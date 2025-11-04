use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, TwoAdicField};
use p3_interpolation::interpolate_subgroup;
use p3_matrix::dense::RowMajorMatrix;

use crate::{
    constant::K_SKIP_SUMCHECK,
    poly::multilinear::MultilinearPoint,
    sumcheck::sumcheck_polynomial::SumcheckPolynomial,
    whir::proof::{InitialPhase, WhirProof},
};
use crate::whir::verifier::errors::VerifierError;

/// Extracts a sequence of `(SumcheckPolynomial, folding_randomness)` pairs from the verifier transcript,
/// and computes the corresponding `MultilinearPoint` folding randomness in reverse order.
///
/// This function reads from the Fiat–Shamir transcript to simulate verifier interaction
/// in the sumcheck protocol. For each round, it recovers:
/// - One univariate polynomial (usually degree ≤ 2) sent by the prover.
/// - One challenge scalar chosen by the verifier (folding randomness).
///
/// ## Modes
///
/// - **Standard mode** (`is_univariate_skip = false`):
///   Each round represents a single variable being folded.
///   The polynomial is evaluated at 3 points, typically `{0, 1, r}` for quadratic reduction.
///
/// - **Univariate skip mode** (`is_univariate_skip = true`):
///   The first `K_SKIP_SUMCHECK` variables are folded simultaneously by evaluating a single univariate polynomial
///   over a coset of size `2^{k+1}`. This yields a larger polynomial but avoids several later rounds.
///
/// # Arguments
///
/// - `proof`: Proof data that is being verified.
/// - `challenger`: The verifier's Fiat–Shamir transcript state.
/// - `rounds`: Total number of variables being folded.
/// - `proof_rounds_offset`: Offset into `proof.rounds` where this sumcheck's data starts.
///   For the initial phase with statement, pass 0. For subsequent sumcheck invocations,
///   pass the cumulative number of rounds consumed from `proof.rounds` so far.
/// - `pow_bits`: Optional proof-of-work difficulty (0 disables PoW).
/// - `is_univariate_skip`: If true, apply the univariate skip optimization on the first `K_SKIP_SUMCHECK` variables.
///
/// # Returns
///
/// - A `MultilinearPoint` of folding randomness values in reverse order.
pub(crate) fn verify_sumcheck_rounds<EF, F, Challenger, const DIGEST_ELEMS: usize>(
    proof: &mut WhirProof<F, EF, DIGEST_ELEMS>,
    challenger: &mut Challenger,
    claimed_sum: &mut EF,
    rounds: usize,
    proof_rounds_offset: usize,
    pow_bits: usize,
    is_univariate_skip: bool,
) -> Result<MultilinearPoint<EF>, VerifierError>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    // Calculate how many `(poly, rand)` pairs to expect based on skip mode
    //
    // If skipping: we do 1 large round for the skip, and the remaining normally
    let effective_rounds = if is_univariate_skip && (rounds >= K_SKIP_SUMCHECK) {
        1 + (rounds - K_SKIP_SUMCHECK)
    } else {
        rounds
    };

    // Preallocate vector to hold the randomness values
    let mut randomness = Vec::with_capacity(effective_rounds);

    // Handle the univariate skip case
    if is_univariate_skip && (rounds >= K_SKIP_SUMCHECK) {
        // Read `2^{k+1}` evaluations (size of coset domain) for the skipping polynomial
        // and get PoW witness after the skip round
        let (evals, skip_pow) = match &proof.initial_phase {
            InitialPhase::WithStatementSkip { skip_evaluations, skip_pow } => {
                let evals: [EF; 1 << (K_SKIP_SUMCHECK + 1)] = skip_evaluations.as_slice()
                    .try_into()
                    .expect("skip_evaluations has wrong length");
                (evals, skip_pow)
            }
            _ => panic!("Expected WithStatementSkip variant"),
        };

        // Interpolate into a univariate polynomial (over the coset domain)
        let poly = evals.to_vec();

        // Verify that the sum over the subgroup H of size 2^k matches the claimed sum.
        //
        // The prover sends evaluations on a coset of H.
        // The even-indexed evaluations correspond to the points in H itself.
        let actual_sum: EF = poly.iter().step_by(2).copied().sum();
        if actual_sum != *claimed_sum {
            return Err(VerifierError::SumcheckFailed {
                round: 0,
                expected: claimed_sum.to_string(),
                actual: actual_sum.to_string(),
            });
        }

        // Observe the skip evaluations (Fiat-Shamir)
        let flattened = EF::flatten_to_base(evals.to_vec());
        challenger.observe_slice(&flattened);

        // Optional: apply proof-of-work query
        if let Some(witnesses) = skip_pow {
            if let Some(&first_witness) = witnesses.first() {
                let _ = challenger.check_witness(pow_bits, first_witness);
            }
        }

        // Sample the challenge scalar r₀ ∈ 𝔽 for this round
        let rand = challenger.sample_algebra_element();

        // Update the claimed sum using the univariate polynomial and randomness.
        //
        // We interpolate the univariate polynomial at the randomness point.
        *claimed_sum = interpolate_subgroup(&RowMajorMatrix::new_col(poly), rand)[0];

        // Record this round’s randomness
        randomness.push(rand);
    }

    // Continue with the remaining sumcheck rounds
    let start_round = if is_univariate_skip && rounds >= K_SKIP_SUMCHECK {
        K_SKIP_SUMCHECK // skip the first k rounds
    } else {
        0
    };

    // Check if initial_phase has WithStatement data available
    let has_initial_statement = !is_univariate_skip
        && matches!(&proof.initial_phase, InitialPhase::WithStatement { sumcheck } if !sumcheck.polynomial_evaluations.is_empty());


    for i in start_round..rounds {
        // Extract the first and third evaluations of the sumcheck polynomial
        // and derive the second evaluation from the latest sum
        // Detect if this is the initial phase call
        // offset==0 means initial phase, offset>0 means subsequent
        let is_initial_phase_call = has_initial_statement && proof_rounds_offset == 0;

        // Determine where to read this round's data from
        let rounds_index = if is_initial_phase_call && i > 0 {
            // Initial phase call, rounds after i=0: read from proof.rounds[i-1]
            i - 1
        } else if proof_rounds_offset == 0 && is_univariate_skip {
            // Univariate skip case
            i - start_round
        } else if proof_rounds_offset > 0 {
            // Subsequent call: offset is encoded as (actual_offset + 1) to distinguish from initial phase
            // So we need to subtract 1 to get the actual offset into proof.rounds
            (proof_rounds_offset - 1) + i
        } else {
            // Fallback
            proof_rounds_offset + i
        };

        // Check if we should use initial_phase
        let use_initial_phase = is_initial_phase_call && i == 0;

        let (c0, c1, c2, pow_witnesses) = if use_initial_phase {
            // First round of initial phase with statement (non-skip case)
            let sumcheck_data = match &proof.initial_phase {
                InitialPhase::WithStatement { sumcheck } => sumcheck,
                _ => unreachable!("Already checked has_initial_statement"),
            };

            let [c0, c1, c2] = sumcheck_data.polynomial_evaluations[0];
            let pow_witness = sumcheck_data.pow_witnesses.as_ref().and_then(|w| w.first().copied());
            (c0, c1, c2, pow_witness)
        } else {
            // Read from proof.rounds
            let round_proof = proof.rounds.get(rounds_index)
                .ok_or_else(|| VerifierError::SumcheckFailed {
                    round: i,
                    expected: "valid round data".to_string(),
                    actual: format!("missing round data at proof.rounds[{}], proof.rounds.len()={}", rounds_index, proof.rounds.len()),
                })?;

            let [c0, c1, c2] = round_proof.sumcheck.polynomial_evaluations[0];
            let pow_witness = round_proof.sumcheck.pow_witnesses.as_ref().and_then(|w| w.first().copied());
            (c0, c1, c2, pow_witness)
        };

        if *claimed_sum != c0 + c1 {
            return Err(VerifierError::SumcheckFailed {
                round: i,
                expected: claimed_sum.to_string(),
                actual: (c0 + c1).to_string(),
            });
        }

        // Optional PoW interaction (grinding resistance)
        if let Some(witness) = pow_witnesses {
            let _ = challenger.check_witness(pow_bits, witness);
        }

        // Sample the next verifier folding randomness rᵢ
        let rand: EF = challenger.sample_algebra_element();

        // Update claimed sum using folding randomness
        *claimed_sum = SumcheckPolynomial::new(vec![c0, c1, c2])
            .evaluate_on_standard_domain(&MultilinearPoint::new(vec![rand]));

        // Store this round's randomness
        randomness.push(rand);
    }

    // We should reverse the order of the randomness points:
    // This is because the randomness points are originally reverted at the end of the sumcheck rounds.
    randomness.reverse();

    Ok(MultilinearPoint::new(randomness))
}
#[cfg(test)]
mod tests {
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::{DuplexChallenger, CanObserve};
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField, BasedVectorSpace};
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::{SeedableRng, rngs::SmallRng};

    use super::*;
    use crate::{
        parameters::{FoldingFactor, ProtocolParameters, errors::SecurityAssumption},
        poly::coeffs::CoefficientList,
        sumcheck::sumcheck_single::SumcheckSingle,
        whir::{constraints::statement::Statement, parameters::WhirConfig},
    };

    type F = BabyBear;
    type EF4 = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;

    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

    // Digest size matches MyCompress output size (the 3rd parameter of TruncatedPermutation)
    const DIGEST_ELEMS: usize = 8;

    /// Constructs a default WHIR configuration for testing
    fn default_whir_config(
        num_variables: usize,
    ) -> WhirConfig<EF4, F, MyHash, MyCompress, MyChallenger> {
        // Create hash and compression functions for the Merkle tree
        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);

        let merkle_hash = MyHash::new(perm.clone());
        let merkle_compress = MyCompress::new(perm);

        // Construct WHIR protocol parameters
        let whir_params = ProtocolParameters {
            initial_statement: true,
            security_level: 32,
            pow_bits: 0,
            rs_domain_initial_reduction_factor: 1,
            folding_factor: FoldingFactor::Constant(2),
            merkle_hash,
            merkle_compress,
            soundness_type: SecurityAssumption::UniqueDecoding,
            starting_log_inv_rate: 1,
            univariate_skip: false,
        };

        // Combine protocol and polynomial parameters into a single config
        WhirConfig::new(num_variables, whir_params)
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_read_sumcheck_rounds_variants() {
        // Define a multilinear polynomial in 3 variables:
        // f(X0, X1, X2) = 1 + 2*X2 + 3*X1 + 4*X1*X2
        //              + 5*X0 + 6*X0*X2 + 7*X0*X1 + 8*X0*X1*X2
        let c1 = F::from_u64(1);
        let c2 = F::from_u64(2);
        let c3 = F::from_u64(3);
        let c4 = F::from_u64(4);
        let c5 = F::from_u64(5);
        let c6 = F::from_u64(6);
        let c7 = F::from_u64(7);
        let c8 = F::from_u64(8);

        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4, c5, c6, c7, c8]);

        // Define the actual polynomial function over EF4
        let f = |x0: EF4, x1: EF4, x2: EF4| {
            x2 * c2
                + x1 * c3
                + x1 * x2 * c4
                + x0 * c5
                + x0 * x2 * c6
                + x0 * x1 * c7
                + x0 * x1 * x2 * c8
                + c1
        };

        let n_vars = coeffs.num_variables();
        assert_eq!(n_vars, 3);

        // Create a constraint system with evaluations of f at various points
        let mut statement = Statement::initialize(n_vars);

        let x_000 = MultilinearPoint::new(vec![EF4::ZERO, EF4::ZERO, EF4::ZERO]);
        let x_100 = MultilinearPoint::new(vec![EF4::ONE, EF4::ZERO, EF4::ZERO]);
        let x_110 = MultilinearPoint::new(vec![EF4::ONE, EF4::ONE, EF4::ZERO]);
        let x_111 = MultilinearPoint::new(vec![EF4::ONE, EF4::ONE, EF4::ONE]);
        let x_011 = MultilinearPoint::new(vec![EF4::ZERO, EF4::ONE, EF4::ONE]);

        let f_000 = f(EF4::ZERO, EF4::ZERO, EF4::ZERO);
        let f_100 = f(EF4::ONE, EF4::ZERO, EF4::ZERO);
        let f_110 = f(EF4::ONE, EF4::ONE, EF4::ZERO);
        let f_111 = f(EF4::ONE, EF4::ONE, EF4::ONE);
        let f_011 = f(EF4::ZERO, EF4::ONE, EF4::ONE);

        statement.add_evaluated_constraint(x_000, f_000);
        statement.add_evaluated_constraint(x_100, f_100);
        statement.add_evaluated_constraint(x_110, f_110);
        statement.add_evaluated_constraint(x_111, f_111);
        statement.add_evaluated_constraint(x_011, f_011);

        let folding_factor = 3;
        let pow_bits = 0;

        // Create protocol parameters
        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);

        let merkle_hash = MyHash::new(perm.clone());
        let merkle_compress = MyCompress::new(perm.clone());

        let whir_params = ProtocolParameters {
            initial_statement: true,
            security_level: 32,
            pow_bits,
            rs_domain_initial_reduction_factor: 1,
            folding_factor: FoldingFactor::Constant(2),
            merkle_hash,
            merkle_compress,
            soundness_type: SecurityAssumption::UniqueDecoding,
            starting_log_inv_rate: 1,
            univariate_skip: false,
        };

        // Initialize WhirProof structure
        let mut proof: WhirProof<F, EF4, DIGEST_ELEMS> = WhirProof::from_protocol_parameters(&whir_params, n_vars);

        // Create prover challenger
        let mut prover_challenger = MyChallenger::new(perm.clone());

        // Run prover-side sumcheck
        let (_, _) = SumcheckSingle::<F, EF4>::from_base_evals(
            &coeffs.to_evaluations(),
            &statement,
            EF4::ONE,
            &mut proof,
            &mut prover_challenger,
            folding_factor,
            pow_bits,
        );

        // Manually verify expected sumcheck rounds
        let (_, mut expected_initial_sum) = statement.combine::<F>(EF4::ONE);
        let mut current_sum = expected_initial_sum;
        let mut expected: Vec<(SumcheckPolynomial<EF4>, EF4)> = Vec::with_capacity(folding_factor);

        // Create manual verification challenger (same seed to get same randomness)
        let mut manual_challenger = MyChallenger::new(perm.clone());

        // Extract and verify each sumcheck round
        let initial_sumcheck_data = match &proof.initial_phase {
            InitialPhase::WithStatement { sumcheck } => sumcheck,
            _ => panic!("Expected WithStatement variant"),
        };

        let [c0, c1, c2] = initial_sumcheck_data.polynomial_evaluations[0];

        assert_eq!(current_sum, c0 + c1, "Sumcheck failed at initial round");

        let poly = SumcheckPolynomial::new(vec![c0, c1, c2]);

        // Sample random challenge r_i ∈ EF4 and evaluate h_i(r_i)
        let r: EF4 = manual_challenger.sample_algebra_element();
        current_sum = poly.evaluate_on_standard_domain(&MultilinearPoint::new(vec![r]));

        expected.push((poly, r));

        for i in 0..(folding_factor - 1) {
            // Get the 3 evaluations of sumcheck polynomial h_i(X) at X = 0, 1, 2
            // After the initial phase, subsequent rounds are stored in proof.rounds[i]
            let [c0, c1, c2] = proof.rounds[i].sumcheck
                .polynomial_evaluations[0];

            assert_eq!(current_sum, c0 + c1, "Sumcheck failed at round {}", i + 1);

            let poly = SumcheckPolynomial::new(vec![c0, c1, c2]);

            // Sample random challenge r_i ∈ EF4 and evaluate h_i(r_i)
            let r: EF4 = manual_challenger.sample_algebra_element();
            current_sum = poly.evaluate_on_standard_domain(&MultilinearPoint::new(vec![r]));

            expected.push((poly, r));
        }

        // Create verifier challenger (same seed to get same randomness)
        let mut verifier_challenger = MyChallenger::new(perm);

        // Call verify_sumcheck_rounds
        let randomness = verify_sumcheck_rounds(
            &mut proof,
            &mut verifier_challenger,
            &mut expected_initial_sum,
            folding_factor,
            0, // proof_rounds_offset for initial phase
            pow_bits,
            false, // is_univariate_skip
        )
        .unwrap();

        // Check that number of parsed rounds is correct
        assert_eq!(randomness.num_variables(), folding_factor);

        // Reconstruct the expected MultilinearPoint from reversed order of expected randomness
        let expected_randomness =
            MultilinearPoint::new(expected.iter().map(|&(_, r)| r).rev().collect());
        assert_eq!(
            randomness, expected_randomness,
            "Mismatch in full MultilinearPoint folding randomness"
        );

    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_read_sumcheck_rounds_with_univariate_skip() {
        // -------------------------------------------------------------
        // Define a multilinear polynomial in K = K_SKIP_SUMCHECK + 3 variables.
        // The number of coefficients must be 2^K.
        // -------------------------------------------------------------
        const K_SKIP: usize = K_SKIP_SUMCHECK;
        const NUM_VARS: usize = K_SKIP + 3;
        let num_points = 1 << NUM_VARS;

        // Construct simple deterministic coefficients f = [1, 2, ..., 2^K]
        let coeffs: Vec<F> = (1..=num_points).map(F::from_u64).collect();
        let coeffs = CoefficientList::new(coeffs);

        assert_eq!(coeffs.num_variables(), NUM_VARS);

        // -------------------------------------------------------------
        // Construct a Statement by evaluating f at several Boolean points
        // These evaluations will serve as equality constraints
        // -------------------------------------------------------------
        let mut statement = Statement::initialize(NUM_VARS);
        for i in 0..5 {
            let bool_point: Vec<_> = (0..NUM_VARS)
                .map(|j| {
                    if (i >> j) & 1 == 1 {
                        EF4::ONE
                    } else {
                        EF4::ZERO
                    }
                })
                .collect();
            let ml_point = MultilinearPoint::new(bool_point.clone());
            let expected_val = coeffs.evaluate(&ml_point);
            statement.add_evaluated_constraint(ml_point, expected_val);
        }

        let pow_bits = 0;

        // Create protocol parameters with univariate skip enabled
        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);

        let merkle_hash = MyHash::new(perm.clone());
        let merkle_compress = MyCompress::new(perm.clone());

        let whir_params = ProtocolParameters {
            initial_statement: true,
            security_level: 32,
            pow_bits,
            rs_domain_initial_reduction_factor: 1,
            folding_factor: FoldingFactor::Constant(NUM_VARS),
            merkle_hash,
            merkle_compress,
            soundness_type: SecurityAssumption::UniqueDecoding,
            starting_log_inv_rate: 1,
            univariate_skip: true,
        };

        // Initialize WhirProof structure
        let mut proof: WhirProof<F, EF4, DIGEST_ELEMS> = WhirProof::from_protocol_parameters(&whir_params, NUM_VARS);

        // Create prover challenger
        let mut prover_challenger = MyChallenger::new(perm.clone());

        // Compute expected sum before running prover
        let (_, mut expected_sum) = statement.combine::<F>(EF4::ONE);

        // Run prover-side sumcheck with univariate skip
        let (_, _) = SumcheckSingle::<F, EF4>::with_skip(
            &coeffs.to_evaluations(),
            &statement,
            EF4::ONE,
            &mut proof,
            &mut prover_challenger,
            NUM_VARS,
            pow_bits,
            K_SKIP,
        );

        // Manually extract and verify expected sumcheck rounds
        let mut current_sum = expected_sum;
        let mut expected = Vec::new();

        // Create manual verification challenger (same seed to get same randomness)
        let mut manual_challenger = MyChallenger::new(perm.clone());

        // Extract skip round data
        let skip_data = match &proof.initial_phase {
            InitialPhase::WithStatementSkip { skip_evaluations, .. } => skip_evaluations,
            _ => panic!("Expected WithStatementSkip variant"),
        };

        // First skipped round (wide DFT LDE)
        let evals: [EF4; 1 << (K_SKIP + 1)] = skip_data.as_slice()
            .try_into()
            .expect("skip_evaluations has wrong length");

        // Verify that the sum over the subgroup H matches the expected sum
        let actual_sum: EF4 = evals.iter().step_by(2).copied().sum();
        assert_eq!(current_sum, actual_sum, "Skip round sum mismatch");

        // Observe the skip evaluations before sampling the challenge (Fiat-Shamir)
        let flattened: Vec<F> = EF4::flatten_to_base(evals.to_vec());
        manual_challenger.observe_slice(&flattened);

        let poly = SumcheckPolynomial::new(evals.to_vec());
        let r0: EF4 = manual_challenger.sample_algebra_element();
        expected.push((poly.clone(), r0));

        let mat = RowMajorMatrix::new(evals.to_vec(), 1);
        current_sum = interpolate_subgroup(&mat, r0)[0];

        // Remaining quadratic rounds from proof.rounds
        for i in 0..(NUM_VARS - K_SKIP) {
            let [c0, c1, c2] = proof.rounds[i].sumcheck.polynomial_evaluations[0];

            assert_eq!(current_sum, c0 + c1, "Sumcheck failed at round {}", K_SKIP + i);

            let poly = SumcheckPolynomial::new(vec![c0, c1, c2]);
            let r: EF4 = manual_challenger.sample_algebra_element();
            current_sum = poly.evaluate_on_standard_domain(&MultilinearPoint::new(vec![r]));
            expected.push((poly, r));
        }

        // Create verifier challenger (same seed to get same randomness)
        let mut verifier_challenger = MyChallenger::new(perm);

        // Call verify_sumcheck_rounds with skip enabled
        let randomness = verify_sumcheck_rounds(
            &mut proof,
            &mut verifier_challenger,
            &mut expected_sum,
            NUM_VARS,
            0, // proof_rounds_offset for initial phase
            pow_bits,
            true, // is_univariate_skip
        )
        .unwrap();

        // Check length:
        // - 1 randomness for the first K skipped rounds
        // - 1 randomness for each regular round
        assert_eq!(randomness.num_variables(), NUM_VARS - K_SKIP + 1);

        // Reconstruct the expected MultilinearPoint from reversed order of expected randomness
        let expected_randomness =
            MultilinearPoint::new(expected.iter().map(|&(_, r)| r).rev().collect());
        assert_eq!(
            randomness, expected_randomness,
            "Mismatch in full MultilinearPoint folding randomness"
        );
    }
}
