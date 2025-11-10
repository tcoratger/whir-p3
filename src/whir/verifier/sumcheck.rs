use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, TwoAdicField};
use p3_interpolation::interpolate_subgroup;
use p3_matrix::dense::RowMajorMatrix;

use crate::{
    constant::K_SKIP_SUMCHECK,
    poly::multilinear::MultilinearPoint,
    sumcheck::sumcheck_polynomial::SumcheckPolynomial,
    whir::proof::{InitialPhase, SumcheckData, WhirRoundProof},
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
/// - `pow_bits`: Optional proof-of-work difficulty (0 disables PoW).
/// - `is_univariate_skip`: If true, apply the univariate skip optimization on the first `K_SKIP_SUMCHECK` variables.
///
/// Verify the initial sumcheck phase based on the InitialPhase variant.
///
/// # Returns
///
/// - A `MultilinearPoint` of folding randomness values in reverse order.
pub(crate) fn verify_initial_sumcheck_rounds<F, EF, Challenger>(
    initial_phase: &InitialPhase<EF, F>,
    challenger: &mut Challenger,
    claimed_sum: &mut EF,
    rounds: usize,
    pow_bits: usize,
    is_univariate_skip: bool,
) -> Result<MultilinearPoint<EF>, VerifierError>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    match initial_phase {
        InitialPhase::WithStatementSkip { skip_evaluations, skip_pow, sumcheck } => {
            // Handle univariate skip optimization
            if !is_univariate_skip || rounds < K_SKIP_SUMCHECK {
                return Err(VerifierError::SumcheckFailed {
                    round: 0,
                    expected: "univariate skip optimization enabled".to_string(),
                    actual: "WithStatementSkip phase without skip conditions".to_string(),
                });
            }

            // Verify skip round evaluations size
            let skip_size = 1 << (K_SKIP_SUMCHECK + 1);
            if skip_evaluations.len() != skip_size {
                return Err(VerifierError::SumcheckFailed {
                    round: 0,
                    expected: format!("{} evaluations", skip_size),
                    actual: format!("{} evaluations", skip_evaluations.len()),
                });
            }

            // Verify sum over subgroup H (every other element starting from 0)
            let actual_sum: EF = skip_evaluations.iter().step_by(2).copied().sum();
            if actual_sum != *claimed_sum {
                return Err(VerifierError::SumcheckFailed {
                    round: 0,
                    expected: claimed_sum.to_string(),
                    actual: actual_sum.to_string(),
                });
            }

            // Observe the skip evaluations for Fiat-Shamir
            let flattened: Vec<F> = EF::flatten_to_base(skip_evaluations.to_vec());
            challenger.observe_slice(&flattened);

            // Verify PoW if present
            if pow_bits > 0 {
                if let Some(pow_witnesses) = skip_pow {
                    if !pow_witnesses.is_empty() {
                        let _ = challenger.check_witness(pow_bits, pow_witnesses[0]);
                    }
                }
            }

            // Sample challenge for the skip round
            let r_skip: EF = challenger.sample_algebra_element();

            // Interpolate to get the new claimed sum after skip folding
            let mat = RowMajorMatrix::new(skip_evaluations.to_vec(), 1);
            *claimed_sum = interpolate_subgroup(&mat, r_skip)[0];

            // Now process the remaining standard sumcheck rounds after the skip
            let mut randomness = Vec::with_capacity(1 + (rounds - K_SKIP_SUMCHECK));
            randomness.push(r_skip);

            // The remaining (rounds - K_SKIP_SUMCHECK) rounds are standard sumcheck rounds
            // stored in the sumcheck field
            let remaining_rounds = rounds - K_SKIP_SUMCHECK;
            for i in 0..remaining_rounds.min(sumcheck.polynomial_evaluations.len()) {
                let [c0, c1, c2] = sumcheck.polynomial_evaluations[i];

                // Verify sumcheck equation: h(0) + h(1) = claimed_sum
                if *claimed_sum != c0 + c1 {
                    return Err(VerifierError::SumcheckFailed {
                        round: K_SKIP_SUMCHECK + i,
                        expected: claimed_sum.to_string(),
                        actual: (c0 + c1).to_string(),
                    });
                }

                // Observe polynomial evaluations
                challenger.observe_slice(&EF::flatten_to_base(vec![c0, c1, c2]));

                // Verify PoW if present
                if pow_bits > 0 {
                    if let Some(ref pow_witnesses) = sumcheck.pow_witnesses {
                        if i < pow_witnesses.len() {
                            let _ = challenger.check_witness(pow_bits, pow_witnesses[i]);
                        }
                    }
                }

                // Create polynomial and sample challenge
                let poly = SumcheckPolynomial::new(vec![c0, c1, c2]);
                let r: EF = challenger.sample_algebra_element();

                // Update claimed sum for next round
                *claimed_sum = poly.evaluate_on_standard_domain(&MultilinearPoint::new(vec![r]));
                randomness.push(r);
            }

            // Reverse for the expected order
            randomness.reverse();
            Ok(MultilinearPoint::new(randomness))
        }

        InitialPhase::WithStatement { sumcheck } => {
            // Standard initial sumcheck without skip
            if is_univariate_skip && rounds >= K_SKIP_SUMCHECK {
                return Err(VerifierError::SumcheckFailed {
                    round: 0,
                    expected: "WithStatementSkip phase".to_string(),
                    actual: "WithStatement phase when skip should be enabled".to_string(),
                });
            }

            let mut randomness = Vec::with_capacity(rounds);

            // Process all sumcheck rounds in the initial phase
            for i in 0..rounds.min(sumcheck.polynomial_evaluations.len()) {
                let [c0, c1, c2] = sumcheck.polynomial_evaluations[i];

                // Verify sumcheck equation: h(0) + h(1) = claimed_sum
                if *claimed_sum != c0 + c1 {
                    return Err(VerifierError::SumcheckFailed {
                        round: i,
                        expected: claimed_sum.to_string(),
                        actual: (c0 + c1).to_string(),
                    });
                }

                // Observe polynomial evaluations
                challenger.observe_slice(&EF::flatten_to_base(vec![c0, c1, c2]));

                // Verify PoW if present
                if pow_bits > 0 {
                    if let Some(ref pow_witnesses) = sumcheck.pow_witnesses {
                        if i < pow_witnesses.len() {
                            let _ = challenger.check_witness(pow_bits, pow_witnesses[i]);
                        }
                    }
                }

                // Create polynomial and sample challenge
                let poly = SumcheckPolynomial::new(vec![c0, c1, c2]);
                let r: EF = challenger.sample_algebra_element();

                // Update claimed sum for next round
                *claimed_sum = poly.evaluate_on_standard_domain(&MultilinearPoint::new(vec![r]));
                randomness.push(r);
            }

            // Reverse for the expected order
            randomness.reverse();
            Ok(MultilinearPoint::new(randomness))
        }

        InitialPhase::WithoutStatement { .. } => {
            // No sumcheck in the initial phase for WithoutStatement
            // The folding randomness is sampled directly in the main verifier
            Ok(MultilinearPoint::new(Vec::new()))
        }
    }
}

/// Verify sumcheck rounds from a WhirRoundProof.
///
/// # Returns
///
/// - A `MultilinearPoint` of folding randomness values in reverse order.
pub(crate) fn verify_sumcheck_rounds<F, EF, Challenger, const DIGEST_ELEMS: usize>(
    round_proof: &WhirRoundProof<F, EF, DIGEST_ELEMS>,
    challenger: &mut Challenger,
    claimed_sum: &mut EF,
    rounds: usize,
    pow_bits: usize,
) -> Result<MultilinearPoint<EF>, VerifierError>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    let sumcheck = &round_proof.sumcheck;

    if sumcheck.polynomial_evaluations.len() != rounds {
        return Err(VerifierError::SumcheckFailed {
            round: 0,
            expected: format!("{} rounds", rounds),
            actual: format!("{} rounds in proof", sumcheck.polynomial_evaluations.len()),
        });
    }

    let mut randomness = Vec::with_capacity(rounds);

    for i in 0..rounds {
        let [c0, c1, c2] = sumcheck.polynomial_evaluations[i];

        // Verify sumcheck equation: h(0) + h(1) = claimed_sum
        if *claimed_sum != c0 + c1 {
            return Err(VerifierError::SumcheckFailed {
                round: i,
                expected: claimed_sum.to_string(),
                actual: (c0 + c1).to_string(),
            });
        }

        // Observe polynomial evaluations
        challenger.observe_slice(&EF::flatten_to_base(vec![c0, c1, c2]));

        // Verify PoW if present
        if pow_bits > 0 {
            if let Some(ref pow_witnesses) = sumcheck.pow_witnesses {
                if i < pow_witnesses.len() {
                    let _ = challenger.check_witness(pow_bits, pow_witnesses[i]);
                }
            }
        }

        // Create polynomial and sample challenge
        let poly = SumcheckPolynomial::new(vec![c0, c1, c2]);
        let r: EF = challenger.sample_algebra_element();

        // Update claimed sum for next round
        *claimed_sum = poly.evaluate_on_standard_domain(&MultilinearPoint::new(vec![r]));
        randomness.push(r);
    }

    // Reverse for the expected order
    randomness.reverse();
    Ok(MultilinearPoint::new(randomness))
}

/// Verify the final sumcheck rounds.
///
/// # Returns
///
/// - A `MultilinearPoint` of folding randomness values in reverse order.
pub(crate) fn verify_final_sumcheck_rounds<F, EF, Challenger>(
    final_sumcheck: &Option<SumcheckData<EF, F>>,
    challenger: &mut Challenger,
    claimed_sum: &mut EF,
    rounds: usize,
    pow_bits: usize,
) -> Result<MultilinearPoint<EF>, VerifierError>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    if rounds == 0 {
        // No final sumcheck expected
        return Ok(MultilinearPoint::new(Vec::new()));
    }

    let sumcheck = final_sumcheck.as_ref().ok_or_else(|| VerifierError::SumcheckFailed {
        round: 0,
        expected: format!("{} final sumcheck rounds", rounds),
        actual: "None".to_string(),
    })?;

    if sumcheck.polynomial_evaluations.len() != rounds {
        return Err(VerifierError::SumcheckFailed {
            round: 0,
            expected: format!("{} rounds", rounds),
            actual: format!("{} rounds in proof", sumcheck.polynomial_evaluations.len()),
        });
    }

    let mut randomness = Vec::with_capacity(rounds);

    for i in 0..rounds {
        let [c0, c1, c2] = sumcheck.polynomial_evaluations[i];

        // Verify sumcheck equation: h(0) + h(1) = claimed_sum
        if *claimed_sum != c0 + c1 {
            return Err(VerifierError::SumcheckFailed {
                round: i,
                expected: claimed_sum.to_string(),
                actual: (c0 + c1).to_string(),
            });
        }

        // Observe polynomial evaluations
        challenger.observe_slice(&EF::flatten_to_base(vec![c0, c1, c2]));

        // Verify PoW if present
        if pow_bits > 0 {
            if let Some(ref pow_witnesses) = sumcheck.pow_witnesses {
                if i < pow_witnesses.len() {
                    let _ = challenger.check_witness(pow_bits, pow_witnesses[i]);
                }
            }
        }

        // Create polynomial and sample challenge
        let poly = SumcheckPolynomial::new(vec![c0, c1, c2]);
        let r: EF = challenger.sample_algebra_element();

        // Update claimed sum for next round
        *claimed_sum = poly.evaluate_on_standard_domain(&MultilinearPoint::new(vec![r]));
        randomness.push(r);
    }

    // Reverse for the expected order
    randomness.reverse();
    Ok(MultilinearPoint::new(randomness))
}
#[cfg(test)]
mod tests {
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::{DuplexChallenger, CanObserve};
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField, BasedVectorSpace};
    use p3_matrix::dense::RowMajorMatrix;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::{SeedableRng, rngs::SmallRng};

    use super::*;
    use crate::{
        parameters::{FoldingFactor, ProtocolParameters, errors::SecurityAssumption},
        poly::coeffs::CoefficientList,
        sumcheck::sumcheck_single::SumcheckSingle,
        whir::{constraints::statement::Statement, parameters::WhirConfig},
    };
    use crate::whir::proof::WhirProof;

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

        // Observe polynomial evaluations (must match what verify_initial_sumcheck_rounds does)
        let flattened: Vec<F> = EF4::flatten_to_base(vec![c0, c1, c2]);
        manual_challenger.observe_slice(&flattened);

        // Sample random challenge r_i ∈ EF4 and evaluate h_i(r_i)
        let r: EF4 = manual_challenger.sample_algebra_element();
        current_sum = poly.evaluate_on_standard_domain(&MultilinearPoint::new(vec![r]));

        expected.push((poly, r));

        for i in 0..(folding_factor - 1) {
            // Get the 3 evaluations of sumcheck polynomial h_i(X) at X = 0, 1, 2
            // After the initial round, subsequent rounds within the same folding_factor
            // are also stored in initial_phase.sumcheck
            let [c0, c1, c2] = initial_sumcheck_data.polynomial_evaluations[i + 1];

            assert_eq!(current_sum, c0 + c1, "Sumcheck failed at round {}", i + 1);

            let poly = SumcheckPolynomial::new(vec![c0, c1, c2]);

            // Observe polynomial evaluations (must match what verify_sumcheck_rounds does)
            let flattened: Vec<F> = EF4::flatten_to_base(vec![c0, c1, c2]);
            manual_challenger.observe_slice(&flattened);

            // Sample random challenge r_i ∈ EF4 and evaluate h_i(r_i)
            let r: EF4 = manual_challenger.sample_algebra_element();
            current_sum = poly.evaluate_on_standard_domain(&MultilinearPoint::new(vec![r]));

            expected.push((poly, r));
        }

        // Create verifier challenger (same seed to get same randomness)
        let mut verifier_challenger = MyChallenger::new(perm);

        // Call verify_initial_sumcheck_rounds
        let randomness = verify_initial_sumcheck_rounds(
            &proof.initial_phase,
            &mut verifier_challenger,
            &mut expected_initial_sum,
            folding_factor,
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

        // Extract the sumcheck data that contains the remaining rounds after skip
        let skip_sumcheck_data = match &proof.initial_phase {
            InitialPhase::WithStatementSkip { sumcheck, .. } => sumcheck,
            _ => panic!("Expected WithStatementSkip variant"),
        };

        // Remaining quadratic rounds are stored in initial_phase.sumcheck
        for i in 0..(NUM_VARS - K_SKIP) {
            let [c0, c1, c2] = skip_sumcheck_data.polynomial_evaluations[i];

            assert_eq!(current_sum, c0 + c1, "Sumcheck failed at round {}", K_SKIP + i);

            let poly = SumcheckPolynomial::new(vec![c0, c1, c2]);

            // Observe polynomial evaluations (must match what verify_sumcheck_rounds does)
            let flattened: Vec<F> = EF4::flatten_to_base(vec![c0, c1, c2]);
            manual_challenger.observe_slice(&flattened);

            let r: EF4 = manual_challenger.sample_algebra_element();
            current_sum = poly.evaluate_on_standard_domain(&MultilinearPoint::new(vec![r]));
            expected.push((poly, r));
        }

        // Create verifier challenger (same seed to get same randomness)
        let mut verifier_challenger = MyChallenger::new(perm);

        // Call verify_initial_sumcheck_rounds with skip enabled
        // This now handles both the skip part AND the remaining rounds in the initial phase
        // It returns all randomness values together: 1 for skip + (NUM_VARS - K_SKIP) for remaining
        let randomness = verify_initial_sumcheck_rounds(
            &proof.initial_phase,
            &mut verifier_challenger,
            &mut expected_sum,
            NUM_VARS,
            pow_bits,
            true, // is_univariate_skip
        )
        .unwrap();

        // Check combined length:
        // - 1 randomness for the first K skipped rounds
        // - 1 randomness for each regular round
        assert_eq!(randomness.num_variables(), 1 + (NUM_VARS - K_SKIP));

        // Reconstruct the expected MultilinearPoint from reversed order of expected randomness
        let expected_randomness =
            MultilinearPoint::new(expected.iter().map(|&(_, r)| r).rev().collect());
        assert_eq!(
            randomness, expected_randomness,
            "Mismatch in full MultilinearPoint folding randomness"
        );
    }
}
