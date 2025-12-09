use alloc::{format, string::ToString, vec, vec::Vec};

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, TwoAdicField};
use p3_interpolation::interpolate_subgroup;
use p3_matrix::dense::RowMajorMatrix;

use crate::{
    constant::K_SKIP_SUMCHECK,
    fiat_shamir::grinding::check_pow_grinding,
    poly::multilinear::MultilinearPoint,
    whir::{
        proof::{InitialPhase, SumcheckData, SumcheckSkipData, WhirRoundProof},
        verifier::VerifierError,
    },
};

/// Verifies standard sumcheck rounds and extracts folding randomness from the transcript.
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
/// - `polynomial_evaluations`: The polynomial evaluations (c0, c2) for each round.
/// - `pow_witnesses`: Optional PoW witnesses from the prover.
/// - `challenger`: The Fiat-Shamir challenger for transcript management.
/// - `claimed_sum`: Mutable reference to the claimed sum, updated each round.
/// - `pow_bits`: Proof-of-work difficulty (0 disables PoW).
///
/// # Returns
///
/// - A `MultilinearPoint` of folding randomness values in reverse order.
///   Common helper function to verify standard sumcheck rounds
fn verify_standard_sumcheck_rounds<F, EF, Challenger>(
    polynomial_evaluations: &[[EF; 2]],
    pow_witnesses: Option<&[F]>,
    challenger: &mut Challenger,
    claimed_sum: &mut EF,
    pow_bits: usize,
) -> Result<Vec<EF>, VerifierError>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    let mut randomness = Vec::with_capacity(polynomial_evaluations.len());

    for (i, &[c0, c2]) in polynomial_evaluations.iter().enumerate() {
        // Derive h(1) from the sumcheck equation: h(0) + h(1) = claimed_sum
        let h_1 = *claimed_sum - c0;

        // Observe only the sent polynomial evaluations (c0 and c2)
        challenger.observe_algebra_slice(&[c0, c2]);

        // Verify PoW if present
        check_pow_grinding(
            challenger,
            pow_witnesses.and_then(|w| w.get(i).copied()),
            pow_bits,
        )?;

        // Sample challenge
        let r: EF = challenger.sample_algebra_element();

        // Update claimed sum for next round using direct quadratic formula:
        // h(X) = c0 + c1*X + c2*X^2 where c1 = h(1) - c0 - c2
        // h(r) = c2*r^2 + c1*r + c0 = c2*r^2 + (h(1) - c0 - c2)*r + c0
        *claimed_sum = c2 * r.square() + (h_1 - c0 - c2) * r + c0;
        randomness.push(r);
    }

    Ok(randomness)
}

pub(crate) fn verify_initial_sumcheck_rounds<F, EF, Challenger>(
    initial_phase: &InitialPhase<EF, F>,
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
    match initial_phase {
        InitialPhase::WithStatementSkip(SumcheckSkipData {
            evaluations: skip_evaluations,
            pow: skip_pow,
            sumcheck,
        }) => {
            // Handle univariate skip optimization
            if rounds < K_SKIP_SUMCHECK {
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
                    expected: format!("{skip_size} evaluations"),
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
            challenger.observe_algebra_slice(skip_evaluations);

            check_pow_grinding(challenger, *skip_pow, pow_bits)?;

            // Sample challenge for the skip round
            let r_skip: EF = challenger.sample_algebra_element();

            // Interpolate to get the new claimed sum after skip folding
            let mat = RowMajorMatrix::new(skip_evaluations.clone(), 1);
            *claimed_sum = interpolate_subgroup(&mat, r_skip)[0];

            // Now process the remaining standard sumcheck rounds after the skip
            let remaining_rounds = rounds - K_SKIP_SUMCHECK;
            let mut randomness = vec![r_skip];

            let standard_randomness = verify_standard_sumcheck_rounds(
                &sumcheck.polynomial_evaluations[0..remaining_rounds],
                sumcheck.pow_witnesses.as_deref(),
                challenger,
                claimed_sum,
                pow_bits,
            )?;

            randomness.extend(standard_randomness);
            Ok(MultilinearPoint::new(randomness))
        }

        InitialPhase::WithStatement { sumcheck } => {
            // Standard initial sumcheck without skip
            let randomness = verify_standard_sumcheck_rounds(
                &sumcheck.polynomial_evaluations,
                sumcheck.pow_witnesses.as_deref(),
                challenger,
                claimed_sum,
                pow_bits,
            )?;

            Ok(MultilinearPoint::new(randomness))
        }

        InitialPhase::WithoutStatement { pow_witnesses } => {
            // No sumcheck - just sample folding randomness directly
            let randomness: Vec<EF> = (0..rounds)
                .map(|_| challenger.sample_algebra_element())
                .collect();

            // Check PoW
            check_pow_grinding(challenger, *pow_witnesses, pow_bits)?;

            Ok(MultilinearPoint::new(randomness))
        }

        InitialPhase::WithStatementSvo { sumcheck } => {
            // Fallback to WithStatement behavior (WithStatementSvo not yet implemented)
            let randomness = verify_standard_sumcheck_rounds(
                &sumcheck.polynomial_evaluations,
                sumcheck.pow_witnesses.as_deref(),
                challenger,
                claimed_sum,
                pow_bits,
            )?;

            Ok(MultilinearPoint::new(randomness))
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
            expected: format!("{rounds} rounds"),
            actual: format!("{} rounds in proof", sumcheck.polynomial_evaluations.len()),
        });
    }

    let randomness = verify_standard_sumcheck_rounds(
        &sumcheck.polynomial_evaluations,
        sumcheck.pow_witnesses.as_deref(),
        challenger,
        claimed_sum,
        pow_bits,
    )?;

    Ok(MultilinearPoint::new(randomness))
}

/// Verify the final sumcheck rounds.
///
/// # Returns
///
/// - A `MultilinearPoint` of folding randomness values in reverse order.
pub(crate) fn verify_final_sumcheck_rounds<F, EF, Challenger>(
    final_sumcheck: Option<&SumcheckData<EF, F>>,
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

    let sumcheck = final_sumcheck.ok_or_else(|| VerifierError::SumcheckFailed {
        round: 0,
        expected: format!("{rounds} final sumcheck rounds"),
        actual: "None".to_string(),
    })?;

    if sumcheck.polynomial_evaluations.len() != rounds {
        return Err(VerifierError::SumcheckFailed {
            round: 0,
            expected: format!("{rounds} rounds"),
            actual: format!("{} rounds in proof", sumcheck.polynomial_evaluations.len()),
        });
    }

    let randomness = verify_standard_sumcheck_rounds(
        &sumcheck.polynomial_evaluations,
        sumcheck.pow_witnesses.as_deref(),
        challenger,
        claimed_sum,
        pow_bits,
    )?;
    Ok(MultilinearPoint::new(randomness))
}
#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::{SeedableRng, rngs::SmallRng};

    use super::*;
    use crate::{
        fiat_shamir::domain_separator::{DomainSeparator, SumcheckParams},
        parameters::{FoldingFactor, ProtocolParameters, errors::SecurityAssumption},
        poly::{coeffs::CoefficientList, evals::EvaluationsList},
        sumcheck::sumcheck_single::SumcheckSingle,
        whir::{
            constraints::{Constraint, statement::EqStatement},
            parameters::InitialPhaseConfig,
            proof::{InitialPhase, WhirProof},
        },
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
    fn create_proof_from_test_protocol_params(
        num_variables: usize,
        folding_factor: FoldingFactor,
        initial_phase_config: InitialPhaseConfig,
    ) -> WhirProof<F, EF4, DIGEST_ELEMS> {
        // Create hash and compression functions for the Merkle tree
        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);

        let merkle_hash = MyHash::new(perm.clone());
        let merkle_compress = MyCompress::new(perm);

        // Construct WHIR protocol parameters
        let whir_params = ProtocolParameters {
            initial_phase_config,
            security_level: 32,
            pow_bits: 0,
            rs_domain_initial_reduction_factor: 1,
            folding_factor,
            merkle_hash,
            merkle_compress,
            soundness_type: SecurityAssumption::UniqueDecoding,
            starting_log_inv_rate: 1,
        };

        // Combine protocol and polynomial parameters into a single config
        WhirProof::from_protocol_parameters(&whir_params, num_variables)
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
        let mut statement = EqStatement::initialize(n_vars);

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

        // Set up domain separator
        // - Add sumcheck
        let mut domsep: DomainSeparator<EF4, F> = DomainSeparator::new(vec![]);
        domsep.add_sumcheck(&SumcheckParams {
            rounds: folding_factor,
            pow_bits,
            univariate_skip: None,
        });

        let mut rng = SmallRng::seed_from_u64(1);
        let challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));
        let mut prover_challenger = challenger.clone();

        let constraint = Constraint::new_eq_only(EF4::ONE, statement.clone());

        // Initialize proof and challenger
        let mut proof = create_proof_from_test_protocol_params(
            n_vars,
            FoldingFactor::Constant(folding_factor),
            InitialPhaseConfig::WithStatementClassic,
        );
        domsep.observe_domain_separator(&mut prover_challenger);

        // Extract sumcheck data from the initial phase
        let InitialPhase::WithStatement { ref mut sumcheck } = proof.initial_phase else {
            panic!("Expected WithStatement variant");
        };

        // Instantiate the prover with base field coefficients
        let (_, _) = SumcheckSingle::<F, EF4>::from_base_evals(
            &coeffs.to_evaluations(),
            sumcheck,
            &mut prover_challenger,
            folding_factor,
            pow_bits,
            &constraint,
        );

        // Reconstruct verifier state to simulate the rounds
        let mut verifier_challenger = challenger;
        domsep.observe_domain_separator(&mut verifier_challenger);

        // Save a fresh copy for verify_initial_sumcheck_rounds
        let mut verifier_challenger_for_verify = verifier_challenger.clone();

        let mut t = EvaluationsList::zero(statement.num_variables());
        let mut expected_initial_sum = EF4::ZERO;
        statement.combine_hypercube::<F, false>(&mut t, &mut expected_initial_sum, EF4::ONE);
        // Start with the claimed sum before folding
        let mut current_sum = expected_initial_sum;

        let mut expected = Vec::with_capacity(folding_factor);

        // Extract and verify each sumcheck round
        let InitialPhase::WithStatement {
            sumcheck: initial_sumcheck_data,
        } = &proof.initial_phase
        else {
            panic!("Expected WithStatement variant")
        };

        // First round: read c_0 = h(0) and c_2 (quadratic coefficient)
        let [c_0, c_2] = initial_sumcheck_data.polynomial_evaluations[0];
        let h_1 = current_sum - c_0;

        // Observe polynomial evaluations (must match what verify_initial_sumcheck_rounds does)
        verifier_challenger.observe_algebra_slice(&[c_0, c_2]);

        // Sample random challenge r_i ∈ EF4 and evaluate h_i(r_i)
        let r: EF4 = verifier_challenger.sample_algebra_element();
        // h(r) = c_2 * r^2 + (h(1) - c_0 - c_2) * r + c_0
        current_sum = c_2 * r.square() + (h_1 - c_0 - c_2) * r + c_0;
        expected.push(r);

        for i in 0..folding_factor - 1 {
            // Read c_0 = h(0) and c_2 (quadratic coefficient), derive h(1) = claimed_sum - c_0
            let [c_0, c_2] = initial_sumcheck_data.polynomial_evaluations[i + 1];
            let h_1 = current_sum - c_0;

            // Observe polynomial evaluations
            verifier_challenger.observe_algebra_slice(&[c_0, c_2]);

            // Sample random challenge r
            let r: EF4 = verifier_challenger.sample_algebra_element();
            // h(r) = c_2 * r^2 + (h(1) - c_0 - c_2) * r + c_0
            current_sum = c_2 * r.square() + (h_1 - c_0 - c_2) * r + c_0;

            if pow_bits > 0 {
                // verifier_state.challenge_pow::<Blake3PoW>(pow_bits).unwrap();
            }

            expected.push(r);
        }

        let randomness = verify_initial_sumcheck_rounds(
            &proof.initial_phase,
            &mut verifier_challenger_for_verify,
            &mut expected_initial_sum,
            folding_factor,
            pow_bits,
        )
        .unwrap();

        // Check that number of parsed rounds is correct
        assert_eq!(randomness.num_variables(), folding_factor);

        // Reconstruct the expected MultilinearPoint from expected randomness
        let expected_randomness = MultilinearPoint::new(expected);
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
        let mut statement = EqStatement::initialize(NUM_VARS);
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

        let folding_factor = NUM_VARS;
        let pow_bits = 0;

        // Set up domain separator
        let mut domsep: DomainSeparator<EF4, F> = DomainSeparator::new(vec![]);
        domsep.add_sumcheck(&SumcheckParams {
            rounds: folding_factor,
            pow_bits,
            univariate_skip: Some(K_SKIP),
        });

        let mut rng = SmallRng::seed_from_u64(1);
        let challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));
        let mut prover_challenger = challenger.clone();

        let constraint = Constraint::new_eq_only(EF4::ONE, statement.clone());

        // Initialize proof and challenger
        let mut proof = create_proof_from_test_protocol_params(
            NUM_VARS,
            FoldingFactor::Constant(folding_factor),
            InitialPhaseConfig::WithStatementUnivariateSkip,
        );
        domsep.observe_domain_separator(&mut prover_challenger);

        // Extract skip data from the initial phase
        let InitialPhase::WithStatementSkip(ref mut skip_data) = proof.initial_phase else {
            panic!("Expected WithStatementSkip variant");
        };

        // Instantiate the prover with base field coefficients and univariate skip
        let (_, _) = SumcheckSingle::<F, EF4>::with_skip(
            &coeffs.to_evaluations(),
            skip_data,
            &mut prover_challenger,
            folding_factor,
            pow_bits,
            K_SKIP,
            &constraint,
        );

        // Reconstruct verifier state to simulate the rounds
        let mut verifier_challenger = challenger;
        domsep.observe_domain_separator(&mut verifier_challenger);

        // Save a fresh copy for verify_initial_sumcheck_rounds
        let mut verifier_challenger_for_verify = verifier_challenger.clone();

        let mut t = EvaluationsList::zero(statement.num_variables());
        let mut expected_initial_sum = EF4::ZERO;
        statement.combine_hypercube::<F, false>(&mut t, &mut expected_initial_sum, EF4::ONE);
        // Start with the claimed sum before folding
        let mut current_sum = expected_initial_sum;

        let mut expected = Vec::new();

        // Extract skip data from the proof for verification replay
        let InitialPhase::WithStatementSkip(skip_data) = &proof.initial_phase else {
            panic!("Expected WithStatementSkip variant");
        };

        // First skipped round (wide DFT LDE)
        let skip_evaluations = &skip_data.evaluations;

        // Verify sum over subgroup H (every other element starting from 0)
        let actual_sum: EF4 = skip_evaluations.iter().step_by(2).copied().sum();
        assert_eq!(actual_sum, current_sum, "Skip round sum mismatch");

        // Observe the skip evaluations for Fiat-Shamir
        verifier_challenger.observe_algebra_slice(skip_evaluations);

        // Sample challenge for the skip round
        let r_skip: EF4 = verifier_challenger.sample_algebra_element();
        expected.push(r_skip);

        // Interpolate to get the new claimed sum after skip folding
        let mat = RowMajorMatrix::new(skip_evaluations.clone(), 1);
        current_sum = interpolate_subgroup(&mat, r_skip)[0];

        // Remaining quadratic rounds after the skip
        let remaining_rounds = folding_factor - K_SKIP;
        for i in 0..remaining_rounds {
            // Read c_0 = h(0) and c_2 (quadratic coefficient), derive h(1) = claimed_sum - c_0
            let [c_0, c_2] = skip_data.sumcheck.polynomial_evaluations[i];
            let h_1 = current_sum - c_0;

            // Observe polynomial evaluations
            verifier_challenger.observe_algebra_slice(&[c_0, c_2]);

            // Sample random challenge r
            let r: EF4 = verifier_challenger.sample_algebra_element();
            // h(r) = c_2 * r^2 + (h(1) - c_0 - c_2) * r + c_0
            current_sum = c_2 * r.square() + (h_1 - c_0 - c_2) * r + c_0;

            expected.push(r);
        }

        let randomness = verify_initial_sumcheck_rounds(
            &proof.initial_phase,
            &mut verifier_challenger_for_verify,
            &mut expected_initial_sum,
            folding_factor,
            pow_bits,
        )
        .unwrap();

        // Check length:
        // - 1 randomness for the first K skipped rounds
        // - 1 randomness for each regular round
        assert_eq!(randomness.num_variables(), folding_factor - K_SKIP + 1);

        // Reconstruct the expected MultilinearPoint from the expected randomness
        let expected_randomness = MultilinearPoint::new(expected);
        assert_eq!(
            randomness, expected_randomness,
            "Mismatch in full MultilinearPoint folding randomness"
        );
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_read_sumcheck_rounds_svo() {
        // Multilinear polynomial in 6 variables
        const NUM_VARS: usize = 6;
        let num_points = 1 << NUM_VARS;

        let coeffs: Vec<F> = (1..=num_points).map(F::from_u64).collect();
        let coeffs = CoefficientList::new(coeffs);

        assert_eq!(coeffs.num_variables(), NUM_VARS);

        // Create a constraint system with evaluations of f at a point
        let mut statement = EqStatement::initialize(NUM_VARS);
        let constraint_point: Vec<_> = (0..NUM_VARS)
            .map(|j| if j % 2 == 0 { EF4::ONE } else { EF4::ZERO })
            .collect();
        let ml_point = MultilinearPoint::new(constraint_point);
        let expected_val = coeffs.evaluate(&ml_point);
        statement.add_evaluated_constraint(ml_point, expected_val);

        let folding_factor = NUM_VARS;
        let pow_bits = 0;

        // Set up domain separator
        let mut domsep: DomainSeparator<EF4, F> = DomainSeparator::new(vec![]);
        domsep.add_sumcheck(&SumcheckParams {
            rounds: folding_factor,
            pow_bits,
            univariate_skip: None,
        });

        let mut rng = SmallRng::seed_from_u64(1);
        let challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));
        let mut prover_challenger = challenger.clone();

        let constraint = Constraint::new_eq_only(EF4::ONE, statement.clone());

        // Initialize proof and challenger
        let mut proof = create_proof_from_test_protocol_params(
            NUM_VARS,
            FoldingFactor::Constant(folding_factor),
            InitialPhaseConfig::WithStatementSvo,
        );
        domsep.observe_domain_separator(&mut prover_challenger);

        // Extract sumcheck data from the initial phase
        let InitialPhase::WithStatementSvo { ref mut sumcheck } = proof.initial_phase else {
            panic!("Expected WithStatementSvo variant");
        };

        // Instantiate the prover with base field coefficients using SVO
        let (_, _) = SumcheckSingle::<F, EF4>::from_base_evals(
            &coeffs.to_evaluations(),
            sumcheck,
            &mut prover_challenger,
            folding_factor,
            pow_bits,
            &constraint,
        );

        // Reconstruct verifier state to simulate the rounds
        let mut verifier_challenger = challenger;
        domsep.observe_domain_separator(&mut verifier_challenger);

        // Save a fresh copy for verify_initial_sumcheck_rounds
        let mut verifier_challenger_for_verify = verifier_challenger.clone();

        let (_, mut expected_initial_sum) = constraint.combine_new();
        // Start with the claimed sum before folding
        let mut current_sum = expected_initial_sum;

        let mut expected = Vec::with_capacity(folding_factor);

        // Extract sumcheck data from the proof for verification replay
        let InitialPhase::WithStatementSvo {
            sumcheck: svo_sumcheck,
        } = &proof.initial_phase
        else {
            panic!("Expected WithStatementSvo variant")
        };

        for i in 0..folding_factor {
            // Read c_0 = h(0) and c_2 (quadratic coefficient), derive h(1) = claimed_sum - c_0
            let [c_0, c_2] = svo_sumcheck.polynomial_evaluations[i];
            let h_1 = current_sum - c_0;

            // Observe polynomial evaluations
            verifier_challenger.observe_algebra_slice(&[c_0, c_2]);

            // Sample random challenge r
            let r: EF4 = verifier_challenger.sample_algebra_element();
            // h(r) = c_2 * r^2 + (h(1) - c_0 - c_2) * r + c_0
            current_sum = c_2 * r.square() + (h_1 - c_0 - c_2) * r + c_0;

            expected.push(r);
        }

        let randomness = verify_initial_sumcheck_rounds(
            &proof.initial_phase,
            &mut verifier_challenger_for_verify,
            &mut expected_initial_sum,
            folding_factor,
            pow_bits,
        )
        .unwrap();

        // Check that number of parsed rounds is correct
        assert_eq!(randomness.num_variables(), folding_factor);

        // Reconstruct the expected MultilinearPoint from expected randomness
        let expected_randomness = MultilinearPoint::new(expected);
        assert_eq!(
            randomness, expected_randomness,
            "Mismatch in full MultilinearPoint folding randomness for SVO"
        );
    }
}
