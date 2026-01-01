use alloc::{string::ToString, vec, vec::Vec};

use p3_field::{ExtensionField, TwoAdicField};
use p3_interpolation::interpolate_subgroup;
use p3_matrix::dense::RowMajorMatrix;

use crate::{
    constant::K_SKIP_SUMCHECK,
    fiat_shamir::transcript::{Challenge, Pow, Reader},
    poly::multilinear::MultilinearPoint,
    whir::{parameters::InitialPhase, verifier::VerifierError},
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
/// - `pow_witnesses`: PoW witnesses from the prover.
/// - `challenger`: The Fiat-Shamir challenger for transcript management.
/// - `claimed_sum`: Mutable reference to the claimed sum, updated each round.
/// - `pow_bits`: Proof-of-work difficulty (0 disables PoW).
///
/// # Returns
///
/// - A `MultilinearPoint` of folding randomness values in reverse order.
///   Common helper function to verify standard sumcheck rounds
pub(crate) fn verify_standard_sumcheck_rounds<F, EF, Transcript>(
    transcript: &mut Transcript,
    number_of_rounds: usize,
    claimed_sum: &mut EF,
    pow_bits: usize,
) -> Result<MultilinearPoint<EF>, VerifierError>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Transcript: Reader<EF> + Challenge<EF> + Pow<F>,
{
    let vars = (0..number_of_rounds)
        .map(|_| {
            let c0 = transcript.read()?;
            let c2 = transcript.read()?;
            let h1 = *claimed_sum - c0;
            transcript.pow(pow_bits)?;
            let r = transcript.sample();
            // Update claimed sum for next round using direct quadratic formula:
            // h(X) = c0 + c1*X + c2*X^2 where c1 = h(1) - c0 - c2
            // h(r) = c2*r^2 + c1*r + c0 = c2*r^2 + (h(1) - c0 - c2)*r + c0
            *claimed_sum = c2 * r.square() + (h1 - c0 - c2) * r + c0;
            Ok(r)
        })
        .collect::<Result<Vec<EF>, VerifierError>>()?;
    Ok(MultilinearPoint::new(vars))
}

pub(crate) fn verify_initial_sumcheck_rounds<F, EF, Transcript>(
    transcript: &mut Transcript,
    initial_phase: InitialPhase,
    claimed_sum: &mut EF,
    num_rounds: usize,
    pow_bits: usize,
) -> Result<MultilinearPoint<EF>, VerifierError>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Transcript: Reader<F> + Reader<EF> + Challenge<EF> + Pow<F>,
{
    match initial_phase {
        InitialPhase::WithStatementSkip => {
            // Handle univariate skip optimization
            if num_rounds < K_SKIP_SUMCHECK {
                return Err(VerifierError::SumcheckFailed {
                    round: 0,
                    expected: "univariate skip optimization enabled".to_string(),
                    actual: "WithStatementSkip phase without skip conditions".to_string(),
                });
            }

            let skip_evaluations: Vec<EF> = transcript.read_many(1 << (K_SKIP_SUMCHECK + 1))?;
            // Verify sum over subgroup H (every other element starting from 0)
            let actual_sum: EF = skip_evaluations.iter().step_by(2).copied().sum();
            if actual_sum != *claimed_sum {
                return Err(VerifierError::SumcheckFailed {
                    round: 0,
                    expected: claimed_sum.to_string(),
                    actual: actual_sum.to_string(),
                });
            }

            // Verify pow
            transcript.pow(pow_bits)?;
            // Sample challenge for the skip round
            let r_skip = transcript.sample();

            // Interpolate to get the new claimed sum after skip folding
            let mat = RowMajorMatrix::new(skip_evaluations, 1);
            *claimed_sum = interpolate_subgroup(&mat, r_skip)[0];

            // Now process the remaining standard sumcheck rounds after the skip
            let remaining_rounds = num_rounds - K_SKIP_SUMCHECK;
            let mut randomness = vec![r_skip];

            randomness.extend(verify_standard_sumcheck_rounds(
                transcript,
                remaining_rounds,
                claimed_sum,
                pow_bits,
            )?);
            Ok(MultilinearPoint::new(randomness))
        }

        InitialPhase::WithoutStatement => {
            // No sumcheck - just sample folding randomness directly
            let randomness = transcript.sample_many(num_rounds);
            transcript.pow(pow_bits)?;
            Ok(MultilinearPoint::new(randomness))
        }

        InitialPhase::WithStatementClassic | InitialPhase::WithStatementSvo => Ok(
            verify_standard_sumcheck_rounds(transcript, num_rounds, claimed_sum, pow_bits)?,
        ),
    }
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
        fiat_shamir::{
            domain_separator::{DomainSeparator, SumcheckParams},
            transcript::{FiatShamirReader, FiatShamirWriter},
        },
        poly::evals::EvaluationsList,
        sumcheck::sumcheck_single::SumcheckSingle,
        whir::{
            constraints::{Constraint, statement::EqStatement},
            parameters::InitialPhase,
        },
    };

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;

    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

    // Digest size matches MyCompress output size (the 3rd parameter of TruncatedPermutation)
    const DIGEST_ELEMS: usize = 8;

    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_read_sumcheck_rounds_variants() {
        // Define a multilinear polynomial in 3 variables:
        // f(X0, X1, X2) = 1 + 2*X2 + 3*X1 + 4*X1*X2
        //              + 5*X0 + 6*X0*X2 + 7*X0*X1 + 8*X0*X1*X2
        let e1 = F::from_u64(1);
        let e2 = F::from_u64(2);
        let e3 = F::from_u64(3);
        let e4 = F::from_u64(4);
        let e5 = F::from_u64(5);
        let e6 = F::from_u64(6);
        let e7 = F::from_u64(7);
        let e8 = F::from_u64(8);

        let evals = EvaluationsList::new(vec![
            e1,
            e1 + e2,
            e1 + e3,
            e1 + e2 + e3 + e4,
            e1 + e5,
            e1 + e2 + e5 + e6,
            e1 + e3 + e5 + e7,
            e1 + e2 + e3 + e4 + e5 + e6 + e7 + e8,
        ]);

        // Define the actual polynomial function over EF4
        let f = |x0: EF, x1: EF, x2: EF| {
            x2 * e2
                + x1 * e3
                + x1 * x2 * e4
                + x0 * e5
                + x0 * x2 * e6
                + x0 * x1 * e7
                + x0 * x1 * x2 * e8
                + e1
        };

        let n_vars = evals.num_variables();
        assert_eq!(n_vars, 3);

        // Create a constraint system with evaluations of f at various points
        let mut statement = EqStatement::initialize(n_vars);

        let x_000 = MultilinearPoint::new(vec![EF::ZERO, EF::ZERO, EF::ZERO]);
        let x_100 = MultilinearPoint::new(vec![EF::ONE, EF::ZERO, EF::ZERO]);
        let x_110 = MultilinearPoint::new(vec![EF::ONE, EF::ONE, EF::ZERO]);
        let x_111 = MultilinearPoint::new(vec![EF::ONE, EF::ONE, EF::ONE]);
        let x_011 = MultilinearPoint::new(vec![EF::ZERO, EF::ONE, EF::ONE]);

        let f_000 = f(EF::ZERO, EF::ZERO, EF::ZERO);
        let f_100 = f(EF::ONE, EF::ZERO, EF::ZERO);
        let f_110 = f(EF::ONE, EF::ONE, EF::ZERO);
        let f_111 = f(EF::ONE, EF::ONE, EF::ONE);
        let f_011 = f(EF::ZERO, EF::ONE, EF::ONE);

        statement.add_evaluated_constraint(x_000, f_000);
        statement.add_evaluated_constraint(x_100, f_100);
        statement.add_evaluated_constraint(x_110, f_110);
        statement.add_evaluated_constraint(x_111, f_111);
        statement.add_evaluated_constraint(x_011, f_011);

        let folding_factor = 3;
        let pow_bits = 0;

        // Set up domain separator
        // - Add sumcheck
        let mut domsep: DomainSeparator<EF, F> = DomainSeparator::new(vec![]);
        domsep.add_sumcheck(&SumcheckParams {
            rounds: folding_factor,
            pow_bits,
            univariate_skip: None,
        });

        let mut rng = SmallRng::seed_from_u64(1);
        let mut challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));
        domsep.observe_domain_separator(&mut challenger);
        let mut transcript = FiatShamirWriter::init(challenger.clone());

        let constraint = Constraint::new_eq_only(EF::ONE, statement.clone());
        // Instantiate the prover with base field coefficients
        let (_, _) = SumcheckSingle::<F, EF>::from_base_evals(
            &mut transcript,
            &evals,
            folding_factor,
            pow_bits,
            &constraint,
        )
        .unwrap();

        let proof = transcript.finalize();
        let mut transcript = FiatShamirReader::init(proof.clone(), challenger.clone());

        let mut t = EvaluationsList::zero(statement.num_variables());
        let mut expected_initial_sum = EF::ZERO;
        statement.combine_hypercube::<F, false>(&mut t, &mut expected_initial_sum, EF::ONE);
        // Start with the claimed sum before folding
        let mut current_sum = expected_initial_sum;

        let mut expected = Vec::with_capacity(folding_factor);

        // First round: read c_0 = h(0) and c_2 (quadratic coefficient)
        let c_0: EF = transcript.read().unwrap();
        let c_2: EF = transcript.read().unwrap();
        let h_1 = current_sum - c_0;
        transcript.pow(pow_bits).unwrap();

        // Sample random challenge r_i ∈ EF4 and evaluate h_i(r_i)
        let r: EF = transcript.sample();
        // h(r) = c_2 * r^2 + (h(1) - c_0 - c_2) * r + c_0
        current_sum = c_2 * r.square() + (h_1 - c_0 - c_2) * r + c_0;
        expected.push(r);

        for _ in 0..folding_factor - 1 {
            // Read c_0 = h(0) and c_2 (quadratic coefficient), derive h(1) = claimed_sum - c_0
            let c_0: EF = transcript.read().unwrap();
            let c_2: EF = transcript.read().unwrap();
            let h_1 = current_sum - c_0;
            transcript.pow(pow_bits).unwrap();

            // Sample random challenge r
            let r: EF = transcript.sample();
            // h(r) = c_2 * r^2 + (h(1) - c_0 - c_2) * r + c_0
            current_sum = c_2 * r.square() + (h_1 - c_0 - c_2) * r + c_0;

            expected.push(r);
        }

        let mut transcript = FiatShamirReader::init(proof, challenger.clone());
        let randomness = verify_initial_sumcheck_rounds(
            &mut transcript,
            InitialPhase::WithStatementClassic,
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
        let evals = EvaluationsList::new((1..=num_points).map(F::from_u64).collect());

        assert_eq!(evals.num_variables(), NUM_VARS);

        // -------------------------------------------------------------
        // Construct a Statement by evaluating f at several Boolean points
        // These evaluations will serve as equality constraints
        // -------------------------------------------------------------
        let mut statement = EqStatement::initialize(NUM_VARS);
        for i in 0..5 {
            let bool_point: Vec<_> = (0..NUM_VARS)
                .map(|j| if (i >> j) & 1 == 1 { EF::ONE } else { EF::ZERO })
                .collect();
            let ml_point = MultilinearPoint::new(bool_point.clone());
            let expected_val = evals.evaluate_hypercube_base(&ml_point);
            statement.add_evaluated_constraint(ml_point, expected_val);
        }

        let folding_factor = NUM_VARS;
        let pow_bits = 0;

        // Set up domain separator
        let mut domsep: DomainSeparator<EF, F> = DomainSeparator::new(vec![]);
        domsep.add_sumcheck(&SumcheckParams {
            rounds: folding_factor,
            pow_bits,
            univariate_skip: Some(K_SKIP),
        });

        let mut rng = SmallRng::seed_from_u64(1);
        let mut challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));
        domsep.observe_domain_separator(&mut challenger);

        let mut transcript = FiatShamirWriter::init(challenger.clone());
        let constraint = Constraint::new_eq_only(EF::ONE, statement.clone());

        // Instantiate the prover with base field coefficients and univariate skip
        let (_, _) = SumcheckSingle::<F, EF>::with_skip(
            &evals,
            &mut transcript,
            folding_factor,
            pow_bits,
            K_SKIP,
            &constraint,
        )
        .unwrap();

        // Reconstruct verifier state to simulate the rounds
        let proof = transcript.finalize();
        let mut transcript = FiatShamirReader::init(proof.clone(), challenger.clone());

        let mut t = EvaluationsList::zero(statement.num_variables());
        let mut expected_initial_sum = EF::ZERO;
        statement.combine_hypercube::<F, false>(&mut t, &mut expected_initial_sum, EF::ONE);
        // Start with the claimed sum before folding
        let mut current_sum = expected_initial_sum;

        let mut expected = Vec::new();

        // First skipped round (wide DFT LDE)
        let skip_evaluations = &transcript.read_many(1 << (K_SKIP + 1)).unwrap();

        // Verify sum over subgroup H (every other element starting from 0)
        let actual_sum: EF = skip_evaluations.iter().step_by(2).copied().sum();
        assert_eq!(actual_sum, current_sum, "Skip round sum mismatch");

        transcript.pow(pow_bits).unwrap();
        // Sample challenge for the skip round
        let r_skip: EF = transcript.sample();
        expected.push(r_skip);

        // Interpolate to get the new claimed sum after skip folding
        let mat = RowMajorMatrix::new(skip_evaluations.clone(), 1);
        current_sum = interpolate_subgroup(&mat, r_skip)[0];

        // Remaining quadratic rounds after the skip
        let remaining_rounds = folding_factor - K_SKIP;
        for _ in 0..remaining_rounds {
            // Read c_0 = h(0) and c_2 (quadratic coefficient), derive h(1) = claimed_sum - c_0
            let c_0: EF = transcript.read().unwrap();
            let c_2: EF = transcript.read().unwrap();
            let h_1 = current_sum - c_0;
            transcript.pow(pow_bits).unwrap();

            // Sample random challenge r
            let r: EF = transcript.sample();
            // h(r) = c_2 * r^2 + (h_1 - c_0 - c_2) * r + c_0
            current_sum = c_2 * r.square() + (h_1 - c_0 - c_2) * r + c_0;

            expected.push(r);
        }

        let mut transcript = FiatShamirReader::init(proof, challenger.clone());
        let randomness = verify_initial_sumcheck_rounds(
            &mut transcript,
            InitialPhase::WithStatementSkip,
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

        let evals = EvaluationsList::new((1..=num_points).map(F::from_u64).collect());

        assert_eq!(evals.num_variables(), NUM_VARS);

        // Create a constraint system with evaluations of f at a point
        let mut statement = EqStatement::initialize(NUM_VARS);
        let constraint_point: Vec<_> = (0..NUM_VARS)
            .map(|j| if j % 2 == 0 { EF::ONE } else { EF::ZERO })
            .collect();
        let ml_point = MultilinearPoint::new(constraint_point);
        let expected_val = evals.evaluate_hypercube_base(&ml_point);
        statement.add_evaluated_constraint(ml_point, expected_val);

        let folding_factor = NUM_VARS;
        let pow_bits = 0;

        // Set up domain separator
        let mut domsep: DomainSeparator<EF, F> = DomainSeparator::new(vec![]);
        domsep.add_sumcheck(&SumcheckParams {
            rounds: folding_factor,
            pow_bits,
            univariate_skip: None,
        });

        let mut rng = SmallRng::seed_from_u64(1);
        let mut challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));
        domsep.observe_domain_separator(&mut challenger);
        let mut transcript = FiatShamirWriter::init(challenger.clone());

        let constraint = Constraint::new_eq_only(EF::ONE, statement.clone());

        // Instantiate the prover with base field coefficients using SVO
        let (_, _) = SumcheckSingle::<F, EF>::from_base_evals(
            &mut transcript,
            &evals,
            folding_factor,
            pow_bits,
            &constraint,
        )
        .unwrap();

        let proof = transcript.finalize();
        let mut transcript = FiatShamirReader::init(proof.clone(), challenger.clone());

        let (_, mut expected_initial_sum) = constraint.combine_new();
        // Start with the claimed sum before folding
        let mut current_sum = expected_initial_sum;

        let mut expected = Vec::with_capacity(folding_factor);

        for _ in 0..folding_factor {
            // Read c_0 = h(0) and c_2 (quadratic coefficient), derive h(1) = claimed_sum - c_0
            let c_0: EF = transcript.read().unwrap();
            let c_2: EF = transcript.read().unwrap();
            let h_1 = current_sum - c_0;
            transcript.pow(pow_bits).unwrap();

            // Sample random challenge r
            let r: EF = transcript.sample();
            // h(r) = c_2 * r^2 + (h(1) - c_0 - c_2) * r + c_0
            current_sum = c_2 * r.square() + (h_1 - c_0 - c_2) * r + c_0;

            expected.push(r);
        }

        let mut transcript = FiatShamirReader::init(proof, challenger.clone());
        let randomness = verify_initial_sumcheck_rounds(
            &mut transcript,
            InitialPhase::WithStatementSvo,
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
