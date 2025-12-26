use alloc::{vec, vec::Vec};

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::{DuplexChallenger, FieldChallenger, GrindingChallenger};
use p3_field::{PrimeCharacteristicRing, TwoAdicField, extension::BinomialExtensionField};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::{Rng, SeedableRng, rngs::SmallRng};

use super::sumcheck_single::SumcheckSingle;
use crate::{
    constant::K_SKIP_SUMCHECK,
    fiat_shamir::domain_separator::DomainSeparator,
    parameters::{FoldingFactor, ProtocolParameters, errors::SecurityAssumption},
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    whir::{
        constraints::{
            Constraint,
            evaluator::ConstraintPolyEvaluator,
            statement::{DomainStatement, EqStatement},
        },
        parameters::InitialPhaseConfig,
        proof::{InitialPhase, SumcheckData, WhirProof},
        verifier::sumcheck::{
            verify_final_sumcheck_rounds, verify_initial_sumcheck_rounds, verify_sumcheck_rounds,
        },
    },
};

type F = BabyBear;
type EF = BinomialExtensionField<BabyBear, 4>;
type Perm = Poseidon2BabyBear<16>;

type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

/// Creates a fresh `DomainSeparator` and `DuplexChallenger` using a fixed RNG seed.
fn domainsep_and_challenger() -> (DomainSeparator<EF, F>, MyChallenger) {
    // Initialize a small random number generator with a fixed seed.
    let mut rng = SmallRng::seed_from_u64(1);

    // Construct a Poseidon2 permutation instance using 128 bits of security from the RNG
    let perm = Perm::new_from_rng_128(&mut rng);

    // Create a new duplex challenger over the field `F` with this permutation
    let challenger = MyChallenger::new(perm);

    // Return a fresh (empty) domain separator and the challenger
    (DomainSeparator::new(vec![]), challenger)
}

fn create_test_protocol_params(
    folding_factor: FoldingFactor,
    initial_phase_config: InitialPhaseConfig,
) -> ProtocolParameters<MyHash, MyCompress> {
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);

    ProtocolParameters {
        initial_phase_config,
        security_level: 32,
        pow_bits: 0,
        rs_domain_initial_reduction_factor: 1,
        folding_factor,
        merkle_hash: MyHash::new(perm.clone()),
        merkle_compress: MyCompress::new(perm),
        soundness_type: SecurityAssumption::UniqueDecoding,
        starting_log_inv_rate: 1,
    }
}

fn make_constraint<Challenger>(
    challenger: &mut Challenger,
    constraint_evals: &mut Vec<EF>,
    num_vars: usize,
    num_eqs: usize,
    num_sels: usize,
    poly: &EvaluationsList<F>,
) -> Constraint<EF>
where
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    // To simulate stir point derivation derive domain generator
    let omega = F::two_adic_generator(num_vars);

    // Create a new empty eq and select statements of that arity
    let mut eq_statement = EqStatement::initialize(num_vars);
    let mut sel_statement = DomainStatement::initialize(num_vars, num_vars);

    // - Sample `num_eqs` univariate challenge points.
    // - Evaluate the sumcheck polynomial on them.
    // - Collect (point, eval) pairs for use in the statement and constraint aggregation.
    (0..num_eqs).for_each(|_| {
        // Sample a univariate field element from the prover's challenger.
        let point: EF = challenger.sample_algebra_element();

        // Expand it into a `num_vars`-dimensional multilinear point.
        let point = MultilinearPoint::expand_from_univariate(point, num_vars);

        // Evaluate the current sumcheck polynomial at the sampled point.
        let eval = poly.evaluate_hypercube_base(&point);

        // Store evaluation for verifier to read later.
        constraint_evals.push(eval);

        // Add the evaluation result to the transcript for Fiat-Shamir soundness.
        challenger.observe_algebra_element(eval);

        // Add the evaluation constraint: poly(point) == eval.
        eq_statement.add_evaluated_constraint(point, eval);
    });

    // - Sample `num_sels` univariate challenge points.
    // - Evaluate the sumcheck polynomial on them.
    // - Collect (var, eval) pairs for use in the statement and constraint aggregation.
    (0..num_sels).for_each(|_| {
        // Simulate stir point derivation
        let index: usize = challenger.sample_bits(num_vars);
        let var = omega.exp_u64(index as u64);

        // Evaluate the current sumcheck polynomial as univariate at the sampled variable.
        let eval = poly
            .iter()
            .rfold(EF::ZERO, |result, &coeff| result * var + coeff);

        // Store evaluation for verifier to read later.
        constraint_evals.push(eval);

        // Add the evaluation result to the transcript for Fiat-Shamir soundness.
        challenger.observe_algebra_element(eval);

        // Add the evaluation constraint: poly(point) == eval.
        sel_statement.add_constraint(index, eval);
    });

    // Return the constructed constraint with the alpha used for linear combination.
    let alpha: EF = challenger.sample_algebra_element();

    Constraint::new(alpha, eq_statement, sel_statement)
}

fn make_constraint_ext<Challenger>(
    challenger: &mut Challenger,
    constraint_evals: &mut Vec<EF>,
    num_vars: usize,
    num_eqs: usize,
    num_sels: usize,
    poly: &EvaluationsList<EF>,
) -> Constraint<EF>
where
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    // To simulate stir point derivation derive domain generator
    let omega = F::two_adic_generator(num_vars);

    // Create a new empty eq and select statements of that arity
    let mut eq_statement = EqStatement::initialize(num_vars);
    let mut sel_statement = DomainStatement::initialize(num_vars, num_vars);

    // - Sample `num_eqs` univariate challenge points.
    // - Evaluate the sumcheck polynomial on them.
    // - Collect (point, eval) pairs for use in the statement and constraint aggregation.
    (0..num_eqs).for_each(|_| {
        // Sample a univariate field element from the prover's challenger.
        let point: EF = challenger.sample_algebra_element();

        // Expand it into a `num_vars`-dimensional multilinear point.
        let point = MultilinearPoint::expand_from_univariate(point, num_vars);

        // Evaluate the current sumcheck polynomial at the sampled point.
        let eval = poly.evaluate_hypercube_ext::<F>(&point);

        // Store evaluation for verifier to read later.
        constraint_evals.push(eval);

        // Add the evaluation result to the transcript for Fiat-Shamir soundness.
        challenger.observe_algebra_element(eval);

        // Add the evaluation constraint: poly(point) == eval.
        eq_statement.add_evaluated_constraint(point, eval);
    });

    // - Sample `num_sels` univariate challenge points.
    // - Evaluate the sumcheck polynomial on them.
    // - Collect (var, eval) pairs for use in the statement and constraint aggregation.
    (0..num_sels).for_each(|_| {
        // Simulate stir point derivation
        let index: usize = challenger.sample_bits(num_vars);

        let var = omega.exp_u64(index as u64);

        // Evaluate the current sumcheck polynomial as univariate at the sampled variable.
        let eval = poly
            .iter()
            .rfold(EF::ZERO, |result, &coeff| result * var + coeff);

        // Store evaluation for verifier to read later.
        constraint_evals.push(eval);

        // Add the evaluation result to the transcript for Fiat-Shamir soundness.
        challenger.observe_algebra_element(eval);

        // Add the evaluation constraint: poly(point) == eval.
        sel_statement.add_constraint(index, eval);
    });

    // Return the constructed constraint with the alpha used for linear combination.
    let alpha: EF = challenger.sample_algebra_element();

    Constraint::new(alpha, eq_statement, sel_statement)
}

fn read_constraint<Challenger>(
    challenger: &mut Challenger,
    constraint_evals: &[EF],
    num_vars: usize,
    num_eqs: usize,
    num_sels: usize,
) -> Constraint<EF>
where
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    // Create a new statement that will hold all reconstructed constraints.
    let mut eq_statement = EqStatement::initialize(num_vars);

    // For each point, sample a challenge and read its corresponding evaluation from the proof.
    for &eval in constraint_evals.iter().take(num_eqs) {
        // Sample a univariate challenge and expand to a multilinear point.
        let point =
            MultilinearPoint::expand_from_univariate(challenger.sample_algebra_element(), num_vars);

        // Observe the evaluation to keep the challenger synchronized (must match prover)
        challenger.observe_algebra_element(eval);

        // Add the constraint: poly(point) == eval.
        eq_statement.add_evaluated_constraint(point, eval);
    }

    // Create a new statement that will hold all reconstructed constraints.
    let mut sel_statement = DomainStatement::initialize(num_vars, num_vars);

    // For each point, sample a challenge and read its corresponding evaluation from the proof.
    for i in 0..num_sels {
        // Simulate stir point derivation
        let index: usize = challenger.sample_bits(num_vars);

        // Read the committed evaluation corresponding to this point from constraint_evals.
        // Sel evaluations are stored after eq evaluations.
        let eval = constraint_evals[num_eqs + i];

        // Observe the evaluation to keep the challenger synchronized (must match prover)
        challenger.observe_algebra_element(eval);

        // Add the constraint: poly(point) == eval.
        sel_statement.add_constraint(index, eval);
    }

    Constraint::new(
        challenger.sample_algebra_element(),
        eq_statement,
        sel_statement,
    )
}

/// Runs an end-to-end prover-verifier test for the `SumcheckSingle` protocol with nested folding.
/// This test:
/// - Initializes a random multilinear polynomial over `F`.
/// - Runs the prover through several rounds of sumcheck folding.
/// - Verifies the transcript using the verifier.
/// - Checks that all reconstructed constraints match the original ones.
/// - Verifies that the final sum satisfies:
///   \begin{equation}
///   \text{sum} = f(r) \cdot \text{eq}(z, r)
///   \end{equation}
///
/// # Panics
/// Panics if:
/// - Any intermediate or final round does not produce expected sizes.
/// - Any constraint mismatches.
/// - The verifier-side evaluation differs from the expected one.
///
/// # Arguments
/// - `num_vars`: Number of variables of the initial polynomial.
/// - `folding_factors`: List of how many variables to fold per round.
/// - `num_eqs`: Number of equality statements to apply at each stage.
/// - `num_sels`: Number of select statements to apply at each stage.
#[allow(clippy::too_many_lines)]
fn run_sumcheck_test(
    num_vars: usize,
    folding_factor: FoldingFactor,
    num_eqs: &[usize],
    num_sels: &[usize],
) {
    let (num_rounds, final_rounds) = folding_factor.compute_number_of_rounds(num_vars);
    assert_eq!(num_eqs.len(), num_rounds + 1);
    assert_eq!(num_sels.len(), num_rounds + 1);
    folding_factor.check_validity(num_vars).unwrap();

    // Initialize a random multilinear polynomial with 2^num_vars evaluations.
    let mut rng = SmallRng::seed_from_u64(1);
    let poly = EvaluationsList::new((0..1 << num_vars).map(|_| rng.random()).collect());

    // PROVER
    let (domsep, challenger) = domainsep_and_challenger();
    let mut prover_challenger = challenger.clone();

    // Initialize proof and challenger
    let params =
        create_test_protocol_params(folding_factor, InitialPhaseConfig::WithStatementClassic);
    let mut proof = WhirProof::<F, EF, 8>::from_protocol_parameters(&params, num_vars);
    domsep.observe_domain_separator(&mut prover_challenger);

    // Store constraint evaluations for each round (prover writes, verifier reads)
    let mut all_constraint_evals: Vec<Vec<EF>> = Vec::new();

    // Create the initial constraint statement
    let mut constraint_evals: Vec<EF> = Vec::new();
    let constraint = make_constraint(
        &mut prover_challenger,
        &mut constraint_evals,
        num_vars,
        num_eqs[0],
        num_sels[0],
        &poly,
    );
    all_constraint_evals.push(constraint_evals);

    // ROUND 0
    let folding0 = folding_factor.at_round(0);
    // Extract sumcheck data from the initial phase
    let InitialPhase::WithStatement { ref mut sumcheck } = proof.initial_phase else {
        panic!("Expected WithStatement variant");
    };
    let (mut sumcheck, mut prover_randomness) = SumcheckSingle::from_base_evals(
        &poly,
        sumcheck,
        &mut prover_challenger,
        folding0,
        0,
        &constraint,
    );

    // Track how many variables remain to fold
    let mut num_vars_inter = num_vars - folding0;

    // INTERMEDIATE ROUNDS
    for (round, (&num_eq_points, &num_sel_points)) in
        num_eqs.iter().zip(num_sels.iter()).enumerate().skip(1)
    {
        let folding = folding_factor.at_round(round);
        // Sample new evaluation constraints and combine them into the sumcheck state
        let mut constraint_evals: Vec<EF> = Vec::new();
        let constraint = make_constraint_ext(
            &mut prover_challenger,
            &mut constraint_evals,
            num_vars_inter,
            num_eq_points,
            num_sel_points,
            &sumcheck.evals(),
        );
        all_constraint_evals.push(constraint_evals);

        // Compute and apply the next folding round
        let mut sumcheck_data: SumcheckData<EF, F> = SumcheckData::default();
        prover_randomness.extend(&sumcheck.compute_sumcheck_polynomials(
            &mut sumcheck_data,
            &mut prover_challenger,
            folding,
            0,
            Some(constraint),
        ));
        proof.rounds[round - 1].sumcheck = sumcheck_data;

        num_vars_inter -= folding;

        // Check that the number of variables and evaluations match the expected values
        assert_eq!(sumcheck.num_variables(), num_vars_inter);
        assert_eq!(sumcheck.num_evals(), 1 << num_vars_inter);
    }

    // Ensure we've folded all variables.
    assert_eq!(num_vars_inter, final_rounds);

    // FINAL ROUND
    let mut sumcheck_data: SumcheckData<EF, F> = SumcheckData::default();
    prover_randomness.extend(&sumcheck.compute_sumcheck_polynomials(
        &mut sumcheck_data,
        &mut prover_challenger,
        final_rounds,
        0,
        None,
    ));
    proof.set_final_sumcheck_data(sumcheck_data);
    let final_folded_value = sumcheck.evals().as_constant().unwrap();

    assert_eq!(sumcheck.num_variables(), 0);
    assert_eq!(sumcheck.num_evals(), 1);

    // Final folded value must match f(r)
    assert_eq!(
        poly.evaluate_hypercube_base(&prover_randomness),
        final_folded_value
    );
    // Commit final result to Fiat-Shamir transcript
    prover_challenger.observe_algebra_element(final_folded_value);

    // VERIFIER
    let mut verifer_challenger = challenger;

    // Running total for the verifier’s sum of constraint combinations
    let mut sum = EF::ZERO;

    // Point `r` is constructed over rounds using verifier-chosen challenges
    let mut verifier_randomness = MultilinearPoint::new(vec![]);

    // All constraints used by the verifier across rounds
    let mut constraints = vec![];

    // Recompute the same variable count as prover had
    let mut num_vars_inter = num_vars;

    // Apply domain separator to verifier challenger
    domsep.observe_domain_separator(&mut verifer_challenger);

    // VERIFY INITIAL ROUND (round 0)
    {
        // Reconstruct round constraint from transcript
        let constraint = read_constraint(
            &mut verifer_challenger,
            &all_constraint_evals[0],
            num_vars_inter,
            num_eqs[0],
            num_sels[0],
        );
        // Accumulate the weighted sum of constraint values
        constraint.combine_evals(&mut sum);
        // Save constraints for later equality check
        constraints.push(constraint);

        // Verify initial sumcheck rounds using the initial phase
        let folding = folding_factor.at_round(0);
        verifier_randomness.extend(
            &verify_initial_sumcheck_rounds(
                &proof.initial_phase,
                &mut verifer_challenger,
                &mut sum,
                folding,
                0,
            )
            .unwrap(),
        );

        num_vars_inter -= folding;
    }

    // VERIFY INTERMEDIATE ROUNDS (rounds 1 to num_rounds)
    for (round_idx, (&num_eq_points, &num_sel_points)) in
        num_eqs.iter().zip(num_sels.iter()).enumerate().skip(1)
    {
        // Reconstruct round constraint from transcript
        let constraint = read_constraint(
            &mut verifer_challenger,
            &all_constraint_evals[round_idx],
            num_vars_inter,
            num_eq_points,
            num_sel_points,
        );
        // Accumulate the weighted sum of constraint values
        constraint.combine_evals(&mut sum);
        // Save constraints for later equality check
        constraints.push(constraint);

        // Extend r with verifier's folding challenges
        // Note: proof.rounds[round_idx - 1] because rounds are 0-indexed but we start at round 1
        let folding = folding_factor.at_round(round_idx);
        verifier_randomness.extend(
            &verify_sumcheck_rounds(
                &proof.rounds[round_idx - 1],
                &mut verifer_challenger,
                &mut sum,
                folding,
                0,
            )
            .unwrap(),
        );

        num_vars_inter -= folding;
    }

    // Final round check
    verifier_randomness.extend(
        &verify_final_sumcheck_rounds(
            proof.final_sumcheck.as_ref(),
            &mut verifer_challenger,
            &mut sum,
            final_rounds,
            0,
        )
        .unwrap(),
    );

    // Check that the randomness vectors are the same
    assert_eq!(prover_randomness, verifier_randomness);

    // CHECK EQ(z, r) WEIGHT POLY
    //
    // No skip optimization, so the first round is treated as a standard sumcheck round.
    let evaluator = ConstraintPolyEvaluator::new(num_vars, folding_factor, None);
    let weights =
        evaluator.eval_constraints_poly::<F, _>(&constraints, &verifier_randomness.reversed());

    // CHECK SUM == f(r) * weights(z, r)
    assert_eq!(sum, final_folded_value * weights);
}

/// Runs an end-to-end prover-verifier test for the `SumcheckSingle` protocol with skipping enabled.
///
/// This variant uses the univariate skip optimization: in the first round, `K_SKIP_SUMCHECK`
/// variables are folded at once using a low-degree extension (LDE) and DFT interpolation.
///
/// It checks:
/// - Correct commitment and folding of the prover.
/// - Verifier consistency and challenge reconstruction.
/// - Final algebraic relation:
///   \begin{equation}
///   \text{sum} = f(r) \cdot \text{eq}(z, r)
///   \end{equation}
///
/// # Arguments
/// - `num_vars`: Number of variables of the initial polynomial.
/// - `folding_factors`: List of how many variables to fold per round.
/// - `num_eqs`: Number of equality statements to apply at each stage.
/// - `num_sels`: Number of select statements to apply at each stage.
#[allow(clippy::too_many_lines)]
fn run_sumcheck_test_skips(
    num_vars: usize,
    folding_factor: FoldingFactor,
    num_eq_points: &[usize],
    num_sel_points: &[usize],
) {
    let (num_rounds, final_rounds) = folding_factor.compute_number_of_rounds(num_vars);
    assert_eq!(num_eq_points.len(), num_rounds + 1);
    assert_eq!(num_sel_points.len(), num_rounds + 1);
    folding_factor.check_validity(num_vars).unwrap();

    // SETUP POLYNOMIAL
    //
    // Generate a random multilinear polynomial of arity `num_vars`
    let mut rng = SmallRng::seed_from_u64(1);
    let poly = EvaluationsList::new((0..1 << num_vars).map(|_| rng.random()).collect());

    // PROVER SIDE
    let (domsep, challenger) = domainsep_and_challenger();
    let mut prover_challenger = challenger.clone();

    // Initialize proof and challenger
    let params = create_test_protocol_params(
        folding_factor,
        InitialPhaseConfig::WithStatementUnivariateSkip,
    );
    let mut proof = WhirProof::<F, EF, 8>::from_protocol_parameters(&params, num_vars);
    domsep.observe_domain_separator(&mut prover_challenger);

    // Store constraint evaluations for each round (prover writes, verifier reads)
    let mut all_constraint_evals: Vec<Vec<EF>> = Vec::new();

    // Sample and commit initial evaluation constraints
    let mut constraint_evals: Vec<EF> = Vec::new();
    let constraint = make_constraint(
        &mut prover_challenger,
        &mut constraint_evals,
        num_vars,
        num_eq_points[0],
        num_sel_points[0],
        &poly,
    );
    all_constraint_evals.push(constraint_evals);
    constraint.validate_for_skip_case();

    // ROUND 0
    // Initialize sumcheck with univariate skip (skips K_SKIP_SUMCHECK)
    let folding0 = folding_factor.at_round(0);
    // Extract skip data from the initial phase
    let InitialPhase::WithStatementSkip(ref mut skip_data) = proof.initial_phase else {
        panic!("Expected WithStatementSkip variant");
    };
    let (mut sumcheck, mut prover_randomness) = SumcheckSingle::with_skip(
        &poly,
        skip_data,
        &mut prover_challenger,
        folding0,
        0,
        K_SKIP_SUMCHECK,
        &constraint,
    );

    // Track how many variables remain after folding
    let mut num_vars_inter = num_vars - folding0;

    // INTERMEDIATE ROUNDS
    for (round, (&num_eq_points, &num_sel_points)) in num_eq_points
        .iter()
        .zip(num_sel_points.iter())
        .enumerate()
        .skip(1)
    {
        let folding = folding_factor.at_round(round);
        // Sample new evaluation constraints and combine them into the sumcheck state
        let mut constraint_evals: Vec<EF> = Vec::new();
        let constraint = make_constraint_ext(
            &mut prover_challenger,
            &mut constraint_evals,
            num_vars_inter,
            num_eq_points,
            num_sel_points,
            &sumcheck.evals(),
        );
        all_constraint_evals.push(constraint_evals);

        // Fold the sumcheck polynomial again and extend randomness vector
        let mut sumcheck_data: SumcheckData<EF, F> = SumcheckData::default();
        prover_randomness.extend(&sumcheck.compute_sumcheck_polynomials(
            &mut sumcheck_data,
            &mut prover_challenger,
            folding,
            0,
            Some(constraint),
        ));
        proof.rounds[round - 1].sumcheck = sumcheck_data;

        num_vars_inter -= folding;

        // Sanity check: number of variables and evaluations should be correct
        assert_eq!(sumcheck.num_variables(), num_vars_inter);
        assert_eq!(sumcheck.num_evals(), 1 << num_vars_inter);
    }

    // Ensure we've folded all variables.
    assert_eq!(num_vars_inter, final_rounds);

    // FINAL ROUND
    let mut sumcheck_data: SumcheckData<EF, F> = SumcheckData::default();
    prover_randomness.extend(&sumcheck.compute_sumcheck_polynomials(
        &mut sumcheck_data,
        &mut prover_challenger,
        final_rounds,
        0,
        None,
    ));
    proof.set_final_sumcheck_data(sumcheck_data);

    // After final round, polynomial must collapse to a constant
    assert_eq!(sumcheck.num_variables(), 0);
    assert_eq!(sumcheck.num_evals(), 1);

    // Final constant should be f(r), where r is the accumulated challenge point
    let final_folded_value = sumcheck.evals().as_constant().unwrap();

    // Verify that the final folded value matches the polynomial evaluation under skip semantics.
    //
    // The prover's randomness is in the format: [r_skip, r_1, r_2, ..., r_j], where:
    //   - r_skip is the single challenge for the k skipped variables (first element)
    //   - r_1, ..., r_j are the j challenges for the non-skipped variables
    //
    // Verify: f̂(r_skip, r_1, ..., rj) == final_folded_value
    assert_eq!(
        poly.evaluate_with_univariate_skip(&prover_randomness, K_SKIP_SUMCHECK),
        final_folded_value
    );
    // Commit final result to Fiat-Shamir transcript
    prover_challenger.observe_algebra_element(final_folded_value);

    // VERIFIER SIDE
    let mut verifier_challenger = challenger;

    // Running total for the verifier's sum of constraint combinations
    let mut sum = EF::ZERO;

    // Point `r` is constructed over rounds using verifier-chosen challenges
    let mut verifier_randomness = MultilinearPoint::new(vec![]);

    // All constraints used by the verifier across rounds
    let mut constraints = vec![];

    // Recompute the same variable count as prover had
    let mut num_vars_inter = num_vars;

    // Apply domain separator to verifier challenger
    domsep.observe_domain_separator(&mut verifier_challenger);

    // VERIFY INITIAL ROUND (round 0)
    {
        // Reconstruct round constraint from transcript
        let constraint = read_constraint(
            &mut verifier_challenger,
            &all_constraint_evals[0],
            num_vars_inter,
            num_eq_points[0],
            num_sel_points[0],
        );
        // Accumulate the weighted sum of constraint values
        constraint.combine_evals(&mut sum);
        // Save constraints for later equality check
        constraints.push(constraint);

        // Verify initial sumcheck rounds using the initial phase (with skip)
        let folding = folding_factor.at_round(0);
        verifier_randomness.extend(
            &verify_initial_sumcheck_rounds(
                &proof.initial_phase,
                &mut verifier_challenger,
                &mut sum,
                folding,
                0,
            )
            .unwrap(),
        );

        num_vars_inter -= folding;
    }

    // VERIFY INTERMEDIATE ROUNDS (rounds 1 to num_rounds)
    for (round_idx, (&num_eq_pts, &num_sel_pts)) in num_eq_points
        .iter()
        .zip(num_sel_points.iter())
        .enumerate()
        .skip(1)
    {
        // Reconstruct round constraint from transcript
        let constraint = read_constraint(
            &mut verifier_challenger,
            &all_constraint_evals[round_idx],
            num_vars_inter,
            num_eq_pts,
            num_sel_pts,
        );
        // Accumulate the weighted sum of constraint values
        constraint.combine_evals(&mut sum);
        // Save constraints for later equality check
        constraints.push(constraint);

        // Extend r with verifier's folding challenges
        let folding = folding_factor.at_round(round_idx);
        verifier_randomness.extend(
            &verify_sumcheck_rounds(
                &proof.rounds[round_idx - 1],
                &mut verifier_challenger,
                &mut sum,
                folding,
                0,
            )
            .unwrap(),
        );

        num_vars_inter -= folding;
    }

    // FINAL FOLDING
    verifier_randomness.extend(
        &verify_final_sumcheck_rounds(
            proof.final_sumcheck.as_ref(),
            &mut verifier_challenger,
            &mut sum,
            final_rounds,
            0,
        )
        .unwrap(),
    );

    // Check that the randomness vectors are the same
    assert_eq!(prover_randomness, verifier_randomness);

    // EVALUATE EQ(z, r) VIA CONSTRAINTS
    //
    // Evaluate eq(z, r) using the unified constraint evaluation function.
    let evaluator = ConstraintPolyEvaluator::new(num_vars, folding_factor, Some(K_SKIP_SUMCHECK));
    let weights = evaluator.eval_constraints_poly::<F, _>(&constraints, &verifier_randomness);

    // FINAL SUMCHECK CHECK
    //
    // Main equation: sum == f(r) · eq(z, r)
    assert_eq!(sum, final_folded_value * weights);
}

/// Runs an end-to-end prover-verifier test for the `SumcheckSingle` protocol with SVO optimization.
#[allow(clippy::too_many_lines)]
fn run_sumcheck_test_svo(
    num_vars: usize,
    folding_factor: FoldingFactor,
    num_eqs: &[usize],
    num_sels: &[usize],
) {
    let (num_rounds, final_rounds) = folding_factor.compute_number_of_rounds(num_vars);
    assert_eq!(num_eqs.len(), num_rounds + 1);
    assert_eq!(num_sels.len(), num_rounds + 1);
    folding_factor.check_validity(num_vars).unwrap();

    // Initialize a random multilinear polynomial with 2^num_vars evaluations.
    let mut rng = SmallRng::seed_from_u64(1);
    let poly = EvaluationsList::new((0..1 << num_vars).map(|_| rng.random()).collect());

    // PROVER
    let (domsep, challenger) = domainsep_and_challenger();
    let mut prover_challenger = challenger.clone();

    // Initialize proof and challenger
    let params = create_test_protocol_params(folding_factor, InitialPhaseConfig::WithStatementSvo);
    let mut proof = WhirProof::<F, EF, 8>::from_protocol_parameters(&params, num_vars);
    domsep.observe_domain_separator(&mut prover_challenger);

    // Store constraint evaluations for each round (prover writes, verifier reads)
    let mut all_constraint_evals: Vec<Vec<EF>> = Vec::new();

    // Create the initial constraint statement
    let mut constraint_evals: Vec<EF> = Vec::new();
    let constraint = make_constraint(
        &mut prover_challenger,
        &mut constraint_evals,
        num_vars,
        num_eqs[0],
        num_sels[0],
        &poly,
    );
    all_constraint_evals.push(constraint_evals);

    // ROUND 0
    let folding0 = folding_factor.at_round(0);
    // Extract sumcheck data from the initial phase
    let InitialPhase::WithStatementSvo { ref mut sumcheck } = proof.initial_phase else {
        panic!("Expected WithStatementSvo variant");
    };
    let (mut sumcheck, mut prover_randomness) = SumcheckSingle::from_base_evals_svo(
        &poly,
        sumcheck,
        &mut prover_challenger,
        folding0,
        0,
        &constraint,
    );

    // Track how many variables remain to fold
    let mut num_vars_inter = num_vars - folding0;

    // INTERMEDIATE ROUNDS
    for (round, (&num_eq_points, &num_sel_points)) in
        num_eqs.iter().zip(num_sels.iter()).enumerate().skip(1)
    {
        let folding = folding_factor.at_round(round);
        // Sample new evaluation constraints and combine them into the sumcheck state
        let mut constraint_evals: Vec<EF> = Vec::new();
        let constraint = make_constraint_ext(
            &mut prover_challenger,
            &mut constraint_evals,
            num_vars_inter,
            num_eq_points,
            num_sel_points,
            &sumcheck.evals(),
        );
        all_constraint_evals.push(constraint_evals);

        // Compute and apply the next folding round
        let mut sumcheck_data: SumcheckData<EF, F> = SumcheckData::default();
        prover_randomness.extend(&sumcheck.compute_sumcheck_polynomials(
            &mut sumcheck_data,
            &mut prover_challenger,
            folding,
            0,
            Some(constraint),
        ));
        proof.rounds[round - 1].sumcheck = sumcheck_data;

        num_vars_inter -= folding;

        // Check that the number of variables and evaluations match the expected values
        assert_eq!(sumcheck.num_variables(), num_vars_inter);
        assert_eq!(sumcheck.num_evals(), 1 << num_vars_inter);
    }

    // Ensure we've folded all variables.
    assert_eq!(num_vars_inter, final_rounds);

    // FINAL ROUND
    let mut sumcheck_data: SumcheckData<EF, F> = SumcheckData::default();
    prover_randomness.extend(&sumcheck.compute_sumcheck_polynomials(
        &mut sumcheck_data,
        &mut prover_challenger,
        final_rounds,
        0,
        None,
    ));
    proof.set_final_sumcheck_data(sumcheck_data);

    assert_eq!(sumcheck.num_variables(), 0);
    assert_eq!(sumcheck.num_evals(), 1);

    // Final folded value must match f(r)
    let final_folded_value = sumcheck.evals().as_constant().unwrap();
    assert_eq!(
        poly.evaluate_hypercube_base(&prover_randomness),
        final_folded_value
    );
    // Commit final result to Fiat-Shamir transcript
    prover_challenger.observe_algebra_element(final_folded_value);

    // VERIFIER
    let mut verifier_challenger = challenger;

    // Running total for the verifier's sum of constraint combinations
    let mut sum = EF::ZERO;

    // Point `r` is constructed over rounds using verifier-chosen challenges
    let mut verifier_randomness = MultilinearPoint::new(vec![]);

    // All constraints used by the verifier across rounds
    let mut constraints = vec![];

    // Recompute the same variable count as prover had
    let mut num_vars_inter = num_vars;

    // Apply domain separator to verifier challenger
    domsep.observe_domain_separator(&mut verifier_challenger);

    // VERIFY INITIAL ROUND (round 0)
    {
        // Reconstruct round constraint from transcript
        let constraint = read_constraint(
            &mut verifier_challenger,
            &all_constraint_evals[0],
            num_vars_inter,
            num_eqs[0],
            num_sels[0],
        );
        // Accumulate the weighted sum of constraint values
        constraint.combine_evals(&mut sum);
        // Save constraints for later equality check
        constraints.push(constraint);

        // Verify initial sumcheck rounds
        let folding = folding_factor.at_round(0);
        verifier_randomness.extend(
            &verify_initial_sumcheck_rounds(
                &proof.initial_phase,
                &mut verifier_challenger,
                &mut sum,
                folding,
                0,
            )
            .unwrap(),
        );

        num_vars_inter -= folding;
    }

    // VERIFY INTERMEDIATE ROUNDS
    for (round_idx, (&num_eq_pts, &num_sel_pts)) in
        num_eqs.iter().zip(num_sels.iter()).enumerate().skip(1)
    {
        // Reconstruct round constraint from transcript
        let constraint = read_constraint(
            &mut verifier_challenger,
            &all_constraint_evals[round_idx],
            num_vars_inter,
            num_eq_pts,
            num_sel_pts,
        );
        // Accumulate the weighted sum of constraint values
        constraint.combine_evals(&mut sum);
        // Save constraints for later equality check
        constraints.push(constraint);

        // Extend r with verifier's folding challenges
        let folding = folding_factor.at_round(round_idx);
        verifier_randomness.extend(
            &verify_sumcheck_rounds(
                &proof.rounds[round_idx - 1],
                &mut verifier_challenger,
                &mut sum,
                folding,
                0,
            )
            .unwrap(),
        );

        num_vars_inter -= folding;
    }

    // FINAL FOLDING
    verifier_randomness.extend(
        &verify_final_sumcheck_rounds(
            proof.final_sumcheck.as_ref(),
            &mut verifier_challenger,
            &mut sum,
            final_rounds,
            0,
        )
        .unwrap(),
    );

    // Check that the randomness vectors are the same
    assert_eq!(prover_randomness, verifier_randomness);

    // CHECK EQ(z, r) WEIGHT POLY
    //
    // No skip optimization, so the first round is treated as a standard sumcheck round.
    let evaluator = ConstraintPolyEvaluator::new(num_vars, folding_factor, None);
    let weights =
        evaluator.eval_constraints_poly::<F, _>(&constraints, &verifier_randomness.reversed());

    // CHECK SUM == f(r) * weights(z, r)
    assert_eq!(sum, final_folded_value * weights);
}

#[test]
fn test_sumcheck_prover_without_skip() {
    let mut rng = SmallRng::seed_from_u64(0);

    for num_vars in 0..=10 {
        for folding_factor in 1..=num_vars {
            for _ in 0..100 {
                let folding_factor = FoldingFactor::Constant(folding_factor);
                let num_rounds = folding_factor.compute_number_of_rounds(num_vars).0 + 1;
                let num_eq_points = (0..num_rounds)
                    .map(|_| rng.random_range(0..=2))
                    .collect::<Vec<_>>();
                let num_sel_points = (0..num_rounds)
                    .map(|_| rng.random_range(0..=2))
                    .collect::<Vec<_>>();
                run_sumcheck_test(num_vars, folding_factor, &num_eq_points, &num_sel_points);
            }
        }
    }
}

#[test]
fn test_sumcheck_prover_svo() {
    for num_vars in &[6, 8, 10, 12, 14, 16, 18, 20, 22, 24] {
        let folding_factor = *num_vars;
        let folding_factor = FoldingFactor::Constant(folding_factor);
        let num_eq_points = vec![1];
        let num_sel_points = vec![0];
        run_sumcheck_test_svo(*num_vars, folding_factor, &num_eq_points, &num_sel_points);
    }
}

#[test]
fn test_sumcheck_prover_with_skip() {
    let mut rng = SmallRng::seed_from_u64(0);

    for num_vars in 8..=10 {
        for folding_factor in 2..=num_vars {
            for _ in 0..100 {
                let folding_factor = FoldingFactor::Constant(folding_factor);
                for k_skip in 2..folding_factor.at_round(0) {
                    if k_skip < K_SKIP_SUMCHECK {
                        continue;
                    }

                    let num_rounds = folding_factor.compute_number_of_rounds(num_vars).0 + 1;
                    let num_eq_points = (0..num_rounds)
                        .map(|_| rng.random_range(0..=2))
                        .collect::<Vec<usize>>();
                    let mut num_sel_points = (0..num_rounds)
                        .map(|_| rng.random_range(0..=2))
                        .collect::<Vec<usize>>();
                    num_sel_points[0] = 0;
                    run_sumcheck_test_skips(
                        num_vars,
                        folding_factor,
                        &num_eq_points,
                        &num_sel_points,
                    );
                }
            }
        }
    }
}
