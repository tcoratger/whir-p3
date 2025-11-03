use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::{DuplexChallenger, FieldChallenger, GrindingChallenger};
use p3_field::{
    ExtensionField, Field, PrimeCharacteristicRing, TwoAdicField, extension::BinomialExtensionField,
};
use p3_interpolation::interpolate_subgroup;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::{Rng, SeedableRng, rngs::SmallRng};

use super::sumcheck_single::SumcheckSingle;
use crate::{
    constant::K_SKIP_SUMCHECK,
    fiat_shamir::{
        domain_separator::DomainSeparator, prover::ProverState, verifier::VerifierState,
    },
    parameters::FoldingFactor,
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    whir::{
        constraints::{
            evaluator::{Constraint, ConstraintPolyEvaluator},
            statement::{EqStatement, SelectStatement},
        },
        verifier::sumcheck::{
            verify_sumcheck_rounds, verify_sumcheck_rounds_svo,
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

/// Constructs a fresh `ProverState` using a domain separator and challenger.
fn prover() -> ProverState<F, EF, MyChallenger> {
    // Create a domain separator and challenger using deterministic RNG
    let (domsep, challenger) = domainsep_and_challenger();

    // Use the domain separator to construct a new prover state with the given challenger
    domsep.to_prover_state(challenger)
}

/// Constructs a fresh `VerifierState` from a given proof, using a domain separator and challenger.
fn verifier(proof: Vec<F>) -> VerifierState<F, EF, MyChallenger> {
    // Create a domain separator and challenger using deterministic RNG
    let (domsep, challenger) = domainsep_and_challenger();

    // Use the domain separator to construct a new verifier state with the provided proof and challenger
    domsep.to_verifier_state(proof, challenger)
}

fn make_constraint<Challenger>(
    prover: &mut ProverState<F, EF, Challenger>,
    num_vars: usize,
    num_eqs: usize,
    num_sels: usize,
    poly: &EvaluationsList<F>,
) -> Constraint<F, EF>
where
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    // To simulate stir point derivation derive domain generator
    let omega = F::two_adic_generator(num_vars);

    // Create a new empty eq and select statements of that arity
    let mut eq_statement = EqStatement::initialize(num_vars);
    let mut sel_statement = SelectStatement::initialize(num_vars);

    // - Sample `num_eqs` univariate challenge points.
    // - Evaluate the sumcheck polynomial on them.
    // - Collect (point, eval) pairs for use in the statement and constraint aggregation.
    (0..num_eqs).for_each(|_| {
        // Sample a univariate field element from the prover's challenger.
        let point = prover.sample();

        // Expand it into a `num_vars`-dimensional multilinear point.
        let point = MultilinearPoint::expand_from_univariate(point, num_vars);

        // Evaluate the current sumcheck polynomial at the sampled point.
        let eval = poly.evaluate_hypercube(&point);

        // Add the evaluation result to the transcript for Fiat-Shamir soundness.
        prover.add_extension_scalar(eval);

        // Add the evaluation constraint: poly(point) == eval.
        eq_statement.add_evaluated_constraint(point, eval);
    });

    // - Sample `num_sels` univariate challenge points.
    // - Evaluate the sumcheck polynomial on them.
    // - Collect (var, eval) pairs for use in the statement and constraint aggregation.
    (0..num_sels).for_each(|_| {
        // Simulate stir point derivation
        let index: usize = prover.sample_bits(num_vars);
        let var = omega.exp_u64(index as u64);

        // Evaluate the current sumcheck polynomial as univariate at the sampled variable.
        let eval = poly
            .iter()
            .rfold(EF::ZERO, |result, &coeff| result * var + coeff);

        // Add the evaluation result to the transcript for Fiat-Shamir soundness.
        prover.add_extension_scalar(eval);

        // Add the evaluation constraint: poly(point) == eval.
        sel_statement.add_constraint(var, eval);
    });

    // Return the constructed constraint with the alpha used for linear combination.
    Constraint::new(prover.sample(), eq_statement, sel_statement)
}

fn make_constraint_ext<Challenger>(
    prover: &mut ProverState<F, EF, Challenger>,
    num_vars: usize,
    num_eqs: usize,
    num_sels: usize,
    poly: &EvaluationsList<EF>,
) -> Constraint<F, EF>
where
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    // To simulate stir point derivation derive domain generator
    let omega = F::two_adic_generator(num_vars);

    // Create a new empty eq and select statements of that arity
    let mut eq_statement = EqStatement::initialize(num_vars);
    let mut sel_statement = SelectStatement::initialize(num_vars);

    // - Sample `num_eqs` univariate challenge points.
    // - Evaluate the sumcheck polynomial on them.
    // - Collect (point, eval) pairs for use in the statement and constraint aggregation.
    (0..num_eqs).for_each(|_| {
        // Sample a univariate field element from the prover's challenger.
        let point = prover.sample();

        // Expand it into a `num_vars`-dimensional multilinear point.
        let point = MultilinearPoint::expand_from_univariate(point, num_vars);

        // Evaluate the current sumcheck polynomial at the sampled point.
        let eval = poly.evaluate_hypercube(&point);

        // Add the evaluation result to the transcript for Fiat-Shamir soundness.
        prover.add_extension_scalar(eval);

        // Add the evaluation constraint: poly(point) == eval.
        eq_statement.add_evaluated_constraint(point, eval);
    });

    // - Sample `num_sels` univariate challenge points.
    // - Evaluate the sumcheck polynomial on them.
    // - Collect (var, eval) pairs for use in the statement and constraint aggregation.
    (0..num_sels).for_each(|_| {
        // Simulate stir point derivation
        let index: usize = prover.sample_bits(num_vars);
        let var = omega.exp_u64(index as u64);

        // Evaluate the current sumcheck polynomial as univariate at the sampled variable.
        let eval = poly
            .iter()
            .rfold(EF::ZERO, |result, &coeff| result * var + coeff);

        // Add the evaluation result to the transcript for Fiat-Shamir soundness.
        prover.add_extension_scalar(eval);

        // Add the evaluation constraint: poly(point) == eval.
        sel_statement.add_constraint(var, eval);
    });

    // Return the constructed constraint with the alpha used for linear combination.
    Constraint::new(prover.sample(), eq_statement, sel_statement)
}

fn read_constraint<Challenger>(
    verifier: &mut VerifierState<F, EF, Challenger>,
    num_vars: usize,
    num_eqs: usize,
    num_sels: usize,
) -> Constraint<F, EF>
where
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    // Create a new statement that will hold all reconstructed constraints.
    let mut eq_statement = EqStatement::initialize(num_vars);

    // For each point, sample a challenge and read its corresponding evaluation from the transcript.
    for _ in 0..num_eqs {
        // Sample a univariate challenge and expand to a multilinear point.
        let point = MultilinearPoint::expand_from_univariate(verifier.sample(), num_vars);

        // Read the committed evaluation corresponding to this point from the proof data.
        let eval = verifier.next_extension_scalar().unwrap();

        // Add the constraint: poly(point) == eval.
        eq_statement.add_evaluated_constraint(point, eval);
    }

    // Create a new statement that will hold all reconstructed constraints.
    let mut sel_statement = SelectStatement::<F, EF>::initialize(num_vars);

    // To simulate stir point derivation derive domain generator
    let omega = F::two_adic_generator(num_vars);

    // For each point, sample a challenge and read its corresponding evaluation from the transcript.
    for _ in 0..num_sels {
        // Simulate stir point derivation
        let index: usize = verifier.sample_bits(num_vars);
        let var = omega.exp_u64(index as u64);

        // Read the committed evaluation corresponding to this point from the proof data.
        let eval = verifier.next_extension_scalar().unwrap();

        // Add the constraint: poly(point) == eval.
        sel_statement.add_constraint(var, eval);
    }

    Constraint::new(verifier.sample(), eq_statement, sel_statement)
}

/// Helper to extend a `MultilinearPoint` by creating a new one.
fn extend_point<F: Field>(
    point: &MultilinearPoint<F>,
    extension: &MultilinearPoint<F>,
) -> MultilinearPoint<F> {
    let new_coords: Vec<_> = extension
        .as_slice()
        .iter()
        .chain(point.iter())
        .copied()
        .collect();
    MultilinearPoint::new(new_coords)
}

/// Runs an end-to-end prover-verifier test for the `SumcheckSingle` protocol with nested folding.
///
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
    let prover = &mut prover();

    // Create the initial constraint statement
    let constraint = make_constraint(prover, num_vars, num_eqs[0], num_sels[0], &poly);

    // ROUND 0
    let folding0 = folding_factor.at_round(0);
    let (mut sumcheck, mut prover_randomness) =
        SumcheckSingle::from_base_evals(&poly, prover, folding0, 0, &constraint);

    // Track how many variables remain to fold
    let mut num_vars_inter = num_vars - folding0;

    // INTERMEDIATE ROUNDS
    for (round, (&num_eq_points, &num_sel_points)) in
        num_eqs.iter().zip(num_sels.iter()).enumerate().skip(1)
    {
        let folding = folding_factor.at_round(round);
        // Sample new evaluation constraints and combine them into the sumcheck state
        let constraint = make_constraint_ext(
            prover,
            num_vars_inter,
            num_eq_points,
            num_sel_points,
            &sumcheck.evals,
        );

        // Compute and apply the next folding round
        prover_randomness = extend_point(
            &prover_randomness,
            &sumcheck.compute_sumcheck_polynomials(prover, folding, 0, Some(constraint)),
        );

        num_vars_inter -= folding;

        // Check that the number of variables and evaluations match the expected values
        assert_eq!(sumcheck.evals.num_variables(), num_vars_inter);
        assert_eq!(sumcheck.evals.num_evals(), 1 << num_vars_inter);
    }

    // Ensure we’ve folded all variables.
    assert_eq!(num_vars_inter, final_rounds);

    // FINAL ROUND
    prover_randomness = extend_point(
        &prover_randomness,
        &sumcheck.compute_sumcheck_polynomials(prover, final_rounds, 0, None),
    );

    assert_eq!(sumcheck.evals.num_variables(), 0);
    assert_eq!(sumcheck.evals.num_evals(), 1);

    // Final folded value must match f(r)
    let final_folded_value = sumcheck.evals.as_constant().unwrap();
    assert_eq!(
        poly.evaluate_hypercube(&prover_randomness),
        final_folded_value
    );
    // Commit final result to Fiat-Shamir transcript
    prover.add_extension_scalar(final_folded_value);

    // Save proof data to pass to verifier
    let proof = prover.proof_data().to_vec();

    // VERIFIER
    let verifier = &mut verifier(proof);

    // Running total for the verifier’s sum of constraint combinations
    let mut sum = EF::ZERO;

    // Point `r` is constructed over rounds using verifier-chosen challenges
    let mut verifier_randomness = MultilinearPoint::new(vec![]);

    // All constraints used by the verifier across rounds
    let mut constraints = vec![];

    // Recompute the same variable count as prover had
    let mut num_vars_inter = num_vars;

    // VERIFY EACH ROUND
    for (round_idx, (&num_eq_points, &num_sel_points)) in
        num_eqs.iter().zip(num_sels.iter()).enumerate()
    {
        // Reconstruct round constraint from transcript
        let constraint = read_constraint(verifier, num_vars_inter, num_eq_points, num_sel_points);
        // Accumulate the weighted sum of constraint values
        constraint.combine_evals(&mut sum);
        // Save constraints for later equality check
        constraints.push(constraint);

        // Extend r with verifier's folding challenges
        let folding = folding_factor.at_round(round_idx);
        verifier_randomness = extend_point(
            &verifier_randomness,
            &verify_sumcheck_rounds(verifier, &mut sum, folding, 0, false).unwrap(),
        );

        num_vars_inter -= folding;
    }

    // Final round check
    verifier_randomness = extend_point(
        &verifier_randomness,
        &verify_sumcheck_rounds(verifier, &mut sum, final_rounds, 0, false).unwrap(),
    );

    // Check that the randomness vectors are the same
    assert_eq!(prover_randomness, verifier_randomness);

    // Final folded constant from transcript
    assert_eq!(
        final_folded_value,
        verifier.next_extension_scalar().unwrap()
    );

    // CHECK EQ(z, r) WEIGHT POLY
    //
    // No skip optimization, so the first round is treated as a standard sumcheck round.
    let evaluator = ConstraintPolyEvaluator::new(num_vars, folding_factor, None);
    let weights = evaluator.eval_constraints_poly(&constraints, &verifier_randomness);

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
    let prover = &mut prover();

    // Sample and commit initial evaluation constraints
    let constraint = make_constraint(prover, num_vars, num_eq_points[0], num_sel_points[0], &poly);
    constraint.validate_for_skip_case();

    // ROUND 0
    // Initialize sumcheck with univariate skip (skips K_SKIP_SUMCHECK)
    let folding0 = folding_factor.at_round(0);
    let (mut sumcheck, mut prover_randomness) =
        SumcheckSingle::with_skip(&poly, prover, folding0, 0, K_SKIP_SUMCHECK, &constraint);

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
        let constraint = make_constraint_ext(
            prover,
            num_vars_inter,
            num_eq_points,
            num_sel_points,
            &sumcheck.evals,
        );

        // Fold the sumcheck polynomial again and extend randomness vector
        prover_randomness = extend_point(
            &prover_randomness,
            &sumcheck.compute_sumcheck_polynomials(prover, folding, 0, Some(constraint)),
        );

        num_vars_inter -= folding;

        // Sanity check: number of variables and evaluations should be correct
        assert_eq!(sumcheck.evals.num_variables(), num_vars_inter);
        assert_eq!(sumcheck.evals.num_evals(), 1 << num_vars_inter);
    }

    // Ensure we’ve folded all variables.
    assert_eq!(num_vars_inter, final_rounds);

    // FINAL ROUND
    prover_randomness = extend_point(
        &prover_randomness,
        &sumcheck.compute_sumcheck_polynomials(prover, final_rounds, 0, None),
    );

    // After final round, polynomial must collapse to a constant
    assert_eq!(sumcheck.evals.num_variables(), 0);
    assert_eq!(sumcheck.evals.num_evals(), 1);

    // Final constant should be f(r), where r is the accumulated challenge point
    let final_folded_value = sumcheck.evals.as_constant().unwrap();
    // Final constant should be f̂(r0, r_{k+1..}) under skip semantics
    assert_eq!(
        eval_with_skip::<F, EF>(&poly, K_SKIP_SUMCHECK, &prover_randomness),
        final_folded_value
    );
    // Commit final result to Fiat-Shamir transcript
    prover.add_extension_scalar(final_folded_value);

    // Extract proof data for verifier
    let proof = prover.proof_data().to_vec();

    // VERIFIER SIDE
    let verifier = &mut verifier(proof);

    // Running total for the verifier’s sum of constraint combinations
    let mut sum = EF::ZERO;

    // Point `r` is constructed over rounds using verifier-chosen challenges
    let mut verifier_randomness = MultilinearPoint::new(vec![]);

    // All constraints used by the verifier across rounds
    let mut constraints = vec![];

    // Recompute the same variable count as prover had
    let mut num_vars_inter = num_vars;

    // VERIFY EACH ROUND
    for (round_idx, (&num_eq_points, &num_sel_points)) in
        num_eq_points.iter().zip(num_sel_points.iter()).enumerate()
    {
        // Reconstruct round constraint from transcript
        let constraint = read_constraint(verifier, num_vars_inter, num_eq_points, num_sel_points);
        // Accumulate the weighted sum of constraint values
        constraint.combine_evals(&mut sum);
        // Save constraints for later equality check
        constraints.push(constraint);

        // Extend r with verifier's folding randomness
        //
        // The skip optimization is only applied to the first round.
        let is_skip_round = round_idx == 0;
        let folding = folding_factor.at_round(round_idx);
        verifier_randomness = extend_point(
            &verifier_randomness,
            &verify_sumcheck_rounds(verifier, &mut sum, folding, 0, is_skip_round).unwrap(),
        );

        num_vars_inter -= folding;
    }

    // FINAL FOLDING
    verifier_randomness = extend_point(
        &verifier_randomness,
        &verify_sumcheck_rounds(verifier, &mut sum, final_rounds, 0, false).unwrap(),
    );

    // Check that the randomness vectors are the same
    assert_eq!(prover_randomness, verifier_randomness);

    // Final constant from transcript must match prover's
    assert_eq!(
        final_folded_value,
        verifier.next_extension_scalar().unwrap()
    );

    // EVALUATE EQ(z, r) VIA CONSTRAINTS
    //
    // Evaluate eq(z, r) using the unified constraint evaluation function.
    let evaluator = ConstraintPolyEvaluator::new(num_vars, folding_factor, Some(K_SKIP_SUMCHECK));
    let weights = evaluator.eval_constraints_poly(&constraints, &verifier_randomness);

    // FINAL SUMCHECK CHECK
    //
    // Main equation: sum == f(r) · eq(z, r)
    assert_eq!(sum, final_folded_value * weights);
}

/// Evaluate f with the same "univariate skip" semantics the prover uses:
/// - Treat the first k variables as a single univariate slot over the 2^k subgroup,
/// - interpolate at r_all[0],
/// - then evaluate the remaining (n - k) variables at r_all[1..].
fn eval_with_skip<F, EF>(
    poly: &EvaluationsList<F>,
    k_skip: usize,
    r_all: &MultilinearPoint<EF>,
) -> EF
where
    F: TwoAdicField,
    EF: TwoAdicField + ExtensionField<F>,
{
    let n = poly.num_variables();
    assert!(k_skip > 0 && k_skip <= n);

    // After with_skip() reverses, layout is:
    // [ r_for_remaining_vars...,  r_skip ]
    let need = 1 + (n - k_skip);
    assert!(
        r_all.num_variables() >= need,
        "need {} challenges (1 + n - k), got {}",
        need,
        r_all.num_variables()
    );

    // - r0 is the **last** element (skip challenge),
    // - "rest" are the first n-k challenges for the remaining variables.
    let r0 = *r_all.last_variable().unwrap();
    let rest = r_all.get_subpoint_over_range(0..(n - k_skip));

    // Reshape f into 2^k × 2^{n-k}, interpolate along the skipped dimension at r0,
    // then evaluate the resulting EF-table on the remaining variables.
    let width = 1 << (n - k_skip);
    let f_mat = poly.clone().into_mat(width);
    let folded_row = interpolate_subgroup::<F, EF, _>(&f_mat, r0);

    EvaluationsList::new(folded_row).evaluate_hypercube(&rest)
}

// fn run_sumcheck_test_svo(folding_factors: &[usize], num_points: &[usize]) {
//     // The number of folding stages must match the number of point constraints plus final round.
//     assert_eq!(folding_factors.len(), num_points.len() + 1);
//     // Total number of variables is the sum of all folding amounts.
//     let num_vars = folding_factors.iter().sum::<usize>();

//     // Initialize a random multilinear polynomial with 2^num_vars evaluations.
//     let mut rng = SmallRng::seed_from_u64(1);
//     let poly = EvaluationsList::new((0..1 << num_vars).map(|_| rng.random()).collect());

//     // PROVER
//     let prover = &mut prover();

//     // Create the initial constraint statement: {poly(z_i) = y_i}
//     let statement = make_initial_statement(prover, num_vars, num_points[0], &poly);

//     // Sample random linear combination challenge α₀.
//     let alpha: EF = prover.sample();

//     // ROUND 0
//     let folding = folding_factors[0];
//     let (mut sumcheck, mut prover_randomness) =
//         SumcheckSingle::from_base_evals_svo(&poly, &statement, alpha, prover, folding, 0);

//     // Track how many variables remain to fold
//     let mut num_vars_inter = num_vars - folding;

//     // INTERMEDIATE ROUNDS
//     for (&folding, &num_points) in folding_factors
//         .iter()
//         .skip(1)
//         .zip(num_points.iter().skip(1))
//     {
//         // Add additional equality constraints for intermediate rounds
//         let _ = make_inter_statement(prover, num_points, &mut sumcheck);

//         // Compute and apply the next folding round
//         prover_randomness = extend_point(
//             &prover_randomness,
//             &sumcheck.compute_sumcheck_polynomials(prover, folding, 0),
//         );

//         num_vars_inter -= folding;

//         // Check that the number of variables and evaluations match the expected values
//         assert_eq!(sumcheck.evals.num_variables(), num_vars_inter);
//         assert_eq!(sumcheck.evals.num_evals(), 1 << num_vars_inter);
//     }

//     // FINAL ROUND
//     let final_rounds = *folding_factors.last().unwrap();
//     prover_randomness = extend_point(
//         &prover_randomness,
//         &sumcheck.compute_sumcheck_polynomials(prover, final_rounds, 0),
//     );

//     // Ensure we’ve folded all variables.
//     //assert_eq!(num_vars_inter, final_rounds);
//     assert_eq!(sumcheck.evals.num_variables(), 0);
//     assert_eq!(sumcheck.evals.num_evals(), 1);

//     // Final folded value must match f(r)
//     let final_folded_value = sumcheck.evals.as_constant().unwrap();

//     assert_eq!(poly.evaluate(&prover_randomness), final_folded_value);
//     prover.add_extension_scalar(final_folded_value);

//     // Save proof data to pass to verifier
//     let proof = prover.proof_data().to_vec();

//     // VERIFIER
//     let verifier = &mut verifier(proof);
//     let mut sum = EF::ZERO;
//     let mut verifier_randomness = MultilinearPoint::new(vec![]);
//     let mut alphas = vec![];
//     let mut constraints = vec![];
//     let mut num_vars_inter = num_vars;

//     // VERIFY EACH ROUND
//     for (&folding, &num_points) in folding_factors.iter().zip(num_points.iter()) {
//         // Reconstruct statement from transcript
//         let st = read_statement(verifier, num_vars_inter, num_points);

//         // Draw αᵢ for linear combination
//         let alpha = verifier.sample();
//         alphas.push(alpha);

//         // Combine all constraint sums with powers of αᵢ
//         combine_constraints(&mut sum, &st.constraints, alpha);

//         // Save constraints for later equality test
//         constraints.push(st.constraints.clone());

//         // Extend r with verifier's folding challenges
//         verifier_randomness = extend_point(
//             &verifier_randomness,
//             &verify_sumcheck_rounds_svo(verifier, &mut sum, folding, 0).unwrap(),
//         );

//         num_vars_inter -= folding;
//     }

//     // Check reconstructed constraints match original ones
//     for (expected, actual) in constraints
//         .clone()
//         .iter()
//         .flatten()
//         .zip(statement.constraints.clone().iter())
//     {
//         assert_eq!(expected, actual);
//     }

//     // Final round check
//     let final_rounds = *folding_factors.last().unwrap();
//     verifier_randomness = extend_point(
//         &verifier_randomness,
//         &verify_sumcheck_rounds_svo(verifier, &mut sum, final_rounds, 0).unwrap(),
//     );

//     // Check that the randomness vectors are the same
//     assert_eq!(prover_randomness, verifier_randomness);

//     // Final folded constant from transcript
//     let final_folded_value_transcript = verifier.next_extension_scalar().unwrap();
//     assert_eq!(final_folded_value, final_folded_value_transcript);

//     // CHECK EQ(z, r) WEIGHT POLY
//     //
//     // No skip optimization, so the first round is treated as a standard sumcheck round.
//     let eq_eval = eval_constraints_poly::<F, EF>(
//         false,
//         num_vars,
//         0,
//         folding_factors,
//         &constraints,
//         &alphas,
//         &verifier_randomness,
//     );

//     // CHECK SUM == f(r) * eq(z, r)
//     assert_eq!(sum, final_folded_value_transcript * eq_eval);
// }

fn run_sumcheck_test_svo(
    num_vars: usize,
    folding_factor: FoldingFactor,
    // Maybe we can remove these two inputs:
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
    let prover = &mut prover();

    // Create the initial constraint statement
    let constraint = make_constraint(prover, num_vars, num_eqs[0], num_sels[0], &poly);

    // ROUND 0
    let folding0 = folding_factor.at_round(0);
    let (mut sumcheck, mut prover_randomness) =
        SumcheckSingle::from_base_evals_svo(&poly, prover, folding0, 0, &constraint);

    // Track how many variables remain to fold
    let mut num_vars_inter = num_vars - folding0;

    // INTERMEDIATE ROUNDS
    for (round, (&num_eq_points, &num_sel_points)) in
        num_eqs.iter().zip(num_sels.iter()).enumerate().skip(1)
    {
        let folding = folding_factor.at_round(round);
        // Sample new evaluation constraints and combine them into the sumcheck state
        let constraint = make_constraint_ext(
            prover,
            num_vars_inter,
            num_eq_points,
            num_sel_points,
            &sumcheck.evals,
        );

        // Compute and apply the next folding round
        prover_randomness = extend_point(
            &prover_randomness,
            &sumcheck.compute_sumcheck_polynomials(prover, folding, 0, Some(constraint)),
        );

        num_vars_inter -= folding;

        // Check that the number of variables and evaluations match the expected values
        assert_eq!(sumcheck.evals.num_variables(), num_vars_inter);
        assert_eq!(sumcheck.evals.num_evals(), 1 << num_vars_inter);
    }

    // Ensure we’ve folded all variables.
    assert_eq!(num_vars_inter, final_rounds);

    // FINAL ROUND
    prover_randomness = extend_point(
        &prover_randomness,
        &sumcheck.compute_sumcheck_polynomials(prover, final_rounds, 0, None),
    );

    assert_eq!(sumcheck.evals.num_variables(), 0);
    assert_eq!(sumcheck.evals.num_evals(), 1);

    // Final folded value must match f(r)
    let final_folded_value = sumcheck.evals.as_constant().unwrap();
    assert_eq!(
        poly.evaluate_hypercube(&prover_randomness),
        final_folded_value
    );
    // Commit final result to Fiat-Shamir transcript
    prover.add_extension_scalar(final_folded_value);

    // Save proof data to pass to verifier
    let proof = prover.proof_data().to_vec();

    // VERIFIER
    let verifier = &mut verifier(proof);

    // Running total for the verifier’s sum of constraint combinations
    let mut sum = EF::ZERO;

    // Point `r` is constructed over rounds using verifier-chosen challenges
    let mut verifier_randomness = MultilinearPoint::new(vec![]);

    // All constraints used by the verifier across rounds
    let mut constraints = vec![];

    // Recompute the same variable count as prover had
    let mut num_vars_inter = num_vars;

    // VERIFY EACH ROUND
    for (round_idx, (&num_eq_points, &num_sel_points)) in
        num_eqs.iter().zip(num_sels.iter()).enumerate()
    {
        // Reconstruct round constraint from transcript
        let constraint = read_constraint(verifier, num_vars_inter, num_eq_points, num_sel_points);
        // Accumulate the weighted sum of constraint values
        constraint.combine_evals(&mut sum);
        // Save constraints for later equality check
        constraints.push(constraint);

        // Extend r with verifier's folding challenges
        let folding = folding_factor.at_round(round_idx);
        verifier_randomness = extend_point(
            &verifier_randomness,
            &verify_sumcheck_rounds_svo(verifier, &mut sum, folding, 0).unwrap(),
        );

        num_vars_inter -= folding;
    }

    // Final round check
    verifier_randomness = extend_point(
        &verifier_randomness,
        &verify_sumcheck_rounds_svo(verifier, &mut sum, final_rounds, 0).unwrap(),
    );

    // Check that the randomness vectors are the same
    assert_eq!(prover_randomness, verifier_randomness);

    // Final folded constant from transcript
    assert_eq!(
        final_folded_value,
        verifier.next_extension_scalar().unwrap()
    );

    // CHECK EQ(z, r) WEIGHT POLY
    //
    // No skip optimization, so the first round is treated as a standard sumcheck round.
    let evaluator = ConstraintPolyEvaluator::new(num_vars, folding_factor, None);
    let weights = evaluator.eval_constraints_poly(&constraints, &verifier_randomness);

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
    // let mut rng = SmallRng::seed_from_u64(0);
    for num_vars in 6..=10 {
        let folding_factor = num_vars;
        let folding_factor = FoldingFactor::Constant(folding_factor);
        let num_eq_points = vec![1];
        let num_sel_points = vec![0]; 
        run_sumcheck_test_svo(num_vars, folding_factor, &num_eq_points, &num_sel_points);
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
