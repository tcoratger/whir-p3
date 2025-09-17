use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::{DuplexChallenger, FieldChallenger, GrindingChallenger};
use p3_field::{
    ExtensionField, Field, PrimeCharacteristicRing, TwoAdicField, extension::BinomialExtensionField,
};
use p3_interpolation::interpolate_subgroup;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use proptest::prelude::*;
use rand::{Rng, SeedableRng, rngs::SmallRng};

use super::sumcheck_single::SumcheckSingle;
use crate::{
    constant::K_SKIP_SUMCHECK,
    fiat_shamir::{
        domain_separator::DomainSeparator, prover::ProverState, verifier::VerifierState,
    },
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    whir::{constraints::statement::Statement, verifier::sumcheck::verify_sumcheck_rounds},
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

/// Creates an initial `Statement` with evaluation constraints sampled from the prover's challenger.
///
/// This function samples `num_points` univariate field elements from the prover,
/// expands each into a full `MultilinearPoint` of arity `num_vars`, evaluates the provided
/// multilinear polynomial at each point, and constructs a set of equality constraints of the form:
/// \begin{equation}
///     \text{poly}(\vec{z}_i) = y_i
/// \end{equation}
///
/// where each point $\vec{z}_i$ is sampled via Fiat-Shamir, and each evaluation $y_i$ is added to the transcript.
///
/// # Arguments
/// - `prover`: The mutable `ProverState` which manages Fiat-Shamir sampling and proof data.
/// - `num_vars`: The total number of variables in the multilinear polynomial.
/// - `num_points`: The number of evaluation points to sample from the prover.
/// - `poly`: The multilinear polynomial (in evaluation form) to be evaluated.
///
/// # Returns
/// A `Statement<EF>` containing `num_points` evaluation constraints for the verifier to later check.
fn make_initial_statement<Challenger>(
    prover: &mut ProverState<F, EF, Challenger>,
    num_vars: usize,
    num_points: usize,
    poly: &EvaluationsList<F>,
) -> Statement<EF>
where
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    // Initialize the statement to hold the evaluation constraints.
    let mut statement = Statement::initialize(num_vars);

    // In a single pass, sample each point, evaluate the polynomial, commit it,
    // and insert the resulting constraint into the statement.
    for _ in 0..num_points {
        // Sample a univariate challenge value and lift it to a multilinear point.
        let point = MultilinearPoint::expand_from_univariate(prover.sample(), num_vars);

        // Evaluate the polynomial at this point.
        let eval = poly.evaluate(&point);

        // Record the evaluation in the transcript for Fiat-Shamir soundness.
        prover.add_extension_scalar(eval);

        // Add the constraint: poly(point) = eval.
        statement.add_evaluated_constraint(point, eval);
    }

    // Return the complete statement.
    statement
}

/// Constructs an intermediate `Statement` from the current `sumcheck` polynomial,
/// and registers a new equality constraint with random linear combination weights.
///
/// This function is used between sumcheck rounds to:
/// 1. Sample `num_points` new evaluation points via Fiat-Shamir.
/// 2. Evaluate the current sumcheck polynomial at those points.
/// 3. Commit those evaluations to the transcript for soundness.
/// 4. Add constraints to the current `Statement`.
/// 5. Draw a random scalar `alpha`, and use its powers to define a new linear combination
///    of constraints that must evaluate to the same sum. This supports recursive soundness.
///
/// # Arguments
/// - `prover`: The mutable `ProverState` used for Fiat-Shamir sampling and logging commitments.
/// - `num_points`: Number of evaluation points to sample and constrain.
/// - `sumcheck`: The current `SumcheckSingle` object representing the prover's polynomial state.
///
/// # Returns
/// A tuple `(statement, alpha)`:
/// - `statement`: The new `Statement<EF>` containing point-wise evaluation constraints.
/// - `alpha`: The random scalar used to linearly combine those constraints.
fn make_inter_statement<Challenger>(
    prover: &mut ProverState<F, EF, Challenger>,
    num_points: usize,
    sumcheck: &mut SumcheckSingle<F, EF>,
) -> (Statement<EF>, EF)
where
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    // Determine how many variables are left in the current sumcheck polynomial.
    let num_vars = sumcheck.num_variables();

    // Create a new empty statement of that arity (for evaluation constraints).
    let mut statement = Statement::initialize(num_vars);

    // - Sample `num_points` univariate challenge points.
    // - Evaluate the sumcheck polynomial on them.
    // - Collect (point, eval) pairs for use in the statement and constraint aggregation.
    let (points, evals): (Vec<_>, Vec<_>) = (0..num_points)
        .map(|_| {
            // Sample a univariate field element from the prover's challenger.
            let point = prover.sample();

            // Expand it into a `num_vars`-dimensional multilinear point.
            let point = MultilinearPoint::expand_from_univariate(point, num_vars);

            // Evaluate the current sumcheck polynomial at the sampled point.
            let eval = sumcheck.evals.evaluate(&point);

            // Add the evaluation result to the transcript for Fiat-Shamir soundness.
            prover.add_extension_scalar(eval);

            // Add the evaluation constraint: poly(point) == eval.
            statement.add_evaluated_constraint(point.clone(), eval);

            // Return the sampled point and its evaluation.
            (point, eval)
        })
        .unzip();

    // Sample a random extension field element `alpha` to serve as a combination coefficient.
    let alpha: EF = prover.sample();

    // Compute powers of alpha up to `num_points` and register the new equality constraint.
    //
    // This enforces that the weighted sum of these evaluations equals the claimed value.
    sumcheck.add_new_equality(&points, &evals, &alpha.powers().collect_n(num_points));

    // Return the constructed statement and the alpha used for linear combination.
    (statement, alpha)
}

/// Reconstructs a `Statement` from the verifier's transcript using Fiat-Shamir sampling.
///
/// This function performs the verifier-side equivalent of `make_initial_statement` or `make_inter_statement`.
/// It:
/// 1. Samples `num_points` univariate challenge points.
/// 2. Expands them to `num_vars`-dimensional multilinear points.
/// 3. Reads the corresponding committed evaluations from the proof data.
/// 4. Constructs a `Statement` encoding these constraints: `poly(point) == eval`.
///
/// # Arguments
/// - `verifier`: The mutable `VerifierState` which provides Fiat-Shamir sampling and reads proof scalars.
/// - `num_vars`: The number of variables in the multilinear domain.
/// - `num_points`: The number of evaluation constraints to reconstruct.
///
/// # Returns
/// A `Statement<EF>` populated with `num_points` evaluation constraints for verification.
fn read_statement<Challenger>(
    verifier: &mut VerifierState<F, EF, Challenger>,
    num_vars: usize,
    num_points: usize,
) -> Statement<EF>
where
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    // Create a new statement that will hold all reconstructed constraints.
    let mut statement = Statement::initialize(num_vars);

    // For each point, sample a challenge and read its corresponding evaluation from the transcript.
    for _ in 0..num_points {
        // Sample a univariate challenge and expand to a multilinear point.
        let point = MultilinearPoint::expand_from_univariate(verifier.sample(), num_vars);

        // Read the committed evaluation corresponding to this point from the proof data.
        let eval = verifier.next_extension_scalar().unwrap();

        // Add the constraint: poly(point) == eval.
        statement.add_evaluated_constraint(point, eval);
    }

    // Return the fully reconstructed statement.
    statement
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

/// Evaluates the final combined weight polynomial `W(r)` by recursively applying
/// the correct evaluation method for each round of constraints.
///
/// This function is the verifier's master function for the final check of the sumcheck protocol. It
/// correctly handles the recursive nature of the proof, where constraints are defined over
/// progressively smaller domains.
///
/// It intelligently handles the **univariate skip** optimization: if the flag is set,
/// it applies a special, skip-aware evaluation method for the first round's constraints;
/// otherwise, it treats all rounds as standard sumcheck rounds.
///
/// # Arguments
/// * `univariate_skip`: A boolean flag indicating if the first round used the skip optimization.
/// * `num_vars`: The total number of variables in the original polynomial.
/// * `k_skip`: The number of variables that were skipped (only used if `univariate_skip` is true).
/// * `folding_factors`: The number of variables folded in each round.
/// * `constraints`: A list of constraint sets, one for each round of the protocol.
/// * `alphas`: The random scalars used to linearly combine the constraints in each round.
/// * `final_challenges`: The final, full `n`-dimensional challenge point `r`.
///
/// # Returns
/// The evaluation `W(r)` as a single field element.
fn eval_constraints_poly<F, EF>(
    univariate_skip: bool,
    num_vars: usize,
    k_skip: usize,
    folding_factors: &[usize],
    statements: &[Statement<EF>],
    alphas: &[EF],
    final_challenges: &MultilinearPoint<EF>,
) -> EF
where
    F: TwoAdicField,
    EF: TwoAdicField + ExtensionField<F>,
{
    let mut eq_eval = EF::ZERO;
    let mut num_vars_at_round = num_vars;
    let mut challenges_for_round = final_challenges.clone();

    // Iterate through each round where constraints were introduced.
    for (round_idx, round_statement) in statements.iter().enumerate() {
        let alpha = alphas[round_idx];
        let alpha_pows = alpha.powers().collect_n(round_statement.len());

        // Determine if this specific round should be evaluated using the skip method.
        let is_skip_round = round_idx == 0 && univariate_skip;

        // Calculate the total contribution from this round's constraints.
        let round_contribution: EF = round_statement
            .iter()
            .zip(alpha_pows)
            .map(|((point, _), alpha_pow)| {
                let single_eval = if is_skip_round {
                    // ROUND 0 with SKIP: Use the special skip-aware evaluation.
                    // The constraints for this round are over the full `num_vars` domain.
                    assert_eq!(point.num_variables(), num_vars);
                    point.eq_poly_with_skip(final_challenges, k_skip)
                } else {
                    // STANDARD ROUND: Use the standard multilinear evaluation.
                    // The constraints and challenge point are over the smaller `num_vars_at_round` domain.
                    assert_eq!(point.num_variables(), num_vars_at_round);
                    point.eq_poly(&challenges_for_round)
                };
                alpha_pow * single_eval
            })
            .sum();

        eq_eval += round_contribution;

        // After processing the round, shrink the domain for the next iteration's challenges.
        if round_idx < statements.len() - 1 {
            num_vars_at_round -= folding_factors[round_idx];
            challenges_for_round = final_challenges.get_subpoint_over_range(0..num_vars_at_round);
        }
    }

    eq_eval
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
/// - `folding_factors`: List of how many variables to fold per round.
/// - `num_points`: Number of equality constraints to apply at each stage.
///   Must be one shorter than `folding_factors` (initial + intermediate).
fn run_sumcheck_test(folding_factors: &[usize], num_points: &[usize]) {
    // The number of folding stages must match the number of point constraints plus final round.
    assert_eq!(folding_factors.len(), num_points.len() + 1);
    // Total number of variables is the sum of all folding amounts.
    let num_vars = folding_factors.iter().sum::<usize>();

    // Initialize a random multilinear polynomial with 2^num_vars evaluations.
    let mut rng = SmallRng::seed_from_u64(1);
    let poly = EvaluationsList::new((0..1 << num_vars).map(|_| rng.random()).collect());

    // PROVER
    let prover = &mut prover();

    // Create the initial constraint statement: {poly(z_i) = y_i}
    let statement = make_initial_statement(prover, num_vars, num_points[0], &poly);

    // Sample random linear combination challenge α₀.
    let alpha: EF = prover.sample();

    // ROUND 0
    let folding = folding_factors[0];
    let (mut sumcheck, mut prover_randomness) =
        SumcheckSingle::from_base_evals(&poly, &statement, alpha, prover, folding, 0);

    // Track how many variables remain to fold
    let mut num_vars_inter = num_vars - folding;

    // INTERMEDIATE ROUNDS
    for (&folding, &num_points) in folding_factors
        .iter()
        .skip(1)
        .zip(num_points.iter().skip(1))
    {
        // Add additional equality constraints for intermediate rounds
        let _ = make_inter_statement(prover, num_points, &mut sumcheck);

        // Compute and apply the next folding round
        prover_randomness = extend_point(
            &prover_randomness,
            &sumcheck.compute_sumcheck_polynomials(prover, folding, 0),
        );

        num_vars_inter -= folding;

        // Check that the number of variables and evaluations match the expected values
        assert_eq!(sumcheck.evals.num_variables(), num_vars_inter);
        assert_eq!(sumcheck.evals.num_evals(), 1 << num_vars_inter);
    }

    // FINAL ROUND
    let final_rounds = *folding_factors.last().unwrap();
    prover_randomness = extend_point(
        &prover_randomness,
        &sumcheck.compute_sumcheck_polynomials(prover, final_rounds, 0),
    );

    // Ensure we’ve folded all variables.
    assert_eq!(num_vars_inter, final_rounds);
    assert_eq!(sumcheck.evals.num_variables(), 0);
    assert_eq!(sumcheck.evals.num_evals(), 1);

    // Final folded value must match f(r)
    let final_folded_value = sumcheck.evals.as_constant().unwrap();
    assert_eq!(poly.evaluate(&prover_randomness), final_folded_value);
    prover.add_extension_scalar(final_folded_value);

    // Save proof data to pass to verifier
    let proof = prover.proof_data().to_vec();

    // VERIFIER
    let verifier = &mut verifier(proof);
    let mut sum = EF::ZERO;
    let mut verifier_randomness = MultilinearPoint::new(vec![]);
    let mut alphas = vec![];
    let mut constraints = vec![];
    let mut num_vars_inter = num_vars;

    // VERIFY EACH ROUND
    for (&folding, &num_points) in folding_factors.iter().zip(num_points.iter()) {
        // Reconstruct statement from transcript
        let st = read_statement(verifier, num_vars_inter, num_points);

        // Draw αᵢ for linear combination
        let alpha = verifier.sample();
        alphas.push(alpha);

        // Combine all constraint sums with powers of αᵢ
        st.combine_evals(&mut sum, alpha);

        // Save constraints for later equality test
        constraints.push(st.clone());

        // Extend r with verifier's folding challenges
        verifier_randomness = extend_point(
            &verifier_randomness,
            &verify_sumcheck_rounds(verifier, &mut sum, folding, 0, false).unwrap(),
        );

        num_vars_inter -= folding;
    }

    // Check reconstructed constraints match original ones
    for (expected, actual) in constraints
        .clone()
        .iter()
        .flat_map(Statement::iter)
        .zip(statement.iter())
    {
        assert_eq!(expected, actual);
    }

    // Final round check
    let final_rounds = *folding_factors.last().unwrap();
    verifier_randomness = extend_point(
        &verifier_randomness,
        &verify_sumcheck_rounds(verifier, &mut sum, final_rounds, 0, false).unwrap(),
    );

    // Check that the randomness vectors are the same
    assert_eq!(prover_randomness, verifier_randomness);

    // Final folded constant from transcript
    let final_folded_value_transcript = verifier.next_extension_scalar().unwrap();
    assert_eq!(final_folded_value, final_folded_value_transcript);

    // CHECK EQ(z, r) WEIGHT POLY
    //
    // No skip optimization, so the first round is treated as a standard sumcheck round.
    let eq_eval = eval_constraints_poly::<F, EF>(
        false,
        num_vars,
        0,
        folding_factors,
        &constraints,
        &alphas,
        &verifier_randomness,
    );
    // CHECK SUM == f(r) * eq(z, r)
    assert_eq!(sum, final_folded_value_transcript * eq_eval);
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
/// - `folding_factors`: A list of folding amounts (how many variables are folded each round).
/// - `num_points`: Number of equality constraints applied in each round, except the final one.
fn run_sumcheck_test_skips(folding_factors: &[usize], num_points: &[usize]) {
    // Sanity check: folding_factors should have one more element than num_points
    assert_eq!(folding_factors.len(), num_points.len() + 1);

    // Total number of variables in the original multilinear polynomial
    let num_vars = folding_factors.iter().sum::<usize>();

    // SETUP POLYNOMIAL
    //
    // Generate a random multilinear polynomial of arity `num_vars`
    let mut rng = SmallRng::seed_from_u64(1);
    let poly = EvaluationsList::new((0..1 << num_vars).map(|_| rng.random()).collect());

    // PROVER SIDE
    let prover = &mut prover();

    // Sample and commit initial evaluation constraints for poly(z_i) = y_i
    let statement = make_initial_statement(prover, num_vars, num_points[0], &poly);

    // Sample challenge α₀ used to linearly combine the initial constraints
    let alpha: EF = prover.sample();

    let folding = folding_factors[0];

    // Initialize sumcheck with univariate skip (skips K_SKIP_SUMCHECK variables)
    let (mut sumcheck, mut prover_randomness) = SumcheckSingle::with_skip(
        &poly,
        &statement,
        alpha,
        prover,
        folding,
        0,
        K_SKIP_SUMCHECK,
    );

    // Track how many variables remain after folding
    let mut num_vars_inter = num_vars - folding;

    // INTERMEDIATE ROUNDS
    for (&folding, &num_pts) in folding_factors
        .iter()
        .skip(1)
        .zip(num_points.iter().skip(1))
    {
        // Sample new evaluation constraints and combine them into the sumcheck state
        let _ = make_inter_statement(prover, num_pts, &mut sumcheck);

        // Fold the sumcheck polynomial again and extend randomness vector
        prover_randomness = extend_point(
            &prover_randomness,
            &sumcheck.compute_sumcheck_polynomials(prover, folding, 0),
        );
        num_vars_inter -= folding;

        // Sanity check: number of variables and evaluations should be correct
        assert_eq!(sumcheck.evals.num_variables(), num_vars_inter);
        assert_eq!(sumcheck.evals.num_evals(), 1 << num_vars_inter);
    }

    // FINAL ROUND
    let final_rounds = *folding_factors.last().unwrap();
    prover_randomness = extend_point(
        &prover_randomness,
        &sumcheck.compute_sumcheck_polynomials(prover, final_rounds, 0),
    );

    // After final round, polynomial must collapse to a constant
    assert_eq!(num_vars_inter, final_rounds);
    assert_eq!(sumcheck.evals.num_variables(), 0);
    assert_eq!(sumcheck.evals.num_evals(), 1);

    // Final constant should be f(r), where r is the accumulated challenge point
    let constant = sumcheck.evals.as_constant().unwrap();

    // Final constant should be f̂(r0, r_{k+1..}) under skip semantics
    let expected = eval_with_skip::<F, EF>(&poly, K_SKIP_SUMCHECK, &prover_randomness);
    assert_eq!(constant, expected);

    // Commit final result to Fiat-Shamir transcript
    prover.add_extension_scalar(constant);

    // Extract proof data for verifier
    let proof = prover.proof_data().to_vec();

    // VERIFIER SIDE
    let verifier = &mut verifier(proof);

    // Running total for the verifier’s sum of constraint combinations
    let mut sum = EF::ZERO;

    // Point `r` is constructed over rounds using verifier-chosen challenges
    let mut verifier_randomness = MultilinearPoint::new(vec![]);

    // Track challenge alphas used for combining constraints in each round
    let mut alphas = vec![];

    // All constraints used by the verifier across rounds
    let mut constraints = vec![];

    // Recompute the same variable count as prover had
    let mut num_vars_inter = num_vars;

    // VERIFY EACH ROUND
    for (round_idx, (&folding, &num_pts)) in
        folding_factors.iter().zip(num_points.iter()).enumerate()
    {
        // Reconstruct the equality statement from the proof stream
        let statement = read_statement(verifier, num_vars_inter, num_pts);

        // Sample αᵢ for combining constraints
        let alpha = verifier.sample();
        alphas.push(alpha);

        // Accumulate the weighted sum of constraint values
        statement.combine_evals(&mut sum, alpha);

        // Save constraints for later equality check
        constraints.push(statement.clone());

        // Extend r with verifier's folding randomness
        //
        // The skip optimization is only applied to the first round.
        let is_skip_round = round_idx == 0;
        verifier_randomness = extend_point(
            &verifier_randomness,
            &verify_sumcheck_rounds(verifier, &mut sum, folding, 0, is_skip_round).unwrap(),
        );

        num_vars_inter -= folding;
    }

    // FINAL FOLDING
    let final_rounds = *folding_factors.last().unwrap();
    verifier_randomness = extend_point(
        &verifier_randomness,
        &verify_sumcheck_rounds(verifier, &mut sum, final_rounds, 0, false).unwrap(),
    );

    // Check that the randomness vectors are the same
    assert_eq!(prover_randomness, verifier_randomness);

    // Final constant from transcript must match prover's
    let constant_verifier = verifier.next_extension_scalar().unwrap();
    assert_eq!(constant_verifier, constant);

    // EVALUATE EQ(z, r) VIA CONSTRAINTS
    //
    // Evaluate eq(z, r) using the unified constraint evaluation function.
    let eq_eval = eval_constraints_poly::<F, EF>(
        true, // univariate_skip is true for this test
        num_vars,
        K_SKIP_SUMCHECK,
        folding_factors,
        &constraints,
        &alphas,
        &verifier_randomness,
    );

    // FINAL SUMCHECK CHECK
    //
    // Main equation: sum == f(r) · eq(z, r)
    assert_eq!(sum, constant * eq_eval);
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

    EvaluationsList::new(folded_row).evaluate(&rest)
}

#[test]
fn test_sumcheck_prover() {
    // Test for the simplest case: classical sumcheck.
    run_sumcheck_test(&[1, 0], &[1]);
    run_sumcheck_test(&[1, 1], &[1]);
    run_sumcheck_test(&[1, 1], &[9]);
    run_sumcheck_test(&[4, 1], &[9]);
    run_sumcheck_test(&[1, 4], &[9]);
    run_sumcheck_test(&[1, 1, 1], &[1, 1]);
    run_sumcheck_test(&[4, 1, 4], &[9, 9]);
    run_sumcheck_test(&[4, 4, 1], &[9, 9]);
    run_sumcheck_test(&[1, 4, 4], &[9, 9]);

    // Test for the sumcheck with univariate skip optimization.
    run_sumcheck_test_skips(&[6, 1], &[1]);
    run_sumcheck_test_skips(&[6, 1], &[5]);
    run_sumcheck_test_skips(&[6, 4], &[3]);
    run_sumcheck_test_skips(&[6, 0], &[4]);
    run_sumcheck_test_skips(&[8, 2], &[3]);
    run_sumcheck_test_skips(&[6, 2, 2], &[3, 3]);
}

proptest! {
    #[test]
    fn prop_sumcheck_prover_classic(
        (folds, points) in
            // Non-final folds:
            // - Choose 2..=3 rounds so total length becomes 3..=4 after appending the final fold.
            // - Each round folds 1..=5 variables.
            prop::collection::vec(1usize..=5, 2..=3)
            // Final fold ∈ 0..=4
            .prop_flat_map(|prefix| {
                (0usize..=4).prop_map(move |last| {
                    let mut f = prefix.clone();
                    f.push(last);
                    f
                })
            })
            // Keep overall arity modest: sum(folds) ≤ 15.
            .prop_filter("sum(folds) must be small", |f| f.iter().sum::<usize>() <= 15)
            // Build the per-round constraint counts (no choice for the final fold):
            // - points.len() = folds.len() - 1
            // - each in 1..=4.
            .prop_flat_map(|f| {
                let len_pts = f.len() - 1;
                prop::collection::vec(1usize..=4, len_pts)
                    .prop_map(move |pts| (f.clone(), pts))
            })
    ) {
        // Run the complete classic (no-skip) prover/verifier test with this schedule.
        run_sumcheck_test(&folds, &points);
    }
}

proptest! {
    #[test]
    fn prop_sumcheck_prover_with_skip(
        (folds, points) in
            // Build a three-round folding schedule suited for the skip variant:
            // - First round: must be at least the skip threshold so the optimization actually triggers.
            //   We allow a small bump above the threshold to vary the difficulty a bit.
            // - Middle round: ensure there's another real folding step to keep things multi-round.
            // - Final round: allow either a no-op tail or a small final fold.
            (
                0usize..=3,   // small bump above the skip threshold (or none)
                1usize..=4,   // middle fold is a small positive amount
                0usize..=4    // final fold may be zero or small
            )
            // Materialize the schedule: [first >= K_SKIP_SUMCHECK, middle, last]
            .prop_map(|(delta, mid, last)| vec![K_SKIP_SUMCHECK + delta, mid, last])
            // Keep the overall arity modest so evaluation tables stay quick to build and check..
            .prop_filter("sum(folds) should remain modest", |f| f.iter().sum::<usize>() <= K_SKIP_SUMCHECK + 6)
            // Produce exactly one constraint-count per non-final round.
            // For three folds, we generate two constraint counts — each at least one — to ensure
            // every active round contributes a real equality check without blowing up runtime.
            .prop_flat_map(|f| {
                prop::collection::vec(1usize..=4, f.len() - 1)
                    .prop_map(move |pts| (f.clone(), pts))
            })
    ) {
        // Run the end-to-end test using the skip-aware prover/verifier path.
        run_sumcheck_test_skips(&folds, &points);
    }
}
