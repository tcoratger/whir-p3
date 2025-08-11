use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::{DuplexChallenger, FieldChallenger, GrindingChallenger};
use p3_field::{
    ExtensionField, Field, PrimeCharacteristicRing, TwoAdicField, extension::BinomialExtensionField,
};
use p3_interpolation::interpolate_subgroup;
use p3_matrix::dense::RowMajorMatrix;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::{Rng, SeedableRng, distr::StandardUniform, rngs::SmallRng};

use super::sumcheck_single::SumcheckSingle;
use crate::{
    fiat_shamir::{
        domain_separator::DomainSeparator, prover::ProverState, verifier::VerifierState,
    },
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    sumcheck::K_SKIP_SUMCHECK,
    whir::{
        statement::{Statement, constraint::Constraint, weights::Weights},
        verifier::sumcheck::verify_sumcheck_rounds,
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

/// Generates a vector of `n` random field elements using the provided RNG.
fn rand_vec<F>(mut rng: impl Rng, n: usize) -> Vec<F>
where
    StandardUniform: rand::distr::Distribution<F>,
{
    (0..n).map(|_| rng.random()).collect()
}

/// Generates a random `MultilinearPoint` of arity `k` over the given field.
fn rand_point<F>(rng: impl Rng, k: usize) -> MultilinearPoint<F>
where
    StandardUniform: rand::distr::Distribution<F>,
{
    MultilinearPoint(rand_vec(rng, k))
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
    let mut statement = Statement::new(num_vars);

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
        statement.add_constraint(Weights::evaluation(point), eval);
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
    let mut statement = Statement::new(num_vars);

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
            statement.add_constraint(Weights::evaluation(point.clone()), eval);

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
    let mut statement = Statement::new(num_vars);

    // For each point, sample a challenge and read its corresponding evaluation from the transcript.
    for _ in 0..num_points {
        // Sample a univariate challenge and expand to a multilinear point.
        let point = MultilinearPoint::expand_from_univariate(verifier.sample(), num_vars);

        // Read the committed evaluation corresponding to this point from the proof data.
        let eval = verifier.next_extension_scalar().unwrap();

        // Add the constraint: poly(point) == eval.
        statement.add_constraint(Weights::evaluation(point), eval);
    }

    // Return the fully reconstructed statement.
    statement
}

impl<F> MultilinearPoint<F>
where
    F: Field,
{
    /// Extends a `MultilinearPoint` with another `MultilinearPoint`.
    fn extend(&mut self, rest: &Self) {
        self.0 = rest.iter().chain(self.iter()).copied().collect::<Vec<_>>();
    }
}

/// Combines a list of constraints into a single linear combination using powers of `alpha`,
/// and updates the running claimed sum in place.
///
/// This is used during sumcheck verification to aggregate multiple equality constraints:
/// each constraint contributes its claimed sum weighted by a distinct power of a random scalar `alpha`.
/// This aggregation ensures soundness via the Schwartz-Zippel lemma by reducing multiple claims
/// into a single probabilistic check.
///
/// Specifically, given constraints with claimed sums $c_i$, the verifier checks that:
/// \begin{equation}
///     \text{combined\_sum} = \sum_{i=0}^{n-1} \alpha^i \cdot c_i
/// \end{equation}
///
/// # Arguments
/// - `claimed_sum`: Mutable reference to the total accumulated claimed sum so far. Updated in place.
/// - `constraints`: A slice of `Constraint<EF>` each containing a known `sum` field.
/// - `alpha`: A random extension field element used to weight the constraints.
///
/// # Returns
/// A `Vec<EF>` containing the powers of `alpha` used to combine each constraint.
fn combine_constraints<EF: Field>(
    claimed_sum: &mut EF,
    constraints: &[Constraint<EF>],
    alpha: EF,
) -> Vec<EF> {
    // Compute powers of alpha: [1, alpha, alpha², ..., alpha^{n-1}]
    let alpha = alpha.powers().collect_n(constraints.len());

    // Compute the weighted sum of all constraints using the corresponding power of alpha
    let weighted_sum: EF = constraints
        .iter()
        .zip(&alpha)
        .map(|(c, &rand)| rand * c.sum)
        .sum();

    // Add the result to the claimed sum
    *claimed_sum += weighted_sum;

    // Return the powers of alpha for use in combining weight functions later
    alpha
}

fn eval_constraints_poly<EF: Field>(
    mut num_variables: usize,
    folding_factor: &[usize],
    constraints: &[Vec<Constraint<EF>>],
    alphas: &[EF],
    mut point: MultilinearPoint<EF>,
) -> EF {
    let mut value = EF::ZERO;
    assert_eq!(alphas.len(), constraints.len());

    for (round, (alphas, constraints)) in alphas.iter().zip(constraints.iter()).enumerate() {
        let alphas = alphas.powers().collect_n(constraints.len());
        assert_eq!(alphas.len(), constraints.len());
        if round > 0 {
            num_variables -= folding_factor[round - 1];
            point = MultilinearPoint(point[..num_variables].to_vec());
        }
        value += constraints
            .iter()
            .zip(alphas)
            .map(|(constraint, alpha)| alpha * constraint.weights.compute(&point))
            .sum::<EF>();
    }
    value
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
    let poly = EvaluationsList::new(rand_vec(&mut rng, 1 << num_vars));

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
        make_inter_statement(prover, num_points, &mut sumcheck);

        // Compute and apply the next folding round
        prover_randomness.extend(&sumcheck.compute_sumcheck_polynomials(
            prover,
            folding_factors[1],
            0,
        ));
        num_vars_inter -= folding;

        // Check that the number of variables and evaluations match the expected values
        assert_eq!(sumcheck.evals.num_variables(), num_vars_inter);
        assert_eq!(sumcheck.evals.num_evals(), 1 << num_vars_inter);
    }

    // FINAL ROUND
    let final_rounds = *folding_factors.last().unwrap();
    prover_randomness.extend(&sumcheck.compute_sumcheck_polynomials(prover, final_rounds, 0));

    // Ensure we’ve folded all variables.
    assert_eq!(num_vars_inter, final_rounds);
    assert_eq!(sumcheck.evals.num_variables(), 0);
    assert_eq!(sumcheck.evals.num_evals(), 1);

    // Final folded value must match f(r)
    let final_folded_value = sumcheck.evals[0];
    assert_eq!(poly.evaluate(&prover_randomness), final_folded_value);
    prover.add_extension_scalar(final_folded_value);

    // Save proof data to pass to verifier
    let proof = prover.proof_data().to_vec();

    // VERIFIER
    let verifier = &mut verifier(proof);
    let mut sum = EF::ZERO;
    let mut verifier_randomness = MultilinearPoint(vec![]);
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
        combine_constraints(&mut sum, &st.constraints, alpha);

        // Save constraints for later equality test
        constraints.push(st.constraints.clone());

        // Extend r with verifier's folding challenges
        verifier_randomness
            .extend(&verify_sumcheck_rounds(verifier, &mut sum, folding, 0, false).unwrap());

        num_vars_inter -= folding;
    }

    // Check reconstructed constraints match original ones
    for (expected, actual) in constraints
        .clone()
        .iter()
        .flatten()
        .zip(statement.constraints.clone().iter())
    {
        assert_eq!(expected, actual);
    }

    // Final round check
    let final_rounds = *folding_factors.last().unwrap();
    verifier_randomness
        .extend(&verify_sumcheck_rounds(verifier, &mut sum, final_rounds, 0, false).unwrap());

    // Check that the randomness vectors are the same
    assert_eq!(prover_randomness, verifier_randomness);

    // Final folded constant from transcript
    let final_folded_value_transcript = verifier.next_extension_scalar().unwrap();
    assert_eq!(final_folded_value, final_folded_value_transcript);

    // CHECK EQ(z, r) WEIGHT POLY
    let eq_eval = eval_constraints_poly(
        num_vars,
        folding_factors,
        &constraints,
        &alphas,
        verifier_randomness,
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
#[allow(clippy::collection_is_never_read)]
fn run_sumcheck_test_skips(folding_factors: &[usize], num_points: &[usize]) {
    // Sanity check: folding_factors should have one more element than num_points
    assert_eq!(folding_factors.len(), num_points.len() + 1);

    // Total number of variables in the original multilinear polynomial
    let num_vars = folding_factors.iter().sum::<usize>();

    // SETUP POLYNOMIAL
    //
    // Generate a random multilinear polynomial of arity `num_vars`
    let mut rng = SmallRng::seed_from_u64(1);
    let poly = EvaluationsList::new(rand_vec(&mut rng, 1 << num_vars));

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
        make_inter_statement(prover, num_pts, &mut sumcheck);

        // Fold the sumcheck polynomial again and extend randomness vector
        prover_randomness.extend(&sumcheck.compute_sumcheck_polynomials(prover, folding, 0));
        num_vars_inter -= folding;

        // Sanity check: number of variables and evaluations should be correct
        assert_eq!(sumcheck.evals.num_variables(), num_vars_inter);
        assert_eq!(sumcheck.evals.num_evals(), 1 << num_vars_inter);
    }

    // FINAL ROUND
    let final_rounds = *folding_factors.last().unwrap();
    prover_randomness.extend(&sumcheck.compute_sumcheck_polynomials(prover, final_rounds, 0));

    // After final round, polynomial must collapse to a constant
    assert_eq!(num_vars_inter, final_rounds);
    assert_eq!(sumcheck.evals.num_variables(), 0);
    assert_eq!(sumcheck.evals.num_evals(), 1);

    // Final constant should be f(r), where r is the accumulated challenge point
    let constant = sumcheck.evals[0];

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
    let mut verifier_randomness = MultilinearPoint(vec![]);

    // Track challenge alphas used for combining constraints in each round
    let mut alphas = vec![];

    // All constraints used by the verifier across rounds
    let mut constraints = vec![];

    // Recompute the same variable count as prover had
    let mut num_vars_inter = num_vars;

    // VERIFY EACH ROUND
    for (&folding, &num_pts) in folding_factors.iter().zip(num_points.iter()) {
        // Reconstruct the equality statement from the proof stream
        let statement = read_statement(verifier, num_vars_inter, num_pts);

        // Sample αᵢ for combining constraints
        let alpha = verifier.sample();
        alphas.push(alpha);

        // Accumulate the weighted sum of constraint values
        combine_constraints(&mut sum, &statement.constraints, alpha);

        // Save constraints for later equality check
        constraints.push(statement.constraints.clone());

        // Extend r with verifier's folding randomness
        verifier_randomness
            .extend(&verify_sumcheck_rounds(verifier, &mut sum, folding, 0, true).unwrap());

        num_vars_inter -= folding;
    }

    // FINAL FOLDING
    let final_rounds = *folding_factors.last().unwrap();
    verifier_randomness
        .extend(&verify_sumcheck_rounds(verifier, &mut sum, final_rounds, 0, true).unwrap());

    // Check that the randomness vectors are the same
    assert_eq!(prover_randomness, verifier_randomness);

    // Final constant from transcript must match prover's
    let constant_verifier = verifier.next_extension_scalar().unwrap();
    assert_eq!(constant_verifier, constant);

    // TODO: This should be fixed somehow for the univariate skip to work
    // // EVALUATE EQ(z, r) VIA CONSTRAINTS
    // //
    // // Evaluate eq(z, r) using the multilinear constraint interpolation
    // let eq_eval = eval_constraints_poly(
    //     num_vars,
    //     folding_factors,
    //     &constraints,
    //     &alphas,
    //     verifier_randomness.clone(),
    // );

    // // FINAL SUMCHECK CHECK
    // //
    // // Main equation: sum == f(r) · eq(z, r)
    // assert_eq!(sum, constant * eq_eval);

    // EVALUATE EQ(z, r) VIA CONSTRAINTS — using the exact skip pipeline as the prover.
    let k_skip = K_SKIP_SUMCHECK;

    // In this test we only have one round of constraints combined with a single alpha.
    let constraints_round0 = &constraints[0];
    let alpha_round0 = alphas[0];
    let alpha_pows = alpha_round0.powers().collect_n(constraints_round0.len());

    // 1) Combine the equality weights into a single weight table W over B^n.
    let mut w_all = EvaluationsList::new(vec![EF::ZERO; 1 << num_vars]);
    for (c, &a_i) in constraints_round0.iter().zip(&alpha_pows) {
        // Accumulate eq_{z_i}(·) with coefficient a_i.
        c.weights.accumulate::<F, true>(&mut w_all, a_i);
    }

    // Number of variables for the multilinear polynomial f(X)
    let n = poly.num_variables();
    // Number of remaining variables after skipping k
    let num_remaining_vars = n - k_skip;
    // Number of columns = 2^{n-k}
    let width = 1 << num_remaining_vars;

    // Reorganize the weights into a matrix
    let w_mat = RowMajorMatrix::new(w_all.to_vec(), width);

    // Evaluate folded weights at the skip challenge, then on the remaining variables.
    let r_skip = *verifier_randomness
        .0
        .last()
        .expect("skip challenge present");
    let folded_w_row = interpolate_subgroup(&w_mat, r_skip);
    let w_folded = EvaluationsList::new(folded_w_row);

    // Rest of challenges are the first (n - k) entries, in order.
    let r_rest = MultilinearPoint(verifier_randomness.0[..(num_vars - k_skip)].to_vec());

    // This is Σ α^i·eq(z_i, r) evaluated with the same domain/transform as the prover.
    let eq_eval = w_folded.evaluate(&r_rest);

    // FINAL SUMCHECK CHECK
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
        r_all.len() >= need,
        "need {} challenges (1 + n - k), got {}",
        need,
        r_all.len()
    );

    // - r0 is the **last** element (skip challenge),
    // - "rest" are the first n-k challenges for the remaining variables.
    let r0 = *r_all.last().unwrap();
    let rest = MultilinearPoint(r_all[..(n - k_skip)].to_vec());

    // Reshape f into 2^k × 2^{n-k}, interpolate along the skipped dimension at r0,
    // then evaluate the resulting EF-table on the remaining variables.
    let width = 1 << (n - k_skip);
    let f_mat = RowMajorMatrix::new(poly.to_vec(), width);
    let folded_row = interpolate_subgroup::<F, EF, _>(&f_mat, r0);

    EvaluationsList::new(folded_row).evaluate(&rest)
}

#[test]
fn test_sumcheck_prover() {
    run_sumcheck_test(&[1, 0], &[1]);
    run_sumcheck_test(&[1, 1], &[1]);
    run_sumcheck_test(&[1, 1], &[9]);
    run_sumcheck_test(&[4, 1], &[9]);
    run_sumcheck_test(&[1, 4], &[9]);
    run_sumcheck_test(&[1, 1, 1], &[1, 1]);
    run_sumcheck_test(&[4, 1, 4], &[9, 9]);
    run_sumcheck_test(&[4, 4, 1], &[9, 9]);
    run_sumcheck_test(&[1, 4, 4], &[9, 9]);

    // Folds 6 variables with skip, then 1 standard. Total 7 variables. 1 constraint.
    run_sumcheck_test_skips(&[6, 1], &[1]);

    // Tests the system with a higher number of initial equality constraints.
    // Total 7 variables (6 skipped, 1 standard). 5 constraints.
    run_sumcheck_test_skips(&[6, 1], &[5]);

    // Increases the total number of variables, with more processed in the final standard round.
    // Total 10 variables (6 skipped, 4 standard). 3 constraints.
    run_sumcheck_test_skips(&[6, 4], &[3]);

    // The final round folds 0 variables, testing an edge case.
    // Total 6 variables (all 6 skipped). 4 constraints.
    run_sumcheck_test_skips(&[6, 0], &[4]);

    // The `with_skip` function will perform one skip of `K_SKIP_SUMCHECK` (6) variables,
    // followed by 2 standard folding rounds, all within the first logical round.
    // Total 10 variables (8 folded in round 0, 2 in final round). 3 constraints.
    run_sumcheck_test_skips(&[8, 2], &[3]);

    // This provides a more complex scenario with a skip round, an intermediate standard
    // round with new constraints, and a final round.
    // Total 10 variables (6 skipped, then 2, then 2). 3 initial and 3 intermediate constraints.
    run_sumcheck_test_skips(&[6, 2, 2], &[3, 3]);
}
