use alloc::{vec, vec::Vec};

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::{DuplexChallenger, FieldChallenger, GrindingChallenger};
use p3_field::{
    BasedVectorSpace, PrimeCharacteristicRing, TwoAdicField, extension::BinomialExtensionField,
};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::{Rng, SeedableRng, rngs::SmallRng};

use super::sumcheck_single::SumcheckSingle;
use crate::{
    constant::K_SKIP_SUMCHECK,
    fiat_shamir::{
        domain_separator::DomainSeparator, prover::ProverState, verifier::VerifierState,
    },
    parameters::{FoldingFactor, ProtocolParameters, errors::SecurityAssumption},
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    whir::{
        constraints::{
            Constraint,
            evaluator::ConstraintPolyEvaluator,
            statement::{EqStatement, SelectStatement},
        },
        parameters::InitialPhaseConfig,
        proof::{InitialPhase, WhirProof, WhirRoundProof},
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

/// Constructs a fresh `VerifierState` from a given proof, using a domain separator and challenger.
fn verifier(proof: Vec<F>) -> VerifierState<F, EF, MyChallenger> {
    // Create a domain separator and challenger using deterministic RNG
    let (domsep, challenger) = domainsep_and_challenger();

    // Use the domain separator to construct a new verifier state with the provided proof and challenger
    domsep.to_verifier_state(proof, challenger)
}

fn make_constraint<Challenger>(
    prover: &mut ProverState<F, EF, Challenger>,
    challenger: &mut Challenger,
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
        // Keep challenger_rf in sync
        let _point_rf: EF = challenger.sample_algebra_element();

        // Expand it into a `num_vars`-dimensional multilinear point.
        let point = MultilinearPoint::expand_from_univariate(point, num_vars);

        // Evaluate the current sumcheck polynomial at the sampled point.
        let eval = poly.evaluate_hypercube(&point);

        // Add the evaluation result to the transcript for Fiat-Shamir soundness.
        prover.add_extension_scalar(eval);
        // Keep challenger_rf in sync
        challenger.observe_slice(&EF::flatten_to_base(vec![eval]));

        // Add the evaluation constraint: poly(point) == eval.
        eq_statement.add_evaluated_constraint(point, eval);
    });

    // - Sample `num_sels` univariate challenge points.
    // - Evaluate the sumcheck polynomial on them.
    // - Collect (var, eval) pairs for use in the statement and constraint aggregation.
    (0..num_sels).for_each(|_| {
        // Simulate stir point derivation
        let index: usize = prover.sample_bits(num_vars);
        // Keep challenger_rf in sync
        let _index_rf: usize = challenger.sample_bits(num_vars);
        let var = omega.exp_u64(index as u64);

        // Evaluate the current sumcheck polynomial as univariate at the sampled variable.
        let eval = poly
            .iter()
            .rfold(EF::ZERO, |result, &coeff| result * var + coeff);

        // Add the evaluation result to the transcript for Fiat-Shamir soundness.
        prover.add_extension_scalar(eval);
        // Keep challenger_rf in sync
        challenger.observe_slice(&EF::flatten_to_base(vec![eval]));

        // Add the evaluation constraint: poly(point) == eval.
        sel_statement.add_constraint(var, eval);
    });

    // Return the constructed constraint with the alpha used for linear combination.
    let alpha = prover.sample();
    // Keep challenger_rf in sync
    let alpha_rf: EF = challenger.sample_algebra_element();
    assert_eq!(
        alpha, alpha_rf,
        "External challenger and prover_state challenger diverged"
    );
    Constraint::new(alpha, eq_statement, sel_statement)
}

fn make_constraint_ext<Challenger>(
    prover: &mut ProverState<F, EF, Challenger>,
    challenger: &mut Challenger,
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
        // Keep challenger_rf in sync
        let point_rf: EF = challenger.sample_algebra_element();
        assert_eq!(
            point, point_rf,
            "External challenger and prover_state challenger diverged"
        );
        // Expand it into a `num_vars`-dimensional multilinear point.
        let point = MultilinearPoint::expand_from_univariate(point, num_vars);

        // Evaluate the current sumcheck polynomial at the sampled point.
        let eval = poly.evaluate_hypercube(&point);

        // Add the evaluation result to the transcript for Fiat-Shamir soundness.
        prover.add_extension_scalar(eval);
        // Keep challenger_rf in sync
        challenger.observe_slice(&EF::flatten_to_base(vec![eval]));

        // Add the evaluation constraint: poly(point) == eval.
        eq_statement.add_evaluated_constraint(point, eval);
    });

    // - Sample `num_sels` univariate challenge points.
    // - Evaluate the sumcheck polynomial on them.
    // - Collect (var, eval) pairs for use in the statement and constraint aggregation.
    (0..num_sels).for_each(|_| {
        // Simulate stir point derivation
        let index: usize = prover.sample_bits(num_vars);
        // Keep challenger_rf in sync
        let index_rf: usize = challenger.sample_bits(num_vars);
        assert_eq!(
            index, index_rf,
            "External challenger and prover_state challenger diverged"
        );

        let var = omega.exp_u64(index as u64);

        // Evaluate the current sumcheck polynomial as univariate at the sampled variable.
        let eval = poly
            .iter()
            .rfold(EF::ZERO, |result, &coeff| result * var + coeff);

        // Add the evaluation result to the transcript for Fiat-Shamir soundness.
        prover.add_extension_scalar(eval);
        // Keep challenger_rf in sync
        challenger.observe_slice(&EF::flatten_to_base(vec![eval]));

        // Add the evaluation constraint: poly(point) == eval.
        sel_statement.add_constraint(var, eval);
    });

    // Return the constructed constraint with the alpha used for linear combination.
    let alpha = prover.sample();
    let alpha_rf: EF = challenger.sample_algebra_element();
    assert_eq!(
        alpha, alpha_rf,
        "External challenger and prover_state challenger diverged"
    );
    Constraint::new(alpha, eq_statement, sel_statement)
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
    let (domsep, challenger_for_prover) = domainsep_and_challenger();
    let prover = &mut domsep.to_prover_state(challenger_for_prover);

    // Initialize proof and challenger
    let params =
        create_test_protocol_params(folding_factor, InitialPhaseConfig::WithStatementClassic);
    let mut proof = WhirProof::<F, EF, 8>::from_protocol_parameters(&params, num_vars);
    let mut rng = SmallRng::seed_from_u64(1);
    let mut challenger_rf = MyChallenger::new(Perm::new_from_rng_128(&mut rng));
    domsep.observe_domain_separator(&mut challenger_rf);

    // Create the initial constraint statement
    let constraint = make_constraint(
        prover,
        &mut challenger_rf,
        num_vars,
        num_eqs[0],
        num_sels[0],
        &poly,
    );

    // ROUND 0
    let folding0 = folding_factor.at_round(0);
    // Extract sumcheck data from the initial phase
    let InitialPhase::WithStatement { ref mut sumcheck } = proof.initial_phase else {
        panic!("Expected WithStatement variant");
    };
    let (mut sumcheck, mut prover_randomness) = SumcheckSingle::from_base_evals(
        &poly,
        prover,
        sumcheck,
        &mut challenger_rf,
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
        let constraint = make_constraint_ext(
            prover,
            &mut challenger_rf,
            num_vars_inter,
            num_eq_points,
            num_sel_points,
            &sumcheck.evals,
        );

        proof.rounds.push(WhirRoundProof::default());

        // Compute and apply the next folding round
        prover_randomness.extend(&sumcheck.compute_sumcheck_polynomials(
            prover,
            &mut proof,
            &mut challenger_rf,
            folding,
            0,
            false,
            Some(constraint),
        ));

        num_vars_inter -= folding;

        // Check that the number of variables and evaluations match the expected values
        assert_eq!(sumcheck.evals.num_variables(), num_vars_inter);
        assert_eq!(sumcheck.evals.num_evals(), 1 << num_vars_inter);
    }

    // Ensure we've folded all variables.
    assert_eq!(num_vars_inter, final_rounds);

    // FINAL ROUND
    prover_randomness.extend(&sumcheck.compute_sumcheck_polynomials(
        prover,
        &mut proof,
        &mut challenger_rf,
        final_rounds,
        0,
        true,
        None,
    ));
    let final_folded_value = sumcheck.evals.as_constant().unwrap();

    assert_eq!(sumcheck.evals.num_variables(), 0);
    assert_eq!(sumcheck.evals.num_evals(), 1);

    // Final folded value must match f(r)
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
        verifier_randomness.extend(
            &verify_sumcheck_rounds(
                verifier,
                &mut sum,
                folding,
                0,
                InitialPhaseConfig::WithStatementClassic,
            )
            .unwrap(),
        );

        num_vars_inter -= folding;
    }

    // Final round check
    verifier_randomness.extend(
        &verify_sumcheck_rounds(
            verifier,
            &mut sum,
            final_rounds,
            0,
            InitialPhaseConfig::WithStatementClassic,
        )
        .unwrap(),
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
    let weights = evaluator.eval_constraints_poly(&constraints, &verifier_randomness.reversed());

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
    let (domsep, challenger_for_prover) = domainsep_and_challenger();
    let prover = &mut domsep.to_prover_state(challenger_for_prover);

    // Initialize proof and challenger
    let params = create_test_protocol_params(
        folding_factor,
        InitialPhaseConfig::WithStatementUnivariateSkip,
    );
    let mut proof = WhirProof::<F, EF, 8>::from_protocol_parameters(&params, num_vars);
    let mut rng = SmallRng::seed_from_u64(1);
    let mut challenger_rf = MyChallenger::new(Perm::new_from_rng_128(&mut rng));
    domsep.observe_domain_separator(&mut challenger_rf);

    // Sample and commit initial evaluation constraints
    let constraint = make_constraint(
        prover,
        &mut challenger_rf,
        num_vars,
        num_eq_points[0],
        num_sel_points[0],
        &poly,
    );
    constraint.validate_for_skip_case();

    // ROUND 0
    // Initialize sumcheck with univariate skip (skips K_SKIP_SUMCHECK)
    let folding0 = folding_factor.at_round(0);
    // Extract skip fields from the initial phase
    let InitialPhase::WithStatementSkip {
        ref mut skip_evaluations,
        ref mut skip_pow,
        ref mut sumcheck,
    } = proof.initial_phase
    else {
        panic!("Expected WithStatementSkip variant");
    };
    let (mut sumcheck, mut prover_randomness) = SumcheckSingle::with_skip(
        &poly,
        prover,
        skip_evaluations,
        skip_pow,
        sumcheck,
        &mut challenger_rf,
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
        let constraint = make_constraint_ext(
            prover,
            &mut challenger_rf,
            num_vars_inter,
            num_eq_points,
            num_sel_points,
            &sumcheck.evals,
        );

        // Fold the sumcheck polynomial again and extend randomness vector
        prover_randomness.extend(&sumcheck.compute_sumcheck_polynomials(
            prover,
            &mut proof,
            &mut challenger_rf,
            folding,
            0,
            false,
            Some(constraint),
        ));

        num_vars_inter -= folding;

        // Sanity check: number of variables and evaluations should be correct
        assert_eq!(sumcheck.evals.num_variables(), num_vars_inter);
        assert_eq!(sumcheck.evals.num_evals(), 1 << num_vars_inter);
    }

    // Ensure we've folded all variables.
    assert_eq!(num_vars_inter, final_rounds);

    // FINAL ROUND
    prover_randomness.extend(&sumcheck.compute_sumcheck_polynomials(
        prover,
        &mut proof,
        &mut challenger_rf,
        final_rounds,
        0,
        true,
        None,
    ));

    // After final round, polynomial must collapse to a constant
    assert_eq!(sumcheck.evals.num_variables(), 0);
    assert_eq!(sumcheck.evals.num_evals(), 1);

    // Final constant should be f(r), where r is the accumulated challenge point
    let final_folded_value = sumcheck.evals.as_constant().unwrap();

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
        let initial_phase = if round_idx == 0 {
            InitialPhaseConfig::WithStatementUnivariateSkip
        } else {
            InitialPhaseConfig::WithStatementClassic
        };
        let folding = folding_factor.at_round(round_idx);
        verifier_randomness.extend(
            &verify_sumcheck_rounds(verifier, &mut sum, folding, 0, initial_phase).unwrap(),
        );

        num_vars_inter -= folding;
    }

    // FINAL FOLDING
    verifier_randomness.extend(
        &verify_sumcheck_rounds(
            verifier,
            &mut sum,
            final_rounds,
            0,
            InitialPhaseConfig::WithStatementClassic,
        )
        .unwrap(),
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
    let (domsep, challenger_for_prover) = domainsep_and_challenger();
    let prover = &mut domsep.to_prover_state(challenger_for_prover);

    // Initialize proof and challenger
    let params = create_test_protocol_params(folding_factor, InitialPhaseConfig::WithStatementSvo);
    let mut proof = WhirProof::<F, EF, 8>::from_protocol_parameters(&params, num_vars);
    let mut rng = SmallRng::seed_from_u64(1);
    let mut challenger_rf = MyChallenger::new(Perm::new_from_rng_128(&mut rng));
    domsep.observe_domain_separator(&mut challenger_rf);

    // Create the initial constraint statement
    let constraint = make_constraint(
        prover,
        &mut challenger_rf,
        num_vars,
        num_eqs[0],
        num_sels[0],
        &poly,
    );

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
            &mut challenger_rf,
            num_vars_inter,
            num_eq_points,
            num_sel_points,
            &sumcheck.evals,
        );

        // Compute and apply the next folding round
        prover_randomness.extend(&sumcheck.compute_sumcheck_polynomials(
            prover,
            &mut proof,
            &mut challenger_rf,
            folding,
            0,
            false,
            Some(constraint),
        ));

        num_vars_inter -= folding;

        // Check that the number of variables and evaluations match the expected values
        assert_eq!(sumcheck.evals.num_variables(), num_vars_inter);
        assert_eq!(sumcheck.evals.num_evals(), 1 << num_vars_inter);
    }

    // Ensure we've folded all variables.
    assert_eq!(num_vars_inter, final_rounds);

    // FINAL ROUND
    prover_randomness.extend(&sumcheck.compute_sumcheck_polynomials(
        prover,
        &mut proof,
        &mut challenger_rf,
        final_rounds,
        0,
        true,
        None,
    ));

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
        verifier_randomness.extend(
            &verify_sumcheck_rounds(
                verifier,
                &mut sum,
                folding,
                0,
                InitialPhaseConfig::WithStatementSvo,
            )
            .unwrap(),
        );

        num_vars_inter -= folding;
    }

    // Final round check
    verifier_randomness.extend(
        &verify_sumcheck_rounds(
            verifier,
            &mut sum,
            final_rounds,
            0,
            InitialPhaseConfig::WithStatementSvo,
        )
        .unwrap(),
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
    let weights = evaluator.eval_constraints_poly(&constraints, &verifier_randomness.reversed());

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
