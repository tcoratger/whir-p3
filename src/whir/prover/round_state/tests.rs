use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};
use p3_matrix::dense::DenseMatrix;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::{SeedableRng, rngs::SmallRng};

use crate::{
    dft::EvalsDft,
    parameters::{FoldingFactor, ProtocolParameters, errors::SecurityAssumption},
    poly::{coeffs::CoefficientList, evals::EvaluationsList, multilinear::MultilinearPoint},
    whir::{
        WhirConfig,
        committer::{Witness, writer::CommitmentWriter},
        constraints::statement::Statement,
        prover::{Prover, round_state::RoundState},
        proof::WhirProof,
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

/// Create a WHIR protocol configuration for test scenarios.
///
/// This utility function builds a `WhirConfig` using the provided parameters:
/// - `num_variables`: Number of variables in the multilinear polynomial.
/// - `protocol_parameters`: The protocol parameters for WHIR.
///
/// The returned config can be used to initialize a prover and set up domain commitments
/// for round state construction in WHIR tests.
fn make_test_config(
    num_variables: usize,
    protocol_parameters: ProtocolParameters<MyHash, MyCompress>
) -> WhirConfig<EF4, F, MyHash, MyCompress, MyChallenger> {
    // Combine the multivariate and protocol parameters into a full WHIR config
    WhirConfig::new(num_variables, protocol_parameters)
}

/// Create test protocol parameters with the specified configuration.
///
/// This helper function creates a `ProtocolParameters` structure with:
/// - `initial_statement`: Whether to start with an initial sumcheck statement.
/// - `folding_factor`: Number of variables to fold per round.
/// - `pow_bits`: Difficulty of the proof-of-work challenge used in Fiat-Shamir.
fn make_test_params(
    initial_statement: bool,
    folding_factor: usize,
    pow_bits: usize
) -> ProtocolParameters<MyHash, MyCompress> {
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);

    let merkle_hash = MyHash::new(perm.clone());
    let merkle_compress = MyCompress::new(perm);

    // Define the core protocol parameters for WHIR, customizing behavior based
    // on whether to start with an initial sumcheck and how to fold the polynomial.
    ProtocolParameters {
        initial_statement,
        security_level: 80,
        pow_bits,
        rs_domain_initial_reduction_factor: 1,
        folding_factor: FoldingFactor::Constant(folding_factor),
        merkle_hash,
        merkle_compress,
        soundness_type: SecurityAssumption::CapacityBound,
        starting_log_inv_rate: 1,
        univariate_skip: false,
    }
}

/// Prepare the challenger, WhirProof, and Merkle commitment for a test polynomial.
///
/// This helper sets up the challenger with domain separation and commits to a polynomial
/// using a `CommitmentWriter`. It returns:
/// - the initialized challenger with domain separation applied,
/// - the `WhirProof` structure,
/// - and a `Witness` object containing the committed polynomial and Merkle data.
///
/// This follows the pattern used in main.rs for the new API.
#[allow(clippy::type_complexity)]
fn setup_domain_and_commitment(
    params: &WhirConfig<EF4, F, MyHash, MyCompress, MyChallenger>,
    poly: EvaluationsList<F>,
) -> (
    MyChallenger,
    WhirProof<F, EF4, DIGEST_ELEMS>,
    Witness<EF4, F, DenseMatrix<F>, DIGEST_ELEMS>,
) {
    use p3_challenger::CanObserve;
    use crate::whir::proof::InitialPhase;

    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);

    // Build ProtocolParameters from WhirConfig fields
    let protocol_params = ProtocolParameters {
        initial_statement: params.initial_statement,
        security_level: params.security_level,
        pow_bits: params.starting_folding_pow_bits,
        folding_factor: params.folding_factor,
        merkle_hash: params.merkle_hash.clone(),
        merkle_compress: params.merkle_compress.clone(),
        soundness_type: params.soundness_type,
        starting_log_inv_rate: params.starting_log_inv_rate,
        rs_domain_initial_reduction_factor: 1,
        univariate_skip: false,
    };

    // Create WhirProof structure from protocol parameters
    let mut whir_proof = WhirProof::from_protocol_parameters(&protocol_params, poly.num_variables());

    // Create challenger and apply domain separation by observing proof structure
    let mut challenger = MyChallenger::new(perm);

    // Domain separation: observe proof structure (matching main.rs pattern)
    challenger.observe(F::from_u64(whir_proof.rounds.len() as u64));
    challenger.observe(F::from_u64(whir_proof.final_queries.len() as u64));

    // Observe initial phase variant
    let phase_tag = match &whir_proof.initial_phase {
        InitialPhase::WithoutStatement { .. } => 0,
        InitialPhase::WithStatement { .. } => 1,
        InitialPhase::WithStatementSkip { .. } => 2,
    };
    challenger.observe(F::from_u64(phase_tag as u64));

    // Create a committer using the protocol configuration
    let committer = CommitmentWriter::new(params);

    // Perform DFT-based commitment to the polynomial
    let witness = committer
        .commit(&EvalsDft::<F>::default(), &mut whir_proof, &mut challenger, poly);

    // Return all initialized components needed for round state setup
    (challenger, whir_proof, witness)
}

#[test]
fn test_no_initial_statement_no_sumcheck() {
    // Number of variables in the multilinear polynomial
    let num_variables = 2;

    // Create protocol parameters and WHIR config with:
    // - no initial sumcheck,
    // - folding factor 2,
    // - no PoW grinding.
    let params = make_test_params(false, 2, 0);
    let config = make_test_config(num_variables, params);

    // Define a polynomial
    let poly = EvaluationsList::new(vec![F::from_u64(3); 1 << num_variables]);

    // Initialize:
    // - challenger with domain separation,
    // - WhirProof structure,
    // - witness containing Merkle tree for `poly`.
    let (mut challenger, mut whir_proof, witness) = setup_domain_and_commitment(&config, poly);

    // Create an empty public statement (no constraints)
    let statement = Statement::<EF4>::initialize(num_variables);

    // Initialize the round state using the setup configuration and witness
    let state = RoundState::initialize_first_round_state(
        &Prover(&config),
        &mut whir_proof,
        &mut challenger,
        statement,
        witness,
    );

    // Folding factor was 2, so we expect 2 sampled folding randomness values
    assert_eq!(state.folding_randomness.num_variables(), 2);

    // Full randomness vector should be padded up to `num_variables`
    assert_eq!(state.randomness_vec.len(), num_variables);

    // Since this is the first round, no Merkle data for folded rounds should exist
    assert!(state.merkle_prover_data.is_none());
}

#[test]
fn test_initial_statement_with_folding_factor_3() {
    // Set the number of variables in the multilinear polynomial
    let num_variables = 3;

    // Create protocol parameters and WHIR configuration with:
    // - initial statement enabled (sumcheck will run),
    // - folding factor = 3 (fold all variables in the first round),
    // - PoW disabled.
    let params = make_test_params(true, 3, 0);
    let config = make_test_config(num_variables, params);

    // Define the multilinear polynomial:
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

    let poly = CoefficientList::new(vec![c1, c2, c3, c4, c5, c6, c7, c8]).to_evaluations();

    // Manual redefinition of the same polynomial as a function for evaluation
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

    // Add a single equality constraint to the statement: f(1,1,1) = expected value
    let mut statement = Statement::<EF4>::initialize(num_variables);
    statement.add_evaluated_constraint(
        MultilinearPoint::new(vec![EF4::ONE, EF4::ONE, EF4::ONE]),
        f(EF4::ONE, EF4::ONE, EF4::ONE),
    );

    // Set up the challenger, proof, and witness for this configuration
    let (mut challenger, mut whir_proof, witness) = setup_domain_and_commitment(&config, poly);

    // Run the first round state initialization (this will trigger sumcheck)
    let state = RoundState::initialize_first_round_state(
        &Prover(&config),
        &mut whir_proof,
        &mut challenger,
        statement,
        witness,
    );

    // Extract the constructed sumcheck prover and folding randomness
    let sumcheck = &state.sumcheck_prover;
    let sumcheck_randomness = state.folding_randomness.clone();

    // With a folding factor of 3, all variables are collapsed in 1 round, so we expect only 1 evaluation left
    assert_eq!(sumcheck.evals.num_evals(), 1);

    // The value of f at the folding point should match the evaluation
    let eval_at_point = sumcheck.evals.as_slice()[0];
    let expected = f(
        sumcheck_randomness[0],
        sumcheck_randomness[1],
        sumcheck_randomness[2],
    );
    assert_eq!(eval_at_point, expected);

    // Check that dot product of evaluations and weights matches the final sum
    let dot_product: EF4 = sumcheck
        .evals
        .iter()
        .zip(&sumcheck.weights)
        .map(|(&f, &w)| f * w)
        .sum();
    assert_eq!(dot_product, sumcheck.sum);

    // Verify that the `randomness_vec` (which is in reverse variable order) matches the expected layout
    assert_eq!(
        state.randomness_vec,
        vec![
            sumcheck_randomness[2],
            sumcheck_randomness[1],
            sumcheck_randomness[0]
        ]
    );

    // The `folding_randomness` should store values in forward order (X0, X1, X2)
    assert_eq!(
        state.folding_randomness.as_slice(),
        vec![
            sumcheck_randomness[0],
            sumcheck_randomness[1],
            sumcheck_randomness[2]
        ]
    );

    // No folded Merkle tree data should exist at this point
    assert!(state.merkle_prover_data.is_none());
}

#[test]
fn test_zero_poly_multiple_constraints() {
    // Use a polynomial with 3 variables
    let num_variables = 3;

    // Build protocol parameters and WHIR config with an initial statement, folding factor 1, and no PoW
    let params = make_test_params(true, 1, 0);
    let config = make_test_config(num_variables, params);

    // Define a zero polynomial: f(X) = 0 for all X
    let poly = EvaluationsList::new(vec![F::ZERO; 1 << num_variables]);

    // Generate challenger, proof, and Merkle commitment witness for the poly
    let (mut challenger, mut whir_proof, witness) = setup_domain_and_commitment(&config, poly);

    // Create a new statement with multiple constraints
    let mut statement = Statement::<EF4>::initialize(num_variables);

    // Add one equality constraint per Boolean input: f(x) = 0 for all x ∈ {0,1}³
    for i in 0..1 << num_variables {
        let point = (0..num_variables)
            .map(|b| EF4::from_u64(((i >> b) & 1) as u64))
            .collect();
        statement.add_evaluated_constraint(MultilinearPoint::new(point), EF4::ZERO);
    }

    // Initialize the first round of the WHIR protocol with the zero polynomial and constraints
    let state = RoundState::initialize_first_round_state(
        &Prover(&config),
        &mut whir_proof,
        &mut challenger,
        statement,
        witness,
    );

    // Extract the sumcheck prover and folding randomness
    let sumcheck = &state.sumcheck_prover;
    let sumcheck_randomness = state.folding_randomness.clone();

    for (f, w) in sumcheck.evals.iter().zip(&sumcheck.weights) {
        // Each evaluation should be 0
        assert_eq!(*f, EF4::ZERO);
        // Their contribution to the weighted sum should also be 0
        assert_eq!(*f * *w, EF4::ZERO);
    }
    // Final claimed sum is 0
    assert_eq!(sumcheck.sum, EF4::ZERO);

    // Folding randomness should have length equal to the folding factor (1)
    assert_eq!(sumcheck_randomness.num_variables(), 1);

    // The `randomness_vec` is populated in reverse variable order, padded with 0s
    assert_eq!(
        state.randomness_vec,
        vec![sumcheck_randomness[0], EF4::ZERO, EF4::ZERO]
    );

    // Confirm that folding randomness matches exactly
    assert_eq!(
        state.folding_randomness,
        MultilinearPoint::new(vec![sumcheck_randomness[0]])
    );

    // No Merkle commitment data for folded rounds yet
    assert!(state.merkle_prover_data.is_none());
}

#[test]
fn test_initialize_round_state_with_initial_statement() {
    // Use a polynomial in 3 variables
    let num_variables = 3;

    // Set PoW grinding difficulty (used in Fiat-Shamir)
    let pow_bits = 4;

    // Build protocol parameters and WHIR configuration with:
    // - initial statement enabled,
    // - folding factor of 1 (fold one variable in the first round),
    // - PoW bits enabled.
    let params = make_test_params(true, 1, pow_bits);
    let config = make_test_config(num_variables, params);

    // Define a multilinear polynomial:
    // f(X0, X1, X2) = 1 + 2*X2 + 3*X1 + 4*X1*X2 + 5*X0 + 6*X0*X2 + 7*X0*X1 + 8*X0*X1*X2
    let c1 = F::from_u64(1);
    let c2 = F::from_u64(2);
    let c3 = F::from_u64(3);
    let c4 = F::from_u64(4);
    let c5 = F::from_u64(5);
    let c6 = F::from_u64(6);
    let c7 = F::from_u64(7);
    let c8 = F::from_u64(8);
    let poly = CoefficientList::new(vec![c1, c2, c3, c4, c5, c6, c7, c8]).to_evaluations();

    // Equivalent function for evaluating the polynomial manually
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

    // Construct a statement with one evaluation constraint at the point (1, 0, 1)
    let mut statement = Statement::<EF4>::initialize(num_variables);
    statement.add_evaluated_constraint(
        MultilinearPoint::new(vec![EF4::ONE, EF4::ZERO, EF4::ONE]),
        f(EF4::ONE, EF4::ZERO, EF4::ONE),
    );

    // Set up challenger, proof, and produce commitment + witness
    let (mut challenger, mut whir_proof, witness) = setup_domain_and_commitment(&config, poly);

    // Run the first round initialization
    let state = RoundState::initialize_first_round_state(
        &Prover(&config),
        &mut whir_proof,
        &mut challenger,
        statement,
        witness,
    );

    // Unwrap the sumcheck prover and get the sampled folding randomness
    let sumcheck = &state.sumcheck_prover;
    let sumcheck_randomness = &state.folding_randomness;

    // Evaluate f at (32636, 9876, r0) and match it with the sumcheck's recovered evaluation
    let evals_f = &sumcheck.evals;
    assert_eq!(
        evals_f.evaluate(&MultilinearPoint::new(vec![
            EF4::from_u64(32636),
            EF4::from_u64(9876)
        ])),
        f(
            EF4::from_u64(32636),
            EF4::from_u64(9876),
            sumcheck_randomness[0]
        )
    );

    let evals_f = evals_f.as_slice();
    let weights = sumcheck.weights.as_slice();

    // Manually verify that ⟨f, w⟩ = claimed sum
    let dot_product = evals_f[0] * weights[0]
        + evals_f[1] * weights[1]
        + evals_f[2] * weights[2]
        + evals_f[3] * weights[3];
    assert_eq!(dot_product, sumcheck.sum);

    // No Merkle tree data has been created for folded rounds yet
    assert!(state.merkle_prover_data.is_none());

    // The randomness_vec must contain the sampled folding randomness, reversed and zero-padded
    assert_eq!(
        state.randomness_vec,
        vec![sumcheck_randomness[0], EF4::ZERO, EF4::ZERO]
    );

    // The folding randomness must match what was sampled by the sumcheck
    assert_eq!(
        state.folding_randomness,
        MultilinearPoint::new(vec![sumcheck_randomness[0]])
    );
}
