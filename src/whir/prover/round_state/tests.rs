use alloc::vec;

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};
use p3_matrix::dense::DenseMatrix;
use p3_merkle_tree::MerkleTree;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::{SeedableRng, rngs::SmallRng};

use crate::{
    fiat_shamir::domain_separator::DomainSeparator,
    parameters::{FoldingFactor, ProtocolParameters, errors::SecurityAssumption},
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    whir::{
        committer::writer::CommitmentWriter,
        constraints::statement::initial::InitialStatement,
        parameters::{SumcheckStrategy, WhirConfig},
        proof::WhirProof,
        prover::round_state::RoundState,
    },
};

type F = BabyBear;
type EF4 = BinomialExtensionField<F, 4>;
type Perm = Poseidon2BabyBear<16>;

type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

const DIGEST_ELEMS: usize = 8;

/// Create a WHIR protocol configuration for test scenarios.
///
/// This utility function builds a `WhirConfig` using the provided parameters:
/// - `num_variables`: Number of variables in the multilinear polynomial.
/// - `initial_phase_config`: Configuration for the initial phase.
/// - `folding_factor`: Number of variables to fold per round.
/// - `pow_bits`: Difficulty of the proof-of-work challenge used in Fiat-Shamir.
///
/// The returned config can be used to initialize a prover and set up domain commitments
/// for round state construction in WHIR tests.
fn make_test_config(
    num_variables: usize,
    folding_factor: usize,
    pow_bits: usize,
) -> WhirConfig<EF4, F, MyHash, MyCompress, MyChallenger> {
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);

    let merkle_hash = MyHash::new(perm.clone());
    let merkle_compress = MyCompress::new(perm);

    // Define the core protocol parameters for WHIR, customizing behavior based
    // on whether to start with an initial sumcheck and how to fold the polynomial.
    let protocol_params = ProtocolParameters {
        security_level: 80,
        pow_bits,
        rs_domain_initial_reduction_factor: 1,
        folding_factor: FoldingFactor::Constant(folding_factor),
        merkle_hash,
        merkle_compress,
        soundness_type: SecurityAssumption::CapacityBound,
        starting_log_inv_rate: 1,
    };

    // Combine the multivariate and protocol parameters into a full WHIR config
    WhirConfig::new(num_variables, protocol_params)
}

/// Prepare the Fiat-Shamir domain, prover state, and Merkle commitment for a test polynomial.
///
/// This helper sets up the necessary transcript (`DomainSeparator`) and
/// commits to a polynomial using a `CommitmentWriter`. It returns:
/// - the initialized domain separator,
/// - the `ProverState` transcript context for Fiat-Shamir interaction,
/// - and a `Witness` object containing the committed polynomial and Merkle data.
///
/// This is used as a boilerplate step before running the first WHIR round.
#[allow(clippy::type_complexity)]
fn setup_domain_and_commitment(
    params: &WhirConfig<EF4, F, MyHash, MyCompress, MyChallenger>,
    poly: EvaluationsList<F>,
) -> (
    WhirProof<F, EF4, F, DIGEST_ELEMS>,
    MyChallenger,
    MerkleTree<F, F, DenseMatrix<F>, DIGEST_ELEMS>,
) {
    // Build ProtocolParameters from WhirConfig fields
    let protocol_params = ProtocolParameters {
        security_level: params.security_level,
        pow_bits: params.starting_folding_pow_bits,
        folding_factor: params.folding_factor,
        merkle_hash: params.merkle_hash.clone(),
        merkle_compress: params.merkle_compress.clone(),
        soundness_type: params.soundness_type,
        starting_log_inv_rate: params.starting_log_inv_rate,
        rs_domain_initial_reduction_factor: 1,
    };

    // Create WhirProof structure from protocol parameters
    let whir_proof = WhirProof::from_protocol_parameters(&protocol_params, poly.num_variables());

    // Create a new Fiat-Shamir domain separator.
    let mut domsep = DomainSeparator::new(vec![]);

    // Observe the public statement into the transcript for binding.
    domsep.commit_statement::<_, _, _, 8>(params);

    // Reserve transcript space for WHIR proof messages.
    domsep.add_whir_proof::<_, _, _, 8>(params);

    // Convert the domain separator into a mutable prover-side transcript.
    let mut rng = SmallRng::seed_from_u64(1);
    let mut prover_challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));
    domsep.observe_domain_separator(&mut prover_challenger);

    // Create a committer using the protocol configuration (Merkle parameters, hashers, etc.).
    let committer = CommitmentWriter::new(params);
    let mut initial_statement = InitialStatement::new(poly, 0, SumcheckStrategy::default());
    let mut proof =
        WhirProof::from_protocol_parameters(&protocol_params, initial_statement.num_variables());

    // Perform DFT-based commitment to the polynomial, producing a prover data
    // which includes the Merkle tree and polynomial values.
    let prover_data = committer
        .commit::<_, <F as p3_field::Field>::Packing, F, <F as p3_field::Field>::Packing, DIGEST_ELEMS>(
            &Radix2DFTSmallBatch::<F>::default(),
            &mut proof,
            &mut prover_challenger,
            &mut initial_statement,
        )
        .unwrap();

    // Return all initialized components needed for round state setup.
    (whir_proof, prover_challenger, prover_data)
}

#[test]
fn test_no_initial_statement_no_sumcheck() {
    // Number of variables in the multilinear polynomial
    let num_variables = 2;

    // Create a WHIR protocol config with:
    // - no initial sumcheck,
    // - folding factor 2,
    // - no PoW grinding.
    let config = make_test_config(num_variables, 2, 0);
    let folding0 = config.folding_factor.at_round(0);

    // Define a polynomial
    let poly = EvaluationsList::new(vec![F::from_u64(3); 1 << num_variables]);

    // Initialize:
    // - domain separator for Fiat-Shamir transcript,
    // - prover state,
    // - prover data containing Merkle tree for `poly`.
    let (mut proof, mut challenger, prover_data) =
        setup_domain_and_commitment(&config, poly.clone());

    // Create an empty public statement (no constraints)
    let statement = InitialStatement::new(poly, folding0, SumcheckStrategy::default());

    // Initialize the round state using the setup configuration and prover data
    let state = RoundState::initialize_first_round_state(
        &mut proof.initial_sumcheck,
        &mut challenger,
        &statement,
        prover_data,
        config.folding_factor.at_round(0),
        0,
    )
    .unwrap();

    // Folding factor was 2, so we expect 2 sampled folding randomness values
    assert_eq!(state.folding_randomness.num_variables(), 2);

    // Since this is the first round, no Merkle data for folded rounds should exist
    assert!(state.merkle_prover_data.is_none());
}

#[test]
fn test_initial_statement_with_folding_factor_3() {
    // Set the number of variables in the multilinear polynomial
    let num_variables = 3;

    // Create a WHIR configuration with:
    // - initial statement enabled (sumcheck will run),
    // - folding factor = 3 (fold all variables in the first round),
    // - PoW disabled.
    let config = make_test_config(num_variables, 3, 0);
    let folding0 = config.folding_factor.at_round(0);

    // Define the multilinear polynomial:
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

    let poly = EvaluationsList::new(vec![
        e1,
        e1 + e2,
        e1 + e3,
        e1 + e2 + e3 + e4,
        e1 + e5,
        e1 + e2 + e5 + e6,
        e1 + e3 + e5 + e7,
        e1 + e2 + e3 + e4 + e5 + e6 + e7 + e8,
    ]);

    // Manual redefinition of the same polynomial as a function for evaluation
    let f = |x0: EF4, x1: EF4, x2: EF4| {
        x2 * e2
            + x1 * e3
            + x1 * x2 * e4
            + x0 * e5
            + x0 * x2 * e6
            + x0 * x1 * e7
            + x0 * x1 * x2 * e8
            + e1
    };

    // Set up the domain separator, prover state, and prover data for this configuration
    let (mut proof, mut challenger_rf, prover_data) =
        setup_domain_and_commitment(&config, poly.clone());

    // Add a single equality constraint to the statement: f(1,1,1) = expected value
    // Create an empty public statement (no constraints)
    let mut statement = InitialStatement::new(poly, folding0, SumcheckStrategy::default());
    // Evaluate at point (1, 1, 1)
    let _ = statement.evaluate(&MultilinearPoint::new(vec![EF4::ONE, EF4::ONE, EF4::ONE]));

    // Run the first round state initialization (this will trigger sumcheck)
    let state = RoundState::initialize_first_round_state(
        &mut proof.initial_sumcheck,
        &mut challenger_rf,
        &statement,
        prover_data,
        config.folding_factor.at_round(0),
        0,
    )
    .unwrap();

    // Extract the constructed sumcheck prover and folding randomness
    let sumcheck = &state.sumcheck_prover;
    let sumcheck_randomness = state.folding_randomness.clone();

    // With a folding factor of 3, all variables are collapsed in 1 round, so we expect only 1 evaluation left
    assert_eq!(sumcheck.poly.num_evals(), 1);

    // The value of f at the folding point should match the evaluation
    let eval_at_point = sumcheck.evals().as_slice()[0];
    let expected = f(
        sumcheck_randomness[0],
        sumcheck_randomness[1],
        sumcheck_randomness[2],
    );
    assert_eq!(eval_at_point, expected);

    // Check that dot product of evaluations and weights matches the final sum
    let dot_product: EF4 = sumcheck.poly.dot_product();
    assert_eq!(dot_product, sumcheck.sum);

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

    // Build a WHIR config with an initial statement, folding factor 1, and no PoW
    let config = make_test_config(num_variables, 1, 0);
    let folding0 = config.folding_factor.at_round(0);

    // Define a zero polynomial: f(X) = 0 for all X
    let poly = EvaluationsList::new(vec![F::ZERO; 1 << num_variables]);

    // Generate domain separator, prover state, and Merkle commitment witness for the poly
    let (mut proof, mut challenger_rf, prover_data) =
        setup_domain_and_commitment(&config, poly.clone());

    // Create a new statement with multiple constraints
    let mut statement = InitialStatement::new(poly, folding0, SumcheckStrategy::default());

    // Add one equality constraint per Boolean input: f(x) = 0 for all x ∈ {0,1}³
    for i in 0..1 << num_variables {
        let point = (0..num_variables)
            .map(|b| EF4::from_u64(((i >> b) & 1) as u64))
            .collect();
        let eval = statement.evaluate(&MultilinearPoint::new(point));
        assert_eq!(eval, EF4::ZERO);
    }

    // Initialize the first round of the WHIR protocol with the zero polynomial and constraints
    let state = RoundState::initialize_first_round_state(
        &mut proof.initial_sumcheck,
        &mut challenger_rf,
        &statement,
        prover_data,
        folding0,
        0,
    )
    .unwrap();

    // Extract the sumcheck prover and folding randomness
    let sumcheck = &state.sumcheck_prover;
    let sumcheck_randomness = state.folding_randomness.clone();

    for (f, w) in sumcheck.evals().iter().zip(&sumcheck.weights()) {
        // Each evaluation should be 0
        assert_eq!(*f, EF4::ZERO);
        // Their contribution to the weighted sum should also be 0
        assert_eq!(*f * *w, EF4::ZERO);
    }
    // Final claimed sum is 0
    assert_eq!(sumcheck.sum, EF4::ZERO);

    // Folding randomness should have length equal to the folding factor (1)
    assert_eq!(sumcheck_randomness.num_variables(), 1);

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

    // Build a WHIR configuration with:
    // - initial statement enabled,
    // - folding factor of 1 (fold one variable in the first round),
    // - PoW bits enabled.
    let config = make_test_config(num_variables, 1, pow_bits);
    let folding0 = config.folding_factor.at_round(0);

    // Define a multilinear polynomial:
    // f(X0, X1, X2) = 1 + 2*X2 + 3*X1 + 4*X1*X2 + 5*X0 + 6*X0*X2 + 7*X0*X1 + 8*X0*X1*X2
    let e1 = F::from_u64(1);
    let e2 = F::from_u64(2);
    let e3 = F::from_u64(3);
    let e4 = F::from_u64(4);
    let e5 = F::from_u64(5);
    let e6 = F::from_u64(6);
    let e7 = F::from_u64(7);
    let e8 = F::from_u64(8);

    let poly = EvaluationsList::new(vec![
        e1,
        e1 + e2,
        e1 + e3,
        e1 + e2 + e3 + e4,
        e1 + e5,
        e1 + e2 + e5 + e6,
        e1 + e3 + e5 + e7,
        e1 + e2 + e3 + e4 + e5 + e6 + e7 + e8,
    ]);

    // Equivalent function for evaluating the polynomial manually
    let f = |x0: EF4, x1: EF4, x2: EF4| {
        x2 * e2
            + x1 * e3
            + x1 * x2 * e4
            + x0 * e5
            + x0 * x2 * e6
            + x0 * x1 * e7
            + x0 * x1 * x2 * e8
            + e1
    };

    // Set up Fiat-Shamir domain and produce commitment + prover data
    // Generate domain separator, prover state, and Merkle commitment prover data for the poly
    let (mut proof, mut challenger_rf, prover_data) =
        setup_domain_and_commitment(&config, poly.clone());

    // Construct a statement with one evaluation constraint at the point (1, 0, 1)
    let mut statement = InitialStatement::new(poly, folding0, SumcheckStrategy::default());
    let _ = statement.evaluate(&MultilinearPoint::new(vec![EF4::ONE, EF4::ZERO, EF4::ONE]));

    // Run the first round initialization
    let state = RoundState::initialize_first_round_state(
        &mut proof.initial_sumcheck,
        &mut challenger_rf,
        &statement,
        prover_data,
        folding0,
        pow_bits,
    )
    .expect("RoundState initialization failed");

    // Unwrap the sumcheck prover and get the sampled folding randomness
    let sumcheck = &state.sumcheck_prover;
    let sumcheck_randomness = &state.folding_randomness;

    // Evaluate f at (32636, 9876, r0) and match it with the sumcheck's recovered evaluation
    let evals_f = &sumcheck.evals();
    assert_eq!(
        evals_f.evaluate_hypercube_ext::<F>(&MultilinearPoint::new(vec![
            EF4::from_u64(32636),
            EF4::from_u64(9876)
        ])),
        f(
            sumcheck_randomness[0],
            EF4::from_u64(32636),
            EF4::from_u64(9876),
        )
    );

    // Manually verify that ⟨f, w⟩ = claimed sum
    assert_eq!(sumcheck.poly.dot_product(), sumcheck.sum);

    // No Merkle tree data has been created for folded rounds yet
    assert!(state.merkle_prover_data.is_none());

    // The folding randomness must match what was sampled by the sumcheck
    assert_eq!(
        state.folding_randomness,
        MultilinearPoint::new(vec![sumcheck_randomness[0]])
    );
}
