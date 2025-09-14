use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::{SeedableRng, rngs::SmallRng};

use whir_p3::{
    dft::EvalsDft,
    fiat_shamir::domain_separator::DomainSeparator,
    parameters::{FoldingFactor, ProtocolParameters, errors::SecurityAssumption},
    poly::{coeffs::CoefficientList, evals::EvaluationsList, multilinear::MultilinearPoint},
    whir::{
        committer::{reader::CommitmentReader, writer::CommitmentWriter},
        parameters::WhirConfig,
        prover::Prover,
        statement::Statement,
        verifier::Verifier,
    },
};

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type Perm = Poseidon2BabyBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

fn main() {
    // Test with all ones polynomial
    test_whir_simple(false); // without skip
    test_whir_simple(true);  // with skip

    // Test with random polynomial
    test_whir_random(false); // without skip
    test_whir_random(true);  // with skip
}

fn test_whir_simple(use_univariate_skip: bool) {
    let num_variables = 5;
    let num_coeffs = 1 << num_variables;

    // Create hash and compression functions
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
        folding_factor: FoldingFactor::Constant(5),
        merkle_hash,
        merkle_compress,
        soundness_type: SecurityAssumption::JohnsonBound,
        starting_log_inv_rate: 1,
        univariate_skip: use_univariate_skip,
    };

    let params = WhirConfig::<EF, F, MyHash, MyCompress, MyChallenger>::new(num_variables, whir_params);

    // Define polynomial with all coefficients set to 1
    let polynomial = CoefficientList::new(vec![F::ONE; num_coeffs]).to_evaluations();

    // Sample 1 multilinear point
    let points: Vec<_> = vec![
        MultilinearPoint::new((0..num_variables).map(|i| EF::from_u64(i as u64)).collect())
    ];

    // Construct statement
    let mut statement = Statement::<EF>::new(num_variables);
    for point in &points {
        let eval = polynomial.evaluate(point);
        statement.add_constraint(point.clone(), eval);
    }

    // Run WHIR protocol
    run_whir(&params, polynomial, statement, use_univariate_skip);
}

fn test_whir_random(use_univariate_skip: bool) {
    let num_variables = 5;
    let num_coeffs = 1 << num_variables;

    // Create hash and compression functions
    let mut rng = SmallRng::seed_from_u64(42);
    let perm = Perm::new_from_rng_128(&mut rng);
    let merkle_hash = MyHash::new(perm.clone());
    let merkle_compress = MyCompress::new(perm);

    // Construct WHIR protocol parameters
    let whir_params = ProtocolParameters {
        initial_statement: true,
        security_level: 32,
        pow_bits: 0,
        rs_domain_initial_reduction_factor: 1,
        folding_factor: FoldingFactor::Constant(5),
        merkle_hash,
        merkle_compress,
        soundness_type: SecurityAssumption::JohnsonBound,
        starting_log_inv_rate: 1,
        univariate_skip: use_univariate_skip,
    };

    let params = WhirConfig::<EF, F, MyHash, MyCompress, MyChallenger>::new(num_variables, whir_params);

    // Define polynomial with random coefficients
    use rand::Rng;
    let mut rng = SmallRng::seed_from_u64(42);
    let polynomial: EvaluationsList<F> =
        EvaluationsList::new((0..num_coeffs).map(|_| rng.random()).collect());

    // Sample 1 multilinear point
    let points: Vec<_> = vec![
        MultilinearPoint::new((0..num_variables).map(|i| EF::from_u64(i as u64)).collect())
    ];

    // Construct statement
    let mut statement = Statement::<EF>::new(num_variables);
    for point in &points {
        let eval = polynomial.evaluate(point);
        statement.add_constraint(point.clone(), eval);
    }

    // Run WHIR protocol
    run_whir(&params, polynomial, statement, use_univariate_skip);
}

fn run_whir(
    params: &WhirConfig<EF, F, MyHash, MyCompress, MyChallenger>,
    polynomial: EvaluationsList<F>,
    statement: Statement<EF>,
    use_univariate_skip: bool
) {
    println!("Running WHIR with univariate_skip = {}", use_univariate_skip);

    // Define domain separator
    let mut domainsep = DomainSeparator::new(vec![]);
    domainsep.commit_statement::<_, _, _, 32>(params);
    domainsep.add_whir_proof::<_, _, _, 32>(params);

    let mut rng = SmallRng::seed_from_u64(1);
    let challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));

    // Initialize prover state
    let mut prover_state = domainsep.to_prover_state(challenger.clone());

    // Commit to polynomial
    let committer = CommitmentWriter::new(params);
    let dft_committer = EvalsDft::<F>::default();
    let witness = committer
        .commit(&dft_committer, &mut prover_state, polynomial)
        .unwrap();

    let prover = Prover(params);
    let dft_prover = EvalsDft::<F>::default();

    // Generate proof
    prover
        .prove(&dft_prover, &mut prover_state, statement.clone(), witness)
        .unwrap();

    let checkpoint_prover: EF = prover_state.sample();

    // Verify proof
    let commitment_reader = CommitmentReader::new(params);
    let verifier = Verifier::new(params);

    let mut verifier_state =
        domainsep.to_verifier_state(prover_state.proof_data().to_vec(), challenger);

    let parsed_commitment = commitment_reader
        .parse_commitment::<8>(&mut verifier_state)
        .unwrap();

    verifier
        .verify(&mut verifier_state, &parsed_commitment, &statement)
        .unwrap();

    let checkpoint_verifier: EF = verifier_state.sample();
    assert_eq!(checkpoint_prover, checkpoint_verifier);

    println!("WHIR test completed successfully!");
}