use clap::Parser;
use p3_baby_bear::BabyBear;
use p3_blake3::Blake3;
use p3_dft::Radix2DitParallel;
use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};
use p3_goldilocks::Goldilocks;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use tracing_forest::{ForestLayer, util::LevelFilter};
use tracing_subscriber::{EnvFilter, Registry, layer::SubscriberExt, util::SubscriberInitExt};
use whir_p3::{
    fiat_shamir::{domain_separator::DomainSeparator, pow::blake3::Blake3PoW},
    parameters::{
        FoldingFactor, MultivariateParameters, ProtocolParameters, default_max_pow,
        errors::SecurityAssumption,
    },
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    whir::{
        committer::{reader::CommitmentReader, writer::CommitmentWriter},
        parameters::WhirConfig,
        prover::Prover,
        statement::{Statement, weights::Weights},
        verifier::Verifier,
    },
};

type F = Goldilocks;
type EF = BinomialExtensionField<F, 2>;
type _F = BabyBear;
type _EF = BinomialExtensionField<_F, 5>;
type ByteHash = Blake3;
type FieldHash = SerializingHasher<ByteHash>;
type MyCompress = CompressionFunctionFromHasher<ByteHash, 2, 32>;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short = 'l', long, default_value = "100")]
    security_level: usize,

    #[arg(short = 'p', long)]
    pow_bits: Option<usize>,

    #[arg(short = 'd', long, default_value = "24")]
    num_variables: usize,

    #[arg(short = 'e', long = "evaluations", default_value = "1")]
    num_evaluations: usize,

    #[arg(short = 'r', long, default_value = "1")]
    rate: usize,

    #[arg(short = 'k', long = "fold", default_value = "4")]
    folding_factor: usize,

    #[arg(long = "sec", default_value = "CapacityBound")]
    soundness_type: SecurityAssumption,
}

fn main() {
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .init();

    let mut args = Args::parse();

    if args.pow_bits.is_none() {
        args.pow_bits = Some(default_max_pow(args.num_variables, args.rate));
    }

    // Runs as a PCS
    let security_level = args.security_level;
    let pow_bits = args.pow_bits.unwrap();
    let num_variables = args.num_variables;
    let starting_rate = args.rate;
    let folding_factor = FoldingFactor::Constant(args.folding_factor);
    let soundness_type = args.soundness_type;
    let num_evaluations = args.num_evaluations;

    if num_evaluations == 0 {
        println!("Warning: running as PCS but no evaluations specified.");
    }

    // Create hash and compression functions for the Merkle tree
    let byte_hash = ByteHash {};
    let merkle_hash = FieldHash::new(byte_hash);
    let merkle_compress = MyCompress::new(byte_hash);

    let num_coeffs = 1 << num_variables;

    let mv_params = MultivariateParameters::<EF>::new(num_variables);

    // Construct WHIR protocol parameters
    let whir_params = ProtocolParameters::<_, _> {
        initial_statement: true,
        security_level,
        pow_bits,
        folding_factor,
        merkle_hash,
        merkle_compress,
        soundness_type,
        starting_log_inv_rate: starting_rate,
    };

    let params = WhirConfig::<EF, F, FieldHash, MyCompress, Blake3PoW>::new(mv_params, whir_params);

    dbg!(&params);

    // Define a polynomial with all coefficients set to 1 (i.e., constant 1 polynomial)
    let polynomial = EvaluationsList::new((0..num_coeffs).map(F::from_u64).collect());

    // Sample `num_points` random multilinear points in the Boolean hypercube
    let points: Vec<_> = (0..num_evaluations)
        .map(|_| MultilinearPoint((0..num_variables).map(|i| EF::from_u64(i as u64)).collect()))
        .collect();

    // Construct a new statement with the correct number of variables
    let mut statement = Statement::<EF>::new(num_variables);

    // Add constraints for each sampled point (equality constraints)
    for point in &points {
        let eval = polynomial.evaluate_at_extension(point);
        let weights = Weights::evaluation(point.clone());
        statement.add_constraint(weights, eval);
    }

    // Define the Fiat-Shamir domain separator pattern for committing and proving
    let mut domainsep = DomainSeparator::new("🌪️");
    domainsep.commit_statement(&params);
    domainsep.add_whir_proof(&params);

    println!("=========================================");
    println!("Whir (PCS) 🌪️");
    if !params.check_pow_bits() {
        println!("WARN: more PoW bits required than what specified.");
    }

    // Initialize the Merlin transcript from the IOPattern
    let mut prover_state = domainsep.to_prover_state();

    // Commit to the polynomial and produce a witness
    let committer = CommitmentWriter::new(&params);

    let dft_committer = Radix2DitParallel::<F>::default();

    let witness = committer
        .commit(&dft_committer, &mut prover_state, polynomial)
        .unwrap();

    // Generate a proof using the prover
    let prover = Prover(&params);

    let dft_prover = Radix2DitParallel::<F>::default();

    // Generate a STARK proof for the given statement and witness
    prover
        .prove(&dft_prover, &mut prover_state, statement.clone(), witness)
        .unwrap();

    // Create a commitment reader
    let commitment_reader = CommitmentReader::new(&params);

    // Create a verifier with matching parameters
    let verifier = Verifier::new(&params);

    let narg_string = prover_state.narg_string().to_vec();
    let proof_size = narg_string.len();

    // Reconstruct verifier's view of the transcript using the DomainSeparator and prover's data
    let mut verifier_state = domainsep.to_verifier_state(&narg_string);

    // Parse the commitment
    let parsed_commitment = commitment_reader
        .parse_commitment::<32>(&mut verifier_state)
        .unwrap();

    verifier
        .verify(&mut verifier_state, &parsed_commitment, &statement)
        .unwrap();

    println!("Proof size: {:.1} KiB", proof_size as f64 / 1024.0);
}
