use std::time::Instant;

use clap::Parser;
use p3_baby_bear::BabyBear;
use p3_blake3::Blake3;
use p3_dft::Radix2DitSmallBatch;
use p3_field::extension::BinomialExtensionField;
use p3_goldilocks::Goldilocks;
use p3_keccak::KeccakF;
use p3_koala_bear::KoalaBear;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use rand::{Rng, SeedableRng, rngs::StdRng};
use tracing_forest::{ForestLayer, util::LevelFilter};
use tracing_subscriber::{EnvFilter, Registry, layer::SubscriberExt, util::SubscriberInitExt};
use whir_p3::{
    fiat_shamir::{
        DefaultHash, DefaultPerm, domain_separator::DomainSeparator, keccak::KECCAK_WIDTH_BYTES,
        pow::blake3::Blake3PoW,
    },
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
type __F = KoalaBear;
type __EF = BinomialExtensionField<__F, 4>;
type ByteHash = Blake3;
type FieldHash = SerializingHasher<ByteHash>;
type MyCompress = CompressionFunctionFromHasher<ByteHash, 2, 32>;
type Perm = DefaultPerm;
type FiatShamirHash = DefaultHash;
type W = u8;
const PERM_WIDTH: usize = KECCAK_WIDTH_BYTES;

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

    #[arg(long = "initial-rs-reduction", default_value = "3")]
    rs_domain_initial_reduction_factor: usize,
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
    let rs_domain_initial_reduction_factor = args.rs_domain_initial_reduction_factor;

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
        rs_domain_initial_reduction_factor,
    };

    let params = WhirConfig::<
        EF,
        F,
        FieldHash,
        MyCompress,
        Blake3PoW,
        Perm,
        FiatShamirHash,
        W,
        PERM_WIDTH,
    >::new(mv_params, whir_params);

    dbg!(&params);

    let mut rng = StdRng::seed_from_u64(0);
    let polynomial = EvaluationsList::<F>::new((0..num_coeffs).map(|_| rng.random()).collect());

    // Sample `num_points` random multilinear points in the Boolean hypercube
    let points: Vec<_> = (0..num_evaluations)
        .map(|_| MultilinearPoint::rand(&mut rng, num_variables))
        .collect();

    // Construct a new statement with the correct number of variables
    let mut statement = Statement::<EF>::new(num_variables);

    // Add constraints for each sampled point (equality constraints)
    for point in &points {
        let eval = polynomial.evaluate(point);
        let weights = Weights::evaluation(point.clone());
        statement.add_constraint(weights, eval);
    }

    // Define the Fiat-Shamir domain separator pattern for committing and proving
    let mut domainsep = DomainSeparator::new("🌪️", KeccakF);
    domainsep.commit_statement(&params);
    domainsep.add_whir_proof(&params);

    println!("=========================================");
    println!("Whir (PCS) 🌪️");
    if !params.check_pow_bits() {
        println!("WARN: more PoW bits required than what specified.");
    }

    // Initialize the Merlin transcript from the IOPattern
    let mut prover_state = domainsep.to_prover_state::<_, 32>();

    // Commit to the polynomial and produce a witness
    let committer = CommitmentWriter::new(&params);

    let dft = Radix2DitSmallBatch::<F>::new(1 << params.max_fft_size());

    let time = Instant::now();
    let witness = committer
        .commit(&dft, &mut prover_state, polynomial)
        .unwrap();
    let commit_time = time.elapsed();

    // Generate a proof using the prover
    let prover = Prover(&params);

    // Generate a proof for the given statement and witness
    let time = Instant::now();
    prover
        .prove(&dft, &mut prover_state, statement.clone(), witness)
        .unwrap();
    let opening_time = time.elapsed();

    // Create a commitment reader
    let commitment_reader = CommitmentReader::new(&params);

    // Create a verifier with matching parameters
    let verifier = Verifier::new(&params);

    let narg_string = prover_state.narg_string().to_vec();
    let proof_size = narg_string.len();

    // Reconstruct verifier's view of the transcript using the DomainSeparator and prover's data
    let mut verifier_state = domainsep.to_verifier_state::<_, 32>(&narg_string);

    // Parse the commitment
    let parsed_commitment = commitment_reader
        .parse_commitment::<32>(&mut verifier_state)
        .unwrap();

    let verif_time = Instant::now();
    verifier
        .verify(&mut verifier_state, &parsed_commitment, &statement)
        .unwrap();
    let verify_time = verif_time.elapsed();

    println!("\nCommitment time: {} ms", commit_time.as_millis());
    println!("Opening time: {} ms", opening_time.as_millis());
    println!("Proof size: {:.1} KiB", proof_size as f64 / 1024.0);
    println!("Verification time: {} μs", verify_time.as_micros());
}
