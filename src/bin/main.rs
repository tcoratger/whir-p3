use std::time::Instant;

use clap::Parser;
use p3_baby_bear::BabyBear;
use p3_challenger::DuplexChallenger;
use p3_field::{Field, extension::BinomialExtensionField};
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_koala_bear::KoalaBear;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::{
    Rng, SeedableRng,
    rngs::{SmallRng, StdRng},
};
use tracing_forest::{ForestLayer, util::LevelFilter};
use tracing_subscriber::{EnvFilter, Registry, layer::SubscriberExt, util::SubscriberInitExt};
use whir_p3::{
    dft::EvalsDft,
    fiat_shamir::domain_separator::DomainSeparator,
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
type Perm = Poseidon2Goldilocks<16>;

type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

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
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);

    let merkle_hash = MyHash::new(perm.clone());
    let merkle_compress = MyCompress::new(perm);

    let rs_domain_initial_reduction_factor = args.rs_domain_initial_reduction_factor;

    let num_coeffs = 1 << num_variables;

    let mv_params = MultivariateParameters::<EF>::new(num_variables);

    // Construct WHIR protocol parameters
    let whir_params = ProtocolParameters {
        initial_statement: true,
        security_level,
        pow_bits,
        folding_factor,
        merkle_hash,
        merkle_compress,
        soundness_type,
        starting_log_inv_rate: starting_rate,
        rs_domain_initial_reduction_factor,
        univariate_skip: false,
    };

    let params = WhirConfig::<EF, F, MyHash, MyCompress, MyChallenger>::new(mv_params, whir_params);

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
    let mut domainsep = DomainSeparator::new(vec![]);
    domainsep.commit_statement::<_, _, _, 32>(&params);
    domainsep.add_whir_proof::<_, _, _, 32>(&params);

    println!("=========================================");
    println!("Whir (PCS) 🌪️");
    if !params.check_pow_bits() {
        println!("WARN: more PoW bits required than what specified.");
    }

    let mut rng = SmallRng::seed_from_u64(1);
    let challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));

    // Initialize the Merlin transcript from the IOPattern
    let mut prover_state = domainsep.to_prover_state(challenger.clone());

    // Commit to the polynomial and produce a witness
    let committer = CommitmentWriter::new(&params);

    let dft = EvalsDft::<F>::new(1 << params.max_fft_size());

    let time = Instant::now();
    let witness = committer
        .commit::<<F as Field>::Packing, <F as Field>::Packing, 8>(
            &dft,
            &mut prover_state,
            polynomial,
        )
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

    // Reconstruct verifier's view of the transcript using the DomainSeparator and prover's data
    let mut verifier_state =
        domainsep.to_verifier_state(prover_state.proof_data.clone(), challenger);

    // Parse the commitment
    let parsed_commitment = commitment_reader
        .parse_commitment::<8>(&mut verifier_state)
        .unwrap();

    let verif_time = Instant::now();
    verifier
        .verify(&mut verifier_state, &parsed_commitment, &statement)
        .unwrap();
    let verify_time = verif_time.elapsed();

    println!(
        "\nProving time: {} ms (commit: {} ms, opening: {} ms)",
        commit_time.as_millis() + opening_time.as_millis(),
        commit_time.as_millis(),
        opening_time.as_millis()
    );
    println!("Verification time: {} μs", verify_time.as_micros());
}
