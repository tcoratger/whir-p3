use core::fmt::Debug;
use std::time::Instant;

use rand::distr::StandardUniform;
use rand::prelude::Distribution;

use clap::Parser;
use p3_baby_bear::BabyBear;
use p3_challenger::{DuplexChallenger, FieldChallenger};
use p3_commit::{BatchOpening, ExtensionMmcs, Pcs, PolynomialSpace};
use p3_dft::{Radix2DFTSmallBatch, Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{
    extension::BinomialExtensionField, ExtensionField, Field, PrimeField32, PrimeField64,
    TwoAdicField,
};
use p3_fri::{FriParameters, FriProof, TwoAdicFriPcs};
use p3_goldilocks::Goldilocks;
use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CryptographicPermutation, PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::{
    prove, verify, DebugConstraintBuilder, ProverConstraintFolder, StarkConfig, StarkGenericConfig,
    SymbolicAirBuilder, VerifierConstraintFolder,
};
use rand::{
    rngs::{SmallRng, StdRng},
    Rng, SeedableRng,
};
use tracing_forest::{util::LevelFilter, ForestLayer};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Registry};
use whir_p3::{
    fiat_shamir::domain_separator::DomainSeparator,
    parameters::{errors::SecurityAssumption, FoldingFactor, ProtocolParameters, DEFAULT_MAX_POW},
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    whir::{
        committer::{reader::CommitmentReader, writer::CommitmentWriter},
        constraints::statement::EqStatement,
        parameters::{SumcheckOptimization, WhirConfig},
        proof::WhirProof,
        prover::Prover,
        verifier::Verifier,
    },
};

type F = KoalaBear;
type EF = BinomialExtensionField<F, 4>;
type _F = BabyBear;
type _EF = BinomialExtensionField<_F, 5>;
type __F = Goldilocks;
type __EF = BinomialExtensionField<__F, 2>;

type Poseidon16 = Poseidon2KoalaBear<16>;
type Poseidon24 = Poseidon2KoalaBear<24>;

type MerkleHash = PaddingFreeSponge<Poseidon24, 24, 16, 8>; // leaf hashing
type MerkleCompress = TruncatedPermutation<Poseidon16, 2, 8, 16>; // 2-to-1 compression
type MyChallenger = DuplexChallenger<F, Poseidon16, 16, 8>;
type DFT = Radix2DFTSmallBatch<F>;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short = 'l', long, default_value = "90")]
    security_level: usize,

    #[arg(short = 'p', long)]
    pow_bits: Option<usize>,

    #[arg(short = 'd', long, default_value = "25")]
    num_variables: usize,

    #[arg(short = 'e', long = "evaluations", default_value = "1")]
    num_evaluations: usize,

    #[arg(short = 'r', long, default_value = "1")]
    rate: usize,

    #[arg(short = 'k', long = "fold", default_value = "5")]
    folding_factor: usize,

    #[arg(long = "sec", default_value = "CapacityBound")]
    soundness_type: SecurityAssumption,

    #[arg(long = "initial-rs-reduction", default_value = "3")]
    rs_domain_initial_reduction_factor: usize,
}

// Types related to using Poseidon2 in the Merkle tree.
pub(crate) type Poseidon2Sponge<Perm24> = PaddingFreeSponge<Perm24, 24, 16, 8>;
pub(crate) type Poseidon2Compression<Perm16> = TruncatedPermutation<Perm16, 2, 8, 16>;
pub(crate) type Poseidon2MerkleMmcs<F, Perm16, Perm24> = MerkleTreeMmcs<
    <F as Field>::Packing,
    <F as Field>::Packing,
    Poseidon2Sponge<Perm24>,
    Poseidon2Compression<Perm16>,
    8,
>;

/// General context handling that stores things we need in WHIR as well as FRI
struct Context {
    rng: StdRng,
    polynomial: EvaluationsList<F>,
    poseidon16: Poseidon16,
    poseidon24: Poseidon24,
    num_coeffs: usize,
    num_evaluations: usize,
    challenger: MyChallenger,
}

/// Initialize the `Context` object storing things used in both WHIR and FRI
fn init_context() -> Context {
    let mut args = Args::parse(); // we parse again in `prepare_config`, but well..
    let num_coeffs = 1 << args.num_variables;
    let num_evaluations = args.num_evaluations;

    let mut rng = StdRng::seed_from_u64(0);
    let polynomial = EvaluationsList::<F>::new((0..num_coeffs).map(|_| rng.random()).collect());
    let poseidon16 = Poseidon16::new_from_rng_128(&mut rng);
    let poseidon24 = Poseidon24::new_from_rng_128(&mut rng);

    // IMPORTANT: We obviously need to *clone* this challenger for every prove / verify call,
    // otherwise transcript state would persist
    let challenger = MyChallenger::new(poseidon16.clone());

    Context {
        rng,
        polynomial,
        poseidon16,
        poseidon24,
        num_coeffs,
        num_evaluations,
        challenger,
    }
}

/// Prepare the `WhirConfig` used for WHIR and a few fields in FRI
fn prepare_config(ctx: &Context) -> WhirConfig<EF, F, MerkleHash, MerkleCompress, MyChallenger> {
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    let _ = Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .try_init();

    let mut args = Args::parse();

    if args.pow_bits.is_none() {
        args.pow_bits = Some(DEFAULT_MAX_POW);
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

    let merkle_hash = MerkleHash::new(ctx.poseidon24.clone());
    let merkle_compress = MerkleCompress::new(ctx.poseidon16.clone());

    let rs_domain_initial_reduction_factor = args.rs_domain_initial_reduction_factor;

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
        sumcheck_optimization: SumcheckOptimization::Classic,
    };

    let params = WhirConfig::<EF, F, MerkleHash, MerkleCompress, MyChallenger>::new(
        num_variables,
        whir_params,
    );

    params
}

fn run_whir(ctx: &mut Context) {
    let args = Args::parse();

    let params = prepare_config(ctx);
    let dft = Radix2DFTSmallBatch::<F>::new(1 << params.max_fft_size());

    // Sample `num_points` random multilinear points in the Boolean hypercube
    let points: Vec<_> = (0..ctx.num_evaluations)
        .map(|_| MultilinearPoint::rand(&mut ctx.rng, params.num_variables))
        .collect();

    // Construct a new statement with the correct number of variables
    let mut statement = EqStatement::<EF>::initialize(params.num_variables);

    // Add constraints for each sampled point (equality constraints)
    for point in &points {
        statement.add_unevaluated_constraint_hypercube(point.clone(), &ctx.polynomial);
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

    // Initialize the Merlin transcript from the IOPattern
    let mut prover_state = domainsep.to_prover_state(ctx.challenger.clone());

    // Commit to the polynomial and produce a witness
    let committer = CommitmentWriter::new(&params);

    let dft = Radix2DFTSmallBatch::<F>::new(1 << params.max_fft_size());

    let mut proof = WhirProof::<F, EF, 8>::default();

    let time = Instant::now();
    let witness = committer
        .commit(
            &dft,
            &mut prover_state,
            &mut proof,
            &mut ctx.challenger.clone(),
            ctx.polynomial.clone(),
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
        domainsep.to_verifier_state(prover_state.proof_data().to_vec(), ctx.challenger.clone());

    let mut verifier_challenger = ctx.challenger.clone();
    // Parse the commitment
    let parsed_commitment = commitment_reader
        .parse_commitment::<8>(&mut verifier_state, &proof, &mut verifier_challenger)
        .unwrap();

    let verif_time = Instant::now();
    verifier
        .verify(
            &mut verifier_state,
            &parsed_commitment,
            statement,
            &proof,
            &mut verifier_challenger,
        )
        .unwrap();
    let verify_time = verif_time.elapsed();

    println!(
        "\nProving time: {} ms (commit: {} ms, opening: {} ms)",
        commit_time.as_millis() + opening_time.as_millis(),
        commit_time.as_millis(),
        opening_time.as_millis()
    );
    let proof_size = prover_state.proof_data().len() as f64 * (F::ORDER_U64 as f64).log2() / 8.0;
    println!("proof size: {:.2} KiB", proof_size / 1024.0);
    println!("Verification time: {} μs", verify_time.as_micros());
}

/// Creates a set of `FriParameters` suitable for benchmarking.
/// These parameters represents numbers used in Valida
pub const fn create_benchmark_fri_params<Mmcs>(mmcs: Mmcs) -> FriParameters<Mmcs> {
    FriParameters {
        log_blowup: 1,
        log_final_poly_len: 0,
        num_queries: 40,
        proof_of_work_bits: 8,
        mmcs,
    }
}

/// Report the result of the proof.
///
/// Either print that the proof was successful or panic and return the error.
#[inline]
pub fn report_result(result: Result<(), impl Debug>) {
    if let Err(e) = result {
        panic!("{e:?}");
    } else {
        println!("Proof Verified Successfully")
    }
}

/// Returns the size of the FRI proof in bytes
fn calc_fri_proof_size(
    opened_values: Vec<Vec<Vec<Vec<EF>>>>,
    proof: FriProof<
        EF,
        ExtensionMmcs<F, EF, Poseidon2MerkleMmcs<F, Poseidon16, Poseidon24>>,
        F,
        Vec<BatchOpening<F, Poseidon2MerkleMmcs<F, Poseidon16, Poseidon24>>>,
    >,
) -> usize {
    let opening_bytes = bincode::serialize(&opened_values).expect("serialize openings");
    let proof_bytes = bincode::serialize(&proof).expect("serialize proof");
    opening_bytes.len() + proof_bytes.len()
}

fn run_fri(ctx: &mut Context) {
    // WHIR setup, to reuse parameters for FRI (that are applicable)
    let params = prepare_config(ctx);

    // TODO: The DFT size might be different for FRI and WHIR, no?
    // Comment on WhirConfig for max_fft_size says:
    // /// Returns the log2 size of the largest FFT
    // /// (At commitment we perform 2^folding_factor FFT of size 2^max_fft_size)
    // but folding factor will be different?
    let dft = Radix2DFTSmallBatch::<F>::new(1 << params.max_fft_size());
    // Set up MMCS and TwoAdicFriPcs
    let val_mmcs = Poseidon2MerkleMmcs::<F, Poseidon16, Poseidon24>::new(
        params.merkle_hash.clone(),
        params.merkle_compress.clone(),
    );
    let challenge_mmcs = ExtensionMmcs::<F, EF, _>::new(val_mmcs.clone());
    let fri_params = create_benchmark_fri_params(challenge_mmcs);
    let pcs = TwoAdicFriPcs::<F, DFT, _, _>::new(dft, val_mmcs, fri_params);

    println!("\n\n=========================================");
    println!("FRI (PCS) 🍳️");

    let log_height = params.num_variables;
    let trace_height = 1 << log_height;

    // Define the number of columns we split the evaluations into
    // TODO: could make this a CL arg?
    const LOG_NUM_COLS: usize = 5;
    const NUM_COLS: usize = 1 << LOG_NUM_COLS; // 32

    // Construct a domain of the required size based on the height of the "trace". We split the evaluations
    // `Context::polynomial` into a `trace_height x NUM_COLS` `RowMajorMatrix`.
    // NOTE: In KoalaBear the F::TWO_ADICITY is 24. So the height can at most be 2^{23}.
    let domain = TwoAdicMultiplicativeCoset::new(F::GENERATOR, log_height - LOG_NUM_COLS)
        .expect("log height too large");

    // Convert the polynomial evaluations into a RowMajorMatrix, as used by FRI with `NUM_COLS` columns.
    // We need an iterator for the Pcs::commit, so wrap with `once`
    let matrix_iter = std::iter::once((
        domain,
        RowMajorMatrix::new(ctx.polynomial.as_slice().to_vec(), NUM_COLS),
    ));

    // Commit to the matrix
    let commit_time = Instant::now();
    let (commitment, prover_data) = Pcs::<EF, MyChallenger>::commit(&pcs, matrix_iter.clone());
    let commitment_time = commit_time.elapsed();

    // Randomly sample the correct number of opening points
    let open_points: Vec<EF> = (0..ctx.num_evaluations)
        .map(|_| ctx.rng.random::<EF>())
        .collect();
    let num_chunks = ctx.polynomial.num_evals() / trace_height;
    let points = vec![open_points.clone(); num_chunks];
    // Generate the opening proof
    let open_time = Instant::now();
    let (opened_values, proof) =
        pcs.open(vec![(&prover_data, points)], &mut ctx.challenger.clone());
    let opening_time = open_time.elapsed();

    // Construct the points needed for the verifier
    let verifier_points = matrix_iter
        .zip(&opened_values[0]) // first and only commitment
        .map(|((domain, _), mat_openings)| {
            let openings = open_points
                .iter()
                .copied()
                .zip(mat_openings.iter().cloned())
                .collect();
            (domain, openings)
        })
        .collect();

    // Verify the opening proof
    let verif_time = Instant::now();
    let res = pcs.verify(
        vec![(commitment, verifier_points)],
        &proof,
        &mut ctx.challenger.clone(),
    );
    let verify_time = verif_time.elapsed();

    report_result(res);

    println!(
        "\nProving time: {} ms (commit: {} ms, opening: {} ms)",
        commitment_time.as_millis() + opening_time.as_millis(),
        commitment_time.as_millis(),
        opening_time.as_millis()
    );
    let proof_size = calc_fri_proof_size(opened_values, proof) as f64;
    println!("proof size: {:.2} KiB", proof_size / 1024.0);
    println!("Verification time: {} μs", verify_time.as_micros());
}

fn main() {
    let mut ctx = init_context();

    // 1. First run WHIR
    run_whir(&mut ctx);

    // 2. Now run FRI
    run_fri(&mut ctx);
}
