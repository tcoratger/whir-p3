use core::fmt::Debug;
use std::time::Instant;

use rand::distr::StandardUniform;
use rand::prelude::Distribution;

use clap::Parser;
use p3_air::{Air, BaseAir};
use p3_baby_bear::BabyBear;
use p3_challenger::{DuplexChallenger, FieldChallenger};
use p3_commit::{ExtensionMmcs, PolynomialSpace};
use p3_dft::{Radix2DFTSmallBatch, Radix2DitParallel, TwoAdicSubgroupDft};
use p3_examples::airs::{ExampleHashAir, ProofObjective};
use p3_field::{
    extension::BinomialExtensionField, ExtensionField, Field, PrimeField32, PrimeField64,
    TwoAdicField,
};
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_goldilocks::Goldilocks;
use p3_koala_bear::{GenericPoseidon2LinearLayersKoalaBear, KoalaBear, Poseidon2KoalaBear};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_poseidon2::GenericPoseidon2LinearLayers;
use p3_poseidon2_air::{RoundConstants, VectorizedPoseidon2Air};
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
pub(crate) type Poseidon2StarkConfig<F, EF, DFT, Perm16, Perm24> = StarkConfig<
    TwoAdicFriPcs<
        F,
        DFT,
        Poseidon2MerkleMmcs<F, Perm16, Perm24>,
        ExtensionMmcs<F, EF, Poseidon2MerkleMmcs<F, Perm16, Perm24>>,
    >,
    EF,
    DuplexChallenger<F, Perm24, 24, 16>,
>;

// /// An AIR for a hash function used for example proofs and benchmarking.
// ///
// /// A key feature is the ability to randomly generate a trace which proves
// /// the output of some number of hashes using a given hash function.
// pub trait ExampleHashAir<F: Field, SC: StarkGenericConfig>:
//     BaseAir<F>
//     + for<'a> Air<DebugConstraintBuilder<'a, F>>
//     + Air<SymbolicAirBuilder<F>>
//     + for<'a> Air<ProverConstraintFolder<'a, SC>>
//     + for<'a> Air<VerifierConstraintFolder<'a, SC>>
// {
//     fn generate_trace_rows(
//         &self,
//         num_hashes: usize,
//         extra_capacity_bits: usize,
//     ) -> RowMajorMatrix<F>
//     where
//         StandardUniform: Distribution<F>;
// }
//
//impl<
//        F: PrimeField64,
//        Domain: PolynomialSpace<Val = F>,
//        EF: ExtensionField<F>,
//        Challenger: FieldChallenger<F>,
//        Pcs: p3_commit::Pcs<EF, Challenger, Domain = Domain>,
//        SC: StarkGenericConfig<Pcs = Pcs, Challenge = EF, Challenger = Challenger>,
//        LinearLayers: GenericPoseidon2LinearLayers<WIDTH>,
//        const WIDTH: usize,
//        const SBOX_DEGREE: u64,
//        const SBOX_REGISTERS: usize,
//        const HALF_FULL_ROUNDS: usize,
//        const PARTIAL_ROUNDS: usize,
//        const VECTOR_LEN: usize,
//    > ExampleHashAir<F, SC>
//    for VectorizedPoseidon2Air<
//        F,
//        LinearLayers,
//        WIDTH,
//        SBOX_DEGREE,
//        SBOX_REGISTERS,
//        HALF_FULL_ROUNDS,
//        PARTIAL_ROUNDS,
//        VECTOR_LEN,
//    >
//{
//    #[inline]
//    fn generate_trace_rows(
//        &self,
//        num_hashes: usize,
//        extra_capacity_bits: usize,
//    ) -> RowMajorMatrix<F>
//    where
//        StandardUniform: Distribution<F>,
//    {
//        self.generate_vectorized_trace_rows(num_hashes, extra_capacity_bits)
//    }
//}
//
fn prepare_params() -> WhirConfig<EF, F, MerkleHash, MerkleCompress, MyChallenger> {
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
    let poseidon16 = Poseidon16::new_from_rng_128(&mut rng);
    let poseidon24 = Poseidon24::new_from_rng_128(&mut rng);

    let merkle_hash = MerkleHash::new(poseidon24);
    let merkle_compress = MerkleCompress::new(poseidon16.clone());

    let rs_domain_initial_reduction_factor = args.rs_domain_initial_reduction_factor;

    let num_coeffs = 1 << num_variables;

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

fn run_whir() {
    let args = Args::parse();

    let num_variables = args.num_variables;
    let num_evaluations = args.num_evaluations;
    let num_coeffs = 1 << num_variables;
    let params = prepare_params();

    let mut rng = StdRng::seed_from_u64(0);
    let polynomial = EvaluationsList::<F>::new((0..num_coeffs).map(|_| rng.random()).collect());

    // Sample `num_points` random multilinear points in the Boolean hypercube
    let points: Vec<_> = (0..num_evaluations)
        .map(|_| MultilinearPoint::rand(&mut rng, num_variables))
        .collect();

    // Construct a new statement with the correct number of variables
    let mut statement = EqStatement::<EF>::initialize(num_variables);

    // Add constraints for each sampled point (equality constraints)
    for point in &points {
        statement.add_unevaluated_constraint_hypercube(point.clone(), &polynomial);
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

    let poseidon16 = Poseidon16::new_from_rng_128(&mut rng);
    let challenger = MyChallenger::new(poseidon16);
    let mut prover_challenger = challenger.clone();

    // Initialize the Merlin transcript from the IOPattern
    let mut prover_state = domainsep.to_prover_state(challenger.clone());

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
            &mut prover_challenger,
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
    let mut verifier_challenger = challenger.clone();
    let mut verifier_state =
        domainsep.to_verifier_state(prover_state.proof_data().to_vec(), challenger);

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

/// Produce a MerkleTreeMmcs from a pair of cryptographic field permutations.
///
/// The first permutation will be used for compression and the second for more sponge hashing.
/// Currently this is only intended to be used with a pair of Poseidon2 hashes of with 16 and 24
/// but this can easily be generalised in future if we desire.
const fn get_poseidon2_mmcs<
    F: Field,
    Perm16: CryptographicPermutation<[F; 16]> + CryptographicPermutation<[F::Packing; 16]>,
    Perm24: CryptographicPermutation<[F; 24]> + CryptographicPermutation<[F::Packing; 24]>,
>(
    perm16: Perm16,
    perm24: Perm24,
) -> Poseidon2MerkleMmcs<F, Perm16, Perm24> {
    let hash = Poseidon2Sponge::new(perm24);

    let compress = Poseidon2Compression::new(perm16);

    Poseidon2MerkleMmcs::<F, _, _>::new(hash, compress)
}

/// Creates a set of `FriParameters` suitable for benchmarking.
/// These parameters represent typical settings used in production-like scenarios.
pub const fn create_benchmark_fri_params<Mmcs>(mmcs: Mmcs) -> FriParameters<Mmcs> {
    FriParameters {
        log_blowup: 1,
        log_final_poly_len: 0,
        num_queries: 100,
        proof_of_work_bits: 16,
        mmcs,
    }
}

pub fn prove_monty31_poseidon2<
    F: PrimeField32 + TwoAdicField,
    EF: ExtensionField<F>,
    DFT: TwoAdicSubgroupDft<F>,
    Perm16: CryptographicPermutation<[F; 16]> + CryptographicPermutation<[F::Packing; 16]>,
    Perm24: CryptographicPermutation<[F; 24]> + CryptographicPermutation<[F::Packing; 24]>,
    PG: ExampleHashAir<F, Poseidon2StarkConfig<F, EF, DFT, Perm16, Perm24>>,
>(
    proof_goal: PG,
    dft: DFT,
    num_hashes: usize,
    perm16: Perm16,
    perm24: Perm24,
) -> Result<(), impl Debug>
where
    StandardUniform: Distribution<F>,
{
    let val_mmcs = get_poseidon2_mmcs::<F, _, _>(perm16, perm24.clone());

    let challenge_mmcs = ExtensionMmcs::<F, EF, _>::new(val_mmcs.clone());
    let fri_params = create_benchmark_fri_params(challenge_mmcs);

    let trace = proof_goal.generate_trace_rows(num_hashes, fri_params.log_blowup);

    let pcs = TwoAdicFriPcs::new(dft, val_mmcs, fri_params);
    let challenger = DuplexChallenger::new(perm24);

    let config = Poseidon2StarkConfig::new(pcs, challenger);

    let proof = prove(&config, &proof_goal, trace, &vec![]);
    //report_proof_size(&proof);

    verify(&config, &proof_goal, &proof, &vec![])
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

// General constants for constructing the Poseidon2 AIR.
const P2_WIDTH: usize = 16;
const P2_HALF_FULL_ROUNDS: usize = 4;
const P2_LOG_VECTOR_LEN: usize = 3;
const P2_VECTOR_LEN: usize = 1 << P2_LOG_VECTOR_LEN;

fn run_fri() {
    let args = Args::parse();
    let num_evaluations = args.num_evaluations;
    let trace_height = 1 << args.num_variables;

    let num_hashes = {
        println!("Proving 2^{} native Poseidon-2 permutations", {
            args.num_variables + P2_LOG_VECTOR_LEN
        });
        trace_height << P2_LOG_VECTOR_LEN
    };

    // WHIR setup, to reuse parameters for FRI (that are applicable)
    let params = prepare_params();

    // WARNING: Use a real cryptographic PRNG in applications!!
    let mut rng = SmallRng::seed_from_u64(1);

    type EF = BinomialExtensionField<KoalaBear, 4>;

    let proof_goal = {
        let constants = RoundConstants::from_rng(&mut rng);

        // Field specific constants for constructing the Poseidon2 AIR.
        const SBOX_DEGREE: u64 = 3;
        const SBOX_REGISTERS: usize = 0;
        const PARTIAL_ROUNDS: usize = 20;

        let p2_air: VectorizedPoseidon2Air<
            KoalaBear,
            GenericPoseidon2LinearLayersKoalaBear,
            P2_WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            P2_HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
            P2_VECTOR_LEN,
        > = VectorizedPoseidon2Air::new(constants);
        ProofObjective::Poseidon2(p2_air)
    };

    let dft = Radix2DFTSmallBatch::<F>::new(1 << params.max_fft_size());
    //let dft = Radix2DitParallel::default();
    //let dft = RecursiveDft::new(trace_height << 1);
    // match args.discrete_fourier_transform {
    //    DftOptions::RecursiveDft => DftChoice::Recursive(RecursiveDft::new(trace_height << 1)),
    //    DftOptions::Radix2DitParallel => DftChoice::Parallel(Radix2DitParallel::default()),
    //    DftOptions::None => panic!(
    //        "Please specify what dft to use. Options are recursive-dft and radix-2-dit-parallel"
    //    ),
    //};

    let perm16 = Poseidon2KoalaBear::<16>::new_from_rng_128(&mut rng);
    let perm24 = Poseidon2KoalaBear::<24>::new_from_rng_128(&mut rng);
    let result =
        prove_monty31_poseidon2::<_, EF, _, _, _, _>(proof_goal, dft, num_hashes, perm16, perm24);
    report_result(result);
}

#[allow(clippy::too_many_lines)]
fn main() {
    // 1. First run WHIR
    run_whir();

    // 2. Now run FRI
    run_fri();
}
