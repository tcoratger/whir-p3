use criterion::{Criterion, criterion_group, criterion_main};
use p3_challenger::DuplexChallenger;
use p3_field::extension::BinomialExtensionField;
use p3_goldilocks::Poseidon2Goldilocks;
use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::{
    Rng, SeedableRng,
    rngs::{SmallRng, StdRng},
};
use whir_p3::{
    dft::EvalsDft,
    fiat_shamir::domain_separator::DomainSeparator,
    parameters::{DEFAULT_MAX_POW, FoldingFactor, ProtocolParameters, errors::SecurityAssumption},
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    whir::{
        committer::writer::CommitmentWriter, constraints::statement::Statement,
        parameters::WhirConfig, prover::Prover,
    },
};

type F = KoalaBear;
type EF = BinomialExtensionField<F, 4>;
type Perm = Poseidon2Goldilocks<16>;

type Poseidon16 = Poseidon2KoalaBear<16>;
type Poseidon24 = Poseidon2KoalaBear<24>;

type MerkleHash = PaddingFreeSponge<Poseidon24, 24, 16, 8>; // leaf hashing
type MerkleCompress = TruncatedPermutation<Poseidon16, 2, 8, 16>; // 2-to-1 compression
type MyChallenger = DuplexChallenger<F, Poseidon16, 16, 8>;

#[allow(clippy::type_complexity)]
fn prepare_inputs() -> (
    WhirConfig<EF, F, MerkleHash, MerkleCompress, MyChallenger>,
    EvalsDft<F>,
    EvaluationsList<F>,
    Statement<EF>,
    MyChallenger,
    DomainSeparator<EF, F>,
) {
    // Protocol parameter configuration

    // Target cryptographic security in bits.
    let security_level = 100;

    // Number of Boolean variables in the multilinear polynomial. Polynomial has 2^24 coefficients.
    let num_variables = 24;

    // Number of PoW bits required, computed based on the domain size and rate.
    let pow_bits = DEFAULT_MAX_POW;

    // Folding factor `k`: number of variables folded per round in the sumcheck.
    let folding_factor = FoldingFactor::Constant(4);

    // Low-degree extension (LDE) blowup factor: inverse of `rate`.
    let starting_rate = 1;

    // RS code initial domain size reduction factor (controls the LDE domain size).
    let rs_domain_initial_reduction_factor = 3;

    // Create multivariate polynomial and hash setup

    // Define the hash functions for Merkle tree and compression.
    let mut rng = SmallRng::seed_from_u64(1);
    let poseidon16 = Poseidon16::new_from_rng_128(&mut rng);
    let poseidon24 = Poseidon24::new_from_rng_128(&mut rng);

    let merkle_hash = MerkleHash::new(poseidon24);
    let merkle_compress = MerkleCompress::new(poseidon16.clone());

    // Type of soundness assumption used in the IOP model.
    let soundness_type = SecurityAssumption::CapacityBound;

    // Assemble the protocol-level parameters.
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

    // Combine multivariate and protocol parameters into a unified WHIR config.
    let params = WhirConfig::new(num_variables, whir_params);

    // Sample random multilinear polynomial

    // Total number of coefficients = 2^num_variables.
    let num_coeffs = 1 << num_variables;

    // Use a fixed-seed RNG to ensure deterministic benchmark inputs.
    let mut rng = StdRng::seed_from_u64(0);

    // Sample a random multilinear polynomial over `F`, represented by its evaluations.
    let polynomial = EvaluationsList::<F>::new((0..num_coeffs).map(|_| rng.random()).collect());

    // Build a simple constraint system with one point

    // Sample a random Boolean point in {0,1}^num_variables.
    let point = MultilinearPoint::rand(&mut rng, num_variables);

    // Create a new WHIR `Statement` with one constraint.
    let mut statement = Statement::<EF>::initialize(num_variables);
    statement.add_unevaluated_constraint(point, &polynomial);

    // Fiat-Shamir setup

    // Create a domain separator for transcript hashing.
    let mut domainsep = DomainSeparator::new(vec![]);

    // Commit protocol parameters and proof type to the domain separator.
    domainsep.commit_statement::<_, _, _, 32>(&params);
    domainsep.add_whir_proof::<_, _, _, 32>(&params);

    // Instantiate the Fiat-Shamir challenger from an empty seed and Keccak.
    let challenger = MyChallenger::new(poseidon16);

    // DFT backend setup

    // Construct a Radix-2 FFT backend that supports small batch DFTs over `F`.
    let dft = EvalsDft::<F>::new(1 << params.max_fft_size());

    // Return all preprocessed components needed to run commit/prove/verify benchmarks.
    (params, dft, polynomial, statement, challenger, domainsep)
}

fn benchmark_commit_and_prove(c: &mut Criterion) {
    let (params, dft, polynomial, statement, challenger, domainsep) = prepare_inputs();

    c.bench_function("commit", |b| {
        b.iter(|| {
            let mut prover_state = domainsep.to_prover_state(challenger.clone());
            let committer = CommitmentWriter::new(&params);
            let _witness = committer
                .commit(&dft, &mut prover_state, polynomial.clone())
                .unwrap();
        });
    });

    c.bench_function("prove", |b| {
        b.iter(|| {
            let mut prover_state = domainsep.to_prover_state(challenger.clone());
            let committer = CommitmentWriter::new(&params);
            let witness = committer
                .commit(&dft, &mut prover_state, polynomial.clone())
                .unwrap();

            let prover = Prover(&params);
            prover
                .prove(&dft, &mut prover_state, statement.clone(), witness)
                .unwrap();
        });
    });
}

criterion_group!(benches, benchmark_commit_and_prove);
criterion_main!(benches);
