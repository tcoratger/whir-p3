use criterion::{Criterion, criterion_group, criterion_main};
use p3_challenger::DuplexChallenger;
use p3_field::{Field, extension::BinomialExtensionField};
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::{
    Rng, SeedableRng,
    rngs::{SmallRng, StdRng},
};
use whir_p3::{
    dft::EvalsDft,
    fiat_shamir::domain_separator::DomainSeparator,
    parameters::{
        DEFAULT_MAX_POW, FoldingFactor, MultivariateParameters, ProtocolParameters,
        errors::SecurityAssumption,
    },
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    whir::{
        committer::writer::CommitmentWriter,
        parameters::WhirConfig,
        prover::Prover,
        statement::{Statement, weights::Weights},
    },
};

type F = Goldilocks;
type EF = BinomialExtensionField<F, 2>;
type Perm = Poseidon2Goldilocks<16>;

type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

#[allow(clippy::type_complexity)]
fn prepare_inputs() -> (
    WhirConfig<EF, F, MyHash, MyCompress, MyChallenger>,
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

    // Instantiate parameters for a degree-24 multilinear polynomial over EF.
    let mv_params = MultivariateParameters::<EF>::new(num_variables);

    // Define the hash functions for Merkle tree and compression.
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);

    let merkle_hash = MyHash::new(perm.clone());
    let merkle_compress = MyCompress::new(perm);

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
    let params = WhirConfig::new(mv_params, whir_params);

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

    // Evaluate the polynomial at that point to get the expected value.
    let eval = polynomial.evaluate(&point);

    // Construct a constraint: enforces that the polynomial evaluates to `eval` at `point`.
    let weights = Weights::evaluation(point);

    // Create a new WHIR `Statement` with one constraint.
    let mut statement = Statement::<EF>::new(num_variables);
    statement.add_constraint(weights, eval);

    // Fiat-Shamir setup

    // Create a domain separator for transcript hashing.
    let mut domainsep = DomainSeparator::new(vec![]);

    // Commit protocol parameters and proof type to the domain separator.
    domainsep.commit_statement::<_, _, _, 32>(&params);
    domainsep.add_whir_proof::<_, _, _, 32>(&params);

    // Instantiate the Fiat-Shamir challenger from an empty seed and Keccak.
    let mut rng = SmallRng::seed_from_u64(1);
    let challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));

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
                .commit::<<F as Field>::Packing, <F as Field>::Packing, 8>(
                    &dft,
                    &mut prover_state,
                    polynomial.clone(),
                )
                .unwrap();
        });
    });

    c.bench_function("prove", |b| {
        b.iter(|| {
            let mut prover_state = domainsep.to_prover_state(challenger.clone());
            let committer = CommitmentWriter::new(&params);
            let witness = committer
                .commit::<<F as Field>::Packing, <F as Field>::Packing, 8>(
                    &dft,
                    &mut prover_state,
                    polynomial.clone(),
                )
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
