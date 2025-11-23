use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_challenger::{DuplexChallenger, FieldChallenger, GrindingChallenger};
use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::{Rng, SeedableRng, rngs::SmallRng};
use whir::{
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    sumcheck::sumcheck_single::SumcheckSingle,
    whir::constraints::statement::EqStatement,
};
use whir_p3::{
    self as whir,
    fiat_shamir::{domain_separator::DomainSeparator, prover::ProverState},
    parameters::{FoldingFactor, ProtocolParameters, errors::SecurityAssumption},
    whir::{constraints::Constraint, parameters::SumcheckOptimization, proof::WhirProof},
};

type F = KoalaBear;
type EF = BinomialExtensionField<F, 8>;
type Poseidon16 = Poseidon2KoalaBear<16>;
type MyHash = PaddingFreeSponge<Poseidon16, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Poseidon16, 2, 8, 16>;
type MyChallenger = DuplexChallenger<F, Poseidon16, 16, 8>;

const NUM_CONSTRAINTS: usize = 1;
const FOLDING_FACTOR: usize = 5;
const POW_BITS: usize = 0;

/// Helper to create protocol parameters for benchmarking
fn create_test_protocol_params_classic(
    folding_factor: FoldingFactor,
) -> ProtocolParameters<MyHash, MyCompress> {
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Poseidon16::new_from_rng_128(&mut rng);

    ProtocolParameters {
        initial_statement: true,
        security_level: 32,
        pow_bits: 0,
        rs_domain_initial_reduction_factor: 1,
        folding_factor,
        merkle_hash: MyHash::new(perm.clone()),
        merkle_compress: MyCompress::new(perm),
        soundness_type: SecurityAssumption::UniqueDecoding,
        starting_log_inv_rate: 1,
        sumcheck_optimization: SumcheckOptimization::Classic,
    }
}

fn setup_domsep_and_challenger() -> (DomainSeparator<EF, F>, MyChallenger) {
    let mut rng = SmallRng::seed_from_u64(0);
    let poseidon = Poseidon16::new_from_rng_128(&mut rng);
    let challenger = MyChallenger::new(poseidon);
    let domsep = DomainSeparator::new(vec![]);
    (domsep, challenger)
}

fn generate_poly(num_vars: usize) -> EvaluationsList<F> {
    let mut rng = SmallRng::seed_from_u64(1 + num_vars as u64);
    EvaluationsList::new((0..1 << num_vars).map(|_| rng.random()).collect())
}

/// Helper to generate an initial statement with a few constraints.
fn generate_statement<C>(
    prover: &mut ProverState<F, EF, C>,
    num_vars: usize,
    poly: &EvaluationsList<F>,
    num_constraints: usize,
) -> EqStatement<EF>
where
    C: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    let mut statement = EqStatement::initialize(num_vars);
    for _ in 0..num_constraints {
        let point = MultilinearPoint::expand_from_univariate(prover.sample(), num_vars);
        statement.add_unevaluated_constraint_hypercube(point, poly);
    }
    statement
}

fn bench_sumcheck_prover_svo(c: &mut Criterion) {
    let mut group = c.benchmark_group("SumcheckProver");
    // Use a smaller sample size for long-running benchmarks
    group.sample_size(10);

    // Define the range of variable counts to benchmark.
    for num_vars in &[16, 18, 20, 22, 24] {
        // Generate a large polynomial to use for this set of benchmarks.
        let poly = generate_poly(*num_vars);

        // Classic benchmark - folding all variables in one round
        let params_classic =
            create_test_protocol_params_classic(FoldingFactor::Constant(*num_vars));
        group.bench_with_input(BenchmarkId::new("Classic", *num_vars), &poly, |b, poly| {
            b.iter(|| {
                // Setup fresh for each iteration
                let (domsep, challenger_for_prover) = setup_prover_and_challenger();
                let mut prover = domsep.to_prover_state(challenger_for_prover);

                // Initialize proof and challenger (refactored approach)
                let mut proof =
                    WhirProof::<F, EF, 8>::from_protocol_parameters(&params_classic, *num_vars);
                let mut rng = SmallRng::seed_from_u64(1);
                let mut challenger_rf = MyChallenger::new(Poseidon16::new_from_rng_128(&mut rng));
                domsep.observe_domain_separator(&mut challenger_rf);

                // Create constraint
                let statement = generate_statement(&mut prover, *num_vars, poly, 3);
                let constraint = Constraint::new_eq_only(prover.sample(), statement);
                // Keep challenger_rf in sync
                let _alpha_rf: EF = challenger_rf.sample_algebra_element();

                // Fold all variables in one round
                SumcheckSingle::from_base_evals(
                    poly,
                    &mut prover,
                    &mut proof,
                    &mut challenger_rf,
                    *num_vars,
                    0,
                    &constraint,
                );
            });
        });

        // SVO benchmark - using SVO optimization (still uses old ProverState approach)
        group.bench_with_input(BenchmarkId::new("SVO", *num_vars), &poly, |b, poly| {
            b.iter(|| {
                // Setup fresh for each iteration (SVO still uses old approach)
                let (domsep, challenger_for_prover) = setup_prover_and_challenger();
                let mut prover = domsep.to_prover_state(challenger_for_prover);

                // Create constraint
                let statement = generate_statement(&mut prover, *num_vars, poly, 3);
                let constraint = Constraint::new_eq_only(prover.sample(), statement);

                // Fold all variables using SVO optimization (old approach)
                SumcheckSingle::from_base_evals_svo(poly, &mut prover, *num_vars, 0, &constraint);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_sumcheck_prover_svo);
criterion_main!(benches);
