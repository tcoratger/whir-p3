use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_challenger::{DuplexChallenger, FieldChallenger, GrindingChallenger};
use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear};
use rand::{Rng, SeedableRng, rngs::StdRng};
use whir::{
    fiat_shamir::{domain_separator::DomainSeparator, prover::ProverState},
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    sumcheck::sumcheck_single::SumcheckSingle,
    whir::constraints::{evaluator::Constraint, statement::EqStatement},
};
use whir_p3 as whir;

type F = KoalaBear;
type EF = BinomialExtensionField<F, 8>;
type Poseidon16 = Poseidon2KoalaBear<16>;
type MyChallenger = DuplexChallenger<F, Poseidon16, 16, 8>;

const NUM_CONSTRAINTS: usize = 1;
const FOLDING_FACTOR: usize = 5;
const POW_BITS: usize = 0;

fn setup_prover() -> ProverState<F, EF, MyChallenger> {
    let mut rng = StdRng::seed_from_u64(0);
    let poseidon = Poseidon16::new_from_rng_128(&mut rng);
    let challenger = MyChallenger::new(poseidon);
    DomainSeparator::new(vec![]).to_prover_state(challenger)
}

fn generate_poly(num_vars: usize) -> EvaluationsList<F> {
    let mut rng = StdRng::seed_from_u64(1 + num_vars as u64);
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

        let mut prover = setup_prover();
        let statement = generate_statement(&mut prover, *num_vars, &poly, 3);
        let constraint = Constraint::new_eq_only(prover.sample(), statement.clone());

        group.bench_with_input(BenchmarkId::new("Classic", *num_vars), &poly, |b, poly| {
            b.iter(|| {
                SumcheckSingle::from_base_evals(poly, &mut prover, *num_vars, 0, &constraint);
            });
        });
        group.bench_with_input(BenchmarkId::new("SVO", *num_vars), &poly, |b, poly| {
            b.iter(|| {
                SumcheckSingle::from_base_evals_svo(poly, &mut prover, *num_vars, 0, &constraint);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_sumcheck_prover_svo);
criterion_main!(benches);
