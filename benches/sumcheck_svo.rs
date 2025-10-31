use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_challenger::DuplexChallenger;
use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::hint::black_box;
use whir::{
    fiat_shamir::{domain_separator::DomainSeparator, prover::ProverState},
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    sumcheck::sumcheck_single::SumcheckSingle,
    whir::statement::{Statement, point::ConstraintPoint},
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

fn generate_statement(
    num_vars: usize,
    poly: &EvaluationsList<F>,
    num_constraints: usize,
) -> Statement<EF> {
    let mut rng = StdRng::seed_from_u64(42 + num_vars as u64);
    let mut statement = Statement::new(num_vars);
    for _ in 0..num_constraints {
        let point = MultilinearPoint::rand(&mut rng, num_vars);
        let eval = poly.evaluate(&point);
        statement.add_constraint(ConstraintPoint::new(point), eval);
    }
    statement
}

fn bench_sumcheck_prover_svo(c: &mut Criterion) {
    let mut group = c.benchmark_group("SumcheckProver");
    group.sample_size(30);

    for &num_vars in &[16, 18, 20, 22] {
        let poly = generate_poly(num_vars);
        let statement = generate_statement(num_vars, &poly, NUM_CONSTRAINTS);

        group.bench_with_input(
            BenchmarkId::new("Classic", num_vars),
            &num_vars,
            |b, &_num_vars| {
                b.iter(|| {
                    let mut prover = setup_prover();
                    let combination_randomness: EF = prover.sample();
                    let result = SumcheckSingle::from_base_evals(
                        &poly,
                        &statement,
                        combination_randomness,
                        &mut prover,
                        FOLDING_FACTOR,
                        POW_BITS,
                    );
                    black_box(result);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("SVO + Round", num_vars),
            &num_vars,
            |b, &_num_vars| {
                b.iter(|| {
                    let mut prover = setup_prover();
                    let combination_randomness: EF = prover.sample();
                    let result = SumcheckSingle::from_base_evals_svo(
                        &poly,
                        &statement,
                        combination_randomness,
                        &mut prover,
                        FOLDING_FACTOR,
                        POW_BITS,
                    );
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_sumcheck_prover_svo);
criterion_main!(benches);
