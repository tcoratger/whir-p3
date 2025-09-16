use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::{DuplexChallenger, FieldChallenger, GrindingChallenger};
use p3_field::extension::BinomialExtensionField;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use whir_p3::{
    fiat_shamir::{domain_separator::DomainSeparator, prover::ProverState},
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    sumcheck::sumcheck_single::SumcheckSingle,
    whir::constraints::statement::Statement,
};

type F = BabyBear;
type EF = BinomialExtensionField<BabyBear, 4>;
type Perm = Poseidon2BabyBear<16>;
type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

/// Helper to create a fresh prover state for each benchmark iteration.
fn setup_prover() -> ProverState<F, EF, MyChallenger> {
    let mut rng = SmallRng::seed_from_u64(0);
    let perm = Perm::new_from_rng_128(&mut rng);
    let challenger = MyChallenger::new(perm);
    DomainSeparator::new(vec![]).to_prover_state(challenger)
}

/// Helper to generate a random multilinear polynomial.
fn generate_poly(num_vars: usize) -> EvaluationsList<F> {
    let mut rng = SmallRng::seed_from_u64(1);
    let evals = (0..1 << num_vars).map(|_| rng.random()).collect();
    EvaluationsList::new(evals)
}

/// Helper to generate an initial statement with a few constraints.
fn generate_statement<C>(
    prover: &mut ProverState<F, EF, C>,
    num_vars: usize,
    poly: &EvaluationsList<F>,
    num_constraints: usize,
) -> Statement<EF>
where
    C: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    let mut statement = Statement::initialize(num_vars);
    for _ in 0..num_constraints {
        let point = MultilinearPoint::expand_from_univariate(prover.sample(), num_vars);
        statement.add_unevaluated_constraint(point, poly);
    }
    statement
}

/// Main benchmark function to test the sumcheck prover.
fn bench_sumcheck_prover(c: &mut Criterion) {
    let mut group = c.benchmark_group("SumcheckProver");
    // Use a smaller sample size for long-running benchmarks
    group.sample_size(10);

    // Define the range of variable counts to benchmark.
    for num_vars in &[16, 18, 20, 22, 24] {
        // Generate a large polynomial to use for this set of benchmarks.
        let poly = generate_poly(*num_vars);

        // Benchmark for the classic, round-by-round sumcheck
        let classic_folding_schedule = [*num_vars / 2, num_vars - (*num_vars / 2)];

        let mut prover = setup_prover();
        let statement = generate_statement(&mut prover, *num_vars, &poly, 3);
        let combination_randomness: EF = prover.sample();

        group.bench_with_input(BenchmarkId::new("Classic", *num_vars), &poly, |b, poly| {
            b.iter(|| {
                let (mut sumcheck, _) = SumcheckSingle::from_base_evals(
                    poly,
                    &statement,
                    combination_randomness,
                    &mut prover,
                    classic_folding_schedule[0],
                    0,
                );

                // Run the remaining folding rounds
                if classic_folding_schedule.len() > 1 {
                    sumcheck.compute_sumcheck_polynomials(
                        &mut prover,
                        classic_folding_schedule[1],
                        0,
                    );
                }
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_sumcheck_prover);
criterion_main!(benches);
