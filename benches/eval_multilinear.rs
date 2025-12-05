use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use whir_p3::poly::{
    evals::{eval_multilinear_base, eval_multilinear_ext},
    multilinear::MultilinearPoint,
};

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;

fn bench_eval_multilinear_base(c: &mut Criterion) {
    let mut group = c.benchmark_group("eval_multilinear_base");

    for num_vars in (8..=22).step_by(2) {
        let num_evals = 1 << num_vars;

        let throughput = Throughput::Bytes((num_evals * core::mem::size_of::<F>()) as u64);
        group.throughput(throughput);

        // Setup closure: Generate random evaluation data and a random point.
        let setup = |n_vars| {
            let mut rng = SmallRng::seed_from_u64(1);
            let evals: Vec<F> = (0..1 << n_vars).map(|_| rng.random()).collect();

            let point_vec: Vec<EF> = (0..n_vars).map(|_| rng.random()).collect();
            let point = MultilinearPoint::new(point_vec);

            (evals, point)
        };

        let bench_id = BenchmarkId::new("packed-split", num_vars);
        group.bench_with_input(bench_id, &num_vars, |b, &n_vars| {
            let routine = |(evals, point): (Vec<F>, MultilinearPoint<EF>)| {
                let _ = std::hint::black_box(eval_multilinear_base(
                    std::hint::black_box(&evals),
                    std::hint::black_box(&point),
                ));
            };
            b.iter_batched(|| setup(n_vars), routine, criterion::BatchSize::SmallInput);
        });
    }
    group.finish();
}

fn bench_eval_multilinear_ext(c: &mut Criterion) {
    let mut group = c.benchmark_group("eval_multilinear_ext");

    for num_vars in (8..=22).step_by(2) {
        let num_evals = 1 << num_vars;

        let throughput = Throughput::Bytes((num_evals * core::mem::size_of::<F>()) as u64);
        group.throughput(throughput);

        // Setup closure: Generate random evaluation data and a random point.
        let setup = |n_vars| {
            let mut rng = SmallRng::seed_from_u64(1);
            let evals: Vec<EF> = (0..1 << n_vars).map(|_| rng.random()).collect();

            let point_vec: Vec<EF> = (0..n_vars).map(|_| rng.random()).collect();
            let point = MultilinearPoint::new(point_vec);

            (evals, point)
        };

        let bench_id = BenchmarkId::new("packed-split", num_vars);
        group.bench_with_input(bench_id, &num_vars, |b, &n_vars| {
            let routine = |(evals, point): (Vec<EF>, MultilinearPoint<EF>)| {
                let _ = std::hint::black_box(eval_multilinear_ext::<F, _>(
                    std::hint::black_box(&evals),
                    std::hint::black_box(&point),
                ));
            };
            b.iter_batched(|| setup(n_vars), routine, criterion::BatchSize::SmallInput);
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_eval_multilinear_base,
    bench_eval_multilinear_ext
);
criterion_main!(benches);
