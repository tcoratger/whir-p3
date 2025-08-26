use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use rand::Rng;
use whir_p3::{
    poly::multilinear::MultilinearPoint, sumcheck::sumcheck_polynomial::SumcheckPolynomial,
};

type F = BabyBear;

fn benchmark_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("evaluate_on_standard_domain");
    group.sample_size(10);

    let mut rng = rand::rng();

    // Define a range of dimensions to benchmark.
    for &n_vars in [5, 10, 11, 12, 13, 14, 15].iter() {
        let num_evals = 3usize.pow(n_vars as u32);

        let evaluations: Vec<F> = (0..num_evals).map(|_| rng.random()).collect();
        let poly = SumcheckPolynomial::new(evaluations, n_vars);
        let point = MultilinearPoint::rand(&mut rng, n_vars);

        let id = BenchmarkId::from_parameter(format!("n_vars={}", n_vars));

        group.bench_with_input(id, &(poly, point), |b, (poly, point)| {
            b.iter(|| {
                let _ = poly.evaluate_on_standard_domain(black_box(point));
            });
        });
    }
    group.finish();
}

criterion_group!(benches, benchmark_evaluation);
criterion_main!(benches);
