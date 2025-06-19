use std::{collections::HashSet, hint::black_box};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use rand::{Rng, SeedableRng, rngs::StdRng};
use whir_p3::poly::dense::WhirDensePolynomial;

// Field type used for the benchmark.
type F = BabyBear;

/// Generate random interpolation points for a given polynomial degree.
///
/// Returns `degree + 1` unique (x, y) pairs from a random polynomial.
fn generate_interpolation_points(rng: &mut StdRng, degree: usize) -> Vec<(F, F)> {
    let poly = WhirDensePolynomial::<F>::random(rng, degree);
    let mut points = Vec::with_capacity(degree + 1);
    let mut used_x = HashSet::with_capacity(degree + 1);

    while points.len() < degree + 1 {
        let x: F = rng.random();
        if used_x.insert(x) {
            let y = poly.evaluate(x);
            points.push((x, y));
        }
    }

    points
}

/// Benchmark Lagrange interpolation for various input sizes.
fn lagrange_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Lagrange Interpolation");
    let mut rng = StdRng::seed_from_u64(42);

    // Test different polynomial degrees.
    for degree in [16, 32, 64, 128, 256, 512] {
        let points = generate_interpolation_points(&mut rng, degree);

        group.bench_with_input(BenchmarkId::from_parameter(degree), &degree, |b, _| {
            b.iter(|| {
                WhirDensePolynomial::<F>::lagrange_interpolation(black_box(&points));
            });
        });
    }

    group.finish();
}

criterion_group!(benches, lagrange_benchmark);
criterion_main!(benches);
