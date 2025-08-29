use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::KoalaBear;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use whir_p3::poly::{coeffs::CoefficientList, multilinear::MultilinearPoint};

type F = KoalaBear;
type EF4 = BinomialExtensionField<F, 4>;

fn generate_test_case(num_variables: usize) -> (CoefficientList<F>, MultilinearPoint<EF4>) {
    let mut rng = SmallRng::seed_from_u64(1);
    let num_coeffs = 1 << num_variables;
    let coeffs = (0..num_coeffs).map(|_| rng.random()).collect();
    let point = MultilinearPoint::new((0..num_variables).map(|_| rng.random()).collect());
    (CoefficientList::new(coeffs), point)
}

fn benchmark_evaluate(c: &mut Criterion) {
    let mut group = c.benchmark_group("evaluate");

    for &num_variables in &[4, 8, 12, 16, 20, 25] {
        let (poly, point) = generate_test_case(num_variables);

        group.bench_with_input(
            BenchmarkId::from_parameter(num_variables),
            &num_variables,
            |b, &_num_variables| {
                b.iter_batched(
                    || (poly.clone(), point.clone()),
                    |(poly, point)| {
                        let _ = poly.evaluate(&point);
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_evaluate);
criterion_main!(benches);
