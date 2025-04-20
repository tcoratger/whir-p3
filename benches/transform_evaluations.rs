use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_dft::Radix2DitParallel;
use p3_field::{Field, PrimeCharacteristicRing, extension::BinomialExtensionField};
use rand::{Rng, rng};
use whir_p3::{parameters::FoldType, poly::fold::transform_evaluations};

type F = BabyBear;
type EF4 = BinomialExtensionField<F, 4>;

fn generate_random_evals(size: usize) -> Vec<EF4> {
    let mut rng = rng();
    (0..size).map(|_| rng.random()).collect()
}

fn bench_transform_evaluations_ef4(c: &mut Criterion) {
    let mut group = c.benchmark_group("transform_evaluations_ef4");

    for &log_domain_size in &[16, 20] {
        let domain_size = 1 << log_domain_size;

        for &folding_factor in &[4, 5] {
            let folding_factor_exp = 1 << folding_factor;

            // Skip incompatible configs
            if domain_size % folding_factor_exp != 0 {
                continue;
            }

            let evals = generate_random_evals(domain_size);
            let domain_gen = EF4::from_u64(0x5ee9_9486);
            let domain_gen_inv = domain_gen.inverse();

            let dft = Radix2DitParallel::<F>::default();

            group.bench_with_input(
                BenchmarkId::from_parameter(format!(
                    "2^{log_domain_size} elements, 2^{folding_factor} folding"
                )),
                &evals,
                |b, data| {
                    let mut data = data.clone();
                    b.iter(|| {
                        transform_evaluations::<F, EF4, _>(
                            &mut data,
                            &dft,
                            FoldType::ProverHelps,
                            domain_gen,
                            domain_gen_inv,
                            folding_factor,
                        );
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench_transform_evaluations_ef4);
criterion_main!(benches);
