use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_dft::Radix2DitParallel;
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, extension::BinomialExtensionField};
use p3_monty_31::dft::RecursiveDft;
use rand::{Rng, rng};
use whir_p3::ntt::{expand_from_coeff, expand_from_coeff_plonky3};

type F = BabyBear;
type EF4 = BinomialExtensionField<F, 4>;

fn generate_random_coeffs(size: usize) -> Vec<EF4> {
    let mut rng = rng();
    (0..size)
        .map(|_| {
            EF4::from_basis_coefficients_iter(
                [
                    F::from_u32(rng.random()),
                    F::from_u32(rng.random()),
                    F::from_u32(rng.random()),
                    F::from_u32(rng.random()),
                ]
                .into_iter(),
            )
        })
        .collect()
}

fn bench_expand_from_coeff(c: &mut Criterion) {
    let mut group = c.benchmark_group("expand_from_coeff");

    // Try different coefficient sizes
    for &log_n in &[18, 20] {
        let n = 1 << log_n;
        let coeffs = generate_random_coeffs(n);

        // let dft = RecursiveDft::<EF4>::default();
        let dft = Radix2DitParallel::<EF4>::default();

        for &expansion in &[4, 8] {
            group.bench_with_input(
                BenchmarkId::new(format!("n={n}"), expansion),
                &expansion,
                |b, &e| {
                    b.iter(|| {
                        // let _ = expand_from_coeff(&coeffs, e);
                        let _ = expand_from_coeff_plonky3(&dft, &coeffs, e);
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench_expand_from_coeff);
criterion_main!(benches);
