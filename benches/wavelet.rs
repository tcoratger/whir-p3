use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
use whir_p3::poly::wavelet::Radix2WaveletKernel;

/// Benchmark the single‑matrix in‑place transform on matrices whose height is a power of two.
fn bench_wavelet_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("wavelet_transform_in_place");

    // Heights 2^10 (=1 024) up to 2^20 (≈1 M) rows.
    for &pow in &[10u32, 12, 14, 16, 18, 20] {
        let rows = 1usize << pow;

        let kernel = Radix2WaveletKernel::default();

        // Prepare a deterministic data set so each run does the same amount of work.
        let values: Vec<BabyBear> = (1..=rows as u64).map(BabyBear::from_u64).collect();

        group.throughput(Throughput::Elements(rows as u64));
        group.bench_with_input(BenchmarkId::from_parameter(pow), &rows, |b, &_rows| {
            b.iter_batched(
                // 1.  Setup: clone the values into a fresh matrix so every iteration starts clean.
                || RowMajorMatrix::new_col(values.clone()),
                // 2.  Do the work we want to measure.
                |mat| kernel.wavelet_transform_batch(mat),
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_wavelet_transform,);
criterion_main!(benches);
