use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_field::PrimeCharacteristicRing;
use whir_p3::utils::parallel_clone;

fn benchmark_clones(c: &mut Criterion) {
    let mut group = c.benchmark_group("Parallel Clone Comparison");

    let sizes = [
        1 << 12,
        1 << 14,
        1 << 15,
        1 << 16,
        1 << 18,
        1 << 20,
        1 << 22,
    ];

    for size in &sizes {
        let src: Vec<BabyBear> = (0..*size).map(|i| BabyBear::from_u64(i as u64)).collect();
        let mut dst = vec![BabyBear::ZERO; *size];

        // Measure throughput
        let throughput = Throughput::Bytes((size * std::mem::size_of::<BabyBear>()) as u64);
        group.throughput(throughput);

        // Benchmark the standard library's clone_from_slice
        group.bench_with_input(
            BenchmarkId::new("slice::clone_from_slice", size),
            &src,
            |b, src| {
                b.iter(|| {
                    dst.clone_from_slice(black_box(src));
                });
            },
        );

        // Benchmark your parallel_clone function
        group.bench_with_input(BenchmarkId::new("parallel_clone", size), &src, |b, src| {
            b.iter(|| {
                parallel_clone(black_box(src), black_box(&mut dst));
            });
        });
    }

    group.finish();
}

// Register the benchmark group with criterion
criterion_group!(benches, benchmark_clones);
// Generate the main function to run the benchmarks
criterion_main!(benches);
