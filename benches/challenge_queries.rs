use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use whir_p3::{fiat_shamir::ChallengSampler, whir::utils::get_challenge_stir_queries};

type F = BabyBear;

// Simple test challenger for benchmarking
struct TestChallenger {
    counter: usize,
}

impl TestChallenger {
    fn new() -> Self {
        Self { counter: 0 }
    }
}

impl ChallengSampler<F> for TestChallenger {
    fn sample(&mut self) -> F {
        unimplemented!("Not needed for this benchmark")
    }

    fn sample_bits(&mut self, bits: usize) -> usize {
        self.counter += 1;
        // Simple pseudo-random number generation
        (self.counter.wrapping_mul(1103515245).wrapping_add(12345) >> 16) & ((1 << bits) - 1)
    }
}

fn create_challenger() -> TestChallenger {
    TestChallenger::new()
}

// Original implementation (for comparison)
fn get_challenge_stir_queries_original(
    domain_size: usize,
    folding_factor: usize,
    num_queries: usize,
    challenger: &mut TestChallenger,
) -> Vec<usize> {
    use itertools::Itertools;
    use p3_util::log2_ceil_usize;

    let folded_domain_size = domain_size >> folding_factor;
    let domain_size_bits = log2_ceil_usize(folded_domain_size);

    (0..num_queries)
        .map(|_| challenger.sample_bits(domain_size_bits) % folded_domain_size)
        .sorted_unstable()
        .dedup()
        .collect()
}

fn bench_challenge_queries_small(c: &mut Criterion) {
    let mut group = c.benchmark_group("challenge_queries_small");

    // Small-scale test: 16 queries, 8 bits each = 128 total bits
    let domain_size = 1 << 20; // 1M
    let folding_factor = 12; // Fold to 256
    let num_queries = 16;

    group.bench_function("original", |b| {
        b.iter(|| {
            let mut challenger = create_challenger();
            black_box(get_challenge_stir_queries_original(
                black_box(domain_size),
                black_box(folding_factor),
                black_box(num_queries),
                black_box(&mut challenger),
            ))
        })
    });

    group.bench_function("optimized", |b| {
        b.iter(|| {
            let mut challenger = create_challenger();
            black_box(
                get_challenge_stir_queries::<TestChallenger, F>(
                    black_box(domain_size),
                    black_box(folding_factor),
                    black_box(num_queries),
                    black_box(&mut challenger),
                )
                .unwrap(),
            )
        })
    });

    group.finish();
}

fn bench_challenge_queries_medium(c: &mut Criterion) {
    let mut group = c.benchmark_group("challenge_queries_medium");

    // Medium-scale test: 64 queries, 12 bits each = 768 total bits
    let domain_size = 1 << 22; // 4M
    let folding_factor = 10; // Fold to 4096
    let num_queries = 64;

    group.bench_function("original", |b| {
        b.iter(|| {
            let mut challenger = create_challenger();
            black_box(get_challenge_stir_queries_original(
                black_box(domain_size),
                black_box(folding_factor),
                black_box(num_queries),
                black_box(&mut challenger),
            ))
        })
    });

    group.bench_function("optimized", |b| {
        b.iter(|| {
            let mut challenger = create_challenger();
            black_box(
                get_challenge_stir_queries::<TestChallenger, F>(
                    black_box(domain_size),
                    black_box(folding_factor),
                    black_box(num_queries),
                    black_box(&mut challenger),
                )
                .unwrap(),
            )
        })
    });

    group.finish();
}

fn bench_challenge_queries_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("challenge_queries_large");

    // Large-scale test: 128 queries, 16 bits each = 2048 total bits (exceeds threshold)
    let domain_size = 1 << 24; // 16M
    let folding_factor = 8; // Fold to 65536
    let num_queries = 128;

    group.bench_function("original", |b| {
        b.iter(|| {
            let mut challenger = create_challenger();
            black_box(get_challenge_stir_queries_original(
                black_box(domain_size),
                black_box(folding_factor),
                black_box(num_queries),
                black_box(&mut challenger),
            ))
        })
    });

    group.bench_function("optimized", |b| {
        b.iter(|| {
            let mut challenger = create_challenger();
            black_box(
                get_challenge_stir_queries::<TestChallenger, F>(
                    black_box(domain_size),
                    black_box(folding_factor),
                    black_box(num_queries),
                    black_box(&mut challenger),
                )
                .unwrap(),
            )
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_challenge_queries_small,
    bench_challenge_queries_medium,
    bench_challenge_queries_large
);
criterion_main!(benches);
