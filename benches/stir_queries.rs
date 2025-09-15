use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_field::extension::BinomialExtensionField;
use rand::{SeedableRng, rngs::SmallRng};
use whir_p3::{
    fiat_shamir::{domain_separator::DomainSeparator, prover::ProverState},
    whir::utils::get_challenge_stir_queries,
};

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type Perm = Poseidon2BabyBear<16>;
type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

fn create_prover_state() -> ProverState<F, EF, MyChallenger> {
    let mut rng = SmallRng::seed_from_u64(42);
    let perm = Perm::new_from_rng_128(&mut rng);
    let challenger = DuplexChallenger::new(perm);
    let domainsep = DomainSeparator::new(vec![]);
    domainsep.to_prover_state(challenger)
}

fn bench_stir_queries(c: &mut Criterion) {
    let mut group = c.benchmark_group("stir_queries");

    // Small case: Few queries, small domain
    group.bench_function("small_4_queries_512_domain", |b| {
        b.iter(|| {
            let mut prover_state = create_prover_state();
            get_challenge_stir_queries::<_, F, EF>(
                black_box(512), // domain_size
                black_box(3),   // folding_factor (domain becomes 64)
                black_box(4),   // num_queries
                black_box(&mut prover_state),
            )
            .unwrap()
        });
    });

    // Medium case: Moderate queries, medium domain
    group.bench_function("medium_16_queries_4k_domain", |b| {
        b.iter(|| {
            let mut prover_state = create_prover_state();
            get_challenge_stir_queries::<_, F, EF>(
                black_box(4096), // domain_size
                black_box(4),    // folding_factor (domain becomes 256)
                black_box(16),   // num_queries
                black_box(&mut prover_state),
            )
            .unwrap()
        });
    });

    // Large case: Many queries, large domain
    group.bench_function("large_64_queries_64k_domain", |b| {
        b.iter(|| {
            let mut prover_state = create_prover_state();
            get_challenge_stir_queries::<_, F, EF>(
                black_box(65536), // domain_size
                black_box(6),     // folding_factor (domain becomes 1024)
                black_box(64),    // num_queries
                black_box(&mut prover_state),
            )
            .unwrap()
        });
    });

    // Very large case: Extreme scenario
    group.bench_function("very_large_256_queries_1m_domain", |b| {
        b.iter(|| {
            let mut prover_state = create_prover_state();
            get_challenge_stir_queries::<_, F, EF>(
                black_box(1_048_576), // domain_size (1M)
                black_box(10),        // folding_factor (domain becomes 1024)
                black_box(256),       // num_queries
                black_box(&mut prover_state),
            )
            .unwrap()
        });
    });

    // Edge case: Many queries, tiny bits per query
    group.bench_function("edge_100_queries_tiny_bits", |b| {
        b.iter(|| {
            let mut prover_state = create_prover_state();
            get_challenge_stir_queries::<_, F, EF>(
                black_box(64),  // domain_size
                black_box(2),   // folding_factor (domain becomes 16, needs 4 bits)
                black_box(100), // num_queries (lots of queries, few bits each)
                black_box(&mut prover_state),
            )
            .unwrap()
        });
    });

    group.finish();
}

criterion_group!(benches, bench_stir_queries);
criterion_main!(benches);
