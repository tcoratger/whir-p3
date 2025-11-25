use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::{DuplexChallenger, FieldChallenger, GrindingChallenger};
use p3_field::extension::BinomialExtensionField;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::{Rng, SeedableRng, rngs::SmallRng};
use whir_p3::{
    fiat_shamir::{domain_separator::DomainSeparator, prover::ProverState},
    parameters::{FoldingFactor, ProtocolParameters, errors::SecurityAssumption},
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    sumcheck::sumcheck_single::SumcheckSingle,
    whir::{
        constraints::{Constraint, statement::EqStatement},
        parameters::SumcheckOptimization,
        proof::WhirProof,
    },
};

type F = BabyBear;
type EF = BinomialExtensionField<BabyBear, 4>;
type Perm = Poseidon2BabyBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

/// Helper to create protocol parameters for benchmarking
fn create_test_protocol_params(
    folding_factor: FoldingFactor,
) -> ProtocolParameters<MyHash, MyCompress> {
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);

    ProtocolParameters {
        initial_statement: true,
        security_level: 32,
        pow_bits: 0,
        rs_domain_initial_reduction_factor: 1,
        folding_factor,
        merkle_hash: MyHash::new(perm.clone()),
        merkle_compress: MyCompress::new(perm),
        soundness_type: SecurityAssumption::UniqueDecoding,
        starting_log_inv_rate: 1,
        sumcheck_optimization: SumcheckOptimization::Classic,
    }
}

/// Helper to create a fresh domain separator and challenger for each benchmark iteration.
fn setup_domsep_and_challenger() -> (DomainSeparator<EF, F>, MyChallenger) {
    let mut rng = SmallRng::seed_from_u64(0);
    let perm = Perm::new_from_rng_128(&mut rng);
    let challenger = MyChallenger::new(perm);
    let domsep = DomainSeparator::new(vec![]);
    (domsep, challenger)
}

/// Helper to generate a random multilinear polynomial.
fn generate_poly(num_vars: usize) -> EvaluationsList<F> {
    let mut rng = SmallRng::seed_from_u64(1);
    let evals = (0..1 << num_vars).map(|_| rng.random()).collect();
    EvaluationsList::new(evals)
}

/// Helper to generate an initial statement with a few constraints.
fn generate_statement<C>(
    prover: &mut ProverState<F, EF, C>,
    num_vars: usize,
    poly: &EvaluationsList<F>,
    num_constraints: usize,
) -> EqStatement<EF>
where
    C: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    let mut statement = EqStatement::initialize(num_vars);
    for _ in 0..num_constraints {
        let point = MultilinearPoint::expand_from_univariate(prover.sample(), num_vars);
        statement.add_unevaluated_constraint_hypercube(point, poly);
    }
    statement
}

/// Main benchmark function to test the sumcheck prover.
fn bench_sumcheck_prover(c: &mut Criterion) {
    let mut group = c.benchmark_group("SumcheckProver");
    // Use a smaller sample size for long-running benchmarks
    group.sample_size(10);

    // Define the range of variable counts to benchmark.
    for num_vars in &[16, 18, 20, 22, 24] {
        // Generate a large polynomial to use for this set of benchmarks.
        let poly = generate_poly(*num_vars);

        // Benchmark for the classic, round-by-round sumcheck
        let classic_folding_schedule = [*num_vars / 2, num_vars - (*num_vars / 2)];

        // Create parameters with a dummy folding factor (we'll use manual schedule)
        let params = create_test_protocol_params(FoldingFactor::Constant(2));

        group.bench_with_input(BenchmarkId::new("Classic", *num_vars), &poly, |b, poly| {
            b.iter(|| {
                // Setup fresh for each iteration
                let (domsep, challenger_for_prover) = setup_domsep_and_challenger();
                let mut prover = domsep.to_prover_state(challenger_for_prover);

                // Initialize proof and challenger (refactored approach)
                let mut proof = WhirProof::<F, EF, 8>::from_protocol_parameters(&params, *num_vars);
                let mut rng = SmallRng::seed_from_u64(1);
                let mut challenger_rf = MyChallenger::new(Perm::new_from_rng_128(&mut rng));
                domsep.observe_domain_separator(&mut challenger_rf);

                // Create constraint
                let statement = generate_statement(&mut prover, *num_vars, poly, 3);
                let constraint = Constraint::new_eq_only(prover.sample(), statement);
                // Keep challenger_rf in sync
                let _alpha_rf: EF = challenger_rf.sample_algebra_element();

                // First round - fold first half of variables
                let (mut sumcheck, _) = SumcheckSingle::from_base_evals(
                    poly,
                    &mut prover,
                    &mut proof,
                    &mut challenger_rf,
                    classic_folding_schedule[0],
                    0,
                    &constraint,
                );

                // Second round - fold remaining variables
                if classic_folding_schedule.len() > 1 && classic_folding_schedule[1] > 0 {
                    sumcheck.compute_sumcheck_polynomials(
                        &mut prover,
                        &mut proof,
                        &mut challenger_rf,
                        classic_folding_schedule[1],
                        0,
                        true, // final round
                        None,
                    );
                }
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_sumcheck_prover);
criterion_main!(benches);
