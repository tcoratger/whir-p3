use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::{DuplexChallenger, FieldChallenger, GrindingChallenger};
use p3_field::extension::BinomialExtensionField;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::{Rng, SeedableRng, rngs::SmallRng};
use whir_p3::{
    fiat_shamir::domain_separator::DomainSeparator,
    parameters::{FoldingFactor, ProtocolParameters, errors::SecurityAssumption},
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    sumcheck::sumcheck_single::SumcheckSingle,
    whir::{
        constraints::{Constraint, statement::EqStatement},
        parameters::InitialPhaseConfig,
        proof::{InitialPhase, SumcheckData, WhirProof},
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
        initial_phase_config: InitialPhaseConfig::WithStatementClassic,
        security_level: 32,
        pow_bits: 0,
        rs_domain_initial_reduction_factor: 1,
        folding_factor,
        merkle_hash: MyHash::new(perm.clone()),
        merkle_compress: MyCompress::new(perm),
        soundness_type: SecurityAssumption::UniqueDecoding,
        starting_log_inv_rate: 1,
    }
}

/// Helper to create a fresh domain separator and challenger for each benchmark iteration.
fn setup_challenger() -> MyChallenger {
    let mut rng = SmallRng::seed_from_u64(0);
    let perm = Perm::new_from_rng_128(&mut rng);
    MyChallenger::new(perm)
}

/// Helper to generate a random multilinear polynomial.
fn generate_poly(num_vars: usize) -> EvaluationsList<F> {
    let mut rng = SmallRng::seed_from_u64(1);
    let evals = (0..1 << num_vars).map(|_| rng.random()).collect();
    EvaluationsList::new(evals)
}

/// Helper to generate an initial statement with a few constraints.
fn generate_statement<C>(
    challenger: &mut C,
    num_vars: usize,
    poly: &EvaluationsList<F>,
    num_constraints: usize,
) -> EqStatement<EF>
where
    C: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    let mut statement = EqStatement::initialize(num_vars);
    for _ in 0..num_constraints {
        let point =
            MultilinearPoint::expand_from_univariate(challenger.sample_algebra_element(), num_vars);
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

        // Setup domain separator
        let domsep: DomainSeparator<EF, F> = DomainSeparator::new(vec![]);

        group.bench_with_input(BenchmarkId::new("Classic", *num_vars), &poly, |b, poly| {
            b.iter(|| {
                // Setup fresh challenger for each iteration
                let mut challenger = setup_challenger();
                domsep.observe_domain_separator(&mut challenger);

                // Initialize proof
                let mut proof =
                    WhirProof::<F, EF, F, 8>::from_protocol_parameters(&params, *num_vars);

                // Create constraint using challenger directly
                let statement = generate_statement(&mut challenger, *num_vars, poly, 3);
                let alpha: EF = challenger.sample_algebra_element();
                let constraint = Constraint::new_eq_only(alpha, statement);

                // Extract sumcheck data from the initial phase
                let InitialPhase::WithStatement { ref mut sumcheck } = proof.initial_phase else {
                    panic!("Expected WithStatement variant");
                };

                // First round - fold first half of variables
                let (mut sumcheck_prover, _) = SumcheckSingle::from_base_evals(
                    poly,
                    sumcheck,
                    &mut challenger,
                    classic_folding_schedule[0],
                    0,
                    &constraint,
                );

                // Second round - fold remaining variables
                if classic_folding_schedule.len() > 1 && classic_folding_schedule[1] > 0 {
                    let mut sumcheck_data: SumcheckData<EF, F> = SumcheckData::default();
                    sumcheck_prover.compute_sumcheck_polynomials(
                        &mut sumcheck_data,
                        &mut challenger,
                        classic_folding_schedule[1],
                        0,
                        None,
                    );
                    proof.set_final_sumcheck_data(sumcheck_data);
                }
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_sumcheck_prover);
criterion_main!(benches);
