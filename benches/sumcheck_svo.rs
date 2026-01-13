use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_challenger::{DuplexChallenger, FieldChallenger, GrindingChallenger};
use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::{Rng, SeedableRng, rngs::SmallRng};
use whir::{
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    sumcheck::sumcheck_single::SumcheckSingle,
    whir::constraints::statement::EqStatement,
};
use whir_p3::{
    self as whir,
    fiat_shamir::domain_separator::DomainSeparator,
    parameters::{FoldingFactor, ProtocolParameters, errors::SecurityAssumption},
    whir::{
        constraints::Constraint,
        parameters::InitialPhaseConfig,
        proof::{InitialPhase, SumcheckData, WhirProof},
    },
};

type F = KoalaBear;
type EF = BinomialExtensionField<F, 8>;
type Poseidon16 = Poseidon2KoalaBear<16>;
type MyHash = PaddingFreeSponge<Poseidon16, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Poseidon16, 2, 8, 16>;
type MyChallenger = DuplexChallenger<F, Poseidon16, 16, 8>;

/// Helper to create protocol parameters for benchmarking
fn create_test_protocol_params_classic(
    folding_factor: FoldingFactor,
) -> ProtocolParameters<MyHash, MyCompress> {
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Poseidon16::new_from_rng_128(&mut rng);

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

fn setup_challenger() -> MyChallenger {
    let mut rng = SmallRng::seed_from_u64(0);
    let poseidon = Poseidon16::new_from_rng_128(&mut rng);
    MyChallenger::new(poseidon)
}

fn generate_poly(num_vars: usize) -> EvaluationsList<F> {
    let mut rng = SmallRng::seed_from_u64(1 + num_vars as u64);
    EvaluationsList::new((0..1 << num_vars).map(|_| rng.random()).collect())
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

fn bench_sumcheck_prover_svo(c: &mut Criterion) {
    let mut group = c.benchmark_group("SumcheckProver");
    // Use a smaller sample size for long-running benchmarks
    group.sample_size(10);

    // Setup domain separator
    let domsep: DomainSeparator<EF, F> = DomainSeparator::new(vec![]);

    // Define the range of variable counts to benchmark.
    for num_vars in &[16, 18, 20, 22, 24] {
        // Generate a large polynomial to use for this set of benchmarks.
        let poly = generate_poly(*num_vars);

        // Classic benchmark - folding all variables in one round
        let params_classic =
            create_test_protocol_params_classic(FoldingFactor::Constant(*num_vars));
        group.bench_with_input(BenchmarkId::new("Classic", *num_vars), &poly, |b, poly| {
            b.iter(|| {
                // Setup fresh challenger for each iteration
                let mut challenger = setup_challenger();
                domsep.observe_domain_separator(&mut challenger);

                // Initialize proof
                let mut proof =
                    WhirProof::<F, EF, 8>::from_protocol_parameters(&params_classic, *num_vars);

                // Create constraint using challenger directly
                let statement = generate_statement(&mut challenger, *num_vars, poly, 3);
                let alpha: EF = challenger.sample_algebra_element();
                let constraint = Constraint::new_eq_only(alpha, statement);

                // Extract sumcheck data from the initial phase
                let InitialPhase::WithStatement { ref mut sumcheck } = proof.initial_phase else {
                    panic!("Expected WithStatement variant");
                };

                // Fold all variables in one round
                SumcheckSingle::from_base_evals(
                    poly,
                    sumcheck,
                    &mut challenger,
                    *num_vars,
                    0,
                    &constraint,
                );
            });
        });

        // SVO benchmark - using SVO optimization
        group.bench_with_input(BenchmarkId::new("SVO", *num_vars), &poly, |b, poly| {
            b.iter(|| {
                // Setup fresh challenger for each iteration
                let mut challenger = setup_challenger();
                domsep.observe_domain_separator(&mut challenger);

                // Create constraint using challenger directly
                let statement = generate_statement(&mut challenger, *num_vars, poly, 1);
                let alpha: EF = challenger.sample_algebra_element();
                let constraint = Constraint::new_eq_only(alpha, statement);

                // Create sumcheck data
                let mut sumcheck_data: SumcheckData<EF, F> = SumcheckData::default();

                // Fold all variables using SVO optimization
                SumcheckSingle::from_base_evals_svo(
                    poly,
                    &mut sumcheck_data,
                    &mut challenger,
                    *num_vars,
                    0,
                    &constraint,
                );
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_sumcheck_prover_svo);
criterion_main!(benches);
