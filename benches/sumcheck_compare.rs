use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::{DuplexChallenger, FieldChallenger};
use p3_field::extension::BinomialExtensionField;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use whir_p3::{
    fiat_shamir::domain_separator::DomainSeparator,
    parameters::{FoldingFactor, ProtocolParameters, errors::SecurityAssumption},
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    sumcheck::sumcheck_prover::Sumcheck,
    whir::{
        constraints::statement::initial::InitialStatement,
        parameters::SumcheckStrategy,
        proof::{SumcheckData, WhirProof},
    },
};

type F = BabyBear;
type EF = BinomialExtensionField<BabyBear, 4>;
type Perm = Poseidon2BabyBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

const NUM_VARS_TO_BENCH: [usize; 5] = [16, 18, 20, 22, 24];
const NUM_CONSTRAINTS: usize = 3;
const FIRST_ROUND_FOLDING: usize = 4;

fn create_test_protocol_params(
    first_round_folding: usize,
) -> ProtocolParameters<MyHash, MyCompress> {
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);

    ProtocolParameters {
        security_level: 32,
        pow_bits: 0,
        rs_domain_initial_reduction_factor: 1,
        folding_factor: FoldingFactor::Constant(first_round_folding),
        merkle_hash: MyHash::new(perm.clone()),
        merkle_compress: MyCompress::new(perm),
        soundness_type: SecurityAssumption::UniqueDecoding,
        starting_log_inv_rate: 1,
    }
}

fn setup_challenger() -> MyChallenger {
    let mut rng = SmallRng::seed_from_u64(0);
    let perm = Perm::new_from_rng_128(&mut rng);
    MyChallenger::new(perm)
}

fn generate_poly(num_vars: usize) -> EvaluationsList<F> {
    let mut rng = SmallRng::seed_from_u64(1);
    let evals = (0..1 << num_vars).map(|_| rng.random()).collect();
    EvaluationsList::new(evals)
}

fn sample_constraint_points(num_vars: usize, num_constraints: usize) -> Vec<MultilinearPoint<EF>> {
    let domsep: DomainSeparator<EF, F> = DomainSeparator::new(vec![]);
    let mut challenger = setup_challenger();
    domsep.observe_domain_separator(&mut challenger);

    (0..num_constraints)
        .map(|_| {
            MultilinearPoint::expand_from_univariate(challenger.sample_algebra_element(), num_vars)
        })
        .collect()
}

fn build_statement(
    poly: &EvaluationsList<F>,
    points: &[MultilinearPoint<EF>],
    strategy: SumcheckStrategy,
) -> InitialStatement<F, EF> {
    let mut statement = InitialStatement::new(poly.clone(), FIRST_ROUND_FOLDING, strategy);
    for point in points {
        let _ = statement.evaluate(point);
    }
    statement
}

fn run_sumcheck(
    params: &ProtocolParameters<MyHash, MyCompress>,
    base_challenger: &MyChallenger,
    statement: &InitialStatement<F, EF>,
    num_vars: usize,
) {
    let mut challenger = base_challenger.clone();
    let mut proof = WhirProof::<F, EF, F, 8>::from_protocol_parameters(params, num_vars);

    let (mut sumcheck_prover, _) = Sumcheck::from_base_evals(
        &mut proof.initial_sumcheck,
        &mut challenger,
        FIRST_ROUND_FOLDING,
        0,
        statement,
    );

    let remaining_rounds = num_vars - FIRST_ROUND_FOLDING;
    if remaining_rounds > 0 {
        let mut sumcheck_data = SumcheckData::default();
        sumcheck_prover.compute_sumcheck_polynomials(
            &mut sumcheck_data,
            &mut challenger,
            remaining_rounds,
            0,
            None,
        );
        proof.set_final_sumcheck_data(sumcheck_data);
    }
}

fn bench_sumcheck_compare(c: &mut Criterion) {
    let mut group = c.benchmark_group("SumcheckCompare");
    group.sample_size(10);

    for num_vars in NUM_VARS_TO_BENCH {
        let params = create_test_protocol_params(FIRST_ROUND_FOLDING);
        let poly = generate_poly(num_vars);
        let points = sample_constraint_points(num_vars, NUM_CONSTRAINTS);

        let classic_statement = build_statement(&poly, &points, SumcheckStrategy::Classic);
        let svo_statement = build_statement(&poly, &points, SumcheckStrategy::Svo);

        assert_eq!(classic_statement.normalize(), svo_statement.normalize());

        let domsep: DomainSeparator<EF, F> = DomainSeparator::new(vec![]);
        let mut challenger = setup_challenger();
        domsep.observe_domain_separator(&mut challenger);

        group.bench_with_input(
            BenchmarkId::new("Classic", num_vars),
            &classic_statement,
            |b, statement| {
                b.iter(|| run_sumcheck(&params, &challenger, statement, num_vars));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Svo", num_vars),
            &svo_statement,
            |b, statement| {
                b.iter(|| run_sumcheck(&params, &challenger, statement, num_vars));
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_sumcheck_compare);
criterion_main!(benches);
