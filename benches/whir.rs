use criterion::{Criterion, criterion_group, criterion_main};
use whir_p3::{
    parameters::{FoldingFactor, errors::SecurityAssumption},
    whir::make_whir_things,
};

fn benchmark_whir(c: &mut Criterion) {
    let polynomial_sizes = [16, 18, 20, 22];
    let folding_factor = FoldingFactor::Constant(4);
    let num_points = 2;
    let soundness_type = SecurityAssumption::UniqueDecoding;
    let pow_bits = 10;
    let rs_domain_initial_reduction_factor = 1;

    for num_variables in polynomial_sizes {
        c.bench_function(&format!("whir_end_to_end_size {num_variables}"), |b| {
            b.iter(|| {
                make_whir_things(
                    num_variables,
                    folding_factor,
                    num_points,
                    soundness_type,
                    pow_bits,
                    rs_domain_initial_reduction_factor,
                );
            });
        });
    }
}

criterion_group!(benches, benchmark_whir);
criterion_main!(benches);
