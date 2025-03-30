use whir_p3::{
    parameters::{FoldType, FoldingFactor, SoundnessType},
    whir::make_whir_things,
};

fn main() {
    let num_variables = 6;
    let folding_factor = FoldingFactor::Constant(2);
    let num_points = 1;
    let soundness_type = SoundnessType::ProvableList;
    let pow_bits = 0;
    let fold_type = FoldType::Naive;

    make_whir_things(
        num_variables,
        folding_factor,
        num_points,
        soundness_type,
        pow_bits,
        fold_type,
    );
}
