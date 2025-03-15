use crate::parameters::{FoldType, FoldingFactor, MultivariateParameters, SoundnessType};
use std::marker::PhantomData;

#[derive(Debug, Clone)]
pub struct RoundConfig {
    pub pow_bits: f64,
    pub folding_pow_bits: f64,
    pub num_queries: usize,
    pub ood_samples: usize,
    pub log_inv_rate: usize,
}

#[derive(Debug, Clone)]
pub struct WhirConfig<F, PowStrategy> {
    pub mv_parameters: MultivariateParameters<F>,
    pub soundness_type: SoundnessType,
    pub security_level: usize,
    pub max_pow_bits: usize,

    pub committment_ood_samples: usize,
    // The WHIR protocol can prove either:
    // 1. The commitment is a valid low degree polynomial. In that case, the initial statement is
    //    set to false.
    // 2. The commitment is a valid folded polynomial, and an additional polynomial evaluation
    //    statement. In that case, the initial statement is set to true.
    pub initial_statement: bool,
    // pub starting_domain: Domain<F>,
    pub starting_log_inv_rate: usize,
    pub starting_folding_pow_bits: f64,

    pub folding_factor: FoldingFactor,
    pub round_parameters: Vec<RoundConfig>,
    pub fold_optimisation: FoldType,

    pub final_queries: usize,
    pub final_pow_bits: f64,
    pub final_log_inv_rate: usize,
    pub final_sumcheck_rounds: usize,
    pub final_folding_pow_bits: f64,

    // PoW parameters
    pub pow_strategy: PhantomData<PowStrategy>,
}
