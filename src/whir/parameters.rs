use std::{f64::consts::LOG2_10, marker::PhantomData};

use p3_field::{ExtensionField, Field, TwoAdicField};

use crate::{
    domain::Domain,
    parameters::{
        FoldType, FoldingFactor, MultivariateParameters, ProtocolParameters,
        errors::SecurityAssumption,
    },
};

#[derive(Debug, Clone)]
pub struct RoundConfig {
    pub pow_bits: f64,
    pub folding_pow_bits: f64,
    pub num_queries: usize,
    pub ood_samples: usize,
    pub log_inv_rate: usize,
}

#[derive(Debug, Clone)]
pub struct WhirConfig<EF, F, H, C, PowStrategy>
where
    F: Field + TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
{
    pub mv_parameters: MultivariateParameters<EF>,
    pub soundness_type: SecurityAssumption,
    pub security_level: usize,
    pub max_pow_bits: usize,

    pub committment_ood_samples: usize,
    // The WHIR protocol can prove either:
    // 1. The commitment is a valid low degree polynomial. In that case, the initial statement is
    //    set to false.
    // 2. The commitment is a valid folded polynomial, and an additional polynomial evaluation
    //    statement. In that case, the initial statement is set to true.
    pub initial_statement: bool,
    pub starting_domain: Domain<EF, F>,
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

    // Merkle tree parameters
    pub merkle_hash: H,
    pub merkle_compress: C,
}

impl<EF, F, H, C, PowStrategy> WhirConfig<EF, F, H, C, PowStrategy>
where
    F: Field + TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
{
    #[allow(clippy::too_many_lines)]
    pub fn new(
        mv_parameters: MultivariateParameters<EF>,
        whir_parameters: ProtocolParameters<H, C>,
    ) -> Self {
        whir_parameters
            .folding_factor
            .check_validity(mv_parameters.num_variables)
            .unwrap();

        let protocol_security_level = whir_parameters
            .security_level
            .saturating_sub(whir_parameters.pow_bits);
        let field_size_bits = EF::bits();
        let mut log_inv_rate = whir_parameters.starting_log_inv_rate;
        let mut num_variables = mv_parameters.num_variables;

        let starting_domain = Domain::new(1 << mv_parameters.num_variables, log_inv_rate)
            .expect("Should have found an appropriate domain - check Field 2 adicity?");

        let (num_rounds, final_sumcheck_rounds) = whir_parameters
            .folding_factor
            .compute_number_of_rounds(mv_parameters.num_variables);

        let committment_ood_samples = if whir_parameters.initial_statement {
            whir_parameters.soundness_type.determine_ood_samples(
                whir_parameters.security_level,
                num_variables,
                log_inv_rate,
                field_size_bits,
            )
        } else {
            0
        };

        let starting_folding_pow_bits = if whir_parameters.initial_statement {
            Self::folding_pow_bits(
                whir_parameters.security_level,
                whir_parameters.soundness_type,
                field_size_bits,
                num_variables,
                log_inv_rate,
            )
        } else {
            {
                let prox_gaps_error = whir_parameters.soundness_type.prox_gaps_error(
                    num_variables,
                    log_inv_rate,
                    field_size_bits,
                    2,
                ) + (whir_parameters.folding_factor.at_round(0) as f64)
                    .log2();
                (whir_parameters.security_level as f64 - prox_gaps_error).max(0.0)
            }
        };

        let mut round_parameters = Vec::with_capacity(num_rounds);
        num_variables -= whir_parameters.folding_factor.at_round(0);
        for round in 0..num_rounds {
            // Queries are set w.r.t. to old rate, while the rest to the new rate
            let next_rate = log_inv_rate + (whir_parameters.folding_factor.at_round(round) - 1);

            let num_queries = whir_parameters
                .soundness_type
                .queries(protocol_security_level, log_inv_rate);

            let ood_samples = whir_parameters.soundness_type.determine_ood_samples(
                whir_parameters.security_level,
                num_variables,
                next_rate,
                field_size_bits,
            );

            let query_error = whir_parameters
                .soundness_type
                .queries_error(log_inv_rate, num_queries);
            let combination_error = Self::rbr_soundness_queries_combination(
                whir_parameters.soundness_type,
                field_size_bits,
                num_variables,
                next_rate,
                ood_samples,
                num_queries,
            );

            let pow_bits = 0_f64
                .max(whir_parameters.security_level as f64 - (query_error.min(combination_error)));

            let folding_pow_bits = Self::folding_pow_bits(
                whir_parameters.security_level,
                whir_parameters.soundness_type,
                field_size_bits,
                num_variables,
                next_rate,
            );

            round_parameters.push(RoundConfig {
                pow_bits,
                folding_pow_bits,
                num_queries,
                ood_samples,
                log_inv_rate,
            });

            num_variables -= whir_parameters.folding_factor.at_round(round + 1);
            log_inv_rate = next_rate;
        }

        let final_queries = whir_parameters
            .soundness_type
            .queries(protocol_security_level, log_inv_rate);

        let final_pow_bits = 0_f64.max(
            whir_parameters.security_level as f64
                - whir_parameters
                    .soundness_type
                    .queries_error(log_inv_rate, final_queries),
        );

        let final_folding_pow_bits =
            0_f64.max(whir_parameters.security_level as f64 - (field_size_bits - 1) as f64);

        Self {
            security_level: whir_parameters.security_level,
            max_pow_bits: whir_parameters.pow_bits,
            initial_statement: whir_parameters.initial_statement,
            committment_ood_samples,
            mv_parameters,
            starting_domain,
            soundness_type: whir_parameters.soundness_type,
            starting_log_inv_rate: whir_parameters.starting_log_inv_rate,
            starting_folding_pow_bits,
            folding_factor: whir_parameters.folding_factor,
            round_parameters,
            final_queries,
            final_pow_bits,
            final_sumcheck_rounds,
            final_folding_pow_bits,
            fold_optimisation: whir_parameters.fold_optimisation,
            final_log_inv_rate: log_inv_rate,
            pow_strategy: PhantomData,
            merkle_hash: whir_parameters.merkle_hash,
            merkle_compress: whir_parameters.merkle_compress,
        }
    }

    pub fn n_rounds(&self) -> usize {
        self.round_parameters.len()
    }

    pub fn check_pow_bits(&self) -> bool {
        let max_bits = self.max_pow_bits as f64;

        // Check the main pow bits values
        if self.starting_folding_pow_bits > max_bits
            || self.final_pow_bits > max_bits
            || self.final_folding_pow_bits > max_bits
        {
            return false;
        }

        // Check all round parameters
        self.round_parameters
            .iter()
            .all(|r| r.pow_bits <= max_bits && r.folding_pow_bits <= max_bits)
    }

    // Compute the proximity gaps term of the fold
    #[must_use]
    pub const fn rbr_soundness_fold_prox_gaps(
        soundness_type: SecurityAssumption,
        field_size_bits: usize,
        num_variables: usize,
        log_inv_rate: usize,
        log_eta: f64,
    ) -> f64 {
        // Recall, at each round we are only folding by two at a time
        let error = match soundness_type {
            SecurityAssumption::CapacityBound => (num_variables + log_inv_rate) as f64 - log_eta,
            SecurityAssumption::JohnsonBound => {
                LOG2_10 + 3.5 * log_inv_rate as f64 + 2. * num_variables as f64
            }
            SecurityAssumption::UniqueDecoding => (num_variables + log_inv_rate) as f64,
        };

        field_size_bits as f64 - error
    }

    #[must_use]
    pub const fn rbr_soundness_fold_sumcheck(
        soundness_type: SecurityAssumption,
        field_size_bits: usize,
        num_variables: usize,
        log_inv_rate: usize,
    ) -> f64 {
        let list_size = soundness_type.list_size_bits(num_variables, log_inv_rate);

        field_size_bits as f64 - (list_size + 1.)
    }

    #[must_use]
    pub fn folding_pow_bits(
        security_level: usize,
        soundness_type: SecurityAssumption,
        field_size_bits: usize,
        num_variables: usize,
        log_inv_rate: usize,
    ) -> f64 {
        let prox_gaps_error =
            soundness_type.prox_gaps_error(num_variables, log_inv_rate, field_size_bits, 2);
        let sumcheck_error = Self::rbr_soundness_fold_sumcheck(
            soundness_type,
            field_size_bits,
            num_variables,
            log_inv_rate,
        );

        let error = prox_gaps_error.min(sumcheck_error);

        0_f64.max(security_level as f64 - error)
    }

    #[must_use]
    pub fn rbr_soundness_queries_combination(
        soundness_type: SecurityAssumption,
        field_size_bits: usize,
        num_variables: usize,
        log_inv_rate: usize,
        ood_samples: usize,
        num_queries: usize,
    ) -> f64 {
        let list_size = soundness_type.list_size_bits(num_variables, log_inv_rate);

        let log_combination = ((ood_samples + num_queries) as f64).log2();

        field_size_bits as f64 - (log_combination + list_size + 1.)
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};

    use super::*;

    type Poseidon2Compression<Perm16> = TruncatedPermutation<Perm16, 2, 8, 16>;
    type Poseidon2Sponge<Perm24> = PaddingFreeSponge<Perm24, 24, 16, 8>;

    /// Generates default WHIR parameters
    const fn default_whir_params()
    -> ProtocolParameters<Poseidon2Sponge<u8>, Poseidon2Compression<u8>> {
        ProtocolParameters {
            initial_statement: true,
            security_level: 100,
            pow_bits: 20,
            folding_factor: FoldingFactor::ConstantFromSecondRound(4, 4),
            merkle_hash: Poseidon2Sponge::new(44), // Just a placeholder
            merkle_compress: Poseidon2Compression::new(55), // Just a placeholder
            soundness_type: SecurityAssumption::CapacityBound,
            fold_optimisation: FoldType::ProverHelps,
            starting_log_inv_rate: 1,
        }
    }

    #[test]
    fn test_whir_config_creation() {
        let params = default_whir_params();

        let mv_params = MultivariateParameters::<BabyBear>::new(10);
        let config = WhirConfig::<
            BabyBear,
            BabyBear,
            Poseidon2Sponge<u8>,
            Poseidon2Compression<u8>,
            (),
        >::new(mv_params, params);

        assert_eq!(config.security_level, 100);
        assert_eq!(config.max_pow_bits, 20);
        assert_eq!(config.soundness_type, SecurityAssumption::CapacityBound);
        assert!(config.initial_statement);
    }

    #[test]
    fn test_n_rounds() {
        let params = default_whir_params();
        let mv_params = MultivariateParameters::<BabyBear>::new(10);
        let config = WhirConfig::<
            BabyBear,
            BabyBear,
            Poseidon2Sponge<u8>,
            Poseidon2Compression<u8>,
            (),
        >::new(mv_params, params);

        assert_eq!(config.n_rounds(), config.round_parameters.len());
    }

    #[test]
    fn test_folding_pow_bits() {
        let field_size_bits = 64;
        let soundness = SecurityAssumption::CapacityBound;

        let pow_bits = WhirConfig::<BabyBear, BabyBear, u8, u8, ()>::folding_pow_bits(
            100, // Security level
            soundness,
            field_size_bits,
            10, // Number of variables
            5,  // Log inverse rate
        );

        // PoW bits should never be negative
        assert!(pow_bits >= 0.);
    }

    #[test]
    fn test_check_pow_bits_within_limits() {
        let params = default_whir_params();
        let mv_params = MultivariateParameters::<BabyBear>::new(10);
        let mut config = WhirConfig::<
            BabyBear,
            BabyBear,
            Poseidon2Sponge<u8>,
            Poseidon2Compression<u8>,
            (),
        >::new(mv_params, params);

        // Set all values within limits
        config.max_pow_bits = 20;
        config.starting_folding_pow_bits = 15.0;
        config.final_pow_bits = 18.0;
        config.final_folding_pow_bits = 19.5;

        // Ensure all rounds are within limits
        config.round_parameters = vec![
            RoundConfig {
                pow_bits: 17.0,
                folding_pow_bits: 19.0,
                num_queries: 5,
                ood_samples: 2,
                log_inv_rate: 3,
            },
            RoundConfig {
                pow_bits: 18.0,
                folding_pow_bits: 19.5,
                num_queries: 6,
                ood_samples: 2,
                log_inv_rate: 4,
            },
        ];

        assert!(
            config.check_pow_bits(),
            "All values are within limits, check_pow_bits should return true."
        );
    }

    #[test]
    fn test_check_pow_bits_starting_folding_exceeds() {
        let params = default_whir_params();
        let mv_params = MultivariateParameters::<BabyBear>::new(10);
        let mut config = WhirConfig::<
            BabyBear,
            BabyBear,
            Poseidon2Sponge<u8>,
            Poseidon2Compression<u8>,
            (),
        >::new(mv_params, params);

        config.max_pow_bits = 20;
        config.starting_folding_pow_bits = 21.0; // Exceeds max_pow_bits
        config.final_pow_bits = 18.0;
        config.final_folding_pow_bits = 19.5;

        assert!(
            !config.check_pow_bits(),
            "Starting folding pow bits exceeds max_pow_bits, should return false."
        );
    }

    #[test]
    fn test_check_pow_bits_final_pow_exceeds() {
        let params = default_whir_params();
        let mv_params = MultivariateParameters::<BabyBear>::new(10);
        let mut config = WhirConfig::<
            BabyBear,
            BabyBear,
            Poseidon2Sponge<u8>,
            Poseidon2Compression<u8>,
            (),
        >::new(mv_params, params);

        config.max_pow_bits = 20;
        config.starting_folding_pow_bits = 15.0;
        config.final_pow_bits = 21.0; // Exceeds max_pow_bits
        config.final_folding_pow_bits = 19.5;

        assert!(
            !config.check_pow_bits(),
            "Final pow bits exceeds max_pow_bits, should return false."
        );
    }

    #[test]
    fn test_check_pow_bits_round_pow_exceeds() {
        let params = default_whir_params();
        let mv_params = MultivariateParameters::<BabyBear>::new(10);
        let mut config = WhirConfig::<
            BabyBear,
            BabyBear,
            Poseidon2Sponge<u8>,
            Poseidon2Compression<u8>,
            (),
        >::new(mv_params, params);

        config.max_pow_bits = 20;
        config.starting_folding_pow_bits = 15.0;
        config.final_pow_bits = 18.0;
        config.final_folding_pow_bits = 19.5;

        // One round's pow_bits exceeds limit
        config.round_parameters = vec![RoundConfig {
            pow_bits: 21.0, // Exceeds max_pow_bits
            folding_pow_bits: 19.0,
            num_queries: 5,
            ood_samples: 2,
            log_inv_rate: 3,
        }];

        assert!(
            !config.check_pow_bits(),
            "A round has pow_bits exceeding max_pow_bits, should return false."
        );
    }

    #[test]
    fn test_check_pow_bits_round_folding_pow_exceeds() {
        let params = default_whir_params();
        let mv_params = MultivariateParameters::<BabyBear>::new(10);
        let mut config = WhirConfig::<
            BabyBear,
            BabyBear,
            Poseidon2Sponge<u8>,
            Poseidon2Compression<u8>,
            (),
        >::new(mv_params, params);

        config.max_pow_bits = 20;
        config.starting_folding_pow_bits = 15.0;
        config.final_pow_bits = 18.0;
        config.final_folding_pow_bits = 19.5;

        // One round's folding_pow_bits exceeds limit
        config.round_parameters = vec![RoundConfig {
            pow_bits: 19.0,
            folding_pow_bits: 21.0, // Exceeds max_pow_bits
            num_queries: 5,
            ood_samples: 2,
            log_inv_rate: 3,
        }];

        assert!(
            !config.check_pow_bits(),
            "A round has folding_pow_bits exceeding max_pow_bits, should return false."
        );
    }

    #[test]
    fn test_check_pow_bits_exactly_at_limit() {
        let params = default_whir_params();
        let mv_params = MultivariateParameters::<BabyBear>::new(10);
        let mut config = WhirConfig::<
            BabyBear,
            BabyBear,
            Poseidon2Sponge<u8>,
            Poseidon2Compression<u8>,
            (),
        >::new(mv_params, params);

        config.max_pow_bits = 20;
        config.starting_folding_pow_bits = 20.0;
        config.final_pow_bits = 20.0;
        config.final_folding_pow_bits = 20.0;

        config.round_parameters = vec![RoundConfig {
            pow_bits: 20.0,
            folding_pow_bits: 20.0,
            num_queries: 5,
            ood_samples: 2,
            log_inv_rate: 3,
        }];

        assert!(
            config.check_pow_bits(),
            "All pow_bits are exactly at max_pow_bits, should return true."
        );
    }

    #[test]
    fn test_check_pow_bits_all_exceed() {
        let params = default_whir_params();
        let mv_params = MultivariateParameters::<BabyBear>::new(10);
        let mut config = WhirConfig::<
            BabyBear,
            BabyBear,
            Poseidon2Sponge<u8>,
            Poseidon2Compression<u8>,
            (),
        >::new(mv_params, params);

        config.max_pow_bits = 20;
        config.starting_folding_pow_bits = 22.0;
        config.final_pow_bits = 23.0;
        config.final_folding_pow_bits = 24.0;

        config.round_parameters = vec![RoundConfig {
            pow_bits: 25.0,
            folding_pow_bits: 26.0,
            num_queries: 5,
            ood_samples: 2,
            log_inv_rate: 3,
        }];

        assert!(
            !config.check_pow_bits(),
            "All values exceed max_pow_bits, should return false."
        );
    }
}
