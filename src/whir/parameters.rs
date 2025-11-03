use std::{f64::consts::LOG2_10, marker::PhantomData};

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field, TwoAdicField};

use crate::parameters::{FoldingFactor, ProtocolParameters, errors::SecurityAssumption};

#[derive(Debug, Clone)]
pub struct RoundConfig<F> {
    pub pow_bits: usize,
    pub folding_pow_bits: usize,
    pub num_queries: usize,
    pub ood_samples: usize,
    pub log_inv_rate: usize,
    pub num_variables: usize,
    pub folding_factor: usize,
    pub domain_size: usize,
    pub folded_domain_gen: F,
}

#[derive(Debug, Clone)]
pub struct WhirConfig<EF, F, Hash, C, Challenger>
where
    F: Field,
    EF: ExtensionField<F>,
{
    pub num_variables: usize,
    pub soundness_type: SecurityAssumption,
    pub security_level: usize,
    pub max_pow_bits: usize,

    pub commitment_ood_samples: usize,
    // The WHIR protocol can prove either:
    // 1. The commitment is a valid low degree polynomial. In that case, the initial statement is
    //    set to false.
    // 2. The commitment is a valid folded polynomial, and an additional polynomial evaluation
    //    statement. In that case, the initial statement is set to true.
    pub initial_statement: bool,
    pub starting_log_inv_rate: usize,
    pub starting_folding_pow_bits: usize,

    pub folding_factor: FoldingFactor,
    pub rs_domain_initial_reduction_factor: usize,
    pub round_parameters: Vec<RoundConfig<F>>,

    pub final_queries: usize,
    pub final_pow_bits: usize,
    pub final_log_inv_rate: usize,
    pub final_sumcheck_rounds: usize,
    pub final_folding_pow_bits: usize,

    // Merkle tree parameters
    pub merkle_hash: Hash,
    pub merkle_compress: C,

    // Univariate skip optimization
    pub univariate_skip: bool,

    pub _base_field: PhantomData<F>,
    pub _extension_field: PhantomData<EF>,
    pub _challenger: PhantomData<Challenger>,
}

impl<EF, F, Hash, C, Challenger> WhirConfig<EF, F, Hash, C, Challenger>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    #[allow(clippy::too_many_lines)]
    pub fn new(num_variables: usize, whir_parameters: ProtocolParameters<Hash, C>) -> Self {
        // We need to store the initial number of variables for the final composition.
        let initial_num_variables = num_variables;
        whir_parameters
            .folding_factor
            .check_validity(num_variables)
            .unwrap();

        assert!(
            whir_parameters.rs_domain_initial_reduction_factor
                <= whir_parameters.folding_factor.at_round(0),
            "Increasing the code rate is not a good idea"
        );

        let protocol_security_level = whir_parameters
            .security_level
            .saturating_sub(whir_parameters.pow_bits);
        let field_size_bits = EF::bits();
        let mut log_inv_rate = whir_parameters.starting_log_inv_rate;
        let mut num_variables = num_variables;

        let log_domain_size = num_variables + log_inv_rate;
        let mut domain_size: usize = 1 << log_domain_size;

        // We could theorically tolerate a bigger `log_folded_domain_size` (up to EF::TWO_ADICITY), but this would reduce performance:
        // 1) Because the FFT twiddle factors would be in the Extension Field
        // 2) Because all the equality polynomials from WHIR queries would be in the Extension Field
        //
        // Note that this does not restrict the amount of data committed, as long as folding_factor_0 > EF::TWO_ADICITY - F::TWO_ADICITY
        let log_folded_domain_size = log_domain_size - whir_parameters.folding_factor.at_round(0);
        assert!(
            log_folded_domain_size <= F::TWO_ADICITY,
            "Increase folding_factor_0"
        );

        let (num_rounds, final_sumcheck_rounds) = whir_parameters
            .folding_factor
            .compute_number_of_rounds(num_variables);

        let commitment_ood_samples = if whir_parameters.initial_statement {
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
            let rs_reduction_factor = if round == 0 {
                whir_parameters.rs_domain_initial_reduction_factor
            } else {
                1
            };
            let next_rate = log_inv_rate
                + (whir_parameters.folding_factor.at_round(round) - rs_reduction_factor);

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
            let folding_factor = whir_parameters.folding_factor.at_round(round);
            let next_folding_factor = whir_parameters.folding_factor.at_round(round + 1);
            let folded_domain_gen =
                F::two_adic_generator(domain_size.ilog2() as usize - folding_factor);

            round_parameters.push(RoundConfig {
                pow_bits: pow_bits as usize,
                folding_pow_bits: folding_pow_bits as usize,
                num_queries,
                ood_samples,
                log_inv_rate,
                num_variables,
                folding_factor,
                domain_size,
                folded_domain_gen,
            });

            num_variables -= next_folding_factor;
            log_inv_rate = next_rate;
            domain_size >>= rs_reduction_factor;
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
            commitment_ood_samples,
            num_variables: initial_num_variables,
            soundness_type: whir_parameters.soundness_type,
            starting_log_inv_rate: whir_parameters.starting_log_inv_rate,
            starting_folding_pow_bits: starting_folding_pow_bits as usize,
            folding_factor: whir_parameters.folding_factor,
            rs_domain_initial_reduction_factor: whir_parameters.rs_domain_initial_reduction_factor,
            round_parameters,
            final_queries,
            final_pow_bits: final_pow_bits as usize,
            final_sumcheck_rounds,
            final_folding_pow_bits: final_folding_pow_bits as usize,
            final_log_inv_rate: log_inv_rate,
            merkle_hash: whir_parameters.merkle_hash,
            merkle_compress: whir_parameters.merkle_compress,
            univariate_skip: whir_parameters.univariate_skip,
            _base_field: PhantomData,
            _extension_field: PhantomData,
            _challenger: PhantomData,
        }
    }

    /// Returns the size of the initial evaluation domain.
    ///
    /// This is the size of the domain used to evaluate the original multilinear polynomial
    /// before any folding or reduction steps are applied in the WHIR protocol.
    ///
    /// It is computed as:
    ///
    /// \begin{equation}
    /// 2^{\text{num\_variables} + \text{starting\_log\_inv\_rate}}
    /// \end{equation}
    ///
    /// - `num_variables` is the number of variables in the original multivariate polynomial.
    /// - `starting_log_inv_rate` is the initial inverse rate of the Reed–Solomon code,
    ///   controlling redundancy relative to the degree.
    ///
    /// # Returns
    /// A power-of-two value representing the number of evaluation points in the starting domain.
    pub const fn starting_domain_size(&self) -> usize {
        1 << (self.num_variables + self.starting_log_inv_rate)
    }

    pub const fn n_rounds(&self) -> usize {
        self.round_parameters.len()
    }

    pub const fn rs_reduction_factor(&self, round: usize) -> usize {
        if round == 0 {
            self.rs_domain_initial_reduction_factor
        } else {
            1
        }
    }

    pub fn log_inv_rate_at(&self, round: usize) -> usize {
        let mut res = self.starting_log_inv_rate;
        for r in 0..round {
            res += self.folding_factor.at_round(r);
            res -= self.rs_reduction_factor(r);
        }
        res
    }

    pub fn merkle_tree_height(&self, round: usize) -> usize {
        self.log_inv_rate_at(round) + self.num_variables - self.folding_factor.total_number(round)
    }

    pub const fn n_vars_of_final_polynomial(&self) -> usize {
        self.num_variables - self.folding_factor.total_number(self.n_rounds())
    }

    /// Returns the log2 size of the largest FFT
    /// (At commitment we perform 2^folding_factor FFT of size 2^max_fft_size)
    pub const fn max_fft_size(&self) -> usize {
        self.num_variables + self.starting_log_inv_rate - self.folding_factor.at_round(0)
    }

    pub fn check_pow_bits(&self) -> bool {
        let max_bits = self.max_pow_bits;

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

    /// Compute the synthetic or derived `RoundConfig` for the final phase.
    ///
    /// - If no folding rounds were configured, constructs a fallback config
    ///   based on the starting domain and folding factor.
    /// - If rounds were configured, derives the final config by adapting
    ///   the last round’s values for the final folding phase.
    ///
    /// This is used by the verifier when verifying the final polynomial,
    /// ensuring consistent challenge selection and STIR constraint handling.
    pub fn final_round_config(&self) -> RoundConfig<F> {
        if self.round_parameters.is_empty() {
            // Fallback: no folding rounds, use initial domain setup
            RoundConfig {
                num_variables: self.num_variables - self.folding_factor.at_round(0),
                folding_factor: self.folding_factor.at_round(self.n_rounds()),
                num_queries: self.final_queries,
                pow_bits: self.final_pow_bits,
                domain_size: self.starting_domain_size(),
                folded_domain_gen: F::two_adic_generator(
                    self.starting_domain_size().ilog2() as usize - self.folding_factor.at_round(0),
                ),
                ood_samples: 0, // no OOD in synthetic final phase
                folding_pow_bits: self.final_folding_pow_bits,
                log_inv_rate: self.starting_log_inv_rate,
            }
        } else {
            let rs_reduction_factor = self.rs_reduction_factor(self.n_rounds() - 1);
            let folding_factor = self.folding_factor.at_round(self.n_rounds());

            // Derive final round config from last round, adjusted for next fold
            let last = self.round_parameters.last().unwrap();
            let domain_size = last.domain_size >> rs_reduction_factor;
            let folded_domain_gen = F::two_adic_generator(
                domain_size.ilog2() as usize - self.folding_factor.at_round(self.n_rounds()),
            );

            RoundConfig {
                num_variables: last.num_variables - folding_factor,
                folding_factor,
                num_queries: self.final_queries,
                pow_bits: self.final_pow_bits,
                domain_size,
                folded_domain_gen,
                ood_samples: last.ood_samples,
                folding_pow_bits: self.final_folding_pow_bits,
                log_inv_rate: last.log_inv_rate,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_field::PrimeCharacteristicRing;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};

    use super::*;

    type F = BabyBear;
    type Poseidon2Compression<Perm16> = TruncatedPermutation<Perm16, 2, 8, 16>;
    type Poseidon2Sponge<Perm24> = PaddingFreeSponge<Perm24, 24, 16, 8>;
    type Perm = Poseidon2BabyBear<16>;
    type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

    /// Generates default WHIR parameters
    const fn default_whir_params()
    -> ProtocolParameters<Poseidon2Sponge<u8>, Poseidon2Compression<u8>> {
        ProtocolParameters {
            initial_statement: true,
            security_level: 100,
            pow_bits: 20,
            rs_domain_initial_reduction_factor: 1,
            folding_factor: FoldingFactor::ConstantFromSecondRound(4, 4),
            merkle_hash: Poseidon2Sponge::new(44), // Just a placeholder
            merkle_compress: Poseidon2Compression::new(55), // Just a placeholder
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: 1,
            univariate_skip: false,
        }
    }

    #[test]
    fn test_whir_config_creation() {
        let params = default_whir_params();

        let config =
            WhirConfig::<F, F, Poseidon2Sponge<u8>, Poseidon2Compression<u8>, MyChallenger>::new(
                10, params,
            );

        assert_eq!(config.security_level, 100);
        assert_eq!(config.max_pow_bits, 20);
        assert_eq!(config.soundness_type, SecurityAssumption::CapacityBound);
        assert!(config.initial_statement);
    }

    #[test]
    fn test_n_rounds() {
        let params = default_whir_params();
        let config =
            WhirConfig::<F, F, Poseidon2Sponge<u8>, Poseidon2Compression<u8>, MyChallenger>::new(
                10, params,
            );

        assert_eq!(config.n_rounds(), config.round_parameters.len());
    }

    #[test]
    fn test_folding_pow_bits() {
        let field_size_bits = 64;
        let soundness = SecurityAssumption::CapacityBound;

        let pow_bits = WhirConfig::<F, F, u8, u8, MyChallenger>::folding_pow_bits(
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
        let mut config =
            WhirConfig::<F, F, Poseidon2Sponge<u8>, Poseidon2Compression<u8>, MyChallenger>::new(
                10, params,
            );

        // Set all values within limits
        config.max_pow_bits = 20;
        config.starting_folding_pow_bits = 15;
        config.final_pow_bits = 18;
        config.final_folding_pow_bits = 19;

        // Ensure all rounds are within limits
        config.round_parameters = vec![
            RoundConfig {
                pow_bits: 17,
                folding_pow_bits: 19,
                num_queries: 5,
                ood_samples: 2,
                log_inv_rate: 3,
                num_variables: 10,
                folding_factor: 2,
                domain_size: 10,
                folded_domain_gen: F::from_u64(2),
            },
            RoundConfig {
                pow_bits: 18,
                folding_pow_bits: 19,
                num_queries: 6,
                ood_samples: 2,
                log_inv_rate: 4,
                num_variables: 10,
                folding_factor: 2,
                domain_size: 10,
                folded_domain_gen: F::from_u64(2),
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
        let mut config =
            WhirConfig::<F, F, Poseidon2Sponge<u8>, Poseidon2Compression<u8>, MyChallenger>::new(
                10, params,
            );

        config.max_pow_bits = 20;
        config.starting_folding_pow_bits = 21; // Exceeds max_pow_bits
        config.final_pow_bits = 18;
        config.final_folding_pow_bits = 19;

        assert!(
            !config.check_pow_bits(),
            "Starting folding pow bits exceeds max_pow_bits, should return false."
        );
    }

    #[test]
    fn test_check_pow_bits_final_pow_exceeds() {
        let params = default_whir_params();
        let mut config =
            WhirConfig::<F, F, Poseidon2Sponge<u8>, Poseidon2Compression<u8>, MyChallenger>::new(
                10, params,
            );

        config.max_pow_bits = 20;
        config.starting_folding_pow_bits = 15;
        config.final_pow_bits = 21; // Exceeds max_pow_bits
        config.final_folding_pow_bits = 19;

        assert!(
            !config.check_pow_bits(),
            "Final pow bits exceeds max_pow_bits, should return false."
        );
    }

    #[test]
    fn test_check_pow_bits_round_pow_exceeds() {
        let params = default_whir_params();
        let mut config =
            WhirConfig::<F, F, Poseidon2Sponge<u8>, Poseidon2Compression<u8>, MyChallenger>::new(
                10, params,
            );

        config.max_pow_bits = 20;
        config.starting_folding_pow_bits = 15;
        config.final_pow_bits = 18;
        config.final_folding_pow_bits = 19;

        // One round's pow_bits exceeds limit
        config.round_parameters = vec![RoundConfig {
            pow_bits: 21, // Exceeds max_pow_bits
            folding_pow_bits: 19,
            num_queries: 5,
            ood_samples: 2,
            log_inv_rate: 3,
            num_variables: 10,
            folding_factor: 2,
            domain_size: 10,
            folded_domain_gen: F::from_u64(2),
        }];

        assert!(
            !config.check_pow_bits(),
            "A round has pow_bits exceeding max_pow_bits, should return false."
        );
    }

    #[test]
    fn test_check_pow_bits_round_folding_pow_exceeds() {
        let params = default_whir_params();
        let mut config =
            WhirConfig::<F, F, Poseidon2Sponge<u8>, Poseidon2Compression<u8>, MyChallenger>::new(
                10, params,
            );

        config.max_pow_bits = 20;
        config.starting_folding_pow_bits = 15;
        config.final_pow_bits = 18;
        config.final_folding_pow_bits = 19;

        // One round's folding_pow_bits exceeds limit
        config.round_parameters = vec![RoundConfig {
            pow_bits: 19,
            folding_pow_bits: 21, // Exceeds max_pow_bits
            num_queries: 5,
            ood_samples: 2,
            log_inv_rate: 3,
            num_variables: 10,
            folding_factor: 2,
            domain_size: 10,
            folded_domain_gen: F::from_u64(2),
        }];

        assert!(
            !config.check_pow_bits(),
            "A round has folding_pow_bits exceeding max_pow_bits, should return false."
        );
    }

    #[test]
    fn test_check_pow_bits_exactly_at_limit() {
        let params = default_whir_params();
        let mut config =
            WhirConfig::<F, F, Poseidon2Sponge<u8>, Poseidon2Compression<u8>, MyChallenger>::new(
                10, params,
            );

        config.max_pow_bits = 20;
        config.starting_folding_pow_bits = 20;
        config.final_pow_bits = 20;
        config.final_folding_pow_bits = 20;

        config.round_parameters = vec![RoundConfig {
            pow_bits: 20,
            folding_pow_bits: 20,
            num_queries: 5,
            ood_samples: 2,
            log_inv_rate: 3,
            num_variables: 10,
            folding_factor: 2,
            domain_size: 10,
            folded_domain_gen: F::from_u64(2),
        }];

        assert!(
            config.check_pow_bits(),
            "All pow_bits are exactly at max_pow_bits, should return true."
        );
    }

    #[test]
    fn test_check_pow_bits_all_exceed() {
        let params = default_whir_params();
        let mut config =
            WhirConfig::<F, F, Poseidon2Sponge<u8>, Poseidon2Compression<u8>, MyChallenger>::new(
                10, params,
            );

        config.max_pow_bits = 20;
        config.starting_folding_pow_bits = 22;
        config.final_pow_bits = 23;
        config.final_folding_pow_bits = 24;

        config.round_parameters = vec![RoundConfig {
            pow_bits: 25,
            folding_pow_bits: 26,
            num_queries: 5,
            ood_samples: 2,
            log_inv_rate: 3,
            num_variables: 10,
            folding_factor: 2,
            domain_size: 10,
            folded_domain_gen: F::from_u64(2),
        }];

        assert!(
            !config.check_pow_bits(),
            "All values exceed max_pow_bits, should return false."
        );
    }
}
