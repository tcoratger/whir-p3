use crate::constant::K_SKIP_SUMCHECK;

/// STIR protocol configuration for WHIR proximity testing.
///
/// Controls folding behavior and optimizations during STIR query processing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StirConfig {
    /// Whether this is the initial statement round.
    ///
    /// Enables univariate skip optimization.
    pub initial_statement: bool,

    /// Whether to apply univariate skip optimization.
    ///
    /// Reduces sumcheck rounds by K_SKIP_SUMCHECK.
    pub univariate_skip: bool,

    /// Number of variables to fold per STIR iteration.
    ///
    /// Must be >= K_SKIP_SUMCHECK for univariate skip.
    pub folding_factor_at_round: usize,
}

impl StirConfig {
    /// Creates a new STIR configuration with conservative default values.
    ///
    /// Default configuration:
    /// - `initial_statement`: `false` - Standard non-initial rounds
    /// - `univariate_skip`: `false` - No optimization applied
    /// - `folding_factor_at_round`: `0` - No folding
    ///
    /// These defaults ensure correctness but may not be optimal for performance.
    /// Use the builder pattern for more control over configuration.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            initial_statement: false,
            univariate_skip: false,
            folding_factor_at_round: 0,
        }
    }

    /// Creates a builder for constructing an optimized `StirConfig`.
    #[must_use]
    pub const fn builder() -> StirConfigBuilder {
        StirConfigBuilder::new()
    }

    /// Determines if univariate skip optimization should be applied for the given round.
    ///
    /// The univariate skip optimization allows skipping K_SKIP_SUMCHECK variables
    /// during the sumcheck protocol when specific conditions are met.
    /// This significantly improves verifier efficiency.
    ///
    /// # Conditions for Univariate Skip
    ///
    /// All of the following must be true:
    /// 1. `initial_statement` is `true` - Must be the initial statement round
    /// 2. `round_index` is `0` - Only applies to the first round
    /// 3. `univariate_skip` is `true` - Optimization must be enabled
    /// 4. `folding_factor_at_round` >= `K_SKIP_SUMCHECK` - Sufficient folding power
    ///
    /// # Arguments
    ///
    /// * `round_index` - The current STIR round index (0-based)
    ///
    /// # Returns
    ///
    /// `true` if univariate skip should be applied, `false` otherwise
    #[must_use]
    pub const fn should_apply_univariate_skip(&self, round_index: usize) -> bool {
        self.initial_statement
            && round_index == 0
            && self.univariate_skip
            && self.folding_factor_at_round >= K_SKIP_SUMCHECK
    }
}

impl Default for StirConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for constructing [`StirConfig`] instances with method chaining.
///
/// Provides a fluent interface for configuring STIR protocol parameters.
/// Enables precise control over optimizations and folding behavior.
///
/// # Usage Pattern
///
/// ```ignore
/// let config = StirConfig::builder()
///     .initial_statement(true)          // Enable initial statement mode
///     .univariate_skip(true)            // Enable univariate skip optimization
///     .folding_factor_at_round(k)       // Set folding parameter
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct StirConfigBuilder {
    /// Internal state for initial_statement configuration
    initial_statement: bool,
    /// Internal state for univariate_skip configuration
    univariate_skip: bool,
    /// Internal state for folding_factor_at_round configuration
    folding_factor_at_round: usize,
}

impl StirConfigBuilder {
    /// Creates a new builder with conservative default values.
    ///
    /// Starts with safe defaults that ensure correctness but may not be optimal.
    #[must_use]
    const fn new() -> Self {
        Self {
            initial_statement: false,
            univariate_skip: false,
            folding_factor_at_round: 0,
        }
    }

    /// Configures whether this represents an initial statement round.
    ///
    /// The initial statement round has special properties in WHIR that enable
    /// the univariate skip optimization. Set to `true` for the first round
    /// of STIR processing when the prover is establishing the initial claim.
    ///
    /// # Arguments
    ///
    /// * `initial_statement` - `true` for initial statement rounds, `false` otherwise
    ///
    /// # Returns
    ///
    /// Self for method chaining
    #[must_use]
    pub const fn with_initial_statement(mut self, initial_statement: bool) -> Self {
        self.initial_statement = initial_statement;
        self
    }

    /// Configures whether to enable univariate skip optimization.
    ///
    /// The univariate skip optimization reduces the number of sumcheck rounds
    /// by K_SKIP_SUMCHECK when specific conditions are met. This significantly
    /// improves verifier performance but requires careful parameter tuning.
    ///
    /// Only effective when combined with `initial_statement(true)` and sufficient
    /// `folding_factor_at_round`.
    ///
    /// # Arguments
    ///
    /// * `univariate_skip` - `true` to enable optimization, `false` for standard mode
    ///
    /// # Returns
    ///
    /// Self for method chaining
    #[must_use]
    pub const fn with_univariate_skip(mut self, univariate_skip: bool) -> Self {
        self.univariate_skip = univariate_skip;
        self
    }

    /// Sets the folding factor for the current round.
    ///
    /// The folding factor `k` determines how many variables are reduced in each
    /// STIR iteration. Higher values reduce round complexity but increase query
    /// alphabet size from F to F^(2^k).
    ///
    /// Must be >= K_SKIP_SUMCHECK to enable univariate skip optimization.
    /// Typical values are 2-4 for optimal performance.
    ///
    /// # Arguments
    ///
    /// * `folding_factor_at_round` - Number of variables to fold per round
    ///
    /// # Returns
    ///
    /// Self for method chaining
    #[must_use]
    pub const fn with_folding_factor_at_round(mut self, folding_factor_at_round: usize) -> Self {
        self.folding_factor_at_round = folding_factor_at_round;
        self
    }

    /// Builds the final [`StirConfig`] with the configured parameters.
    ///
    /// Consumes the builder and produces a fully configured STIR configuration
    /// that can be used with the WHIR proximity testing protocol.
    ///
    /// # Returns
    ///
    /// A configured `StirConfig` instance ready for use
    #[must_use]
    pub const fn build(self) -> StirConfig {
        StirConfig {
            initial_statement: self.initial_statement,
            univariate_skip: self.univariate_skip,
            folding_factor_at_round: self.folding_factor_at_round,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = StirConfig::default();
        assert!(!config.initial_statement);
        assert!(!config.univariate_skip);
        assert_eq!(config.folding_factor_at_round, 0);
    }

    #[test]
    fn test_builder_pattern() {
        let config = StirConfig::builder()
            .with_initial_statement(true)
            .with_univariate_skip(true)
            .with_folding_factor_at_round(4)
            .build();

        assert!(config.initial_statement);
        assert!(config.univariate_skip);
        assert_eq!(config.folding_factor_at_round, 4);
    }

    #[test]
    fn test_should_apply_univariate_skip() {
        let config = StirConfig::builder()
            .with_initial_statement(true)
            .with_univariate_skip(true)
            .with_folding_factor_at_round(K_SKIP_SUMCHECK)
            .build();

        assert!(config.should_apply_univariate_skip(0));
        assert!(!config.should_apply_univariate_skip(1));

        let config_no_skip = StirConfig::builder()
            .with_initial_statement(true)
            .with_univariate_skip(false)
            .with_folding_factor_at_round(K_SKIP_SUMCHECK)
            .build();

        assert!(!config_no_skip.should_apply_univariate_skip(0));
    }
}
