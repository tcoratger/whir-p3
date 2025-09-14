use crate::constant::K_SKIP_SUMCHECK;

/// Configuration for STIR proof handling operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StirConfig {
    /// Whether this is an initial statement round
    pub initial_statement: bool,
    /// Whether to apply univariate skip optimization
    pub univariate_skip: bool,
    /// The folding factor for the current round
    pub folding_factor_at_round: usize,
}

impl StirConfig {
    /// Creates a new configuration with default values.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let config = StirConfig::new();
    /// ```
    #[must_use]
    pub const fn new() -> Self {
        Self {
            initial_statement: false,
            univariate_skip: false,
            folding_factor_at_round: 0,
        }
    }

    /// Creates a builder for constructing a `StirConfig`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let config = StirConfig::builder()
    ///     .initial_statement(true)
    ///     .build();
    /// ```
    #[must_use]
    pub const fn builder() -> StirConfigBuilder {
        StirConfigBuilder::new()
    }

    /// Determines if univariate skip should be applied for the given round.
    ///
    /// This encapsulates the complex logic for determining when the univariate
    /// skip optimization can be safely applied.
    ///
    /// # Arguments
    ///
    /// * `round_index` - The current round index (0-based)
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

/// Builder for constructing [`StirConfig`] instances.
#[derive(Debug, Clone)]
pub struct StirConfigBuilder {
    initial_statement: bool,
    univariate_skip: bool,
    folding_factor_at_round: usize,
}

impl StirConfigBuilder {
    /// Creates a new builder with default values.
    #[must_use]
    const fn new() -> Self {
        Self {
            initial_statement: false,
            univariate_skip: false,
            folding_factor_at_round: 0,
        }
    }

    /// Sets whether this is an initial statement round.
    ///
    /// # Arguments
    ///
    /// * `initial_statement` - Whether this is an initial statement round
    #[must_use]
    pub const fn initial_statement(mut self, initial_statement: bool) -> Self {
        self.initial_statement = initial_statement;
        self
    }

    /// Sets whether to apply univariate skip optimization.
    ///
    /// # Arguments
    ///
    /// * `univariate_skip` - Whether to apply univariate skip
    #[must_use]
    pub const fn univariate_skip(mut self, univariate_skip: bool) -> Self {
        self.univariate_skip = univariate_skip;
        self
    }

    /// Sets the folding factor for the current round.
    ///
    /// # Arguments
    ///
    /// * `folding_factor_at_round` - The folding factor for the current round
    #[must_use]
    pub const fn folding_factor_at_round(mut self, folding_factor_at_round: usize) -> Self {
        self.folding_factor_at_round = folding_factor_at_round;
        self
    }

    /// Builds the final [`StirConfig`].
    ///
    /// # Returns
    ///
    /// A configured `StirConfig` instance
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
            .initial_statement(true)
            .univariate_skip(true)
            .folding_factor_at_round(4)
            .build();

        assert!(config.initial_statement);
        assert!(config.univariate_skip);
        assert_eq!(config.folding_factor_at_round, 4);
    }

    #[test]
    fn test_should_apply_univariate_skip() {
        let config = StirConfig::builder()
            .initial_statement(true)
            .univariate_skip(true)
            .folding_factor_at_round(K_SKIP_SUMCHECK)
            .build();

        assert!(config.should_apply_univariate_skip(0));
        assert!(!config.should_apply_univariate_skip(1));

        let config_no_skip = StirConfig::builder()
            .initial_statement(true)
            .univariate_skip(false)
            .folding_factor_at_round(K_SKIP_SUMCHECK)
            .build();

        assert!(!config_no_skip.should_apply_univariate_skip(0));
    }
}
