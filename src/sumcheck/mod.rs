pub mod sumcheck_polynomial;
pub mod sumcheck_single;
pub mod sumcheck_single_skip;
pub mod utils;

/// Number of variables skipped in the first round of the Sumcheck protocol using univariate skip.
///
/// Instead of reducing one variable per round, we reduce `K_SKIP_SUMCHECK` variables at once
/// by evaluating over a multiplicative subgroup of size `2^k`.
///
/// This optimization keeps the work in the base field and reduces the number of expensive
/// extension field rounds.
pub const K_SKIP_SUMCHECK: usize = 5;
