/// Number of variables skipped in the first round of the Sumcheck protocol using univariate skip.
///
/// Instead of reducing one variable per round, we reduce `K_SKIP_SUMCHECK` variables at once
/// by evaluating over a multiplicative subgroup of size `2^k`.
///
/// This optimization keeps the work in the base field and reduces the number of expensive
/// extension field rounds.
pub const K_SKIP_SUMCHECK: usize = 5;

/// The number of variables at which the multilinear evaluation algorithm switches
/// from a recursive to a non-recursive, chunk-based strategy.
///
/// ## Rationale
///
/// The choice of algorithm for multilinear evaluation involves a trade-off:
///
/// - **Recursive Method (`5 <= n < 20`):** This approach is simple and efficient for a moderate
///   number of variables. It splits the problem in half at each step and uses `rayon::join`
///   for parallelism. However, for a very large number of variables, the deep recursion
///   stack can become a performance bottleneck.
///
/// - **Non-Recursive Method (`n >= 20`):** This method is optimized for wide parallelism on very
///   large inputs. It splits the evaluation point, precomputes basis polynomial evaluations,
///   and processes the main evaluation list in large, contiguous chunks. This avoids deep
///   recursion and utilizes memory more effectively, but has a higher setup cost.
///
/// The value `20` is an empirically chosen threshold representing the crossover point where the
/// benefits of the non-recursive strategy begin to outweigh its setup overhead.
pub const MLE_RECURSION_THRESHOLD: usize = 20;
