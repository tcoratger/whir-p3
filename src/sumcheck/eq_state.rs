//! Stateful evaluator for the equality polynomial in an SVO-based sumcheck.
//!
//! This module implements the state management described as Algorithm 6 in
//! “Speeding Up Sum-Check Proving” (eprint.iacr.org/2025/1117).
//!
//! The main goal is to provide a single, coherent object that tracks all state
//! needed to evaluate the equality polynomial as the sumcheck protocol progresses.
//! It combines two ideas:
//!
//! 1) Incremental tracking of the scalar
//!
//!    At round i, we need the prefix equality value
//!
//!    \begin{equation}
//!    \text{eq}(w_{0..i}, r_{0..i}).
//!    \end{equation}
//!
//!    This is the product of all equality checks for the variables that have
//!    already been bound. Instead of recomputing this from scratch every time,
//!    it is updated in place in constant time whenever a new challenge is bound.
//!
//! 2) Precomputed equality tables for the SVO (Algorithm 5) phase
//!
//!    Once we enter the SVO phase, we repeatedly evaluate polynomials t_i(X)
//!    that depend on equality polynomials over the remaining variables on the
//!    left and right halves. All required tables are precomputed during
//!    initialization and stored as stacks. Each round consumes (pops) the
//!    appropriate table, so there is no need to index or recompute them inside
//!    the main sumcheck loop.
//!
//! Throughout its lifetime, the object maintains a set of simple invariants:
//!
//! - `bound_count = i` is the current round index (0-based).
//! - `current_scalar = eq(w[0..i], r[0..i])` is the cumulative equality
//!   scalar after binding the first i variables.
//! - While `i < l/2`, the top of `eq_l_stack` corresponds to the current round.
//! - While `i ≥ l/2`, the top of `eq_tail_stack` corresponds to the current round.

use alloc::vec::Vec;

use p3_field::Field;

use crate::poly::{evals::EvaluationsList, multilinear::MultilinearPoint};

/// Stateful manager for all equality-polynomial data in an SVO-based sumcheck.
///
/// The object is parameterized by:
///
/// - A witness point w of length l.
/// - A starting round index `START_ROUND` (const generic) that marks when the SVO (Algorithm 5)
///   phase begins.
///
/// It then exposes:
///
/// - An incremental view of the linear equality polynomial at each round.
/// - A sequence of precomputed tables used to evaluate the t_i(X) polynomials
///   in the SVO phase.
/// - The final scalar eq(w, r) once all variables have been bound.
#[derive(Debug)]
pub struct SumcheckEqState<'a, F: Field, const START_ROUND: usize> {
    /// Witness point w with l Boolean coordinates.
    ///
    /// This is the point whose equality with the verifier challenges r controls
    /// the equality polynomial. The ith coordinate is denoted w_i.
    w: &'a MultilinearPoint<F>,

    /// Cumulative equality scalar for the prefix of already bound variables.
    ///
    /// After i rounds have been bound, this is
    ///
    /// \begin{equation}
    ///     \text{current_scalar} = \text{eq}(w_{0..i}, r_{0..i}).
    /// \end{equation}
    ///
    /// It is updated in place in each call to `bind`.
    current_scalar: F,

    /// Number of variables that have been bound so far.
    ///
    /// If `bound_count = i`, then rounds 0, 1, …, i−1 are already processed,
    /// and round i is the next one to be handled.
    bound_count: usize,

    /// Precomputed equality table on the right half of variables.
    ///
    /// This stores evaluations of
    ///
    /// \begin{equation}
    ///     \text{eq}(w_{l/2..l}, x_R)
    /// \end{equation}
    ///
    /// seen as a multilinear polynomial in the right-half variables x_R.
    ///
    /// It is shared across all rounds in the first SVO sub-phase
    /// (rounds in `[start_round, l/2)`).
    eq_r: EvaluationsList<F>,

    /// Stack of precomputed left-side equality tables for the SVO phase.
    ///
    /// For each round i in the range `[start_round, l/2)`, there is a table
    /// with evaluations of
    ///
    /// \begin{equation}
    ///     \text{eq}(w_{i+1..l/2}, x_L).
    /// \end{equation}
    ///
    /// The tables are stored so that `eq_l_stack.last()` corresponds to the
    /// current round.
    eq_l_stack: Vec<EvaluationsList<F>>,

    /// Stack of precomputed tail equality tables for the later SVO phase.
    ///
    /// For each round i in `[l/2, l)`, there is a table with evaluations of
    ///
    /// \begin{equation}
    ///     \text{eq}(w_{i+1..l}, x_{\text{tail}}).
    /// \end{equation}
    eq_tail_stack: Vec<EvaluationsList<F>>,

    /// Half the total number of variables, l / 2.
    ///
    /// This is the split point between the left and right halves of the witness.
    half_l: usize,

    /// Number of variables on the right side, that is, in x_R.
    ///
    /// This value is derived from the length of `eq_r`.
    num_vars_x_r: usize,
}

impl<'a, F: Field, const START_ROUND: usize> SumcheckEqState<'a, F, START_ROUND> {
    /// Constructs a new equality-polynomial state and precomputes all tables.
    ///
    /// The constructor performs the entire setup phase for Algorithm 5:
    ///
    /// - It splits the witness into left and right halves at index `l / 2`.
    /// - It precomputes:
    ///   - The right-side equality table eq(w_{l/2..l}, x_R).
    ///   - All left-side tables eq(w_{i+1..l/2}, x_L) needed for rounds
    ///     `START_ROUND` through `l/2 − 1`.
    ///   - All tail tables eq(w_{i+1..l}, x_tail) needed for rounds `l/2` through `l − 1`.
    ///
    /// Once this is done, the main sumcheck loop does not need to recompute
    /// or index into these tables; it only asks for the current table and
    /// then advances the state via `bind`.
    ///
    /// Arguments:
    ///
    /// - `w`: the witness point with l Boolean coordinates.
    ///
    /// The starting round index `START_ROUND` is provided as a const generic parameter.
    pub fn new(w: &'a MultilinearPoint<F>) -> Self {
        let num_vars = w.num_variables();
        let half_l = num_vars / 2;

        // Precomputation for t_i(X) tables.

        // Right-side table: eq(w_{l/2..l}, x_R).
        //
        // This is a multilinear equality polynomial in the x_R variables,
        // evaluated over the full hypercube of x_R.
        let eq_r = EvaluationsList::new_from_point(&w.0[half_l..], F::ONE);
        let num_vars_x_r = eq_r.num_variables();

        // Left-side tables for rounds in [START_ROUND, l/2).
        //
        // For round i in this range, we need eq(w_{i+1..half_l}, x_L).
        let mut eq_l_stack: Vec<_> = (START_ROUND..half_l)
            .map(|i| {
                let round = i + 1;
                EvaluationsList::new_from_point(&w.0[round..half_l], F::ONE)
            })
            .collect();

        // The tables are generated in increasing order of i;
        //
        // We reverse the list so that the last element corresponds to the
        // current round and can be popped.
        eq_l_stack.reverse();

        // Tail tables for rounds in [l/2, l).
        //
        // For round i in this range, we need eq(w_{i+1..l}, x_tail).
        let mut eq_tail_stack: Vec<_> = (half_l..num_vars)
            .map(|i| {
                let round = i + 1;
                EvaluationsList::new_from_point(&w.0[round..], F::ONE)
            })
            .collect();

        // Same idea: reverse so the top of the stack matches the current round.
        eq_tail_stack.reverse();

        Self {
            w,
            current_scalar: F::ONE, // eq of an empty prefix is 1
            bound_count: 0,         // no variables bound yet
            eq_r,
            eq_l_stack,
            eq_tail_stack,
            half_l,
            num_vars_x_r,
        }
    }

    /// Binds the next witness coordinate to a verifier challenge.
    ///
    /// This method advances the sumcheck by one round. It performs two updates:
    ///
    /// 1) It incorporates the new challenge r into the incremental equality scalar:
    ///
    ///    after this call, the scalar represents eq(w_{0..i+1}, r_{0..i+1}).
    ///
    /// 2) If we are in the SVO phase (Algorithm 5), it discards the tables that
    ///    were used for the round that just finished by popping from the
    ///    relevant stack. This keeps the stacks and `bound_count` in sync.
    ///
    /// The round index used here is the current `bound_count`. Concretely:
    ///
    /// - Before the call, `bound_count = i` and the next variable is w_i.
    /// - After the call:
    ///   - `bound_count = i + 1`,
    ///   - the scalar equals eq(w_{0..i+1}, r_{0..i+1}),
    ///   - and the table for round i has been popped if we were in the SVO phase.
    ///
    /// Panics:
    ///
    /// - If all variables are already bound and we attempt to bind one more.
    #[inline]
    pub fn bind(&mut self, r: F) {
        assert!(
            !self.is_fully_bound(),
            "Cannot bind more challenges: already bound {}/{}",
            self.bound_count,
            self.w.num_variables()
        );

        // Index of the current round.
        let i = self.bound_count;

        // Witness value w_i for this round.
        let w_i = self.w.0[i];

        // Update the incremental equality scalar.
        //
        // The binary equality polynomial for a single coordinate is
        //
        //   eq(w_i, r) = (1 - w_i)(1 - r) + w_i r
        //
        // which can be written as
        //
        //   1 - w_i - r + 2 w_i r
        //
        // for efficiency. We multiply the existing scalar by this new factor.
        self.current_scalar *= F::ONE - w_i - r + (w_i * r).double();

        // If we are at or past the start of the SVO phase, we have just finished
        // a round that used some precomputed table. That table belongs to the
        // round with index i and must now be discarded.
        if i >= START_ROUND {
            if i < self.half_l {
                // Rounds in [START_ROUND, l/2): tables live in eq_l_stack.
                let _ = self
                    .eq_l_stack
                    .pop()
                    .expect("eq_l_stack is empty, logic error in SVO phase");
            } else if i < self.w.num_variables() - 1 {
                // Rounds in [l/2, l-1): tables live in eq_tail_stack.
                //
                // For the very last round, there is no table to pop.
                let _ = self
                    .eq_tail_stack
                    .pop()
                    .expect("eq_tail_stack is empty, logic error in SVO phase");
            }
        }

        // Move on to the next round.
        self.bound_count += 1;
    }

    /// Returns the evaluations of the current linear equality polynomial.
    ///
    /// At round i, before binding w_i, we consider the polynomial
    ///
    /// \begin{equation}
    ///     l_i(X) = \text{eq}(w_{0..i}, r_{0..i}) \cdot \text{eq}(w_i, X).
    /// \end{equation}
    ///
    /// The method returns the pair $[l_i(0), l_i(1)]$.
    ///
    /// These are computed using:
    ///
    /// - eq(w_i, 0) = 1 − w_i,
    /// - eq(w_i, 1) = w_i.
    ///
    /// Panics:
    ///
    /// - If all variables have already been bound, since there is no next
    ///   variable to build a linear polynomial from.
    #[inline]
    #[must_use]
    pub fn current_linear_evals(&self) -> [F; 2] {
        assert!(
            !self.is_fully_bound(),
            "No more variables to evaluate: all {}/{} variables bound",
            self.bound_count,
            self.w.num_variables()
        );

        // Index of the next variable to be bound.
        let i = self.bound_count;

        // Witness value for this variable.
        let w_i = self.w.0[i];

        // l_i(0) = current_scalar * eq(w_i, 0) = current_scalar * (1 - w_i)
        let l_0 = self.current_scalar * (F::ONE - w_i);

        // l_i(1) = current_scalar * eq(w_i, 1) = current_scalar * w_i
        let l_1 = self.current_scalar * w_i;

        [l_0, l_1]
    }

    /// Returns the precomputed tables needed to evaluate t_i(X) in the current round.
    ///
    /// This is only meaningful once the SVO phase has begun, that is,
    /// when `bound_count >= START_ROUND`.
    ///
    /// The returned pair `(eq_left, eq_right)` depends on the current round i:
    ///
    /// - For rounds i in `[START_ROUND, l/2)`:
    ///   - `eq_left` is the top of the `eq_l_stack`, corresponding to eq(w_{i+1..l/2}, x_L).
    ///   - `eq_right` is the fixed `eq_r` table, corresponding to eq(w_{l/2..l}, x_R).
    ///
    /// - For rounds i in `[l/2, l)`:
    ///   - `eq_left` is the top of the `eq_tail_stack`, corresponding to eq(w_{i+1..l}, x_tail).
    ///   - `eq_right` is an empty slice, since only the tail table is used.
    ///
    /// The method provides these slices directly, without requiring the caller
    /// to compute indices or reason about stack offsets.
    ///
    /// Panics:
    ///
    /// - If called before `START_ROUND`, since no SVO tables are in use yet.
    /// - If all variables have been bound.
    #[inline]
    #[must_use]
    pub fn current_t_poly_tables(&self) -> (&[F], &[F]) {
        assert!(
            self.bound_count >= START_ROUND,
            "Not in Algorithm 5 (SVO) phase yet: bound_count={}, start_round={}",
            self.bound_count,
            START_ROUND
        );
        assert!(
            !self.is_fully_bound(),
            "All variables bound: no more tables available"
        );

        if self.bound_count < self.half_l {
            // Early SVO rounds: use a left table from eq_l_stack and the fixed right table eq_r.
            let eq_l = self
                .eq_l_stack
                .last()
                .expect("eq_l_stack is empty, logic error")
                .as_slice();
            (eq_l, self.eq_r.as_slice())
        } else {
            // Later SVO rounds: use only a tail table from eq_tail_stack.
            let eq_tail = self
                .eq_tail_stack
                .last()
                .expect("eq_tail_stack is empty, logic error")
                .as_slice();
            (eq_tail, &[])
        }
    }

    /// Returns the number of variables in the right-hand block x_R.
    #[inline]
    #[must_use]
    pub const fn num_vars_x_r(&self) -> usize {
        self.num_vars_x_r
    }

    /// Returns l / 2, the split point between left and right halves of w.
    #[inline]
    #[must_use]
    pub const fn half_l(&self) -> usize {
        self.half_l
    }

    /// Returns the final equality scalar eq(w, r) once all variables are bound.
    ///
    /// This is only valid when the sumcheck has finished.
    ///
    /// Panics:
    ///
    /// - If called before all variables are bound.
    #[inline]
    #[must_use]
    pub const fn final_scalar(&self) -> F {
        assert!(self.is_fully_bound(), "Not fully bound");
        self.current_scalar
    }

    /// Returns true if all variables of the witness have been bound.
    #[inline]
    #[must_use]
    pub const fn is_fully_bound(&self) -> bool {
        self.bound_count == self.w.num_variables()
    }

    /// Returns the total number of variables in the witness.
    #[inline]
    #[must_use]
    pub const fn num_variables(&self) -> usize {
        self.w.num_variables()
    }

    /// Returns the number of variables bound so far.
    ///
    /// This is the current round index i:
    /// - rounds 0 through i − 1 are in the past,
    /// - round i is the next one to process.
    #[inline]
    #[must_use]
    pub const fn bound_count(&self) -> usize {
        self.bound_count
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    use super::*;

    type F = BabyBear;

    #[test]
    fn test_initialization_precomputes_all_tables() {
        const START_ROUND: usize = 3;

        // Witness point coordinates
        let w0 = F::from_u64(2);
        let w1 = F::from_u64(3);
        let w2 = F::from_u64(5);
        let w3 = F::from_u64(7);
        let w4 = F::from_u64(11);
        let w5 = F::from_u64(13);
        let w6 = F::from_u64(17);
        let w7 = F::from_u64(19);

        // Create a witness point with 8 variables
        //
        // This allows us to test both phases: rounds [3, 4) and [4, 8)
        let w = MultilinearPoint::new(vec![w0, w1, w2, w3, w4, w5, w6, w7]);

        let eq_state = SumcheckEqState::<_, START_ROUND>::new(&w);

        // Initial state: no variables bound yet
        assert_eq!(eq_state.bound_count(), 0);
        assert!(!eq_state.is_fully_bound());

        // Verify the scalar is 1 by checking linear evals: l(0) + l(1) = scalar
        let [l_0, l_1] = eq_state.current_linear_evals();
        assert_eq!(l_0 + l_1, F::ONE); // Confirms scalar = 1

        // Verify metadata
        assert_eq!(eq_state.num_variables(), 8);
        assert_eq!(eq_state.half_l(), 4); // l/2 = 8/2 = 4

        // The right table eq_R = eq(w[4..8], x_R) should have 2^4 = 16 entries
        assert_eq!(eq_state.num_vars_x_r(), 4);

        // Left tables: one for each round in [3, 4) = just round 3
        // So we should have 1 table in the eq_l_stack
        assert_eq!(eq_state.eq_l_stack.len(), 1);

        // Tail tables: one for each round in [4, 8) = rounds 4,5,6,7
        // So we should have 4 tables in the eq_tail_stack
        assert_eq!(eq_state.eq_tail_stack.len(), 4);
    }

    #[test]
    fn test_bind_updates_scalar_incrementally() {
        // Test that bind() correctly maintains the incremental scalar
        // by comparing against the reference implementation

        // Witness point coordinates
        let w0 = F::from_u64(2);
        let w1 = F::from_u64(3);
        let w2 = F::from_u64(5);
        let w3 = F::from_u64(7);
        let w = MultilinearPoint::new(vec![w0, w1, w2, w3]);

        let mut eq_state = SumcheckEqState::<_, 3>::new(&w);

        // Challenge values
        let r0 = F::from_u64(11);
        let r1 = F::from_u64(13);
        let r2 = F::from_u64(17);
        let challenges = [r0, r1, r2];

        for (i, &r) in challenges.iter().enumerate() {
            eq_state.bind(r);
            assert_eq!(eq_state.bound_count(), i + 1);

            // After binding i+1 variables, the scalar should equal eq(w[..i+1], r[..i+1])
            // We verify this by checking the linear evals sum: l(0) + l(1) = scalar
            if i + 1 < w.num_variables() {
                let [l_0, l_1] = eq_state.current_linear_evals();
                let computed_scalar = l_0 + l_1;
                let expected = MultilinearPoint::eval_eq(&w.0[..=i], &challenges[..=i]);
                assert_eq!(
                    computed_scalar,
                    expected,
                    "Scalar mismatch after binding {} variables",
                    i + 1
                );
            }
        }
    }

    #[test]
    fn test_linear_evals_before_algorithm_5() {
        // Before we reach start_round, we only use the incremental scalar
        //
        // The linear polynomial is l_i(X) = current_scalar * eq(w_i, X)

        // Witness point coordinates
        let w0 = F::from_u64(2);
        let w1 = F::from_u64(3);
        let w2 = F::from_u64(5);
        let w3 = F::from_u64(7);
        let w = MultilinearPoint::new(vec![w0, w1, w2, w3]);

        let mut eq_state = SumcheckEqState::<_, 3>::new(&w);

        // Round 0: scalar = 1 (no variables bound yet)
        // l_0(X) = 1 * eq(w0, X) = eq(w0, X)
        // l_0(0) = eq(w0, 0) = 1 - w0
        // l_0(1) = eq(w0, 1) = w0
        let [l_0, l_1] = eq_state.current_linear_evals();
        assert_eq!(l_0, F::ONE - w0);
        assert_eq!(l_1, w0);

        // Challenge value
        let r0 = F::from_u64(11);

        // Bind first variable with challenge r0
        eq_state.bind(r0);

        // Round 1: scalar = eq(w0, r0)
        // l_1(X) = scalar * eq(w1, X)
        let [l_0, l_1] = eq_state.current_linear_evals();

        // Key property: l_0 + l_1 = scalar (since eq(w1, 0) + eq(w1, 1) = 1)
        let scalar = l_0 + l_1;

        // Verify the relationship between linear evals and scalar
        // l_0 = scalar * eq(w1, 0) = scalar * (1 - w1)
        // l_1 = scalar * eq(w1, 1) = scalar * w1
        assert_eq!(l_0, scalar * (F::ONE - w1));
        assert_eq!(l_1, scalar * w1);
    }

    #[test]
    fn test_table_access_first_half_rounds() {
        const START_ROUND: usize = 2;

        // Test accessing precomputed tables for rounds in [start_round, l/2)
        //
        // These rounds use both eq_L (from stack) and eq_R (fixed)

        // Witness point coordinates
        let w0 = F::from_u64(2);
        let w1 = F::from_u64(3);
        let w2 = F::from_u64(5);
        let w3 = F::from_u64(7);
        let w4 = F::from_u64(11);
        let w5 = F::from_u64(13);
        let w = MultilinearPoint::new(vec![w0, w1, w2, w3, w4, w5]);

        let mut eq_state = SumcheckEqState::<_, START_ROUND>::new(&w);

        // Challenge values
        let r0 = F::from_u64(17);
        let r1 = F::from_u64(19);

        // Bind first two variables to reach round 2 (the start of Algorithm 5)
        eq_state.bind(r0);
        eq_state.bind(r1);

        // Now bound_count = 2, so we're at round 2 (within [start_round, l/2) = [2, 3))
        assert_eq!(eq_state.bound_count(), START_ROUND);

        // Get tables for current round
        let (eq_l, eq_r) = eq_state.current_t_poly_tables();

        // eq_R is the fixed table eq(w[3..6], x_R) with 2^3 = 8 entries
        assert_eq!(eq_r.len(), 8);

        // eq_L for round 2 should be eq(w[3..3], x_L) = eq([], x_L) = just [1]
        // (since we need eq(w[round..half_l], x_L) = eq(w[3..3], x_L))
        assert_eq!(eq_l.len(), 1);
        assert_eq!(eq_l[0], F::ONE);

        // Verify eq_R is correctly computed
        let expected_eq_r = EvaluationsList::new_from_point(&w.0[3..6], F::ONE);
        assert_eq!(eq_r, expected_eq_r.as_slice());
    }

    #[test]
    fn test_table_access_second_half_rounds() {
        const START_ROUND: usize = 2;

        // Test accessing precomputed tables for rounds in [l/2, l)
        //
        // These rounds use only eq_tail (from stack), eq_R is empty

        // Witness point coordinates
        let w0 = F::from_u64(2);
        let w1 = F::from_u64(3);
        let w2 = F::from_u64(5);
        let w3 = F::from_u64(7);
        let w4 = F::from_u64(11);
        let w5 = F::from_u64(13);
        let w = MultilinearPoint::new(vec![w0, w1, w2, w3, w4, w5]);

        let mut eq_state = SumcheckEqState::<_, START_ROUND>::new(&w);

        // Challenge values
        let r0 = F::from_u64(17);
        let r1 = F::from_u64(19);
        let r2 = F::from_u64(23);

        // Bind variables to reach round 3 (which is l/2 = 3)
        eq_state.bind(r0);
        eq_state.bind(r1);
        eq_state.bind(r2);

        // Now bound_count = 3, so we're at round 3 (which is l/2)
        assert_eq!(eq_state.bound_count(), 3);
        assert_eq!(eq_state.bound_count(), eq_state.half_l());

        // Get tables for current round
        let (eq_tail, eq_r) = eq_state.current_t_poly_tables();

        // In second half, eq_R should be empty
        assert_eq!(eq_r.len(), 0);

        // eq_tail for round 3 should be eq(w[4..6], x_tail) with 2^2 = 4 entries
        assert_eq!(eq_tail.len(), 4);

        // Verify eq_tail is correctly computed
        let expected_eq_tail = EvaluationsList::new_from_point(&w.0[4..6], F::ONE);
        assert_eq!(eq_tail, expected_eq_tail.as_slice());
    }

    #[test]
    fn test_bind_pops_tables_in_algorithm_5_phase() {
        const START_ROUND: usize = 2;

        // Verify the "throw away" behavior: bind() should pop tables as they're used

        // Witness point coordinates
        let w0 = F::from_u64(2);
        let w1 = F::from_u64(3);
        let w2 = F::from_u64(5);
        let w3 = F::from_u64(7);
        let w = MultilinearPoint::new(vec![w0, w1, w2, w3]);

        let mut eq_state = SumcheckEqState::<_, START_ROUND>::new(&w);

        // Challenge values
        let r0 = F::from_u64(11);
        let r1 = F::from_u64(13);
        let r2 = F::from_u64(17);
        let r3 = F::from_u64(19);

        // Initially, we should have:
        // - eq_l_stack with tables for rounds [2, 2) = empty (since half_l = 2)
        // - eq_tail_stack with tables for rounds [2, 4) = rounds 2, 3
        assert_eq!(eq_state.eq_l_stack.len(), 0);
        assert_eq!(eq_state.eq_tail_stack.len(), 2);

        // Bind first two variables to reach the Algorithm 5 phase
        eq_state.bind(r0);
        eq_state.bind(r1);

        // Now we're at round 2, which is at half_l
        // We should still have both tail tables
        assert_eq!(eq_state.eq_tail_stack.len(), 2);

        // Access tables (doesn't consume them)
        let _ = eq_state.current_t_poly_tables();
        assert_eq!(eq_state.eq_tail_stack.len(), 2);

        // Bind the next variable - this should pop the first tail table
        eq_state.bind(r2);
        assert_eq!(eq_state.eq_tail_stack.len(), 1);

        // Bind the last variable - this should NOT pop (special case for last round)
        eq_state.bind(r3);
        assert_eq!(eq_state.eq_tail_stack.len(), 1); // Still 1, not popped
        assert!(eq_state.is_fully_bound());
    }

    #[test]
    fn test_final_scalar_after_full_binding() {
        // Verify that after binding all variables, final_scalar() returns eq(w, r)

        // Witness point coordinates
        let w0 = F::from_u64(2);
        let w1 = F::from_u64(3);
        let w2 = F::from_u64(5);
        let w = MultilinearPoint::new(vec![w0, w1, w2]);

        let mut eq_state = SumcheckEqState::<_, 2>::new(&w);

        // Challenge values
        let r0 = F::from_u64(7);
        let r1 = F::from_u64(11);
        let r2 = F::from_u64(13);
        let challenges = vec![r0, r1, r2];

        // Bind all variables
        for &r in &challenges {
            eq_state.bind(r);
        }

        assert!(eq_state.is_fully_bound());

        // The final scalar should equal eq(w, r) computed directly
        let expected = MultilinearPoint::eval_eq(&w.0, &challenges);
        assert_eq!(eq_state.final_scalar(), expected);
    }

    #[test]
    fn test_stack_ordering_first_half() {
        const START_ROUND: usize = 2;

        // Verify that stacks are in the correct order (reversed for popping)

        // For a 6-variable witness with start_round = 2:
        // - Rounds in [2, 3): round 2
        // - We need eq_L for round 2, which is eq(w[3..3], x_L)

        // Witness point coordinates
        let w0 = F::from_u64(2);
        let w1 = F::from_u64(3);
        let w2 = F::from_u64(5);
        let w3 = F::from_u64(7);
        let w4 = F::from_u64(11);
        let w5 = F::from_u64(13);
        let w = MultilinearPoint::new(vec![w0, w1, w2, w3, w4, w5]);

        let mut eq_state = SumcheckEqState::<_, START_ROUND>::new(&w);

        // Challenge values
        let r0 = F::from_u64(17);
        let r1 = F::from_u64(19);
        let r2 = F::from_u64(23);

        // Stack should have 1 element (for round 2)
        assert_eq!(eq_state.eq_l_stack.len(), 1);

        // Bind to reach round 2
        eq_state.bind(r0);
        eq_state.bind(r1);

        // Get the table (should be the last element due to reversal)
        let (eq_l, _) = eq_state.current_t_poly_tables();

        // This should be eq(w[3..3], x_L) = eq([], x_L) = [1]
        assert_eq!(eq_l.len(), 1);
        assert_eq!(eq_l[0], F::ONE);

        // After binding r2, the stack should be popped
        eq_state.bind(r2);
        assert_eq!(eq_state.eq_l_stack.len(), 0);
    }

    #[test]
    fn test_stack_ordering_second_half() {
        const START_ROUND: usize = 2;

        // Verify tail stack ordering is correct

        // Witness point coordinates
        let w0 = F::from_u64(2);
        let w1 = F::from_u64(3);
        let w2 = F::from_u64(5);
        let w3 = F::from_u64(7);
        let w = MultilinearPoint::new(vec![w0, w1, w2, w3]);

        let mut eq_state = SumcheckEqState::<_, START_ROUND>::new(&w);

        // Challenge values
        let r0 = F::from_u64(11);
        let r1 = F::from_u64(13);
        let r2 = F::from_u64(17);

        // Tail stack should have 2 elements (for rounds 2 and 3)
        assert_eq!(eq_state.eq_tail_stack.len(), 2);

        // Bind to reach round 2 (which is half_l)
        eq_state.bind(r0);
        eq_state.bind(r1);

        // First table access at round 2
        let (eq_tail_1, _) = eq_state.current_t_poly_tables();
        // Should be eq(w[3..4], x_tail) = eq([w3], x_tail) with 2^1 = 2 entries
        assert_eq!(eq_tail_1.len(), 2);

        // Bind r2 and move to round 3
        eq_state.bind(r2);
        assert_eq!(eq_state.eq_tail_stack.len(), 1); // Popped one

        // Second table access at round 3
        let (eq_tail_2, _) = eq_state.current_t_poly_tables();
        // Should be eq(w[4..4], x_tail) = eq([], x_tail) = [1]
        assert_eq!(eq_tail_2.len(), 1);
        assert_eq!(eq_tail_2[0], F::ONE);
    }

    #[test]
    fn test_edge_case_small_start_round() {
        const START_ROUND: usize = 0;

        // Test with start_round = 0 (Algorithm 5 from the beginning)

        // Witness point coordinates
        let w0 = F::from_u64(2);
        let w1 = F::from_u64(3);
        let w = MultilinearPoint::new(vec![w0, w1]);

        let eq_state = SumcheckEqState::<_, START_ROUND>::new(&w);

        // With 2 variables and start_round = 0:
        // - half_l = 1
        // - Rounds [0, 1): round 0 (first half)
        // - Rounds [1, 2): round 1 (second half)
        assert_eq!(eq_state.eq_l_stack.len(), 1);
        assert_eq!(eq_state.eq_tail_stack.len(), 1);
    }

    #[test]
    fn test_edge_case_start_round_at_half() {
        const START_ROUND: usize = 2; // half_l = 2

        // Test with start_round = l/2 (Algorithm 5 starts at the second half)

        // Witness point coordinates
        let w0 = F::from_u64(2);
        let w1 = F::from_u64(3);
        let w2 = F::from_u64(5);
        let w3 = F::from_u64(7);
        let w = MultilinearPoint::new(vec![w0, w1, w2, w3]);

        let eq_state = SumcheckEqState::<_, START_ROUND>::new(&w);

        // No first-half rounds in Algorithm 5 phase
        assert_eq!(eq_state.eq_l_stack.len(), 0);
        // Second-half rounds: [2, 4)
        assert_eq!(eq_state.eq_tail_stack.len(), 2);
    }

    #[test]
    fn test_correctness_against_manual_computation() {
        const START_ROUND: usize = 2;

        // End-to-end test verifying all components work together correctly
        // We'll manually compute what the tables should contain and verify

        // Witness point coordinates
        let w0 = F::from_u64(2);
        let w1 = F::from_u64(3);
        let w2 = F::from_u64(5);
        let w3 = F::from_u64(7);
        let w = MultilinearPoint::new(vec![w0, w1, w2, w3]);

        let mut eq_state = SumcheckEqState::<_, START_ROUND>::new(&w);

        // Challenge values
        let r0 = F::from_u64(11);
        let r1 = F::from_u64(13);

        // Bind to round 2
        eq_state.bind(r0);
        eq_state.bind(r1);

        // At round 2 (which is half_l = 2), we should use eq_tail
        let (eq_tail, eq_r) = eq_state.current_t_poly_tables();
        assert!(eq_r.is_empty()); // Second half

        // Manually compute what eq_tail should be: eq(w[3..4], x_tail) = eq([w3], x_tail)
        // For x_tail ∈ {0, 1}:
        // eq([w3], [0]) = (1 - w3)(1 - 0) + w3*0 = 1 - w3
        // eq([w3], [1]) = (1 - w3)(1 - 1) + w3*1 = w3
        let expected_0 = F::ONE - w3;
        let expected_1 = w3;
        assert_eq!(eq_tail[0], expected_0);
        assert_eq!(eq_tail[1], expected_1);

        // Verify linear evals are consistent
        // At round 2, the next variable to bind is w2
        let [l_0, l_1] = eq_state.current_linear_evals();
        let scalar = l_0 + l_1; // Recover scalar from linear evals

        // l_0 = scalar * eq(w2, 0) = scalar * (1 - w2)
        // l_1 = scalar * eq(w2, 1) = scalar * w2
        assert_eq!(l_0, scalar * (F::ONE - w2));
        assert_eq!(l_1, scalar * w2);
    }

    #[test]
    #[should_panic(expected = "Cannot bind more challenges")]
    fn test_panic_on_binding_too_many() {
        // Witness point
        let w0 = F::from_u64(2);
        let w = MultilinearPoint::new(vec![w0]);
        let mut eq_state = SumcheckEqState::<_, 0>::new(&w);

        // Challenge values
        let r0 = F::from_u64(3);
        let r1 = F::from_u64(5);

        eq_state.bind(r0); // First bind: ok
        eq_state.bind(r1); // Second bind: should panic
    }

    #[test]
    #[should_panic(expected = "No more variables to evaluate")]
    fn test_panic_on_linear_evals_when_fully_bound() {
        // Witness point
        let w0 = F::from_u64(2);
        let w = MultilinearPoint::new(vec![w0]);
        let mut eq_state = SumcheckEqState::<_, 0>::new(&w);

        // Challenge value
        let r0 = F::from_u64(3);

        eq_state.bind(r0);
        assert!(eq_state.is_fully_bound());

        // Trying to get linear evals when fully bound should panic
        let _ = eq_state.current_linear_evals();
    }

    #[test]
    #[should_panic(expected = "Not in Algorithm 5")]
    fn test_panic_on_table_access_before_start_round() {
        const START_ROUND: usize = 2;

        // Witness point coordinates
        let w0 = F::from_u64(2);
        let w1 = F::from_u64(3);
        let w2 = F::from_u64(5);
        let w3 = F::from_u64(7);
        let w = MultilinearPoint::new(vec![w0, w1, w2, w3]);

        let eq_state = SumcheckEqState::<_, START_ROUND>::new(&w);

        // We're at bound_count = 0, which is before start_round = 2
        // Attempting to access tables should panic
        let _ = eq_state.current_t_poly_tables();
    }

    #[test]
    #[should_panic(expected = "Not fully bound")]
    fn test_panic_on_final_scalar_before_fully_bound() {
        // Witness point coordinates
        let w0 = F::from_u64(2);
        let w1 = F::from_u64(3);
        let w = MultilinearPoint::new(vec![w0, w1]);
        let eq_state = SumcheckEqState::<_, 0>::new(&w);

        // Trying to get final_scalar before binding all variables should panic
        let _ = eq_state.final_scalar();
    }
}
