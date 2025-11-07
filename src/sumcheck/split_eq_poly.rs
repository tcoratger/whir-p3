//! Incremental equality polynomial evaluation used in sumcheck.

use p3_field::Field;

use crate::poly::multilinear::MultilinearPoint;

/// Incremental equality polynomial evaluator for sumcheck protocols.
///
/// This structure maintains a scalar value representing `eq(w[..bound_count], r[..bound_count])`
/// and updates it efficiently as new challenges are bound, avoiding full recomputation.
///
/// ## Mathematical Background
///
/// The equality polynomial `eq(w, r)` for two points w, r ∈ F^n is defined as:
/// ```text
/// eq(w, r) = ∏_{i=0}^{n-1} [(1 - w_i)(1 - r_i) + w_i * r_i]
/// ```
///
/// This can be computed incrementally using the factorization:
/// ```text
/// eq(w[..i+1], r[..i+1]) = eq(w[..i], r[..i]) * eq(w[i], r[i])
/// ```
///
/// where `eq(w[i], r[i]) = 1 - w[i] - r[i] + 2*w[i]*r[i]`
/// (using an algebraic identity to reduce multiplications).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitEqPolynomial<'a, F> {
    /// The multilinear point w.
    ///
    /// We often call this the witness point in sumcheck.
    w: &'a MultilinearPoint<F>,
    /// Current accumulated scalar: eq(w[..bound_count], r[..bound_count]).
    ///
    /// ## Invariant
    /// This equals the product:
    ///
    /// ```text
    /// ∏_{i=0}^{bound_count-1} eq(w[i], r[i]),
    /// ```
    ///
    /// where eq(a, b) = (1 - a)(1 - b) + a*b.
    current_scalar: F,
    /// Number of variables bound so far.
    ///
    /// Equivalently, this is the index of the next variable to bind.
    ///
    /// Invariant: 0 ≤ bound_count ≤ w.len()
    bound_count: usize,
}

impl<'a, F: Field> SplitEqPolynomial<'a, F> {
    /// Creates a new split equality polynomial from a multilinear point.
    ///
    /// Initially, no variables are bound, so `current_scalar = 1` (empty product).
    ///
    /// ## Arguments
    /// * `w`: The multilinear point (witness point)
    #[inline]
    #[must_use]
    pub const fn new(w: &'a MultilinearPoint<F>) -> Self {
        Self {
            w,
            current_scalar: F::ONE,
            bound_count: 0,
        }
    }

    /// Binds the next variable with challenge `r`.
    ///
    /// Updates the internal scalar incrementally:
    /// ```text
    /// current_scalar *= eq(w[bound_count], r)
    ///                = (1 - w[bound_count])(1 - r) + w[bound_count] * r
    ///                = 1 - w[bound_count] - r + 2 * w[bound_count] * r
    /// ```
    ///
    /// The last form uses an algebraic identity to reduce the number of multiplications.
    ///
    /// ## Panics
    /// Panics if all variables have already been bound (`bound_count == w.len()`).
    #[inline]
    pub fn bind(&mut self, r: F) {
        assert!(
            self.bound_count < self.w.num_variables(),
            "Cannot bind more challenges than variables: already bound {}/{}",
            self.bound_count,
            self.w.num_variables()
        );

        let w_i = self.w.0[self.bound_count];
        // eq(w_i, r) = (1 - w_i)(1 - r) + w_i * r
        //            = 1 - w_i - r + 2 * w_i * r
        self.current_scalar *= F::ONE - w_i - r + (w_i * r).double();
        self.bound_count += 1;
    }

    /// Returns the current accumulated scalar `eq(w[..bound_count], r[..bound_count])`.
    ///
    /// This is the product of all `eq(w[i], r[i])` for `i < bound_count`.
    #[inline]
    #[must_use]
    pub const fn current_scalar(&self) -> F {
        self.current_scalar
    }

    /// Returns the evaluations of the linear polynomial for the next variable.
    ///
    /// Computes:
    /// ```text
    /// [current_scalar * (1 - w[bound_count]), current_scalar * w[bound_count]]
    /// ```
    ///
    /// These are the evaluations at X = 0 and X = 1 of:
    /// ```text
    /// eq(w[..bound_count+1], (r[..bound_count], X))
    /// ```
    ///
    /// This is the linear polynomial `l(X)` used in sumcheck rounds.
    ///
    /// ## Panics
    /// Panics if all variables have already been bound.
    #[inline]
    #[must_use]
    pub fn current_evals(&self) -> [F; 2] {
        assert!(
            self.bound_count < self.w.num_variables(),
            "No more variables to evaluate: all {}/{} variables bound",
            self.bound_count,
            self.w.num_variables()
        );

        let w_i = self.w.0[self.bound_count];
        [
            self.current_scalar * (F::ONE - w_i),
            self.current_scalar * w_i,
        ]
    }

    /// Returns the number of variables bound so far.
    ///
    /// This is also the index of the next variable to bind.
    #[inline]
    #[must_use]
    pub const fn bound_count(&self) -> usize {
        self.bound_count
    }

    /// Returns the total number of variables.
    #[inline]
    #[must_use]
    pub const fn num_variables(&self) -> usize {
        self.w.num_variables()
    }

    /// Returns `true` if all variables have been bound.
    #[inline]
    #[must_use]
    pub const fn is_fully_bound(&self) -> bool {
        self.bound_count == self.w.num_variables()
    }

    /// Returns the remaining number of variables to bind.
    #[inline]
    #[must_use]
    pub const fn remaining_variables(&self) -> usize {
        self.w.num_variables() - self.bound_count
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    use super::*;
    use crate::poly::multilinear::MultilinearPoint;

    type F = BabyBear;

    #[test]
    fn test_split_eq_poly_initialization() {
        // Create witness point w in F^3
        let w = MultilinearPoint::new(vec![F::from_u64(2), F::from_u64(3), F::from_u64(5)]);
        let split_eq = SplitEqPolynomial::new(&w);

        // Initially, no variables bound: eq([], []) = 1 (empty product)
        assert_eq!(split_eq.current_scalar(), F::ONE);
        // No challenges consumed yet
        assert_eq!(split_eq.bound_count(), 0);
        // Total number of variables to bind
        assert_eq!(split_eq.num_variables(), 3);
        // Not yet fully bound
        assert!(!split_eq.is_fully_bound());
        // All 3 variables still need to be bound
        assert_eq!(split_eq.remaining_variables(), 3);
    }

    #[test]
    fn test_split_eq_poly_single_bind() {
        // Witness w
        let w = MultilinearPoint::new(vec![F::from_u64(2), F::from_u64(3)]);
        let mut split_eq = SplitEqPolynomial::new(&w);

        // Bind first variable with challenge r0
        let r0 = F::from_u64(5);
        split_eq.bind(r0);

        // After one bind: current_scalar = eq(w[0], r0)
        //
        // Verify against direct computation using the full eq formula
        let expected = MultilinearPoint::eval_eq(&w.0[..1], &[r0]);
        assert_eq!(split_eq.current_scalar(), expected);
        // One variable bound
        assert_eq!(split_eq.bound_count(), 1);
        // One variable remaining
        assert_eq!(split_eq.remaining_variables(), 1);
    }

    #[test]
    fn test_split_eq_poly_multiple_binds() {
        // Witness w
        let w = MultilinearPoint::new(vec![F::from_u64(2), F::from_u64(3), F::from_u64(5)]);
        let mut split_eq = SplitEqPolynomial::new(&w);

        let r0 = F::from_u64(7);
        let r1 = F::from_u64(11);

        // After first bind: current_scalar = eq(w[0], r0)
        split_eq.bind(r0);
        let expected_1 = MultilinearPoint::eval_eq(&w.0[..1], &[r0]);
        assert_eq!(split_eq.current_scalar(), expected_1);

        // After second bind: current_scalar = eq(w[0..2], r[0..2])
        //
        // Computed incrementally as: previous_scalar * eq(w[1], r1)
        split_eq.bind(r1);
        let expected_2 = MultilinearPoint::eval_eq(&w.0[..2], &[r0, r1]);
        assert_eq!(split_eq.current_scalar(), expected_2);

        // Two variables bound, one remaining
        assert_eq!(split_eq.bound_count(), 2);
        assert!(!split_eq.is_fully_bound());
    }

    #[test]
    fn test_split_eq_poly_fully_bound() {
        // Witness w
        let w = MultilinearPoint::new(vec![F::from_u64(2), F::from_u64(3)]);
        let mut split_eq = SplitEqPolynomial::new(&w);

        let r0 = F::from_u64(5);
        let r1 = F::from_u64(7);

        // Bind all variables
        split_eq.bind(r0);
        split_eq.bind(r1);

        // Final scalar = eq(w, r)
        let expected = MultilinearPoint::eval_eq(&w.0, &[r0, r1]);
        assert_eq!(split_eq.current_scalar(), expected);
        // All variables bound
        assert_eq!(split_eq.bound_count(), 2);
        assert!(split_eq.is_fully_bound());
        // No remaining variables
        assert_eq!(split_eq.remaining_variables(), 0);
    }

    #[test]
    #[should_panic(expected = "Cannot bind more challenges than variables")]
    fn test_split_eq_poly_bind_too_many() {
        // Single variable witness w
        let w = MultilinearPoint::new(vec![F::from_u64(2)]);
        let mut split_eq = SplitEqPolynomial::new(&w);

        // First bind succeeds
        split_eq.bind(F::from_u64(3));
        // Second bind should panic: already fully bound
        split_eq.bind(F::from_u64(5));
    }

    #[test]
    fn test_split_eq_poly_current_evals() {
        // Witness w
        let w = MultilinearPoint::new(vec![F::from_u64(2), F::from_u64(3)]);
        let mut split_eq = SplitEqPolynomial::new(&w);

        // Initially: linear polynomial l(X) = eq((w[0]), (X)) over X ∈ {0,1}
        //
        // Returns [l(0), l(1)] = [eq(w[0], 0), eq(w[0], 1)] = [1-w[0], w[0]]
        let [eval_0, eval_1] = split_eq.current_evals();
        assert_eq!(eval_0, F::ONE - w.0[0]);
        assert_eq!(eval_1, w.0[0]);

        // Bind first variable with r0
        let r0 = F::from_u64(5);
        split_eq.bind(r0);

        // Now: linear polynomial l(X) = eq((w[0],w[1]), (r0,X)) over X ∈ {0,1}
        //
        // Returns [l(0), l(1)] = [scalar*(1-w[1]), scalar*w[1]]
        // where scalar = eq(w[0], r0)
        let [eval_0, eval_1] = split_eq.current_evals();
        let scalar = split_eq.current_scalar();
        assert_eq!(eval_0, scalar * (F::ONE - w.0[1]));
        assert_eq!(eval_1, scalar * w.0[1]);

        // Linear interpolation property: l(0) + l(1) = 2 * eq(w[..i], r[..i])
        //
        // Since we work in characteristic p, this simplifies to the scalar sum property
        assert_eq!(eval_0 + eval_1, scalar);
    }

    #[test]
    #[should_panic(expected = "No more variables to evaluate")]
    fn test_split_eq_poly_current_evals_fully_bound() {
        // Single variable w
        let w = MultilinearPoint::new(vec![F::from_u64(2)]);
        let mut split_eq = SplitEqPolynomial::new(&w);

        // Bind the only variable
        split_eq.bind(F::from_u64(3));
        // Attempting to get linear evals when fully bound should panic
        // (no next variable to evaluate over)
        let _ = split_eq.current_evals();
    }

    #[test]
    fn test_split_eq_poly_zero_variables() {
        // Edge case: empty witness w = () in F^0
        let w = MultilinearPoint::<F>::new(vec![]);
        let split_eq = SplitEqPolynomial::new(&w);

        // Empty product equals 1
        assert_eq!(split_eq.current_scalar(), F::ONE);
        // No variables to bind
        assert_eq!(split_eq.bound_count(), 0);
        assert_eq!(split_eq.num_variables(), 0);
        // Already fully bound since there are no variables
        assert!(split_eq.is_fully_bound());
    }

    #[test]
    fn test_split_eq_poly_one_variable() {
        // Single variable case
        let w = MultilinearPoint::new(vec![F::from_u64(2)]);
        let mut split_eq = SplitEqPolynomial::new(&w);

        // Linear polynomial l(X) = eq(w[0], X) = (1-w[0])(1-X) + w[0]*X
        //
        // Evaluations: [l(0), l(1)] = [1-w[0], w[0]] in field
        let [eval_0, eval_1] = split_eq.current_evals();
        assert_eq!(eval_0, F::ONE - F::from_u64(2));
        assert_eq!(eval_1, F::from_u64(2));

        // Bind with challenge r0
        let r0 = F::from_u64(3);
        split_eq.bind(r0);

        // Final scalar
        let expected = MultilinearPoint::eval_eq(&w.0, &[r0]);
        assert_eq!(split_eq.current_scalar(), expected);
        assert!(split_eq.is_fully_bound());
    }
}
