use alloc::vec::Vec;
use core::{
    ops::{Index, RangeBounds},
    slice::SliceIndex,
};

use itertools::Itertools;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_interpolation::interpolate_subgroup;
use p3_matrix::dense::RowMajorMatrix;
use rand::{
    Rng,
    distr::{Distribution, StandardUniform},
};

use crate::poly::evals::EvaluationsList;

/// A point `(x_1, ..., x_n)` in `F^n` for some field `F`.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct MultilinearPoint<F>(pub(crate) Vec<F>);

impl<F: Field> From<Vec<F>> for MultilinearPoint<F> {
    fn from(vars: Vec<F>) -> Self {
        Self::new(vars)
    }
}

impl<F: Field> From<&[F]> for MultilinearPoint<F> {
    fn from(vars: &[F]) -> Self {
        Self::new(vars.to_vec())
    }
}

impl<F> MultilinearPoint<F>
where
    F: Field,
{
    /// Construct a new `MultilinearPoint` from a vector of field elements.
    #[must_use]
    pub const fn new(coords: Vec<F>) -> Self {
        Self(coords)
    }

    /// Returns the number of variables (dimension `n`).
    #[inline]
    #[must_use]
    pub const fn num_variables(&self) -> usize {
        self.0.len()
    }

    /// Return a reference to the slice of field elements
    /// defining the point.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[F] {
        &self.0
    }

    /// Return an iterator over the field elements making up the point.
    #[inline]
    pub fn iter(&self) -> core::slice::Iter<'_, F> {
        self.0.iter()
    }

    /// Return a sub-point over the specified range of variables.
    #[inline]
    #[must_use]
    pub fn get_subpoint_over_range<R: RangeBounds<usize> + SliceIndex<[F], Output = [F]>>(
        &self,
        range: R,
    ) -> Self {
        Self(self.0[range].to_vec())
    }

    /// Return a reference to the last variable in the point, if it exists.
    ///
    /// Returns None if the point is empty.
    #[inline]
    #[must_use]
    pub fn last_variable(&self) -> Option<&F> {
        self.0.last()
    }

    /// Converts a univariate evaluation point into a multilinear one.
    ///
    /// Uses the bijection:
    /// ```ignore
    /// f(x_1, ..., x_n) <-> g(y) := f(y^(2^(n-1)), ..., y^4, y^2, y)
    /// ```
    /// Meaning:
    /// ```ignore
    /// x_1^i_1 * ... * x_n^i_n <-> y^i
    /// ```
    /// where `(i_1, ..., i_n)` is the **big-endian** binary decomposition of `i`.
    ///
    /// Reversing the order ensures the **big-endian** convention.
    pub fn expand_from_univariate(point: F, num_variables: usize) -> Self {
        let mut res: Vec<F> = F::zero_vec(num_variables);
        let mut cur = point;

        // Fill big-endian: [y^(2^(n-1)), ..., y^2, y]
        // Loop from the last index down to the first.
        for i in (0..num_variables).rev() {
            res[i] = cur;
            cur = cur.square();
        }

        Self(res)
    }

    /// Computes the equality polynomial `eq(p, q)` for two points given as slices.
    ///
    /// The **equality polynomial** for two vectors is:
    /// ```ignore
    /// eq(p, q) = ∏ (p_i * q_i + (1 - p_i) * (1 - q_i))
    /// ```
    ///
    /// This is a static method that avoids allocating `MultilinearPoint` wrappers
    /// when you already have slices.
    ///
    /// # Panics
    /// Panics if `p` and `q` have different lengths.
    #[must_use]
    #[inline]
    pub fn eval_eq(p: &[F], q: &[F]) -> F {
        assert_eq!(
            p.len(),
            q.len(),
            "Points must have the same number of variables"
        );

        // This uses the algebraic identity:
        // l * r + (1 - l) * (1 - r) = 1 + 2 * l * r - l - r
        // to avoid unnecessary multiplications.
        p.iter()
            .zip(q)
            .map(|(&l, &r)| F::ONE + l * r.double() - l - r)
            .product()
    }

    /// Computes `eq(c, p)`, where `p` is another `MultilinearPoint`.
    ///
    /// The **equality polynomial** for two vectors is:
    /// ```ignore
    /// eq(s1, s2) = ∏ (s1_i * s2_i + (1 - s1_i) * (1 - s2_i))
    /// ```
    #[must_use]
    #[inline]
    pub fn eq_poly(&self, point: &Self) -> F {
        Self::eval_eq(self.as_slice(), point.as_slice())
    }

    /// Computes `select(c, p)`, where `p` is another `MultilinearPoint`.
    ///
    /// The **selection polynomial** for two vectors is:
    /// ```ignore
    /// select(s1, s2) = ∏ (s1_i * s2_i - s2_i + 1)
    /// ```
    #[must_use]
    pub fn select_poly<EF: ExtensionField<F>>(&self, point: &MultilinearPoint<EF>) -> EF {
        assert_eq!(self.num_variables(), point.num_variables());

        self.into_iter()
            .zip(point)
            .map(|(&l, &r)| r * (l - F::ONE) + F::ONE)
            .product()
    }

    /// Computes equality polynomial evaluations over the inner variables for SVO-based sumcheck.
    ///
    /// This method is used in the Small Value Optimization (SVO) algorithm described in
    /// <https://eprint.iacr.org/2025/1117>, specifically in Procedure 9.
    ///
    /// Given a point `w = (w_0, ..., w_{n-1})` and parameters:
    /// - `NUM_SVO_ROUNDS`: The number of initial SVO rounds (typically 3)
    /// - `l = n`: The total number of variables
    /// - `l/2`: Half of the total variables
    ///
    /// This computes the equality polynomial `eq(w_{NUM_SVO_ROUNDS}, ..., w_{NUM_SVO_ROUNDS + l/2 - 1}; x)`
    /// evaluated at all points `x ∈ {0,1}^{l/2}`.
    ///
    /// ## Mathematical Background
    ///
    /// The equality polynomial for a point `w` over `k` variables is:
    /// ```ignore
    ///     eq(w; x) = ∏_{i=0}^{k-1} (w_i * x_i + (1 - w_i) * (1 - x_i))
    /// ```
    ///
    /// This method evaluates this polynomial at all `2^{l/2}` hypercube points by extracting
    /// the **middle** or **inner** segment of variables from the full point:
    /// - The first `NUM_SVO_ROUNDS` variables are skipped (used in earlier rounds)
    /// - The next `l/2` variables form the "inner" segment used for this computation
    /// - The remaining variables are used separately in the "outer" computation
    ///
    /// ## Returns
    ///
    /// An `EvaluationsList<F>` containing `2^{l/2}` field elements, where the `i`-th element
    /// is `eq(w_inner; binary(i))`, with `w_inner` being the extracted inner variables.
    ///
    /// ## Example
    ///
    /// Given a 10-variable witness:
    /// - `w` (`l=10`) -> `l/2 = 5`
    /// - `NUM_SVO_ROUNDS = 3` (`k=3`):
    ///
    /// The variables are partitioned as:
    ///
    /// `w = (w_0, w_1, w_2 | w_3, w_4, w_5, w_6, w_7 | w_8, w_9)`
    ///       \___________/   \______________________/  \_____/
    ///        Prefix (k=3)        Inner (l/2=5)         Outer
    ///
    /// This function will:
    /// 1.  Extract the **Inner** slice: `w_inner = (w_3, w_4, w_5, w_6, w_7)`.
    /// 2.  Compute `eq(w_inner, x)` for all `x` in `{0,1}^5`.
    pub fn svo_e_in_table<const NUM_SVO_ROUNDS: usize>(&self) -> EvaluationsList<F> {
        let half_l = self.num_variables() / 2;
        EvaluationsList::new_from_point(
            &self.get_subpoint_over_range(NUM_SVO_ROUNDS..NUM_SVO_ROUNDS + half_l),
            F::ONE,
        )
    }

    /// Computes equality polynomial evaluations over the outer variables for each SVO round.
    ///
    /// This method is used in the Small Value Optimization (SVO) algorithm described in
    /// <https://eprint.iacr.org/2025/1117>, specifically in Procedure 9.
    ///
    /// For each round `i` in `[0, NUM_SVO_ROUNDS)`, this computes the equality polynomial
    /// over the **"outer"** variables—those that are neither in the first `i+1` prefix variables
    /// nor in the middle "inner" segment.
    ///
    /// ## Mathematical Background
    ///
    /// For round `i`, we compute:
    /// ```ignore
    ///     E_out[i](y) = eq(w_{outer_i}; y)
    /// ```
    /// where `w_{outer_i}` is formed by concatenating:
    /// - `w[i+1..NUM_SVO_ROUNDS]`: Prefix variables after round `i` but before inner segment
    /// - `w[half_l + NUM_SVO_ROUNDS..]`: Variables after the inner segment (outer suffix)
    ///
    /// ## Variable Partitioning
    ///
    /// Given a witness `w = (w_0, ..., w_{n-1})` with `l = n` variables:
    ///
    /// ```ignore
    /// For round i=0:
    ///   w = (w_0 | w_1, w_2 | w_3, ..., w_7 | w_8, w_9)
    ///        \__/  \______/   \___________/  \_______/
    ///        used   prefix    inner (l/2=5)    outer
    ///
    /// For round i=1:
    ///   w = (w_0, w_1 | w_2 | w_3, ..., w_7 | w_8, w_9)
    ///        \______/   \_/   \___________/   \______/
    ///          used    prefix  inner (l/2=5)   outer
    ///
    /// For round i=2:
    ///   w = (w_0, w_1, w_2 | w_3, ..., w_7 | w_8, w_9)
    ///        \___________/   \___________/  \_______/
    ///            used        inner (l/2=5)    outer
    /// ```
    ///
    /// The "outer" for round `i` consists of:
    /// - Variables `w[i+1..NUM_SVO_ROUNDS]` (the remaining prefix)
    /// - Variables `w[half_l + NUM_SVO_ROUNDS..]` (the suffix after inner)
    ///
    /// ## Returns
    ///
    /// An array of `NUM_SVO_ROUNDS` evaluation lists, where `E_out[i]` contains
    /// `2^(# outer variables for round i)` field elements representing
    /// `eq(w_{outer_i}; x)` for all hypercube points `x`.
    ///
    /// ## Example
    ///
    /// For a 10-variable point with `NUM_SVO_ROUNDS = 3`:
    /// - Round 0: outer = (w_1, w_2, w_8, w_9) → 2^4 = 16 evaluations
    /// - Round 1: outer = (w_2, w_8, w_9) → 2^3 = 8 evaluations
    /// - Round 2: outer = (w_8, w_9) → 2^2 = 4 evaluations
    pub fn svo_e_out_tables<const NUM_SVO_ROUNDS: usize>(
        &self,
    ) -> [EvaluationsList<F>; NUM_SVO_ROUNDS] {
        // Compute l/2: the split point between inner and outer suffix
        let half_l = self.num_variables() / 2;

        // For each round, we'll extract the "outer" variables and compute their eq evaluations
        // The length of the outer segment: total variables - half_l (inner) - 1 (current round var)
        let w_out_len = self.num_variables() - half_l - 1;

        core::array::from_fn(|round| {
            // Build the outer variable slice for this round by concatenating two segments:
            // - Segment 1: Variables from (round+1) to NUM_SVO_ROUNDS (remaining prefix)
            // - Segment 2: Variables from (half_l + NUM_SVO_ROUNDS) onwards (outer suffix)
            let mut w_out = Vec::with_capacity(w_out_len);

            // Extract segment 1: w[round+1..NUM_SVO_ROUNDS]
            // These are the prefix variables not yet used in earlier rounds
            w_out.extend_from_slice(&self.0[round + 1..NUM_SVO_ROUNDS]);

            // Extract segment 2: w[half_l + NUM_SVO_ROUNDS..]
            // These are the variables after the inner segment (the outer suffix)
            w_out.extend_from_slice(&self.0[half_l + NUM_SVO_ROUNDS..]);

            // Compute eq(w_out; x) for all x in the hypercube
            EvaluationsList::new_from_point(&w_out.into(), F::ONE)
        })
    }

    /// Evaluates the equality polynomial `eq(self, X)` at a folded challenge point.
    ///
    /// This method is used in protocols that "skip" folding rounds by providing a single challenge
    /// for multiple variables.
    #[must_use]
    pub fn eq_poly_with_skip<EF>(&self, r_all: &MultilinearPoint<EF>, k_skip: usize) -> EF
    where
        F: TwoAdicField,
        EF: TwoAdicField + ExtensionField<F>,
    {
        // The total number of variables `n` is inferred from the challenge point `r_all`
        // and the number of skipped variables `k_skip`.
        let n = r_all.num_variables() + k_skip - 1;

        // The point `self` (z) must be defined over the full n-variable domain.
        assert_eq!(
            self.num_variables(),
            n,
            "Constraint point must have the same number of variables as the full domain"
        );

        // Construct the evaluation table for the polynomial eq_z(X).
        // This creates a list of 2^n values, where only the entry at index `z` is ONE.
        let evals = EvaluationsList::new_from_point(self, F::ONE);

        // Reshape the flat list of 2^n evaluations into a `2^k_skip x 2^(n-k_skip)` matrix.
        // Rows correspond to the skipped variables (X0, ..., X_{k_skip-1}).
        // Columns correspond to the remaining variables.
        let num_remaining_vars = n - k_skip;
        let width = 1 << num_remaining_vars;
        let mat = evals.into_mat(width);

        // Deconstruct the challenge object `r_all`.
        // The last element is the challenge for the `k_skip` variables being folded.
        let r_skip = *r_all
            .last_variable()
            .expect("skip challenge must be present");
        // The first `n - k_skip` elements are for the remaining variables.
        let r_rest = MultilinearPoint::new(r_all.as_slice()[..num_remaining_vars].to_vec());

        // Perform the two-stage evaluation.
        // Fold the skipped variables by interpolating each column at `r_skip`.
        let folded_row = interpolate_subgroup(&mat, r_skip);

        // Evaluate the new, smaller polynomial at the remaining challenges `r_rest`.
        EvaluationsList::new(folded_row).evaluate_hypercube_ext(&r_rest)
    }

    /// Returns a new `MultilinearPoint` with the variables in reversed order.
    #[must_use]
    pub fn reversed(&self) -> Self {
        Self(self.0.iter().rev().copied().collect())
    }

    /// Helper to extend a `MultilinearPoint` by creating a new one.
    #[cfg(test)]
    pub fn extend(&mut self, other: &Self) {
        self.0.extend_from_slice(&other.0);
    }

    /// Transposes points so same-index variables are aligned in rows.
    pub(crate) fn transpose(points: &[Self], rev_order: bool) -> RowMajorMatrix<F> {
        let k = points
            .iter()
            .map(Self::num_variables)
            .all_equal_value()
            .unwrap();
        let n = points.len();
        let mut flat = F::zero_vec(k * n);
        points.iter().enumerate().for_each(|(i, point)| {
            point.iter().enumerate().for_each(|(j, &cur)| {
                if rev_order {
                    flat[(k - 1 - j) * n + i] = cur;
                } else {
                    flat[j * n + i] = cur;
                }
            });
        });
        RowMajorMatrix::new(flat, n)
    }

    /// Given a position splits the point into two sub-points.
    pub(crate) fn split_at(&self, pos: usize) -> (Self, Self) {
        (
            Self::new(self.0.split_at(pos).0.to_vec()),
            Self::new(self.0.split_at(pos).1.to_vec()),
        )
    }
}

impl<F> MultilinearPoint<F>
where
    StandardUniform: Distribution<F>,
{
    pub fn rand<R: Rng>(rng: &mut R, num_variables: usize) -> Self {
        Self((0..num_variables).map(|_| rng.random()).collect())
    }
}

impl<'a, F> IntoIterator for &'a MultilinearPoint<F> {
    type Item = &'a F;
    type IntoIter = core::slice::Iter<'a, F>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<F> IntoIterator for MultilinearPoint<F> {
    type Item = F;
    type IntoIter = alloc::vec::IntoIter<F>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<F> Index<usize> for MultilinearPoint<F> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

#[cfg(test)]
#[allow(clippy::identity_op, clippy::erasing_op)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};
    use p3_matrix::dense::RowMajorMatrix;
    use proptest::prelude::*;
    use rand::{SeedableRng, rngs::SmallRng};

    use super::*;

    type F = BabyBear;
    type EF4 = BinomialExtensionField<F, 4>;

    #[test]
    fn test_num_variables() {
        let point = MultilinearPoint::<F>(vec![F::from_u64(1), F::from_u64(0), F::from_u64(1)]);
        assert_eq!(point.num_variables(), 3);
    }

    #[test]
    fn test_expand_from_univariate_single_variable() {
        let point = F::from_u64(3);
        let expanded = MultilinearPoint::expand_from_univariate(point, 1);

        // For n = 1, we expect [y]
        assert_eq!(expanded.0, vec![point]);
    }

    #[test]
    fn test_expand_from_univariate_two_variables() {
        let point = F::from_u64(2);
        let expanded = MultilinearPoint::expand_from_univariate(point, 2);

        // For n = 2, we expect [y^2, y]
        let expected = vec![point * point, point];
        assert_eq!(expanded.0, expected);
    }

    #[test]
    fn test_expand_from_univariate_three_variables() {
        let point = F::from_u64(5);
        let expanded = MultilinearPoint::expand_from_univariate(point, 3);

        // For n = 3, we expect [y^4, y^2, y]
        let expected = vec![point.exp_u64(4), point.exp_u64(2), point];
        assert_eq!(expanded.0, expected);
    }

    #[test]
    fn test_expand_from_univariate_large_variables() {
        let point = F::from_u64(7);
        let expanded = MultilinearPoint::expand_from_univariate(point, 5);

        // For n = 5, we expect [y^16, y^8, y^4, y^2, y]
        let expected = vec![
            point.exp_u64(16),
            point.exp_u64(8),
            point.exp_u64(4),
            point.exp_u64(2),
            point,
        ];
        assert_eq!(expanded.0, expected);
    }

    #[test]
    fn test_expand_from_univariate_identity() {
        let point = F::ONE;
        let expanded = MultilinearPoint::expand_from_univariate(point, 4);

        // Since 1^k = 1 for all k, the result should be [1, 1, 1, 1]
        let expected = vec![F::ONE; 4];
        assert_eq!(expanded.0, expected);
    }

    #[test]
    fn test_expand_from_univariate_zero() {
        let point = F::ZERO;
        let expanded = MultilinearPoint::expand_from_univariate(point, 4);

        // Since 0^k = 0 for all k, the result should be [0, 0, 0, 0]
        let expected = vec![F::ZERO; 4];
        assert_eq!(expanded.0, expected);
    }

    #[test]
    fn test_expand_from_univariate_empty() {
        let point = F::from_u64(9);
        let expanded = MultilinearPoint::expand_from_univariate(point, 0);

        // No variables should return an empty vector
        assert_eq!(expanded.0, vec![]);
    }

    #[test]
    fn test_expand_from_univariate_powers_correctness() {
        let point = F::from_u64(3);
        let expanded = MultilinearPoint::expand_from_univariate(point, 6);

        // For n = 6, we expect [y^32, y^16, y^8, y^4, y^2, y]
        let expected = vec![
            point.exp_u64(32),
            point.exp_u64(16),
            point.exp_u64(8),
            point.exp_u64(4),
            point.exp_u64(2),
            point,
        ];
        assert_eq!(expanded.0, expected);
    }

    #[test]
    fn test_eq_poly_outside_all_zeros() {
        let ml_point1 = MultilinearPoint(vec![F::ZERO; 4]);
        let ml_point2 = MultilinearPoint(vec![F::ZERO; 4]);

        assert_eq!(ml_point1.eq_poly(&ml_point2), F::ONE);
    }

    #[test]
    fn test_eq_poly_outside_all_ones() {
        let ml_point1 = MultilinearPoint(vec![F::ONE; 4]);
        let ml_point2 = MultilinearPoint(vec![F::ONE; 4]);

        assert_eq!(ml_point1.eq_poly(&ml_point2), F::ONE);
    }

    #[test]
    fn test_eq_poly_outside_mixed_match() {
        let ml_point1 = MultilinearPoint(vec![F::ONE, F::ZERO, F::ONE, F::ZERO]);
        let ml_point2 = MultilinearPoint(vec![F::ONE, F::ZERO, F::ONE, F::ZERO]);

        assert_eq!(ml_point1.eq_poly(&ml_point2), F::ONE);
    }

    #[test]
    fn test_eq_poly_outside_mixed_mismatch() {
        let ml_point1 = MultilinearPoint(vec![F::ONE, F::ZERO, F::ONE, F::ZERO]);
        let ml_point2 = MultilinearPoint(vec![F::ZERO, F::ONE, F::ZERO, F::ONE]);

        assert_eq!(ml_point1.eq_poly(&ml_point2), F::ZERO);
    }

    #[test]
    fn test_eval_eq_static_method() {
        // Test that eval_eq works correctly with slices
        let p = [F::from_u64(2), F::from_u64(3), F::from_u64(5)];
        let q = [F::from_u64(7), F::from_u64(11), F::from_u64(13)];

        // Compute using static method
        let result = MultilinearPoint::eval_eq(&p, &q);

        // Compute manually: eq(p,q) = ∏ (p_i * q_i + (1 - p_i) * (1 - q_i))
        let expected = (F::ONE + p[0] * q[0].double() - p[0] - q[0])
            * (F::ONE + p[1] * q[1].double() - p[1] - q[1])
            * (F::ONE + p[2] * q[2].double() - p[2] - q[2]);

        assert_eq!(result, expected);

        // Test that it matches eq_poly
        let ml_p = MultilinearPoint::new(p.to_vec());
        let ml_q = MultilinearPoint::new(q.to_vec());
        assert_eq!(result, ml_p.eq_poly(&ml_q));
    }

    #[test]
    fn test_eq_poly_outside_single_variable_match() {
        let ml_point1 = MultilinearPoint(vec![F::ONE]);
        let ml_point2 = MultilinearPoint(vec![F::ONE]);

        assert_eq!(ml_point1.eq_poly(&ml_point2), F::ONE);
    }

    #[test]
    fn test_eq_poly_outside_single_variable_mismatch() {
        let ml_point1 = MultilinearPoint(vec![F::ONE]);
        let ml_point2 = MultilinearPoint(vec![F::ZERO]);

        assert_eq!(ml_point1.eq_poly(&ml_point2), F::ZERO);
    }

    #[test]
    fn test_eq_poly_outside_manual_comparison() {
        // Construct the first multilinear point with arbitrary non-binary field values
        let x00 = F::from_u8(17);
        let x01 = F::from_u8(56);
        let x02 = F::from_u8(5);
        let x03 = F::from_u8(12);
        let ml_point1 = MultilinearPoint(vec![x00, x01, x02, x03]);

        // Construct the second multilinear point with different non-binary field values
        let x10 = F::from_u8(43);
        let x11 = F::from_u8(5);
        let x12 = F::from_u8(54);
        let x13 = F::from_u8(242);
        let ml_point2 = MultilinearPoint(vec![x10, x11, x12, x13]);

        // Compute the equality polynomial between ml_point1 and ml_point2
        let result = ml_point1.eq_poly(&ml_point2);

        // Manually compute the expected result of the equality polynomial:
        // eq(c, p) = ∏ (c_i * p_i + (1 - c_i) * (1 - p_i))
        // This formula evaluates to 1 iff c_i == p_i for all i, and < 1 otherwise
        let expected = (x00 * x10 + (F::ONE - x00) * (F::ONE - x10))
            * (x01 * x11 + (F::ONE - x01) * (F::ONE - x11))
            * (x02 * x12 + (F::ONE - x02) * (F::ONE - x12))
            * (x03 * x13 + (F::ONE - x03) * (F::ONE - x13));

        // Assert that the method and manual computation yield the same result
        assert_eq!(result, expected);
    }

    #[test]
    fn test_eq_poly_outside_large_match() {
        let ml_point1 = MultilinearPoint(vec![
            F::ONE,
            F::ONE,
            F::ZERO,
            F::ONE,
            F::ZERO,
            F::ONE,
            F::ONE,
            F::ZERO,
        ]);
        let ml_point2 = MultilinearPoint(vec![
            F::ONE,
            F::ONE,
            F::ZERO,
            F::ONE,
            F::ZERO,
            F::ONE,
            F::ONE,
            F::ZERO,
        ]);

        assert_eq!(ml_point1.eq_poly(&ml_point2), F::ONE);
    }

    #[test]
    fn test_eq_poly_outside_large_mismatch() {
        let ml_point1 = MultilinearPoint(vec![
            F::ONE,
            F::ONE,
            F::ZERO,
            F::ONE,
            F::ZERO,
            F::ONE,
            F::ONE,
            F::ZERO,
        ]);
        let ml_point2 = MultilinearPoint(vec![
            F::ONE,
            F::ONE,
            F::ZERO,
            F::ONE,
            F::ZERO,
            F::ONE,
            F::ONE,
            F::ONE, // Last bit differs
        ]);

        assert_eq!(ml_point1.eq_poly(&ml_point2), F::ZERO);
    }

    #[test]
    fn test_eq_poly_outside_empty_vector() {
        let ml_point1 = MultilinearPoint::<F>(vec![]);
        let ml_point2 = MultilinearPoint::<F>(vec![]);

        assert_eq!(ml_point1.eq_poly(&ml_point2), F::ONE);
    }

    #[test]
    #[should_panic]
    fn test_eq_poly_outside_different_lengths() {
        let ml_point1 = MultilinearPoint(vec![F::ONE, F::ZERO]);
        let ml_point2 = MultilinearPoint(vec![F::ONE, F::ZERO, F::ONE]);

        // Should panic because lengths do not match
        let _ = ml_point1.eq_poly(&ml_point2);
    }

    #[test]
    fn test_multilinear_point_rand_not_all_same() {
        const K: usize = 20; // Number of trials
        const N: usize = 10; // Number of variables

        let mut rng = SmallRng::seed_from_u64(1);

        let mut all_same_count = 0;

        for _ in 0..K {
            let point = MultilinearPoint::<F>::rand(&mut rng, N);
            let first = point.0[0];

            // Check if all coordinates are the same as the first one
            if point.into_iter().all(|x| x == first) {
                all_same_count += 1;
            }
        }

        // If all K trials are completely uniform, the RNG is suspicious
        assert!(
            all_same_count < K,
            "rand generated uniform points in all {K} trials"
        );
    }

    proptest! {
        #[test]
        fn proptest_eq_poly_outside_matches_manual(
            (coords1, coords2) in prop::collection::vec(0u8..=250, 1..=8).prop_flat_map(|v1| {
                let len = v1.len();
                prop::collection::vec(0u8..=250, len).prop_map(move |v2| (v1.clone(), v2))
            })
        ) {
            // Convert both u8 vectors to field elements
            let p1 = MultilinearPoint(coords1.iter().copied().map(F::from_u8).collect());
            let p2 = MultilinearPoint(coords2.iter().copied().map(F::from_u8).collect());

            // Evaluate eq_poly
            let result = p1.eq_poly(&p2);

            // Compute expected value using manual formula:
            // eq(c, p) = ∏ (c_i * p_i + (1 - c_i)(1 - p_i))
            let expected = p1.into_iter().zip(p2).fold(F::ONE, |acc, (a, b)| {
                acc * (a * b + (F::ONE - a) * (F::ONE - b))
            });

            prop_assert_eq!(result, expected);
        }
    }

    #[test]
    fn test_eq_poly_with_skip() {
        // SETUP:
        // - n = 3 total variables: (X0, X1, X2).
        // - The constraint point `z` is defined over the full n=3 variables.
        // - k_skip = 2 variables to skip: X0, X1.
        let n = 3;
        let k_skip = 2;

        // The weight polynomial is W(X) = eq_z(X0, X1, X2), where z=(2,3,4).
        // The constraint point MUST be full-dimensional.
        let point = MultilinearPoint::new(vec![F::from_u32(2), F::from_u32(3), F::from_u32(4)]);

        // The verifier's full challenge object `r_all`.
        // It has (n - k_skip) + 1 = (3 - 2) + 1 = 2 elements.
        // - r_rest for remaining variable (X2).
        // - r_skip for the combined (X0, X1) domain.
        let r_rest = MultilinearPoint::new(vec![EF4::from_u32(5)]);
        let r_skip = EF4::from_u32(7);
        let r_all = MultilinearPoint::new([r_rest.as_slice(), &[r_skip]].concat());

        // ACTION: Compute W(r) using the function under test.
        let result = point.eq_poly_with_skip(&r_all, k_skip);

        // MANUAL VERIFICATION:
        // 1. Manually construct the full 8-element table for W(X) = eq_z(X0, X1, X2).
        let z0 = EF4::from(point.as_slice()[0]);
        let z1 = EF4::from(point.as_slice()[1]);
        let z2 = EF4::from(point.as_slice()[2]);
        let mut full_evals_vec = Vec::with_capacity(1 << n);
        for i in 0..(1 << n) {
            // Index `i` corresponds to the point (x0, x1, x2)
            let x0 = EF4::from_u32((i >> 2) & 1);
            let x1 = EF4::from_u32((i >> 1) & 1);
            let x2 = EF4::from_u32(i & 1);
            let term0 = z0 * x0 + (EF4::ONE - z0) * (EF4::ONE - x0);
            let term1 = z1 * x1 + (EF4::ONE - z1) * (EF4::ONE - x1);
            let term2 = z2 * x2 + (EF4::ONE - z2) * (EF4::ONE - x2);
            full_evals_vec.push(term0 * term1 * term2);
        }

        // Reshape into a 4x2 matrix (Rows: (X0,X1), Cols: X2).
        let num_remaining = n - k_skip;
        let mat = RowMajorMatrix::new(full_evals_vec, 1 << num_remaining);

        // Interpolate each column at r_skip to fold the (X0, X1) variables.
        let folded_row = interpolate_subgroup(&mat, r_skip);

        // The `folded_row` is a new 1-variable polynomial, W'(X2).
        let final_poly = EvaluationsList::new(folded_row);

        // Evaluate this final polynomial at the remaining challenge, r_rest.
        let expected = final_poly.evaluate_hypercube_ext::<F>(&r_rest);

        assert_eq!(
            result, expected,
            "Manual skip evaluation for Evaluation weight should match"
        );
    }

    #[test]
    fn test_eq_poly_with_skip_evaluation_all_vars() {
        // SETUP:
        // - n = 5 total variables: (X0, X1, X2, X3, X4).
        // - The constraint point `z` is defined over the full n=5 variables.
        // - k_skip = 5 variables to skip (all of them).
        // - This leaves 0 remaining variables.
        let n = 5;
        let k_skip = 5;

        // The weight polynomial is W(X) = eq_z(X0..X4), where z is a random 5-element point.
        let point = MultilinearPoint::new(vec![
            F::from_u32(2),
            F::from_u32(3),
            F::from_u32(5),
            F::from_u32(7),
            F::from_u32(11),
        ]);

        // The verifier's challenge object `r_all`.
        // It has (n - k_skip) + 1 = (5 - 5) + 1 = 1 element.
        // - r_rest is an empty vector for the 0 remaining variables.
        // - r_skip is the single challenge for the combined (X0..X4) domain.
        let r_rest = MultilinearPoint::new(vec![]);
        let r_skip = EF4::from_u32(13);
        let r_all = MultilinearPoint::new(vec![r_skip]);

        // Compute W(r) using the function under test.
        let result = point.eq_poly_with_skip(&r_all, k_skip);

        // MANUAL VERIFICATION:
        // Manually construct the full 2^5=32 element table for W(X) = eq_z(X0..X4).
        let mut full_evals_vec = Vec::with_capacity(1 << n);
        for i in 0..(1 << n) {
            // The evaluation of eq_z(x) is the product of n individual terms.
            let eq_val: EF4 = (0..n)
                .map(|j| {
                    // Get the j-th coordinate of the constraint point z.
                    let z_j = EF4::from(point.as_slice()[j]);
                    // Get the j-th coordinate of the hypercube point x by checking the j-th bit of i.
                    let x_j = EF4::from_u32((i >> (n - 1 - j)) & 1);
                    // Calculate the j-th term of the product.
                    z_j * x_j + (EF4::ONE - z_j) * (EF4::ONE - x_j)
                })
                .product();
            full_evals_vec.push(eq_val);
        }

        // Reshape into a 32x1 matrix (Rows: (X0..X4), Cols: empty).
        let num_remaining = n - k_skip;
        let mat = RowMajorMatrix::new(full_evals_vec, 1 << num_remaining);

        // Interpolate the single column at r_skip to fold all 5 variables.
        let folded_row = interpolate_subgroup(&mat, r_skip);

        // The `folded_row` is a new 0-variable polynomial (a constant).
        let final_poly = EvaluationsList::new(folded_row);

        // Evaluate this constant polynomial. The point `r_rest` is empty.
        let expected = final_poly.evaluate_hypercube_ext::<F>(&r_rest);

        // The result of interpolation should be a single scalar.
        assert_eq!(
            final_poly.num_evals(),
            1,
            "Folding all variables should result in a single value"
        );

        assert_eq!(
            result, expected,
            "Manual skip evaluation for n=k_skip should match"
        );
    }

    #[test]
    fn test_svo_eq_evals_inner_basic() {
        // SETUP: 10-variable point with NUM_SVO_ROUNDS = 3
        // Point: w = (w_0, w_1, ..., w_9)
        // Half: l/2 = 10/2 = 5
        // Inner variables: w_3, w_4, w_5, w_6, w_7 (from index 3 to 7 inclusive)
        const NUM_SVO_ROUNDS: usize = 3;

        let w: Vec<F> = (0..10).map(F::from_u32).collect();
        let point = MultilinearPoint::new(w);

        // Compute inner eq evaluations using the method under test
        let evals = point.svo_e_in_table::<NUM_SVO_ROUNDS>();

        // VERIFICATION: Should return 2^5 = 32 evaluations
        assert_eq!(evals.num_evals(), 32);

        // Extract the inner variables manually for comparison
        let w_inner = &point.as_slice()[NUM_SVO_ROUNDS..NUM_SVO_ROUNDS + 5];
        assert_eq!(
            w_inner,
            &[
                F::from_u32(3),
                F::from_u32(4),
                F::from_u32(5),
                F::from_u32(6),
                F::from_u32(7)
            ]
        );

        // Verify ALL 32 evaluations using naive computation
        //
        // For each index i in [0, 31], compute eq(w_inner, binary(i)) naively
        for i in 0usize..32 {
            // Extract binary representation of i to get hypercube point x
            //
            // eq(w; x) = ∏_{j=0}^{4} (w_j * x_j + (1 - w_j) * (1 - x_j))
            let expected: F = (0..5)
                .map(|j| {
                    // Extract j-th bit of i (from most significant to least significant)
                    let x_j = F::from_u32(((i >> (4 - j)) & 1) as u32);
                    let w_j = w_inner[j];
                    // Compute the j-th term of the product
                    w_j * x_j + (F::ONE - w_j) * (F::ONE - x_j)
                })
                .product();

            assert_eq!(
                evals.as_slice()[i],
                expected,
                "Mismatch at index {i} (binary: {i:05b})"
            );
        }
    }

    #[test]
    fn test_svo_eq_evals_inner_minimal() {
        // SETUP: Minimal case with 4 variables and NUM_SVO_ROUNDS = 1
        // Point: w = (w_0, w_1, w_2, w_3)
        // Half: l/2 = 4/2 = 2
        // Inner variables: w_1, w_2 (from index 1 to 2 inclusive)
        const NUM_SVO_ROUNDS: usize = 1;

        let p0 = F::from_u32(7);
        let p1 = F::from_u32(11);
        let p2 = F::from_u32(13);
        let p3 = F::from_u32(17);
        let point = MultilinearPoint::new(vec![p0, p1, p2, p3]);

        let evals = point.svo_e_in_table::<NUM_SVO_ROUNDS>();

        // VERIFICATION: Should return 2^2 = 4 evaluations
        assert_eq!(evals.num_evals(), 4);

        #[allow(clippy::range_plus_one)]
        let w_inner = &point.as_slice()[NUM_SVO_ROUNDS..NUM_SVO_ROUNDS + 2];
        assert_eq!(w_inner, &[p1, p2]);

        // Verify ALL 4 evaluations using manual naive computation
        // eq(w; x) = ∏ (w_i * x_i + (1 - w_i) * (1 - x_i))
        let expected = [
            // Index 0 (binary 00): x = (0, 0)
            // eq(w_inner, 00) = (1 - p1) * (1 - p2)
            (F::ONE - p1) * (F::ONE - p2),
            // Index 1 (binary 01): x = (0, 1)
            // eq(w_inner, 01) = (1 - p1) * p2
            (F::ONE - p1) * p2,
            // Index 2 (binary 10): x = (1, 0)
            // eq(w_inner, 10) = p1 * (1 - p2)
            p1 * (F::ONE - p2),
            // Index 3 (binary 11): x = (1, 1)
            // eq(w_inner, 11) = p1 * p2
            p1 * p2,
        ];

        assert_eq!(evals.as_slice(), &expected);
    }

    #[test]
    fn test_svo_eq_evals_inner_different_start_rounds() {
        // SETUP: Test that different NUM_SVO_ROUNDS values extract different slices
        let point = MultilinearPoint::new(vec![
            F::from_u32(2),
            F::from_u32(3),
            F::from_u32(5),
            F::from_u32(7),
            F::from_u32(11),
            F::from_u32(13),
            F::from_u32(17),
            F::from_u32(19),
        ]);

        // With NUM_SVO_ROUNDS = 1, inner variables are w_1, w_2, w_3, w_4
        let evals_1 = point.svo_e_in_table::<1>();
        assert_eq!(evals_1.num_evals(), 16); // 2^4

        // With NUM_SVO_ROUNDS = 2, inner variables are w_2, w_3, w_4, w_5
        let evals_2 = point.svo_e_in_table::<2>();
        assert_eq!(evals_2.num_evals(), 16); // 2^4

        // They should produce different results since they use different variable slices
        assert_ne!(evals_1.as_slice(), evals_2.as_slice());

        // Verify the extraction is correct for NUM_SVO_ROUNDS = 2
        let w_inner = &point.as_slice()[2..6];
        let expected = EvaluationsList::new_from_point(&w_inner.into(), F::ONE);
        assert_eq!(evals_2.as_slice(), expected.as_slice());
    }

    #[test]
    fn test_svo_e_out_tables_sizes() {
        // SETUP: 10-variable point with NUM_SVO_ROUNDS = 3
        // Verify that each round produces the correct number of evaluations
        const NUM_SVO_ROUNDS: usize = 3;

        let w: Vec<F> = (0..10).map(F::from_u32).collect();
        let point = MultilinearPoint::new(w);

        let e_out = point.svo_e_out_tables::<NUM_SVO_ROUNDS>();

        // Round 0: outer = (w_1, w_2, w_8, w_9) → 4 variables → 2^4 = 16 evaluations
        assert_eq!(e_out[0].num_evals(), 16);

        // Round 1: outer = (w_2, w_8, w_9) → 3 variables → 2^3 = 8 evaluations
        assert_eq!(e_out[1].num_evals(), 8);

        // Round 2: outer = (w_8, w_9) → 2 variables → 2^2 = 4 evaluations
        assert_eq!(e_out[2].num_evals(), 4);
    }

    #[test]
    fn test_svo_e_out_tables_minimal() {
        // SETUP: Minimal 4-variable point with NUM_SVO_ROUNDS = 1
        // Point: w = (w_0, w_1, w_2, w_3)
        // half_l = 2
        // Round 0: outer = (w_3) → 1 variable
        const NUM_SVO_ROUNDS: usize = 1;

        let p0 = F::from_u32(7);
        let p1 = F::from_u32(11);
        let p2 = F::from_u32(13);
        let p3 = F::from_u32(17);
        let point = MultilinearPoint::new(vec![p0, p1, p2, p3]);

        let e_out = point.svo_e_out_tables::<NUM_SVO_ROUNDS>();

        // Should have 1 table for 1 round
        assert_eq!(e_out.len(), 1);

        // Round 0: outer variables should be (w_3)
        // Extraction:
        // - w[round+1..NUM_SVO_ROUNDS] = w[1..1] = empty
        // - w[half_l + NUM_SVO_ROUNDS..] = w[2+1..] = w[3..4] = (w_3)
        // This gives us 2^1 = 2 evaluations
        assert_eq!(e_out[0].num_evals(), 2);

        // Verify ALL 2 evaluations using manual computation
        // eq(w_outer, x) for x in {0, 1}
        let expected = [
            // Index 0 (binary 0): eq((p3), (0))
            F::ONE - p3,
            // Index 1 (binary 1): eq((p3), (1))
            p3,
        ];

        assert_eq!(e_out[0].as_slice(), &expected);
    }

    #[test]
    fn test_svo_e_out_tables_comprehensive() {
        // SETUP: 10-variable point with NUM_SVO_ROUNDS = 3
        // Verify ALL evaluations for each round using naive manual computation
        const NUM_SVO_ROUNDS: usize = 3;

        let mut rng = SmallRng::seed_from_u64(123);
        let w: Vec<F> = (0..10).map(|_| rng.random()).collect();
        let point = MultilinearPoint::new(w.clone());

        let e_out = point.svo_e_out_tables::<NUM_SVO_ROUNDS>();

        // Check that we have `NUM_SVO_ROUNDS` rounds
        assert_eq!(e_out.len(), NUM_SVO_ROUNDS);

        // ROUND 0
        //
        // For round 0, outer variables are: (w_1, w_2, w_8, w_9)
        // Partitioning: w = (w_0 | w_1, w_2 | w_3..w_7 | w_8, w_9)
        //                    used   outer      inner      outer
        // This gives 4 variables → 2^4 = 16 evaluations
        assert_eq!(e_out[0].num_evals(), 16);

        // Extract outer variables manually for clarity
        let outer_0 = [w[1], w[2], w[8], w[9]];

        // Verify ALL 16 evaluations using naive computation
        for i in 0usize..16 {
            // For each hypercube point x ∈ {0,1}^4, compute eq(outer_0, x)
            //
            // eq(w; x) = ∏_{j=0}^{3} (w_j * x_j + (1 - w_j) * (1 - x_j))
            let expected: F = (0..4)
                .map(|j| {
                    // Extract j-th bit of i to get the j-th component of x
                    let x_j = F::from_u32(((i >> (3 - j)) & 1) as u32);
                    let w_j = outer_0[j];
                    // Compute the equality polynomial term
                    w_j * x_j + (F::ONE - w_j) * (F::ONE - x_j)
                })
                .product();

            assert_eq!(
                e_out[0].as_slice()[i],
                expected,
                "Round 0: Mismatch at index {i} (binary: {i:04b})"
            );
        }

        // ROUND 1
        //
        // For round 1, outer variables are: (w_2, w_8, w_9)
        // Partitioning: w = (w_0, w_1 | w_2 | w_3..w_7 | w_8, w_9)
        //                    used      outer   inner      outer
        // This gives 3 variables → 2^3 = 8 evaluations
        assert_eq!(e_out[1].num_evals(), 8);

        let outer_1 = [w[2], w[8], w[9]];

        // Verify ALL 8 evaluations using naive computation
        for i in 0usize..8 {
            let expected: F = (0..3)
                .map(|j| {
                    let x_j = F::from_u32(((i >> (2 - j)) & 1) as u32);
                    let w_j = outer_1[j];
                    w_j * x_j + (F::ONE - w_j) * (F::ONE - x_j)
                })
                .product();

            assert_eq!(
                e_out[1].as_slice()[i],
                expected,
                "Round 1: Mismatch at index {i} (binary: {i:03b})"
            );
        }

        // ROUND 2
        //
        // For round 2, outer variables are: (w_8, w_9)
        // Partitioning: w = (w_0, w_1, w_2 | w_3..w_7 | w_8, w_9)
        //                         used        inner      outer
        // This gives 2 variables → 2^2 = 4 evaluations
        assert_eq!(e_out[2].num_evals(), 4);

        let outer_2 = [w[8], w[9]];

        // Verify ALL 4 evaluations using naive computation
        for i in 0usize..4 {
            let expected: F = (0..2)
                .map(|j| {
                    let x_j = F::from_u32(((i >> (1 - j)) & 1) as u32);
                    let w_j = outer_2[j];
                    w_j * x_j + (F::ONE - w_j) * (F::ONE - x_j)
                })
                .product();

            assert_eq!(
                e_out[2].as_slice()[i],
                expected,
                "Round 2: Mismatch at index {i} (binary: {i:02b})"
            );
        }
    }

    #[test]
    fn test_svo_e_out_tables_round_0_manual() {
        // SETUP: 6-variable point with NUM_SVO_ROUNDS = 2
        // Manually verify round 0 computations
        const NUM_SVO_ROUNDS: usize = 2;

        let p0 = F::from_u32(2);
        let p1 = F::from_u32(3);
        let p2 = F::from_u32(5);
        let p3 = F::from_u32(7);
        let p4 = F::from_u32(11);
        let p5 = F::from_u32(13);
        let point = MultilinearPoint::new(vec![p0, p1, p2, p3, p4, p5]);

        let e_out = point.svo_e_out_tables::<NUM_SVO_ROUNDS>();

        // Round 0: outer = (w_1, w_5)
        // half_l = 3, so:
        // - Prefix segment: w[1..2] = (w_1)
        // - Outer suffix: w[3+2..6] = w[5..6] = (w_5)
        // So outer_0 = (w_1, w_5) → 2 variables → 4 evaluations

        assert_eq!(e_out[0].num_evals(), 4);

        // Verify all 4 evaluations manually
        let expected_0 = [
            // Index 0 (binary 00): eq((p1, p5), (0, 0))
            (F::ONE - p1) * (F::ONE - p5),
            // Index 1 (binary 01): eq((p1, p5), (0, 1))
            (F::ONE - p1) * p5,
            // Index 2 (binary 10): eq((p1, p5), (1, 0))
            p1 * (F::ONE - p5),
            // Index 3 (binary 11): eq((p1, p5), (1, 1))
            p1 * p5,
        ];

        assert_eq!(e_out[0].as_slice(), &expected_0);

        // Round 1: outer = (w_5)
        // Round 1: w[2..2] = empty
        //          w[3+2..6] = w[5..6] = (w_5)
        // So outer_1 = (w_5) → 1 variable → 2 evaluations

        assert_eq!(e_out[1].num_evals(), 2);

        let expected_1 = [
            F::ONE - p5, // eq((p5), (0))
            p5,          // eq((p5), (1))
        ];

        assert_eq!(e_out[1].as_slice(), &expected_1);
    }
}
