use std::ops::Deref;

use p3_field::Field;
use rand::{
    Rng,
    distr::{Distribution, StandardUniform},
};

use super::hypercube::BinaryHypercubePoint;

/// A point `(x_1, ..., x_n)` in `F^n` for some field `F`.
///
/// Often, `x_i` are binary. If strictly binary, `BinaryHypercubePoint` is used.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct MultilinearPoint<F>(pub Vec<F>);

impl<F> Deref for MultilinearPoint<F> {
    type Target = Vec<F>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<F> MultilinearPoint<F>
where
    F: Field,
{
    /// Returns the number of variables (dimension `n`).
    #[inline]
    #[must_use]
    pub fn num_variables(&self) -> usize {
        self.len()
    }

    /// Converts a `BinaryHypercubePoint` (bit representation) into a `MultilinearPoint`.
    ///
    /// This maps each bit in the binary integer to `F::ONE` (1) or `F::ZERO` (0) in the field.
    ///
    /// Given `point = b_{n-1} ... b_1 b_0` (big-endian), it produces:
    /// ```ignore
    /// [b_{n-1}, b_{n-2}, ..., b_1, b_0]
    /// ```
    #[must_use]
    pub fn from_binary_hypercube_point(point: BinaryHypercubePoint, num_variables: usize) -> Self {
        Self(
            (0..num_variables)
                .rev()
                .map(|i| F::from_bool((point.0 >> i) & 1 == 1))
                .collect(),
        )
    }

    /// Converts `MultilinearPoint` to a `BinaryHypercubePoint`, assuming values are binary.
    ///
    /// The point is interpreted as a binary number:
    /// ```ignore
    /// b_{n-1} * 2^{n-1} + b_{n-2} * 2^{n-2} + ... + b_1 * 2^1 + b_0 * 2^0
    /// ```
    /// Returns `None` if any coordinate is non-binary.
    pub fn to_hypercube(&self) -> Option<BinaryHypercubePoint> {
        self.iter()
            .try_fold(0, |acc, &coord| {
                if coord == F::ZERO {
                    Some(acc << 1)
                } else if coord == F::ONE {
                    Some((acc << 1) | 1)
                } else {
                    None
                }
            })
            .map(BinaryHypercubePoint)
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
        let mut res = Vec::with_capacity(num_variables);
        let mut cur = point;

        for _ in 0..num_variables {
            res.push(cur);
            cur = cur.square(); // Compute y^(2^k) at each step
        }

        res.reverse();
        Self(res)
    }

    /// Computes the equality polynomial `eq(c, p)`, where `p` is binary.
    ///
    /// The **equality polynomial** is defined as:
    /// ```ignore
    /// eq(c, p) = ∏ (c_i * p_i + (1 - c_i) * (1 - p_i))
    /// ```
    /// which evaluates to `1` if `c == p`, and `0` otherwise.
    ///
    /// `p` is interpreted as a **big-endian** binary number.
    #[must_use]
    pub fn eq_poly(&self, mut point: BinaryHypercubePoint) -> F {
        let n_variables = self.num_variables();
        assert!(*point < (1 << n_variables)); // Ensure correct length

        let mut acc = F::ONE;

        for val in self.iter().rev() {
            let b = *point % 2;
            acc *= if b == 1 { *val } else { F::ONE - *val };
            *point >>= 1;
        }

        acc
    }

    /// Computes `eq(c, p)`, where `p` is a general `MultilinearPoint` (not necessarily binary).
    ///
    /// The **equality polynomial** for two vectors is:
    /// ```ignore
    /// eq(s1, s2) = ∏ (s1_i * s2_i + (1 - s1_i) * (1 - s2_i))
    /// ```
    /// which evaluates to `1` if `s1 == s2`, and `0` otherwise.
    ///
    /// This uses the algebraic identity:
    /// ```ignore
    /// s1_i * s2_i + (1 - s1_i) * (1 - s2_i) = 1 + 2 * s1_i * s2_i - s1_i - s2_i
    /// ```
    /// to avoid unnecessary multiplications.
    #[must_use]
    pub fn eq_poly_outside(&self, point: &Self) -> F {
        assert_eq!(self.num_variables(), point.num_variables());

        let mut acc = F::ONE;

        for (&l, &r) in self.iter().zip(&point.0) {
            // l * r + (1 - l) * (1 - r) = 1 + 2 * l * r - l - r
            // +/- much cheaper than multiplication.
            acc *= F::ONE + l * r.double() - l - r;
        }

        acc
    }

    /// Computes `eq3(c, p)`, the **equality polynomial** for `{0,1,2}^n`.
    ///
    /// `p` is interpreted as a **big-endian** ternary number.
    ///
    /// `eq3(c, p)` is the unique polynomial of **degree ≤ 2** in each variable,
    /// such that:
    /// ```ignore
    /// eq3(c, p) = 1  if c == p
    ///           = 0  otherwise
    /// ```
    /// Uses precomputed values to reduce redundant operations.
    #[must_use]
    pub fn eq_poly3(&self, mut point: usize) -> F {
        let n_variables = self.num_variables();
        assert!(point < 3usize.pow(n_variables as u32));

        let mut acc = F::ONE;

        // Iterate in **little-endian** order and adjust using big-endian convention.
        for &val in self.iter().rev() {
            let val_minus_one = val - F::ONE;
            let val_minus_two = val - F::TWO;

            acc *= match point % 3 {
                0 => val_minus_one * val_minus_two.halve(), // (val - 1)(val - 2) / 2
                1 => -val * val_minus_two,                  // val (val - 2)(-1)
                2 => val * val_minus_one.halve(),           // val (val - 1) / 2
                _ => unreachable!(),
            };
            point /= 3;
        }

        acc
    }
}

impl<F> From<F> for MultilinearPoint<F> {
    fn from(value: F) -> Self {
        Self(vec![value])
    }
}

impl<F> MultilinearPoint<F>
where
    StandardUniform: Distribution<F>,
{
    pub fn rand<R: Rng>(rng: &mut R, num_variables: usize) -> Self {
        Self(
            (0..num_variables)
                .map(|_| rng.sample(StandardUniform))
                .collect(),
        )
    }
}

#[cfg(test)]
#[allow(clippy::identity_op, clippy::cast_sign_loss, clippy::erasing_op)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use proptest::prelude::*;
    use rand::rng;

    use super::*;

    type F = BabyBear;

    #[test]
    fn test_num_variables() {
        let point = MultilinearPoint::<F>(vec![F::from_u64(1), F::from_u64(0), F::from_u64(1)]);
        assert_eq!(point.num_variables(), 3);
    }

    #[test]
    fn test_from_binary_hypercube_point_all_zeros() {
        let num_variables = 5;
        // Represents (0,0,0,0,0)
        let binary_point = BinaryHypercubePoint(0);
        let ml_point =
            MultilinearPoint::<F>::from_binary_hypercube_point(binary_point, num_variables);

        let expected = vec![F::ZERO; num_variables];
        assert_eq!(ml_point.0, expected);
    }

    #[test]
    fn test_from_binary_hypercube_point_all_ones() {
        let num_variables = 4;
        // Represents (1,1,1,1)
        let binary_point = BinaryHypercubePoint((1 << num_variables) - 1);
        let ml_point =
            MultilinearPoint::<F>::from_binary_hypercube_point(binary_point, num_variables);

        let expected = vec![F::ONE; num_variables];
        assert_eq!(ml_point.0, expected);
    }

    #[test]
    fn test_from_binary_hypercube_point_mixed_bits() {
        let num_variables = 6;
        // Represents (1,0,1,0,1,0)
        let binary_point = BinaryHypercubePoint(0b10_1010);
        let ml_point =
            MultilinearPoint::<F>::from_binary_hypercube_point(binary_point, num_variables);

        let expected = vec![F::ONE, F::ZERO, F::ONE, F::ZERO, F::ONE, F::ZERO];
        assert_eq!(ml_point.0, expected);
    }

    #[test]
    fn test_from_binary_hypercube_point_truncation() {
        let num_variables = 3;
        // Should only use last 3 bits (101)
        let binary_point = BinaryHypercubePoint(0b10101);
        let ml_point =
            MultilinearPoint::<F>::from_binary_hypercube_point(binary_point, num_variables);

        let expected = vec![F::ONE, F::ZERO, F::ONE];
        assert_eq!(ml_point.0, expected);
    }

    #[test]
    fn test_from_binary_hypercube_point_expansion() {
        let num_variables = 8;
        // Represents (0,0,0,0,1,0,1,0)
        let binary_point = BinaryHypercubePoint(0b1010);
        let ml_point =
            MultilinearPoint::<F>::from_binary_hypercube_point(binary_point, num_variables);

        let expected = vec![
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ONE,
            F::ZERO,
            F::ONE,
            F::ZERO,
        ];
        assert_eq!(ml_point.0, expected);
    }

    #[test]
    fn test_to_hypercube_all_zeros() {
        let point = MultilinearPoint(vec![F::ZERO, F::ZERO, F::ZERO, F::ZERO]);
        assert_eq!(point.to_hypercube(), Some(BinaryHypercubePoint(0)));
    }

    #[test]
    fn test_to_hypercube_all_ones() {
        let point = MultilinearPoint(vec![F::ONE, F::ONE, F::ONE, F::ONE]);
        assert_eq!(point.to_hypercube(), Some(BinaryHypercubePoint(0b1111)));
    }

    #[test]
    fn test_to_hypercube_mixed_bits() {
        let point = MultilinearPoint(vec![F::ONE, F::ZERO, F::ONE, F::ZERO]);
        assert_eq!(point.to_hypercube(), Some(BinaryHypercubePoint(0b1010)));
    }

    #[test]
    fn test_to_hypercube_single_bit() {
        let point = MultilinearPoint(vec![F::ONE]);
        assert_eq!(point.to_hypercube(), Some(BinaryHypercubePoint(1)));

        let point = MultilinearPoint(vec![F::ZERO]);
        assert_eq!(point.to_hypercube(), Some(BinaryHypercubePoint(0)));
    }

    #[test]
    fn test_to_hypercube_large_binary_number() {
        let point = MultilinearPoint(vec![
            F::ONE,
            F::ONE,
            F::ZERO,
            F::ONE,
            F::ZERO,
            F::ONE,
            F::ONE,
            F::ZERO,
        ]);
        assert_eq!(
            point.to_hypercube(),
            Some(BinaryHypercubePoint(0b1101_0110))
        );
    }

    #[test]
    fn test_to_hypercube_non_binary_values() {
        let invalid_value = F::from_u64(2);
        let point = MultilinearPoint(vec![F::ONE, invalid_value, F::ZERO]);
        assert_eq!(point.to_hypercube(), None);
    }

    #[test]
    fn test_to_hypercube_empty_vector() {
        let point = MultilinearPoint::<F>(vec![]);
        assert_eq!(point.to_hypercube(), Some(BinaryHypercubePoint(0)));
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
    fn test_eq_poly_all_zeros() {
        // Multilinear point (0,0,0,0)
        let ml_point = MultilinearPoint(vec![F::ZERO; 4]);
        let binary_point = BinaryHypercubePoint(0b0000);

        // eq_poly should evaluate to 1 since c_i = p_i = 0
        assert_eq!(ml_point.eq_poly(binary_point), F::ONE);
    }

    #[test]
    fn test_eq_poly_all_ones() {
        // Multilinear point (1,1,1,1)
        let ml_point = MultilinearPoint(vec![F::ONE; 4]);
        let binary_point = BinaryHypercubePoint(0b1111);

        // eq_poly should evaluate to 1 since c_i = p_i = 1
        assert_eq!(ml_point.eq_poly(binary_point), F::ONE);
    }

    #[test]
    fn test_eq_poly_mixed_bits_match() {
        // Multilinear point (1,0,1,0)
        let ml_point = MultilinearPoint(vec![F::ONE, F::ZERO, F::ONE, F::ZERO]);
        let binary_point = BinaryHypercubePoint(0b1010);

        // eq_poly should evaluate to 1 since c_i = p_i for all i
        assert_eq!(ml_point.eq_poly(binary_point), F::ONE);
    }

    #[test]
    fn test_eq_poly_mixed_bits_mismatch() {
        // Multilinear point (1,0,1,0)
        let ml_point = MultilinearPoint(vec![F::ONE, F::ZERO, F::ONE, F::ZERO]);
        let binary_point = BinaryHypercubePoint(0b1100); // Differs at second bit

        // eq_poly should evaluate to 0 since there is at least one mismatch
        assert_eq!(ml_point.eq_poly(binary_point), F::ZERO);
    }

    #[test]
    fn test_eq_poly_single_variable_match() {
        // Multilinear point (1)
        let ml_point = MultilinearPoint(vec![F::ONE]);
        let binary_point = BinaryHypercubePoint(0b1);

        // eq_poly should evaluate to 1 since c_1 = p_1 = 1
        assert_eq!(ml_point.eq_poly(binary_point), F::ONE);
    }

    #[test]
    fn test_eq_poly_single_variable_mismatch() {
        // Multilinear point (1)
        let ml_point = MultilinearPoint(vec![F::ONE]);
        let binary_point = BinaryHypercubePoint(0b0);

        // eq_poly should evaluate to 0 since c_1 != p_1
        assert_eq!(ml_point.eq_poly(binary_point), F::ZERO);
    }

    #[test]
    fn test_eq_poly_large_binary_number_match() {
        // Multilinear point (1,1,0,1,0,1,1,0)
        let ml_point = MultilinearPoint(vec![
            F::ONE,
            F::ONE,
            F::ZERO,
            F::ONE,
            F::ZERO,
            F::ONE,
            F::ONE,
            F::ZERO,
        ]);
        let binary_point = BinaryHypercubePoint(0b1101_0110);

        // eq_poly should evaluate to 1 since c_i = p_i for all i
        assert_eq!(ml_point.eq_poly(binary_point), F::ONE);
    }

    #[test]
    fn test_eq_poly_large_binary_number_mismatch() {
        // Multilinear point (1,1,0,1,0,1,1,0)
        let ml_point = MultilinearPoint(vec![
            F::ONE,
            F::ONE,
            F::ZERO,
            F::ONE,
            F::ZERO,
            F::ONE,
            F::ONE,
            F::ZERO,
        ]);
        let binary_point = BinaryHypercubePoint(0b1101_0111); // Last bit differs

        // eq_poly should evaluate to 0 since there is a mismatch
        assert_eq!(ml_point.eq_poly(binary_point), F::ZERO);
    }

    #[test]
    fn test_eq_poly_empty_vector() {
        // Empty Multilinear Point
        let ml_point = MultilinearPoint::<F>(vec![]);
        let binary_point = BinaryHypercubePoint(0);

        // eq_poly should evaluate to 1 since both are trivially equal
        assert_eq!(ml_point.eq_poly(binary_point), F::ONE);
    }

    #[test]
    fn test_eq_poly_outside_all_zeros() {
        let ml_point1 = MultilinearPoint(vec![F::ZERO; 4]);
        let ml_point2 = MultilinearPoint(vec![F::ZERO; 4]);

        assert_eq!(ml_point1.eq_poly_outside(&ml_point2), F::ONE);
    }

    #[test]
    fn test_eq_poly_outside_all_ones() {
        let ml_point1 = MultilinearPoint(vec![F::ONE; 4]);
        let ml_point2 = MultilinearPoint(vec![F::ONE; 4]);

        assert_eq!(ml_point1.eq_poly_outside(&ml_point2), F::ONE);
    }

    #[test]
    fn test_eq_poly_outside_mixed_match() {
        let ml_point1 = MultilinearPoint(vec![F::ONE, F::ZERO, F::ONE, F::ZERO]);
        let ml_point2 = MultilinearPoint(vec![F::ONE, F::ZERO, F::ONE, F::ZERO]);

        assert_eq!(ml_point1.eq_poly_outside(&ml_point2), F::ONE);
    }

    #[test]
    fn test_eq_poly_outside_mixed_mismatch() {
        let ml_point1 = MultilinearPoint(vec![F::ONE, F::ZERO, F::ONE, F::ZERO]);
        let ml_point2 = MultilinearPoint(vec![F::ZERO, F::ONE, F::ZERO, F::ONE]);

        assert_eq!(ml_point1.eq_poly_outside(&ml_point2), F::ZERO);
    }

    #[test]
    fn test_eq_poly_outside_single_variable_match() {
        let ml_point1 = MultilinearPoint(vec![F::ONE]);
        let ml_point2 = MultilinearPoint(vec![F::ONE]);

        assert_eq!(ml_point1.eq_poly_outside(&ml_point2), F::ONE);
    }

    #[test]
    fn test_eq_poly_outside_single_variable_mismatch() {
        let ml_point1 = MultilinearPoint(vec![F::ONE]);
        let ml_point2 = MultilinearPoint(vec![F::ZERO]);

        assert_eq!(ml_point1.eq_poly_outside(&ml_point2), F::ZERO);
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
        let result = ml_point1.eq_poly_outside(&ml_point2);

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

        assert_eq!(ml_point1.eq_poly_outside(&ml_point2), F::ONE);
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

        assert_eq!(ml_point1.eq_poly_outside(&ml_point2), F::ZERO);
    }

    #[test]
    fn test_eq_poly_outside_empty_vector() {
        let ml_point1 = MultilinearPoint::<F>(vec![]);
        let ml_point2 = MultilinearPoint::<F>(vec![]);

        assert_eq!(ml_point1.eq_poly_outside(&ml_point2), F::ONE);
    }

    #[test]
    #[should_panic]
    fn test_eq_poly_outside_different_lengths() {
        let ml_point1 = MultilinearPoint(vec![F::ONE, F::ZERO]);
        let ml_point2 = MultilinearPoint(vec![F::ONE, F::ZERO, F::ONE]);

        // Should panic because lengths do not match
        let _ = ml_point1.eq_poly_outside(&ml_point2);
    }

    #[test]
    fn test_eq_poly3_all_zeros() {
        let ml_point = MultilinearPoint(vec![F::ZERO; 4]);
        // (0,0,0,0) in base 3 = 0 * 3^3 + 0 * 3^2 + 0 * 3^1 + 0 * 3^0 = 0
        let ternary_point = 0;

        assert_eq!(ml_point.eq_poly3(ternary_point), F::ONE);
    }

    #[test]
    fn test_eq_poly3_all_ones() {
        let ml_point = MultilinearPoint(vec![F::ONE; 4]);
        // (1,1,1,1) in base 3 = 1 * 3^3 + 1 * 3^2 + 1 * 3^1 + 1 * 3^0
        let ternary_point = 1 * 3_i32.pow(3) + 1 * 3_i32.pow(2) + 1 * 3_i32.pow(1) + 1;

        assert_eq!(ml_point.eq_poly3(ternary_point as usize), F::ONE);
    }

    #[test]
    fn test_eq_poly3_all_twos() {
        let two = F::ONE + F::ONE;
        let ml_point = MultilinearPoint(vec![two; 4]);
        // (2,2,2,2) in base 3 = 2 * 3^3 + 2 * 3^2 + 2 * 3^1 + 2 * 3^0
        let ternary_point = 2 * 3_i32.pow(3) + 2 * 3_i32.pow(2) + 2 * 3_i32.pow(1) + 2;

        assert_eq!(ml_point.eq_poly3(ternary_point as usize), F::ONE);
    }

    #[test]
    fn test_eq_poly3_mixed_match() {
        let two = F::ONE + F::ONE;
        let ml_point = MultilinearPoint(vec![two, F::ONE, F::ZERO, F::ONE]);
        // (2,1,0,1) in base 3 = 2 * 3^3 + 1 * 3^2 + 0 * 3^1 + 1 * 3^0
        let ternary_point = 2 * 3_i32.pow(3) + 1 * 3_i32.pow(2) + 0 * 3_i32.pow(1) + 1;

        assert_eq!(ml_point.eq_poly3(ternary_point as usize), F::ONE);
    }

    #[test]
    fn test_eq_poly3_mixed_mismatch() {
        let two = F::ONE + F::ONE;
        let ml_point = MultilinearPoint(vec![two, F::ONE, F::ZERO, F::ONE]);
        // (2,2,0,1) differs at the second coordinate
        let ternary_point = 2 * 3_i32.pow(3) + 2 * 3_i32.pow(2) + 0 * 3_i32.pow(1) + 1;

        assert_eq!(ml_point.eq_poly3(ternary_point as usize), F::ZERO);
    }

    #[test]
    fn test_eq_poly3_single_variable_match() {
        let ml_point = MultilinearPoint(vec![F::ONE]);
        // (1) in base 3 = 1
        let ternary_point = 1;

        assert_eq!(ml_point.eq_poly3(ternary_point), F::ONE);
    }

    #[test]
    fn test_eq_poly3_single_variable_mismatch() {
        let ml_point = MultilinearPoint(vec![F::ONE]);
        // (2) in base 3 = 2
        let ternary_point = 2;

        assert_eq!(ml_point.eq_poly3(ternary_point), F::ZERO);
    }

    #[test]
    fn test_eq_poly3_large_match() {
        let two = F::ONE + F::ONE;
        let ml_point = MultilinearPoint(vec![two, F::ONE, F::ZERO, F::ONE, F::ZERO, two, F::ONE]);
        // (2,1,0,1,0,2,1) in base 3 = 2 * 3^6 + 1 * 3^5 + 0 * 3^4 + 1 * 3^3 + 0 * 3^2 + 2 * 3^1 + 1
        // * 3^0
        let ternary_point = 2 * 3_i32.pow(6)
            + 1 * 3_i32.pow(5)
            + 0 * 3_i32.pow(4)
            + 1 * 3_i32.pow(3)
            + 0 * 3_i32.pow(2)
            + 2 * 3_i32.pow(1)
            + 1;

        assert_eq!(ml_point.eq_poly3(ternary_point as usize), F::ONE);
    }

    #[test]
    fn test_eq_poly3_large_mismatch() {
        let two = F::ONE + F::ONE;
        let ml_point = MultilinearPoint(vec![two, F::ONE, F::ZERO, F::ONE, F::ZERO, two, F::ONE]);
        // (2,1,0,1,1,2,1) differs at the fifth coordinate
        let ternary_point = 2 * 3_i32.pow(6)
            + 1 * 3_i32.pow(5)
            + 0 * 3_i32.pow(4)
            + 1 * 3_i32.pow(3)
            + 1 * 3_i32.pow(2)
            + 2 * 3_i32.pow(1)
            + 1;

        assert_eq!(ml_point.eq_poly3(ternary_point as usize), F::ZERO);
    }

    #[test]
    fn test_eq_poly3_empty_vector() {
        let ml_point = MultilinearPoint::<F>(vec![]);
        let ternary_point = 0;

        assert_eq!(ml_point.eq_poly3(ternary_point), F::ONE);
    }

    #[test]
    #[should_panic]
    fn test_eq_poly3_invalid_ternary_value() {
        let ml_point = MultilinearPoint(vec![F::ONE, F::ZERO]);
        let ternary_point = 9; // Invalid ternary representation (not in {0,1,2})

        let _ = ml_point.eq_poly3(ternary_point);
    }

    #[test]
    fn test_equality() {
        let point = MultilinearPoint(vec![F::from_u64(0), F::from_u64(0)]);
        assert_eq!(point.eq_poly(BinaryHypercubePoint(0b00)), F::from_u64(1));
        assert_eq!(point.eq_poly(BinaryHypercubePoint(0b01)), F::from_u64(0));
        assert_eq!(point.eq_poly(BinaryHypercubePoint(0b10)), F::from_u64(0));
        assert_eq!(point.eq_poly(BinaryHypercubePoint(0b11)), F::from_u64(0));

        let point = MultilinearPoint(vec![F::from_u64(1), F::from_u64(0)]);
        assert_eq!(point.eq_poly(BinaryHypercubePoint(0b00)), F::from_u64(0));
        assert_eq!(point.eq_poly(BinaryHypercubePoint(0b01)), F::from_u64(0));
        assert_eq!(point.eq_poly(BinaryHypercubePoint(0b10)), F::from_u64(1));
        assert_eq!(point.eq_poly(BinaryHypercubePoint(0b11)), F::from_u64(0));
    }

    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn test_equality3() {
        let point = MultilinearPoint(vec![F::from_u64(0), F::from_u64(0)]);

        assert_eq!(point.eq_poly3(0), F::from_u64(1));
        assert_eq!(point.eq_poly3(1), F::from_u64(0));
        assert_eq!(point.eq_poly3(2), F::from_u64(0));
        assert_eq!(point.eq_poly3(3), F::from_u64(0));
        assert_eq!(point.eq_poly3(4), F::from_u64(0));
        assert_eq!(point.eq_poly3(5), F::from_u64(0));
        assert_eq!(point.eq_poly3(6), F::from_u64(0));
        assert_eq!(point.eq_poly3(7), F::from_u64(0));
        assert_eq!(point.eq_poly3(8), F::from_u64(0));

        let point = MultilinearPoint(vec![F::from_u64(1), F::from_u64(0)]);

        assert_eq!(point.eq_poly3(0), F::from_u64(0));
        assert_eq!(point.eq_poly3(1), F::from_u64(0));
        assert_eq!(point.eq_poly3(2), F::from_u64(0));
        assert_eq!(point.eq_poly3(3), F::from_u64(1)); // 3 corresponds to ternary (1,0)
        assert_eq!(point.eq_poly3(4), F::from_u64(0));
        assert_eq!(point.eq_poly3(5), F::from_u64(0));
        assert_eq!(point.eq_poly3(6), F::from_u64(0));
        assert_eq!(point.eq_poly3(7), F::from_u64(0));
        assert_eq!(point.eq_poly3(8), F::from_u64(0));

        let point = MultilinearPoint(vec![F::from_u64(0), F::from_u64(2)]);

        assert_eq!(point.eq_poly3(0), F::from_u64(0));
        assert_eq!(point.eq_poly3(1), F::from_u64(0));
        assert_eq!(point.eq_poly3(2), F::from_u64(1)); // 2 corresponds to ternary (0,2)
        assert_eq!(point.eq_poly3(3), F::from_u64(0));
        assert_eq!(point.eq_poly3(4), F::from_u64(0));
        assert_eq!(point.eq_poly3(5), F::from_u64(0));
        assert_eq!(point.eq_poly3(6), F::from_u64(0));
        assert_eq!(point.eq_poly3(7), F::from_u64(0));
        assert_eq!(point.eq_poly3(8), F::from_u64(0));

        let point = MultilinearPoint(vec![F::from_u64(2), F::from_u64(2)]);

        assert_eq!(point.eq_poly3(0), F::from_u64(0));
        assert_eq!(point.eq_poly3(1), F::from_u64(0));
        assert_eq!(point.eq_poly3(2), F::from_u64(0));
        assert_eq!(point.eq_poly3(3), F::from_u64(0));
        assert_eq!(point.eq_poly3(4), F::from_u64(0));
        assert_eq!(point.eq_poly3(5), F::from_u64(0));
        assert_eq!(point.eq_poly3(6), F::from_u64(0));
        assert_eq!(point.eq_poly3(7), F::from_u64(0));
        assert_eq!(point.eq_poly3(8), F::from_u64(1)); // 8 corresponds to ternary (2,2)
    }

    #[test]
    fn test_multilinear_point_rand_not_all_same() {
        const K: usize = 20; // Number of trials
        const N: usize = 10; // Number of variables

        let mut rng = rng();

        let mut all_same_count = 0;

        for _ in 0..K {
            let point = MultilinearPoint::<F>::rand(&mut rng, N);
            let first = point.0[0];

            // Check if all coordinates are the same as the first one
            if point.0.iter().all(|&x| x == first) {
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

            // Evaluate eq_poly_outside
            let result = p1.eq_poly_outside(&p2);

            // Compute expected value using manual formula:
            // eq(c, p) = ∏ (c_i * p_i + (1 - c_i)(1 - p_i))
            let expected = p1.0.iter().zip(&p2.0).fold(F::ONE, |acc, (&a, &b)| {
                acc * (a * b + (F::ONE - a) * (F::ONE - b))
            });

            prop_assert_eq!(result, expected);
        }
    }
}
