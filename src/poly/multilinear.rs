use super::hypercube::BinaryHypercubePoint;
use p3_field::Field;

/// Point (x_1,..., x_n) in F^n for some n. Often, the x_i are binary.
/// For the latter case, we also have `BinaryHypercubePoint`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultilinearPoint<F>(pub Vec<F>);

impl<F> MultilinearPoint<F>
where
    F: Field,
{
    /// returns the number of variables.
    pub fn n_variables(&self) -> usize {
        self.0.len()
    }

    /// Creates a MultilinearPoint from a BinaryHypercubePoint; the latter models the same thing,
    /// but is restricted to binary entries.
    pub fn from_binary_hypercube_point(point: BinaryHypercubePoint, num_variables: usize) -> Self {
        Self(
            (0..num_variables)
                .rev()
                .map(|i| if (point.0 >> i) & 1 == 1 { F::ONE } else { F::ZERO })
                .collect(),
        )
    }

    /// Converts to a BinaryHypercubePoint, provided the MultilinearPoint is actually in {0,1}^n.
    pub fn to_hypercube(&self) -> Option<BinaryHypercubePoint> {
        self.0
            .iter()
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

    /// converts a univariate evaluation point into a multilinear one.
    ///
    /// Notably, consider the usual bijection
    /// {multilinear polys in n variables} <-> {univariate polys of deg < 2^n}
    /// f(x_1,...x_n)  <-> g(y) := f(y^(2^(n-1), ..., y^4, y^2, y).
    /// x_1^i_1 * ... *x_n^i_n <-> y^i, where (i_1,...,i_n) is the (big-endian) binary decomposition
    /// of i.
    ///
    /// expand_from_univariate maps the evaluation points to the multivariate domain, i.e.
    /// f(expand_from_univariate(y)) == g(y).
    /// in a way that is compatible with our endianness choices.
    pub fn expand_from_univariate(point: F, num_variables: usize) -> Self {
        let mut res = Vec::with_capacity(num_variables);
        let mut cur = point;
        for _ in 0..num_variables {
            res.push(cur);
            cur = cur * cur;
        }

        // Reverse so higher power is first
        res.reverse();

        Self(res)
    }

    /// Compute eq(coords,point), where eq is the equality polynomial, where point is binary.
    ///
    /// Recall that the equality polynomial eq(c, p) is defined as eq(c,p) == \prod_i c_i * p_i +
    /// (1-c_i)*(1-p_i). Note that for fixed p, viewed as a polynomial in c, it is the
    /// interpolation polynomial associated to the evaluation point p in the evaluation set {0,1}^n.
    pub fn eq_poly(&self, mut point: BinaryHypercubePoint) -> F {
        let n_variables = self.n_variables();
        // check that the lengths of coords and point match.
        assert!(*point < (1 << n_variables));

        let mut acc = F::ONE;

        for val in self.0.iter().rev() {
            let b = *point % 2;
            acc *= if b == 1 { *val } else { F::ONE - *val };
            *point >>= 1;
        }

        acc
    }

    /// Compute eq(coords,point), where eq is the equality polynomial and where point is not
    /// neccessarily binary.
    ///
    /// Recall that the equality polynomial eq(c, p) is defined as eq(c,p) == \prod_i c_i * p_i +
    /// (1-c_i)*(1-p_i). Note that for fixed p, viewed as a polynomial in c, it is the
    /// interpolation polynomial associated to the evaluation point p in the evaluation set {0,1}^n.
    pub fn eq_poly_outside(&self, point: &Self) -> F {
        assert_eq!(self.n_variables(), point.n_variables());

        let mut acc = F::ONE;

        for (&l, &r) in self.0.iter().zip(&point.0) {
            acc *= l * r + (F::ONE - l) * (F::ONE - r);
        }

        acc
    }

    /// Compute eq3(coords,point), where eq3 is the equality polynomial for {0,1,2}^n and point is
    /// interpreted as an element from {0,1,2}^n via (big Endian) ternary decomposition.
    ///
    /// eq3(coords, point) is the unique polynomial of degree <=2 in each variable, s.t.
    /// for coords, point in {0,1,2}^n, we have:
    /// eq3(coords,point) = 1 if coords == point and 0 otherwise.
    pub fn eq_poly3(&self, mut point: usize) -> F {
        let two_inv = F::TWO.inverse();

        let n_variables = self.n_variables();
        assert!(point < 3usize.pow(n_variables as u32));

        let mut acc = F::ONE;

        // Note: This iterates over the ternary decomposition least-significant trit(?) first.
        // Since our convention is big endian, we reverse the order of coords to account for this.
        for &val in self.0.iter().rev() {
            acc *= match point % 3 {
                0 => (val - F::ONE) * (val - F::TWO) * two_inv,
                1 => val * (val - F::TWO) * (-F::ONE),
                2 => val * (val - F::ONE) * two_inv,
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

#[cfg(test)]
#[allow(
    clippy::identity_op,
    clippy::cast_sign_loss,
    clippy::erasing_op,
    clippy::should_panic_without_expect
)]
mod tests {
    use super::*;
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    #[test]
    fn test_n_variables() {
        let point = MultilinearPoint::<BabyBear>(vec![
            BabyBear::from_u64(1),
            BabyBear::from_u64(0),
            BabyBear::from_u64(1),
        ]);
        assert_eq!(point.n_variables(), 3);
    }

    #[test]
    fn test_from_binary_hypercube_point_all_zeros() {
        let num_variables = 5;
        // Represents (0,0,0,0,0)
        let binary_point = BinaryHypercubePoint(0);
        let ml_point =
            MultilinearPoint::<BabyBear>::from_binary_hypercube_point(binary_point, num_variables);

        let expected = vec![BabyBear::ZERO; num_variables];
        assert_eq!(ml_point.0, expected);
    }

    #[test]
    fn test_from_binary_hypercube_point_all_ones() {
        let num_variables = 4;
        // Represents (1,1,1,1)
        let binary_point = BinaryHypercubePoint((1 << num_variables) - 1);
        let ml_point =
            MultilinearPoint::<BabyBear>::from_binary_hypercube_point(binary_point, num_variables);

        let expected = vec![BabyBear::ONE; num_variables];
        assert_eq!(ml_point.0, expected);
    }

    #[test]
    fn test_from_binary_hypercube_point_mixed_bits() {
        let num_variables = 6;
        // Represents (1,0,1,0,1,0)
        let binary_point = BinaryHypercubePoint(0b10_1010);
        let ml_point =
            MultilinearPoint::<BabyBear>::from_binary_hypercube_point(binary_point, num_variables);

        let expected = vec![
            BabyBear::ONE,
            BabyBear::ZERO,
            BabyBear::ONE,
            BabyBear::ZERO,
            BabyBear::ONE,
            BabyBear::ZERO,
        ];
        assert_eq!(ml_point.0, expected);
    }

    #[test]
    fn test_from_binary_hypercube_point_truncation() {
        let num_variables = 3;
        // Should only use last 3 bits (101)
        let binary_point = BinaryHypercubePoint(0b10101);
        let ml_point =
            MultilinearPoint::<BabyBear>::from_binary_hypercube_point(binary_point, num_variables);

        let expected = vec![BabyBear::ONE, BabyBear::ZERO, BabyBear::ONE];
        assert_eq!(ml_point.0, expected);
    }

    #[test]
    fn test_from_binary_hypercube_point_expansion() {
        let num_variables = 8;
        // Represents (0,0,0,0,1,0,1,0)
        let binary_point = BinaryHypercubePoint(0b1010);
        let ml_point =
            MultilinearPoint::<BabyBear>::from_binary_hypercube_point(binary_point, num_variables);

        let expected = vec![
            BabyBear::ZERO,
            BabyBear::ZERO,
            BabyBear::ZERO,
            BabyBear::ZERO,
            BabyBear::ONE,
            BabyBear::ZERO,
            BabyBear::ONE,
            BabyBear::ZERO,
        ];
        assert_eq!(ml_point.0, expected);
    }

    #[test]
    fn test_to_hypercube_all_zeros() {
        let point =
            MultilinearPoint(vec![BabyBear::ZERO, BabyBear::ZERO, BabyBear::ZERO, BabyBear::ZERO]);
        assert_eq!(point.to_hypercube(), Some(BinaryHypercubePoint(0)));
    }

    #[test]
    fn test_to_hypercube_all_ones() {
        let point =
            MultilinearPoint(vec![BabyBear::ONE, BabyBear::ONE, BabyBear::ONE, BabyBear::ONE]);
        assert_eq!(point.to_hypercube(), Some(BinaryHypercubePoint(0b1111)));
    }

    #[test]
    fn test_to_hypercube_mixed_bits() {
        let point =
            MultilinearPoint(vec![BabyBear::ONE, BabyBear::ZERO, BabyBear::ONE, BabyBear::ZERO]);
        assert_eq!(point.to_hypercube(), Some(BinaryHypercubePoint(0b1010)));
    }

    #[test]
    fn test_to_hypercube_single_bit() {
        let point = MultilinearPoint(vec![BabyBear::ONE]);
        assert_eq!(point.to_hypercube(), Some(BinaryHypercubePoint(1)));

        let point = MultilinearPoint(vec![BabyBear::ZERO]);
        assert_eq!(point.to_hypercube(), Some(BinaryHypercubePoint(0)));
    }

    #[test]
    fn test_to_hypercube_large_binary_number() {
        let point = MultilinearPoint(vec![
            BabyBear::ONE,
            BabyBear::ONE,
            BabyBear::ZERO,
            BabyBear::ONE,
            BabyBear::ZERO,
            BabyBear::ONE,
            BabyBear::ONE,
            BabyBear::ZERO,
        ]);
        assert_eq!(point.to_hypercube(), Some(BinaryHypercubePoint(0b1101_0110)));
    }

    #[test]
    fn test_to_hypercube_non_binary_values() {
        let invalid_value = BabyBear::from_u64(2);
        let point = MultilinearPoint(vec![BabyBear::ONE, invalid_value, BabyBear::ZERO]);
        assert_eq!(point.to_hypercube(), None);
    }

    #[test]
    fn test_to_hypercube_empty_vector() {
        let point = MultilinearPoint::<BabyBear>(vec![]);
        assert_eq!(point.to_hypercube(), Some(BinaryHypercubePoint(0)));
    }

    #[test]
    fn test_expand_from_univariate_single_variable() {
        let point = BabyBear::from_u64(3);
        let expanded = MultilinearPoint::expand_from_univariate(point, 1);

        // For n = 1, we expect [y]
        assert_eq!(expanded.0, vec![point]);
    }

    #[test]
    fn test_expand_from_univariate_two_variables() {
        let point = BabyBear::from_u64(2);
        let expanded = MultilinearPoint::expand_from_univariate(point, 2);

        // For n = 2, we expect [y^2, y]
        let expected = vec![point * point, point];
        assert_eq!(expanded.0, expected);
    }

    #[test]
    fn test_expand_from_univariate_three_variables() {
        let point = BabyBear::from_u64(5);
        let expanded = MultilinearPoint::expand_from_univariate(point, 3);

        // For n = 3, we expect [y^4, y^2, y]
        let expected = vec![point.exp_u64(4), point.exp_u64(2), point];
        assert_eq!(expanded.0, expected);
    }

    #[test]
    fn test_expand_from_univariate_large_variables() {
        let point = BabyBear::from_u64(7);
        let expanded = MultilinearPoint::expand_from_univariate(point, 5);

        // For n = 5, we expect [y^16, y^8, y^4, y^2, y]
        let expected =
            vec![point.exp_u64(16), point.exp_u64(8), point.exp_u64(4), point.exp_u64(2), point];
        assert_eq!(expanded.0, expected);
    }

    #[test]
    fn test_expand_from_univariate_identity() {
        let point = BabyBear::ONE;
        let expanded = MultilinearPoint::expand_from_univariate(point, 4);

        // Since 1^k = 1 for all k, the result should be [1, 1, 1, 1]
        let expected = vec![BabyBear::ONE; 4];
        assert_eq!(expanded.0, expected);
    }

    #[test]
    fn test_expand_from_univariate_zero() {
        let point = BabyBear::ZERO;
        let expanded = MultilinearPoint::expand_from_univariate(point, 4);

        // Since 0^k = 0 for all k, the result should be [0, 0, 0, 0]
        let expected = vec![BabyBear::ZERO; 4];
        assert_eq!(expanded.0, expected);
    }

    #[test]
    fn test_expand_from_univariate_empty() {
        let point = BabyBear::from_u64(9);
        let expanded = MultilinearPoint::expand_from_univariate(point, 0);

        // No variables should return an empty vector
        assert_eq!(expanded.0, vec![]);
    }

    #[test]
    fn test_expand_from_univariate_powers_correctness() {
        let point = BabyBear::from_u64(3);
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
        let ml_point = MultilinearPoint(vec![BabyBear::ZERO; 4]);
        let binary_point = BinaryHypercubePoint(0b0000);

        // eq_poly should evaluate to 1 since c_i = p_i = 0
        assert_eq!(ml_point.eq_poly(binary_point), BabyBear::ONE);
    }

    #[test]
    fn test_eq_poly_all_ones() {
        // Multilinear point (1,1,1,1)
        let ml_point = MultilinearPoint(vec![BabyBear::ONE; 4]);
        let binary_point = BinaryHypercubePoint(0b1111);

        // eq_poly should evaluate to 1 since c_i = p_i = 1
        assert_eq!(ml_point.eq_poly(binary_point), BabyBear::ONE);
    }

    #[test]
    fn test_eq_poly_mixed_bits_match() {
        // Multilinear point (1,0,1,0)
        let ml_point =
            MultilinearPoint(vec![BabyBear::ONE, BabyBear::ZERO, BabyBear::ONE, BabyBear::ZERO]);
        let binary_point = BinaryHypercubePoint(0b1010);

        // eq_poly should evaluate to 1 since c_i = p_i for all i
        assert_eq!(ml_point.eq_poly(binary_point), BabyBear::ONE);
    }

    #[test]
    fn test_eq_poly_mixed_bits_mismatch() {
        // Multilinear point (1,0,1,0)
        let ml_point =
            MultilinearPoint(vec![BabyBear::ONE, BabyBear::ZERO, BabyBear::ONE, BabyBear::ZERO]);
        let binary_point = BinaryHypercubePoint(0b1100); // Differs at second bit

        // eq_poly should evaluate to 0 since there is at least one mismatch
        assert_eq!(ml_point.eq_poly(binary_point), BabyBear::ZERO);
    }

    #[test]
    fn test_eq_poly_single_variable_match() {
        // Multilinear point (1)
        let ml_point = MultilinearPoint(vec![BabyBear::ONE]);
        let binary_point = BinaryHypercubePoint(0b1);

        // eq_poly should evaluate to 1 since c_1 = p_1 = 1
        assert_eq!(ml_point.eq_poly(binary_point), BabyBear::ONE);
    }

    #[test]
    fn test_eq_poly_single_variable_mismatch() {
        // Multilinear point (1)
        let ml_point = MultilinearPoint(vec![BabyBear::ONE]);
        let binary_point = BinaryHypercubePoint(0b0);

        // eq_poly should evaluate to 0 since c_1 != p_1
        assert_eq!(ml_point.eq_poly(binary_point), BabyBear::ZERO);
    }

    #[test]
    fn test_eq_poly_large_binary_number_match() {
        // Multilinear point (1,1,0,1,0,1,1,0)
        let ml_point = MultilinearPoint(vec![
            BabyBear::ONE,
            BabyBear::ONE,
            BabyBear::ZERO,
            BabyBear::ONE,
            BabyBear::ZERO,
            BabyBear::ONE,
            BabyBear::ONE,
            BabyBear::ZERO,
        ]);
        let binary_point = BinaryHypercubePoint(0b1101_0110);

        // eq_poly should evaluate to 1 since c_i = p_i for all i
        assert_eq!(ml_point.eq_poly(binary_point), BabyBear::ONE);
    }

    #[test]
    fn test_eq_poly_large_binary_number_mismatch() {
        // Multilinear point (1,1,0,1,0,1,1,0)
        let ml_point = MultilinearPoint(vec![
            BabyBear::ONE,
            BabyBear::ONE,
            BabyBear::ZERO,
            BabyBear::ONE,
            BabyBear::ZERO,
            BabyBear::ONE,
            BabyBear::ONE,
            BabyBear::ZERO,
        ]);
        let binary_point = BinaryHypercubePoint(0b1101_0111); // Last bit differs

        // eq_poly should evaluate to 0 since there is a mismatch
        assert_eq!(ml_point.eq_poly(binary_point), BabyBear::ZERO);
    }

    #[test]
    fn test_eq_poly_empty_vector() {
        // Empty Multilinear Point
        let ml_point = MultilinearPoint::<BabyBear>(vec![]);
        let binary_point = BinaryHypercubePoint(0);

        // eq_poly should evaluate to 1 since both are trivially equal
        assert_eq!(ml_point.eq_poly(binary_point), BabyBear::ONE);
    }

    #[test]
    fn test_eq_poly_outside_all_zeros() {
        let ml_point1 = MultilinearPoint(vec![BabyBear::ZERO; 4]);
        let ml_point2 = MultilinearPoint(vec![BabyBear::ZERO; 4]);

        assert_eq!(ml_point1.eq_poly_outside(&ml_point2), BabyBear::ONE);
    }

    #[test]
    fn test_eq_poly_outside_all_ones() {
        let ml_point1 = MultilinearPoint(vec![BabyBear::ONE; 4]);
        let ml_point2 = MultilinearPoint(vec![BabyBear::ONE; 4]);

        assert_eq!(ml_point1.eq_poly_outside(&ml_point2), BabyBear::ONE);
    }

    #[test]
    fn test_eq_poly_outside_mixed_match() {
        let ml_point1 =
            MultilinearPoint(vec![BabyBear::ONE, BabyBear::ZERO, BabyBear::ONE, BabyBear::ZERO]);
        let ml_point2 =
            MultilinearPoint(vec![BabyBear::ONE, BabyBear::ZERO, BabyBear::ONE, BabyBear::ZERO]);

        assert_eq!(ml_point1.eq_poly_outside(&ml_point2), BabyBear::ONE);
    }

    #[test]
    fn test_eq_poly_outside_mixed_mismatch() {
        let ml_point1 =
            MultilinearPoint(vec![BabyBear::ONE, BabyBear::ZERO, BabyBear::ONE, BabyBear::ZERO]);
        let ml_point2 =
            MultilinearPoint(vec![BabyBear::ZERO, BabyBear::ONE, BabyBear::ZERO, BabyBear::ONE]);

        assert_eq!(ml_point1.eq_poly_outside(&ml_point2), BabyBear::ZERO);
    }

    #[test]
    fn test_eq_poly_outside_single_variable_match() {
        let ml_point1 = MultilinearPoint(vec![BabyBear::ONE]);
        let ml_point2 = MultilinearPoint(vec![BabyBear::ONE]);

        assert_eq!(ml_point1.eq_poly_outside(&ml_point2), BabyBear::ONE);
    }

    #[test]
    fn test_eq_poly_outside_single_variable_mismatch() {
        let ml_point1 = MultilinearPoint(vec![BabyBear::ONE]);
        let ml_point2 = MultilinearPoint(vec![BabyBear::ZERO]);

        assert_eq!(ml_point1.eq_poly_outside(&ml_point2), BabyBear::ZERO);
    }

    #[test]
    fn test_eq_poly_outside_large_match() {
        let ml_point1 = MultilinearPoint(vec![
            BabyBear::ONE,
            BabyBear::ONE,
            BabyBear::ZERO,
            BabyBear::ONE,
            BabyBear::ZERO,
            BabyBear::ONE,
            BabyBear::ONE,
            BabyBear::ZERO,
        ]);
        let ml_point2 = MultilinearPoint(vec![
            BabyBear::ONE,
            BabyBear::ONE,
            BabyBear::ZERO,
            BabyBear::ONE,
            BabyBear::ZERO,
            BabyBear::ONE,
            BabyBear::ONE,
            BabyBear::ZERO,
        ]);

        assert_eq!(ml_point1.eq_poly_outside(&ml_point2), BabyBear::ONE);
    }

    #[test]
    fn test_eq_poly_outside_large_mismatch() {
        let ml_point1 = MultilinearPoint(vec![
            BabyBear::ONE,
            BabyBear::ONE,
            BabyBear::ZERO,
            BabyBear::ONE,
            BabyBear::ZERO,
            BabyBear::ONE,
            BabyBear::ONE,
            BabyBear::ZERO,
        ]);
        let ml_point2 = MultilinearPoint(vec![
            BabyBear::ONE,
            BabyBear::ONE,
            BabyBear::ZERO,
            BabyBear::ONE,
            BabyBear::ZERO,
            BabyBear::ONE,
            BabyBear::ONE,
            BabyBear::ONE, // Last bit differs
        ]);

        assert_eq!(ml_point1.eq_poly_outside(&ml_point2), BabyBear::ZERO);
    }

    #[test]
    fn test_eq_poly_outside_empty_vector() {
        let ml_point1 = MultilinearPoint::<BabyBear>(vec![]);
        let ml_point2 = MultilinearPoint::<BabyBear>(vec![]);

        assert_eq!(ml_point1.eq_poly_outside(&ml_point2), BabyBear::ONE);
    }

    #[test]
    #[should_panic]
    fn test_eq_poly_outside_different_lengths() {
        let ml_point1 = MultilinearPoint(vec![BabyBear::ONE, BabyBear::ZERO]);
        let ml_point2 = MultilinearPoint(vec![BabyBear::ONE, BabyBear::ZERO, BabyBear::ONE]);

        // Should panic because lengths do not match
        ml_point1.eq_poly_outside(&ml_point2);
    }

    #[test]
    fn test_eq_poly3_all_zeros() {
        let ml_point = MultilinearPoint(vec![BabyBear::ZERO; 4]);
        // (0,0,0,0) in base 3 = 0 * 3^3 + 0 * 3^2 + 0 * 3^1 + 0 * 3^0 = 0
        let ternary_point = 0;

        assert_eq!(ml_point.eq_poly3(ternary_point), BabyBear::ONE);
    }

    #[test]
    fn test_eq_poly3_all_ones() {
        let ml_point = MultilinearPoint(vec![BabyBear::ONE; 4]);
        // (1,1,1,1) in base 3 = 1 * 3^3 + 1 * 3^2 + 1 * 3^1 + 1 * 3^0
        let ternary_point = 1 * 3_i32.pow(3) + 1 * 3_i32.pow(2) + 1 * 3_i32.pow(1) + 1;

        assert_eq!(ml_point.eq_poly3(ternary_point as usize), BabyBear::ONE);
    }

    #[test]
    fn test_eq_poly3_all_twos() {
        let two = BabyBear::ONE + BabyBear::ONE;
        let ml_point = MultilinearPoint(vec![two; 4]);
        // (2,2,2,2) in base 3 = 2 * 3^3 + 2 * 3^2 + 2 * 3^1 + 2 * 3^0
        let ternary_point = 2 * 3_i32.pow(3) + 2 * 3_i32.pow(2) + 2 * 3_i32.pow(1) + 2;

        assert_eq!(ml_point.eq_poly3(ternary_point as usize), BabyBear::ONE);
    }

    #[test]
    fn test_eq_poly3_mixed_match() {
        let two = BabyBear::ONE + BabyBear::ONE;
        let ml_point = MultilinearPoint(vec![two, BabyBear::ONE, BabyBear::ZERO, BabyBear::ONE]);
        // (2,1,0,1) in base 3 = 2 * 3^3 + 1 * 3^2 + 0 * 3^1 + 1 * 3^0
        let ternary_point = 2 * 3_i32.pow(3) + 1 * 3_i32.pow(2) + 0 * 3_i32.pow(1) + 1;

        assert_eq!(ml_point.eq_poly3(ternary_point as usize), BabyBear::ONE);
    }

    #[test]
    fn test_eq_poly3_mixed_mismatch() {
        let two = BabyBear::ONE + BabyBear::ONE;
        let ml_point = MultilinearPoint(vec![two, BabyBear::ONE, BabyBear::ZERO, BabyBear::ONE]);
        // (2,2,0,1) differs at the second coordinate
        let ternary_point = 2 * 3_i32.pow(3) + 2 * 3_i32.pow(2) + 0 * 3_i32.pow(1) + 1;

        assert_eq!(ml_point.eq_poly3(ternary_point as usize), BabyBear::ZERO);
    }

    #[test]
    fn test_eq_poly3_single_variable_match() {
        let ml_point = MultilinearPoint(vec![BabyBear::ONE]);
        // (1) in base 3 = 1
        let ternary_point = 1;

        assert_eq!(ml_point.eq_poly3(ternary_point), BabyBear::ONE);
    }

    #[test]
    fn test_eq_poly3_single_variable_mismatch() {
        let ml_point = MultilinearPoint(vec![BabyBear::ONE]);
        // (2) in base 3 = 2
        let ternary_point = 2;

        assert_eq!(ml_point.eq_poly3(ternary_point), BabyBear::ZERO);
    }

    #[test]
    fn test_eq_poly3_large_match() {
        let two = BabyBear::ONE + BabyBear::ONE;
        let ml_point = MultilinearPoint(vec![
            two,
            BabyBear::ONE,
            BabyBear::ZERO,
            BabyBear::ONE,
            BabyBear::ZERO,
            two,
            BabyBear::ONE,
        ]);
        // (2,1,0,1,0,2,1) in base 3 = 2 * 3^6 + 1 * 3^5 + 0 * 3^4 + 1 * 3^3 + 0 * 3^2 + 2 * 3^1 + 1
        // * 3^0
        let ternary_point = 2 * 3_i32.pow(6) +
            1 * 3_i32.pow(5) +
            0 * 3_i32.pow(4) +
            1 * 3_i32.pow(3) +
            0 * 3_i32.pow(2) +
            2 * 3_i32.pow(1) +
            1;

        assert_eq!(ml_point.eq_poly3(ternary_point as usize), BabyBear::ONE);
    }

    #[test]
    fn test_eq_poly3_large_mismatch() {
        let two = BabyBear::ONE + BabyBear::ONE;
        let ml_point = MultilinearPoint(vec![
            two,
            BabyBear::ONE,
            BabyBear::ZERO,
            BabyBear::ONE,
            BabyBear::ZERO,
            two,
            BabyBear::ONE,
        ]);
        // (2,1,0,1,1,2,1) differs at the fifth coordinate
        let ternary_point = 2 * 3_i32.pow(6) +
            1 * 3_i32.pow(5) +
            0 * 3_i32.pow(4) +
            1 * 3_i32.pow(3) +
            1 * 3_i32.pow(2) +
            2 * 3_i32.pow(1) +
            1;

        assert_eq!(ml_point.eq_poly3(ternary_point as usize), BabyBear::ZERO);
    }

    #[test]
    fn test_eq_poly3_empty_vector() {
        let ml_point = MultilinearPoint::<BabyBear>(vec![]);
        let ternary_point = 0;

        assert_eq!(ml_point.eq_poly3(ternary_point), BabyBear::ONE);
    }

    #[test]
    #[should_panic]
    fn test_eq_poly3_invalid_ternary_value() {
        let ml_point = MultilinearPoint(vec![BabyBear::ONE, BabyBear::ZERO]);
        let ternary_point = 9; // Invalid ternary representation (not in {0,1,2})

        ml_point.eq_poly3(ternary_point);
    }
}
