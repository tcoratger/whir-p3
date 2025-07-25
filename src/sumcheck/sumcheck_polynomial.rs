use p3_field::Field;

use crate::poly::multilinear::MultilinearPoint;

/// Represents a polynomial stored in evaluation form over a ternary domain {0,1,2}^n.
///
/// This structure is uniquely determined by its evaluations over the ternary hypercube.
/// The order of storage follows big-endian lexicographic ordering with respect to the
/// evaluation points.
///
/// Given `n_variables`, the number of stored evaluations is `3^n_variables`.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SumcheckPolynomial<F> {
    /// Number of variables in the polynomial (defines the dimension of the evaluation domain).
    n_variables: usize,
    /// Vector of function evaluations at points in `{0,1,2}^n_variables`, stored in lexicographic
    /// order.
    evaluations: Vec<F>,
}

impl<F> SumcheckPolynomial<F>
where
    F: Field,
{
    /// Creates a new sumcheck polynomial with `n_variables` variables.
    ///
    /// # Parameters:
    /// - `evaluations`: A vector of function values evaluated on `{0,1,2}^n_variables`.
    /// - `n_variables`: The number of variables (determines the evaluation domain size).
    ///
    /// The vector `evaluations` **must** have a length of `3^n_variables`.
    #[must_use]
    pub const fn new(evaluations: Vec<F>, n_variables: usize) -> Self {
        Self {
            n_variables,
            evaluations,
        }
    }

    /// Returns the vector of stored evaluations.
    ///
    /// The order follows lexicographic ordering of the ternary hypercube `{0,1,2}^n_variables`:
    ///
    /// ```ignore
    /// evaluations[i] = h(x_1, x_2, ..., x_n)  where (x_1, ..., x_n) ∈ {0,1,2}^n
    /// ```
    #[must_use]
    pub fn evaluations(&self) -> &[F] {
        &self.evaluations
    }

    /// Computes the sum of function values over the Boolean hypercube `{0,1}^n_variables`.
    ///
    /// Instead of summing over all `3^n` evaluations, this method only sums over points where all
    /// coordinates are 0 or 1.
    ///
    /// Mathematically, this computes:
    /// ```ignore
    /// sum = ∑ f(x_1, ..., x_n)  where  (x_1, ..., x_n) ∈ {0,1}^n
    /// ```
    #[must_use]
    pub fn sum_over_boolean_hypercube(&self) -> F {
        (0..(1 << self.n_variables))
            .map(|point| self.evaluations[self.binary_to_ternary_index(point)])
            .sum()
    }

    /// Converts a binary index `(0..2^n)` to its corresponding ternary index `(0..3^n)`.
    ///
    /// This maps a Boolean hypercube `{0,1}^n` to the ternary hypercube `{0,1,2}^n`.
    ///
    /// Given a binary index:
    /// ```ignore
    /// binary_index = b_{n-1} b_{n-2} ... b_0  (in bits)
    /// ```
    /// The corresponding **ternary index** is computed as:
    /// ```ignore
    /// ternary_index = b_0 * 3^0 + b_1 * 3^1 + ... + b_{n-1} * 3^{n-1}
    /// ```
    ///
    /// # Example:
    /// ```ignore
    /// binary index 0b11  (3 in decimal)  →  ternary index 4
    /// binary index 0b10  (2 in decimal)  →  ternary index 3
    /// binary index 0b01  (1 in decimal)  →  ternary index 1
    /// binary index 0b00  (0 in decimal)  →  ternary index 0
    /// ```
    fn binary_to_ternary_index(&self, mut binary_index: usize) -> usize {
        let mut ternary_index = 0;
        let mut factor = 1;

        for _ in 0..self.n_variables {
            ternary_index += (binary_index & 1) * factor;
            // Move to next bit
            binary_index >>= 1;
            // Increase ternary place value
            factor *= 3;
        }

        ternary_index
    }

    /// Evaluates the polynomial at an arbitrary point in the domain `{0,1,2}^n`.
    ///
    /// Given an interpolation point `point ∈ F^n`, this computes:
    /// ```ignore
    /// f(point) = ∑ evaluations[i] * eq_poly3(i)
    /// ```
    /// where `eq_poly3(i)` is the Lagrange basis polynomial at index `i` in `{0,1,2}^n`.
    ///
    /// This allows evaluating the polynomial at non-discrete inputs beyond `{0,1,2}^n`.
    ///
    /// # Constraints:
    /// - The input `point` must have `n_variables` dimensions.
    #[must_use]
    #[inline]
    pub fn evaluate_at_point(&self, point: &MultilinearPoint<F>) -> F {
        assert_eq!(point.num_variables(), self.n_variables);
        self.evaluations
            .iter()
            .enumerate()
            .map(|(i, &eval)| eval * point.eq_poly3(i))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    use super::*;

    type F = BabyBear;

    #[test]
    #[should_panic]
    fn test_evaluate_at_point_wrong_dimensions() {
        let poly = SumcheckPolynomial::new(vec![F::ZERO; 9], 2);
        let point = MultilinearPoint(vec![F::ONE]); // Wrong dimension  
        assert_eq!(poly.evaluate_at_point(&point), F::ZERO);
    }

    #[test]
    fn test_all_zero_evaluations() {
        let poly = SumcheckPolynomial::new(vec![F::ZERO; 9], 2);
        assert_eq!(poly.sum_over_boolean_hypercube(), F::ZERO);

        let point = MultilinearPoint(vec![F::from_u64(5), F::from_u64(7)]);
        assert_eq!(poly.evaluate_at_point(&point), F::ZERO);
    }

    #[test]
    fn test_single_nonzero_evaluation() {
        let mut evaluations = vec![F::ZERO; 9];
        evaluations[4] = F::from_u64(100); // f(1,1) = 100  
        let poly = SumcheckPolynomial::new(evaluations, 2);

        // Only f(1,1) contributes to boolean sum
        assert_eq!(poly.sum_over_boolean_hypercube(), F::from_u64(100));
    }

    #[test]
    fn test_binary_to_ternary_systematic() {
        for n_vars in 1..=4 {
            let total_evals = 3_usize.pow(n_vars as u32);
            let poly = SumcheckPolynomial::new(vec![F::ZERO; total_evals], n_vars);

            let mut used_indices = std::collections::HashSet::new();
            for binary_idx in 0..(1 << n_vars) {
                let ternary_idx = poly.binary_to_ternary_index(binary_idx);
                assert!(ternary_idx < total_evals);
                assert!(used_indices.insert(ternary_idx), "Duplicate ternary index");
            }
        }
    }

    #[test]
    fn test_binary_to_ternary_index() {
        let poly = SumcheckPolynomial::new(vec![F::ZERO; 9], 2);

        // Binary indices: 0, 1, 2, 3 (for 2 variables: {00, 01, 10, 11})
        // Corresponding ternary indices: 0, 1, 3, 4
        assert_eq!(poly.binary_to_ternary_index(0b00), 0);
        assert_eq!(poly.binary_to_ternary_index(0b01), 1);
        assert_eq!(poly.binary_to_ternary_index(0b10), 3);
        assert_eq!(poly.binary_to_ternary_index(0b11), 4);
    }

    #[test]
    fn test_binary_to_ternary_index_three_vars() {
        let poly = SumcheckPolynomial::new(vec![F::ZERO; 27], 3);

        // Check conversion for all binary points in {0,1}^3
        assert_eq!(poly.binary_to_ternary_index(0b000), 0);
        assert_eq!(poly.binary_to_ternary_index(0b001), 1);
        assert_eq!(poly.binary_to_ternary_index(0b010), 3);
        assert_eq!(poly.binary_to_ternary_index(0b011), 4);
        assert_eq!(poly.binary_to_ternary_index(0b100), 9);
        assert_eq!(poly.binary_to_ternary_index(0b101), 10);
        assert_eq!(poly.binary_to_ternary_index(0b110), 12);
        assert_eq!(poly.binary_to_ternary_index(0b111), 13);
    }

    #[test]
    fn test_sum_over_boolean_hypercube_single_var() {
        // Test case for a single variable (n_variables = 1)
        // Function values at {0,1,2}: f(0) = 3, f(1) = 5, f(2) = 7
        let evaluations = vec![
            F::from_u64(3), // f(0)
            F::from_u64(5), // f(1)
            F::from_u64(7), // f(2)
        ];
        let poly = SumcheckPolynomial::new(evaluations, 1);

        // Sum over {0,1}: f(0) + f(1)
        let expected_sum = F::from_u64(3) + F::from_u64(5);
        assert_eq!(poly.sum_over_boolean_hypercube(), expected_sum);
    }

    #[test]
    fn test_sum_over_boolean_hypercube() {
        // Define a simple function f such that:
        // f(0,0) = 1, f(0,1) = 2, f(0,2) = 3
        // f(1,0) = 4, f(1,1) = 5, f(1,2) = 6
        // f(2,0) = 7, f(2,1) = 8, f(2,2) = 9
        let evaluations: Vec<_> = (1..=9).map(F::from_u64).collect();
        let poly = SumcheckPolynomial::new(evaluations, 2);

        // Sum over {0,1}^2: f(0,0) + f(0,1) + f(1,0) + f(1,1)
        let expected_sum = F::from_u64(1) + F::from_u64(2) + F::from_u64(4) + F::from_u64(5);
        let computed_sum = poly.sum_over_boolean_hypercube();
        assert_eq!(computed_sum, expected_sum);
    }

    #[test]
    fn test_sum_over_boolean_hypercube_three_vars() {
        // Test case for three variables (n_variables = 3)
        // Evaluations indexed lexicographically in {0,1,2}^3:
        //
        // f(0,0,0) = 1  f(0,0,1) = 2  f(0,0,2) = 3
        // f(0,1,0) = 4  f(0,1,1) = 5  f(0,1,2) = 6
        // f(0,2,0) = 7  f(0,2,1) = 8  f(0,2,2) = 9
        //
        // f(1,0,0) = 10 f(1,0,1) = 11 f(1,0,2) = 12
        // f(1,1,0) = 13 f(1,1,1) = 14 f(1,1,2) = 15
        // f(1,2,0) = 16 f(1,2,1) = 17 f(1,2,2) = 18
        //
        // f(2,0,0) = 19 f(2,0,1) = 20 f(2,0,2) = 21
        // f(2,1,0) = 22 f(2,1,1) = 23 f(2,1,2) = 24
        // f(2,2,0) = 25 f(2,2,1) = 26 f(2,2,2) = 27
        let evaluations: Vec<_> = (1..=27).map(F::from_u64).collect();
        let poly = SumcheckPolynomial::new(evaluations, 3);

        // Sum over {0,1}^3
        let expected_sum = F::from_u64(1)
            + F::from_u64(2)
            + F::from_u64(4)
            + F::from_u64(5)
            + F::from_u64(10)
            + F::from_u64(11)
            + F::from_u64(13)
            + F::from_u64(14);

        assert_eq!(poly.sum_over_boolean_hypercube(), expected_sum);
    }

    #[test]
    fn test_linearity_of_evaluate_at_point() {
        let evals1: Vec<_> = (1..=9).map(F::from_u64).collect();
        let evals2: Vec<_> = (10..=18).map(F::from_u64).collect();
        let poly1 = SumcheckPolynomial::new(evals1.clone(), 2);
        let poly2 = SumcheckPolynomial::new(evals2.clone(), 2);

        // Create combined polynomial: poly1 + 2*poly2
        let combined_evals: Vec<_> = evals1
            .iter()
            .zip(evals2.iter())
            .map(|(&e1, &e2)| e1 + e2.double())
            .collect();
        let combined_poly = SumcheckPolynomial::new(combined_evals, 2);

        let point = MultilinearPoint(vec![F::from_u64(3), F::from_u64(7)]);
        let expected = poly1.evaluate_at_point(&point) + poly2.evaluate_at_point(&point).double();
        assert_eq!(combined_poly.evaluate_at_point(&point), expected);
    }

    #[test]
    fn test_evaluate_at_point() {
        // Define a function f where evaluations are hardcoded:
        // f(0,0) = 1, f(0,1) = 2, f(0,2) = 3
        // f(1,0) = 4, f(1,1) = 5, f(1,2) = 6
        // f(2,0) = 7, f(2,1) = 8, f(2,2) = 9
        let evaluations: Vec<_> = (1..=9).map(F::from_u64).collect();
        let poly = SumcheckPolynomial::new(evaluations, 2);

        // Define an evaluation point (0.5, 0.5) as an interpolation between {0,1,2}^2
        let point = MultilinearPoint(vec![F::from_u64(1) / F::from_u64(2); 2]);

        let result = poly.evaluate_at_point(&point);

        // Compute the expected result using the full weighted sum:
        let expected_value = F::from_u64(1) * point.eq_poly3(0)
            + F::from_u64(2) * point.eq_poly3(1)
            + F::from_u64(3) * point.eq_poly3(2)
            + F::from_u64(4) * point.eq_poly3(3)
            + F::from_u64(5) * point.eq_poly3(4)
            + F::from_u64(6) * point.eq_poly3(5)
            + F::from_u64(7) * point.eq_poly3(6)
            + F::from_u64(8) * point.eq_poly3(7)
            + F::from_u64(9) * point.eq_poly3(8);

        assert_eq!(result, expected_value);
    }

    #[test]
    fn test_evaluate_at_point_three_vars() {
        // Define a function with three variables
        let evaluations: Vec<_> = (1..=27).map(F::from_u64).collect();
        let poly = SumcheckPolynomial::new(evaluations, 3);

        // Define an interpolation point (1/2, 1/2, 1/2) in {0,1,2}^3
        let point = MultilinearPoint(vec![F::from_u64(1) / F::from_u64(2); 3]);

        // Compute expected evaluation:
        let expected_value = (0..27)
            .map(|i| poly.evaluations[i] * point.eq_poly3(i))
            .sum::<F>();

        let computed_value = poly.evaluate_at_point(&point);
        assert_eq!(computed_value, expected_value);
    }
}
