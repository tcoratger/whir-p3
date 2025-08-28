use p3_field::{ExtensionField, Field};
use p3_util::log2_strict_usize;

/// Computes the partial contributions to the sumcheck polynomial from two evaluations.
///
/// Given two evaluations of a function and two evaluations of a weight:
/// - \( p(0), p(1) \) and \( w(0), w(1) \)
///
/// this function:
/// - Models \( p(x) = p(0) + (p(1) - p(0)) \cdot x \)
/// - Models \( w(x) = w(0) + (w(1) - w(0)) \cdot x \)
/// - Computes the contributions to:
///
/// \[
/// p(x) \cdot w(x) = \text{const term} + \text{linear term} \cdot x + \text{quadratic term} \cdot x^2
/// \]
///
/// Returns:
/// - The **constant coefficient** (\( p(0) \cdot w(0) \))
/// - The **quadratic coefficient** (\( (p(1) - p(0)) \cdot (w(1) - w(0)) \))
///
/// Note: the linear coefficient is reconstructed globally later.
#[inline]
pub(crate) fn sumcheck_quadratic<F, EF>((p, eq): (&[F], &[EF])) -> (EF, EF)
where
    F: Field,
    EF: ExtensionField<F>,
{
    // Compute the constant coefficient:
    // p(0) * w(0)
    let constant = eq[0] * p[0];

    // Compute the quadratic coefficient:
    // (p(1) - p(0)) * (w(1) - w(0))
    let quadratic = (eq[1] - eq[0]) * (p[1] - p[0]);

    (constant, quadratic)
}

/// Computes the evaluations of the select(p, X) polynomial and adds them to the weights.
///
/// This is the replacement for `eval_eq` that enables the "select trick".
///
/// ```text
/// select(pow(z), b) = z^int(b)
/// ```
///
/// # Arguments
/// * `point`: The multilinear point `p` (which corresponds to `pow(z)`).
/// * `weights`: The slice of weights to be updated.
/// * `combination_randomness`: The random scalar `alpha` to multiply by.
pub fn eval_select<F, EF>(point: &[F], weights: &mut [EF], combination_randomness: EF)
where
    F: Field,
    EF: ExtensionField<F>,
{
    let n = log2_strict_usize(weights.len());
    assert_eq!(
        point.len(),
        n,
        "Point dimension must match number of variables"
    );

    // This is an efficient way to calculate z^int(b) for each b in the hypercube.
    // It is equivalent to the logic in my previous (incorrect) answer but more concise.
    let z_powers = &point;

    // Iterate through each point `b` on the boolean hypercube, represented by its integer value `i`.
    for (i, weight) in weights.iter_mut().enumerate() {
        let mut res = combination_randomness;
        for j in 0..n {
            if (i >> j) & 1 == 1 {
                res *= z_powers[j];
            }
        }
        // Add the computed value to the weights vector.
        *weight += res;
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, extension::BinomialExtensionField};

    use super::*;

    type F = BabyBear;
    type EF4 = BinomialExtensionField<F, 4>;

    #[test]
    fn test_sumcheck_quadratic_base_case() {
        // Define evaluations over the base field
        let p = &[F::from_u64(3), F::from_u64(5)];
        // Define weights over the extension field
        let eq = &[EF4::from_u64(7), EF4::from_u64(11)];

        // Call the function under test
        let (c0, c2) = sumcheck_quadratic((p, eq));

        // Compute expected constant coefficient:
        // p(0) * w(0) = 3 * 7 = 21
        let expected_c0 = EF4::from_u64(21);

        // Compute expected quadratic coefficient:
        // (p(1) - p(0)) * (w(1) - w(0)) = (5 - 3) * (11 - 7) = 2 * 4 = 8
        let expected_c2 = EF4::from_u64(8);

        // Verify output matches expectations
        assert_eq!(c0, expected_c0);
        assert_eq!(c2, expected_c2);
    }

    #[test]
    fn test_sumcheck_quadratic_extension_case() {
        // Define both p and eq over the same extension field
        let p = &[EF4::from_u64(2), EF4::from_u64(6)];
        let eq = &[EF4::from_u64(5), EF4::from_u64(9)];

        // Call the function under test
        let (c0, c2) = sumcheck_quadratic((p, eq));

        // Compute expected constant coefficient:
        // p(0) * w(0) = 2 * 5 = 10
        let expected_c0 = EF4::from_u64(10);

        // Compute expected quadratic coefficient:
        // (p(1) - p(0)) * (w(1) - w(0)) = (6 - 2) * (9 - 5) = 4 * 4 = 16
        let expected_c2 = EF4::from_u64(16);

        // Verify output matches expectations
        assert_eq!(c0, expected_c0);
        assert_eq!(c2, expected_c2);
    }

    #[test]
    fn test_sumcheck_quadratic_zero_values() {
        // Test zero-value case
        let p = &[F::ZERO, F::ZERO];
        let eq = &[EF4::ZERO, EF4::ZERO];
        let (c0, c2) = sumcheck_quadratic((p, eq));
        assert_eq!(c0, EF4::ZERO);
        assert_eq!(c2, EF4::ZERO);
    }

    #[test]
    fn test_sumcheck_quadratic_one_values() {
        // Test unit value case
        let p = &[F::ONE, F::ONE];
        let eq = &[EF4::ONE, EF4::ONE];
        let (c0, c2) = sumcheck_quadratic((p, eq));
        assert_eq!(c0, EF4::ONE);
        assert_eq!(c2, EF4::ZERO); // (1-1) * (1-1) = 0
    }

    #[test]
    fn test_sumcheck_quadratic_large_values() {
        // Test large values
        let p = &[F::from_u64(1000), F::from_u64(2000)];
        let eq = &[EF4::from_u64(500), EF4::from_u64(1500)];
        let (c0, c2) = sumcheck_quadratic((p, eq));
        assert_eq!(c0, EF4::from_u64(500_000)); // 1000 * 500
        assert_eq!(c2, EF4::from_u64(1_000_000)); // 1000 * 1000
    }

    #[test]
    fn test_sumcheck_quadratic_mixed_field_types() {
        // Test : base field is zero but extension field is non-zero
        let p = &[F::ZERO, F::from_u64(5)];
        let eq = &[EF4::from_u64(3), EF4::from_u64(7)];
        let (c0, c2) = sumcheck_quadratic((p, eq));
        assert_eq!(c0, EF4::ZERO); // 0 * 3 = 0
        assert_eq!(c2, EF4::from_u64(20)); // 5 * 4 = 20
    }

    #[test]
    fn test_sumcheck_quadratic_linearity() {
        // Test linearity property：f(a+b) = f(a) + f(b)
        let p1 = &[F::from_u64(1), F::from_u64(2)];
        let p2 = &[F::from_u64(3), F::from_u64(4)];
        let eq = &[EF4::from_u64(5), EF4::from_u64(6)];

        let (c0_1, c2_1) = sumcheck_quadratic((p1, eq));
        let (c0_2, c2_2) = sumcheck_quadratic((p2, eq));

        let p_sum = &[p1[0] + p2[0], p1[1] + p2[1]];
        let (c0_sum, c2_sum) = sumcheck_quadratic((p_sum, eq));

        assert_eq!(c0_sum, c0_1 + c0_2);
        assert_eq!(c2_sum, c2_1 + c2_2);
    }

    #[test]
    fn test_eval_select_basic_2_vars() {
        // ARRANGE
        // The number of variables is 2. The domain size is 2^2 = 4.
        let n_vars = 2;
        // Initialize a weights vector to all zeros. This is where results will be accumulated.
        let mut weights = vec![F::ZERO; 1 << n_vars];
        // Define the point 'z' for the pow(z) map. Let z = 3.
        let z = F::from_u64(3);
        // The point passed to `eval_select` is pow(z) = (z^(2^0), z^(2^1)) = (3, 9).
        let point = &[z, z.square()];
        // The combination randomness (alpha) is set to 1 for simplicity.
        let combination_randomness = F::ONE;

        // ACT
        // Call the function under test.
        eval_select(point, &mut weights, combination_randomness);

        // ASSERT
        // Manually calculate the expected result for each point `b` on the hypercube.
        // The formula is `alpha * z^int(b)`.
        // b = (0,0), int(b) = 0. Expected: 1 * 3^0 = 1.
        assert_eq!(weights[0], F::from_u64(1));
        // b = (0,1), int(b) = 1. Expected: 1 * 3^1 = 3.
        assert_eq!(weights[1], F::from_u64(3));
        // b = (1,0), int(b) = 2. Expected: 1 * 3^2 = 9.
        assert_eq!(weights[2], F::from_u64(9));
        // b = (1,1), int(b) = 3. Expected: 1 * 3^3 = 27.
        assert_eq!(weights[3], F::from_u64(27));
    }

    #[test]
    fn test_eval_select_with_nonzero_alpha() {
        // ARRANGE
        // Same setup as the basic 2-variable test.
        let n_vars = 2;
        let mut weights = vec![F::ZERO; 1 << n_vars];
        let z = F::from_u64(2);
        let point = &[z, z.square()]; // pow(z) = (2, 4)
        // Use a different randomness value, alpha = 5.
        let combination_randomness = F::from_u64(5);

        // ACT
        eval_select(point, &mut weights, combination_randomness);

        // ASSERT
        // The expected results are the same as the basic case, but scaled by alpha.
        // b = (0,0), int(b) = 0. Expected: 5 * 2^0 = 5.
        assert_eq!(weights[0], F::from_u64(5));
        // b = (0,1), int(b) = 1. Expected: 5 * 2^1 = 10.
        assert_eq!(weights[1], F::from_u64(10));
        // b = (1,0), int(b) = 2. Expected: 5 * 2^2 = 20.
        assert_eq!(weights[2], F::from_u64(20));
        // b = (1,1), int(b) = 3. Expected: 5 * 2^3 = 40.
        assert_eq!(weights[3], F::from_u64(40));
    }

    #[test]
    fn test_eval_select_accumulates_correctly() {
        // ARRANGE
        // Initialize weights with pre-existing values [10, 100].
        let mut weights = vec![F::from_u64(10), F::from_u64(100)];
        let z1 = F::from_u64(2);
        let point1 = &[z1]; // pow(z1) = (2)
        let alpha1 = F::from_u64(3);

        // ACT (First Call)
        // First accumulation: add 3*2^int(b) to the weights.
        eval_select(point1, &mut weights, alpha1);

        // ASSERT (First Call)
        // b=0, int(b)=0. Expected: 10 + 3*2^0 = 13.
        assert_eq!(weights[0], F::from_u64(13));
        // b=1, int(b)=1. Expected: 100 + 3*2^1 = 106.
        assert_eq!(weights[1], F::from_u64(106));

        // ARRANGE (Second Call)
        let z2 = F::from_u64(4);
        let point2 = &[z2]; // pow(z2) = (4)
        let alpha2 = F::from_u64(5);

        // ACT (Second Call)
        // Second accumulation: add 5*4^int(b) to the updated weights.
        eval_select(point2, &mut weights, alpha2);

        // ASSERT (Second Call)
        // b=0, int(b)=0. Expected: 13 + 5*4^0 = 18.
        assert_eq!(weights[0], F::from_u64(18));
        // b=1, int(b)=1. Expected: 106 + 5*4^1 = 126.
        assert_eq!(weights[1], F::from_u64(126));
    }

    #[test]
    fn test_eval_select_2_vars() {
        // ARRANGE
        // The setup is identical to the first test, but types are adjusted.
        let n_vars = 2;
        // The weights buffer is now in the extension field.
        let mut weights = vec![EF4::ZERO; 1 << n_vars];
        // The point 'z' is in the base field.
        let z = F::from_u64(3);
        // The pow(z) map is also in the base field.
        let point = &[z, z.square()];
        // The combination randomness is in the extension field.
        let combination_randomness = EF4::from_u64(1);

        // ACT
        eval_select(point, &mut weights, combination_randomness);

        // ASSERT
        // The expected results are the same values, but lifted into the extension field.
        assert_eq!(weights[0], EF4::from_u64(1));
        assert_eq!(weights[1], EF4::from_u64(3));
        assert_eq!(weights[2], EF4::from_u64(9));
        assert_eq!(weights[3], EF4::from_u64(27));
    }

    #[test]
    fn test_eval_select_with_extension_alpha() {
        // ARRANGE
        let n_vars = 1;
        let mut weights = vec![EF4::ZERO; 1 << n_vars];
        // The point is from the base field.
        let point = &[F::from_u64(10)];
        // Create a non-trivial extension field element for alpha.
        let combination_randomness =
            EF4::from_basis_coefficients_slice(&[F::from_u64(2), F::from_u64(3), F::ZERO, F::ZERO])
                .unwrap();

        // ACT
        eval_select(point, &mut weights, combination_randomness);

        // ASSERT
        // b=0, int(b)=0. Expected: alpha * 10^0 = alpha.
        assert_eq!(weights[0], combination_randomness);
        // b=1, int(b)=1. Expected: alpha * 10^1 = 10 * alpha.
        assert_eq!(weights[1], combination_randomness * EF4::from_u64(10));
    }
}
