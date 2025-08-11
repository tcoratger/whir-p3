use p3_field::{ExtensionField, Field};

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

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};

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
}
