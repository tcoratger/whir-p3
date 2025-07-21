use p3_field::{ExtensionField, Field};
use p3_matrix::Matrix;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::poly::{coeffs::CoefficientList, dense::WhirDensePolynomial, evals::EvaluationsList};

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

/// Generates the Lagrange basis polynomials for the domain {0, 1, ..., 2^k - 1}.
/// The i-th polynomial is 1 at `i` and 0 at other points in the domain.
pub fn univariate_selectors<F: Field>(k: usize) -> Vec<WhirDensePolynomial<F>> {
    let n_points = 1 << k;
    (0..n_points)
        .into_par_iter()
        .map(|i| {
            let evals = (0..n_points)
                .map(|j| (F::from_usize(j), if i == j { F::ONE } else { F::ZERO }))
                .collect::<Vec<_>>();
            // CHANGED: Use the correct method from the reference code
            WhirDensePolynomial::lagrange_interpolation(&evals)
                .expect("interpolation should succeed for this construction")
        })
        .collect()
}

/// Folds the first `k` variables of a multilinear polynomial using a set of `2^k` scalars.
/// This is equivalent to evaluating the polynomial at a point defined by the scalars.
pub fn fold_multilinear<F, EF>(evals: &EvaluationsList<F>, scalars: &[EF]) -> EvaluationsList<EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let k = scalars.len().trailing_zeros() as usize;
    assert_eq!(1 << k, scalars.len());

    let n_vars = evals.num_variables();
    assert!(n_vars >= k);
    let new_n_vars = n_vars - k;
    let new_size = 1 << new_n_vars;

    let folded_evals = (0..new_size)
        .into_par_iter()
        .map(|i| {
            // This is the dot product: sum_{j=0}^{2^k-1} scalars[j] * evals[i + j * new_size]
            // It correctly folds the first `k` variables.
            (0..scalars.len())
                .map(|j| scalars[j] * evals.evals()[i + j * new_size])
                .sum()
        })
        .collect();

    EvaluationsList::new(folded_evals)
}

pub fn interpolate_multilinear<F, EF, Mat>(evals_mat: &Mat, point: &[EF]) -> Vec<EF>
where
    F: Field,
    EF: ExtensionField<F>,
    Mat: Matrix<F> + Sync,
{
    let k = point.len();
    let n_evals = 1 << k;
    assert_eq!(evals_mat.width(), n_evals, "Matrix width must be 2^k");

    evals_mat
        .par_rows()
        .map(|row| {
            row.zip(0..n_evals)
                .map(|(p_at_b, b_u32)| {
                    // `p_at_b` is the value from the row iterator, e.g., p(b)
                    let eq_at_r_b = (0..k)
                        .map(|j| {
                            if (b_u32 >> j) & 1 == 1 {
                                point[j]
                            } else {
                                EF::ONE - point[j]
                            }
                        })
                        .product::<EF>();

                    eq_at_r_b * p_at_b
                })
                .sum::<EF>()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};
    use p3_matrix::dense::RowMajorMatrix;

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
    fn test_interpolate_multilinear_k2_single_poly() {
        //------------------------------------------------------------------
        // 1. DEFINE POLYNOMIAL FROM COEFFICIENTS
        //------------------------------------------------------------------
        // Let p(X₀, X₁) = c₀ + c₁X₀ + c₂X₁ + c₃X₀X₁.
        let c0 = F::from_u32(1);
        let c1 = F::from_u32(2);
        let c2 = F::from_u32(3);
        let c3 = F::from_u32(4);

        // Define a lambda for direct evaluation.
        let poly = |x0: EF4, x1: EF4| {
            EF4::from(c0) + x0 * EF4::from(c1) + x1 * EF4::from(c2) + x0 * x1 * EF4::from(c3)
        };

        //------------------------------------------------------------------
        // 2. COMPUTE EVALUATIONS OVER THE HYPERCUBE
        //------------------------------------------------------------------
        // We compute p(b) for b in {(0,0), (1,0), (0,1), (1,1)}.
        // The order corresponds to the integer value of the bits (b₁, b₀).
        let evals = vec![
            poly(F::ZERO.into(), F::ZERO.into()), // p(0,0)
            poly(F::ONE.into(), F::ZERO.into()),  // p(1,0)
            poly(F::ZERO.into(), F::ONE.into()),  // p(0,1)
            poly(F::ONE.into(), F::ONE.into()),   // p(1,1)
        ];
        let evals_mat = RowMajorMatrix::new(evals, 1 << 2);

        //------------------------------------------------------------------
        // 3. INTERPOLATE AND VERIFY
        //------------------------------------------------------------------
        // Choose a random point r = (r₀, r₁) to evaluate at.
        let point: Vec<EF4> = vec![EF4::from_u32(5), EF4::from_u32(10)];

        // The expected result is a direct evaluation using the lambda.
        let expected_eval = poly(point[0], point[1]);
        // expected = 1 + 2*5 + 3*10 + 4*5*10 = 1 + 10 + 30 + 200 = 241
        assert_eq!(expected_eval, EF4::from_u32(241));

        // Use our function to interpolate from the evaluations table.
        let result_vec = interpolate_multilinear(&evals_mat, &point);

        assert_eq!(result_vec.len(), 1);
        assert_eq!(
            result_vec[0], expected_eval,
            "Interpolated value did not match lambda evaluation"
        );
    }

    #[test]
    fn test_interpolate_multilinear_k3_batch_of_two() {
        //------------------------------------------------------------------
        // 1. SETUP
        //------------------------------------------------------------------
        let k = 3;

        // p1(X₀,X₁,X₂) is defined by evals p1(b) = b for b in {0..7}
        let evals1: Vec<F> = (0..8).map(F::from_u32).collect();
        // p2(X₀,X₁,X₂) is defined by evals p2(b) = 2*b for b in {0..7}
        let evals2: Vec<F> = (0..8).map(|i| F::from_u32(i * 2)).collect();

        let all_evals: Vec<F> = evals1.iter().chain(evals2.iter()).copied().collect();
        let evals_mat = RowMajorMatrix::new(all_evals, 1 << k);
        let point: Vec<EF4> = vec![
            EF4::from_u32(3), // r₀
            EF4::from_u32(5), // r₁
            EF4::from_u32(7), // r₂
        ];

        //------------------------------------------------------------------
        // 2. MANUAL COEFFICIENT AND LAMBDA CALCULATION
        //------------------------------------------------------------------
        // p(X₀,X₁,X₂) = c₀ + c₁X₀ + c₂X₁ + c₃X₂ + c₄X₀X₁ + c₅X₀X₂ + c₆X₁X₂ + c₇X₀X₁X₂

        // For p1(b) = b, the binary representation b = 1*b₀ + 2*b₁ + 4*b₂ gives the polynomial.
        // So, p1(X₀, X₁, X₂) = 1*X₀ + 2*X₁ + 4*X₂
        let poly1 = |x0: EF4, x1: EF4, x2: EF4| x0 + x1 * F::TWO + x2 * F::from_u32(4);

        // For p2(b) = 2*b, the polynomial is just twice the one for p1.
        // So, p2(X₀, X₁, X₂) = 2 * (1*X₀ + 2*X₁ + 4*X₂) = 2*X₀ + 4*X₁ + 8*X₂
        let poly2 =
            |x0: EF4, x1: EF4, x2: EF4| x0 * F::TWO + x1 * F::from_u32(4) + x2 * F::from_u32(8);

        // Calculate expected results using the lambda functions.
        let expected_eval1 = poly1(point[0], point[1], point[2]);
        let expected_eval2 = poly2(point[0], point[1], point[2]);
        // expected1 = 3 + 5*2 + 7*4 = 3 + 10 + 28 = 41
        assert_eq!(expected_eval1, EF4::from_u32(41));
        // expected2 = 3*2 + 5*4 + 7*8 = 6 + 20 + 56 = 82
        assert_eq!(expected_eval2, EF4::from_u32(82));

        //------------------------------------------------------------------
        // 3. ACTUAL COMPUTATION & ASSERTION
        //------------------------------------------------------------------
        let result_vec = interpolate_multilinear(&evals_mat, &point);
        assert_eq!(result_vec.len(), 2);
        assert_eq!(
            result_vec[0], expected_eval1,
            "Interpolated value for p1 was incorrect"
        );
        assert_eq!(
            result_vec[1], expected_eval2,
            "Interpolated value for p2 was incorrect"
        );
    }
}
