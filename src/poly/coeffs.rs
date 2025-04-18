use p3_field::{ExtensionField, Field};
#[cfg(feature = "parallel")]
use {
    rayon::{join, prelude::*},
    std::mem::size_of,
};

use super::{dense::WhirDensePolynomial, evals::EvaluationsList};
use crate::{ntt::wavelet::wavelet_transform, poly::multilinear::MultilinearPoint};

/// Represents a multilinear polynomial in coefficient form with `num_variables` variables.
///
/// The coefficients correspond to the **monomials** determined by the binary decomposition of their
/// index. If `num_variables = n`, then `coeffs[j]` corresponds to the monomial:
///
/// ```ignore
/// coeffs[j] * X_0^{b_0} * X_1^{b_1} * ... * X_{n-1}^{b_{n-1}}
/// ```
/// where `(b_0, b_1, ..., b_{n-1})` is the binary representation of `j`, with `b_{n-1}` being
/// the most significant bit.
///
/// **Example** (n = 3, variables X₀, X₁, X₂):
/// - `coeffs[0]` → Constant term (1)
/// - `coeffs[1]` → Coefficient of `X₂`
/// - `coeffs[2]` → Coefficient of `X₁`
/// - `coeffs[4]` → Coefficient of `X₀`
#[derive(Default, Debug, Clone)]
pub struct CoefficientList<F> {
    /// List of coefficients, stored in **lexicographic order**.
    /// For `n` variables, `coeffs.len() == 2^n`.
    coeffs: Vec<F>,
    /// Number of variables in the polynomial.
    num_variables: usize,
}

impl<F> CoefficientList<F>
where
    F: Field,
{
    /// Evaluates the polynomial at an arbitrary point in `F^n`.
    ///
    /// This generalizes evaluation beyond `(0,1)^n`, allowing fractional or arbitrary field
    /// elements.
    ///
    /// Uses multivariate Horner's method via `eval_multivariate()`, which recursively reduces
    /// the evaluation.
    ///
    /// Ensures that:
    /// - `point` has the same number of variables as the polynomial (`n`).
    pub fn evaluate(&self, point: &MultilinearPoint<F>) -> F {
        assert_eq!(self.num_variables, point.num_variables());
        eval_multivariate(&self.coeffs, &point.0)
    }

    /// Interprets self as a univariate polynomial (with coefficients of X^i in order of ascending
    /// i) and evaluates it at each point in `points`. We return the vector of evaluations.
    ///
    /// NOTE: For the `usual` mapping between univariate and multilinear polynomials, the
    /// coefficient ordering is such that for a single point x, we have (extending notation to a
    /// single point) self.evaluate_at_univariate(x) == self.evaluate([x^(2^n), x^(2^{n-1}),
    /// ..., x^2, x])
    pub fn evaluate_at_univariate(&self, points: &[F]) -> Vec<F> {
        // WhirDensePolynomial::from_coefficients_slice converts to a dense univariate polynomial.
        // The coefficient order is "coefficient of 1 first".
        let univariate = WhirDensePolynomial::from_coefficients_slice(&self.coeffs);
        points
            .iter()
            .map(|point| univariate.evaluate(point))
            .collect()
    }

    /// Folds the polynomial along high-indexed variables, reducing its dimensionality.
    ///
    /// Given a multilinear polynomial `f(X₀, ..., X_{n-1})`, this partially evaluates it at
    /// `folding_randomness`, returning a new polynomial in fewer variables:
    ///
    /// ```ignore
    /// f(X₀, ..., X_{m-1}, r₀, r₁, ..., r_k) → g(X₀, ..., X_{m-1})
    /// ```
    /// where `r₀, ..., r_k` are values from `folding_randomness`.
    ///
    /// - The number of variables decreases: `m = n - k`
    /// - Uses multivariate evaluation over chunks of coefficients.
    #[must_use]
    pub fn fold(&self, folding_randomness: &MultilinearPoint<F>) -> Self {
        let folding_factor = folding_randomness.num_variables();
        #[cfg(not(feature = "parallel"))]
        let coeffs = self
            .coeffs
            .chunks_exact(1 << folding_factor)
            .map(|coeffs| eval_multivariate(coeffs, &folding_randomness.0))
            .collect();
        #[cfg(feature = "parallel")]
        let coeffs = self
            .coeffs
            .par_chunks_exact(1 << folding_factor)
            .map(|coeffs| eval_multivariate(coeffs, &folding_randomness.0))
            .collect();

        Self {
            coeffs,
            num_variables: self.num_variables() - folding_factor,
        }
    }

    /// Evaluate self at `point`, where `point` is from a field extension extending the field over
    /// which the polynomial `self` is defined.
    ///
    /// Note that we only support the case where F is a prime field.
    pub fn evaluate_at_extension<EF: ExtensionField<F>>(&self, point: &MultilinearPoint<EF>) -> EF {
        assert_eq!(self.num_variables, point.num_variables());
        eval_extension(&self.coeffs, &point.0, EF::ONE)
    }
}

/// Multivariate Horner’s method for evaluating a polynomial at a general point.
///
/// Given `coeffs` (coefficients in lexicographic order) and `point = (x₀, x₁, ..., x_{n-1})`,
/// this recursively evaluates:
///
/// ```ignore
/// P(x₀, ..., x_{n-1}) = P_0(x₁, ..., x_{n-1}) + x₀ * P_1(x₁, ..., x_{n-1})
/// ```
///
/// where `P_0` and `P_1` are the even-indexed and odd-indexed coefficient subsets.
fn eval_multivariate<F: Field>(coeffs: &[F], point: &[F]) -> F {
    debug_assert_eq!(coeffs.len(), 1 << point.len());
    match point {
        [] => coeffs[0],
        [x] => coeffs[0] + coeffs[1] * *x,
        [x0, x1] => {
            let b0 = coeffs[0] + coeffs[1] * *x1;
            let b1 = coeffs[2] + coeffs[3] * *x1;
            b0 + b1 * *x0
        }
        [x0, x1, x2] => {
            let b00 = coeffs[0] + coeffs[1] * *x2;
            let b01 = coeffs[2] + coeffs[3] * *x2;
            let b10 = coeffs[4] + coeffs[5] * *x2;
            let b11 = coeffs[6] + coeffs[7] * *x2;
            let b0 = b00 + b01 * *x1;
            let b1 = b10 + b11 * *x1;
            b0 + b1 * *x0
        }
        [x0, x1, x2, x3] => {
            let b000 = coeffs[0] + coeffs[1] * *x3;
            let b001 = coeffs[2] + coeffs[3] * *x3;
            let b010 = coeffs[4] + coeffs[5] * *x3;
            let b011 = coeffs[6] + coeffs[7] * *x3;
            let b100 = coeffs[8] + coeffs[9] * *x3;
            let b101 = coeffs[10] + coeffs[11] * *x3;
            let b110 = coeffs[12] + coeffs[13] * *x3;
            let b111 = coeffs[14] + coeffs[15] * *x3;
            let b00 = b000 + b001 * *x2;
            let b01 = b010 + b011 * *x2;
            let b10 = b100 + b101 * *x2;
            let b11 = b110 + b111 * *x2;
            let b0 = b00 + b01 * *x1;
            let b1 = b10 + b11 * *x1;
            b0 + b1 * *x0
        }
        [x, tail @ ..] => {
            let (b0t, b1t) = coeffs.split_at(coeffs.len() / 2);
            #[cfg(not(feature = "parallel"))]
            let (b0t, b1t) = (eval_multivariate(b0t, tail), eval_multivariate(b1t, tail));
            #[cfg(feature = "parallel")]
            let (b0t, b1t) = {
                let work_size: usize = (1 << 15) / size_of::<F>();
                if coeffs.len() > work_size {
                    join(
                        || eval_multivariate(b0t, tail),
                        || eval_multivariate(b1t, tail),
                    )
                } else {
                    (eval_multivariate(b0t, tail), eval_multivariate(b1t, tail))
                }
            };
            b0t + b1t * *x
        }
    }
}

impl<F> CoefficientList<F> {
    /// Creates a `CoefficientList` from a vector of coefficients.
    ///
    /// Ensures that:
    /// - The length is a power of two (`2^n` for some `n`).
    /// - Computes `num_variables` as `log₂(coeffs.len())`.
    pub fn new(coeffs: Vec<F>) -> Self {
        let len = coeffs.len();
        assert!(len.is_power_of_two());
        let num_variables = len.ilog2();

        Self {
            coeffs,
            num_variables: num_variables as usize,
        }
    }

    /// Returns a reference to the stored coefficients.
    #[allow(clippy::missing_const_for_fn)]
    pub fn coeffs(&self) -> &[F] {
        &self.coeffs
    }

    /// Returns the number of variables (`n`).
    ///
    /// Since `coeffs.len() = 2^n`, this returns `log₂(coeffs.len())`.
    pub const fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// Returns the total number of coefficients (`2^n`).
    pub fn num_coeffs(&self) -> usize {
        self.coeffs.len()
    }

    /// Map the polynomial `self` from F[X_1,...,X_n] to E[X_1,...,X_n], where E is a field
    /// extension of F.
    ///
    /// Note that this is currently restricted to the case where F is a prime field.
    pub fn to_extension<E: ExtensionField<F>>(self) -> CoefficientList<E>
    where
        F: Field,
    {
        CoefficientList::new(self.coeffs.into_iter().map(E::from).collect())
    }
}

impl<F> From<CoefficientList<F>> for EvaluationsList<F>
where
    F: Field,
{
    fn from(value: CoefficientList<F>) -> Self {
        let mut evals = value.coeffs;
        wavelet_transform(&mut evals);
        Self::new(evals)
    }
}

/// Recursively evaluates a multilinear polynomial at an extension field point.
///
/// Given `coeffs` in lexicographic order, this computes:
/// ```ignore
/// eval_poly(X_0, ..., X_n) = sum(coeffs[i] * product(X_j for j in S(i)))
/// ```
/// where `S(i)` is the set of variables active in term `i` (based on its binary representation).
///
/// - Uses divide-and-conquer recursion:
///   - Splits `coeffs` into two halves for `X_0 = 0` and `X_0 = 1`.
///   - Recursively evaluates each half.
fn eval_extension<F, E>(coeff: &[F], eval: &[E], scalar: E) -> E
where
    F: Field,
    E: ExtensionField<F>,
{
    debug_assert_eq!(coeff.len(), 1 << eval.len());

    if let Some((&x, tail)) = eval.split_first() {
        let (low, high) = coeff.split_at(coeff.len() / 2);

        #[cfg(feature = "parallel")]
        {
            const PARALLEL_THRESHOLD: usize = 10;
            if tail.len() > PARALLEL_THRESHOLD {
                let (a, b) = rayon::join(
                    || eval_extension(low, tail, scalar),
                    || eval_extension(high, tail, scalar * x),
                );
                return a + b;
            }
        }

        // Default non-parallel execution
        eval_extension(low, tail, scalar) + eval_extension(high, tail, scalar * x)
    } else {
        scalar * E::from(coeff[0])
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};

    use super::*;
    use crate::poly::evals::EvaluationsList;

    #[test]
    fn test_coefficient_list_initialization() {
        let coeffs = vec![
            BabyBear::from_u64(3),
            BabyBear::from_u64(1),
            BabyBear::from_u64(4),
            BabyBear::from_u64(1),
        ];
        let coeff_list = CoefficientList::new(coeffs.clone());

        // Check that the coefficients are stored correctly
        assert_eq!(coeff_list.coeffs(), &coeffs);
        // Since len = 4 = 2^2, we expect num_variables = 2
        assert_eq!(coeff_list.num_variables(), 2);
    }

    #[test]
    fn test_evaluate_multilinear() {
        let coeff0 = BabyBear::from_u64(8);
        let coeff1 = BabyBear::from_u64(2);
        let coeff2 = BabyBear::from_u64(3);
        let coeff3 = BabyBear::from_u64(1);

        let coeffs = vec![coeff0, coeff1, coeff2, coeff3];
        let coeff_list = CoefficientList::new(coeffs);

        let x0 = BabyBear::from_u64(2);
        let x1 = BabyBear::from_u64(3);
        let point = MultilinearPoint(vec![x0, x1]);

        // Expected value based on multilinear evaluation
        let expected_value = coeff0 + coeff1 * x1 + coeff2 * x0 + coeff3 * x0 * x1;
        assert_eq!(coeff_list.evaluate(&point), expected_value);
    }

    #[test]
    fn test_folding() {
        let coeff0 = BabyBear::from_u64(6);
        let coeff1 = BabyBear::from_u64(2);
        let coeff2 = BabyBear::from_u64(4);
        let coeff3 = BabyBear::from_u64(8);

        let coeffs = vec![coeff0, coeff1, coeff2, coeff3];
        let coeff_list = CoefficientList::new(coeffs);

        let folding_value = BabyBear::from_u64(3);
        let folding_point = MultilinearPoint(vec![folding_value]);
        let folded = coeff_list.fold(&folding_point);

        let eval_value = BabyBear::from_u64(5);
        let expected_eval = coeff_list.evaluate(&MultilinearPoint(vec![eval_value, folding_value]));

        // Ensure folding preserves evaluation correctness
        assert_eq!(
            folded.evaluate(&MultilinearPoint(vec![eval_value])),
            expected_eval
        );
    }

    #[test]
    fn test_folding_multiple_variables() {
        let num_variables = 3;
        let coeffs: Vec<BabyBear> = (0..(1 << num_variables)).map(BabyBear::from_u64).collect();
        let coeff_list = CoefficientList::new(coeffs);

        let fold_x1 = BabyBear::from_u64(4);
        let fold_x2 = BabyBear::from_u64(2);
        let folding_point = MultilinearPoint(vec![fold_x1, fold_x2]);

        let folded = coeff_list.fold(&folding_point);

        let eval_x0 = BabyBear::from_u64(6);
        let full_point = MultilinearPoint(vec![eval_x0, fold_x1, fold_x2]);
        let expected_eval = coeff_list.evaluate(&full_point);

        // Ensure correctness of folding and evaluation
        assert_eq!(
            folded.evaluate(&MultilinearPoint(vec![eval_x0])),
            expected_eval
        );
    }

    #[test]
    fn test_coefficient_to_evaluations_conversion() {
        let coeff0 = BabyBear::from_u64(5);
        let coeff1 = BabyBear::from_u64(3);
        let coeff2 = BabyBear::from_u64(7);
        let coeff3 = BabyBear::from_u64(2);

        let coeffs = vec![coeff0, coeff1, coeff2, coeff3];
        let coeff_list = CoefficientList::new(coeffs);

        let evaluations = EvaluationsList::from(coeff_list);

        // Expected results after wavelet transform (manually derived)
        assert_eq!(evaluations[0], coeff0);
        assert_eq!(evaluations[1], coeff0 + coeff1);
        assert_eq!(evaluations[2], coeff0 + coeff2);
        assert_eq!(evaluations[3], coeff0 + coeff1 + coeff2 + coeff3);
    }

    #[test]
    fn test_num_variables_and_coeffs() {
        // 8 = 2^3, so num_variables = 3
        let coeffs = vec![BabyBear::from_u64(1); 8];
        let coeff_list = CoefficientList::new(coeffs);

        assert_eq!(coeff_list.num_variables(), 3);
        assert_eq!(coeff_list.num_coeffs(), 8);
    }

    #[test]
    #[should_panic]
    fn test_coefficient_list_empty() {
        let _coeff_list = CoefficientList::<BabyBear>::new(vec![]);
    }

    #[test]
    #[should_panic]
    fn test_coefficient_list_invalid_size() {
        // 7 is not a power of two
        let _coeff_list = CoefficientList::new(vec![BabyBear::from_u64(1); 7]);
    }

    #[test]
    fn test_folding_and_evaluation() {
        let num_variables = 10;
        let coeffs = (0..(1 << num_variables)).map(BabyBear::from_u64).collect();
        let coeffs_list = CoefficientList::new(coeffs);

        let randomness: Vec<_> = (0..num_variables)
            .map(|i| BabyBear::from_u64(35 * i as u64))
            .collect();
        for k in 0..num_variables {
            let fold_part = randomness[0..k].to_vec();
            let eval_part = randomness[k..randomness.len()].to_vec();

            let fold_random = MultilinearPoint(fold_part.clone());
            let eval_point = MultilinearPoint([eval_part.clone(), fold_part].concat());

            let folded = coeffs_list.fold(&fold_random);
            assert_eq!(
                folded.evaluate(&MultilinearPoint(eval_part)),
                coeffs_list.evaluate(&eval_point)
            );
        }
    }

    #[test]
    fn test_evaluate_at_extension_single_variable() {
        type F = BabyBear;
        type E = BinomialExtensionField<F, 4>;

        // Polynomial f(X) = 3 + 7X in base field
        let coeff0 = F::from_u64(3);
        let coeff1 = F::from_u64(7);
        let coeffs = vec![coeff0, coeff1];
        let coeff_list = CoefficientList::new(coeffs);

        let x = E::from_u64(2); // Evaluation at x = 2 in extension field
        let expected_value = E::from_u64(3) + E::from_u64(7) * x; // f(2) = 3 + 7 * 2
        let eval_result = coeff_list.evaluate_at_extension(&MultilinearPoint(vec![x]));

        assert_eq!(eval_result, expected_value);
    }

    #[test]
    fn test_evaluate_at_extension_two_variables() {
        type F = BabyBear;
        type E = BinomialExtensionField<F, 4>;

        // Polynomial f(X₀, X₁) = 2 + 5X₀ + 3X₁ + 7X₀X₁
        let coeffs = vec![
            F::from_u64(2), // Constant term
            F::from_u64(5), // X₁ term
            F::from_u64(3), // X₀ term
            F::from_u64(7), // X₀X₁ term
        ];
        let coeff_list = CoefficientList::new(coeffs);

        let x0 = E::from_u64(2);
        let x1 = E::from_u64(3);
        let expected_value =
            E::from_u64(2) + E::from_u64(5) * x1 + E::from_u64(3) * x0 + E::from_u64(7) * x0 * x1;
        let eval_result = coeff_list.evaluate_at_extension(&MultilinearPoint(vec![x0, x1]));

        assert_eq!(eval_result, expected_value);
    }

    #[test]
    fn test_evaluate_at_extension_three_variables() {
        type F = BabyBear;
        type E = BinomialExtensionField<F, 4>;

        // Polynomial: f(X₀, X₁, X₂) = 1 + 2X₂ + 3X₁ + 5X₁X₂ + 4X₀ + 6X₀X₂ + 7X₀X₁ + 8X₀X₁X₂
        let coeffs = vec![
            F::from_u64(1), // Constant term (000)
            F::from_u64(2), // X₂ (001)
            F::from_u64(3), // X₁ (010)
            F::from_u64(5), // X₁X₂ (011)
            F::from_u64(4), // X₀ (100)
            F::from_u64(6), // X₀X₂ (101)
            F::from_u64(7), // X₀X₁ (110)
            F::from_u64(8), // X₀X₁X₂ (111)
        ];
        let coeff_list = CoefficientList::new(coeffs);

        let x0 = E::from_u64(2);
        let x1 = E::from_u64(3);
        let x2 = E::from_u64(4);

        // Correct expected value based on the coefficient order
        let expected_value = E::from_u64(1)
            + E::from_u64(2) * x2
            + E::from_u64(3) * x1
            + E::from_u64(5) * x1 * x2
            + E::from_u64(4) * x0
            + E::from_u64(6) * x0 * x2
            + E::from_u64(7) * x0 * x1
            + E::from_u64(8) * x0 * x1 * x2;

        let eval_result = coeff_list.evaluate_at_extension(&MultilinearPoint(vec![x0, x1, x2]));

        assert_eq!(eval_result, expected_value);
    }

    #[test]
    fn test_evaluate_at_extension_zero_polynomial() {
        type F = BabyBear;
        type E = BinomialExtensionField<F, 4>;

        // Zero polynomial f(X) = 0
        let coeff_list = CoefficientList::new(vec![F::ZERO; 4]); // f(X₀, X₁) = 0

        let x0 = E::from_u64(5);
        let x1 = E::from_u64(7);
        let eval_result = coeff_list.evaluate_at_extension(&MultilinearPoint(vec![x0, x1]));

        assert_eq!(eval_result, E::ZERO);
    }

    #[test]
    fn test_evaluate_at_univariate_degree_one() {
        // Polynomial: f(x) = 3 + 4x
        let c0 = BabyBear::from_u64(3);
        let c1 = BabyBear::from_u64(4);
        let coeffs = vec![c0, c1];
        let poly = CoefficientList::new(coeffs);

        let p0 = BabyBear::from_u64(0);
        let p1 = BabyBear::from_u64(1);
        let p2 = BabyBear::from_u64(2);
        let p3 = BabyBear::from_u64(5);
        let points = vec![p0, p1, p2, p3];

        // Manually compute expected values from coeffs
        // f(x) = coeffs[0] + coeffs[1] * x
        let expected = vec![
            c0 + c1 * p0, // 3 + 4 * 0
            c0 + c1 * p1, // 3 + 4 * 1
            c0 + c1 * p2, // 3 + 4 * 2
            c0 + c1 * p3, // 3 + 4 * 5
        ];

        let result = poly.evaluate_at_univariate(&points);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_evaluate_at_univariate_degree_three_multiple_points() {
        // Polynomial: f(x) = 1 + 2x + 3x² + 4x³
        let c0 = BabyBear::from_u64(1);
        let c1 = BabyBear::from_u64(2);
        let c2 = BabyBear::from_u64(3);
        let c3 = BabyBear::from_u64(4);
        let coeffs = vec![c0, c1, c2, c3];
        let poly = CoefficientList::new(coeffs);

        let p0 = BabyBear::from_u64(0);
        let p1 = BabyBear::from_u64(1);
        let p2 = BabyBear::from_u64(2);
        let points = vec![p0, p1, p2];

        // f(x) = c0 + c1*x + c2*x² + c3*x³
        let expected = vec![
            c0 + c1 * p0 + c2 * p0.square() + c3 * p0.square() * p0,
            c0 + c1 * p1 + c2 * p1.square() + c3 * p1.square() * p1,
            c0 + c1 * p2 + c2 * p2.square() + c3 * p2.square() * p2,
        ];

        let result = poly.evaluate_at_univariate(&points);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_evaluate_at_univariate_equivalence_to_multilinear() {
        // Polynomial: f(x) = 5 + 6x + 7x² + 8x³
        let c0 = BabyBear::from_u64(5);
        let c1 = BabyBear::from_u64(6);
        let c2 = BabyBear::from_u64(7);
        let c3 = BabyBear::from_u64(8);
        let coeffs = vec![c0, c1, c2, c3];
        let poly = CoefficientList::new(coeffs);

        let x = BabyBear::from_u64(2);

        let expected = c0 + c1 * x + c2 * x.square() + c3 * x.square() * x;

        let result_univariate = poly.evaluate_at_univariate(&[x])[0];

        let ml_point = MultilinearPoint::expand_from_univariate(x, 2);
        let result_multilinear = poly.evaluate(&ml_point);

        assert_eq!(result_univariate, expected);
        assert_eq!(result_multilinear, expected);
    }
}
