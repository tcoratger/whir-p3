use p3_field::{ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrix;
#[cfg(feature = "parallel")]
use {
    rayon::{join, prelude::*},
    std::mem::size_of,
};

use super::{dense::WhirDensePolynomial, evals::EvaluationsList};
use crate::{ntt::wavelet::wavelet_transform, poly::multilinear::MultilinearPoint};

/// A wrapper enum that holds coefficient data for a multilinear polynomial,
/// either over the base field `F` or an extension field `EF`.
///
/// This abstraction allows operating generically on both base and extension
/// field coefficients, similar to `EvaluationStorage`.
#[derive(Debug, Clone)]
pub(crate) enum CoefficientStorage<F, EF> {
    /// Coefficients over the base field `F`.
    Base(CoefficientList<F>),
    /// Coefficients over the extension field `EF`.
    Extension(CoefficientList<EF>),
}

impl<F, EF> CoefficientStorage<F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    /// Returns the number of variables in the stored coefficient list.
    ///
    /// # Returns
    /// - `usize`: The number of input variables of the underlying multilinear polynomial.
    pub(crate) const fn num_variables(&self) -> usize {
        match self {
            Self::Base(cl) => cl.num_variables(),
            Self::Extension(cl) => cl.num_variables(),
        }
    }

    /// Returns the number of coefficients (`2^num_variables`).
    pub(crate) fn num_coeffs(&self) -> usize {
        match self {
            Self::Base(cl) => cl.num_coeffs(),
            Self::Extension(cl) => cl.num_coeffs(),
        }
    }

    /// Folds the stored polynomial using the provided `folding_randomness`, returning a new
    /// `CoefficientList<EF>` with fewer variables.
    ///
    /// Works generically on both base and extension field representations.
    pub(crate) fn fold(&self, folding_randomness: &MultilinearPoint<EF>) -> CoefficientList<EF> {
        match self {
            Self::Base(cl) => cl.fold(folding_randomness),
            Self::Extension(cl) => cl.fold(folding_randomness),
        }
    }
}

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
    pub fn fold<EF>(&self, folding_randomness: &MultilinearPoint<EF>) -> CoefficientList<EF>
    where
        EF: ExtensionField<F>,
    {
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

        CoefficientList {
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
pub(crate) fn eval_multivariate<F, EF>(coeffs: &[F], point: &[EF]) -> EF
where
    F: Field,
    EF: ExtensionField<F>,
{
    debug_assert_eq!(coeffs.len(), 1 << point.len());
    match point {
        [] => coeffs[0].into(),
        [x] => *x * coeffs[1] + coeffs[0],
        [x0, x1] => {
            let b0 = *x1 * coeffs[1] + coeffs[0];
            let b1 = *x1 * coeffs[3] + coeffs[2];
            b0 + b1 * *x0
        }
        [x0, x1, x2] => {
            let b00 = *x2 * coeffs[1] + coeffs[0];
            let b01 = *x2 * coeffs[3] + coeffs[2];
            let b10 = *x2 * coeffs[5] + coeffs[4];
            let b11 = *x2 * coeffs[7] + coeffs[6];
            let b0 = b00 + b01 * *x1;
            let b1 = b10 + b11 * *x1;
            b0 + b1 * *x0
        }
        [x0, x1, x2, x3] => {
            let b000 = *x3 * coeffs[1] + coeffs[0];
            let b001 = *x3 * coeffs[3] + coeffs[2];
            let b010 = *x3 * coeffs[5] + coeffs[4];
            let b011 = *x3 * coeffs[7] + coeffs[6];
            let b100 = *x3 * coeffs[9] + coeffs[8];
            let b101 = *x3 * coeffs[11] + coeffs[10];
            let b110 = *x3 * coeffs[13] + coeffs[12];
            let b111 = *x3 * coeffs[15] + coeffs[14];
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
}

impl<F> From<CoefficientList<F>> for EvaluationsList<F>
where
    F: Field,
{
    fn from(value: CoefficientList<F>) -> Self {
        let mut evals = RowMajorMatrix::new_col(value.coeffs);
        wavelet_transform(&mut evals.as_view_mut());
        Self::new(evals.values)
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
    use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, extension::BinomialExtensionField};
    use proptest::prelude::*;
    use rand::{Rng, rng};

    use super::*;
    use crate::poly::evals::EvaluationsList;

    type F = BabyBear;
    type EF4 = BinomialExtensionField<F, 4>;

    #[test]
    fn test_coefficient_list_initialization() {
        let coeffs = vec![
            F::from_u64(3),
            F::from_u64(1),
            F::from_u64(4),
            F::from_u64(1),
        ];
        let coeff_list = CoefficientList::new(coeffs.clone());

        // Check that the coefficients are stored correctly
        assert_eq!(coeff_list.coeffs(), &coeffs);
        // Since len = 4 = 2^2, we expect num_variables = 2
        assert_eq!(coeff_list.num_variables(), 2);
    }

    #[test]
    fn test_evaluate_multilinear() {
        let coeff0 = F::from_u64(8);
        let coeff1 = F::from_u64(2);
        let coeff2 = F::from_u64(3);
        let coeff3 = F::from_u64(1);

        let coeffs = vec![coeff0, coeff1, coeff2, coeff3];
        let coeff_list = CoefficientList::new(coeffs);

        let x0 = F::from_u64(2);
        let x1 = F::from_u64(3);
        let point = MultilinearPoint(vec![x0, x1]);

        // Expected value based on multilinear evaluation
        let expected_value = coeff0 + coeff1 * x1 + coeff2 * x0 + coeff3 * x0 * x1;
        assert_eq!(coeff_list.evaluate(&point), expected_value);
    }

    #[test]
    fn test_folding() {
        let coeff0 = F::from_u64(6);
        let coeff1 = F::from_u64(2);
        let coeff2 = F::from_u64(4);
        let coeff3 = F::from_u64(8);

        let coeffs = vec![coeff0, coeff1, coeff2, coeff3];
        let coeff_list = CoefficientList::new(coeffs);

        let folding_value = F::from_u64(3);
        let folding_point = MultilinearPoint(vec![folding_value]);
        let folded = coeff_list.fold(&folding_point);

        let eval_value = F::from_u64(5);
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
        let coeffs: Vec<_> = (0..(1 << num_variables)).map(F::from_u64).collect();
        let coeff_list = CoefficientList::new(coeffs);

        let fold_x1 = F::from_u64(4);
        let fold_x2 = F::from_u64(2);
        let folding_point = MultilinearPoint(vec![fold_x1, fold_x2]);

        let folded = coeff_list.fold(&folding_point);

        let eval_x0 = F::from_u64(6);
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
        let coeff0 = F::from_u64(5);
        let coeff1 = F::from_u64(3);
        let coeff2 = F::from_u64(7);
        let coeff3 = F::from_u64(2);

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
        let coeffs = vec![F::from_u64(1); 8];
        let coeff_list = CoefficientList::new(coeffs);

        assert_eq!(coeff_list.num_variables(), 3);
        assert_eq!(coeff_list.num_coeffs(), 8);
    }

    #[test]
    #[should_panic]
    fn test_coefficient_list_empty() {
        let _coeff_list = CoefficientList::<F>::new(vec![]);
    }

    #[test]
    #[should_panic]
    fn test_coefficient_list_invalid_size() {
        // 7 is not a power of two
        let _coeff_list = CoefficientList::new(vec![F::from_u64(1); 7]);
    }

    #[test]
    fn test_folding_and_evaluation() {
        let num_variables = 10;
        let coeffs = (0..(1 << num_variables)).map(F::from_u64).collect();
        let coeffs_list = CoefficientList::new(coeffs);

        let randomness: Vec<_> = (0..num_variables)
            .map(|i| F::from_u64(35 * i as u64))
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
        // Polynomial f(X) = 3 + 7X in base field
        let coeff0 = F::from_u64(3);
        let coeff1 = F::from_u64(7);
        let coeffs = vec![coeff0, coeff1];
        let coeff_list = CoefficientList::new(coeffs);

        let x = EF4::from_u64(2); // Evaluation at x = 2 in extension field
        let expected_value = EF4::from_u64(3) + EF4::from_u64(7) * x; // f(2) = 3 + 7 * 2
        let eval_result = coeff_list.evaluate_at_extension(&MultilinearPoint(vec![x]));

        assert_eq!(eval_result, expected_value);
    }

    #[test]
    fn test_evaluate_at_extension_two_variables() {
        // Polynomial f(X₀, X₁) = 2 + 5X₀ + 3X₁ + 7X₀X₁
        let coeffs = vec![
            F::from_u64(2), // Constant term
            F::from_u64(5), // X₁ term
            F::from_u64(3), // X₀ term
            F::from_u64(7), // X₀X₁ term
        ];
        let coeff_list = CoefficientList::new(coeffs);

        let x0 = EF4::from_u64(2);
        let x1 = EF4::from_u64(3);
        let expected_value = EF4::from_u64(2)
            + EF4::from_u64(5) * x1
            + EF4::from_u64(3) * x0
            + EF4::from_u64(7) * x0 * x1;
        let eval_result = coeff_list.evaluate_at_extension(&MultilinearPoint(vec![x0, x1]));

        assert_eq!(eval_result, expected_value);
    }

    #[test]
    fn test_evaluate_at_extension_three_variables() {
        // Polynomial: f(X₀, X₁, X₂) = 1 + 2X₂ + 3X₁ + 4X₁X₂ + 5X₀ + 6X₀X₂ + 7X₀X₁ + 8X₀X₁X₂
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

        let x0 = EF4::from_u64(2);
        let x1 = EF4::from_u64(3);
        let x2 = EF4::from_u64(4);

        // Correct expected value based on the coefficient order
        let expected_value = EF4::from_u64(1)
            + EF4::from_u64(2) * x2
            + EF4::from_u64(3) * x1
            + EF4::from_u64(5) * x1 * x2
            + EF4::from_u64(4) * x0
            + EF4::from_u64(6) * x0 * x2
            + EF4::from_u64(7) * x0 * x1
            + EF4::from_u64(8) * x0 * x1 * x2;

        let eval_result = coeff_list.evaluate_at_extension(&MultilinearPoint(vec![x0, x1, x2]));

        assert_eq!(eval_result, expected_value);
    }

    #[test]
    fn test_evaluate_at_extension_zero_polynomial() {
        // Zero polynomial f(X) = 0
        let coeff_list = CoefficientList::new(vec![F::ZERO; 4]); // f(X₀, X₁) = 0

        let x0 = EF4::from_u64(5);
        let x1 = EF4::from_u64(7);
        let eval_result = coeff_list.evaluate_at_extension(&MultilinearPoint(vec![x0, x1]));

        assert_eq!(eval_result, EF4::ZERO);
    }

    #[test]
    fn test_evaluate_at_univariate_degree_one() {
        // Polynomial: f(x) = 3 + 4x
        let c0 = F::from_u64(3);
        let c1 = F::from_u64(4);
        let coeffs = vec![c0, c1];
        let poly = CoefficientList::new(coeffs);

        let p0 = F::from_u64(0);
        let p1 = F::from_u64(1);
        let p2 = F::from_u64(2);
        let p3 = F::from_u64(5);
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
        let c0 = F::from_u64(1);
        let c1 = F::from_u64(2);
        let c2 = F::from_u64(3);
        let c3 = F::from_u64(4);
        let coeffs = vec![c0, c1, c2, c3];
        let poly = CoefficientList::new(coeffs);

        let p0 = F::from_u64(0);
        let p1 = F::from_u64(1);
        let p2 = F::from_u64(2);
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
        let c0 = F::from_u64(5);
        let c1 = F::from_u64(6);
        let c2 = F::from_u64(7);
        let c3 = F::from_u64(8);
        let coeffs = vec![c0, c1, c2, c3];
        let poly = CoefficientList::new(coeffs);

        let x = F::from_u64(2);

        let expected = c0 + c1 * x + c2 * x.square() + c3 * x.square() * x;

        let result_univariate = poly.evaluate_at_univariate(&[x])[0];

        let ml_point = MultilinearPoint::expand_from_univariate(x, 2);
        let result_multilinear = poly.evaluate(&ml_point);

        assert_eq!(result_univariate, expected);
        assert_eq!(result_multilinear, expected);
    }

    #[test]
    fn test_eval_multivariate_with_extension_field_points() {
        let mut rng = rng();

        let c0 = F::from_u64(1);
        let c1 = F::from_u64(2);
        let c2 = F::from_u64(3);
        let c3 = F::from_u64(4);

        let coeffs = vec![c0, c1, c2, c3];
        assert_eq!(coeffs.len(), 4);

        let x0 = F::from_u64(5);
        let x1 = F::from_u64(7);

        let eval_extension = eval_multivariate(&coeffs, &[EF4::from(x0), EF4::from(x1)]);
        let expected = c0 + c1 * x1 + c2 * x0 + c3 * x0 * x1;

        assert_eq!(eval_extension, EF4::from(expected));

        // Compare with base field evaluation
        let eval_base = eval_multivariate(&coeffs, &[x0, x1]);
        assert_eq!(eval_extension, EF4::from(eval_base));

        // Now test with some random extension points
        let e: EF4 = rng.random();
        let f: EF4 = rng.random();

        let eval = eval_multivariate(&coeffs, &[e, f]);

        let expected =
            EF4::from(c0) + EF4::from(c1) * f + EF4::from(c2) * e + EF4::from(c3) * e * f;

        assert_eq!(eval, expected);
    }

    #[test]
    fn test_eval_multivariate_constant_poly() {
        let c = F::from_u64(42);
        let coeffs = vec![c];

        let points: Vec<EF4> = vec![]; // Zero variables

        let result = eval_multivariate(&coeffs, &points);
        assert_eq!(result, EF4::from(c));
    }

    #[test]
    fn test_eval_multivariate_all_zero_point() {
        let coeffs = vec![
            F::from_u64(11), // constant term
            F::from_u64(22), // x₁
            F::from_u64(33), // x₀
            F::from_u64(44), // x₀x₁
        ];
        let zeros = vec![EF4::ZERO, EF4::ZERO];

        let result = eval_multivariate(&coeffs, &zeros);
        assert_eq!(result, EF4::from(coeffs[0])); // Only constant survives
    }

    #[test]
    fn test_eval_multivariate_all_ones_point() {
        let coeffs = vec![
            F::from_u64(5), // 1
            F::from_u64(6), // x₁
            F::from_u64(7), // x₀
            F::from_u64(8), // x₀x₁
        ];

        let one = EF4::ONE;
        let result = eval_multivariate(&coeffs, &[one, one]);

        let expected = EF4::from(coeffs[0])
            + EF4::from(coeffs[1])
            + EF4::from(coeffs[2])
            + EF4::from(coeffs[3]);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_eval_multivariate_three_vars_mixed() {
        // We define a multilinear polynomial in three variables (X₀, X₁, X₂):
        //
        // f(X₀, X₁, X₂) =
        //    1                     (constant term)
        //  + 2·X₂                  (degree-1 term in X₂)
        //  + 3·X₁                  (degree-1 term in X₁)
        //  + 4·X₁·X₂               (degree-2 term X₁X₂)
        //  + 5·X₀                  (degree-1 term in X₀)
        //  + 6·X₀·X₂               (degree-2 term X₀X₂)
        //  + 7·X₀·X₁               (degree-2 term X₀X₁)
        //  + 8·X₀·X₁·X₂            (degree-3 term X₀X₁X₂)

        let coeffs = vec![
            F::from_u64(1), // coeff for constant term
            F::from_u64(2), // coeff for X₂
            F::from_u64(3), // coeff for X₁
            F::from_u64(4), // coeff for X₁·X₂
            F::from_u64(5), // coeff for X₀
            F::from_u64(6), // coeff for X₀·X₂
            F::from_u64(7), // coeff for X₀·X₁
            F::from_u64(8), // coeff for X₀·X₁·X₂
        ];

        // Now define evaluation point (x₀, x₁, x₂) ∈ EF4³:
        let x0 = EF4::from_u64(2); // x₀ = 2 (embedded from base field)
        let x1 = EF4::from_u64(3); // x₁ = 3
        let x2 = EF4::from_basis_coefficients_iter(
            [F::new(9), F::new(0), F::new(1), F::new(0)].into_iter(),
        )
        .unwrap(); // x₂ is a custom EF4 element

        // Evaluate the multilinear polynomial at (x₀, x₁, x₂)
        let result = eval_multivariate(&coeffs, &[x0, x1, x2]);

        // Manually expand the expected result:
        //
        // f(x₀,x₁,x₂) =
        //    1
        //  + 2 * x₂
        //  + 3 * x₁
        //  + 4 * (x₁ * x₂)
        //  + 5 * x₀
        //  + 6 * (x₀ * x₂)
        //  + 7 * (x₀ * x₁)
        //  + 8 * (x₀ * x₁ * x₂)
        //
        // Each coefficient is promoted from F to EF4 before multiplication.

        let expected = EF4::from(coeffs[0])
            + EF4::from(coeffs[1]) * x2
            + EF4::from(coeffs[2]) * x1
            + EF4::from(coeffs[3]) * x1 * x2
            + EF4::from(coeffs[4]) * x0
            + EF4::from(coeffs[5]) * x0 * x2
            + EF4::from(coeffs[6]) * x0 * x1
            + EF4::from(coeffs[7]) * x0 * x1 * x2;

        // Final assertion: evaluated result must match the manual expansion
        assert_eq!(result, expected);
    }

    #[test]
    fn test_fold_with_extension_one_var() {
        // Polynomial: f(X₀, X₁) = 1 + 2X₁ + 3X₀ + 4X₀X₁
        let coeffs = vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ];
        let poly = CoefficientList::new(coeffs);

        // Fold with X₁ = 5 (in EF4)
        let r1 = EF4::from_u64(5);
        let folded = poly.fold(&MultilinearPoint(vec![r1]));

        // Should produce polynomial in X₀ only
        for x0_f in 0..10 {
            let x0 = EF4::from_u64(x0_f);
            let full_point = MultilinearPoint(vec![x0, r1]);
            let folded_point = MultilinearPoint(vec![x0]);

            let expected = poly.evaluate_at_extension(&full_point);
            let actual = folded.evaluate(&folded_point);
            assert_eq!(expected, actual);
        }
    }

    #[test]
    fn test_fold_with_extension_two_vars() {
        // f(X₀, X₁, X₂) = 1 + 2X₂ + 3X₁ + 4X₁X₂ + 5X₀ + 6X₀X₂ + 7X₀X₁ + 8X₀X₁X₂
        let coeffs = (1..=8).map(F::from_u64).collect::<Vec<_>>();
        let poly = CoefficientList::new(coeffs);

        let r1 = EF4::from_u64(9);
        let r2 = EF4::from_basis_coefficients_iter(
            [F::new(2), F::new(0), F::new(3), F::new(1)].into_iter(),
        )
        .unwrap();

        let folded = poly.fold(&MultilinearPoint(vec![r1, r2]));

        for x0_f in 0..10 {
            let x0 = EF4::from_u64(x0_f);
            let full_point = MultilinearPoint(vec![x0, r1, r2]);
            let folded_point = MultilinearPoint(vec![x0]);

            let expected = poly.evaluate_at_extension(&full_point);
            let actual = folded.evaluate(&folded_point);
            assert_eq!(expected, actual);
        }
    }

    #[test]
    fn test_fold_with_zero_extension_randomness() {
        // f(X₀, X₁) = 1 + 2X₁ + 3X₀ + 4X₀X₁
        let coeffs = vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ];
        let poly = CoefficientList::new(coeffs);

        let zero = EF4::ZERO;
        let folded = poly.fold(&MultilinearPoint(vec![zero]));

        // Should be equivalent to evaluating X₁ = 0 in original poly
        for x0_f in 0..5 {
            let x0 = EF4::from_u64(x0_f);
            let full_point = MultilinearPoint(vec![x0, zero]);
            let folded_point = MultilinearPoint(vec![x0]);

            let expected = poly.evaluate_at_extension(&full_point);
            let actual = folded.evaluate(&folded_point);
            assert_eq!(expected, actual);
        }
    }

    #[test]
    fn test_coefficient_storage_num_variables_and_coeffs_base() {
        let coeffs: Vec<F> = (0..8).map(F::from_u64).collect();
        let list = CoefficientList::new(coeffs);
        let storage = CoefficientStorage::<F, EF4>::Base(list);

        assert_eq!(storage.num_variables(), 3); // log2(8)
        assert_eq!(storage.num_coeffs(), 8);
    }

    #[test]
    fn test_coefficient_storage_num_variables_and_coeffs_extension() {
        let coeffs: Vec<EF4> = (0..16).map(EF4::from_u64).collect();
        let list = CoefficientList::new(coeffs);
        let storage = CoefficientStorage::<F, EF4>::Extension(list);

        assert_eq!(storage.num_variables(), 4); // log2(16)
        assert_eq!(storage.num_coeffs(), 16);
    }

    #[test]
    fn test_coefficient_list_to_evaluations_list_three_variables() {
        // Define a multilinear polynomial:
        //
        // f(X₀, X₁, X₂) =
        //    1                  (constant term)
        //  + 2·X₂               (degree-1 term in X₂)
        //  + 3·X₁               (degree-1 term in X₁)
        //  + 4·X₁·X₂            (degree-2 term X₁X₂)
        //  + 5·X₀               (degree-1 term in X₀)
        //  + 6·X₀·X₂            (degree-2 term X₀X₂)
        //  + 7·X₀·X₁            (degree-2 term X₀X₁)
        //  + 8·X₀·X₁·X₂         (degree-3 term X₀X₁X₂)

        let coeffs = vec![
            F::from_u64(1), // index 0: constant
            F::from_u64(2), // index 1: X₂
            F::from_u64(3), // index 2: X₁
            F::from_u64(4), // index 3: X₁·X₂
            F::from_u64(5), // index 4: X₀
            F::from_u64(6), // index 5: X₀·X₂
            F::from_u64(7), // index 6: X₀·X₁
            F::from_u64(8), // index 7: X₀·X₁·X₂
        ];

        let coeff_list = CoefficientList::new(coeffs);

        // Convert to evaluations list via wavelet transform
        let eval_list = EvaluationsList::from(coeff_list);

        // Manually compute expected evaluations.
        //
        // Evaluations correspond to values of f(x₀,x₁,x₂) at all (x₀,x₁,x₂) ∈ {0,1}³
        //
        // Order of points is lex smallest bit last:
        // [ (x₂, x₁, x₀) ] ← binary counting
        //
        // | Index | (X₀, X₁, X₂) | f(X₀,X₁,X₂) |
        // |:-----:|:------------:|:-----------:|
        // |   0   | (0, 0, 0)    | 1
        // |   1   | (0, 0, 1)    | 1 + 2
        // |   2   | (0, 1, 0)    | 1 + 3
        // |   3   | (0, 1, 1)    | 1 + 2 + 3 + 4
        // |   4   | (1, 0, 0)    | 1 + 5
        // |   5   | (1, 0, 1)    | 1 + 2 + 5 + 6
        // |   6   | (1, 1, 0)    | 1 + 3 + 5 + 7
        // |   7   | (1, 1, 1)    | 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8

        let expected = vec![
            F::from_u64(1),
            F::from_u64(1 + 2),
            F::from_u64(1 + 3),
            F::from_u64(1 + 2 + 3 + 4),
            F::from_u64(1 + 5),
            F::from_u64(1 + 2 + 5 + 6),
            F::from_u64(1 + 3 + 5 + 7),
            F::from_u64(1 + 2 + 3 + 4 + 5 + 6 + 7 + 8),
        ];

        assert_eq!(eval_list.evals(), &expected);
    }

    proptest! {
        #[test]
        fn prop_fold_equivalence_f_vs_ef(
            coeffs in proptest::collection::vec(0u64..u64::MAX, 8),
            r0 in 0u64..u64::MAX,
            r1 in 0u64..u64::MAX
        ) {
            let coeffs_f: Vec<F> = coeffs.into_iter().map(F::from_u64).collect();
            let coeffs_ef: Vec<EF4> = coeffs_f.clone().into_iter().map(EF4::from).collect();
            let base = CoefficientList::new(coeffs_f);
            let ext = CoefficientList::new(coeffs_ef);

            let s_base : CoefficientStorage<F, EF4> = CoefficientStorage::Base(base);
            let s_ext : CoefficientStorage<F, EF4> = CoefficientStorage::Extension(ext);

            let folding_point = MultilinearPoint(vec![EF4::from_u64(r0), EF4::from_u64(r1)]);

            let folded_base = s_base.fold(&folding_point);
            let folded_ext = s_ext.fold(&folding_point);

            prop_assert_eq!(folded_base.coeffs(), folded_ext.coeffs());
            prop_assert_eq!(folded_base.num_variables(), folded_ext.num_variables());

            for x0 in 0..4 {
                let x0_ext = EF4::from_u64(x0);
                let point = MultilinearPoint(vec![x0_ext]);
                prop_assert_eq!(folded_base.evaluate(&point), folded_ext.evaluate(&point));
            }
        }
    }
}
