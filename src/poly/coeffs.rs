use super::{evals::EvaluationsList, hypercube::BinaryHypercubePoint};
use crate::{ntt::wavelet::wavelet_transform, poly::multilinear::MultilinearPoint};
use p3_field::Field;
use rayon::join;
use std::mem::size_of;

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
#[derive(Debug, Clone)]
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
    /// Evaluates the polynomial at a binary hypercube point `(0,1)^n`.
    ///
    /// This is a special case of evaluation where the input consists only of 0s and 1s.
    ///
    /// Ensures that:
    /// - `point` is within the valid range `{0, ..., 2^n - 1}`
    /// - The length of `coeffs` matches `2^n`
    ///
    /// Uses a **fast path** by converting the point to a `MultilinearPoint`.
    pub fn evaluate_hypercube(&self, point: BinaryHypercubePoint) -> F {
        assert_eq!(self.coeffs.len(), 1 << self.num_variables);
        assert!(point.0 < (1 << self.num_variables));
        self.evaluate(&MultilinearPoint::from_binary_hypercube_point(point, self.num_variables))
    }

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
        assert_eq!(self.num_variables, point.n_variables());
        eval_multivariate(&self.coeffs, &point.0)
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
        let folding_factor = folding_randomness.n_variables();
        let coeffs = self
            .coeffs
            .chunks_exact(1 << folding_factor)
            .map(|chunk| eval_multivariate(chunk, &folding_randomness.0))
            .collect();

        Self { coeffs, num_variables: self.num_variables() - folding_factor }
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
#[allow(clippy::similar_names)]
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
            let (b0t, b1t) = {
                let work_size = (1 << 15) / size_of::<F>();
                if coeffs.len() > work_size {
                    join(|| eval_multivariate(b0t, tail), || eval_multivariate(b1t, tail))
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

        Self { coeffs, num_variables: num_variables as usize }
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

#[cfg(test)]
#[allow(clippy::should_panic_without_expect, clippy::similar_names)]
mod tests {
    use super::*;
    use crate::poly::evals::EvaluationsList;
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

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
    fn test_evaluate_hypercube() {
        let coeff0 = BabyBear::from_u64(10);
        let coeff1 = BabyBear::from_u64(3);
        let coeff2 = BabyBear::from_u64(5);
        let coeff3 = BabyBear::from_u64(7);

        let coeffs = vec![
            coeff0, // Constant term
            coeff1, // X_2 coefficient
            coeff2, // X_1 coefficient
            coeff3, // X_1 * X_2 coefficient
        ];
        let coeff_list = CoefficientList::new(coeffs);

        // Evaluations at hypercube points (expected values derived manually)
        // f(0,0) = coeffs[0]
        assert_eq!(coeff_list.evaluate_hypercube(BinaryHypercubePoint(0b00)), coeff0);
        // f(0,1) = coeffs[0] + coeffs[1]
        assert_eq!(coeff_list.evaluate_hypercube(BinaryHypercubePoint(0b01)), coeff0 + coeff1);
        // f(1,0) = coeffs[0] + coeffs[2]
        assert_eq!(coeff_list.evaluate_hypercube(BinaryHypercubePoint(0b10)), coeff0 + coeff2);
        // f(1,1) = coeffs[0] + coeffs[1] + coeffs[2] + coeffs[3]
        assert_eq!(
            coeff_list.evaluate_hypercube(BinaryHypercubePoint(0b11)),
            coeff0 + coeff1 + coeff2 + coeff3
        );
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
    fn test_evaluate_single_variable() {
        let coeff0 = BabyBear::from_u64(4);
        let coeff1 = BabyBear::from_u64(7);

        let coeffs = vec![coeff0, coeff1];
        let coeff_list = CoefficientList::new(coeffs);

        // Single-variable polynomial evaluations
        assert_eq!(coeff_list.evaluate_hypercube(BinaryHypercubePoint(0)), coeff0);
        assert_eq!(coeff_list.evaluate_hypercube(BinaryHypercubePoint(1)), coeff0 + coeff1);
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
        assert_eq!(folded.evaluate(&MultilinearPoint(vec![eval_value])), expected_eval);
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
        assert_eq!(folded.evaluate(&MultilinearPoint(vec![eval_x0])), expected_eval);
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

        let randomness: Vec<_> =
            (0..num_variables).map(|i| BabyBear::from_u64(35 * i as u64)).collect();
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
}
