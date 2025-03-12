use crate::poly::{coeffs::CoefficientList, evals::EvaluationsList, multilinear::MultilinearPoint};
use p3_field::Field;
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    join,
    slice::ParallelSlice,
};

use super::proof::SumcheckPolynomial;

/// Implements the single-round sumcheck protocol for verifying a multilinear polynomial evaluation.
///
/// This struct is responsible for:
/// - Transforming a polynomial from coefficient representation into evaluation form.
/// - Constructing and evaluating equality constraints using interpolation.
/// - Computing the sumcheck polynomial, which is a quadratic polynomial in a single variable.
///
/// Given a multilinear polynomial `p(X1, ..., Xn)`, the sumcheck polynomial is computed as:
///
/// ```text
/// S(X) = ∑ p(β) * eq(β)
/// ```
///
/// where:
/// - `β` ranges over evaluation points in `{0,1,2}^k` (with `k=1` in this implementation).
/// - `eq(X)` is the equality polynomial encoding constraints.
/// - The result `S(X)` is a quadratic polynomial in `X`.
///
/// The sumcheck protocol ensures that the claimed sum is correct.
#[derive(Debug)]
pub struct SumcheckSingle<F> {
    /// Evaluations of the polynomial `p(X)`.
    evaluation_of_p: EvaluationsList<F>,
    /// Evaluations of the equality constraint polynomial `eq(X)`.
    evaluation_of_equality: EvaluationsList<F>,
    /// Number of variables `n` in the multilinear polynomial.
    num_variables: usize,
    /// The computed sum, which serves as a verification check in sumcheck.
    sum: F,
}

impl<F> SumcheckSingle<F>
where
    F: Field,
{
    /// Constructs a new `SumcheckSingle` instance from polynomial coefficients.
    ///
    /// This function:
    /// - Converts `coeffs` into evaluation form.
    /// - Initializes an empty equality constraint table.
    /// - Applies equality constraints if provided.
    pub fn new(
        coeffs: CoefficientList<F>,
        points: &[MultilinearPoint<F>],
        combination_randomness: &[F],
        evaluations: &[F],
    ) -> Self {
        assert_eq!(points.len(), combination_randomness.len());
        assert_eq!(points.len(), evaluations.len());
        let num_variables = coeffs.num_variables();

        let mut prover = Self {
            evaluation_of_p: coeffs.into(),
            evaluation_of_equality: EvaluationsList::new(vec![F::ZERO; 1 << num_variables]),
            num_variables,
            sum: F::ZERO,
        };

        prover.add_new_equality(points, combination_randomness, evaluations);
        prover
    }

    /// Adds new equality constraints to the polynomial.
    ///
    /// Computes:
    ///
    /// ```text
    /// eq(X) = ∑ ε_i * eq_{z_i}(X)
    /// ```
    ///
    /// where:
    /// - `ε_i` are weighting factors.
    /// - `eq_{z_i}(X)` is the equality polynomial ensuring `X = z_i`.
    pub fn add_new_equality(
        &mut self,
        points: &[MultilinearPoint<F>],
        combination_randomness: &[F],
        evaluations: &[F],
    ) {
        assert_eq!(combination_randomness.len(), points.len());
        assert_eq!(combination_randomness.len(), evaluations.len());

        points.iter().zip(combination_randomness.iter().zip(evaluations)).for_each(
            |(point, (&rand, &eval))| {
                eval_eq(&point.0, self.evaluation_of_equality.evals_mut(), rand);
                self.sum += rand * eval;
            },
        );
    }

    /// Computes the sumcheck polynomial `S(X)`, which is quadratic.
    ///
    /// The sumcheck polynomial is given by:
    ///
    /// ```text
    /// S(X) = ∑ p(β) * eq(β)
    /// ```
    ///
    /// where `β` are points in `{0,1,2}^1`, and `S(X)` is a **quadratic polynomial**.
    pub fn compute_sumcheck_polynomial(&self) -> SumcheckPolynomial<F> {
        assert!(self.num_variables >= 1);

        // Compute the quadratic coefficients using parallel reduction
        let (c0, c2) = self
            .evaluation_of_p
            .evals()
            .par_chunks_exact(2)
            .zip(self.evaluation_of_equality.evals().par_chunks_exact(2))
            .map(|(p_at, eq_at)| {
                // Convert evaluations to coefficients for the linear fns p and eq.
                let (p_0, p_1) = (p_at[0], p_at[1] - p_at[0]);
                let (eq_0, eq_1) = (eq_at[0], eq_at[1] - eq_at[0]);

                // Now we need to add the contribution of p(x) * eq(x)
                (p_0 * eq_0, p_1 * eq_1)
            })
            .reduce(|| (F::ZERO, F::ZERO), |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2));

        // Compute the middle coefficient using sum rule: sum = 2 * c0 + c1 + c2
        let c1 = self.sum - c0.double() - c2;

        // Evaluate the quadratic polynomial at 0, 1, 2
        let eval_0 = c0;
        let eval_1 = c0 + c1 + c2;
        let eval_2 = eval_1 + c1 + c2 + c2.double();

        SumcheckPolynomial::new(vec![eval_0, eval_1, eval_2], 1)
    }
}

/// Computes the equality polynomial evaluations efficiently.
///
/// Given an evaluation point vector `eval`, the function computes
/// the equality polynomial recursively using the formula:
///
/// ```text
/// eq(X) = ∏ (1 - X_i + 2X_i z_i)
/// ```
///
/// where `z_i` are the constraint points.
fn eval_eq<F: Field>(eval: &[F], out: &mut [F], scalar: F) {
    const PARALLEL_THRESHOLD: usize = 10;

    // Ensure that the output buffer size is correct:
    // It should be of size `2^n`, where `n` is the number of variables.
    debug_assert_eq!(out.len(), 1 << eval.len());

    // Base case: When there are no more variables to process, update the final value.
    if let Some((&x, tail)) = eval.split_first() {
        // Divide the output buffer into two halves: one for `X_i = 0` and one for `X_i = 1`
        let (low, high) = out.split_at_mut(out.len() / 2);

        // Compute weight updates for the two branches:
        // - `s0` corresponds to the case when `X_i = 0`
        // - `s1` corresponds to the case when `X_i = 1`
        //
        // Mathematically, this follows the recurrence:
        // ```text
        // eq_{X1, ..., Xn}(X) = (1 - X_1) * eq_{X2, ..., Xn}(X) + X_1 * eq_{X2, ..., Xn}(X)
        // ```
        let s1 = scalar * x; // Contribution when `X_i = 1`
        let s0 = scalar - s1; // Contribution when `X_i = 0`

        // Use parallel execution if the number of remaining variables is large.
        if tail.len() > PARALLEL_THRESHOLD {
            join(|| eval_eq(tail, low, s0), || eval_eq(tail, high, s1));
        } else {
            eval_eq(tail, low, s0);
            eval_eq(tail, high, s1);
        }
    } else {
        // Leaf case: Add the accumulated scalar to the final output slot.
        out[0] += scalar;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::{coeffs::CoefficientList, multilinear::MultilinearPoint};
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    #[test]
    fn test_sumcheck_single_initialization() {
        // Polynomial with 2 variables: f(X1, X2) = 1 + 2*X1 + 3*X2 + 4*X1*X2
        let c1 = BabyBear::from_u64(1);
        let c2 = BabyBear::from_u64(2);
        let c3 = BabyBear::from_u64(3);
        let c4 = BabyBear::from_u64(4);

        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4]);
        let points = vec![];
        let combination_randomness = vec![];
        let evaluations = vec![];

        let prover = SumcheckSingle::new(coeffs, &points, &combination_randomness, &evaluations);

        // Expected evaluation table after wavelet transform
        let expected_evaluation_of_p = vec![c1, c1 + c2, c1 + c3, c1 + c2 + c3 + c4];

        assert_eq!(prover.evaluation_of_p.evals(), &expected_evaluation_of_p);
        assert_eq!(prover.evaluation_of_equality.evals(), &vec![BabyBear::ZERO; 4]);
        assert_eq!(prover.sum, BabyBear::ZERO);
        assert_eq!(prover.num_variables, 2);
    }

    #[test]
    fn test_sumcheck_single_one_variable() {
        // Polynomial with 1 variable: f(X1) = 1 + 3*X1
        let c1 = BabyBear::from_u64(1);
        let c2 = BabyBear::from_u64(3);

        let coeffs = CoefficientList::new(vec![c1, c2]);
        let points = vec![];
        let combination_randomness = vec![];
        let evaluations = vec![];

        let prover = SumcheckSingle::new(coeffs, &points, &combination_randomness, &evaluations);

        let expected_evaluation_of_p = vec![c1, c1 + c2];

        assert_eq!(prover.evaluation_of_p.evals(), &expected_evaluation_of_p);
        assert_eq!(prover.evaluation_of_equality.evals(), &vec![BabyBear::ZERO; 2]);
        assert_eq!(prover.sum, BabyBear::ZERO);
        assert_eq!(prover.num_variables, 1);
    }

    #[test]
    fn test_sumcheck_single_three_variables() {
        // Polynomial with 3 variables
        let c1 = BabyBear::from_u64(1);
        let c2 = BabyBear::from_u64(2);
        let c3 = BabyBear::from_u64(3);
        let c4 = BabyBear::from_u64(4);
        let c5 = BabyBear::from_u64(5);
        let c6 = BabyBear::from_u64(6);
        let c7 = BabyBear::from_u64(7);
        let c8 = BabyBear::from_u64(8);

        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4, c5, c6, c7, c8]);
        let points = vec![];
        let combination_randomness = vec![];
        let evaluations = vec![];

        let prover = SumcheckSingle::new(coeffs, &points, &combination_randomness, &evaluations);

        let expected_evaluation_of_p = vec![
            c1,
            c1 + c2,
            c1 + c3,
            c1 + c2 + c3 + c4,
            c1 + c5,
            c1 + c2 + c5 + c6,
            c1 + c3 + c5 + c7,
            c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8,
        ];

        assert_eq!(prover.evaluation_of_p.evals(), &expected_evaluation_of_p);
        assert_eq!(prover.evaluation_of_equality.evals(), &vec![BabyBear::ZERO; 8]);
        assert_eq!(prover.sum, BabyBear::ZERO);
        assert_eq!(prover.num_variables, 3);
    }

    #[test]
    fn test_sumcheck_single_with_equality_constraints() {
        // Polynomial with 2 variables
        let c1 = BabyBear::from_u64(1);
        let c2 = BabyBear::from_u64(2);
        let c3 = BabyBear::from_u64(3);
        let c4 = BabyBear::from_u64(4);

        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4]);

        // Introduce an equality constraint at (X1, X2) = (1,0) with randomness 2
        let point = MultilinearPoint(vec![BabyBear::ONE, BabyBear::ZERO]);
        let combination_randomness = vec![BabyBear::from_u64(2)];
        let evaluations = vec![BabyBear::from_u64(5)];

        let prover = SumcheckSingle::new(coeffs, &[point], &combination_randomness, &evaluations);

        // Expected sum update: sum = 2 * 5
        assert_eq!(prover.sum, BabyBear::from_u64(2) * BabyBear::from_u64(5));

        // Expected evaluation table after wavelet transform
        let expected_evaluation_of_p = vec![c1, c1 + c2, c1 + c3, c1 + c2 + c3 + c4];
        assert_eq!(prover.evaluation_of_p.evals(), &expected_evaluation_of_p);
        assert_eq!(prover.num_variables, 2);
    }

    #[test]
    fn test_eval_eq_functionality() {
        let mut output = vec![BabyBear::ZERO; 4]; // n=2 → 2^2 = 4 elements
        let eval = vec![BabyBear::from_u64(1), BabyBear::from_u64(0)]; // (X1, X2) = (1,0)
        let scalar = BabyBear::from_u64(2);

        eval_eq(&eval, &mut output, scalar);

        // Expected results for (X1, X2) = (1,0)
        let expected_output =
            vec![BabyBear::ZERO, BabyBear::ZERO, BabyBear::from_u64(2), BabyBear::ZERO];

        assert_eq!(output, expected_output);
    }

    #[test]
    fn test_sumcheck_single_multiple_constraints() {
        // Polynomial with 3 variables
        let c1 = BabyBear::from_u64(1);
        let c2 = BabyBear::from_u64(2);
        let c3 = BabyBear::from_u64(3);
        let c4 = BabyBear::from_u64(4);
        let c5 = BabyBear::from_u64(5);
        let c6 = BabyBear::from_u64(6);
        let c7 = BabyBear::from_u64(7);
        let c8 = BabyBear::from_u64(8);

        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4, c5, c6, c7, c8]);

        // Constraints: (1,0,1) with randomness 2, (0,1,0) with randomness 3
        let point1 = MultilinearPoint(vec![BabyBear::ONE, BabyBear::ZERO, BabyBear::ONE]);
        let point2 = MultilinearPoint(vec![BabyBear::ZERO, BabyBear::ONE, BabyBear::ZERO]);
        let combination_randomness = vec![BabyBear::from_u64(2), BabyBear::from_u64(3)];
        let evaluations = vec![BabyBear::from_u64(5), BabyBear::from_u64(4)];

        let prover =
            SumcheckSingle::new(coeffs, &[point1, point2], &combination_randomness, &evaluations);

        // Expected sum update: sum = (2 * 5) + (3 * 4)
        assert_eq!(
            prover.sum,
            BabyBear::from_u64(2) * BabyBear::from_u64(5) +
                BabyBear::from_u64(3) * BabyBear::from_u64(4)
        );
    }

    #[test]
    fn test_compute_sumcheck_polynomial_basic() {
        // Polynomial with 2 variables: f(X1, X2) = c1 + c2*X1 + c3*X2 + c4*X1*X2
        let c1 = BabyBear::from_u64(1);
        let c2 = BabyBear::from_u64(2);
        let c3 = BabyBear::from_u64(3);
        let c4 = BabyBear::from_u64(4);

        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4]);
        let points = vec![];
        let combination_randomness = vec![];
        let evaluations = vec![];

        let prover = SumcheckSingle::new(coeffs, &points, &combination_randomness, &evaluations);
        let sumcheck_poly = prover.compute_sumcheck_polynomial();

        // Since no equality constraints, sumcheck_poly should be **zero**
        let expected_evaluations = vec![BabyBear::ZERO; 3];
        assert_eq!(sumcheck_poly.evaluations(), &expected_evaluations);
    }

    #[test]
    fn test_compute_sumcheck_polynomial_with_equality_constraints() {
        // Define a polynomial: f(X1, X2) = c1 + c2*X1 + c3*X2 + c4*X1*X2
        // This represents a multilinear polynomial in two variables.
        let c1 = BabyBear::from_u64(1);
        let c2 = BabyBear::from_u64(2);
        let c3 = BabyBear::from_u64(3);
        let c4 = BabyBear::from_u64(4);

        // Convert the polynomial into coefficient form
        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4]);

        // Introduce an equality constraint at point (X1, X2) = (1,0) with randomness 2
        let point = MultilinearPoint(vec![BabyBear::ONE, BabyBear::ZERO]);
        let combination_randomness = vec![BabyBear::from_u64(2)];
        let evaluations = vec![BabyBear::from_u64(5)];

        // Instantiate the Sumcheck prover with the polynomial and equality constraints
        let prover = SumcheckSingle::new(coeffs, &[point], &combination_randomness, &evaluations);
        let sumcheck_poly = prover.compute_sumcheck_polynomial();

        // Expected sum update: sum = 2 * 5
        assert_eq!(prover.sum, BabyBear::from_u64(2) * BabyBear::from_u64(5));

        // Compute the evaluations of the polynomial at the four possible binary inputs
        let ep_00 = c1; // f(0,0) = c1
        let ep_01 = c1 + c2; // f(0,1) = c1 + c2
        let ep_10 = c1 + c3; // f(1,0) = c1 + c3
        let ep_11 = c1 + c3 + c2 + c4; // f(1,1) = c1 + c3 + c2 + c4

        // Compute the evaluations of the equality constraint polynomial at each binary point
        // Given that the constraint is at (1,0) with randomness 2, the equality function is:
        // \begin{equation}
        // eq(X1, X2) = 2 * (X1 - 1) * (X2 - 0)
        // \end{equation}
        let f_00 = BabyBear::ZERO; // eq(0,0) = 0
        let f_01 = BabyBear::ZERO; // eq(0,1) = 0
        let f_10 = BabyBear::TWO; // eq(1,0) = 2
        let f_11 = BabyBear::ZERO; // eq(1,1) = 0

        // Compute the coefficients of the quadratic polynomial S(X)
        let e0 = ep_00 * f_00 + ep_10 * f_10; // Contribution at X = 0
        let e2 = (ep_01 - ep_00) * (f_01 - f_00) + (ep_11 - ep_10) * (f_11 - f_10); // Quadratic coefficient
        let e1 = prover.sum - e0.double() - e2; // Middle coefficient using sum rule

        // Compute sumcheck polynomial evaluations at {0,1,2}
        let eval_0 = e0;
        let eval_1 = e0 + e1 + e2;
        let eval_2 = eval_1 + e1 + e2 + e2.double();
        let expected_evaluations = vec![eval_0, eval_1, eval_2];
        assert_eq!(sumcheck_poly.evaluations(), &expected_evaluations);
    }

    #[test]
    fn test_compute_sumcheck_polynomial_with_equality_constraints_3vars() {
        // Define a polynomial:
        // f(X1, X2, X3) = c1 + c2*X1 + c3*X2 + c4*X3 + c5*X1*X2 + c6*X1*X3 + c7*X2*X3 + c8*X1*X2*X3
        let c1 = BabyBear::from_u64(1);
        let c2 = BabyBear::from_u64(2);
        let c3 = BabyBear::from_u64(3);
        let c4 = BabyBear::from_u64(4);
        let c5 = BabyBear::from_u64(5);
        let c6 = BabyBear::from_u64(6);
        let c7 = BabyBear::from_u64(7);
        let c8 = BabyBear::from_u64(8);

        // Convert the polynomial into coefficient form
        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4, c5, c6, c7, c8]);

        // Introduce an equality constraint at (X1, X2, X3) = (1,0,1) with randomness 2
        let point = MultilinearPoint(vec![BabyBear::ONE, BabyBear::ZERO, BabyBear::ONE]);
        let combination_randomness = vec![BabyBear::from_u64(2)];
        let evaluations = vec![BabyBear::from_u64(5)];

        // Instantiate the Sumcheck prover
        let prover = SumcheckSingle::new(coeffs, &[point], &combination_randomness, &evaluations);
        let sumcheck_poly = prover.compute_sumcheck_polynomial();

        // Expected sum update: sum = 2 * 5 = 10
        assert_eq!(prover.sum, BabyBear::from_u64(10));

        // Compute polynomial evaluations at the 8 possible binary inputs
        let ep_000 = c1; // f(0,0,0)
        let ep_001 = c1 + c2; // f(0,0,1)
        let ep_010 = c1 + c3; // f(0,1,0)
        let ep_011 = c1 + c2 + c3 + c4; // f(0,1,1)
        let ep_100 = c1 + c5; // f(1,0,0)
        let ep_101 = c1 + c2 + c5 + c6; // f(1,0,1)
        let ep_110 = c1 + c3 + c5 + c7; // f(1,1,0)
        let ep_111 = c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8; // f(1,1,1)

        // Compute the equality constraint evaluations for each binary point
        // Given the constraint at (1,0,1) with randomness 2:
        // \begin{equation}
        // eq(X1, X2, X3) = 2 * (X1 - 1) * (X2 - 0) * (X3 - 1)
        // \end{equation}
        let f_000 = BabyBear::ZERO; // eq(0,0,0) = 0
        let f_001 = BabyBear::ZERO; // eq(0,0,1) = 0
        let f_010 = BabyBear::ZERO; // eq(0,1,0) = 0
        let f_011 = BabyBear::ZERO; // eq(0,1,1) = 0
        let f_100 = BabyBear::ZERO; // eq(1,0,0) = 0
        let f_101 = BabyBear::TWO; // eq(1,0,1) = 2
        let f_110 = BabyBear::ZERO; // eq(1,1,0) = 0
        let f_111 = BabyBear::ZERO; // eq(1,1,1) = 0

        // Compute the coefficients of the quadratic sumcheck polynomial S(X)
        // Contribution at X = 0 (constant term)
        let e0 = ep_000 * f_000 + ep_010 * f_010 + ep_100 * f_100 + ep_110 * f_110;

        // Quadratic coefficient
        let e2 = (ep_001 - ep_000) * (f_001 - f_000) +
            (ep_011 - ep_010) * (f_011 - f_010) +
            (ep_101 - ep_100) * (f_101 - f_100) +
            (ep_111 - ep_110) * (f_111 - f_110);

        // Middle coefficient using sum rule: sum = 2 * e0 + e1 + e2
        let e1 = prover.sum - e0.double() - e2;

        // Compute sumcheck polynomial evaluations at {0,1,2}
        let eval_0 = e0;
        let eval_1 = e0 + e1 + e2;
        let eval_2 = eval_1 + e1 + e2 + e2.double();

        // Hardcoded expected values
        let expected_evaluations = vec![eval_0, eval_1, eval_2];

        // Assert that computed sumcheck polynomial matches expectations
        assert_eq!(sumcheck_poly.evaluations(), &expected_evaluations);
    }
}
