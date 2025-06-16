use p3_field::{ExtensionField, Field};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use tracing::instrument;

use crate::{
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    utils::eval_eq,
};

/// Represents a weight function used in polynomial evaluations.
///
/// A `Weights<F>` instance allows evaluating or accumulating weighted contributions
/// to a multilinear polynomial stored in evaluation form. It supports two modes:
///
/// - Evaluation mode: Represents an equality constraint at a specific `MultilinearPoint<F>`.
/// - Linear mode: Represents a set of per-corner weights stored as `EvaluationsList<F>`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Weights<F> {
    /// Represents a weight function that enforces equality constraints at a specific point.
    Evaluation { point: MultilinearPoint<F> },
    /// Represents a weight function defined as a precomputed set of evaluations.
    Linear { weight: EvaluationsList<F> },
}

impl<F: Field> Weights<F> {
    /// Constructs a weight in evaluation mode, enforcing an equality constraint at `point`.
    ///
    /// Given a multilinear polynomial `p(X)`, this weight evaluates:
    ///
    /// \begin{equation}
    /// w(X) = eq_{z}(X)
    /// \end{equation}
    ///
    /// where `eq_z(X)` is the Lagrange interpolation polynomial enforcing `X = z`.
    #[must_use]
    pub const fn evaluation(point: MultilinearPoint<F>) -> Self {
        Self::Evaluation { point }
    }

    /// Constructs a weight in linear mode, applying a set of precomputed weights.
    ///
    /// This mode allows applying a function `w(X)` stored in `EvaluationsList<F>`:
    ///
    /// \begin{equation}
    /// w(X) = \sum_{i} w_i \cdot X_i
    /// \end{equation}
    ///
    /// where `w_i` are the predefined weight values for each corner of the hypercube.
    #[must_use]
    pub const fn linear(weight: EvaluationsList<F>) -> Self {
        Self::Linear { weight }
    }

    /// Returns the number of variables involved in the weight function.
    #[must_use]
    pub fn num_variables(&self) -> usize {
        match self {
            Self::Evaluation { point } => point.num_variables(),
            Self::Linear { weight } => weight.num_variables(),
        }
    }

    /// Construct weights for a univariate evaluation
    pub fn univariate(point: F, size: usize) -> Self {
        Self::Evaluation {
            point: MultilinearPoint::expand_from_univariate(point, size),
        }
    }

    /// Computes the weighted sum of a polynomial `p(X)` under the current weight function.
    ///
    /// - In linear mode, computes the inner product between the polynomial values and weights:
    ///
    /// \begin{equation}
    /// \sum_{i} p_i \cdot w_i
    /// \end{equation}
    ///
    /// - In evaluation mode, evaluates `p(X)` at the equality constraint point.
    ///
    /// **Precondition:**
    /// If `self` is in linear mode, `poly.num_variables()` must match `weight.num_variables()`.
    #[must_use]
    pub fn evaluate_evals<BF>(&self, poly: &EvaluationsList<BF>) -> F
    where
        BF: Field,
        F: ExtensionField<BF>,
    {
        match self {
            Self::Linear { weight } => {
                assert_eq!(poly.num_variables(), weight.num_variables());
                #[cfg(not(feature = "parallel"))]
                {
                    poly.evals()
                        .iter()
                        .zip(weight.evals().iter())
                        .map(|(p, w)| *w * *p)
                        .sum()
                }
                #[cfg(feature = "parallel")]
                {
                    poly.evals()
                        .par_iter()
                        .zip(weight.evals().par_iter())
                        .map(|(p, w)| *w * *p)
                        .sum()
                }
            }
            Self::Evaluation { point } => poly.evaluate(point),
        }
    }

    /// Accumulates the contribution of the weight function into `accumulator`, scaled by `factor`.
    ///
    /// - In evaluation mode, updates `accumulator` using an equality constraint.
    /// - In linear mode, scales the weight function by `factor` and accumulates it.
    ///
    /// Given a weight function `w(X)` and a factor `λ`, this updates `accumulator` as:
    ///
    /// ```math
    /// a(X) <- a(X) + \lambda \cdot w(X)
    /// ```
    ///
    /// where `a(X)` is the accumulator polynomial.
    ///
    /// **Precondition:**
    /// `accumulator.num_variables()` must match `self.num_variables()`.
    ///
    /// **Warning:**
    /// If INITIALIZED is `false`, the accumulator must be overwritten with the new values.
    #[instrument(skip_all)]
    pub fn accumulate<Base, const INITIALIZED: bool>(
        &self,
        accumulator: &mut EvaluationsList<F>,
        factor: F,
    ) where
        Base: Field,
        F: ExtensionField<Base>,
    {
        assert_eq!(accumulator.num_variables(), self.num_variables());
        match self {
            Self::Evaluation { point } => {
                eval_eq::<Base, F, INITIALIZED>(&point.0, accumulator.evals_mut(), factor);
            }
            Self::Linear { weight } => {
                #[cfg(feature = "parallel")]
                let accumulator_iter = accumulator.evals_mut().par_iter_mut();
                #[cfg(not(feature = "parallel"))]
                let accumulator_iter = accumulator.evals_mut().iter_mut();

                accumulator_iter.enumerate().for_each(|(corner, acc)| {
                    if INITIALIZED {
                        *acc += factor * weight[corner];
                    } else {
                        *acc = factor * weight[corner];
                    }
                });
            }
        }
    }

    /// Evaluates the weight function at a given folding point.
    ///
    /// - In evaluation mode, computes the equality polynomial at the provided point:
    ///
    /// \begin{equation}
    /// w(X) = eq_{\text{point}}(X)
    /// \end{equation}
    ///
    /// This enforces that the polynomial is evaluated exactly at a specific input.
    ///
    /// - In linear mode, interprets the stored weight function as a multilinear polynomial
    ///   and evaluates it at the provided point using multilinear extension.
    ///
    /// **Precondition:**
    /// The input point must have the same number of variables as the weight function.
    ///
    /// **Returns:**
    /// A field element representing the weight function applied to the given point.
    #[must_use]
    pub fn compute(&self, folding_randomness: &MultilinearPoint<F>) -> F {
        match self {
            Self::Evaluation { point } => point.eq_poly_outside(folding_randomness),
            Self::Linear { weight } => weight.evaluate(folding_randomness),
        }
    }

    /// Evaluate the weighted sum with a polynomial in coefficient form.
    #[must_use]
    pub fn evaluate_coeffs(&self, poly: &EvaluationsList<F>) -> F {
        assert_eq!(self.num_variables(), poly.num_variables());
        match self {
            Self::Evaluation { point } => poly.evaluate(point),

            // We intentionally avoid parallel iterators here because this function is only called by the verifier,
            // which is assumed to run on a lightweight device.
            Self::Linear { weight } => weight
                .evals()
                .iter()
                .zip(poly.evals())
                .map(|(&w, &p)| w * p)
                .sum(),
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};

    use super::*;

    type F = BabyBear;
    type EF4 = BinomialExtensionField<F, 4>;

    #[test]
    fn test_weights_evaluation() {
        // Define a point in the multilinear space
        let point = MultilinearPoint(vec![F::ONE, F::ZERO]);
        let weight = Weights::evaluation(point);

        // The number of variables in the weight should match the number of variables in the point
        assert_eq!(weight.num_variables(), 2);
    }

    #[test]
    fn test_weights_linear() {
        // Define a list of evaluation values
        let evals = EvaluationsList::new(vec![F::ONE, F::TWO, F::from_u64(3), F::from_u64(3)]);
        let weight = Weights::linear(evals);

        // The number of variables in the weight should match the number of variables in evals
        assert_eq!(weight.num_variables(), 2);
    }

    #[test]
    fn test_weighted_sum_evaluation() {
        // Define polynomial evaluations at different points
        let e0 = F::from_u64(3);
        let e1 = F::from_u64(5);
        let evals = EvaluationsList::new(vec![e0, e1]);

        // Define an evaluation weight at a specific point
        let point = MultilinearPoint(vec![F::ONE]);
        let weight = Weights::evaluation(point);

        // Expected result: polynomial evaluation at the given point
        let expected = e1;

        assert_eq!(weight.evaluate_evals(&evals), expected);
    }

    #[test]
    fn test_weighted_sum_linear() {
        // Define polynomial evaluations
        let e0 = F::ONE;
        let e1 = F::TWO;
        let evals = EvaluationsList::new(vec![e0, e1]);

        // Define linear weights
        let w0 = F::TWO;
        let w1 = F::from_u64(3);
        let weight_list = EvaluationsList::new(vec![w0, w1]);
        let weight = Weights::linear(weight_list);

        // Compute expected result manually
        //
        // \begin{equation}
        // \sum_{i} e_i \cdot w_i = e_0 \cdot w_0 + e_1 \cdot w_1
        // \end{equation}
        let expected = e0 * w0 + e1 * w1;

        assert_eq!(weight.evaluate_evals(&evals), expected);
    }

    #[test]
    fn test_accumulate_linear() {
        // Initialize an empty accumulator
        let mut accumulator = EvaluationsList::new(vec![F::ZERO, F::ZERO]);

        // Define weights
        let w0 = F::from_u64(2);
        let w1 = F::from_u64(3);
        let weight_list = EvaluationsList::new(vec![w0, w1]);
        let weight = Weights::linear(weight_list);

        // Define a multiplication factor
        let factor = F::from_u64(4);

        // Accumulate weighted values
        weight.accumulate::<_, true>(&mut accumulator, factor);

        // Expected result:
        //
        // \begin{equation}
        // acc_i = factor \cdot w_i
        // \end{equation}
        let expected = vec![
            w0 * factor, // 2 * 4 = 8
            w1 * factor, // 3 * 4 = 12
        ];

        assert_eq!(accumulator.evals(), &expected);
    }

    #[test]
    fn test_accumulate_evaluation() {
        // Initialize an empty accumulator
        let mut accumulator = EvaluationsList::new(vec![F::ZERO, F::ZERO]);

        // Define an evaluation point
        let point = MultilinearPoint(vec![F::ONE]);
        let weight = Weights::evaluation(point.clone());

        // Define a multiplication factor
        let factor = F::from_u64(5);

        // Accumulate weighted values
        weight.accumulate::<_, true>(&mut accumulator, factor);

        // Compute expected result manually
        let mut expected = vec![F::ZERO, F::ZERO];
        eval_eq::<_, _, true>(&point.0, &mut expected, factor);

        assert_eq!(accumulator.evals(), &expected);
    }

    #[test]
    fn test_univariate_weights_one_variable() {
        // y = 3, n = 1 → [3]
        let y = F::from_u64(3);
        let weight = Weights::univariate(y, 1);

        // Expect point to be [3]
        let expected = MultilinearPoint(vec![y]);
        assert_eq!(weight, Weights::evaluation(expected));
    }

    #[test]
    fn test_univariate_weights_two_variables() {
        // y = 4, n = 2 → [y^2, y] = [16, 4]
        let y = F::from_u64(4);
        let weight = Weights::univariate(y, 2);

        let expected = MultilinearPoint(vec![y.square(), y]);
        assert_eq!(weight, Weights::evaluation(expected));
    }

    #[test]
    fn test_univariate_weights_four_variables() {
        // y = 3, n = 4 → [3^8, 3^4, 3^2, 3]
        let y = F::from_u64(3);
        let weight = Weights::univariate(y, 4);

        let expected = MultilinearPoint(vec![y.exp_u64(8), y.exp_u64(4), y.square(), y]);

        assert_eq!(weight, Weights::evaluation(expected));
    }

    #[test]
    fn test_univariate_weights_zero_variables() {
        let y = F::from_u64(10);
        let weight = Weights::univariate(y, 0);

        // Expect empty point
        let expected = MultilinearPoint(vec![]);
        assert_eq!(weight, Weights::evaluation(expected));
    }
}
