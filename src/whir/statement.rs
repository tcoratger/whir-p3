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
#[derive(Clone, Debug)]
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
    pub fn weighted_sum<BF>(&self, poly: &EvaluationsList<BF>) -> F
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
            Self::Evaluation { point } => poly.evaluate_at_extension(point),
        }
    }

    /// Accumulates the contribution of the weight function into `accumulator`, scaled by `factor`.
    ///
    /// - In evaluation mode, updates `accumulator` using an equality constraint.
    /// - In linear mode, scales the weight function by `factor` and accumulates it.
    ///
    /// Given a weight function `w(X)` and a factor `λ`, this updates `accumulator` as:
    ///
    /// \begin{equation}
    /// a(X) \gets a(X) + \lambda \cdot w(X)
    /// \end{equation}
    ///
    /// where `a(X)` is the accumulator polynomial.
    ///
    /// **Precondition:**
    /// `accumulator.num_variables()` must match `self.num_variables()`.
    #[instrument(skip_all)]
    pub fn accumulate(&self, accumulator: &mut EvaluationsList<F>, factor: F) {
        assert_eq!(accumulator.num_variables(), self.num_variables());
        match self {
            Self::Evaluation { point } => {
                eval_eq(&point.0, accumulator.evals_mut(), factor);
            }
            Self::Linear { weight } => {
                #[cfg(feature = "parallel")]
                accumulator
                    .evals_mut()
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(corner, acc)| {
                        *acc += factor * weight[corner];
                    });

                #[cfg(not(feature = "parallel"))]
                accumulator
                    .evals_mut()
                    .iter_mut()
                    .enumerate()
                    .for_each(|(corner, acc)| {
                        *acc += factor * weight[corner];
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
            Self::Linear { weight } => weight.evaluate_at_extension(folding_randomness),
        }
    }
}

/// Represents a single constraint in a polynomial statement.
#[derive(Clone, Debug)]
pub struct Constraint<F> {
    /// The weight function applied to the polynomial.
    ///
    /// This defines how the polynomial is combined or evaluated.
    /// It can represent either a point evaluation or a full set of weights.
    pub weights: Weights<F>,

    /// The expected result of applying the weight to the polynomial.
    ///
    /// This is the scalar value that the weighted sum must match.
    pub sum: F,

    /// If true, the verifier will not evaluate the weight directly.
    ///
    /// Instead, the prover must supply the result as a hint.
    /// This is used for deferred or externally computed evaluations.
    pub defer_evaluation: bool,
}

/// Represents a system of weighted polynomial constraints over a Boolean hypercube.
///
/// A `Statement<F>` consists of multiple constraints, each enforcing a relationship of the form:
///
/// \begin{equation}
/// \sum_{x \in \{0,1\}^n} w_i(x) \cdot p(x) = s_i
/// \end{equation}
///
/// where:
/// - `w_i(x)` is a weight function, either a point evaluation (equality constraint) or a full set of weights.
/// - `p(x)` is a multilinear polynomial over $\{0,1\}^n$ in evaluation form.
/// - `s_i` is the expected sum for the $i$-th constraint.
///
/// These constraints can be combined into a single constraint using a random challenge $\gamma$:
///
/// \begin{equation}
/// W(x) = w_1(x) + \gamma w_2(x) + \gamma^2 w_3(x) + \dots + \gamma^{k-1} w_k(x)
/// \end{equation}
///
/// with a combined expected sum:
///
/// \begin{equation}
/// S = s_1 + \gamma s_2 + \gamma^2 s_3 + \dots + \gamma^{k-1} s_k
/// \end{equation}
///
/// This combined form is used in protocols like sumcheck and zerocheck to reduce many constraints to one.
#[derive(Clone, Debug)]
pub struct Statement<F> {
    /// Number of variables in the multilinear polynomial space (log₂ of evaluation length).
    num_variables: usize,

    /// List of constraints, each pairing a weight function with a target expected sum.
    ///
    /// The weight may be either a concrete evaluation point (enforcing `p(z) = s`)
    /// or a full evaluation vector of weights `w(x)` (enforcing a weighted sum).
    pub constraints: Vec<Constraint<F>>,
}

impl<F: Field> Statement<F> {
    /// Creates an empty `Statement<F>` for polynomials with `num_variables` variables.
    #[must_use]
    pub const fn new(num_variables: usize) -> Self {
        Self {
            num_variables,
            constraints: Vec::new(),
        }
    }

    /// Returns the number of variables defining the polynomial space.
    #[must_use]
    pub const fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// Adds a constraint `(w(X), s)` to the system.
    ///
    /// **Precondition:**
    /// The number of variables in `w(X)` must match `self.num_variables`.
    pub fn add_constraint(&mut self, weights: Weights<F>, sum: F) {
        assert_eq!(weights.num_variables(), self.num_variables());
        self.constraints.push(Constraint {
            weights,
            sum,
            defer_evaluation: false,
        });
    }

    /// Inserts a constraint `(w(X), s)` at the front of the system.
    pub fn add_constraint_in_front(&mut self, weights: Weights<F>, sum: F) {
        assert_eq!(weights.num_variables(), self.num_variables());
        self.constraints.insert(
            0,
            Constraint {
                weights,
                sum,
                defer_evaluation: false,
            },
        );
    }

    /// Inserts multiple constraints at the front of the system.
    ///
    /// Panics if any constraint's number of variables does not match the system.
    pub fn add_constraints_in_front(&mut self, constraints: Vec<(Weights<F>, F)>) {
        // Store the number of variables expected by this statement.
        let n = self.num_variables();

        // Preallocate a vector for the converted constraints to avoid reallocations.
        let mut new_constraints = Vec::with_capacity(constraints.len());

        // Iterate through each (weights, sum) pair in the input.
        for (weights, sum) in constraints {
            // Ensure the number of variables in the weight matches the statement.
            assert_eq!(weights.num_variables(), n);

            // Convert the pair into a full `Constraint` with `defer_evaluation = false`.
            new_constraints.push(Constraint {
                weights,
                sum,
                defer_evaluation: false,
            });
        }

        // Insert all new constraints at the beginning of the existing list.
        self.constraints.splice(0..0, new_constraints);
    }

    /// Combines all constraints into a single aggregated polynomial using a challenge.
    ///
    /// Given a random challenge $\gamma$, the new polynomial is:
    ///
    /// \begin{equation}
    /// W(X) = w_1(X) + \gamma w_2(X) + \gamma^2 w_3(X) + \dots + \gamma^{k-1} w_k(X)
    /// \end{equation}
    ///
    /// with the combined sum:
    ///
    /// \begin{equation}
    /// S = s_1 + \gamma s_2 + \gamma^2 s_3 + \dots + \gamma^{k-1} s_k
    /// \end{equation}
    ///
    /// **Returns:**
    /// - `EvaluationsList<F>`: The combined polynomial `W(X)`.
    /// - `F`: The combined sum `S`.
    #[instrument(skip_all)]
    pub fn combine(&self, challenge: F) -> (EvaluationsList<F>, F) {
        let evaluations_vec = vec![F::ZERO; 1 << self.num_variables];
        let mut combined_evals = EvaluationsList::new(evaluations_vec);
        let (combined_sum, _) = self.constraints.iter().fold(
            (F::ZERO, F::ONE),
            |(mut acc_sum, gamma_pow), constraint| {
                constraint
                    .weights
                    .accumulate(&mut combined_evals, gamma_pow);
                acc_sum += constraint.sum * gamma_pow;
                (acc_sum, gamma_pow * challenge)
            },
        );

        (combined_evals, combined_sum)
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};
    use proptest::prelude::*;

    use super::*;
    use crate::poly::coeffs::CoefficientList;

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

        assert_eq!(weight.weighted_sum(&evals), expected);
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

        assert_eq!(weight.weighted_sum(&evals), expected);
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
        weight.accumulate(&mut accumulator, factor);

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
        weight.accumulate(&mut accumulator, factor);

        // Compute expected result manually
        let mut expected = vec![F::ZERO, F::ZERO];
        eval_eq(&point.0, &mut expected, factor);

        assert_eq!(accumulator.evals(), &expected);
    }

    #[test]
    fn test_statement_combine() {
        // Create a new statement with 1 variable
        let mut statement = Statement::new(1);

        // Define weights
        let w0 = F::from_u64(3);
        let w1 = F::from_u64(5);
        let weight_list = EvaluationsList::new(vec![w0, w1]);
        let weight = Weights::linear(weight_list);

        // Define sum constraint
        let sum = F::from_u64(7);
        statement.add_constraint(weight, sum);

        // Define a challenge factor
        let challenge = F::from_u64(2);

        // Compute combined evaluations and sum
        let (combined_evals, combined_sum) = statement.combine(challenge);

        // Expected evaluations should match the accumulated weights
        let expected_combined_evals = vec![
            w0, // 3
            w1, // 5
        ];

        // Expected sum remains unchanged since there is only one constraint
        let expected_combined_sum = sum;

        assert_eq!(combined_evals.evals(), &expected_combined_evals);
        assert_eq!(combined_sum, expected_combined_sum);
    }

    #[test]
    fn test_statement_with_multiple_constraints() {
        // Create a new statement with 2 variables
        let mut statement = Statement::new(2);

        // Define weights for first constraint (2 variables => 4 evaluations)
        let w0 = F::from_u64(1);
        let w1 = F::from_u64(2);
        let w2 = F::from_u64(3);
        let w3 = F::from_u64(4);
        let weight_list1 = EvaluationsList::new(vec![w0, w1, w2, w3]);
        let weight1 = Weights::linear(weight_list1);

        // Define weights for second constraint (also 2 variables => 4 evaluations)
        let w4 = F::from_u64(5);
        let w5 = F::from_u64(6);
        let w6 = F::from_u64(7);
        let w7 = F::from_u64(8);
        let weight_list2 = EvaluationsList::new(vec![w4, w5, w6, w7]);
        let weight2 = Weights::linear(weight_list2);

        // Define sum constraints
        let sum1 = F::from_u64(5);
        let sum2 = F::from_u64(7);

        // Ensure both weight lists match the expected number of variables
        assert_eq!(weight1.num_variables(), 2);
        assert_eq!(weight2.num_variables(), 2);

        // Add constraints to the statement
        statement.add_constraint(weight1, sum1);
        statement.add_constraint(weight2, sum2);

        // Define a challenge factor
        let challenge = F::from_u64(2);

        // Compute combined evaluations and sum
        let (combined_evals, combined_sum) = statement.combine(challenge);

        // Expected evaluations:
        //
        // \begin{equation}
        // combined = weight_1 + challenge \cdot weight_2
        // \end{equation}
        let expected_combined_evals = vec![
            w0 + challenge * w4, // 1 + 2 * 5 = 11
            w1 + challenge * w5, // 2 + 2 * 6 = 14
            w2 + challenge * w6, // 3 + 2 * 7 = 17
            w3 + challenge * w7, // 4 + 2 * 8 = 20
        ];

        // Expected sum:
        //
        // \begin{equation}
        // S_{combined} = S_1 + challenge \cdot S_2
        // \end{equation}
        let expected_combined_sum = sum1 + challenge * sum2; // 5 + 2 * 7 = 19

        assert_eq!(combined_evals.evals(), &expected_combined_evals);
        assert_eq!(combined_sum, expected_combined_sum);
    }

    #[test]
    fn test_compute_evaluation_weight() {
        // Define an evaluation weight at a specific point
        let point = MultilinearPoint(vec![F::from_u64(3)]);
        let weight = Weights::evaluation(point.clone());

        // Define a randomness point for folding
        let folding_randomness = MultilinearPoint(vec![F::from_u64(2)]);

        // Expected result is the evaluation of eq_poly_outside at the given randomness
        let expected = point.eq_poly_outside(&folding_randomness);

        assert_eq!(weight.compute(&folding_randomness), expected);
    }

    #[test]
    fn test_compute_linear_weight_with_term() {
        // Define a multilinear polynomial in 3 variables:
        // f(X0, X1, X2) = 1 + 2*X2 + 3*X1 + 4*X1*X2
        //              + 5*X0 + 6*X0*X2 + 7*X0*X1 + 8*X0*X1*X2
        let c1 = F::from_u64(1);
        let c2 = F::from_u64(2);
        let c3 = F::from_u64(3);
        let c4 = F::from_u64(4);
        let c5 = F::from_u64(5);
        let c6 = F::from_u64(6);
        let c7 = F::from_u64(7);
        let c8 = F::from_u64(8);

        // Convert the polynomial into coefficient form
        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4, c5, c6, c7, c8]);

        let f = |x0: F, x1: F, x2: F| {
            c1 + c2 * x2
                + c3 * x1
                + c4 * x1 * x2
                + c5 * x0
                + c6 * x0 * x2
                + c7 * x0 * x1
                + c8 * x0 * x1 * x2
        };

        // List of evaluations
        let evaluations = coeffs.to_evaluations();

        // Define a linear weight with a precomputed term
        let weight = Weights::linear(evaluations);

        // Folding randomness should have no effect in linear mode
        let folding_randomness =
            MultilinearPoint(vec![F::from_u64(3), F::from_u64(46), F::from_u64(56)]);

        // Expected result is the proper evaluation of the polynomial
        assert_eq!(
            weight.compute(&folding_randomness),
            f(F::from_u64(3), F::from_u64(46), F::from_u64(56))
        );
    }

    proptest! {
        #[test]
        fn prop_weighted_sum_equivalence_eval_vs_eval_ef4(
            values in prop::collection::vec(0u64..100, 8),
            x0 in 0u64..100,
            x1 in 0u64..100,
            x2 in 0u64..100
        ) {
            // F-based polynomial
            let coeffs: Vec<F> = values.iter().copied().map(F::from_u64).collect();
            let poly = EvaluationsList::new(coeffs);

            // EF4-based polynomial
            let coeffs_ef: Vec<EF4> = values.iter().copied().map(EF4::from_u64).collect();
            let poly_ef = EvaluationsList::new(coeffs_ef);

            let point_ef4 = MultilinearPoint(vec![
                EF4::from_u64(x0),
                EF4::from_u64(x1),
                EF4::from_u64(x2),
            ]);

            // EF4-based weight
            let weight_ef = Weights::<EF4>::evaluation(point_ef4);

            // Comparison between F-based and EF4-based weights
            let result_f = weight_ef.weighted_sum(&poly);
            let result_ef = weight_ef.weighted_sum(&poly_ef);

            prop_assert_eq!(result_f, result_ef);
        }
    }
}
