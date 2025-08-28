use p3_field::{ExtensionField, Field};
use tracing::instrument;
use weights::Weights;

use crate::{
    poly::evals::EvaluationsList, utils::uninitialized_vec, whir::statement::constraint::Constraint,
};

pub mod constraint;
pub mod evaluator;
pub mod weights;

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
    pub fn combine<Base>(&self, challenge: F) -> (EvaluationsList<F>, F)
    where
        Base: Field,
        F: ExtensionField<Base>,
    {
        // If there are no constraints, the combination is:
        // - The combined polynomial W(X) is identically zero (all evaluations = 0).
        // - The combined expected sum S is zero.
        if self.constraints.is_empty() {
            return (
                EvaluationsList::new(F::zero_vec(1 << self.num_variables)),
                F::ZERO,
            );
        }

        // Compute the number of evaluation points: 2^(num_variables).
        let len = 1usize << self.num_variables;

        // Allocate the vector for combined evaluations without initializing it to zero.
        // Safety: we guarantee that the very first accumulate() call will overwrite
        // the entire buffer before any read occurs.
        let mut combined = EvaluationsList::new(unsafe { uninitialized_vec::<F>(len) });

        // Separate the first constraint from the rest.
        // This allows us to treat the first one specially:
        //   - We overwrite the buffer.
        //   - We avoid a runtime branch in the main loop.
        let (first, rest) = self.constraints.split_first().unwrap();

        // Start with γ^0 = 1 for the first constraint.
        let mut gamma = F::ONE;

        // Apply the first constraint's weights directly into the buffer,
        // overwriting any uninitialized values.
        first
            .weights
            .accumulate::<Base, false>(&mut combined, gamma);

        // Initialize the combined expected sum with the first term: s_1 * γ^0.
        let mut sum = first.sum * gamma;

        // Process the remaining constraints.
        for c in rest {
            // Update γ to γ^i for this constraint.
            gamma *= challenge;

            // Add this constraint's weighted polynomial evaluations into the buffer
            c.weights.accumulate::<Base, true>(&mut combined, gamma);

            // Add this constraint's contribution to the combined expected sum:
            sum += c.sum * gamma;
        }

        // Return:
        // - The combined polynomial W(X) in evaluation form.
        // - The combined expected sum S.
        (combined, sum)
    }

    #[must_use]
    pub fn num_deref_constraints(&self) -> usize {
        self.constraints
            .iter()
            .filter(|constraint| constraint.defer_evaluation)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};
    use proptest::prelude::*;

    use super::*;
    use crate::{poly::coeffs::CoefficientList, whir::MultilinearPoint};

    type F = BabyBear;
    type EF4 = BinomialExtensionField<F, 4>;

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

        assert_eq!(&*combined_evals, &expected_combined_evals);
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

        assert_eq!(&*combined_evals, &expected_combined_evals);
        assert_eq!(combined_sum, expected_combined_sum);
    }

    #[test]
    fn test_compute_evaluation_weight() {
        // Define an evaluation weight at a specific point
        let point = MultilinearPoint::new(vec![F::from_u64(3)]);
        let weight = Weights::evaluation(point.clone());

        // Define a randomness point for folding
        let folding_randomness = MultilinearPoint::new(vec![F::from_u64(2)]);

        // Expected result is the evaluation of eq_poly at the given randomness
        let expected = point.eq_poly(&folding_randomness);

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
            MultilinearPoint::new(vec![F::from_u64(3), F::from_u64(46), F::from_u64(56)]);

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

            let point_ef4 = MultilinearPoint::new(vec![
                EF4::from_u64(x0),
                EF4::from_u64(x1),
                EF4::from_u64(x2),
            ]);

            // EF4-based weight
            let weight_ef = Weights::<EF4>::evaluation(point_ef4);

            // Comparison between F-based and EF4-based weights
            let result_f = weight_ef.evaluate_evals(&poly);
            let result_ef = weight_ef.evaluate_evals(&poly_ef);

            prop_assert_eq!(result_f, result_ef);
        }
    }

    #[test]
    fn test_evaluate_in_evaluation_mode() {
        // Define a multilinear polynomial:
        // f(X0, X1) = 3 + 4*X1 + 5*X0 + 6*X0*X1
        let c0 = F::from_u64(3);
        let c1 = F::from_u64(4);
        let c2 = F::from_u64(5);
        let c3 = F::from_u64(6);

        let f = |x0: F, x1: F| c0 + c1 * x1 + c2 * x0 + c3 * x0 * x1;

        // Coefficient order: [1, X1, X0, X0*X1]
        let coeffs = CoefficientList::new(vec![c0, c1, c2, c3]);

        // Create the weight in Evaluation mode at point (1, 0)
        let point = MultilinearPoint::new(vec![F::ONE, F::ZERO]);
        let weight = Weights::evaluation(point);

        // Manually compute expected f(1, 0):
        let expected = f(F::ONE, F::ZERO);

        // Check that Weights::evaluate returns the correct result
        assert_eq!(weight.evaluate_evals(&coeffs.to_evaluations()), expected);
    }

    #[test]
    fn test_evaluate_in_linear_mode() {
        // Define a multilinear polynomial f(X0, X1) = 3 + 4*X1 + 5*X0 + 6*X0*X1
        let c0 = F::from_u64(3);
        let c1 = F::from_u64(4);
        let c2 = F::from_u64(5);
        let c3 = F::from_u64(6);

        let coeffs = CoefficientList::new(vec![c0, c1, c2, c3]);

        // Evaluate f at every Boolean input in {0,1}²
        // Corner ordering: (0,0), (0,1), (1,0), (1,1)
        let evals = coeffs.clone().to_evaluations();

        // Define linear weights w_i for each corner:
        let w0 = F::from_u64(10);
        let w1 = F::from_u64(11);
        let w2 = F::from_u64(12);
        let w3 = F::from_u64(13);
        let weights = Weights::linear(EvaluationsList::new(vec![w0, w1, w2, w3]));

        // Expected weighted sum:
        // sum_i w_i * f_i
        let expected = w0 * evals[0] + w1 * evals[1] + w2 * evals[2] + w3 * evals[3];

        // Call the evaluate method
        let result = weights.evaluate_evals(&coeffs.to_evaluations());

        // Check that the evaluation matches the expected sum
        assert_eq!(result, expected);
    }

    #[test]
    #[should_panic]
    fn test_evaluate_panics_on_mismatched_vars() {
        // Define a polynomial with 1 variable
        let coeffs = CoefficientList::new(vec![F::ONE, F::TWO]);

        // Define a weight expecting 2 variables (linear mode)
        let weights = Weights::linear(EvaluationsList::new(vec![F::ONE; 4]));

        // This should panic because num_variables mismatch
        let _ = weights.evaluate_evals(&coeffs.to_evaluations());
    }
}
