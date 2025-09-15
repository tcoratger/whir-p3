use p3_field::{ExtensionField, Field};
use tracing::instrument;

use crate::{
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    whir::statement::constraint::Constraint,
};

pub mod constraint;
pub mod evaluator;

/// Represents a system of polynomial evaluation constraints over a Boolean hypercube.
///
/// A `Statement` consists of multiple constraints, each enforcing a relationship of the form:
///
/// ```ignore
/// p(z_i) = s_i
/// ```
///
/// where:
/// - `p(X)` is a multilinear polynomial over `{0,1}^n`.
/// - `z_i` is a specific point in the multilinear domain.
/// - `s_i` is the expected evaluation of `p` at `z_i`.
///
/// These individual constraints are combined into a single probabilistic check using a random
/// challenge `γ`. This is done by creating a combined weight polynomial `W(X)` and a combined
/// expected evaluation `S`.
///
/// The combined weight polynomial is a random linear combination of the equality polynomials for each point:
///
/// ```ignore
/// W(X) = Σ γ^(i-1) ⋅ eq_{z_i}(X)
/// ```
///
/// The combined expected evaluation is a random linear combination of the individual expected evaluations:
///
/// ```ignore
/// S = \sum γ^(i-1) ⋅ s_i
/// ```
///
/// This combined form `(W(X), S)` is then used in protocols like sumcheck to verify all original
/// constraints at once.
#[derive(Clone, Debug)]
pub struct Statement<F> {
    /// Number of variables in the multilinear polynomial space.
    num_variables: usize,
    /// List of constraints, each pairing an evaluation point with a target evaluation.
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

    /// Adds an evaluation constraint `p(z) = s` to the system.
    ///
    /// # Panics
    /// Panics if the number of variables in the `point` does not match the statement.
    pub fn add_constraint(&mut self, point: MultilinearPoint<F>, expected_evaluation: F) {
        assert_eq!(point.num_variables(), self.num_variables());
        self.constraints
            .push(Constraint::new(point, expected_evaluation));
    }

    /// Inserts an evaluation constraint `p(z) = s` at the front of the system.
    ///
    /// # Panics
    /// Panics if the number of variables in the `point` does not match the statement.
    pub fn add_constraint_in_front(&mut self, point: MultilinearPoint<F>, expected_evaluation: F) {
        assert_eq!(point.num_variables(), self.num_variables());
        self.constraints
            .insert(0, Constraint::new(point, expected_evaluation));
    }

    /// Inserts multiple constraints at the front of the system.
    ///
    /// Panics if any constraint's number of variables does not match the system.
    pub fn add_constraints_in_front(&mut self, constraints: Vec<(MultilinearPoint<F>, F)>) {
        // Store the number of variables expected by this statement.
        let n = self.num_variables();

        // Preallocate a vector for the converted constraints to avoid reallocations.
        let mut new_constraints = Vec::with_capacity(constraints.len());

        // Iterate through each (weights, sum) pair in the input.
        for (point, expected_evaluation) in constraints {
            // Ensure the number of variables in the weight matches the statement.
            assert_eq!(point.num_variables(), n);

            // Convert the pair into a full `Constraint` with `defer_evaluation = false`.
            new_constraints.push(Constraint::new(point, expected_evaluation));
        }

        // Insert all new constraints at the beginning of the existing list.
        self.constraints.splice(0..0, new_constraints);
    }

    /// Combines all constraints into a single aggregated polynomial and expected sum using a challenge.
    ///
    /// # Returns
    /// - `EvaluationsList<F>`: The evaluations of the combined weight polynomial `W(X)`.
    /// - `F`: The combined expected evaluation `S`.
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

        // Separate the first constraint from the rest.
        // This allows us to treat the first one specially:
        //   - We overwrite the buffer.
        //   - We avoid a runtime branch in the main loop.
        let (first, rest) = self.constraints.split_first().unwrap();

        // Initialize the combined evaluations with the first constraint's polynomial.
        let mut combined = EvaluationsList::new_from_point(first.point(), F::ONE);

        // Initialize the combined expected sum with the first term: s_1 * γ^0 = s_1.
        let mut gamma = F::ONE;
        let mut sum = first.expected_eval();

        // Process the remaining constraints.
        for c in rest {
            // Update γ to γ^i for this constraint.
            gamma *= challenge;

            // Add this constraint's weighted polynomial evaluations into the buffer
            combined.accumulate(c.point(), gamma);

            // Add this constraint's contribution to the combined expected sum:
            sum += c.expected_eval() * gamma;
        }

        // Return:
        // - The combined polynomial W(X) in evaluation form.
        // - The combined expected sum S.
        (combined, sum)
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};

    use super::*;
    use crate::whir::MultilinearPoint;

    type F = BabyBear;
    type EF4 = BinomialExtensionField<F, 4>;

    #[test]
    fn test_statement_combine_single_constraint() {
        let mut statement = Statement::new(1);
        let point = MultilinearPoint::new(vec![F::ONE]);
        let expected_eval = F::from_u64(7);
        statement.add_constraint(point.clone(), expected_eval);

        let challenge = F::from_u64(2); // This is unused with one constraint.
        let (combined_evals, combined_sum) = statement.combine::<F>(challenge);

        // Expected evals for eq_z(X) where z = (1).
        // For x=0, eq=0. For x=1, eq=1.
        let expected_combined_evals_vec = EvaluationsList::new_from_point(&point, F::ONE);

        assert_eq!(combined_evals, expected_combined_evals_vec);
        assert_eq!(combined_sum, expected_eval);
    }

    #[test]
    fn test_statement_with_multiple_constraints() {
        let mut statement = Statement::new(2);

        // Constraint 1: evaluate at z1 = (1,0), expected value 5
        let point1 = MultilinearPoint::new(vec![F::ONE, F::ZERO]);
        let eval1 = F::from_u64(5);
        statement.add_constraint(point1.clone(), eval1);

        // Constraint 2: evaluate at z2 = (0,1), expected value 7
        let point2 = MultilinearPoint::new(vec![F::ZERO, F::ONE]);
        let eval2 = F::from_u64(7);
        statement.add_constraint(point2.clone(), eval2);

        let challenge = F::from_u64(2);
        let (combined_evals, combined_sum) = statement.combine::<F>(challenge);

        // Expected evals: W(X) = eq_z1(X) + challenge * eq_z2(X)
        let mut expected_combined_evals_vec = EvaluationsList::new_from_point(&point1, F::ONE);
        expected_combined_evals_vec.accumulate(&point2, challenge);

        // Expected sum: S = s1 + challenge * s2
        let expected_combined_sum = eval1 + challenge * eval2;

        assert_eq!(combined_evals, expected_combined_evals_vec);
        assert_eq!(combined_sum, expected_combined_sum);
    }

    #[test]
    fn test_compute_evaluation_weight() {
        // Define an evaluation weight at a specific point
        let point = MultilinearPoint::new(vec![F::from_u64(3)]);

        // Define a randomness point for folding
        let folding_randomness = MultilinearPoint::new(vec![F::from_u64(2)]);

        // Expected result is the evaluation of eq_poly at the given randomness
        let expected = point.eq_poly(&folding_randomness);

        assert_eq!(point.eq_poly(&folding_randomness), expected);
    }
}
