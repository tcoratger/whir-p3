use p3_field::Field;

use crate::poly::{evals::EvaluationsList, multilinear::MultilinearPoint};

/// Represents a single constraint in a polynomial statement.
///
/// The constraint is of the form `p(point) = expected_evaluation`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Constraint<F> {
    /// The point at which the polynomial is constrained.
    point: MultilinearPoint<F>,

    /// The expected evaluation that the polynomial should have at `point`.
    expected_evaluation: F,
}

impl<F: Field> Constraint<F> {
    /// Creates a new non-deferred constraint.
    #[must_use]
    pub const fn new(point: MultilinearPoint<F>, expected_evaluation: F) -> Self {
        Self {
            point,
            expected_evaluation,
        }
    }

    /// Verify if a polynomial satisfies the constraint.
    ///
    /// This is used by the verifier.
    #[must_use]
    pub fn verify(&self, poly: &EvaluationsList<F>) -> bool {
        poly.evaluate(&self.point) == self.expected_evaluation
    }

    /// Returns the constraint point.
    #[must_use]
    pub const fn point(&self) -> &MultilinearPoint<F> {
        &self.point
    }

    /// Returns the number of variables in the constraint point.
    #[must_use]
    pub const fn num_variables(&self) -> usize {
        self.point.num_variables()
    }

    /// Returns the claimed evaluation at the constraint point.
    #[must_use]
    pub const fn expected_eval(&self) -> F {
        self.expected_evaluation
    }
}

/// Combines a list of constraints into a single linear combination using powers of `alpha`,
/// and updates the running claimed sum in place.
///
/// # Arguments
/// - `claimed_sum`: Mutable reference to the total accumulated claimed sum so far. Updated in place.
/// - `constraints`: A slice of `Constraint<EF>` each containing a known `sum` field.
/// - `alpha`: A random extension field element used to weight the constraints.
///
/// # Returns
/// A `Vec<EF>` containing the powers of `alpha` used to combine each constraint.
pub fn linear_combine_constraints<F: Field>(
    claimed_eval: &mut F,
    constraints: &[Constraint<F>],
    gamma: F,
) -> Vec<F> {
    let gammas = gamma.powers().collect_n(constraints.len());

    for (constraint, &gamma) in constraints.iter().zip(&gammas) {
        *claimed_eval += constraint.expected_evaluation * gamma;
    }

    gammas
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    use super::*;

    type F = BabyBear;

    #[test]
    fn test_verify_passes_for_simple_linear_polynomial() {
        // Define a simple multilinear polynomial in one variable: f(X_0) = 2 + 3*X_0.
        let f = |x0: F| F::from_u32(2) + F::from_u32(3) * x0;

        // We need the polynomial's evaluations over the boolean hypercube {0, 1}.
        // Calculate f(0) and f(1).
        let f0 = f(F::ZERO);
        let f1 = f(F::ONE);

        // The evaluations are stored in lexicographic order, so [f(0), f(1)].
        let poly_evals = EvaluationsList::new(vec![f0, f1]);

        // Define the constraint: we want to check the polynomial's value at the point X_0 = 1.
        let point = MultilinearPoint::new(vec![F::ONE]);

        // The expected evaluation at X_0 = 1 is f(1), which we already calculated.
        let expected_evaluation = f1;

        // Create the constraint with the point and the expected value.
        let constraint = Constraint {
            point,
            expected_evaluation,
        };

        // Verify that the polynomial (represented by its evaluations) satisfies the constraint.
        assert!(
            constraint.verify(&poly_evals),
            "Constraint should pass for a correct evaluation"
        );
    }

    #[test]
    fn test_verify_fails_for_incorrect_evaluation() {
        // Define the same simple polynomial in one variable: f(X_0) = 2 + 3*X_0.
        let f = |x0: F| F::from_u32(2) + F::from_u32(3) * x0;

        // We need the polynomial's evaluations over the boolean hypercube {0, 1}.
        //
        // The evaluations are stored in lexicographic order, so [f(0), f(1)].
        let poly_evals = EvaluationsList::new(vec![f(F::ZERO), f(F::ONE)]);

        // Define the constraint at the same point, X_0 = 1.
        let point = MultilinearPoint::new(vec![F::ONE]);

        // This time, we set a deliberately incorrect expected evaluation.
        //
        // The correct value of f(1) is 5, but we will claim it is 6.
        let incorrect_expected_evaluation = F::from_u32(6);

        // Create the constraint struct with the point and the incorrect expected value.
        let constraint = Constraint {
            point,
            expected_evaluation: incorrect_expected_evaluation,
        };

        // Verify that the constraint check fails.
        assert!(
            !constraint.verify(&poly_evals),
            "Constraint should fail due to incorrect expected evaluation"
        );
    }

    #[test]
    fn test_verify_passes_for_two_variable_polynomial() {
        // Define a multilinear polynomial in two variables: f(X_0, X_1) = 1 + 2*X_0 + 3*X_1.
        let f = |x0: F, x1: F| F::ONE + F::from_u32(2) * x0 + F::from_u32(3) * x1;

        // We need the polynomial's evaluations over the 2D boolean hypercube: {(0,0), (0,1), (1,0), (1,1)}.
        //
        // The order must be lexicographical for the input variables (X_0, X_1).
        let eval_00 = f(F::ZERO, F::ZERO);
        let eval_01 = f(F::ZERO, F::ONE);
        let eval_10 = f(F::ONE, F::ZERO);
        let eval_11 = f(F::ONE, F::ONE);

        // Create the `EvaluationsList` with these values in the correct order.
        let poly_evals = EvaluationsList::new(vec![eval_00, eval_01, eval_10, eval_11]);

        // Define a constraint at a point outside the hypercube, e.g., (X_0, X_1) = (5, 10).
        let point_x0 = F::from_u32(5);
        let point_x1 = F::from_u32(10);
        let point = MultilinearPoint::new(vec![point_x0, point_x1]);

        // Calculate the correct expected evaluation at this point.
        let expected_evaluation = f(point_x0, point_x1);

        // Create the constraint.
        let constraint = Constraint {
            point,
            expected_evaluation,
        };

        // Verify that the polynomial (represented by its evaluations) satisfies the constraint.
        assert!(
            constraint.verify(&poly_evals),
            "Constraint for a 2-variable polynomial should pass"
        );
    }

    #[test]
    fn test_verify_fails_for_constant_polynomial_with_wrong_value() {
        // Define a constant polynomial f(X_0, X_1) = 42.
        //
        // This means the function's output is always 42, regardless of the input.
        let constant_value = F::from_u32(42);

        // For a constant polynomial with n=2 variables, all 2^2=4 evaluations on the hypercube are the same.
        let poly_evals = EvaluationsList::new(vec![
            constant_value,
            constant_value,
            constant_value,
            constant_value,
        ]);

        // We can pick any arbitrary point to check the constraint.
        let point = MultilinearPoint::new(vec![F::from_u32(123), F::from_u32(456)]);

        // The true value at any point should be 42.
        //
        // We create a constraint with an incorrect value.
        let incorrect_expected_evaluation = F::from_u32(99);

        // Create the constraint with the incorrect expected value.
        let constraint = Constraint {
            point,
            expected_evaluation: incorrect_expected_evaluation,
        };

        // Verify that the check fails.
        //
        // The evaluation of a constant polynomial will always be 42, which is not 99.
        assert!(
            !constraint.verify(&poly_evals),
            "Constraint for a constant polynomial should fail with an incorrect expected value"
        );
    }
}
