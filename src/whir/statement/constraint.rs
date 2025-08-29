use p3_field::Field;

use super::point::ConstraintPoint;
use crate::poly::evals::EvaluationsList;

/// Represents a single constraint in a polynomial statement, of the form `p(z) = s`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Constraint<F> {
    /// The point `z` at which the polynomial is constrained.
    pub point: ConstraintPoint<F>,

    /// The expected evaluation `s` that the polynomial should have at `point`.
    pub expected_evaluation: F,

    /// - If true, the verifier will not evaluate the weight directly.
    /// - If false, the verifier will evaluate the weight directly.
    ///
    /// This is used for deferred or externally computed evaluations.
    pub defer_evaluation: bool,
}

impl<F: Field> Constraint<F> {
    /// Verify if a polynomial satisfies the constraint.
    #[must_use]
    pub fn verify(&self, poly: &EvaluationsList<F>) -> bool {
        poly.evaluate(&self.point.0) == self.expected_evaluation
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    use super::*;
    use crate::{poly::coeffs::CoefficientList, whir::MultilinearPoint};

    type F = BabyBear;

    #[test]
    fn test_verify_passes_for_evaluation_constraint() {
        // Define a multilinear polynomial f(X0) = 2 + 3*X0
        let c0 = F::from_u64(2);
        let c1 = F::from_u64(3);
        let coeffs = CoefficientList::new(vec![c0, c1]);

        let f = |x0: F| c0 + c1 * x0;

        // Define a weight enforcing evaluation at point X0 = 1
        let point = MultilinearPoint::new(vec![F::ONE]);
        let constraint_point = ConstraintPoint::new(point);

        // Compute f(1):
        let expected_evaluation = f(F::ONE);

        let constraint = Constraint {
            point: constraint_point,
            expected_evaluation,
            defer_evaluation: false,
        };

        assert!(
            constraint.verify(&coeffs.to_evaluations()),
            "Constraint should pass"
        );
    }

    #[test]
    fn test_verify_fails_for_wrong_sum_in_evaluation_constraint() {
        // f(X0) = 2 + 3*X0
        let c0 = F::from_u64(2);
        let c1 = F::from_u64(3);
        let coeffs = CoefficientList::new(vec![c0, c1]);

        // Weight: evaluate at X0 = 1
        let point = MultilinearPoint::new(vec![F::ONE]);
        let point = ConstraintPoint::new(point);

        // Wrong sum: f(1) = 5, but sum = 6
        let expected_evaluation = F::from_u64(6);

        let constraint = Constraint {
            point,
            expected_evaluation,
            defer_evaluation: false,
        };

        assert!(
            !constraint.verify(&coeffs.to_evaluations()),
            "Constraint should fail due to incorrect sum"
        );
    }
}
