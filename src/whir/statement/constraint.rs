use p3_field::Field;

use crate::{poly::evals::EvaluationsList, whir::Weights};

/// Represents a single constraint in a polynomial statement.
#[derive(Clone, Debug, Eq, PartialEq)]
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

impl<F: Field> Constraint<F> {
    /// Verify if a polynomial (in coefficient form) satisfies the constraint.
    #[must_use]
    pub fn verify(&self, poly: &EvaluationsList<F>) -> bool {
        self.weights.evaluate_evals(poly) == self.sum
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    use super::*;
    use crate::{
        poly::{coeffs::CoefficientList, evals::EvaluationsList},
        whir::MultilinearPoint,
    };

    type F = BabyBear;

    #[test]
    fn test_verify_passes_for_evaluation_constraint() {
        // Define a multilinear polynomial f(X0) = 2 + 3*X0
        let c0 = F::from_u64(2);
        let c1 = F::from_u64(3);
        let coeffs = CoefficientList::new(vec![c0, c1]);

        let f = |x0: F| c0 + c1 * x0;

        // Define a weight enforcing evaluation at point X0 = 1
        let point = MultilinearPoint(vec![F::ONE]);
        let weights = Weights::evaluation(point);

        // Compute f(1):
        let sum = f(F::ONE);

        let constraint = Constraint {
            weights,
            sum,
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
        let point = MultilinearPoint(vec![F::ONE]);
        let weights = Weights::evaluation(point);

        // Wrong sum: f(1) = 5, but sum = 6
        let sum = F::from_u64(6);

        let constraint = Constraint {
            weights,
            sum,
            defer_evaluation: false,
        };

        assert!(
            !constraint.verify(&coeffs.to_evaluations()),
            "Constraint should fail due to incorrect sum"
        );
    }

    #[test]
    fn test_verify_passes_for_linear_constraint() {
        // f(X0) = 1 + 2*X0
        let c0 = F::from_u64(1);
        let c1 = F::from_u64(2);
        let coeffs = CoefficientList::new(vec![c0, c1]);

        // Evaluate on {0,1}: f(0) = 1, f(1) = 3
        let evals = [F::from_u64(1), F::from_u64(3)];

        // Linear weights: w = [2, 4]
        let w0 = F::from_u64(2);
        let w1 = F::from_u64(4);
        let weights = Weights::linear(EvaluationsList::new(vec![w0, w1]));

        // Compute expected weighted sum:
        let sum = w0 * evals[0] + w1 * evals[1];

        let constraint = Constraint {
            weights,
            sum,
            defer_evaluation: false,
        };

        assert!(
            constraint.verify(&coeffs.to_evaluations()),
            "Linear constraint should pass"
        );
    }

    #[test]
    fn test_verify_passes_for_linear_constraint_3_vars() {
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

        // Coefficient list follows lex order:
        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4, c5, c6, c7, c8]);

        // Evaluate the polynomial at all Boolean points {0,1}^3
        let evals = coeffs.clone().to_evaluations(); // length 8

        // Define linear weights for each Boolean input (arbitrary example)
        let w0 = F::from_u64(1); // weight for (0,0,0)
        let w1 = F::from_u64(2); // weight for (0,0,1)
        let w2 = F::from_u64(3); // weight for (0,1,0)
        let w3 = F::from_u64(4); // weight for (0,1,1)
        let w4 = F::from_u64(5); // weight for (1,0,0)
        let w5 = F::from_u64(6); // weight for (1,0,1)
        let w6 = F::from_u64(7); // weight for (1,1,0)
        let w7 = F::from_u64(8); // weight for (1,1,1)

        let weight_vec = vec![w0, w1, w2, w3, w4, w5, w6, w7];
        let weights = Weights::linear(EvaluationsList::new(weight_vec.clone()));

        // Compute expected weighted sum manually:
        //
        // expected = Σ_i w_i * f_i
        let expected_sum = (0..8)
            .map(|i| weight_vec[i] * evals[i])
            .fold(F::ZERO, |acc, x| acc + x);

        let constraint = Constraint {
            weights,
            sum: expected_sum,
            defer_evaluation: false,
        };

        // The constraint should pass since weights × evaluations == expected_sum
        assert!(
            constraint.verify(&coeffs.to_evaluations()),
            "Linear constraint over 3-variable polynomial should pass"
        );
    }
}
