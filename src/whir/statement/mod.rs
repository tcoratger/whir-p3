use p3_field::{ExtensionField, Field};
use tracing::instrument;

use crate::poly::{evals::EvaluationsList, multilinear::MultilinearPoint};

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
    /// List of evaluation points.
    points: Vec<MultilinearPoint<F>>,
    /// List of target evaluations.
    evaluations: Vec<F>,
}

impl<F: Field> Statement<F> {
    /// Creates an empty `Statement<F>` for polynomials with `num_variables` variables.
    #[must_use]
    pub const fn initialize(num_variables: usize) -> Self {
        Self {
            num_variables,
            points: Vec::new(),
            evaluations: Vec::new(),
        }
    }

    /// Creates a filled `Statement<F>` for polynomials with `num_variables` variables.
    #[must_use]
    pub const fn new(
        num_variables: usize,
        points: Vec<MultilinearPoint<F>>,
        evaluations: Vec<F>,
    ) -> Self {
        Self {
            num_variables,
            points,
            evaluations,
        }
    }

    /// Returns the number of variables defining the polynomial space.
    #[must_use]
    pub const fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// Returns true if the statement contains no constraints.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        debug_assert!(self.points.is_empty() == self.evaluations.is_empty());
        self.points.is_empty()
    }

    /// Returns an iterator over the evaluation constraints in the statement.
    pub fn iter(&self) -> impl Iterator<Item = (&MultilinearPoint<F>, &F)> {
        self.points.iter().zip(self.evaluations.iter())
    }

    /// Returns the number of constraints in the statement.
    #[must_use]
    pub const fn len(&self) -> usize {
        debug_assert!(self.points.len() == self.evaluations.len());
        self.points.len()
    }

    /// Verifies that a given polynomial satisfies all constraints in the statement.
    #[must_use]
    pub fn verify(&self, poly: &EvaluationsList<F>) -> bool {
        self.iter().all(|(point, &expected_eval)| {
            let eval = poly.evaluate(point);
            eval == expected_eval
        })
    }

    /// Concatenates another statement's constraints into this one.
    pub fn concatenate(&mut self, other: &Self) {
        assert_eq!(self.num_variables, other.num_variables);
        self.points.extend_from_slice(&other.points);
        self.evaluations.extend_from_slice(&other.evaluations);
    }

    /// Returns the vector of evaluation points.
    #[must_use]
    pub fn get_points(self) -> Vec<MultilinearPoint<F>> {
        self.points
    }

    /// Adds an evaluation constraint `p(z) = s` to the system.
    ///
    /// This method takes the polynomial `p` and uses it to compute the evaluation `s`.
    ///
    /// # Panics
    /// Panics if the number of variables in the `point` does not match the statement.
    pub fn add_unevaluated_constraint<BF>(
        &mut self,
        point: MultilinearPoint<F>,
        poly: &EvaluationsList<BF>,
    ) where
        BF: Field,
        F: ExtensionField<BF>,
    {
        assert_eq!(point.num_variables(), self.num_variables());
        let eval = poly.evaluate(&point);
        self.points.push(point);
        self.evaluations.push(eval);
    }

    /// Adds an evaluation constraint `p(z) = s` to the system.
    ///
    /// Assumes the evaluation `s` is already known.
    ///
    /// # Panics
    /// Panics if the number of variables in the `point` does not match the statement.
    pub fn add_evaluated_constraint(&mut self, point: MultilinearPoint<F>, eval: F) {
        assert_eq!(point.num_variables(), self.num_variables());
        self.points.push(point);
        self.evaluations.push(eval);
    }

    /// Inserts multiple constraints at the front of the system.
    ///
    /// Panics if any constraint's number of variables does not match the system.
    pub fn add_constraints_in_front(&mut self, points: &[MultilinearPoint<F>], evaluations: &[F]) {
        // Store the number of variables expected by this statement.
        let n = self.num_variables();
        assert_eq!(points.len(), evaluations.len());
        for p in points {
            assert_eq!(p.num_variables(), n);
        }

        self.points.splice(0..0, points.iter().cloned());
        self.evaluations.splice(0..0, evaluations.iter().copied());
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
        if self.points.is_empty() {
            return (
                EvaluationsList::new(F::zero_vec(1 << self.num_variables)),
                F::ZERO,
            );
        }

        // Separate the first constraint from the rest.
        // This allows us to treat the first one specially:
        //   - We overwrite the buffer.
        //   - We avoid a runtime branch in the main loop.
        let (first_point, rest_points) = self.points.split_first().unwrap();
        let (first_eval, rest_evals) = self.evaluations.split_first().unwrap();

        // Initialize the combined evaluations with the first constraint's polynomial.
        let mut combined = EvaluationsList::new_from_point(first_point, F::ONE);

        // Initialize the combined expected sum with the first term: s_1 * γ^0 = s_1.
        let mut gamma = F::ONE;
        let mut sum = *first_eval;

        // Process the remaining constraints.
        for (point, &eval) in rest_points.iter().zip(rest_evals) {
            // Update γ to γ^i for this constraint.
            gamma *= challenge;

            // Add this constraint's weighted polynomial evaluations into the buffer
            combined.accumulate(point, gamma);

            // Add this constraint's contribution to the combined expected sum:
            sum += eval * gamma;
        }

        // Return:
        // - The combined polynomial W(X) in evaluation form.
        // - The combined expected sum S.
        (combined, sum)
    }

    /// Combines a list of evals into a single linear combination using powers of `gamma`,
    /// and updates the running claimed_eval in place.
    ///
    /// # Arguments
    /// - `claimed_eval`: Mutable reference to the total accumulated claimed eval so far. Updated in place.
    /// - `gamma`: A random extension field element used to weight the evals.
    ///
    /// # Returns
    /// A `Vec<F>` containing the powers of `gamma` used to combine each constraint.
    pub fn combine_evals(&self, claimed_eval: &mut F, gamma: F) -> Vec<F> {
        let gammas = gamma.powers().collect_n(self.len());

        for (expected_eval, &gamma) in self.evaluations.iter().zip(&gammas) {
            *claimed_eval += *expected_eval * gamma;
        }

        gammas
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
        let mut statement = Statement::initialize(1);
        let point = MultilinearPoint::new(vec![F::ONE]);
        let expected_eval = F::from_u64(7);
        statement.add_evaluated_constraint(point.clone(), expected_eval);

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
        let mut statement = Statement::initialize(2);

        // Constraint 1: evaluate at z1 = (1,0), expected value 5
        let point1 = MultilinearPoint::new(vec![F::ONE, F::ZERO]);
        let eval1 = F::from_u64(5);
        statement.add_evaluated_constraint(point1.clone(), eval1);

        // Constraint 2: evaluate at z2 = (0,1), expected value 7
        let point2 = MultilinearPoint::new(vec![F::ZERO, F::ONE]);
        let eval2 = F::from_u64(7);
        statement.add_evaluated_constraint(point2.clone(), eval2);

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
