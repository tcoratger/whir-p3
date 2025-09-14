use p3_field::{ExtensionField, Field, TwoAdicField};

use crate::{
    constant::K_SKIP_SUMCHECK,
    poly::multilinear::MultilinearPoint,
    parameters::FoldingFactor,
    whir::statement::{constraint::Constraint, Statement},
};

/// Handles constraint evaluation logic for the WHIR prover.
///
/// This component is responsible for computing constraint evaluations both in
/// standard and univariate skip modes, managing the evaluation point construction,
/// and handling deferred constraints appropriately.
#[derive(Debug)]
pub struct ConstraintEvaluator<EF, F> {
    /// Number of variables in the polynomial
    num_variables: usize,
    /// Folding factor configuration
    folding_factor: FoldingFactor,
    /// Whether univariate skip optimization is enabled
    univariate_skip: bool,
    /// Phantom data for field types
    _phantom: std::marker::PhantomData<(EF, F)>,
}

impl<EF, F> ConstraintEvaluator<EF, F>
where
    EF: ExtensionField<F> + TwoAdicField,
    F: Field + TwoAdicField,
{
    /// Creates a new constraint evaluator.
    pub fn new(
        num_variables: usize,
        folding_factor: FoldingFactor,
        univariate_skip: bool,
    ) -> Self {
        Self {
            num_variables,
            folding_factor,
            univariate_skip,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Computes the constraint evaluation point based on the current configuration.
    ///
    /// In univariate skip mode, this uses a reduced dimensionality evaluation point
    /// that matches the final polynomial structure. In standard mode, it uses the
    /// full randomness vector.
    ///
    /// # Arguments
    ///
    /// * `randomness_vec` - Vector of randomness values from sumcheck rounds
    /// * `n_vars_final` - Number of variables in the final polynomial
    ///
    /// # Returns
    ///
    /// The evaluation point for constraints and list of deferred constraint evaluations
    pub fn compute_evaluation(
        &self,
        randomness_vec: &[EF],
        n_vars_final: usize,
        statement: &Statement<EF>,
    ) -> (MultilinearPoint<EF>, Vec<EF>) {
        let constraint_eval = if self.should_use_skip_evaluation(n_vars_final) {
            self.create_skip_evaluation_point(randomness_vec, n_vars_final)
        } else {
            MultilinearPoint::new(randomness_vec.to_vec())
        };

        let deferred_evaluations = self.compute_deferred_constraints(statement, &constraint_eval);

        (constraint_eval, deferred_evaluations)
    }

    /// Determines if univariate skip evaluation should be used.
    fn should_use_skip_evaluation(&self, n_vars_final: usize) -> bool {
        self.univariate_skip
            && self.folding_factor.at_round(0) >= K_SKIP_SUMCHECK
            && n_vars_final < self.num_variables
    }

    /// Creates the evaluation point for univariate skip mode.
    ///
    /// Takes the appropriate number of elements from the reversed randomness vector
    /// to match the expected final polynomial dimensionality.
    fn create_skip_evaluation_point(
        &self,
        randomness_vec: &[EF],
        expected_len: usize,
    ) -> MultilinearPoint<EF> {
        MultilinearPoint::new(
            randomness_vec
                .iter()
                .rev()
                .take(expected_len)
                .copied()
                .collect(),
        )
    }

    /// Computes evaluations for all deferred constraints.
    ///
    /// Deferred constraints use either standard eq_poly evaluation or skip-aware
    /// eq_poly_with_skip evaluation based on the current configuration.
    fn compute_deferred_constraints(
        &self,
        statement: &Statement<EF>,
        constraint_eval: &MultilinearPoint<EF>,
    ) -> Vec<EF> {
        statement
            .constraints
            .iter()
            .filter(|constraint| constraint.defer_evaluation)
            .map(|constraint| self.evaluate_single_constraint(constraint, constraint_eval))
            .collect()
    }

    /// Evaluates a single constraint using the appropriate evaluation method.
    fn evaluate_single_constraint(
        &self,
        constraint: &Constraint<EF>,
        constraint_eval: &MultilinearPoint<EF>,
    ) -> EF {
        let uses_skip = self.should_use_skip_for_constraint(constraint, constraint_eval);

        if uses_skip {
            constraint
                .point
                .eq_poly_with_skip(constraint_eval, K_SKIP_SUMCHECK)
        } else {
            constraint.point.eq_poly(constraint_eval)
        }
    }

    /// Determines if skip evaluation should be used for a specific constraint.
    fn should_use_skip_for_constraint(
        &self,
        constraint: &Constraint<EF>,
        constraint_eval: &MultilinearPoint<EF>,
    ) -> bool {
        self.univariate_skip
            && self.folding_factor.at_round(0) >= K_SKIP_SUMCHECK
            && constraint.point.num_variables() == self.num_variables
            && constraint_eval.num_variables() < self.num_variables
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    #[test]
    fn test_skip_evaluation_detection() {
        let evaluator = ConstraintEvaluator::<EF, F>::new(
            6,
            FoldingFactor::ConstantFromSecondRound(5, 2),
            true,
        );

        assert!(evaluator.should_use_skip_evaluation(1));
        assert!(!evaluator.should_use_skip_evaluation(6));
    }

    #[test]
    fn test_standard_evaluation_detection() {
        let evaluator = ConstraintEvaluator::<EF, F>::new(
            6,
            FoldingFactor::Constant(2),
            false,
        );

        assert!(!evaluator.should_use_skip_evaluation(1));
        assert!(!evaluator.should_use_skip_evaluation(6));
    }
}