use p3_field::Field;

use crate::{
    poly::{coeffs::CoefficientList, multilinear::MultilinearPoint},
    sumcheck::sumcheck_polynomial::SumcheckPolynomial,
};

/// Represents a single folding round in the WHIR protocol.
///
/// This structure enables recursive compression and verification of a Reed–Solomon
/// proximity test under algebraic constraints.
#[derive(Default, Debug, Clone)]
pub(crate) struct ParsedRound<F> {
    /// Folding randomness vector used in this round.
    pub(crate) folding_randomness: MultilinearPoint<F>,
    /// Out-of-domain query points.
    pub(crate) ood_points: Vec<F>,
    /// OOD answers at each query point.
    pub(crate) ood_answers: Vec<F>,
    /// Indexes of STIR constraint polynomials used in this round.
    pub(crate) stir_challenges_indexes: Vec<usize>,
    /// STIR constraint evaluation points.
    pub(crate) stir_challenges_points: Vec<F>,
    /// Answers to the STIR constraints at each evaluation point.
    pub(crate) stir_challenges_answers: Vec<Vec<F>>,
    /// Randomness used to linearly combine constraints.
    pub(crate) combination_randomness: Vec<F>,
    /// Sumcheck messages and challenge values for verifying correctness.
    pub(crate) sumcheck_rounds: Vec<(SumcheckPolynomial<F>, F)>,
    /// Inverse of the domain generator used in this round.
    pub(crate) domain_gen_inv: F,
}

/// Represents a fully parsed and structured WHIR proof.
///
/// The structure is designed to support recursive verification and evaluation
/// of folded functions under STIR-style constraints.
#[derive(Default, Clone)]
pub(crate) struct ParsedProof<F> {
    /// Initial random coefficients used to combine constraints before folding.
    pub(crate) initial_combination_randomness: Vec<F>,
    /// Initial sumcheck messages and challenges for the first constraint.
    pub(crate) initial_sumcheck_rounds: Vec<(SumcheckPolynomial<F>, F)>,
    /// All folding rounds, each reducing the problem dimension.
    pub(crate) rounds: Vec<ParsedRound<F>>,
    /// Inverse of the domain generator used in the final round.
    pub(crate) final_domain_gen_inv: F,
    /// Indexes of the final constraint polynomials.
    pub(crate) final_randomness_indexes: Vec<usize>,
    /// Evaluation points for the final constraint polynomials.
    pub(crate) final_randomness_points: Vec<F>,
    /// Evaluation results of the final constraints.
    pub(crate) final_randomness_answers: Vec<Vec<F>>,
    /// Folding randomness used in the final recursive step.
    pub(crate) final_folding_randomness: MultilinearPoint<F>,
    /// Final sumcheck proof for verifying the last constraint.
    pub(crate) final_sumcheck_rounds: Vec<(SumcheckPolynomial<F>, F)>,
    /// Challenge vector used to evaluate the last polynomial.
    pub(crate) final_sumcheck_randomness: MultilinearPoint<F>,
    /// Coefficients of the final small polynomial.
    pub(crate) final_coefficients: CoefficientList<F>,
    /// Evaluation values of the statement being proven at a random point.
    pub(crate) statement_values_at_random_point: Vec<F>,
}

impl<F> ParsedProof<F>
where
    F: Field,
{
    /// Computes all intermediate fold evaluations using prover-assisted folding.
    ///
    /// For each round, this evaluates the STIR answers as multilinear polynomials
    /// at the provided folding randomness point. This simulates what the verifier
    /// would receive in a sound recursive sumcheck-based proximity test.
    ///
    /// Returns:
    /// - A vector of vectors, where each inner vector contains the evaluated result
    ///   of each multilinear polynomial at its corresponding folding point.
    pub(crate) fn compute_folds_helped(&self) -> Vec<Vec<F>> {
        // Closure to apply folding evaluation logic.
        let evaluate_answers = |answers: &[Vec<F>], randomness: &MultilinearPoint<F>| {
            answers
                .iter()
                .map(|answers| CoefficientList::new(answers.clone()).evaluate(randomness))
                .collect()
        };

        let mut result: Vec<_> = self
            .rounds
            .iter()
            .map(|round| {
                evaluate_answers(&round.stir_challenges_answers, &round.folding_randomness)
            })
            .collect();

        result.push(evaluate_answers(
            &self.final_randomness_answers,
            &self.final_folding_randomness,
        ));
        result
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    use super::*;

    type F = BabyBear;

    #[test]
    fn test_compute_folds_helped_basic_case() {
        // Define a simple coefficient list with four values
        // This represents a polynomial over `{X1, X2}`
        let stir_challenges_answers = vec![
            F::from_u64(1), // f(0,0)
            F::from_u64(2), // f(0,1)
            F::from_u64(3), // f(1,0)
            F::from_u64(4), // f(1,1)
        ];

        // Define a simple coefficient list with four values
        // This represents a polynomial over `{X1, X2}`
        let final_randomness_answers = vec![
            F::from_u64(5), // f(0,0)
            F::from_u64(6), // f(0,1)
            F::from_u64(7), // f(1,0)
            F::from_u64(8), // f(1,1)
        ];

        // The folding randomness values `(5,6)` will be applied to interpolate the polynomial.
        // This means we are evaluating the polynomial at `X1=5, X2=6`.
        let folding_randomness = MultilinearPoint(vec![F::from_u64(5), F::from_u64(6)]);

        // Final folding randomness values `(55,66)` will be applied to compute the last fold.
        // This means we are evaluating the polynomial at `X1=55, X2=66`.
        let final_folding_randomness = MultilinearPoint(vec![F::from_u64(55), F::from_u64(66)]);

        let single_round = ParsedRound {
            folding_randomness,
            stir_challenges_answers: vec![stir_challenges_answers],
            ..Default::default()
        };

        let proof = ParsedProof {
            rounds: vec![single_round],
            final_folding_randomness,
            final_randomness_answers: vec![final_randomness_answers],
            ..Default::default()
        };

        let folds = proof.compute_folds_helped();

        // Expected first-round evaluation:
        // f(5,6) = 1 + 2(6) + 3(5) + 4(5)(6) = 148
        let expected_rounds = vec![
            CoefficientList::new(vec![
                F::from_u64(1),
                F::from_u64(2),
                F::from_u64(3),
                F::from_u64(4),
            ])
            .evaluate(&MultilinearPoint(vec![F::from_u64(5), F::from_u64(6)])),
        ];

        // Expected final round evaluation:
        // f(55,66) = 5 + 6(66) + 7(55) + 8(55)(66) = 14718
        let expected_final_round = vec![
            CoefficientList::new(vec![
                F::from_u64(5),
                F::from_u64(6),
                F::from_u64(7),
                F::from_u64(8),
            ])
            .evaluate(&MultilinearPoint(vec![F::from_u64(55), F::from_u64(66)])),
        ];

        assert_eq!(folds, vec![expected_rounds, expected_final_round]);
    }

    #[test]
    fn test_compute_folds_helped_single_variable() {
        let stir_challenges_answers = vec![
            F::from_u64(2), // f(0)
            F::from_u64(5), // f(1)
        ];

        let folding_randomness = MultilinearPoint(vec![F::from_u64(3)]); // Evaluating at X1=3

        let single_round = ParsedRound {
            folding_randomness,
            stir_challenges_answers: vec![stir_challenges_answers],
            ..Default::default()
        };

        let proof = ParsedProof {
            rounds: vec![single_round],
            final_folding_randomness: MultilinearPoint(vec![F::from_u64(7)]), /* Evaluating at X1=7 */
            final_randomness_answers: vec![vec![F::from_u64(8), F::from_u64(10)]],
            ..Default::default()
        };

        let folds = proof.compute_folds_helped();

        // Compute expected evaluation at X1=3:
        // f(3) = 2 + 5(3) = 17
        let expected_rounds = vec![
            CoefficientList::new(vec![F::from_u64(2), F::from_u64(5)])
                .evaluate(&MultilinearPoint(vec![F::from_u64(3)])),
        ];

        // Compute expected final round evaluation at X1=7:
        // f(7) = 8 + 10(7) = 78
        let expected_final_round = vec![
            CoefficientList::new(vec![F::from_u64(8), F::from_u64(10)])
                .evaluate(&MultilinearPoint(vec![F::from_u64(7)])),
        ];

        assert_eq!(folds, vec![expected_rounds, expected_final_round]);
    }

    #[test]
    fn test_compute_folds_helped_all_zeros() {
        let stir_challenges_answers = vec![F::ZERO; 4];

        let proof = ParsedProof {
            rounds: vec![ParsedRound {
                folding_randomness: MultilinearPoint(vec![F::from_u64(4), F::from_u64(5)]),
                stir_challenges_answers: vec![stir_challenges_answers.clone()],
                ..Default::default()
            }],
            final_folding_randomness: MultilinearPoint(vec![F::from_u64(10), F::from_u64(20)]),
            final_randomness_answers: vec![stir_challenges_answers],
            ..Default::default()
        };

        let folds = proof.compute_folds_helped();

        // Since all coefficients are zero, every evaluation must be zero.
        assert_eq!(folds, vec![vec![F::ZERO], vec![F::ZERO]]);
    }
}
