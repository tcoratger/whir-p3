use p3_field::{Field, TwoAdicField};

use crate::{
    poly::{coeffs::CoefficientList, multilinear::MultilinearPoint},
    sumcheck::sumcheck_polynomial::SumcheckPolynomial,
};

#[derive(Default, Debug, Clone)]
pub(crate) struct ParsedRound<F> {
    pub(crate) folding_randomness: MultilinearPoint<F>,
    pub(crate) ood_points: Vec<F>,
    pub(crate) ood_answers: Vec<F>,
    pub(crate) stir_challenges_indexes: Vec<usize>,
    pub(crate) stir_challenges_points: Vec<F>,
    pub(crate) stir_challenges_answers: Vec<Vec<F>>,
    pub(crate) combination_randomness: Vec<F>,
    pub(crate) sumcheck_rounds: Vec<(SumcheckPolynomial<F>, F)>,
    pub(crate) domain_gen_inv: F,
}

#[derive(Default, Clone)]
pub(crate) struct ParsedProof<F> {
    pub(crate) initial_combination_randomness: Vec<F>,
    pub(crate) initial_sumcheck_rounds: Vec<(SumcheckPolynomial<F>, F)>,
    pub(crate) rounds: Vec<ParsedRound<F>>,
    pub(crate) final_domain_gen_inv: F,
    pub(crate) final_randomness_indexes: Vec<usize>,
    pub(crate) final_randomness_points: Vec<F>,
    pub(crate) final_randomness_answers: Vec<Vec<F>>,
    pub(crate) final_folding_randomness: MultilinearPoint<F>,
    pub(crate) final_sumcheck_rounds: Vec<(SumcheckPolynomial<F>, F)>,
    pub(crate) final_sumcheck_randomness: MultilinearPoint<F>,
    pub(crate) final_coefficients: CoefficientList<F>,
    pub(crate) statement_values_at_random_point: Vec<F>,
}

impl<F: Field + TwoAdicField> ParsedProof<F> {
    pub(crate) fn compute_folds_helped(&self) -> Vec<Vec<F>> {
        let mut result: Vec<_> = self
            .rounds
            .iter()
            .map(|round| {
                round
                    .stir_challenges_answers
                    .iter()
                    .map(|answers| {
                        CoefficientList::new(answers.clone()).evaluate(&round.folding_randomness)
                    })
                    .collect()
            })
            .collect();

        // Add final round if needed
        if !self.final_randomness_answers.is_empty() {
            result.push(
                self.final_randomness_answers
                    .iter()
                    .map(|answers| {
                        CoefficientList::new(answers.clone())
                            .evaluate(&self.final_folding_randomness)
                    })
                    .collect(),
            );
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    #[test]
    fn test_compute_folds_helped_basic_case() {
        // Define a simple coefficient list with four values
        // This represents a polynomial over `{X1, X2}`
        let stir_challenges_answers = vec![
            BabyBear::from_u64(1), // f(0,0)
            BabyBear::from_u64(2), // f(0,1)
            BabyBear::from_u64(3), // f(1,0)
            BabyBear::from_u64(4), // f(1,1)
        ];

        // Define a simple coefficient list with four values
        // This represents a polynomial over `{X1, X2}`
        let final_randomness_answers = vec![
            BabyBear::from_u64(5), // f(0,0)
            BabyBear::from_u64(6), // f(0,1)
            BabyBear::from_u64(7), // f(1,0)
            BabyBear::from_u64(8), // f(1,1)
        ];

        // The folding randomness values `(5,6)` will be applied to interpolate the polynomial.
        // This means we are evaluating the polynomial at `X1=5, X2=6`.
        let folding_randomness =
            MultilinearPoint(vec![BabyBear::from_u64(5), BabyBear::from_u64(6)]);

        // Final folding randomness values `(55,66)` will be applied to compute the last fold.
        // This means we are evaluating the polynomial at `X1=55, X2=66`.
        let final_folding_randomness =
            MultilinearPoint(vec![BabyBear::from_u64(55), BabyBear::from_u64(66)]);

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
                BabyBear::from_u64(1),
                BabyBear::from_u64(2),
                BabyBear::from_u64(3),
                BabyBear::from_u64(4),
            ])
            .evaluate(&MultilinearPoint(vec![BabyBear::from_u64(5), BabyBear::from_u64(6)])),
        ];

        // Expected final round evaluation:
        // f(55,66) = 5 + 6(66) + 7(55) + 8(55)(66) = 14718
        let expected_final_round = vec![
            CoefficientList::new(vec![
                BabyBear::from_u64(5),
                BabyBear::from_u64(6),
                BabyBear::from_u64(7),
                BabyBear::from_u64(8),
            ])
            .evaluate(&MultilinearPoint(vec![BabyBear::from_u64(55), BabyBear::from_u64(66)])),
        ];

        assert_eq!(folds, vec![expected_rounds, expected_final_round]);
    }

    #[test]
    fn test_compute_folds_helped_empty_proof() {
        let proof = ParsedProof::<BabyBear> {
            rounds: vec![], // No rounds
            final_folding_randomness: MultilinearPoint(vec![BabyBear::from_u64(1)]),
            final_randomness_answers: vec![],
            ..Default::default()
        };

        let folds = proof.compute_folds_helped();

        // Since there are no rounds, folds should be empty.
        assert!(folds.is_empty());
    }

    #[test]
    fn test_compute_folds_helped_single_variable() {
        let stir_challenges_answers = vec![
            BabyBear::from_u64(2), // f(0)
            BabyBear::from_u64(5), // f(1)
        ];

        let folding_randomness = MultilinearPoint(vec![BabyBear::from_u64(3)]); // Evaluating at X1=3

        let single_round = ParsedRound {
            folding_randomness,
            stir_challenges_answers: vec![stir_challenges_answers],
            ..Default::default()
        };

        let proof = ParsedProof {
            rounds: vec![single_round],
            final_folding_randomness: MultilinearPoint(vec![BabyBear::from_u64(7)]), /* Evaluating at X1=7 */
            final_randomness_answers: vec![vec![BabyBear::from_u64(8), BabyBear::from_u64(10)]],
            ..Default::default()
        };

        let folds = proof.compute_folds_helped();

        // Compute expected evaluation at X1=3:
        // f(3) = 2 + 5(3) = 17
        let expected_rounds = vec![
            CoefficientList::new(vec![BabyBear::from_u64(2), BabyBear::from_u64(5)])
                .evaluate(&MultilinearPoint(vec![BabyBear::from_u64(3)])),
        ];

        // Compute expected final round evaluation at X1=7:
        // f(7) = 8 + 10(7) = 78
        let expected_final_round = vec![
            CoefficientList::new(vec![BabyBear::from_u64(8), BabyBear::from_u64(10)])
                .evaluate(&MultilinearPoint(vec![BabyBear::from_u64(7)])),
        ];

        assert_eq!(folds, vec![expected_rounds, expected_final_round]);
    }

    #[test]
    fn test_compute_folds_helped_all_zeros() {
        let stir_challenges_answers = vec![BabyBear::ZERO; 4];

        let proof = ParsedProof {
            rounds: vec![ParsedRound {
                folding_randomness: MultilinearPoint(vec![
                    BabyBear::from_u64(4),
                    BabyBear::from_u64(5),
                ]),
                stir_challenges_answers: vec![stir_challenges_answers.clone()],
                ..Default::default()
            }],
            final_folding_randomness: MultilinearPoint(vec![
                BabyBear::from_u64(10),
                BabyBear::from_u64(20),
            ]),
            final_randomness_answers: vec![stir_challenges_answers],
            ..Default::default()
        };

        let folds = proof.compute_folds_helped();

        // Since all coefficients are zero, every evaluation must be zero.
        assert_eq!(folds, vec![vec![BabyBear::ZERO], vec![BabyBear::ZERO]]);
    }
}
