use std::fmt::Debug;

use p3_commit::{ExtensionMmcs, Mmcs};
use p3_field::{ExtensionField, Field, PrimeField64, TwoAdicField};
use p3_matrix::Dimensions;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CryptographicHasher, Hash, PseudoCompressionFunction};
use serde::{Deserialize, Serialize};

use super::{Verifier, parsed_round::ParsedRound};
use crate::{
    fiat_shamir::{
        errors::{ProofError, ProofResult},
        pow::traits::PowStrategy,
        verifier::VerifierState,
    },
    poly::{coeffs::CoefficientList, multilinear::MultilinearPoint},
    sumcheck::sumcheck_polynomial::SumcheckPolynomial,
    utils::expand_randomness,
    whir::{
        committer::reader::ParsedCommitment, prover::proof::WhirProof,
        utils::get_challenge_stir_queries,
    },
};

/// Represents a fully parsed and structured WHIR proof.
///
/// The structure is designed to support recursive verification and evaluation
/// of folded functions under STIR-style constraints.
#[derive(Default, Clone, Debug)]
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
    #[allow(clippy::too_many_lines)]
    pub(crate) fn from_prover_output<SF, H, C, PS, const DIGEST_ELEMS: usize>(
        verifier: &Verifier<'_, F, SF, H, C, PS>,
        verifier_state: &mut VerifierState<'_, F, SF>,
        parsed_commitment: &ParsedCommitment<F, Hash<SF, u8, DIGEST_ELEMS>>,
        statement_points_len: usize,
        whir_proof: &WhirProof<SF, F, DIGEST_ELEMS>,
    ) -> ProofResult<Self>
    where
        H: CryptographicHasher<SF, [u8; DIGEST_ELEMS]> + Sync,
        C: PseudoCompressionFunction<[u8; DIGEST_ELEMS], 2> + Sync,
        [u8; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
        SF: Field + TwoAdicField + PrimeField64,
        F: ExtensionField<SF> + TwoAdicField,
        PS: PowStrategy,
    {
        let mmcs = MerkleTreeMmcs::new(
            verifier.params.merkle_hash.clone(),
            verifier.params.merkle_compress.clone(),
        );

        let extension_mmcs = ExtensionMmcs::new(mmcs.clone());

        let mut sumcheck_rounds = Vec::new();
        let mut folding_randomness;
        let initial_combination_randomness;

        if verifier.params.initial_statement {
            // Derive combination randomness and first sumcheck polynomial
            let [combination_randomness_gen] = verifier_state.challenge_scalars()?;
            initial_combination_randomness = expand_randomness(
                combination_randomness_gen,
                parsed_commitment.ood_points.len() + statement_points_len,
            );

            // Initial sumcheck
            sumcheck_rounds.reserve_exact(verifier.params.folding_factor.at_round(0));
            for _ in 0..verifier.params.folding_factor.at_round(0) {
                let sumcheck_poly_evals: [_; 3] = verifier_state.next_scalars()?;
                let sumcheck_poly = SumcheckPolynomial::new(sumcheck_poly_evals.to_vec(), 1);
                let [folding_randomness_single] = verifier_state.challenge_scalars()?;
                sumcheck_rounds.push((sumcheck_poly, folding_randomness_single));

                if verifier.params.starting_folding_pow_bits > 0. {
                    verifier_state
                        .challenge_pow::<PS>(verifier.params.starting_folding_pow_bits)?;
                }
            }

            folding_randomness =
                MultilinearPoint(sumcheck_rounds.iter().map(|&(_, r)| r).rev().collect());
        } else {
            assert_eq!(parsed_commitment.ood_points.len(), 0);
            assert_eq!(statement_points_len, 0);

            initial_combination_randomness = vec![F::ONE];

            let mut folding_randomness_vec =
                vec![F::ZERO; verifier.params.folding_factor.at_round(0)];
            verifier_state.fill_challenge_scalars(&mut folding_randomness_vec)?;
            folding_randomness = MultilinearPoint(folding_randomness_vec);

            // PoW
            if verifier.params.starting_folding_pow_bits > 0. {
                verifier_state.challenge_pow::<PS>(verifier.params.starting_folding_pow_bits)?;
            }
        }

        let mut prev_root = parsed_commitment.root;
        let mut domain_gen = verifier.params.starting_domain.backing_domain.group_gen();
        let mut exp_domain_gen =
            domain_gen.exp_u64(1 << verifier.params.folding_factor.at_round(0));
        let mut domain_gen_inv = verifier
            .params
            .starting_domain
            .backing_domain
            .group_gen_inv();
        let mut domain_size = verifier.params.starting_domain.size();
        let mut rounds = vec![];

        for r in 0..verifier.params.n_rounds() {
            let round_params = &verifier.params.round_parameters[r];
            let fold_r = verifier.params.folding_factor.at_round(r);

            let new_root = verifier_state.read_digest()?;

            let mut ood_points = vec![F::ZERO; round_params.ood_samples];
            let mut ood_answers = vec![F::ZERO; round_params.ood_samples];
            if round_params.ood_samples > 0 {
                verifier_state.fill_challenge_scalars(&mut ood_points)?;
                verifier_state.fill_next_scalars(&mut ood_answers)?;
            }

            let stir_challenges_indexes = get_challenge_stir_queries(
                domain_size,
                fold_r,
                round_params.num_queries,
                verifier_state,
            )?;

            let mut stir_challenges_points = Vec::with_capacity(stir_challenges_indexes.len());

            // Verify Merkle openings using `verify_batch`
            let dimensions = vec![Dimensions {
                height: domain_size >> fold_r,
                width: 1 << fold_r,
            }];

            let mut stir_challenges_answers = Vec::new();

            for (i, &stir_challenges_index) in stir_challenges_indexes.iter().enumerate() {
                stir_challenges_points.push(exp_domain_gen.exp_u64(stir_challenges_index as u64));
                if r == 0 {
                    let (answers, merkle_proof) = &whir_proof.commitment_merkle_paths;
                    let indexed_answers = answers[i].iter().map(|v| v.as_base().unwrap()).collect();

                    mmcs.verify_batch(
                        &prev_root,
                        &dimensions,
                        stir_challenges_index,
                        &[indexed_answers],
                        &merkle_proof[i],
                    )
                    .map_err(|_| ProofError::InvalidProof)?;
                    stir_challenges_answers = answers
                        .iter()
                        .map(|inner| inner.iter().map(|&f_el| f_el.into()).collect())
                        .collect();
                } else {
                    let (answers, merkle_proof) = &whir_proof.merkle_paths[r - 1];
                    extension_mmcs
                        .verify_batch(
                            &prev_root,
                            &dimensions,
                            stir_challenges_index,
                            &[answers[i].clone()],
                            &merkle_proof[i],
                        )
                        .map_err(|_| ProofError::InvalidProof)?;
                    stir_challenges_answers.clone_from(answers);
                }
            }

            if round_params.pow_bits > 0. {
                verifier_state.challenge_pow::<PS>(round_params.pow_bits)?;
            }

            let [combination_randomness_gen] = verifier_state.challenge_scalars()?;
            let combination_randomness = expand_randomness(
                combination_randomness_gen,
                stir_challenges_indexes.len() + round_params.ood_samples,
            );

            let mut sumcheck_rounds =
                Vec::with_capacity(verifier.params.folding_factor.at_round(r + 1));

            for _ in 0..verifier.params.folding_factor.at_round(r + 1) {
                let sumcheck_poly_evals: [_; 3] = verifier_state.next_scalars()?;
                let sumcheck_poly = SumcheckPolynomial::new(sumcheck_poly_evals.to_vec(), 1);
                let [folding_randomness_single] = verifier_state.challenge_scalars()?;
                sumcheck_rounds.push((sumcheck_poly, folding_randomness_single));

                if round_params.folding_pow_bits > 0. {
                    verifier_state.challenge_pow::<PS>(round_params.folding_pow_bits)?;
                }
            }

            let new_folding_randomness =
                MultilinearPoint(sumcheck_rounds.iter().map(|&(_, r)| r).rev().collect());

            rounds.push(ParsedRound {
                folding_randomness: folding_randomness.clone(),
                ood_points,
                ood_answers,
                stir_challenges_indexes,
                stir_challenges_points,
                stir_challenges_answers: stir_challenges_answers.clone(),
                combination_randomness,
                sumcheck_rounds,
                domain_gen_inv,
            });

            folding_randomness = new_folding_randomness;

            prev_root = new_root;
            domain_gen = domain_gen * domain_gen;
            exp_domain_gen =
                domain_gen.exp_u64(1 << verifier.params.folding_factor.at_round(r + 1));
            domain_gen_inv = domain_gen_inv * domain_gen_inv;
            domain_size /= 2;
        }

        let mut final_coefficients = vec![F::ZERO; 1 << verifier.params.final_sumcheck_rounds];
        verifier_state.fill_next_scalars(&mut final_coefficients)?;
        let final_coefficients = CoefficientList::new(final_coefficients);

        let fold_last = verifier
            .params
            .folding_factor
            .at_round(verifier.params.n_rounds());
        let final_randomness_indexes = get_challenge_stir_queries(
            domain_size,
            fold_last,
            verifier.params.final_queries,
            verifier_state,
        )?;
        let final_randomness_points: Vec<_> = final_randomness_indexes
            .iter()
            .map(|index| exp_domain_gen.exp_u64(*index as u64))
            .collect();

        let dimensions = vec![Dimensions {
            width: 1 << fold_last,
            height: domain_size >> fold_last,
        }];

        // Final Merkle verification
        let final_randomness_answers = if whir_proof.merkle_paths.is_empty() {
            let (commitment_randomness_answers, commitment_merkle_proof) =
                &whir_proof.commitment_merkle_paths;

            for (i, &stir_challenges_index) in final_randomness_indexes.iter().enumerate() {
                mmcs.verify_batch(
                    &prev_root,
                    &dimensions,
                    stir_challenges_index,
                    &[commitment_randomness_answers[i].clone()],
                    &commitment_merkle_proof[i],
                )
                .map_err(|_| ProofError::InvalidProof)?;
            }

            commitment_randomness_answers
                .iter()
                .map(|inner| inner.iter().map(|&f_el| f_el.into()).collect())
                .collect()
        } else {
            let (final_randomness_answers, final_merkle_proof) =
                &whir_proof.merkle_paths[whir_proof.merkle_paths.len() - 1];

            for (i, &stir_challenges_index) in final_randomness_indexes.iter().enumerate() {
                extension_mmcs
                    .verify_batch(
                        &prev_root,
                        &dimensions,
                        stir_challenges_index,
                        &[final_randomness_answers[i].clone()],
                        &final_merkle_proof[i],
                    )
                    .map_err(|_| ProofError::InvalidProof)?;
            }

            final_randomness_answers.clone()
        };

        if verifier.params.final_pow_bits > 0. {
            verifier_state.challenge_pow::<PS>(verifier.params.final_pow_bits)?;
        }

        let mut final_sumcheck_rounds = Vec::with_capacity(verifier.params.final_sumcheck_rounds);
        for _ in 0..verifier.params.final_sumcheck_rounds {
            let sumcheck_poly_evals: [_; 3] = verifier_state.next_scalars()?;
            let sumcheck_poly = SumcheckPolynomial::new(sumcheck_poly_evals.to_vec(), 1);
            let [folding_randomness_single] = verifier_state.challenge_scalars()?;
            final_sumcheck_rounds.push((sumcheck_poly, folding_randomness_single));

            if verifier.params.final_folding_pow_bits > 0. {
                verifier_state.challenge_pow::<PS>(verifier.params.final_folding_pow_bits)?;
            }
        }

        let final_sumcheck_randomness = MultilinearPoint(
            final_sumcheck_rounds
                .iter()
                .map(|&(_, r)| r)
                .rev()
                .collect(),
        );

        Ok(Self {
            initial_combination_randomness,
            initial_sumcheck_rounds: sumcheck_rounds,
            rounds,
            final_domain_gen_inv: domain_gen_inv,
            final_folding_randomness: folding_randomness,
            final_randomness_indexes,
            final_randomness_points,
            final_randomness_answers,
            final_sumcheck_rounds,
            final_sumcheck_randomness,
            final_coefficients,
            statement_values_at_random_point: whir_proof.statement_values_at_random_point.clone(),
        })
    }

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
        self.rounds
            .iter()
            .map(|r| {
                r.stir_challenges_answers
                    .iter()
                    .map(|a| CoefficientList::new(a.clone()).evaluate(&r.folding_randomness))
                    .collect()
            })
            .chain(std::iter::once(
                self.final_randomness_answers
                    .iter()
                    .map(|a| {
                        CoefficientList::new(a.clone()).evaluate(&self.final_folding_randomness)
                    })
                    .collect(),
            ))
            .collect()
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
