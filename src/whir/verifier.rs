use crate::{
    merkle_tree::{InnerDigest, KeccakDigest, WhirChallenger},
    parameters::FoldType,
    poly::{fold::compute_fold, multilinear::MultilinearPoint},
    whir::{parameters::WhirConfig, parsed_proof::ParsedProof},
};
use p3_baby_bear::BabyBear;
use p3_challenger::CanSample;
use p3_field::{Field, PackedValue, PrimeCharacteristicRing, TwoAdicField};
use std::iter;

use super::statement::{StatementVerifier, VerifierWeights};

#[derive(Clone)]
struct ParsedCommitment<F, D> {
    root: D,
    ood_points: Vec<F>,
    ood_answers: Vec<F>,
}

#[derive(Debug)]
pub struct Verifier<F, PowStrategy, Perm16, Perm24>
where
    F: Field + TwoAdicField,
    <F as PrimeCharacteristicRing>::PrimeSubfield: TwoAdicField,
{
    params: WhirConfig<F, PowStrategy, Perm16, Perm24>,
    two_inv: F,
}

impl<PowStrategy, Perm16, Perm24> Verifier<BabyBear, PowStrategy, Perm16, Perm24> {
    pub fn new(params: WhirConfig<BabyBear, PowStrategy, Perm16, Perm24>) -> Self {
        Self { params, two_inv: BabyBear::TWO.inverse() }
    }

    fn parse_commitment(
        &self,
        challenger: &mut WhirChallenger<BabyBear>,
    ) -> ParsedCommitment<BabyBear, KeccakDigest<BabyBear>> {
        // Read the Merkle root from the challenger
        let root = KeccakDigest(challenger.sample_array());

        // Sample OOD points if required
        let ood_points = if self.params.committment_ood_samples > 0 {
            (0..self.params.committment_ood_samples).map(|_| challenger.sample()).collect()
        } else {
            vec![]
        };

        // Sample OOD answers corresponding to the OOD points
        let ood_answers = ood_points.iter().map(|_| challenger.sample()).collect();

        ParsedCommitment { root, ood_points, ood_answers }
    }

    fn compute_w_poly(
        &self,
        parsed_commitment: &ParsedCommitment<BabyBear, KeccakDigest<BabyBear>>,
        statement: &StatementVerifier<BabyBear>,
        proof: &ParsedProof<BabyBear>,
    ) -> BabyBear {
        let mut num_variables = self.params.mv_parameters.num_variables;

        let mut folding_randomness = MultilinearPoint(
            iter::once(&proof.final_sumcheck_randomness.0)
                .chain(iter::once(&proof.final_folding_randomness.0))
                .chain(proof.rounds.iter().rev().map(|r| &r.folding_randomness.0))
                .flatten()
                .copied()
                .collect(),
        );

        let mut new_constraints = Vec::new();
        for (point, evaluation) in parsed_commitment
            .ood_points
            .clone()
            .into_iter()
            .zip(parsed_commitment.ood_answers.clone())
        {
            let weights = VerifierWeights::evaluation(MultilinearPoint::expand_from_univariate(
                point,
                num_variables,
            ));
            new_constraints.push((weights, evaluation));
        }
        let mut proof_values_iter = proof.statement_values_at_random_point.iter();
        for constraint in &statement.constraints {
            match &constraint.0 {
                VerifierWeights::Evaluation { point } => {
                    new_constraints
                        .push((VerifierWeights::evaluation(point.clone()), constraint.1));
                }
                VerifierWeights::Linear { .. } => {
                    let term = proof_values_iter
                        .next()
                        .expect("Not enough proof statement values for linear constraints");
                    new_constraints
                        .push((VerifierWeights::linear(num_variables, Some(*term)), constraint.1));
                }
            }
        }
        let mut value = new_constraints
            .iter()
            .zip(&proof.initial_combination_randomness)
            .map(|((weight, _), randomness)| *randomness * weight.compute(&folding_randomness))
            .sum();

        for (round, round_proof) in proof.rounds.iter().enumerate() {
            num_variables -= self.params.folding_factor.at_round(round);
            folding_randomness = MultilinearPoint(folding_randomness.0[..num_variables].to_vec());

            let ood_points = &round_proof.ood_points;
            let stir_challenges_points = &round_proof.stir_challenges_points;
            let stir_challenges =
                ood_points.iter().chain(stir_challenges_points).copied().map(|univariate| {
                    MultilinearPoint::expand_from_univariate(univariate, num_variables)
                    // TODO:
                    // Maybe refactor outside
                });

            let sum_of_claims = stir_challenges
                .into_iter()
                .map(|point| point.eq_poly_outside(&folding_randomness))
                .zip(&round_proof.combination_randomness)
                .map(|(point, &rand)| point * rand)
                .sum();

            value += sum_of_claims;
        }

        value
    }

    fn compute_folds(&self, parsed: &ParsedProof<BabyBear>) -> Vec<Vec<BabyBear>> {
        match self.params.fold_optimisation {
            FoldType::Naive => self.compute_folds_full(parsed),
            FoldType::ProverHelps => parsed.compute_folds_helped(),
        }
    }

    fn compute_folds_full(&self, parsed: &ParsedProof<BabyBear>) -> Vec<Vec<BabyBear>> {
        let mut domain_size = self.params.starting_domain.backing_domain.size();

        let mut result = Vec::new();

        for (round_index, round) in parsed.rounds.iter().enumerate() {
            let coset_domain_size = 1 << self.params.folding_factor.at_round(round_index);
            // This is such that coset_generator^coset_domain_size = F::ONE
            //let _coset_generator = domain_gen.pow(&[(domain_size / coset_domain_size) as u64]);
            let coset_generator_inv =
                round.domain_gen_inv.exp_u64((domain_size / coset_domain_size) as u64);

            let evaluations: Vec<_> = round
                .stir_challenges_indexes
                .iter()
                .zip(&round.stir_challenges_answers)
                .map(|(index, answers)| {
                    // The coset is w^index * <w_coset_generator>
                    //let _coset_offset = domain_gen.pow(&[*index as u64]);
                    let coset_offset_inv = round.domain_gen_inv.exp_u64(*index as u64);

                    compute_fold(
                        answers,
                        &round.folding_randomness.0,
                        coset_offset_inv,
                        coset_generator_inv,
                        self.two_inv,
                        self.params.folding_factor.at_round(round_index),
                    )
                })
                .collect();
            result.push(evaluations);
            domain_size /= 2;
        }

        let coset_domain_size = 1 << self.params.folding_factor.at_round(parsed.rounds.len());
        let domain_gen_inv = parsed.final_domain_gen_inv;

        // Final round
        let coset_generator_inv = domain_gen_inv.exp_u64((domain_size / coset_domain_size) as u64);
        let evaluations: Vec<_> = parsed
            .final_randomness_indexes
            .iter()
            .zip(&parsed.final_randomness_answers)
            .map(|(index, answers)| {
                // The coset is w^index * <w_coset_generator>
                //let _coset_offset = domain_gen.pow(&[*index as u64]);
                let coset_offset_inv = domain_gen_inv.exp_u64(*index as u64);

                compute_fold(
                    answers,
                    &parsed.final_folding_randomness.0,
                    coset_offset_inv,
                    coset_generator_inv,
                    self.two_inv,
                    self.params.folding_factor.at_round(parsed.rounds.len()),
                )
            })
            .collect();
        result.push(evaluations);

        result
    }
}
