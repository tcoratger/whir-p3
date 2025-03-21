use super::ProofResult;
use crate::{
    merkle_tree::{KeccakDigest, WhirChallenger},
    parameters::FoldType,
    poly::{coeffs::CoefficientList, fold::compute_fold, multilinear::MultilinearPoint},
    sumcheck::sumcheck_polynomial::SumcheckPolynomial,
    utils::expand_randomness,
    whir::{parameters::WhirConfig, parsed_proof::ParsedProof},
};
use p3_baby_bear::BabyBear;
use p3_challenger::{CanSample, GrindingChallenger};
use p3_field::{Field, PrimeCharacteristicRing, TwoAdicField};
use std::iter;

use super::{
    ProofError, WhirProof,
    fs_utils::get_challenge_stir_queries,
    parsed_proof::ParsedRound,
    statement::{StatementVerifier, VerifierWeights},
};

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

    fn parse_proof(
        &self,
        challenger: &mut WhirChallenger<BabyBear>,
        parsed_commitment: &ParsedCommitment<BabyBear, KeccakDigest<BabyBear>>,
        statement_points_len: usize,
        whir_proof: &WhirProof<BabyBear>,
    ) -> ParsedProof<BabyBear> {
        let mut sumcheck_rounds = Vec::new();
        let mut folding_randomness: MultilinearPoint<BabyBear>;
        let initial_combination_randomness;

        if self.params.initial_statement {
            // Sample combination randomness and first sumcheck polynomial
            let combination_randomness_gen: BabyBear = challenger.sample();
            initial_combination_randomness = expand_randomness(
                combination_randomness_gen,
                parsed_commitment.ood_points.len() + statement_points_len,
            );

            // Initial sumcheck
            sumcheck_rounds.reserve_exact(self.params.folding_factor.at_round(0));
            for _ in 0..self.params.folding_factor.at_round(0) {
                let sumcheck_poly_evals: [BabyBear; 3] = challenger.sample_array();
                let sumcheck_poly = SumcheckPolynomial::new(sumcheck_poly_evals.to_vec(), 1);
                let folding_randomness_single = challenger.sample();
                sumcheck_rounds.push((sumcheck_poly, folding_randomness_single));

                if self.params.starting_folding_pow_bits > 0. {
                    challenger.grind(self.params.starting_folding_pow_bits as usize);
                }
            }

            folding_randomness =
                MultilinearPoint(sumcheck_rounds.iter().map(|&(_, r)| r).rev().collect());
        } else {
            assert_eq!(parsed_commitment.ood_points.len(), 0);
            assert_eq!(statement_points_len, 0);

            initial_combination_randomness = vec![BabyBear::ONE];

            let folding_randomness_vec: Vec<BabyBear> =
                challenger.sample_vec(self.params.folding_factor.at_round(0));
            folding_randomness = MultilinearPoint(folding_randomness_vec);

            // PoW
            if self.params.starting_folding_pow_bits > 0. {
                challenger.grind(self.params.starting_folding_pow_bits as usize);
            }
        }

        let mut prev_root = parsed_commitment.root.clone();
        let mut domain_gen = self.params.starting_domain.backing_domain.group_gen();
        let mut exp_domain_gen = domain_gen.exp_u64(1 << self.params.folding_factor.at_round(0));
        let mut domain_gen_inv = self.params.starting_domain.backing_domain.group_gen_inv();
        let mut domain_size = self.params.starting_domain.size();
        let mut rounds = vec![];

        for r in 0..self.params.n_rounds() {
            // let (merkle_proof, answers) = &whir_proof.merkle_paths[r];
            let round_params = &self.params.round_parameters[r];

            let new_root = KeccakDigest::<BabyBear>(challenger.sample_array());

            let ood_points: Vec<BabyBear> = challenger.sample_vec(round_params.ood_samples);
            let ood_answers: Vec<BabyBear> = challenger.sample_vec(round_params.ood_samples);

            let stir_challenges_indexes = get_challenge_stir_queries::<BabyBear, _>(
                domain_size,
                self.params.folding_factor.at_round(r),
                round_params.num_queries,
                challenger,
            );

            let stir_challenges_points: Vec<_> = stir_challenges_indexes
                .iter()
                .map(|index| exp_domain_gen.exp_u64(*index as u64))
                .collect();

            // if !merkle_proof
            //     .verify(
            //         &self.params.leaf_hash_params,
            //         &self.params.two_to_one_params,
            //         &prev_root,
            //         answers.iter().map(|a| a.as_ref()),
            //     )
            //     .unwrap() ||
            //     merkle_proof.leaf_indexes != stir_challenges_indexes
            // {
            //     return Err(ProofError::InvalidProof);
            // }

            if round_params.pow_bits > 0. {
                challenger.grind(round_params.pow_bits as usize);
            }

            let combination_randomness_gen: BabyBear = challenger.sample();
            let combination_randomness = expand_randomness(
                combination_randomness_gen,
                stir_challenges_indexes.len() + round_params.ood_samples,
            );

            let mut sumcheck_rounds =
                Vec::with_capacity(self.params.folding_factor.at_round(r + 1));

            for _ in 0..self.params.folding_factor.at_round(r + 1) {
                let sumcheck_poly_evals: [BabyBear; 3] = challenger.sample_array();
                let sumcheck_poly = SumcheckPolynomial::new(sumcheck_poly_evals.to_vec(), 1);
                let folding_randomness_single: BabyBear = challenger.sample();
                sumcheck_rounds.push((sumcheck_poly, folding_randomness_single));

                if round_params.folding_pow_bits > 0. {
                    challenger.grind(round_params.folding_pow_bits as usize);
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
                stir_challenges_answers: vec![], // TODO answers.clone(),
                combination_randomness,
                sumcheck_rounds,
                domain_gen_inv,
            });

            folding_randomness = new_folding_randomness;

            prev_root = new_root.clone();
            domain_gen = domain_gen * domain_gen;
            exp_domain_gen = domain_gen.exp_u64(1 << self.params.folding_factor.at_round(r + 1));
            domain_gen_inv = domain_gen_inv * domain_gen_inv;
            domain_size /= 2;
        }

        let final_coefficients: Vec<BabyBear> =
            challenger.sample_vec(self.params.final_sumcheck_rounds);
        let final_coefficients = CoefficientList::new(final_coefficients);

        let final_randomness_indexes = get_challenge_stir_queries::<BabyBear, _>(
            domain_size,
            self.params.folding_factor.at_round(self.params.n_rounds()),
            self.params.final_queries,
            challenger,
        );
        let final_randomness_points: Vec<_> = final_randomness_indexes
            .iter()
            .map(|index| exp_domain_gen.exp_u64(*index as u64))
            .collect();

        // let (final_merkle_proof, final_randomness_answers) =
        //     &whir_proof.merkle_paths[whir_proof.merkle_paths.len() - 1];
        // if !final_merkle_proof
        //     .verify(
        //         &self.params.leaf_hash_params,
        //         &self.params.two_to_one_params,
        //         &prev_root,
        //         final_randomness_answers.iter().map(|a| a.as_ref()),
        //     )
        //     .unwrap()
        //     || final_merkle_proof.leaf_indexes != final_randomness_indexes
        // {
        //     return Err(ProofError::InvalidProof);
        // }

        if self.params.final_pow_bits > 0. {
            challenger.grind(self.params.final_pow_bits as usize);
        }

        let mut final_sumcheck_rounds = Vec::with_capacity(self.params.final_sumcheck_rounds);
        for _ in 0..self.params.final_sumcheck_rounds {
            let sumcheck_poly_evals: [BabyBear; 3] = challenger.sample_array();
            let sumcheck_poly = SumcheckPolynomial::new(sumcheck_poly_evals.to_vec(), 1);
            let folding_randomness_single: BabyBear = challenger.sample();
            final_sumcheck_rounds.push((sumcheck_poly, folding_randomness_single));

            if self.params.final_folding_pow_bits > 0. {
                challenger.grind(self.params.final_folding_pow_bits as usize);
            }
        }

        let final_sumcheck_randomness =
            MultilinearPoint(final_sumcheck_rounds.iter().map(|&(_, r)| r).rev().collect());

        ParsedProof {
            initial_combination_randomness,
            initial_sumcheck_rounds: sumcheck_rounds,
            rounds,
            final_domain_gen_inv: domain_gen_inv,
            final_folding_randomness: folding_randomness,
            final_randomness_indexes,
            final_randomness_points,
            final_randomness_answers: vec![], // TODO final_randomness_answers.clone(),
            final_sumcheck_rounds,
            final_sumcheck_randomness,
            final_coefficients,
            statement_values_at_random_point: whir_proof.statement_values_at_random_point.clone(),
        }
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

    pub fn verify(
        &self,
        challenger: &mut WhirChallenger<BabyBear>,
        statement: &StatementVerifier<BabyBear>,
        whir_proof: &WhirProof<BabyBear>,
    ) -> ProofResult<()> {
        // First, derive all Fiat-Shamir challenges
        let parsed_commitment = self.parse_commitment(challenger);
        let evaluations: Vec<_> = statement.constraints.iter().map(|(_, eval)| *eval).collect();

        let parsed = self.parse_proof(
            challenger,
            &parsed_commitment,
            statement.constraints.len(),
            whir_proof,
        );

        let computed_folds = self.compute_folds(&parsed);
        let mut prev = None;
        if let Some(round) = parsed.initial_sumcheck_rounds.first() {
            // Check the first polynomial
            let (mut prev_poly, mut randomness) = round.clone();
            if prev_poly.sum_over_boolean_hypercube() !=
                parsed_commitment
                    .ood_answers
                    .iter()
                    .copied()
                    .chain(evaluations)
                    .zip(&parsed.initial_combination_randomness)
                    .map(|(ans, &rand)| ans * rand)
                    .sum()
            {
                println!("Initial sumcheck failed");
                return Err(ProofError::InvalidProof);
            }

            // Check the rest of the rounds
            for (sumcheck_poly, new_randomness) in &parsed.initial_sumcheck_rounds[1..] {
                if sumcheck_poly.sum_over_boolean_hypercube() !=
                    prev_poly.evaluate_at_point(&randomness.into())
                {
                    return Err(ProofError::InvalidProof);
                }
                prev_poly = sumcheck_poly.clone();
                randomness = *new_randomness;
            }

            prev = Some((prev_poly, randomness));
        }

        for (round, folds) in parsed.rounds.iter().zip(&computed_folds) {
            let (sumcheck_poly, new_randomness) = &round.sumcheck_rounds[0].clone();

            let values = round.ood_answers.iter().copied().chain(folds.clone());

            let prev_eval = if let Some((prev_poly, randomness)) = prev {
                prev_poly.evaluate_at_point(&randomness.into())
            } else {
                BabyBear::ZERO
            };
            let claimed_sum = prev_eval +
                values
                    .zip(&round.combination_randomness)
                    .map(|(val, &rand)| val * rand)
                    .sum::<BabyBear>();

            if sumcheck_poly.sum_over_boolean_hypercube() != claimed_sum {
                return Err(ProofError::InvalidProof);
            }

            prev = Some((sumcheck_poly.clone(), *new_randomness));

            // Check the rest of the round
            for (sumcheck_poly, new_randomness) in &round.sumcheck_rounds[1..] {
                let (prev_poly, randomness) = prev.unwrap();
                if sumcheck_poly.sum_over_boolean_hypercube() !=
                    prev_poly.evaluate_at_point(&randomness.into())
                {
                    return Err(ProofError::InvalidProof);
                }
                prev = Some((sumcheck_poly.clone(), *new_randomness));
            }
        }

        // Check the foldings computed from the proof match the evaluations of the polynomial
        let final_folds = &computed_folds[computed_folds.len() - 1];
        let final_evaluations =
            parsed.final_coefficients.evaluate_at_univariate(&parsed.final_randomness_points);
        if !final_folds.iter().zip(final_evaluations).all(|(&fold, eval)| fold == eval) {
            return Err(ProofError::InvalidProof);
        }

        // Check the final sumchecks
        if self.params.final_sumcheck_rounds > 0 {
            let prev_sumcheck_poly_eval = if let Some((prev_poly, randomness)) = prev {
                prev_poly.evaluate_at_point(&randomness.into())
            } else {
                BabyBear::ZERO
            };
            let (sumcheck_poly, new_randomness) = &parsed.final_sumcheck_rounds[0].clone();
            let claimed_sum = prev_sumcheck_poly_eval;

            if sumcheck_poly.sum_over_boolean_hypercube() != claimed_sum {
                return Err(ProofError::InvalidProof);
            }

            prev = Some((sumcheck_poly.clone(), *new_randomness));

            // Check the rest of the round
            for (sumcheck_poly, new_randomness) in &parsed.final_sumcheck_rounds[1..] {
                let (prev_poly, randomness) = prev.unwrap();
                if sumcheck_poly.sum_over_boolean_hypercube() !=
                    prev_poly.evaluate_at_point(&randomness.into())
                {
                    return Err(ProofError::InvalidProof);
                }
                prev = Some((sumcheck_poly.clone(), *new_randomness));
            }
        }

        let prev_sumcheck_poly_eval = if let Some((prev_poly, randomness)) = prev {
            prev_poly.evaluate_at_point(&randomness.into())
        } else {
            BabyBear::ZERO
        };

        // Check the final sumcheck evaluation
        let evaluation_of_v_poly = self.compute_w_poly(&parsed_commitment, statement, &parsed);
        let final_value = parsed.final_coefficients.evaluate(&parsed.final_sumcheck_randomness);

        if prev_sumcheck_poly_eval != evaluation_of_v_poly * final_value {
            return Err(ProofError::InvalidProof);
        }

        Ok(())
    }
}
