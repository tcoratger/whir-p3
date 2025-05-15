use std::{fmt::Debug, iter};

use p3_commit::{ExtensionMmcs, Mmcs};
use p3_field::{ExtensionField, Field, PrimeField64, TwoAdicField};
use p3_matrix::Dimensions;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CryptographicHasher, Hash, PseudoCompressionFunction};
use serde::{Deserialize, Serialize};

use super::{
    WhirProof,
    committer::reader::ParsedCommitment,
    parsed_proof::ParsedRound,
    statement::{StatementVerifier, VerifierWeights},
    utils::get_challenge_stir_queries,
};
use crate::{
    fiat_shamir::{
        errors::{ProofError, ProofResult},
        pow::traits::PowStrategy,
        verifier::VerifierState,
    },
    poly::{coeffs::CoefficientList, multilinear::MultilinearPoint},
    sumcheck::sumcheck_polynomial::SumcheckPolynomial,
    utils::expand_randomness,
    whir::{parameters::WhirConfig, parsed_proof::ParsedProof, utils::K_SKIP_SUMCHECK},
};

#[derive(Debug)]
pub struct Verifier<'a, EF, F, H, C, PowStrategy>
where
    F: Field + TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
{
    params: &'a WhirConfig<EF, F, H, C, PowStrategy>,
}

impl<'a, EF, F, H, C, PS> Verifier<'a, EF, F, H, C, PS>
where
    F: Field + TwoAdicField + PrimeField64,
    EF: ExtensionField<F> + TwoAdicField,
    PS: PowStrategy,
{
    pub const fn new(params: &'a WhirConfig<EF, F, H, C, PS>) -> Self {
        Self { params }
    }

    #[allow(clippy::too_many_lines)]
    fn parse_proof<const DIGEST_ELEMS: usize>(
        &self,
        verifier_state: &mut VerifierState<'_, EF, F>,
        parsed_commitment: &ParsedCommitment<EF, Hash<F, u8, DIGEST_ELEMS>>,
        statement_points_len: usize,
        whir_proof: &WhirProof<F, EF, DIGEST_ELEMS>,
    ) -> ProofResult<ParsedProof<EF>>
    where
        H: CryptographicHasher<F, [u8; DIGEST_ELEMS]> + Sync,
        C: PseudoCompressionFunction<[u8; DIGEST_ELEMS], 2> + Sync,
        [u8; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        let mmcs = ExtensionMmcs::new(MerkleTreeMmcs::new(
            self.params.merkle_hash.clone(),
            self.params.merkle_compress.clone(),
        ));

        let commitment_mmcs = MerkleTreeMmcs::new(
            self.params.merkle_hash.clone(),
            self.params.merkle_compress.clone(),
        );

        let mut sumcheck_rounds = Vec::new();
        let mut folding_randomness;
        let initial_combination_randomness;

        if self.params.initial_statement {
            // Derive combination randomness and first sumcheck polynomial
            let [combination_randomness_gen] = verifier_state.challenge_scalars()?;
            initial_combination_randomness = expand_randomness(
                combination_randomness_gen,
                parsed_commitment.ood_points.len() + statement_points_len,
            );

            // Initial sumcheck
            sumcheck_rounds.reserve_exact(self.params.folding_factor.at_round(0));

            let mut is_univariate_skip = false;
            if self.params.folding_factor.at_round(0) >= 2 {
                let sumcheck_poly_evals: [_; 8] = verifier_state.next_scalars()?;
                let sumcheck_poly = SumcheckPolynomial::new(sumcheck_poly_evals.to_vec(), 1);
                let [folding_randomness_single] = verifier_state.challenge_scalars()?;
                sumcheck_rounds.push((sumcheck_poly, folding_randomness_single));
                is_univariate_skip = true;
            }

            let start = if is_univariate_skip {
                K_SKIP_SUMCHECK
            } else {
                0
            };
            for _ in start..self.params.folding_factor.at_round(0) {
                let sumcheck_poly_evals: [_; 3] = verifier_state.next_scalars()?;
                let sumcheck_poly = SumcheckPolynomial::new(sumcheck_poly_evals.to_vec(), 1);
                let [folding_randomness_single] = verifier_state.challenge_scalars()?;
                sumcheck_rounds.push((sumcheck_poly, folding_randomness_single));

                if self.params.starting_folding_pow_bits > 0. {
                    verifier_state.challenge_pow::<PS>(self.params.starting_folding_pow_bits)?;
                }
            }

            folding_randomness =
                MultilinearPoint(sumcheck_rounds.iter().map(|&(_, r)| r).rev().collect());
        } else {
            assert_eq!(parsed_commitment.ood_points.len(), 0);
            assert_eq!(statement_points_len, 0);

            initial_combination_randomness = vec![EF::ONE];

            let mut folding_randomness_vec = vec![EF::ZERO; self.params.folding_factor.at_round(0)];
            verifier_state.fill_challenge_scalars(&mut folding_randomness_vec)?;
            folding_randomness = MultilinearPoint(folding_randomness_vec);

            // PoW
            if self.params.starting_folding_pow_bits > 0. {
                verifier_state.challenge_pow::<PS>(self.params.starting_folding_pow_bits)?;
            }
        }

        let mut prev_root = parsed_commitment.root;
        let mut domain_gen = self.params.starting_domain.backing_domain.group_gen();
        let mut exp_domain_gen = domain_gen.exp_u64(1 << self.params.folding_factor.at_round(0));
        let mut domain_gen_inv = self.params.starting_domain.backing_domain.group_gen_inv();
        let mut domain_size = self.params.starting_domain.size();
        let mut rounds = vec![];

        for r in 0..self.params.n_rounds() {
            let round_params = &self.params.round_parameters[r];
            let fold_r = self.params.folding_factor.at_round(r);

            let new_root = verifier_state.read_digest()?;

            let mut ood_points = vec![EF::ZERO; round_params.ood_samples];
            let mut ood_answers = vec![EF::ZERO; round_params.ood_samples];
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

            let stir_challenges_points = stir_challenges_indexes
                .iter()
                .map(|index| exp_domain_gen.exp_u64(*index as u64))
                .collect();

            // Verify Merkle openings using `verify_batch`
            let dimensions = vec![Dimensions {
                height: domain_size >> fold_r,
                width: 1 << fold_r,
            }];

            let mut stir_challenges_answers = Vec::new();

            for (i, &stir_challenges_index) in stir_challenges_indexes.iter().enumerate() {
                if r == 0 {
                    let (answers, merkle_proof) = &whir_proof.commitment_merkle_paths;
                    let indexed_answers: Vec<F> =
                        answers[i].iter().map(|v| v.as_base().unwrap()).collect();

                    commitment_mmcs
                        .verify_batch(
                            &prev_root,
                            &dimensions,
                            stir_challenges_index,
                            &[indexed_answers],
                            &merkle_proof[i],
                        )
                        .map_err(|_| ProofError::InvalidProof)?;
                    stir_challenges_answers = answers
                        .iter()
                        .map(|inner| inner.iter().map(|&f_el| EF::from(f_el)).collect())
                        .collect();
                } else {
                    let (answers, merkle_proof) = &whir_proof.merkle_paths[r - 1];
                    mmcs.verify_batch(
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
                Vec::with_capacity(self.params.folding_factor.at_round(r + 1));

            for _ in 0..self.params.folding_factor.at_round(r + 1) {
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
            exp_domain_gen = domain_gen.exp_u64(1 << self.params.folding_factor.at_round(r + 1));
            domain_gen_inv = domain_gen_inv * domain_gen_inv;
            domain_size /= 2;
        }

        let mut final_coefficients = vec![EF::ZERO; 1 << self.params.final_sumcheck_rounds];
        verifier_state.fill_next_scalars(&mut final_coefficients)?;
        let final_coefficients = CoefficientList::new(final_coefficients);

        let fold_last = self.params.folding_factor.at_round(self.params.n_rounds());
        let final_randomness_indexes = get_challenge_stir_queries(
            domain_size,
            fold_last,
            self.params.final_queries,
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
        let final_randomness_answers: Vec<Vec<EF>> = if whir_proof.merkle_paths.is_empty() {
            let (commitment_randomness_answers, commitment_merkle_proof) =
                &whir_proof.commitment_merkle_paths;

            for (i, &stir_challenges_index) in final_randomness_indexes.iter().enumerate() {
                commitment_mmcs
                    .verify_batch(
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
                .map(|inner| inner.iter().map(|&f_el| EF::from(f_el)).collect())
                .collect()
        } else {
            let (final_randomness_answers, final_merkle_proof) =
                &whir_proof.merkle_paths[whir_proof.merkle_paths.len() - 1];

            for (i, &stir_challenges_index) in final_randomness_indexes.iter().enumerate() {
                mmcs.verify_batch(
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

        if self.params.final_pow_bits > 0. {
            verifier_state.challenge_pow::<PS>(self.params.final_pow_bits)?;
        }

        let mut final_sumcheck_rounds = Vec::with_capacity(self.params.final_sumcheck_rounds);
        for _ in 0..self.params.final_sumcheck_rounds {
            let sumcheck_poly_evals: [_; 3] = verifier_state.next_scalars()?;
            let sumcheck_poly = SumcheckPolynomial::new(sumcheck_poly_evals.to_vec(), 1);
            let [folding_randomness_single] = verifier_state.challenge_scalars()?;
            final_sumcheck_rounds.push((sumcheck_poly, folding_randomness_single));

            if self.params.final_folding_pow_bits > 0. {
                verifier_state.challenge_pow::<PS>(self.params.final_folding_pow_bits)?;
            }
        }

        let final_sumcheck_randomness = MultilinearPoint(
            final_sumcheck_rounds
                .iter()
                .map(|&(_, r)| r)
                .rev()
                .collect(),
        );

        Ok(ParsedProof {
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

    fn compute_w_poly<const DIGEST_ELEMS: usize>(
        &self,
        parsed_commitment: &ParsedCommitment<EF, Hash<F, u8, DIGEST_ELEMS>>,
        statement: &StatementVerifier<EF>,
        proof: &ParsedProof<EF>,
    ) -> EF {
        let mut num_variables = self.params.mv_parameters.num_variables;

        let mut folding_randomness = MultilinearPoint(
            iter::once(&proof.final_sumcheck_randomness.0)
                .chain(iter::once(&proof.final_folding_randomness.0))
                .chain(proof.rounds.iter().rev().map(|r| &r.folding_randomness.0))
                .flatten()
                .copied()
                .collect(),
        );

        let mut new_constraints: Vec<_> = parsed_commitment
            .ood_points
            .iter()
            .zip(&parsed_commitment.ood_answers)
            .map(|(&point, &eval)| {
                let weights = VerifierWeights::evaluation(
                    MultilinearPoint::expand_from_univariate(point, num_variables),
                );
                (weights, eval)
            })
            .collect();

        let mut proof_values_iter = proof.statement_values_at_random_point.iter();
        for (weights, expected_result) in &statement.constraints {
            match weights {
                VerifierWeights::Evaluation { point } => {
                    new_constraints
                        .push((VerifierWeights::evaluation(point.clone()), *expected_result));
                }
                VerifierWeights::Linear { .. } => {
                    let term = proof_values_iter
                        .next()
                        .expect("Not enough proof statement values for linear constraints");
                    new_constraints.push((
                        VerifierWeights::linear(num_variables, Some(*term)),
                        *expected_result,
                    ));
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

            let stir_challenges = round_proof
                .ood_points
                .iter()
                .chain(&round_proof.stir_challenges_points)
                .map(|&univariate| {
                    MultilinearPoint::expand_from_univariate(univariate, num_variables)
                    // TODO:
                    // Maybe refactor outside
                });

            let sum_of_claims: EF = stir_challenges
                .zip(&round_proof.combination_randomness)
                .map(|(pt, &rand)| pt.eq_poly_outside(&folding_randomness) * rand)
                .sum();

            value += sum_of_claims;
        }

        value
    }

    #[allow(clippy::too_many_lines)]
    pub fn verify<const DIGEST_ELEMS: usize>(
        &self,
        verifier_state: &mut VerifierState<'_, EF, F>,
        parsed_commitment: &ParsedCommitment<EF, Hash<F, u8, DIGEST_ELEMS>>,
        statement: &StatementVerifier<EF>,
        whir_proof: &WhirProof<F, EF, DIGEST_ELEMS>,
    ) -> ProofResult<()>
    where
        H: CryptographicHasher<F, [u8; DIGEST_ELEMS]> + Sync,
        C: PseudoCompressionFunction<[u8; DIGEST_ELEMS], 2> + Sync,
        [u8; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        // First, derive all Fiat-Shamir challenges
        let evaluations: Vec<_> = statement
            .constraints
            .iter()
            .map(|(_, eval)| *eval)
            .collect();

        let parsed = self.parse_proof(
            verifier_state,
            parsed_commitment,
            statement.constraints.len(),
            whir_proof,
        )?;

        let computed_folds = self
            .params
            .fold_optimisation
            .stir_evaluations_verifier(&parsed, self.params);

        let mut prev_sumcheck = None;

        // Initial sumcheck verification
        if let Some((poly, randomness)) = parsed.initial_sumcheck_rounds.first().cloned() {
            if poly.sum_over_boolean_hypercube()
                != parsed_commitment
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

            let mut current = (poly, randomness);

            // Check the rest of the rounds
            for (next_poly, next_rand) in &parsed.initial_sumcheck_rounds[1..] {
                if next_poly.sum_over_boolean_hypercube()
                    != current.0.evaluate_at_point(&current.1.into())
                {
                    println!("Problem with some sumchecks");
                    return Err(ProofError::InvalidProof);
                }
                current = (next_poly.clone(), *next_rand);
            }

            prev_sumcheck = Some(current);
        }

        // Sumcheck rounds
        for (round, folds) in parsed.rounds.iter().zip(&computed_folds) {
            let (sumcheck_poly, new_randomness) = &round.sumcheck_rounds[0];

            let values = round
                .ood_answers
                .iter()
                .copied()
                .chain(folds.iter().copied());

            let prev_eval = prev_sumcheck
                .as_ref()
                .map_or(EF::ZERO, |(p, r)| p.evaluate_at_point(&(*r).into()));

            let claimed_sum = prev_eval
                + values
                    .zip(&round.combination_randomness)
                    .map(|(val, &rand)| val * rand)
                    .sum::<EF>();

            if sumcheck_poly.sum_over_boolean_hypercube() != claimed_sum {
                return Err(ProofError::InvalidProof);
            }

            prev_sumcheck = Some((sumcheck_poly.clone(), *new_randomness));

            // Check the rest of the round
            for (sumcheck_poly, new_randomness) in &round.sumcheck_rounds[1..] {
                let (prev_poly, randomness) = prev_sumcheck.unwrap();
                if sumcheck_poly.sum_over_boolean_hypercube()
                    != prev_poly.evaluate_at_point(&randomness.into())
                {
                    return Err(ProofError::InvalidProof);
                }
                prev_sumcheck = Some((sumcheck_poly.clone(), *new_randomness));
            }
        }

        // Check the foldings computed from the proof match the evaluations of the polynomial
        let final_folds = &computed_folds.last().expect("final folds missing");
        let final_evaluations = parsed
            .final_coefficients
            .evaluate_at_univariate(&parsed.final_randomness_points);
        if !final_folds
            .iter()
            .zip(final_evaluations)
            .all(|(&fold, eval)| fold == eval)
        {
            return Err(ProofError::InvalidProof);
        }

        // Check the final sumchecks
        if self.params.final_sumcheck_rounds > 0 {
            let claimed_sum = prev_sumcheck
                .as_ref()
                .map_or(EF::ZERO, |(p, r)| p.evaluate_at_point(&(*r).into()));

            let (sumcheck_poly, new_randomness) = &parsed.final_sumcheck_rounds[0];

            if sumcheck_poly.sum_over_boolean_hypercube() != claimed_sum {
                return Err(ProofError::InvalidProof);
            }

            prev_sumcheck = Some((sumcheck_poly.clone(), *new_randomness));

            // Check the rest of the round
            for (sumcheck_poly, new_randomness) in &parsed.final_sumcheck_rounds[1..] {
                let (prev_poly, randomness) = prev_sumcheck.unwrap();
                if sumcheck_poly.sum_over_boolean_hypercube()
                    != prev_poly.evaluate_at_point(&randomness.into())
                {
                    return Err(ProofError::InvalidProof);
                }
                prev_sumcheck = Some((sumcheck_poly.clone(), *new_randomness));
            }
        }

        // Final v · w Check
        let prev_sumcheck_poly_eval = prev_sumcheck.map_or(EF::ZERO, |(poly, rand)| {
            poly.evaluate_at_point(&rand.into())
        });

        // Check the final sumcheck evaluation
        let evaluation_of_v_poly = self.compute_w_poly(parsed_commitment, statement, &parsed);
        let final_value = parsed
            .final_coefficients
            .evaluate(&parsed.final_sumcheck_randomness);

        if prev_sumcheck_poly_eval != evaluation_of_v_poly * final_value {
            return Err(ProofError::InvalidProof);
        }

        Ok(())
    }
}
