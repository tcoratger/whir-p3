use std::{fmt::Debug, iter};

use p3_field::{ExtensionField, Field, PrimeField64, TwoAdicField};
use p3_symmetric::{CryptographicHasher, Hash, PseudoCompressionFunction};
use serde::{Deserialize, Serialize};

use super::{
    committer::reader::ParsedCommitment,
    statement::{StatementVerifier, VerifierWeights},
};
use crate::{
    fiat_shamir::{
        errors::{ProofError, ProofResult},
        pow::traits::PowStrategy,
        verifier::VerifierState,
    },
    poly::multilinear::MultilinearPoint,
    whir::{parameters::WhirConfig, prover::proof::WhirProof, verifier::parsed_proof::ParsedProof},
};

pub mod parsed_proof;
pub mod parsed_round;
pub mod utils;

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

        let parsed = ParsedProof::from_prover_output(
            self,
            verifier_state,
            parsed_commitment,
            statement.constraints.len(),
            whir_proof,
        )?;

        let computed_folds = parsed.compute_folds_helped();

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
