use std::{fmt::Debug, iter, ops::Deref};

use p3_field::{ExtensionField, Field, PrimeField64, TwoAdicField};
use p3_symmetric::{CryptographicHasher, Hash, PseudoCompressionFunction};
use serde::{Deserialize, Serialize};
use tracing::instrument;

use super::{
    committer::reader::ParsedCommitment,
    statement::{constraint::Constraint, weights::Weights},
};
use crate::{
    fiat_shamir::{
        errors::{ProofError, ProofResult},
        pow::traits::PowStrategy,
        verifier::VerifierState,
    },
    poly::multilinear::MultilinearPoint,
    whir::{Statement, parameters::WhirConfig, verifier::parsed_proof::ParsedProof},
};

pub mod parsed_proof;
pub mod parsed_round;
pub mod utils;

/// Wrapper around the WHIR verifier configuration.
///
/// This type provides a lightweight, ergonomic interface to verification methods
/// by wrapping a reference to the `WhirConfig`.
#[derive(Debug)]
pub struct Verifier<'a, EF, F, H, C, PowStrategy>(
    /// Reference to the verifier’s configuration containing all round parameters.
    pub(crate) &'a WhirConfig<EF, F, H, C, PowStrategy>,
)
where
    F: Field + TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField;

impl<'a, EF, F, H, C, PS> Verifier<'a, EF, F, H, C, PS>
where
    F: Field + TwoAdicField + PrimeField64,
    EF: ExtensionField<F> + TwoAdicField,
    PS: PowStrategy,
{
    pub const fn new(params: &'a WhirConfig<EF, F, H, C, PS>) -> Self {
        Self(params)
    }

    /// Combine multiple constraints into a single claim using random linear combination.
    ///
    /// This method draws a challenge scalar from the Fiat-Shamir transcript and uses it
    /// to generate a sequence of powers, one for each constraint. These powers serve as
    /// coefficients in a random linear combination of the constraint sums.
    ///
    /// The resulting linear combination is added to `claimed_sum`, which becomes the new
    /// target value to verify in the sumcheck protocol.
    ///
    /// # Arguments
    /// - `verifier_state`: Fiat-Shamir transcript reader.
    /// - `claimed_sum`: Mutable reference to the running sum of combined constraints.
    /// - `constraints`: List of constraints to combine.
    ///
    /// # Returns
    /// A vector of randomness values used to weight each constraint.
    pub fn combine_constraints(
        &self,
        verifier_state: &mut VerifierState<'_, EF, F>,
        claimed_sum: &mut EF,
        constraints: &[Constraint<EF>],
    ) -> ProofResult<Vec<EF>> {
        let [combination_randomness_gen] = verifier_state.challenge_scalars()?;
        let combination_randomness: Vec<_> = combination_randomness_gen
            .powers()
            .take(constraints.len())
            .collect();
        *claimed_sum += constraints
            .iter()
            .zip(&combination_randomness)
            .map(|(c, &rand)| rand * c.sum)
            .sum::<EF>();

        Ok(combination_randomness)
    }

    fn compute_w_poly<const DIGEST_ELEMS: usize>(
        &self,
        parsed_commitment: &ParsedCommitment<EF, Hash<F, u8, DIGEST_ELEMS>>,
        statement: &Statement<EF>,
        proof: &ParsedProof<EF>,
        deferred: &[EF],
    ) -> EF {
        let mut num_variables = self.mv_parameters.num_variables;

        let mut folding_randomness = MultilinearPoint(
            iter::once(&proof.final_sumcheck_randomness.0)
                .chain(iter::once(&proof.final_folding_randomness.0))
                .chain(proof.rounds.iter().rev().map(|r| &r.folding_randomness.0))
                .flatten()
                .copied()
                .collect(),
        );

        let constraints: Vec<_> = parsed_commitment
            .ood_points
            .iter()
            .zip(&parsed_commitment.ood_answers)
            .map(|(&point, &eval)| {
                let weights = Weights::evaluation(MultilinearPoint::expand_from_univariate(
                    point,
                    num_variables,
                ));
                Constraint {
                    weights,
                    sum: eval,
                    defer_evaluation: false,
                }
            })
            .chain(statement.constraints.iter().cloned())
            .collect();

        let mut deferred = deferred.iter().copied();
        let mut value: EF = constraints
            .iter()
            .zip(&proof.initial_combination_randomness)
            .map(|(constraint, randomness)| {
                if constraint.defer_evaluation {
                    deferred.next().unwrap()
                } else {
                    *randomness * constraint.weights.compute(&folding_randomness)
                }
            })
            .sum();

        for (round, round_proof) in proof.rounds.iter().enumerate() {
            num_variables -= self.folding_factor.at_round(round);
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

    #[instrument(skip_all)]
    #[allow(clippy::too_many_lines)]
    pub fn verify<const DIGEST_ELEMS: usize>(
        &self,
        verifier_state: &mut VerifierState<'_, EF, F>,
        parsed_commitment: &ParsedCommitment<EF, Hash<F, u8, DIGEST_ELEMS>>,
        statement: &Statement<EF>,
    ) -> ProofResult<()>
    where
        H: CryptographicHasher<F, [u8; DIGEST_ELEMS]> + Sync,
        C: PseudoCompressionFunction<[u8; DIGEST_ELEMS], 2> + Sync,
        [u8; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        // First, derive all Fiat-Shamir challenges
        let evaluations: Vec<_> = statement.constraints.iter().map(|c| c.sum).collect();

        let parsed = ParsedProof::from_prover_output(
            self,
            verifier_state,
            parsed_commitment,
            statement.constraints.len(),
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
        if self.final_sumcheck_rounds > 0 {
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
        let evaluation_of_v_poly = self.compute_w_poly(
            parsed_commitment,
            statement,
            &parsed,
            &parsed.deferred_weight_evaluations,
        );
        let final_value = parsed
            .final_coefficients
            .evaluate(&parsed.final_sumcheck_randomness);

        if prev_sumcheck_poly_eval != evaluation_of_v_poly * final_value {
            return Err(ProofError::InvalidProof);
        }

        Ok(())
    }
}

impl<EF, F, H, C, PS> Deref for Verifier<'_, EF, F, H, C, PS>
where
    F: Field + TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
{
    type Target = WhirConfig<EF, F, H, C, PS>;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}
