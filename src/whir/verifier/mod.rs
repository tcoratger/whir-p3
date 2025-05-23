use std::{fmt::Debug, ops::Deref};

use p3_commit::{BatchOpeningRef, ExtensionMmcs, Mmcs};
use p3_field::{ExtensionField, Field, PrimeField64, TwoAdicField};
use p3_matrix::Dimensions;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CryptographicHasher, Hash, PseudoCompressionFunction};
use serde::{Deserialize, Serialize};
use tracing::instrument;
use utils::read_sumcheck_rounds;

use super::{
    committer::reader::ParsedCommitment,
    prover::{Leafs, Proof},
    statement::{constraint::Constraint, weights::Weights},
    utils::get_challenge_stir_queries,
};
use crate::{
    fiat_shamir::{
        errors::{ProofError, ProofResult},
        pow::traits::PowStrategy,
        verifier::VerifierState,
    },
    poly::{coeffs::CoefficientList, multilinear::MultilinearPoint},
    whir::{Statement, parameters::WhirConfig},
};

pub mod parsed_proof;
pub mod parsed_round;
pub mod utils;

// TODO: Merge these into RoundConfig
#[derive(Debug)]
pub struct StirChallengParams<F> {
    round_index: usize,
    domain_size: usize,
    num_variables: usize,
    folding_factor: usize,
    num_queries: usize,
    pow_bits: f64,
    domain_gen: F,
    domain_gen_inv: F,
    exp_domain_gen: F,
}

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

    #[instrument(skip_all)]
    #[allow(clippy::too_many_lines)]
    pub fn verify<const DIGEST_ELEMS: usize>(
        &self,
        verifier_state: &mut VerifierState<'_, EF, F>,
        parsed_commitment: &ParsedCommitment<EF, Hash<F, u8, DIGEST_ELEMS>>,
        statement: &Statement<EF>,
    ) -> ProofResult<(MultilinearPoint<EF>, Vec<EF>)>
    where
        H: CryptographicHasher<F, [u8; DIGEST_ELEMS]> + Sync,
        C: PseudoCompressionFunction<[u8; DIGEST_ELEMS], 2> + Sync,
        [u8; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        // Proof agnostic round parameters
        // TODO: Move to RoundConfig
        let mut params = {
            let domain_gen = self.starting_domain.backing_domain.group_gen();
            StirChallengParams {
                round_index: 0,
                domain_size: self.starting_domain.size(),
                num_variables: self.mv_parameters.num_variables - self.folding_factor.at_round(0),
                num_queries: 0,
                folding_factor: 0,
                pow_bits: 0.,
                domain_gen,
                domain_gen_inv: self.starting_domain.backing_domain.group_gen_inv(),
                exp_domain_gen: domain_gen.exp_u64(1 << self.folding_factor.at_round(0)),
            }
        };

        // During the rounds we collect constraints, combination randomness, folding randomness
        // and we update the claimed sum of constraint evaluation.
        let mut round_constraints = Vec::new();
        let mut round_folding_randomness = Vec::new();
        let mut claimed_sum = EF::ZERO;
        let mut prev_commitment = parsed_commitment.clone();

        // Optional initial sumcheck round
        if self.initial_statement {
            // Combine OODS and statement constraints to claimed_sum
            let constraints: Vec<_> = prev_commitment
                .oods_constraints()
                .into_iter()
                .chain(statement.constraints.iter().cloned())
                .collect();
            let combination_randomness =
                self.combine_constraints(verifier_state, &mut claimed_sum, &constraints)?;
            round_constraints.push((combination_randomness, constraints));

            // Initial sumcheck
            let (_, folding_randomness) = read_sumcheck_rounds::<_, _, PS>(
                verifier_state,
                &mut claimed_sum,
                self.folding_factor.at_round(0),
                self.starting_folding_pow_bits,
                false,
            )?;
            round_folding_randomness.push(folding_randomness);
        } else {
            assert_eq!(prev_commitment.ood_points.len(), 0);
            assert!(statement.constraints.is_empty());
            round_constraints.push((vec![], vec![]));

            let mut folding_randomness = EF::zero_vec(self.folding_factor.at_round(0));
            verifier_state.fill_challenge_scalars(&mut folding_randomness)?;
            round_folding_randomness.push(MultilinearPoint(folding_randomness));

            // PoW
            self.verify_proof_of_work(verifier_state, self.starting_folding_pow_bits)?;
        }

        for round_index in 0..self.n_rounds() {
            // Fetch round parameters from config
            let round_params = &self.round_parameters[round_index];
            params.round_index = round_index;
            params.folding_factor = self.folding_factor.at_round(round_index);
            params.num_queries = round_params.num_queries;
            params.pow_bits = round_params.pow_bits;

            // Receive commitment to the folded polynomial (likely encoded at higher expansion)
            let new_commitment = ParsedCommitment::<_, Hash<F, u8, DIGEST_ELEMS>>::parse(
                verifier_state,
                params.num_variables,
                round_params.ood_samples,
            )?;

            // Verify in-domain challenges on the previous commitment.
            let stir_constraints = self.verify_stir_challenges(
                verifier_state,
                &params,
                &prev_commitment,
                round_folding_randomness.last().unwrap(),
                round_index == 0,
            )?;

            // Add out-of-domain and in-domain constraints to claimed_sum
            let constraints: Vec<Constraint<EF>> = new_commitment
                .oods_constraints()
                .into_iter()
                .chain(stir_constraints.into_iter())
                .collect();
            let combination_randomness =
                self.combine_constraints(verifier_state, &mut claimed_sum, &constraints)?;
            round_constraints.push((combination_randomness.clone(), constraints));

            let (_, folding_randomness) = read_sumcheck_rounds::<_, _, PS>(
                verifier_state,
                &mut claimed_sum,
                self.folding_factor.at_round(round_index + 1),
                round_params.folding_pow_bits,
                false,
            )?;
            round_folding_randomness.push(folding_randomness);

            // Update round parameters
            prev_commitment = new_commitment;
            params.num_variables -= params.folding_factor;
            params.domain_gen = params.domain_gen.square();
            params.exp_domain_gen = params
                .domain_gen
                .exp_u64(1 << self.folding_factor.at_round(round_index + 1));
            params.domain_gen_inv = params.domain_gen_inv.square();
            params.domain_size /= 2;
        }

        // Final round parameters.
        params.round_index = self.n_rounds();
        params.num_queries = self.final_queries;
        params.folding_factor = self.folding_factor.at_round(self.n_rounds());
        params.pow_bits = self.final_pow_bits;

        // In the final round we receive the full polynomial instead of a commitment.
        let mut final_coefficients = EF::zero_vec(1 << self.final_sumcheck_rounds);
        verifier_state.fill_next_scalars(&mut final_coefficients)?;
        let final_coefficients = CoefficientList::new(final_coefficients);

        // Verify in-domain challenges on the previous commitment.
        let stir_constraints = self.verify_stir_challenges(
            verifier_state,
            &params,
            &prev_commitment,
            round_folding_randomness.last().unwrap(),
            self.n_rounds() == 0,
        )?;

        // Verify stir constraints directly on final polynomial
        if !stir_constraints
            .iter()
            .all(|c| c.verify(&final_coefficients))
        {
            return Err(ProofError::InvalidProof);
        }

        let (_, final_sumcheck_randomness) = read_sumcheck_rounds::<_, _, PS>(
            verifier_state,
            &mut claimed_sum,
            self.final_sumcheck_rounds,
            self.final_folding_pow_bits,
            false,
        )?;
        round_folding_randomness.push(final_sumcheck_randomness.clone());

        // Compute folding randomness across all rounds.
        let folding_randomness = MultilinearPoint(
            round_folding_randomness
                .into_iter()
                .rev()
                .flat_map(|poly| poly.0.into_iter())
                .collect(),
        );

        // Compute evaluation of weights in folding randomness
        // Some weight computations can be deferred and will be returned for the caller
        // to verify.
        let deferred: Vec<EF> = verifier_state.hint()?;
        let evaluation_of_weights =
            self.eval_constraints_poly(&round_constraints, &deferred, folding_randomness.clone());

        // Check the final sumcheck evaluation
        let final_value = final_coefficients.evaluate(&final_sumcheck_randomness);
        if claimed_sum != evaluation_of_weights * final_value {
            return Err(ProofError::InvalidProof);
        }

        Ok((folding_randomness, deferred))
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

    /// Verify the prover's proof of work (PoW) challenge response.
    ///
    /// If the configured `bits` value is greater than zero, this function checks that
    /// the prover has provided a valid PoW nonce satisfying the difficulty constraint.
    /// This prevents spam and ensures the prover has committed nontrivial effort
    /// before submitting a proof.
    ///
    /// If `bits == 0.`, no proof of work is required and the function returns immediately.
    ///
    /// # Arguments
    /// - `verifier_state`: The verifier’s Fiat-Shamir state.
    /// - `bits`: The number of difficulty bits required for the proof of work.
    ///
    /// # Errors
    /// Returns `ProofError::InvalidProof` if the PoW response is invalid.
    pub fn verify_proof_of_work(
        &self,
        verifier_state: &mut VerifierState<'_, EF, F>,
        bits: f64,
    ) -> ProofResult<()> {
        if bits > 0. {
            verifier_state.challenge_pow::<PS>(bits)?;
        }
        Ok(())
    }

    /// Verify a STIR challenges against a commitment and return the constraints.
    pub fn verify_stir_challenges<const DIGEST_ELEMS: usize>(
        &self,
        verifier_state: &mut VerifierState<'_, EF, F>,
        params: &StirChallengParams<EF>,
        commitment: &ParsedCommitment<EF, Hash<F, u8, DIGEST_ELEMS>>,
        folding_randomness: &MultilinearPoint<EF>,
        leafs_base_field: bool,
    ) -> ProofResult<Vec<Constraint<EF>>>
    where
        H: CryptographicHasher<F, [u8; DIGEST_ELEMS]> + Sync,
        C: PseudoCompressionFunction<[u8; DIGEST_ELEMS], 2> + Sync,
        [u8; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        let stir_challenges_indexes = get_challenge_stir_queries(
            params.domain_size,
            params.folding_factor,
            params.num_queries,
            verifier_state,
        )?;

        let dimensions = vec![Dimensions {
            height: params.domain_size >> params.folding_factor,
            width: 1 << params.folding_factor,
        }];

        let answers = self.verify_merkle_proof(
            verifier_state,
            &commitment.root,
            &stir_challenges_indexes,
            &dimensions,
            leafs_base_field,
        )?;

        self.verify_proof_of_work(verifier_state, params.pow_bits)?;

        // Compute STIR Constraints
        let folds: Vec<EF> = answers
            .into_iter()
            .map(|answers| CoefficientList::new(answers).evaluate(folding_randomness))
            .collect();

        let stir_constraints = stir_challenges_indexes
            .iter()
            .map(|&index| params.exp_domain_gen.exp_u64(index as u64))
            .zip(&folds)
            .map(|(point, &value)| Constraint {
                weights: Weights::univariate(point, params.num_variables),
                sum: value,
                defer_evaluation: false,
            })
            .collect();

        Ok(stir_constraints)
    }

    /// Verify a merkle multi-opening proof for the provided indices.
    pub fn verify_merkle_proof<const DIGEST_ELEMS: usize>(
        &self,
        verifier_state: &mut VerifierState<'_, EF, F>,
        root: &Hash<F, u8, DIGEST_ELEMS>,
        indices: &[usize],
        dimensions: &[Dimensions],
        leafs_base_field: bool,
    ) -> ProofResult<Vec<Vec<EF>>>
    where
        H: CryptographicHasher<F, [u8; DIGEST_ELEMS]> + Sync,
        C: PseudoCompressionFunction<[u8; DIGEST_ELEMS], 2> + Sync,
        [u8; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        let mmcs = MerkleTreeMmcs::new(self.merkle_hash.clone(), self.merkle_compress.clone());

        let extension_mmcs = ExtensionMmcs::new(mmcs.clone());

        let res = if leafs_base_field {
            // Receive claimed leafs
            let answers = verifier_state.hint::<Leafs<F>>()?;

            // Receive merkle proof for leaf indices
            let merkle_proof = verifier_state.hint::<Proof<DIGEST_ELEMS>>()?;

            for (i, &index) in indices.iter().enumerate() {
                mmcs.verify_batch(
                    root,
                    dimensions,
                    index,
                    BatchOpeningRef {
                        opened_values: &[answers[i].clone()],
                        opening_proof: &merkle_proof[i],
                    },
                )
                .map_err(|_| ProofError::InvalidProof)?;
            }

            answers
                .into_iter()
                .map(|inner| inner.iter().map(|&f_el| f_el.into()).collect())
                .collect()
        } else {
            // Receive claimed leafs
            let answers = verifier_state.hint::<Leafs<EF>>()?;

            // Receive merkle proof for leaf indices
            let merkle_proof = verifier_state.hint::<Proof<DIGEST_ELEMS>>()?;

            for (i, &index) in indices.iter().enumerate() {
                extension_mmcs
                    .verify_batch(
                        root,
                        dimensions,
                        index,
                        BatchOpeningRef {
                            opened_values: &[answers[i].clone()],
                            opening_proof: &merkle_proof[i],
                        },
                    )
                    .map_err(|_| ProofError::InvalidProof)?;
            }

            answers
        };

        Ok(res)
    }

    /// Evaluate a batch of constraint polynomials at a given multilinear point.
    ///
    /// This function computes the combined weighted value of constraints across all rounds.
    /// Each constraint is either directly evaluated at the input point (`MultilinearPoint`)
    /// or substituted with a deferred evaluation result, depending on the constraint type.
    ///
    /// The final result is the sum of each constraint's value, scaled by its corresponding
    /// challenge randomness (used in the linear combination step of the sumcheck protocol).
    ///
    /// # Arguments
    /// - `constraints`: A list of tuples, where each tuple corresponds to a round and contains:
    ///     - A vector of challenge randomness values (used to weight each constraint),
    ///     - A vector of `Constraint<EF>` objects for that round.
    /// - `deferred`: Precomputed evaluations used for deferred constraints.
    /// - `point`: The multilinear point at which to evaluate the constraint polynomials.
    ///
    /// # Returns
    /// The combined evaluation result of all weighted constraints across rounds at the given point.
    ///
    /// # Panics
    /// Panics if:
    /// - Any round's `randomness.len()` does not match `constraints.len()`,
    /// - A deferred constraint is encountered but `deferred` has been exhausted.
    fn eval_constraints_poly(
        &self,
        constraints: &[(Vec<EF>, Vec<Constraint<EF>>)],
        deferred: &[EF],
        mut point: MultilinearPoint<EF>,
    ) -> EF {
        let mut num_variables = self.mv_parameters.num_variables;
        let mut deferred = deferred.iter().copied();
        let mut value = EF::ZERO;

        for (round, (randomness, constraints)) in constraints.iter().enumerate() {
            assert_eq!(randomness.len(), constraints.len());
            if round > 0 {
                num_variables -= self.folding_factor.at_round(round - 1);
                point = MultilinearPoint(point.0[..num_variables].to_vec());
            }
            value += constraints
                .iter()
                .zip(randomness)
                .map(|(constraint, &randomness)| {
                    let value = if constraint.defer_evaluation {
                        deferred.next().unwrap()
                    } else {
                        constraint.weights.compute(&point)
                    };
                    value * randomness
                })
                .sum::<EF>();
        }
        value
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
