use std::{fmt::Debug, ops::Deref};

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_commit::{BatchOpeningRef, ExtensionMmcs, Mmcs};
use p3_field::{ExtensionField, Field, Packable, TwoAdicField};
use p3_matrix::Dimensions;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CryptographicHasher, Hash, PseudoCompressionFunction};
use serde::{Deserialize, Serialize};
use tracing::instrument;

use super::{
    committer::reader::ParsedCommitment,
    parameters::RoundConfig,
    statement::{constraint::Constraint, weights::Weights},
    utils::get_challenge_stir_queries,
};
use crate::{
    fiat_shamir::{
        errors::{ProofError, ProofResult},
        verifier::VerifierState,
    },
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    whir::{Statement, parameters::WhirConfig, verifier::sumcheck::verify_sumcheck_rounds},
};

pub mod sumcheck;

/// Wrapper around the WHIR verifier configuration.
///
/// This type provides a lightweight, ergonomic interface to verification methods
/// by wrapping a reference to the `WhirConfig`.
#[derive(Debug)]
pub struct Verifier<'a, EF, F, H, C, Challenger>(
    /// Reference to the verifier’s configuration containing all round parameters.
    pub(crate) &'a WhirConfig<EF, F, H, C, Challenger>,
)
where
    F: Field,
    EF: ExtensionField<F>;

impl<'a, EF, F, H, C, Challenger> Verifier<'a, EF, F, H, C, Challenger>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    pub const fn new(params: &'a WhirConfig<EF, F, H, C, Challenger>) -> Self {
        Self(params)
    }

    #[instrument(skip_all)]
    #[allow(clippy::too_many_lines)]
    pub fn verify<const DIGEST_ELEMS: usize>(
        &self,
        verifier_state: &mut VerifierState<F, EF, Challenger>,
        parsed_commitment: &ParsedCommitment<EF, Hash<F, F, DIGEST_ELEMS>>,
        statement: &Statement<EF>,
    ) -> ProofResult<(MultilinearPoint<EF>, Vec<EF>)>
    where
        H: CryptographicHasher<F, [F; DIGEST_ELEMS]>
            + CryptographicHasher<F::Packing, [F::Packing; DIGEST_ELEMS]>
            + Sync,
        C: PseudoCompressionFunction<[F; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[F::Packing; DIGEST_ELEMS], 2>
            + Sync,
        [F; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
        F: Eq + Packable,
    {
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
            let folding_randomness = verify_sumcheck_rounds(
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
            for folded_randomness in &mut folding_randomness {
                *folded_randomness = verifier_state.sample();
            }
            round_folding_randomness.push(MultilinearPoint(folding_randomness));

            verifier_state.check_pow_grinding(self.starting_folding_pow_bits)?;
        }

        for round_index in 0..self.n_rounds() {
            // Fetch round parameters from config
            let round_params = &self.round_parameters[round_index];

            // Receive commitment to the folded polynomial (likely encoded at higher expansion)
            let new_commitment = ParsedCommitment::<_, Hash<F, F, DIGEST_ELEMS>>::parse(
                verifier_state,
                round_params.num_variables,
                round_params.ood_samples,
            )?;

            // Verify in-domain challenges on the previous commitment.
            let stir_constraints = self.verify_stir_challenges(
                verifier_state,
                round_params,
                &prev_commitment,
                round_folding_randomness.last().unwrap(),
                round_index,
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

            let folding_randomness = verify_sumcheck_rounds(
                verifier_state,
                &mut claimed_sum,
                self.folding_factor.at_round(round_index + 1),
                round_params.folding_pow_bits,
                false,
            )?;

            round_folding_randomness.push(folding_randomness);

            // Update round parameters
            prev_commitment = new_commitment;
        }

        // In the final round we receive the full polynomial instead of a commitment.
        let n_final_coeffs = 1 << self.0.n_vars_of_final_polynomial();
        let final_coefficients = verifier_state.next_extension_scalars_vec(n_final_coeffs)?;
        let final_evaluations = EvaluationsList::new(final_coefficients);

        // Verify in-domain challenges on the previous commitment.
        let stir_constraints = self.verify_stir_challenges(
            verifier_state,
            &self.final_round_config(),
            &prev_commitment,
            round_folding_randomness.last().unwrap(),
            self.n_rounds(),
        )?;

        // Verify stir constraints directly on final polynomial
        stir_constraints
            .iter()
            .all(|c| c.verify(&final_evaluations))
            .then_some(())
            .ok_or(ProofError::InvalidProof)?;

        let final_sumcheck_randomness = verify_sumcheck_rounds(
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
        let deferred =
            verifier_state.next_extension_scalars_vec(statement.num_deref_constraints())?;

        let evaluation_of_weights =
            self.eval_constraints_poly(&round_constraints, &deferred, folding_randomness.clone());

        // Check the final sumcheck evaluation
        let final_value = final_evaluations.evaluate(&final_sumcheck_randomness);
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
        verifier_state: &mut VerifierState<F, EF, Challenger>,
        claimed_sum: &mut EF,
        constraints: &[Constraint<EF>],
    ) -> ProofResult<Vec<EF>> {
        let combination_randomness_gen: EF = verifier_state.sample();
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

    /// Verify STIR in-domain queries and produce associated constraints.
    ///
    /// This method runs the STIR query phase on a given commitment.
    /// It selects random in-domain indices (STIR challenges)
    /// and verifies Merkle proofs for the claimed values at these indices.
    ///
    /// After verification, it evaluates the folded polynomial at these queried points.
    /// It then packages the results as a list of `Constraint` objects,
    /// ready to be combined into the next round’s sumcheck.
    ///
    /// # Arguments
    /// - `verifier_state`: The verifier’s Fiat-Shamir state.
    /// - `params`: Parameters for the current STIR round (domain size, folding factor, etc.).
    /// - `commitment`: The prover’s commitment to the folded polynomial.
    /// - `folding_randomness`: Random point for folding the evaluations.
    /// - `leafs_base_field`: Whether the leaf data is in the base field or extension field.
    ///
    /// # Returns
    /// A vector of `Constraint` objects, each linking a queried domain point
    /// to its evaluated, folded value under the prover’s commitment.
    ///
    /// # Errors
    /// Returns `ProofError::InvalidProof` if Merkle proof verification fails
    /// or the prover’s data does not match the commitment.
    pub fn verify_stir_challenges<const DIGEST_ELEMS: usize>(
        &self,
        verifier_state: &mut VerifierState<F, EF, Challenger>,
        params: &RoundConfig<EF>,
        commitment: &ParsedCommitment<EF, Hash<F, F, DIGEST_ELEMS>>,
        folding_randomness: &MultilinearPoint<EF>,
        round_index: usize,
    ) -> ProofResult<Vec<Constraint<EF>>>
    where
        H: CryptographicHasher<F, [F; DIGEST_ELEMS]>
            + CryptographicHasher<F::Packing, [F::Packing; DIGEST_ELEMS]>
            + Sync,
        C: PseudoCompressionFunction<[F; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[F::Packing; DIGEST_ELEMS], 2>
            + Sync,
        [F; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
        F: Eq + Packable,
    {
        let leafs_base_field = round_index == 0;

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
            round_index,
        )?;

        verifier_state.check_pow_grinding(params.pow_bits)?;

        // Compute STIR Constraints
        let folds: Vec<_> = answers
            .into_iter()
            .map(|answers| EvaluationsList::new(answers).evaluate(folding_randomness))
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

    /// Verify a Merkle multi-opening proof for the provided indices.
    ///
    /// This method checks that the prover’s claimed leaf values at multiple positions
    /// match the committed Merkle root, using batch Merkle proofs.
    /// It supports both base field and extension field leaf types.
    ///
    /// For each queried index:
    /// - It reads the claimed leaf values and associated Merkle proof from the transcript.
    /// - It verifies the Merkle opening against the provided root and dimensions.
    /// - If verification passes, it collects and returns the decoded leaf values.
    ///
    /// # Arguments
    /// - `verifier_state`: The verifier’s Fiat-Shamir transcript state.
    /// - `root`: The Merkle root hash the prover’s claims are verified against.
    /// - `indices`: The list of queried leaf indices.
    /// - `dimensions`: The shape of the underlying matrix being committed (for MMCS verification).
    /// - `leafs_base_field`: Indicates whether leafs are in the base field (`F`) or extension field (`EF`).
    ///
    /// # Returns
    /// A vector of decoded leaf values, one `Vec<EF>` per queried index.
    ///
    /// # Errors
    /// Returns `ProofError::InvalidProof` if any Merkle proof fails verification.
    pub fn verify_merkle_proof<const DIGEST_ELEMS: usize>(
        &self,
        verifier_state: &mut VerifierState<F, EF, Challenger>,
        root: &Hash<F, F, DIGEST_ELEMS>,
        indices: &[usize],
        dimensions: &[Dimensions],
        leafs_base_field: bool,
        round_index: usize,
    ) -> ProofResult<Vec<Vec<EF>>>
    where
        H: CryptographicHasher<F, [F; DIGEST_ELEMS]>
            + CryptographicHasher<F::Packing, [F::Packing; DIGEST_ELEMS]>
            + Sync,
        C: PseudoCompressionFunction<[F; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[F::Packing; DIGEST_ELEMS], 2>
            + Sync,
        [F; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
        F: Eq + Packable,
    {
        // Create a Merkle MMCS instance
        let mmcs = MerkleTreeMmcs::<F::Packing, F::Packing, H, C, DIGEST_ELEMS>::new(
            self.merkle_hash.clone(),
            self.merkle_compress.clone(),
        );

        // Wrap the MMCS in an extension-aware wrapper for EF leaf support.
        let extension_mmcs = ExtensionMmcs::new(mmcs.clone());

        // Branch depending on whether the committed leafs are base field or extension field.
        let res = if leafs_base_field {
            // Merkle leaves
            let mut answers = vec![];
            let merkle_leaf_size = 1 << self.folding_factor.at_round(round_index);
            for _ in 0..indices.len() {
                answers.push(verifier_state.receive_hint_base_scalars(merkle_leaf_size)?);
            }

            // Merkle proofs
            let mut merkle_proofs = Vec::new();
            for _ in 0..indices.len() {
                let mut merkle_path = vec![];
                for _ in 0..self.merkle_tree_height(round_index) {
                    let digest: [F; DIGEST_ELEMS] = verifier_state
                        .receive_hint_base_scalars(DIGEST_ELEMS)?
                        .try_into()
                        .unwrap();
                    merkle_path.push(digest);
                }
                merkle_proofs.push(merkle_path);
            }

            // For each queried index:
            for (i, &index) in indices.iter().enumerate() {
                // Verify the Merkle opening for the claimed leaf against the Merkle root.
                mmcs.verify_batch(
                    root,
                    dimensions,
                    index,
                    BatchOpeningRef {
                        opened_values: &[answers[i].clone()],
                        opening_proof: &merkle_proofs[i],
                    },
                )
                .map_err(|_| ProofError::InvalidProof)?;
            }

            // Convert the base field values to EF and collect them into a result vector.
            answers
                .into_iter()
                .map(|inner| inner.iter().map(|&f_el| f_el.into()).collect())
                .collect()
        } else {
            // Merkle leaves
            let mut answers = vec![];
            let merkle_leaf_size = 1 << self.folding_factor.at_round(round_index);
            for _ in 0..indices.len() {
                answers.push(verifier_state.receive_hint_extension_scalars(merkle_leaf_size)?);
            }

            // Merkle proofs
            let mut merkle_proofs = Vec::new();
            for _ in 0..indices.len() {
                let mut merkle_path = vec![];
                for _ in 0..self.merkle_tree_height(round_index) {
                    let digest: [F; DIGEST_ELEMS] = verifier_state
                        .receive_hint_base_scalars(DIGEST_ELEMS)?
                        .try_into()
                        .unwrap();
                    merkle_path.push(digest);
                }
                merkle_proofs.push(merkle_path);
            }

            // For each queried index:
            for (i, &index) in indices.iter().enumerate() {
                // Verify the Merkle opening against the extension MMCS.
                extension_mmcs
                    .verify_batch(
                        root,
                        dimensions,
                        index,
                        BatchOpeningRef {
                            opened_values: &[answers[i].clone()],
                            opening_proof: &merkle_proofs[i],
                        },
                    )
                    .map_err(|_| ProofError::InvalidProof)?;
            }

            // Return the extension field answers as-is.
            answers
        };

        // Return the verified leaf values.
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

impl<EF, F, H, C, Challenger> Deref for Verifier<'_, EF, F, H, C, Challenger>
where
    F: Field,
    EF: ExtensionField<F>,
{
    type Target = WhirConfig<EF, F, H, C, Challenger>;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}
