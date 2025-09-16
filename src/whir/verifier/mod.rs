use std::{fmt::Debug, ops::Deref};

use errors::VerifierError;
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_commit::{BatchOpeningRef, ExtensionMmcs, Mmcs};
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_interpolation::interpolate_subgroup;
use p3_matrix::Dimensions;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CryptographicHasher, Hash, PseudoCompressionFunction};
use serde::{Deserialize, Serialize};
use tracing::instrument;

use super::{
    committer::reader::ParsedCommitment, parameters::RoundConfig, utils::get_challenge_stir_queries,
};
use crate::{
    constant::K_SKIP_SUMCHECK,
    fiat_shamir::verifier::VerifierState,
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    whir::{
        Statement, constraints::evaluator::ConstraintPolyEvaluator, parameters::WhirConfig,
        verifier::sumcheck::verify_sumcheck_rounds,
    },
};

pub mod errors;
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
    ) -> Result<MultilinearPoint<EF>, VerifierError>
    where
        H: CryptographicHasher<F, [F; DIGEST_ELEMS]> + Sync,
        C: PseudoCompressionFunction<[F; DIGEST_ELEMS], 2> + Sync,
        [F; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        // During the rounds we collect constraints, combination randomness, folding randomness
        // and we update the claimed sum of constraint evaluation.
        let mut round_constraints = Vec::new();
        let mut round_folding_randomness = Vec::new();
        let mut claimed_sum = EF::ZERO;
        let mut prev_commitment = parsed_commitment.clone();

        // Optional initial sumcheck round
        if self.initial_statement {
            // Combine OODS constraints and statement constraints to claimed_sum
            let mut new_statement = prev_commitment.oods_constraints();
            new_statement.concatenate(statement);

            let gamma = verifier_state.sample();
            let combination_randomness = new_statement.combine_evals(&mut claimed_sum, gamma);

            round_constraints.push((combination_randomness, new_statement));

            // Initial sumcheck
            let folding_randomness = verify_sumcheck_rounds(
                verifier_state,
                &mut claimed_sum,
                self.folding_factor.at_round(0),
                self.starting_folding_pow_bits,
                self.univariate_skip,
            )?;
            round_folding_randomness.push(folding_randomness);
        } else {
            assert!(prev_commitment.ood_points.is_empty());
            assert!(statement.is_empty());

            let folding_randomness = MultilinearPoint::new(
                (0..self.folding_factor.at_round(0))
                    .map(|_| verifier_state.sample())
                    .collect::<Vec<_>>(),
            );

            round_folding_randomness.push(folding_randomness);

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
            let mut new_statement = new_commitment.oods_constraints();
            new_statement.concatenate(&stir_constraints);

            let gamma = verifier_state.sample();
            let combination_randomness = new_statement.combine_evals(&mut claimed_sum, gamma);

            round_constraints.push((combination_randomness.clone(), new_statement));

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
        let n_final_coeffs = 1 << self.n_vars_of_final_polynomial();
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
            .verify(&final_evaluations)
            .then_some(())
            .ok_or_else(|| VerifierError::StirChallengeFailed {
                challenge_id: 0,
                details: "STIR constraint verification failed on final polynomial".to_string(),
            })?;

        let final_sumcheck_randomness = verify_sumcheck_rounds(
            verifier_state,
            &mut claimed_sum,
            self.final_sumcheck_rounds,
            self.final_folding_pow_bits,
            false,
        )?;
        round_folding_randomness.push(final_sumcheck_randomness.clone());

        // Compute folding randomness across all rounds.
        let folding_randomness = MultilinearPoint::new(
            round_folding_randomness
                .into_iter()
                .rev()
                .flat_map(IntoIterator::into_iter)
                .collect(),
        );

        let evaluator = ConstraintPolyEvaluator::new(
            self.num_variables,
            self.folding_factor,
            self.univariate_skip,
        );
        let evaluation_of_weights =
            evaluator.eval_constraints_poly(&round_constraints, &folding_randomness);

        // Check the final sumcheck evaluation
        let final_value = final_evaluations.evaluate(&final_sumcheck_randomness);
        if claimed_sum != evaluation_of_weights * final_value {
            return Err(VerifierError::SumcheckFailed {
                round: self.final_sumcheck_rounds,
                expected: (evaluation_of_weights * final_value).to_string(),
                actual: claimed_sum.to_string(),
            });
        }

        Ok(folding_randomness)
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
    /// Returns `VerifierError::MerkleProofInvalid` if Merkle proof verification fails
    /// or the prover’s data does not match the commitment.
    pub fn verify_stir_challenges<const DIGEST_ELEMS: usize>(
        &self,
        verifier_state: &mut VerifierState<F, EF, Challenger>,
        params: &RoundConfig<F>,
        commitment: &ParsedCommitment<EF, Hash<F, F, DIGEST_ELEMS>>,
        folding_randomness: &MultilinearPoint<EF>,
        round_index: usize,
    ) -> Result<Statement<EF>, VerifierError>
    where
        H: CryptographicHasher<F, [F; DIGEST_ELEMS]> + Sync,
        C: PseudoCompressionFunction<[F; DIGEST_ELEMS], 2> + Sync,
        [F; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        let leafs_base_field = round_index == 0;

        // CRITICAL: Verify the prover's proof-of-work before generating challenges.
        //
        // This is the verifier's counterpart to the prover's grinding step and is essential
        // for protocol soundness.
        //
        // The query locations (`stir_challenges_indexes`) we are about to generate are derived
        // from the transcript, which includes the prover's commitment for this round. To prevent
        // a malicious prover from repeatedly trying different commitments until they find one that
        // produces "easy" queries, the protocol forces the prover to perform an expensive
        // proof-of-work (grinding) after they commit.
        //
        // By verifying that proof-of-work *now*, we confirm that the prover "locked in" their
        // commitment at a significant computational cost. This gives us confidence that the
        // challenges we generate are unpredictable and unbiased by a cheating prover.
        verifier_state.check_pow_grinding(params.pow_bits)?;

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

        // Determine if this is the special first round where the univariate skip is applied.
        let is_skip_round = self.initial_statement
            && round_index == 0
            && self.univariate_skip
            && self.folding_factor.at_round(0) >= K_SKIP_SUMCHECK;

        // Compute STIR Constraints
        let folds: Vec<_> = answers
            .into_iter()
            .map(|answer| {
                if is_skip_round {
                    // Case 1: Univariate Skip Round Evaluation
                    //
                    // The `answer` contains evaluations of a polynomial over the `k_skip` variables.
                    let evals = EvaluationsList::new(answer);

                    // Calculate `n-k`, the number of variables that are *not* folded in this skip round.
                    let num_remaining_vars = evals.num_variables() - K_SKIP_SUMCHECK;

                    // Determine the width of the evaluation matrix, which is `2^(n-k)`.
                    let width = 1 << num_remaining_vars;

                    // Reshape the flat `2^n` evaluations into a `2^k x 2^(n-k)` matrix.
                    let mat = evals.into_mat(width);

                    // The `folding_randomness` for a skip round is the special `(n-k)+1` challenge object.
                    let r_all = folding_randomness.clone();

                    // Deconstruct the challenge object `r_all` into its two components.
                    //
                    // The last element is the single challenge `r_skip` used to evaluate the skipped variables.
                    let r_skip = *r_all
                        .last_variable()
                        .expect("skip challenge must be present");
                    // The first `n - k_skip` elements are the challenges `r_rest` for the remaining variables.
                    let r_rest =
                        MultilinearPoint::new(r_all.as_slice()[..num_remaining_vars].to_vec());

                    // Perform the two-stage skip-aware evaluation:
                    //
                    // "Fold" the skipped variables by interpolating the matrix at `r_skip`.
                    let folded_row = interpolate_subgroup(&mat, r_skip);
                    // Evaluate the resulting smaller polynomial at the remaining challenges `r_rest`.
                    EvaluationsList::new(folded_row).evaluate(&r_rest)
                } else {
                    EvaluationsList::new(answer).evaluate(folding_randomness)
                }
            })
            .collect();

        let stir_constraints = stir_challenges_indexes
            .iter()
            .map(|&index| params.folded_domain_gen.exp_u64(index as u64))
            .map(|point| {
                MultilinearPoint::expand_from_univariate(EF::from(point), params.num_variables)
            })
            .collect();

        Ok(Statement::new(
            params.num_variables,
            stir_constraints,
            folds,
        ))
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
    /// Returns `VerifierError::MerkleProofInvalid` if any Merkle proof fails verification.
    pub fn verify_merkle_proof<const DIGEST_ELEMS: usize>(
        &self,
        verifier_state: &mut VerifierState<F, EF, Challenger>,
        root: &Hash<F, F, DIGEST_ELEMS>,
        indices: &[usize],
        dimensions: &[Dimensions],
        leafs_base_field: bool,
        round_index: usize,
    ) -> Result<Vec<Vec<EF>>, VerifierError>
    where
        H: CryptographicHasher<F, [F; DIGEST_ELEMS]> + Sync,
        C: PseudoCompressionFunction<[F; DIGEST_ELEMS], 2> + Sync,
        [F; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        // Create a Merkle MMCS instance
        let mmcs = MerkleTreeMmcs::new(self.merkle_hash.clone(), self.merkle_compress.clone());

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
                .map_err(|_| VerifierError::MerkleProofInvalid {
                    position: index,
                    reason: "Base field Merkle proof verification failed".to_string(),
                })?;
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
                    .map_err(|_| VerifierError::MerkleProofInvalid {
                        position: index,
                        reason: "Extension field Merkle proof verification failed".to_string(),
                    })?;
            }

            // Return the extension field answers as-is.
            answers
        };

        // Return the verified leaf values.
        Ok(res)
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
