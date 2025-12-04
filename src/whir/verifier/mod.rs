use alloc::{format, vec, vec::Vec};
use core::{fmt::Debug, ops::Deref, slice::from_ref};

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
    alloc::string::ToString,
    constant::K_SKIP_SUMCHECK,
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    whir::{
        EqStatement,
        constraints::{Constraint, evaluator::ConstraintPolyEvaluator, statement::SelectStatement},
        parameters::{InitialPhaseConfig, WhirConfig},
        proof::{QueryOpening, WhirProof},
        verifier::sumcheck::{
            verify_final_sumcheck_rounds, verify_initial_sumcheck_rounds, verify_sumcheck_rounds,
        },
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
        proof: &WhirProof<F, EF, DIGEST_ELEMS>,
        challenger: &mut Challenger,
        parsed_commitment: &ParsedCommitment<EF, Hash<F, F, DIGEST_ELEMS>>,
        mut statement: EqStatement<EF>,
    ) -> Result<MultilinearPoint<EF>, VerifierError>
    where
        H: CryptographicHasher<F, [F; DIGEST_ELEMS]> + Sync,
        C: PseudoCompressionFunction<[F; DIGEST_ELEMS], 2> + Sync,
        [F; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        // During the rounds we collect constraints, combination randomness, folding randomness
        // and we update the claimed sum of constraint evaluation.
        let mut constraints = Vec::new();
        let mut round_folding_randomness = Vec::new();
        let mut claimed_eval = EF::ZERO;
        let mut prev_commitment = parsed_commitment.clone();

        // Optional constraint building - only if we have a statement
        if self.initial_phase_config.has_initial_statement() {
            statement.concatenate(&prev_commitment.ood_statement);

            let constraint = Constraint::new(
                challenger.sample_algebra_element(),
                statement,
                SelectStatement::initialize(self.num_variables),
            );
            // Combine claimed evals with combination randomness
            constraint.combine_evals(&mut claimed_eval);
            constraints.push(constraint);
        } else {
            assert!(prev_commitment.ood_statement.is_empty());
            assert!(statement.is_empty());
        }

        // Verify initial sumcheck
        let folding_randomness = verify_initial_sumcheck_rounds(
            &proof.initial_phase,
            challenger,
            &mut claimed_eval,
            self.folding_factor.at_round(0),
            self.starting_folding_pow_bits,
        )?;

        round_folding_randomness.push(folding_randomness);

        for round_index in 0..self.n_rounds() {
            // Fetch round parameters from config
            let round_params = &self.round_parameters[round_index];

            // Receive commitment to the folded polynomial (likely encoded at higher expansion)
            let new_commitment = ParsedCommitment::<_, Hash<F, F, DIGEST_ELEMS>>::parse_with_round(
                proof,
                challenger,
                round_params.num_variables,
                round_params.ood_samples,
                Some(round_index),
            );

            // Verify in-domain challenges on the previous commitment.
            let stir_statement = self.verify_stir_challenges(
                proof,
                challenger,
                round_params,
                &prev_commitment,
                round_folding_randomness.last().unwrap(),
                round_index,
            )?;

            let constraint = Constraint::new(
                challenger.sample_algebra_element(),
                new_commitment.ood_statement.clone(),
                stir_statement,
            );
            constraint.combine_evals(&mut claimed_eval);
            constraints.push(constraint);

            // TODO: SVO optimization is not yet fully implemented
            // Falls back to classic sumcheck for all optimization modes
            let folding_randomness = verify_sumcheck_rounds(
                &proof.rounds[round_index],
                challenger,
                &mut claimed_eval,
                self.folding_factor.at_round(round_index + 1),
                round_params.folding_pow_bits,
            )?;

            round_folding_randomness.push(folding_randomness);

            // Update round parameters
            prev_commitment = new_commitment;
        }

        // In the final round we receive the full polynomial instead of a commitment.
        let Some(final_evaluations) = proof.final_poly.clone() else {
            panic!("Expected final polynomial");
        };

        // Observe the final polynomial to the challenger (flatten to base field first)
        let flatten_base_scalar = EF::flatten_to_base(final_evaluations.as_slice().to_vec());
        challenger.observe_slice(&flatten_base_scalar);

        // Verify in-domain challenges on the previous commitment.
        let stir_statement = self.verify_stir_challenges(
            proof,
            challenger,
            &self.final_round_config(),
            &prev_commitment,
            round_folding_randomness.last().unwrap(),
            self.n_rounds(),
        )?;

        // Verify stir constraints directly on final polynomial
        stir_statement
            .verify(&final_evaluations)
            .then_some(())
            .ok_or_else(|| VerifierError::StirChallengeFailed {
                challenge_id: 0,
                details: "STIR constraint verification failed on final polynomial".to_string(),
            })?;

        // TODO: SVO optimization is not yet fully implemented
        // Falls back to classic sumcheck for all optimization modes
        let final_sumcheck_randomness = verify_final_sumcheck_rounds(
            proof.final_sumcheck.as_ref(),
            challenger,
            &mut claimed_eval,
            self.final_sumcheck_rounds,
            self.final_folding_pow_bits,
        )?;

        round_folding_randomness.push(final_sumcheck_randomness.clone());

        // Compute folding randomness across all rounds.
        let folding_randomness = MultilinearPoint::new(
            round_folding_randomness
                .into_iter()
                .flat_map(IntoIterator::into_iter)
                .collect(),
        );

        // For skip case, don't reverse the randomness (prover stores it in forward order)
        // For non-skip case, reverse it to match the prover's storage
        let is_skip_used = self.initial_phase_config.is_univariate_skip()
            && K_SKIP_SUMCHECK <= self.folding_factor.at_round(0);

        let point_for_eval = if is_skip_used {
            folding_randomness.clone()
        } else {
            folding_randomness.reversed()
        };

        let evaluation_of_weights = ConstraintPolyEvaluator::new(
            self.num_variables,
            self.folding_factor,
            is_skip_used.then_some(K_SKIP_SUMCHECK),
        )
        .eval_constraints_poly(&constraints, &point_for_eval);

        // Check the final sumcheck evaluation
        let final_value = final_evaluations.evaluate_hypercube(&final_sumcheck_randomness);
        if claimed_eval != evaluation_of_weights * final_value {
            return Err(VerifierError::SumcheckFailed {
                round: self.final_sumcheck_rounds,
                expected: (evaluation_of_weights * final_value).to_string(),
                actual: claimed_eval.to_string(),
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
    /// - `proof`: The WHIR proof containing query openings and Merkle proofs.
    /// - `challenger`: The Fiat-Shamir challenger for transcript management.
    /// - `params`: Parameters for the current STIR round (domain size, folding factor, etc.).
    /// - `commitment`: The prover's commitment to the folded polynomial.
    /// - `folding_randomness`: Random point for folding the evaluations.
    /// - `round_index`: The current round index in the protocol.
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
        proof: &crate::whir::proof::WhirProof<F, EF, DIGEST_ELEMS>,
        challenger: &mut Challenger,
        params: &RoundConfig<F>,
        commitment: &ParsedCommitment<EF, Hash<F, F, DIGEST_ELEMS>>,
        folding_randomness: &MultilinearPoint<EF>,
        round_index: usize,
    ) -> Result<SelectStatement<F, EF>, VerifierError>
    where
        H: CryptographicHasher<F, [F; DIGEST_ELEMS]> + Sync,
        C: PseudoCompressionFunction<[F; DIGEST_ELEMS], 2> + Sync,
        [F; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
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
        let pow_witness = if round_index < self.n_rounds() {
            proof
                .get_pow_after_commitment(round_index)
                .ok_or(VerifierError::InvalidRoundIndex { index: round_index })?
        } else {
            // Final round uses final_pow_witness
            proof.final_pow_witness
        };
        if params.pow_bits > 0 && !challenger.check_witness(params.pow_bits, pow_witness) {
            return Err(VerifierError::InvalidPowWitness);
        }

        // Transcript checkpoint after PoW
        if round_index < self.n_rounds() {
            challenger.sample();
        }

        let stir_challenges_indexes = get_challenge_stir_queries::<Challenger, F, EF>(
            params.domain_size,
            params.folding_factor,
            params.num_queries,
            challenger,
        )?;

        let dimensions = vec![Dimensions {
            height: params.domain_size >> params.folding_factor,
            width: 1 << params.folding_factor,
        }];
        let answers = self.verify_merkle_proof(
            proof,
            &commitment.root,
            &stir_challenges_indexes,
            &dimensions,
            round_index,
        )?;

        // Determine if this is the special first round where the univariate skip is applied.
        let is_skip_round = round_index == 0
            && matches!(
                self.initial_phase_config,
                InitialPhaseConfig::WithStatementUnivariateSkip
            )
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
                    EvaluationsList::new(folded_row).evaluate_hypercube(&r_rest)
                } else {
                    EvaluationsList::new(answer).evaluate_hypercube(folding_randomness)
                }
            })
            .collect();

        let stir_constraints = stir_challenges_indexes
            .iter()
            .map(|&index| params.folded_domain_gen.exp_u64(index as u64))
            .collect();

        Ok(SelectStatement::new(
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
    /// - `proof`: The WHIR proof containing query openings and Merkle proofs.
    /// - `root`: The Merkle root hash the prover's claims are verified against.
    /// - `indices`: The list of queried leaf indices.
    /// - `dimensions`: The shape of the underlying matrix being committed (for MMCS verification).
    /// - `round_index`: The current round index to determine which queries to use from the proof.
    ///
    /// # Returns
    /// A vector of decoded leaf values, one `Vec<EF>` per queried index.
    ///
    /// # Errors
    /// Returns `VerifierError::MerkleProofInvalid` if any Merkle proof fails verification.
    pub fn verify_merkle_proof<const DIGEST_ELEMS: usize>(
        &self,
        proof: &WhirProof<F, EF, DIGEST_ELEMS>,
        root: &Hash<F, F, DIGEST_ELEMS>,
        indices: &[usize],
        dimensions: &[Dimensions],
        round_index: usize,
    ) -> Result<Vec<Vec<EF>>, VerifierError>
    where
        H: CryptographicHasher<F, [F; DIGEST_ELEMS]> + Sync,
        C: PseudoCompressionFunction<[F; DIGEST_ELEMS], 2> + Sync,
        [F; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        let mmcs = MerkleTreeMmcs::new(self.merkle_hash.clone(), self.merkle_compress.clone());
        let extension_mmcs = ExtensionMmcs::new(mmcs.clone());

        // Determine which queries to use from the proof structure
        let queries = if round_index == self.n_rounds() {
            &proof.final_queries
        } else {
            &proof
                .rounds
                .get(round_index)
                .ok_or_else(|| VerifierError::MerkleProofInvalid {
                    position: 0,
                    reason: format!("Round {round_index} not found in proof"),
                })?
                .queries
        };

        let mut results = Vec::with_capacity(indices.len());

        for (&index, query) in indices.iter().zip(queries.iter()) {
            let values_ef = match query {
                QueryOpening::Base { values, proof } => {
                    mmcs.verify_batch(
                        root,
                        dimensions,
                        index,
                        BatchOpeningRef {
                            opened_values: from_ref(values),
                            opening_proof: proof,
                        },
                    )
                    .map_err(|_| VerifierError::MerkleProofInvalid {
                        position: index,
                        reason: "Base field Merkle proof verification failed".to_string(),
                    })?;

                    // Convert F -> EF
                    values.iter().map(|&f| f.into()).collect()
                }
                QueryOpening::Extension { values, proof } => {
                    extension_mmcs
                        .verify_batch(
                            root,
                            dimensions,
                            index,
                            BatchOpeningRef {
                                opened_values: from_ref(values),
                                opening_proof: proof,
                            },
                        )
                        .map_err(|_| VerifierError::MerkleProofInvalid {
                            position: index,
                            reason: "Extension field Merkle proof verification failed".to_string(),
                        })?;

                    values.clone()
                }
            };

            results.push(values_ef);
        }

        Ok(results)
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
