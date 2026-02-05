use alloc::{format, vec, vec::Vec};
use core::{fmt::Debug, ops::Deref, slice::from_ref};

use errors::VerifierError;
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::{BatchOpeningRef, ExtensionMmcs, Mmcs};
use p3_field::{ExtensionField, Field, PackedValue, TwoAdicField};
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
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    whir::{
        constraints::{
            Constraint,
            evaluator::ConstraintPolyEvaluator,
            statement::{EqStatement, SelectStatement},
        },
        parameters::WhirConfig,
        proof::{QueryOpening, WhirProof},
        verifier::sumcheck::{verify_final_sumcheck_rounds, verify_sumcheck_rounds},
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
    pub fn verify<P, W, PW, const DIGEST_ELEMS: usize>(
        &self,
        proof: &WhirProof<F, EF, W, DIGEST_ELEMS>,
        challenger: &mut Challenger,
        parsed_commitment: &ParsedCommitment<EF, Hash<F, W, DIGEST_ELEMS>>,
        mut statement: EqStatement<EF>,
    ) -> Result<MultilinearPoint<EF>, VerifierError>
    where
        P: PackedValue<Value = F> + Eq + Send + Sync,
        W: PackedValue<Value = W> + Eq + Send + Sync + Copy,
        PW: PackedValue<Value = W> + Eq + Send + Sync,
        H: CryptographicHasher<F, [W; DIGEST_ELEMS]>
            + CryptographicHasher<P, [PW; DIGEST_ELEMS]>
            + Sync,
        C: PseudoCompressionFunction<[W; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[PW; DIGEST_ELEMS], 2>
            + Sync,
        Challenger: CanObserve<Hash<F, W, DIGEST_ELEMS>>,
        [W; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        // During the rounds we collect constraints, combination randomness, folding randomness
        // and we update the claimed sum of constraint evaluation.
        let mut constraints = Vec::new();
        let mut round_folding_randomness = Vec::new();
        let mut claimed_eval = EF::ZERO;
        let mut prev_commitment = parsed_commitment.clone();

        statement.concatenate(&prev_commitment.ood_statement);

        let constraint = Constraint::new(
            challenger.sample_algebra_element(),
            statement,
            SelectStatement::initialize(self.num_variables),
        );
        // Combine claimed evals with combination randomness
        constraint.combine_evals(&mut claimed_eval);
        constraints.push(constraint);

        let folding_randomness = verify_sumcheck_rounds(
            &proof.initial_sumcheck,
            challenger,
            &mut claimed_eval,
            self.starting_folding_pow_bits,
        )?;

        round_folding_randomness.push(folding_randomness);

        for round_index in 0..self.n_rounds() {
            // Fetch round parameters from config
            let round_params = &self.round_parameters[round_index];

            // Receive commitment to the folded polynomial (likely encoded at higher expansion)
            let new_commitment = ParsedCommitment::<_, Hash<F, W, DIGEST_ELEMS>>::parse_with_round(
                proof,
                challenger,
                round_params.num_variables,
                round_params.ood_samples,
                Some(round_index),
            );

            // Verify in-domain challenges on the previous commitment.
            let stir_statement = self.verify_stir_challenges::<P, W, PW, DIGEST_ELEMS>(
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

            let folding_randomness = verify_sumcheck_rounds(
                &proof.rounds[round_index].sumcheck,
                challenger,
                &mut claimed_eval,
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

        // Observe the final polynomial to the challenger
        challenger.observe_algebra_slice(final_evaluations.as_slice());

        // Verify in-domain challenges on the previous commitment.
        let stir_statement = self.verify_stir_challenges::<P, W, PW, DIGEST_ELEMS>(
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

        let point_for_eval = folding_randomness.reversed();

        let evaluation_of_weights = ConstraintPolyEvaluator::new(self.folding_factor)
            .eval_constraints_poly(&constraints, &point_for_eval);

        // Check the final sumcheck evaluation
        let final_value = final_evaluations.evaluate_hypercube_ext::<F>(&final_sumcheck_randomness);
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
    /// ready to be combined into the next round's sumcheck.
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
    /// to its evaluated, folded value under the prover's commitment.
    ///
    /// # Errors
    /// Returns `VerifierError::MerkleProofInvalid` if Merkle proof verification fails
    /// or the prover's data does not match the commitment.
    pub fn verify_stir_challenges<P, W, PW, const DIGEST_ELEMS: usize>(
        &self,
        proof: &crate::whir::proof::WhirProof<F, EF, W, DIGEST_ELEMS>,
        challenger: &mut Challenger,
        params: &RoundConfig<F>,
        commitment: &ParsedCommitment<EF, Hash<F, W, DIGEST_ELEMS>>,
        folding_randomness: &MultilinearPoint<EF>,
        round_index: usize,
    ) -> Result<SelectStatement<F, EF>, VerifierError>
    where
        P: PackedValue<Value = F> + Eq + Send + Sync,
        W: PackedValue<Value = W> + Eq + Send + Sync,
        PW: PackedValue<Value = W> + Eq + Send + Sync,
        H: CryptographicHasher<F, [W; DIGEST_ELEMS]>
            + CryptographicHasher<P, [PW; DIGEST_ELEMS]>
            + Sync,
        C: PseudoCompressionFunction<[W; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[PW; DIGEST_ELEMS], 2>
            + Sync,
        [W; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
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
        let answers = self.verify_merkle_proof::<P, W, PW, DIGEST_ELEMS>(
            proof,
            &commitment.root,
            &stir_challenges_indexes,
            &dimensions,
            round_index,
        )?;

        // Compute STIR Constraints
        let folds: Vec<_> = answers
            .into_iter()
            .map(|answer| {
                EvaluationsList::new(answer).evaluate_hypercube_ext::<F>(folding_randomness)
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
    /// This method checks that the prover's claimed leaf values at multiple positions
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
    pub fn verify_merkle_proof<P, W, PW, const DIGEST_ELEMS: usize>(
        &self,
        proof: &WhirProof<F, EF, W, DIGEST_ELEMS>,
        root: &Hash<F, W, DIGEST_ELEMS>,
        indices: &[usize],
        dimensions: &[Dimensions],
        round_index: usize,
    ) -> Result<Vec<Vec<EF>>, VerifierError>
    where
        P: PackedValue<Value = F> + Eq + Send + Sync,
        W: PackedValue<Value = W> + Eq + Send + Sync,
        PW: PackedValue<Value = W> + Eq + Send + Sync,
        H: CryptographicHasher<F, [W; DIGEST_ELEMS]>
            + CryptographicHasher<P, [PW; DIGEST_ELEMS]>
            + Sync,
        C: PseudoCompressionFunction<[W; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[PW; DIGEST_ELEMS], 2>
            + Sync,
        [W; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        let mmcs: MerkleTreeMmcs<P, PW, H, C, DIGEST_ELEMS> =
            MerkleTreeMmcs::new(self.merkle_hash.clone(), self.merkle_compress.clone());
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
