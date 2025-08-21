use std::{fmt::Debug, ops::Deref};

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_commit::{BatchOpeningRef, ExtensionMmcs, Mmcs};
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_interpolation::interpolate_subgroup;
use p3_matrix::{Dimensions, dense::RowMajorMatrix};
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
    constant::K_SKIP_SUMCHECK,
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
                self.univariate_skip,
            )?;
            round_folding_randomness.push(folding_randomness);
        } else {
            assert_eq!(prev_commitment.ood_points.len(), 0);
            assert!(statement.constraints.is_empty());
            round_constraints.push((vec![], vec![]));

            let folding_randomness = MultilinearPoint(
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
            self.eval_constraints_poly(&round_constraints, &deferred, &folding_randomness);

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
        let combination_randomness = combination_randomness_gen
            .powers()
            .collect_n(constraints.len());
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
        params: &RoundConfig<F>,
        commitment: &ParsedCommitment<EF, Hash<F, F, DIGEST_ELEMS>>,
        folding_randomness: &MultilinearPoint<EF>,
        round_index: usize,
    ) -> ProofResult<Vec<Constraint<EF>>>
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
                    let mat = RowMajorMatrix::new(evals.to_vec(), width);

                    // The `folding_randomness` for a skip round is the special `(n-k)+1` challenge object.
                    let r_all = folding_randomness.clone();

                    // Deconstruct the challenge object `r_all` into its two components.
                    //
                    // The last element is the single challenge `r_skip` used to evaluate the skipped variables.
                    let r_skip = *r_all.0.last().expect("skip challenge must be present");
                    // The first `n - k_skip` elements are the challenges `r_rest` for the remaining variables.
                    let r_rest = MultilinearPoint(r_all.0[..num_remaining_vars].to_vec());

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
            .zip(&folds)
            .map(|(point, &value)| Constraint {
                weights: Weights::univariate(EF::from(point), params.num_variables),
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

    /// Evaluates the final combined weight polynomial `W(r)` at the challenge point `r`.
    ///
    /// This is the verifier's master function for the final check of the sumcheck protocol. It
    /// correctly handles the recursive nature of the proof, where constraints are defined over
    /// progressively smaller domains.
    ///
    /// Crucially, it also handles the special case where the first round uses the **univariate skip**
    /// optimization, applying a different, skip-aware evaluation method for that round's constraints.
    ///
    /// # Arguments
    /// * `constraints`: A list of constraint sets, one for each round of the protocol. Each set
    ///   is paired with the random `alpha` powers used to linearly combine them.
    /// * `deferred`: A list of pre-computed evaluations for any deferred constraints.
    /// * `point`: The final, full `n`-dimensional challenge point `r` (`folding_randomness`).
    ///
    /// # Returns
    /// The evaluation `W(r)` as a single field element.
    fn eval_constraints_poly(
        &self,
        constraints: &[(Vec<EF>, Vec<Constraint<EF>>)],
        deferred: &[EF],
        point: &MultilinearPoint<EF>,
    ) -> EF {
        // The total number of variables in the original, unfolded polynomial.
        let mut num_variables = self.mv_parameters.num_variables;
        // An iterator for any evaluations that were deferred in the protocol.
        let mut deferred_iter = deferred.iter().copied();
        // The final, accumulated evaluation of W(r).
        let mut value = EF::ZERO;

        // Process the constraints from each round of the protocol.
        for (round_idx, (combination_coeffs, constraints_in_round)) in
            constraints.iter().enumerate()
        {
            // Create a view of the challenge point that matches the domain size for the current round.
            let point_for_round = if round_idx > 0 {
                // If it's not the first round, shrink the number of variables.
                num_variables -= self.folding_factor.at_round(round_idx - 1);
                // The value of the `if` block is the new, sliced point.
                MultilinearPoint(point[..num_variables].to_vec())
            } else {
                // Otherwise, for the first round, use the full, original point.
                point.clone()
            };

            // Check if this is the first round and if the univariate skip optimization is active.
            let is_skip_round = round_idx == 0
                && self.univariate_skip
                && self.folding_factor.at_round(0) >= K_SKIP_SUMCHECK;

            // Calculate the total contribution from this round's constraints.
            let round_sum: EF = constraints_in_round
                .iter()
                .zip(combination_coeffs)
                .map(|(constraint, &coeff)| {
                    // For each constraint, get its evaluation at the appropriate point `r`.
                    let single_eval = if constraint.defer_evaluation {
                        deferred_iter.next().unwrap()
                    } else if is_skip_round {
                        // ROUND 0 with SKIP: Use the special skip-aware evaluation.
                        // The constraint and the full challenge point are over the `num_variables` domain.
                        assert_eq!(constraint.weights.num_variables(), num_variables);
                        constraint
                            .weights
                            .compute_with_skip(&point_for_round, K_SKIP_SUMCHECK)
                    } else {
                        // STANDARD ROUND: Use the standard multilinear evaluation.
                        // The constraint and challenge point are over the (potentially smaller) domain.
                        assert_eq!(constraint.weights.num_variables(), num_variables);
                        constraint.weights.compute(&point_for_round)
                    };
                    // Multiply by its random combination coefficient.
                    single_eval * coeff
                })
                .sum();

            // Add this round's total contribution to the final value.
            value += round_sum;
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

#[cfg(test)]
mod tests {
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use proptest::prelude::*;
    use rand::{Rng, SeedableRng, rngs::SmallRng};

    use super::*;
    use crate::parameters::{
        FoldingFactor, MultivariateParameters, ProtocolParameters, errors::SecurityAssumption,
    };

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;
    type Perm = Poseidon2BabyBear<16>;

    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

    #[test]
    fn test_eval_constraints_poly_non_skip() {
        // -- Test Configuration --
        // We use 20 variables to ensure a non-trivial number of folding rounds.
        let num_vars = 20;
        // A constant folding factor of 5 is used.
        let folding_factor = FoldingFactor::Constant(5);
        // This configuration implies a 3-round folding schedule before the final polynomial:
        // Round 0: 20 vars -> 15 vars
        // Round 1: 15 vars -> 10 vars
        // Round 2: 10 vars ->  5 vars (final polynomial)

        // We will add a varying number of constraints in each round.
        let num_constraints_per_round = &[2, 3, 1];

        // Initialize a deterministic random number generator for reproducibility.
        let mut rng = SmallRng::seed_from_u64(0);

        // -- Cryptographic Primitives & Verifier Config --
        let perm = Perm::new_from_rng_128(&mut rng);
        let merkle_hash = MyHash::new(perm.clone());
        let merkle_compress = MyCompress::new(perm);
        let mv_params = MultivariateParameters::<EF>::new(num_vars);
        let whir_params = ProtocolParameters {
            folding_factor,
            merkle_hash,
            merkle_compress,
            univariate_skip: false,
            initial_statement: true,
            security_level: 90,
            pow_bits: 0,
            soundness_type: SecurityAssumption::UniqueDecoding,
            starting_log_inv_rate: 1,
            rs_domain_initial_reduction_factor: 1,
        };
        let params =
            WhirConfig::<EF, F, MyHash, MyCompress, MyChallenger>::new(mv_params, whir_params);
        let verifier = Verifier::new(&params);

        // -- Random Constraints and Challenges --
        // This block generates the inputs that the verifier would receive in a real proof.
        let mut statements = vec![];
        let mut alphas: Vec<EF> = vec![];
        let mut num_vars_at_round = num_vars;

        // Generate constraints and alpha challenges for each of the 3 rounds.
        for num_constraints in num_constraints_per_round {
            // Create a statement for the current domain size (20, then 15, then 10).
            let mut statement = Statement::new(num_vars_at_round);
            for _ in 0..*num_constraints {
                statement.add_constraint(
                    Weights::evaluation(MultilinearPoint::rand(&mut rng, num_vars_at_round)),
                    rng.random(),
                );
            }
            statements.push(statement);
            // Generate a random combination scalar (alpha) for this round.
            alphas.push(rng.random());
            // Shrink the number of variables for the next round.
            num_vars_at_round -= folding_factor.at_round(0);
        }

        // Assemble the data in the format that `eval_constraints_poly` expects.
        let round_constraints: Vec<_> = statements
            .iter()
            .zip(&alphas)
            .map(|(statement, &alpha)| {
                (
                    alpha.powers().collect_n(statement.constraints.len()),
                    statement.constraints.clone(),
                )
            })
            .collect();

        // Generate the final, full 20-dimensional challenge point `r`.
        let final_point = MultilinearPoint::rand(&mut rng, num_vars);

        // Calculate W(r) using the function under test
        //
        // This is the recursive method we want to validate.
        let result_from_eval_poly =
            verifier.eval_constraints_poly(&round_constraints, &[], &final_point);

        // Manually compute W(r) with explicit recursive evaluation
        let mut expected_result = EF::ZERO;

        // --- Contribution from Round 0 ---
        //
        // Combine the constraints for the first round using its alpha.
        let (w0_combined, _) = statements[0].combine::<F>(alphas[0]);
        // The evaluation point for this round is the full, unsliced 20-variable challenge point.
        let point_round0 = final_point.clone();
        // Add the contribution from this round.
        expected_result += w0_combined.evaluate(&point_round0);

        // --- Contribution from Round 1 ---
        //
        // Combine the constraints for the second round using its alpha.
        let (w1_combined, _) = statements[1].combine::<F>(alphas[1]);
        // The domain has shrunk. The evaluation point is the first 15 variables of the full point.
        let point_round1 = MultilinearPoint(final_point[..15].to_vec());
        // Add the contribution from this round.
        expected_result += w1_combined.evaluate(&point_round1);

        // --- Contribution from Round 2 ---
        //
        // Combine the constraints for the third round.
        let (w2_combined, _) = statements[2].combine::<F>(alphas[2]);
        // The domain shrinks again. The evaluation point is the first 10 variables of the full point.
        let point_round2 = MultilinearPoint(final_point[..10].to_vec());
        // Add the contribution from this round.
        expected_result += w2_combined.evaluate(&point_round2);

        // The result from the recursive function must match the materialized ground truth.
        assert_eq!(result_from_eval_poly, expected_result);
    }

    proptest! {
        #[test]
        fn prop_eval_constraints_poly_non_skip(
            (n, folding_factor_val) in (10..=20usize)
                .prop_flat_map(|n| (
                    Just(n),
                    2..=(n / 2)
                ))
        ) {
            // `Tracks the number of variables remaining before each round.
            let mut num_vars_current = n;
            // The folding factor is constant for all rounds.
            let folding_factor = FoldingFactor::Constant(folding_factor_val);
            // Will store the number of variables folded in each specific round.
            let mut folding_factors_vec = vec![];
            // We simulate the folding process to build the schedule.
            //
            // The protocol folds variables until 0 remain.
            while num_vars_current > 0 {
                // In each round, we fold `folding_factor_val` variables.
                //
                // If this would leave fewer than 0 variables, we fold just enough to reach 0.
                let num_to_fold = std::cmp::min(folding_factor_val, num_vars_current);
                // This check avoids an infinite loop if `num_vars_current` gets stuck.
                if num_to_fold == 0 { break; }
                // Record the number of variables folded in this round.
                folding_factors_vec.push(num_to_fold);
                // Decrease the variable count for the next round.
                num_vars_current -= num_to_fold;
            }
            // The total number of folding rounds.
            let num_rounds = folding_factors_vec.len();

            // Use a seeded RNG for a reproducible test run.
            let mut rng = SmallRng::seed_from_u64(0);
            // For each round, generate a random number of constraints (from 0 to 8).
            let num_constraints_per_round: Vec<usize> = (0..num_rounds)
                .map(|_| rng.random_range(0..=8))
                .collect();

            // -- Cryptographic Primitives & Verifier Config --
            // Set up the necessary cryptographic components for the `WhirConfig`.
            let perm = Perm::new_from_rng_128(&mut rng);
            let merkle_hash = MyHash::new(perm.clone());
            let merkle_compress = MyCompress::new(perm);

            // Define the top-level parameters for the protocol.
            let mv_params = MultivariateParameters::<EF>::new(n);
            let whir_params = ProtocolParameters {
                folding_factor,
                merkle_hash,
                merkle_compress,
                // This test is for the standard, non-skip case.
                univariate_skip: false,
                initial_statement: true,
                security_level: 90,
                pow_bits: 0,
                soundness_type: SecurityAssumption::UniqueDecoding,
                starting_log_inv_rate: 1,
                rs_domain_initial_reduction_factor: 1,
            };
            // Create the complete verifier configuration object.
            let params =
                WhirConfig::<EF, F, MyHash, MyCompress, MyChallenger>::new(mv_params, whir_params);
            let verifier = Verifier::new(&params);

            // -- Random Constraints and Challenges --
            // This block generates the inputs that the verifier would receive in a real proof.
            let mut statements = vec![];
            let mut alphas: Vec<EF> = vec![];
            num_vars_current = n;

            // Generate the statements and alpha challenges for each round based on our dynamic schedule.
            for i in 0..num_rounds {
                // Create a statement for the current domain size (e.g., 20, then 15, then 10...).
                let mut statement = Statement::new(num_vars_current);
                // Add the random number of constraints for this round.
                for _ in 0..num_constraints_per_round[i] {
                    statement.add_constraint(
                        Weights::evaluation(MultilinearPoint::rand(&mut rng, num_vars_current)),
                        rng.random(),
                    );
                }
                statements.push(statement);
                // Generate a random combination scalar (alpha) for this round.
                alphas.push(rng.random());
                // Shrink the number of variables for the next round.
                num_vars_current -= folding_factors_vec[i];
            }

            // Assemble the final data structure in the format required by `eval_constraints_poly`.
            let round_constraints: Vec<_> = statements
                .iter()
                .zip(&alphas)
                .map(|(s, &a)| (a.powers().collect_n(s.constraints.len()), s.constraints.clone()))
                .collect();

            // Generate the final, full n-dimensional challenge point `r`.
            let final_point = MultilinearPoint::rand(&mut rng, n);


            // Calculate W(r) using the function under test
            //
            // This is the recursive method we want to validate.
            let result_from_eval_poly =
                verifier.eval_constraints_poly(&round_constraints, &[], &final_point);


            // Calculate W(r) by materializing and evaluating round-by-round
            //
            // This simpler, more direct method serves as our ground truth.
            let mut expected_result = EF::ZERO;
            let mut num_vars_for_round = n;

            // Loop through each round to calculate its contribution to the final evaluation.
            for i in 0..num_rounds {
                // Combine this round's constraints into a single polynomial `W_i(X)`.
                let (w_combined, _) = statements[i].combine::<F>(alphas[i]);

                // Create the challenge point for this round by taking a prefix slice of the full point `r`.
                let point_for_round = MultilinearPoint(final_point[..num_vars_for_round].to_vec());
                // Evaluate `W_i` at the correctly sliced point.
                let w_eval = w_combined.evaluate(&point_for_round);

                // Add this round's contribution to the total.
                expected_result += w_eval;

                // Shrink the number of variables for the next round's slice.
                num_vars_for_round -= folding_factors_vec[i];
            }

            // The result from the recursive function must match the materialized ground truth.
            prop_assert_eq!(result_from_eval_poly, expected_result);
        }
    }

    #[test]
    fn test_eval_constraints_poly_with_skip() {
        // -- Test Configuration --
        //
        // We use 20 variables to ensure a non-trivial number of folding rounds.
        let num_vars = 20;

        // We use a constant folding factor of `K_SKIP_SUMCHECK` to trigger the skip.
        let folding_factor = FoldingFactor::Constant(K_SKIP_SUMCHECK);

        // This configuration implies a folding schedule:
        // Round 0: 20 vars --(skip 5)--> 15 vars
        // Round 1: 15 vars --(fold 5)--> 10 vars
        // Round 2: 11 vars --(fold 5)-->  5 vars
        // Round 3:  7 vars --(fold 5)-->  0 vars (final polynomial)
        let num_constraints_per_round = &[2, 3, 1, 2];
        let num_rounds = num_constraints_per_round.len();

        // Initialize a deterministic RNG for reproducibility.
        let mut rng = SmallRng::seed_from_u64(0);

        // -- Cryptographic Primitives & Verifier Config --
        let perm = Perm::new_from_rng_128(&mut rng);
        let merkle_hash = MyHash::new(perm.clone());
        let merkle_compress = MyCompress::new(perm);
        let mv_params = MultivariateParameters::<EF>::new(num_vars);
        let whir_params = ProtocolParameters {
            folding_factor,
            merkle_hash,
            merkle_compress,
            // This test is for the skip case.
            univariate_skip: true,
            initial_statement: true,
            security_level: 90,
            pow_bits: 0,
            soundness_type: SecurityAssumption::UniqueDecoding,
            starting_log_inv_rate: 1,
            rs_domain_initial_reduction_factor: 1,
        };
        let params =
            WhirConfig::<EF, F, MyHash, MyCompress, MyChallenger>::new(mv_params, whir_params);
        let verifier = Verifier::new(&params);

        // -- Random Constraints and Challenges --
        let mut statements = vec![];
        let mut alphas: Vec<EF> = vec![];
        let mut num_vars_at_round = num_vars;

        // Generate constraints and alpha challenges for each round.
        for (i, &num_constraints) in num_constraints_per_round
            .iter()
            .enumerate()
            .take(num_rounds)
        {
            let mut statement = Statement::new(num_vars_at_round);
            for _ in 0..num_constraints {
                statement.add_constraint(
                    Weights::evaluation(MultilinearPoint::rand(&mut rng, num_vars_at_round)),
                    rng.random(),
                );
            }
            statements.push(statement);
            alphas.push(rng.random());
            num_vars_at_round -= folding_factor.at_round(i);
        }

        // Assemble the data in the format that `eval_constraints_poly` expects.
        let round_constraints: Vec<_> = statements
            .iter()
            .zip(&alphas)
            .map(|(s, &a)| {
                (
                    a.powers().collect_n(s.constraints.len()),
                    s.constraints.clone(),
                )
            })
            .collect();

        // For a skip protocol, the verifier's final challenge object has a special
        // structure with (n - k_skip) + 1 elements, not n.
        let final_point = MultilinearPoint::<EF>::rand(&mut rng, (num_vars - K_SKIP_SUMCHECK) + 1);

        // Calculate W(r) using the function under test
        let result_from_eval_poly =
            verifier.eval_constraints_poly(&round_constraints, &[], &final_point);

        // Manually compute W(r) with explicit recursive evaluation
        let mut expected_result = EF::ZERO;

        // --- Contribution from Round 0 (Skip Round) ---
        //
        // Combine the constraints for the first round into a single polynomial, W_0(X).
        let (w0_combined, _) = statements[0].combine::<F>(alphas[0]);

        // To evaluate W_0(r) using skip semantics, we follow the same pipeline as the prover:
        // a) Deconstruct the special challenge object `r` into its components:
        // - `r_rest`,
        // - `r_skip`.
        let num_remaining = num_vars - K_SKIP_SUMCHECK;
        let r_rest = MultilinearPoint(final_point[..num_remaining].to_vec());
        let r_skip = *final_point.last().expect("skip challenge must be present");

        // b) Reshape the W_0(X) evaluation table into a matrix.
        let w0_mat = RowMajorMatrix::new(w0_combined.to_vec(), 1 << num_remaining);

        // c) "Fold" the skipped variables by interpolating the matrix at `r_skip`.
        let folded_row = interpolate_subgroup(&w0_mat, r_skip);

        // d) Evaluate the resulting smaller polynomial at the remaining challenges `r_rest`.
        let w0_eval = EvaluationsList::new(folded_row).evaluate(&r_rest);
        expected_result += w0_eval;

        // --- Contribution from Round 1 (Standard Round) ---
        let (w1_combined, _) = statements[1].combine::<F>(alphas[1]);
        // For subsequent rounds, the evaluation point is a prefix slice of the `r_rest` challenges.
        let point_round1 = MultilinearPoint(r_rest[..statements[1].num_variables()].to_vec());
        expected_result += w1_combined.evaluate(&point_round1);

        // --- Contribution from Round 2 (Standard Round) ---
        let (w2_combined, _) = statements[2].combine::<F>(alphas[2]);
        let point_round2 = MultilinearPoint(r_rest[..statements[2].num_variables()].to_vec());
        expected_result += w2_combined.evaluate(&point_round2);

        // --- Contribution from Round 3 (Standard Round) ---
        let (w3_combined, _) = statements[3].combine::<F>(alphas[3]);
        let point_round3 = MultilinearPoint(r_rest[..statements[3].num_variables()].to_vec());
        expected_result += w3_combined.evaluate(&point_round3);

        // The result from the recursive function must match the materialized ground truth.
        assert_eq!(result_from_eval_poly, expected_result);
    }

    proptest! {
        #[test]
        fn prop_eval_constraints_poly_with_skip(
            (n, standard_folding_factor) in (10..=20usize)
                .prop_flat_map(|n| (
                    Just(n),
                    2..=((n - K_SKIP_SUMCHECK) / 2).max(2)
                ))
        ) {
            // Tracks the number of variables remaining before each round.
            let mut num_vars_current = n;
            // - The first round folds K_SKIP_SUMCHECK variables,
            // - Subsequent rounds use the random factor.
            let folding_factor = FoldingFactor::ConstantFromSecondRound(K_SKIP_SUMCHECK, standard_folding_factor);
            // Will store the number of variables folded in each specific round.
            let mut folding_factors_vec = vec![];

            // We simulate the folding process to build the schedule.
            while num_vars_current > 0 {
                let num_to_fold = folding_factor.at_round(folding_factors_vec.len());
                // Ensure we don't overshoot the target of 0 remaining variables.
                let effective_num_to_fold = std::cmp::min(num_to_fold, num_vars_current);
                if effective_num_to_fold == 0 { break; }
                folding_factors_vec.push(effective_num_to_fold);
                num_vars_current -= effective_num_to_fold;
            }
            let num_rounds = folding_factors_vec.len();

            // Use a seeded RNG for a reproducible test run.
            let mut rng = SmallRng::seed_from_u64(0);
            // For each round, generate a random number of constraints (from 0 to 2).
            let num_constraints_per_round: Vec<usize> = (0..num_rounds)
                .map(|_| rng.random_range(0..=2))
                .collect();

            // -- Cryptographic Primitives & Verifier Config --
            let perm = Perm::new_from_rng_128(&mut rng);
            let merkle_hash = MyHash::new(perm.clone());
            let merkle_compress = MyCompress::new(perm);
            let mv_params = MultivariateParameters::<EF>::new(n);
            let whir_params = ProtocolParameters {
                folding_factor,
                merkle_hash,
                merkle_compress,
                // This test is for the skip case.
                univariate_skip: true,
                initial_statement: true,
                security_level: 90,
                pow_bits: 0,
                soundness_type: SecurityAssumption::UniqueDecoding,
                starting_log_inv_rate: 1,
                rs_domain_initial_reduction_factor: 1,
            };
            let params =
                WhirConfig::<EF, F, MyHash, MyCompress, MyChallenger>::new(mv_params, whir_params);
            let verifier = Verifier::new(&params);

            // -- Random Constraints and Challenges --
            let mut statements = vec![];
            let mut alphas: Vec<EF> = vec![];
            num_vars_current = n;

            // Generate the statements and alphas for each round based on our dynamic schedule.
            for i in 0..num_rounds {
                let mut statement = Statement::new(num_vars_current);
                for _ in 0..num_constraints_per_round[i] {
                    statement.add_constraint(
                        Weights::evaluation(MultilinearPoint::rand(&mut rng, num_vars_current)),
                        rng.random(),
                    );
                }
                statements.push(statement);
                alphas.push(rng.random());
                num_vars_current -= folding_factors_vec[i];
            }

            // Assemble the data for the function call.
            let round_constraints: Vec<_> = statements
                .iter()
                .zip(&alphas)
                .map(|(s, &a)| (a.powers().collect_n(s.constraints.len()), s.constraints.clone()))
                .collect();

            // For a skip protocol, the verifier's final challenge object has a special
            // structure with `(n - k_skip) + 1` elements, not `n`.
            let final_point =
                MultilinearPoint::rand(&mut rng, (n - K_SKIP_SUMCHECK) + 1);


            // Calculate W(r) using the function under test
            let result_from_eval_poly =
                verifier.eval_constraints_poly(&round_constraints, &[], &final_point);


            // Calculate W(r) by materializing and evaluating round-by-round
            let mut expected_result = EF::ZERO;
            let mut num_vars_for_round = n;

            // Contribution from Round 0 (Skip Round)
            //
            // Combine the constraints for the first round using its alpha.
            let (w0_combined, _) = statements[0].combine::<F>(alphas[0]);
            // Evaluate W_0(r) using the manual skip evaluation pipeline.
            let num_remaining = n - K_SKIP_SUMCHECK;
            let r_rest = MultilinearPoint(final_point[..num_remaining].to_vec());
            let r_skip = *final_point.last().expect("skip challenge must be present");
            let w0_mat = RowMajorMatrix::new(w0_combined.to_vec(), 1 << num_remaining);
            let folded_row = interpolate_subgroup(&w0_mat, r_skip);
            let w0_eval = EvaluationsList::new(folded_row).evaluate(&r_rest);
            expected_result += w0_eval;
            num_vars_for_round -= folding_factors_vec[0];

            // Contribution from Subsequent Rounds (Standard)
            //
            // Loop through the remaining rounds.
            for i in 1..num_rounds {
                // Combine this round's constraints into a single polynomial `W_i(X)`.
                let (w_combined, _) = statements[i].combine::<F>(alphas[i]);

                // Evaluate `W_i` at the correct prefix slice of the `r_rest` challenges.
                let point_for_round = MultilinearPoint(r_rest[..num_vars_for_round].to_vec());
                let w_eval = w_combined.evaluate(&point_for_round);

                // Add this round's contribution to the total.
                expected_result += w_eval;

                // Shrink the number of variables for the next round's slice.
                num_vars_for_round -= folding_factors_vec[i];
            }

            // The result from the recursive function must match the materialized ground truth.
            prop_assert_eq!(result_from_eval_poly, expected_result);
        }
    }
}
