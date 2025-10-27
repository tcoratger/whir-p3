use std::ops::Deref;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_commit::{ExtensionMmcs, Mmcs};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, TwoAdicField};
use p3_interpolation::interpolate_subgroup;
use p3_matrix::{
    Matrix,
    dense::{DenseMatrix, RowMajorMatrix},
};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use round_state::RoundState;
use serde::{Deserialize, Serialize};
use tracing::{info_span, instrument};

use super::{committer::Witness, constraints::statement::EqStatement, parameters::WhirConfig};
use crate::{
    constant::K_SKIP_SUMCHECK,
    fiat_shamir::{errors::FiatShamirError, prover::ProverState},
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    whir::{
        constraints::{evaluator::Constraint, sel_statement::SelectStatement},
        utils::get_challenge_stir_queries,
    },
};

pub mod round_state;

pub type Proof<W, const DIGEST_ELEMS: usize> = Vec<Vec<[W; DIGEST_ELEMS]>>;
pub type Leafs<F> = Vec<Vec<F>>;

#[derive(Debug)]
pub struct Prover<'a, EF, F, H, C, Challenger, Dft>(
    /// Reference to the protocol configuration shared across prover components.
    pub &'a WhirConfig<EF, F, H, C, Challenger, Dft>,
)
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    Dft: TwoAdicSubgroupDft<F>;

impl<EF, F, H, C, Challenger, Dft> Deref for Prover<'_, EF, F, H, C, Challenger, Dft>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    Dft: TwoAdicSubgroupDft<F>,
{
    type Target = WhirConfig<EF, F, H, C, Challenger, Dft>;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl<EF, F, H, C, Challenger, Dft> Prover<'_, EF, F, H, C, Challenger, Dft>
where
    F: TwoAdicField + Ord,
    EF: ExtensionField<F> + TwoAdicField,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    Dft: TwoAdicSubgroupDft<F>,
{
    /// Validates that the total number of variables expected by the prover configuration
    /// matches the number implied by the folding schedule and the final rounds.
    ///
    /// This ensures that the recursive folding in the sumcheck protocol terminates
    /// precisely at the expected number of final variables.
    ///
    /// # Returns
    /// `true` if the parameter configuration is consistent, `false` otherwise.
    const fn validate_parameters(&self) -> bool {
        self.0.num_variables
            == self.0.folding_factor.total_number(self.0.n_rounds()) + self.0.final_sumcheck_rounds
    }

    /// Validates that the public statement is compatible with the configured number of variables.
    ///
    /// Ensures the following:
    /// - The number of variables in the statement matches the prover's expectations
    /// - If no initial statement is used, the statement must be empty
    ///
    /// # Parameters
    /// - `statement`: The public constraints that the prover will use
    ///
    /// # Returns
    /// `true` if the statement structure is valid for this protocol instance.
    const fn validate_statement(&self, statement: &EqStatement<EF>) -> bool {
        statement.num_variables() == self.0.num_variables
            && (self.0.initial_statement || statement.is_empty())
    }

    /// Validates that the witness satisfies the structural requirements of the WHIR prover.
    ///
    /// Checks the following conditions:
    /// - The number of OOD (out-of-domain) points equals the number of OOD answers
    /// - If no initial statement is used, the OOD data must be empty
    /// - The multilinear witness polynomial must match the expected number of variables
    ///
    /// # Parameters
    /// - `witness`: The private witness to be verified for structural consistency
    ///
    /// # Returns
    /// `true` if the witness structure matches expectations.
    ///
    /// # Panics
    /// - Panics if OOD lengths are inconsistent
    /// - Panics if OOD data is non-empty despite `initial_statement = false`
    const fn validate_witness<const DIGEST_ELEMS: usize>(
        &self,
        witness: &Witness<EF, F, DenseMatrix<F>, DIGEST_ELEMS>,
    ) -> bool {
        if !self.0.initial_statement {
            assert!(witness.ood_statement.is_empty());
        }
        witness.polynomial.num_variables() == self.0.num_variables
    }

    /// Executes the full WHIR prover protocol to produce the proof.
    ///
    /// This function takes the public statement and private witness, performs the
    /// multi-round sumcheck-based polynomial folding protocol using DFTs, and returns
    /// a proof that the witness satisfies the statement.
    ///
    /// The proof includes:
    /// - Merkle authentication paths for each round's polynomial commitments
    /// - Final evaluations of the public linear statement constraints at a random point
    ///
    /// # Parameters
    /// - `dft`: A DFT backend used for evaluations
    /// - `prover_state`: Mutable prover state used across rounds (transcript, randomness, etc.)
    /// - `statement`: The public input, consisting of linear or nonlinear constraints
    /// - `witness`: The private witness satisfying the constraints, including committed values
    ///
    /// # Returns
    /// - The final random evaluation point used to evaluate deferred constraints
    /// - The list of evaluations of all deferred constraints at that point
    ///
    /// # Errors
    /// Returns an error if the witness or statement are invalid, or if a round fails.
    #[instrument(skip_all)]
    pub fn prove<const DIGEST_ELEMS: usize>(
        &self,
        prover_state: &mut ProverState<F, EF, Challenger>,
        statement: EqStatement<EF>,
        witness: Witness<EF, F, DenseMatrix<F>, DIGEST_ELEMS>,
    ) -> Result<MultilinearPoint<EF>, FiatShamirError>
    where
        H: CryptographicHasher<F, [F; DIGEST_ELEMS]>
            + CryptographicHasher<F::Packing, [F::Packing; DIGEST_ELEMS]>
            + Sync,
        C: PseudoCompressionFunction<[F; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[F::Packing; DIGEST_ELEMS], 2>
            + Sync,
        [F; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        // Validate parameters
        assert!(
            self.validate_parameters()
                && self.validate_statement(&statement)
                && self.validate_witness(&witness),
            "Invalid prover parameters, statement, or witness"
        );

        // Initialize the round state with inputs and initial polynomial data
        let mut round_state =
            RoundState::initialize_first_round_state(self, prover_state, statement, witness)?;

        // Run the WHIR protocol round-by-round
        for round in 0..=self.n_rounds() {
            self.round(round, prover_state, &mut round_state)?;
        }

        // Reverse the vector of verifier challenges (used as evaluation point)
        //
        // These challenges were pushed in round order; we reverse them to use as a single
        // evaluation point for final statement consistency checks.
        round_state.randomness_vec.reverse();
        let constraint_eval = MultilinearPoint::new(round_state.randomness_vec);

        Ok(constraint_eval)
    }

    #[instrument(skip_all, fields(round_number = round_index, log_size = self.num_variables - self.folding_factor.total_number(round_index)))]
    #[allow(clippy::too_many_lines)]
    fn round<const DIGEST_ELEMS: usize>(
        &self,
        round_index: usize,
        prover_state: &mut ProverState<F, EF, Challenger>,
        round_state: &mut RoundState<EF, F, F, DenseMatrix<F>, DIGEST_ELEMS>,
    ) -> Result<(), FiatShamirError>
    where
        H: CryptographicHasher<F, [F; DIGEST_ELEMS]>
            + CryptographicHasher<F::Packing, [F::Packing; DIGEST_ELEMS]>
            + Sync,
        C: PseudoCompressionFunction<[F; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[F::Packing; DIGEST_ELEMS], 2>
            + Sync,
        [F; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        let folded_evaluations = &round_state.sumcheck_prover.evals;
        let num_variables = self.num_variables - self.folding_factor.total_number(round_index);
        assert_eq!(num_variables, folded_evaluations.num_variables());

        // Base case: final round reached
        if round_index == self.n_rounds() {
            return self.final_round(round_index, prover_state, round_state);
        }

        let round_params = &self.round_parameters[round_index];

        // Compute the folding factors for later use
        let folding_factor_next = self.folding_factor.at_round(round_index + 1);

        // Compute polynomial evaluations and build Merkle tree
        let domain_reduction = 1 << self.rs_reduction_factor(round_index);
        let new_domain_size = round_state.domain_size / domain_reduction;
        let inv_rate = new_domain_size / folded_evaluations.num_evals();

        // Pad evaluation vector with zeros
        let n = folded_evaluations.num_evals();
        let padded = info_span!("repeating evals").in_scope(|| {
            let mut padded = EF::zero_vec(n * inv_rate);
            padded[..n].copy_from_slice(folded_evaluations.as_slice());
            RowMajorMatrix::new(padded, 1 << folding_factor_next)
        });

        // Perform DFT on the padded evaluations matrix
        let folded_matrix = info_span!("dft", height = padded.height(), width = padded.width())
            .in_scope(|| self.dft.dft_algebra_batch(padded).to_row_major_matrix());

        let mmcs = MerkleTreeMmcs::<F::Packing, F::Packing, H, C, DIGEST_ELEMS>::new(
            self.merkle_hash.clone(),
            self.merkle_compress.clone(),
        );
        let extension_mmcs = ExtensionMmcs::new(mmcs.clone());
        let (root, prover_data) =
            info_span!("commit matrix").in_scope(|| extension_mmcs.commit_matrix(folded_matrix));
        prover_state.add_base_scalars(root.as_ref());

        // Handle OOD (Out-Of-Domain) samples
        let mut ood_statement = EqStatement::initialize(num_variables);
        (0..round_params.ood_samples).for_each(|_| {
            let point =
                MultilinearPoint::expand_from_univariate(prover_state.sample(), num_variables);
            let eval = folded_evaluations.evaluate(&point);
            prover_state.add_extension_scalar(eval);
            ood_statement.add_evaluated_constraint(point, eval);
        });

        // CRITICAL: Perform proof-of-work grinding to finalize the transcript before querying.
        //
        // This is a crucial security step to prevent a malicious prover from influencing the
        // verifier's challenges.
        //
        // The verifier's query locations (the `stir_challenges`) are generated based on the
        // current transcript state, which includes the prover's polynomial commitment (the Merkle
        // root) for this round. Without grinding, a prover could repeatedly try different
        // commitments until they find one that results in "easy" queries, breaking soundness.
        //
        // By forcing the prover to perform this expensive proof-of-work *after* committing but
        // *before* receiving the queries, we make it computationally infeasible to "shop" for
        // favorable challenges. The grinding effectively "locks in" the prover's commitment.
        prover_state.pow_grinding(round_params.pow_bits);

        // STIR Queries
        let stir_challenges_indexes = get_challenge_stir_queries(
            round_state.domain_size,
            self.folding_factor.at_round(round_index),
            round_params.num_queries,
            prover_state,
        )?;
        let stir_vars = stir_challenges_indexes
            .iter()
            .map(|&i| round_state.next_domain_gen.exp_u64(i as u64))
            .collect::<Vec<_>>();
        let mut stir_statement = SelectStatement::initialize(num_variables);

        // Collect Merkle proofs for stir queries
        match &round_state.merkle_prover_data {
            None => {
                let mut answers = Vec::with_capacity(stir_challenges_indexes.len());
                let mut merkle_proofs = Vec::with_capacity(stir_challenges_indexes.len());
                for challenge in &stir_challenges_indexes {
                    let commitment =
                        mmcs.open_batch(*challenge, &round_state.commitment_merkle_prover_data);
                    answers.push(commitment.opened_values[0].clone());
                    merkle_proofs.push(commitment.opening_proof);
                }

                // merkle leaves
                for answer in &answers {
                    prover_state.hint_base_scalars(answer);
                }

                // merkle authentication proof
                for merkle_proof in &merkle_proofs {
                    for digest in merkle_proof {
                        prover_state.hint_base_scalars(digest);
                    }
                }

                // Determine if this is the special first round where the univariate skip is applied.
                let is_skip_round = self.initial_statement
                    && round_index == 0
                    && self.univariate_skip
                    && self.folding_factor.at_round(0) >= K_SKIP_SUMCHECK;

                // Process each set of evaluations retrieved from the Merkle tree openings.
                for (answer, var) in answers.iter().zip(stir_vars.into_iter()) {
                    let evals = EvaluationsList::new(answer.clone());
                    // Fold the polynomial represented by the `answer` evaluations using the verifier's challenge.
                    // The evaluation method depends on whether this is a "skip round" or a "standard round".
                    if is_skip_round {
                        // Case 1: Univariate Skip Round Evaluation
                        //

                        // The challenges for the remaining (non-skipped) variables.
                        let num_remaining_vars = evals.num_variables() - K_SKIP_SUMCHECK;

                        // The width of the matrix corresponds to the number of remaining variables.
                        let width = 1 << num_remaining_vars;

                        // Reshape the `answer` evaluations into the `2^k x 2^(n-k)` matrix format.
                        let mat = evals.into_mat(width);

                        // For a skip round, `folding_randomness` is the special `(n-k)+1` challenge object.
                        let r_all = round_state.folding_randomness.clone();

                        // Deconstruct the special challenge object `r_all`.
                        //
                        // The last element is the single challenge for the `k_skip` variables being folded.
                        let r_skip = *r_all
                            .last_variable()
                            .expect("skip challenge must be present");
                        // The first `n - k_skip` elements are the challenges for the remaining variables.
                        let r_rest = r_all.get_subpoint_over_range(0..num_remaining_vars);

                        // Perform the two-stage skip-aware evaluation:
                        //
                        // "Fold" the skipped variables by interpolating the matrix at `r_skip`.
                        let folded_row = interpolate_subgroup(&mat, r_skip);
                        // Evaluate the resulting smaller polynomial at the remaining challenges `r_rest`.
                        let eval = EvaluationsList::new(folded_row).evaluate(&r_rest);
                        stir_statement.add_constraint(var, eval);
                    } else {
                        // Case 2: Standard Sumcheck Round
                        //
                        // The `answer` represents a standard multilinear polynomial.

                        // Perform a standard multilinear evaluation at the full challenge point `r`.
                        let eval = evals.evaluate(&round_state.folding_randomness);
                        stir_statement.add_constraint(var, eval);
                    }
                }
            }
            Some(data) => {
                let mut answers = Vec::with_capacity(stir_challenges_indexes.len());
                let mut merkle_proofs = Vec::with_capacity(stir_challenges_indexes.len());
                for challenge in &stir_challenges_indexes {
                    let commitment = extension_mmcs.open_batch(*challenge, data);
                    answers.push(commitment.opened_values[0].clone());
                    merkle_proofs.push(commitment.opening_proof);
                }

                // merkle leaves
                for answer in &answers {
                    prover_state.hint_extension_scalars(answer);
                }

                // merkle authentication proof
                for merkle_proof in &merkle_proofs {
                    for digest in merkle_proof {
                        prover_state.hint_base_scalars(digest);
                    }
                }

                // Process each set of evaluations retrieved from the Merkle tree openings.
                for (answer, var) in answers.iter().zip(stir_vars.into_iter()) {
                    // Wrap the evaluations to represent the polynomial.
                    let evals = EvaluationsList::new(answer.clone());
                    // Perform a standard multilinear evaluation at the full challenge point `r`.
                    let eval = evals.evaluate(&round_state.folding_randomness);
                    stir_statement.add_constraint(var, eval);
                }
            }
        }

        let constraint = Constraint::new(prover_state.sample(), ood_statement, stir_statement);
        let folding_randomness = round_state.sumcheck_prover.compute_sumcheck_polynomials(
            prover_state,
            folding_factor_next,
            round_params.folding_pow_bits,
            Some(constraint),
        );

        let start_idx = self.folding_factor.total_number(round_index);
        let dst_randomness =
            &mut round_state.randomness_vec[start_idx..][..folding_randomness.num_variables()];

        for (dst, src) in dst_randomness
            .iter_mut()
            .zip(folding_randomness.iter().rev())
        {
            *dst = *src;
        }

        // Update round state
        round_state.domain_size = new_domain_size;
        round_state.next_domain_gen =
            F::two_adic_generator(new_domain_size.ilog2() as usize - folding_factor_next);
        round_state.folding_randomness = folding_randomness;
        round_state.merkle_prover_data = Some(prover_data);

        Ok(())
    }

    #[instrument(skip_all)]
    fn final_round<const DIGEST_ELEMS: usize>(
        &self,
        round_index: usize,
        prover_state: &mut ProverState<F, EF, Challenger>,
        round_state: &mut RoundState<EF, F, F, DenseMatrix<F>, DIGEST_ELEMS>,
    ) -> Result<(), FiatShamirError>
    where
        H: CryptographicHasher<F, [F; DIGEST_ELEMS]>
            + CryptographicHasher<F::Packing, [F::Packing; DIGEST_ELEMS]>
            + Sync,
        C: PseudoCompressionFunction<[F; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[F::Packing; DIGEST_ELEMS], 2>
            + Sync,
        [F; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        // Directly send coefficients of the polynomial to the verifier.
        prover_state.add_extension_scalars(round_state.sumcheck_prover.evals.as_slice());

        // CRITICAL: Perform proof-of-work grinding to finalize the transcript before querying.
        //
        // This is a crucial security step to prevent a malicious prover from influencing the
        // verifier's challenges.
        //
        // The verifier's query locations (the `stir_challenges`) are generated based on the
        // current transcript state, which includes the prover's polynomial commitment (the Merkle
        // root) for this round. Without grinding, a prover could repeatedly try different
        // commitments until they find one that results in "easy" queries, breaking soundness.
        //
        // By forcing the prover to perform this expensive proof-of-work *after* committing but
        // *before* receiving the queries, we make it computationally infeasible to "shop" for
        // favorable challenges. The grinding effectively "locks in" the prover's commitment.
        prover_state.pow_grinding(self.final_pow_bits);

        // Final verifier queries and answers. The indices are over the folded domain.
        let final_challenge_indexes = get_challenge_stir_queries(
            // The size of the original domain before folding
            round_state.domain_size,
            // The folding factor we used to fold the previous polynomial
            self.folding_factor.at_round(round_index),
            // Number of final verification queries
            self.final_queries,
            prover_state,
        )?;

        // Every query requires opening these many in the previous Merkle tree
        let mmcs = MerkleTreeMmcs::<F::Packing, F::Packing, H, C, DIGEST_ELEMS>::new(
            self.merkle_hash.clone(),
            self.merkle_compress.clone(),
        );
        let extension_mmcs = ExtensionMmcs::new(mmcs.clone());

        match &round_state.merkle_prover_data {
            None => {
                let mut answers = Vec::with_capacity(final_challenge_indexes.len());
                let mut merkle_proofs = Vec::with_capacity(final_challenge_indexes.len());

                for challenge in final_challenge_indexes {
                    let commitment =
                        mmcs.open_batch(challenge, &round_state.commitment_merkle_prover_data);
                    answers.push(commitment.opened_values[0].clone());
                    merkle_proofs.push(commitment.opening_proof);
                }

                // merkle leaves
                for answer in &answers {
                    prover_state.hint_base_scalars(answer);
                }

                // merkle authentication proof
                for merkle_proof in &merkle_proofs {
                    for digest in merkle_proof {
                        prover_state.hint_base_scalars(digest);
                    }
                }
            }

            Some(data) => {
                let mut answers = Vec::with_capacity(final_challenge_indexes.len());
                let mut merkle_proofs = Vec::with_capacity(final_challenge_indexes.len());
                for challenge in final_challenge_indexes {
                    let commitment = extension_mmcs.open_batch(challenge, data);
                    answers.push(commitment.opened_values[0].clone());
                    merkle_proofs.push(commitment.opening_proof);
                }

                // merkle leaves
                for answer in &answers {
                    prover_state.hint_extension_scalars(answer);
                }

                // merkle authentication proof
                for merkle_proof in &merkle_proofs {
                    for digest in merkle_proof {
                        prover_state.hint_base_scalars(digest);
                    }
                }
            }
        }

        // Run final sumcheck if required
        if self.final_sumcheck_rounds > 0 {
            let final_folding_randomness =
                round_state.sumcheck_prover.compute_sumcheck_polynomials(
                    prover_state,
                    self.final_sumcheck_rounds,
                    self.final_folding_pow_bits,
                    None,
                );
            let start_idx = self.folding_factor.total_number(round_index);
            let rand_dst = &mut round_state.randomness_vec
                [start_idx..start_idx + final_folding_randomness.num_variables()];

            for (dst, src) in rand_dst
                .iter_mut()
                .zip(final_folding_randomness.iter().rev())
            {
                *dst = *src;
            }
        }

        Ok(())
    }
}
