use std::ops::Deref;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_commit::{ExtensionMmcs, Mmcs};
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use round::RoundState;
use serde::{Deserialize, Serialize};
use tracing::{info_span, instrument};

use super::{committer::Witness, parameters::WhirConfig, statement::Statement};
use crate::{
    dft::EvalsDft,
    fiat_shamir::{errors::ProofResult, prover::ProverState},
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    utils::parallel_repeat,
    whir::{
        parameters::RoundConfig,
        utils::{get_challenge_stir_queries, sample_ood_points},
    },
};

pub mod round;

pub type Proof<W, const DIGEST_ELEMS: usize> = Vec<Vec<[W; DIGEST_ELEMS]>>;
pub type Leafs<F> = Vec<Vec<F>>;

#[derive(Debug)]
pub struct Prover<'a, EF, F, H, C, Challenger>(
    /// Reference to the protocol configuration shared across prover components.
    pub &'a WhirConfig<EF, F, H, C, Challenger>,
)
where
    F: Field,
    EF: ExtensionField<F>;

impl<EF, F, H, C, Challenger> Deref for Prover<'_, EF, F, H, C, Challenger>
where
    F: Field,
    EF: ExtensionField<F>,
{
    type Target = WhirConfig<EF, F, H, C, Challenger>;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl<EF, F, H, C, Challenger> Prover<'_, EF, F, H, C, Challenger>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    /// Validates that the total number of variables expected by the prover configuration
    /// matches the number implied by the folding schedule and the final rounds.
    ///
    /// This ensures that the recursive folding in the sumcheck protocol terminates
    /// precisely at the expected number of final variables.
    ///
    /// # Returns
    /// `true` if the parameter configuration is consistent, `false` otherwise.
    fn validate_parameters(&self) -> bool {
        self.mv_parameters.num_variables
            == self.folding_factor.total_number(self.n_rounds()) + self.final_sumcheck_rounds
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
    fn validate_statement(&self, statement: &Statement<EF>) -> bool {
        statement.num_variables() == self.mv_parameters.num_variables
            && (self.initial_statement || statement.constraints.is_empty())
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
    fn validate_witness<const DIGEST_ELEMS: usize>(
        &self,
        witness: &Witness<EF, F, DenseMatrix<F>, DIGEST_ELEMS>,
    ) -> bool {
        assert_eq!(witness.ood_points.len(), witness.ood_answers.len());
        if !self.initial_statement {
            assert!(witness.ood_points.is_empty());
        }
        witness.polynomial.num_variables() == self.mv_parameters.num_variables
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
        dft: &EvalsDft<F>,
        prover_state: &mut ProverState<F, EF, Challenger>,
        statement: Statement<EF>,
        witness: Witness<EF, F, DenseMatrix<F>, DIGEST_ELEMS>,
    ) -> ProofResult<(MultilinearPoint<EF>, Vec<EF>)>
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
            self.round(round, dft, prover_state, &mut round_state)?;
        }

        // Reverse the vector of verifier challenges (used as evaluation point)
        //
        // These challenges were pushed in round order; we reverse them to use as a single
        // evaluation point for final statement consistency checks.
        round_state.randomness_vec.reverse();
        let constraint_eval = MultilinearPoint(round_state.randomness_vec);

        // Hints for deferred constraints
        let deferred = round_state
            .statement
            .constraints
            .iter()
            .filter(|constraint| constraint.defer_evaluation)
            .map(|constraint| constraint.weights.compute(&constraint_eval))
            .collect::<Vec<_>>();

        prover_state.hint_extension_scalars(&deferred);

        Ok((constraint_eval, deferred))
    }

    #[instrument(skip_all, fields(round_number = round_index, log_size = self.mv_parameters.num_variables - self.folding_factor.total_number(round_index)))]
    #[allow(clippy::too_many_lines)]
    fn round<const DIGEST_ELEMS: usize>(
        &self,
        round_index: usize,
        dft: &EvalsDft<F>,
        prover_state: &mut ProverState<F, EF, Challenger>,
        round_state: &mut RoundState<EF, F, F, DenseMatrix<F>, DIGEST_ELEMS>,
    ) -> ProofResult<()>
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
        let num_variables =
            self.mv_parameters.num_variables - self.folding_factor.total_number(round_index);
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
        let new_domain = round_state.domain.scale(domain_reduction);
        let inv_rate = new_domain.size() / folded_evaluations.num_evals();
        let folded_matrix = info_span!("fold matrix").in_scope(|| {
            let evals_repeated = info_span!("repeating evals")
                .in_scope(|| parallel_repeat(folded_evaluations.evals(), inv_rate));
            // Do DFT on only interleaved polys to be folded.
            info_span!(
                "dft",
                height = evals_repeated.len() >> folding_factor_next,
                width = 1 << folding_factor_next
            )
            .in_scope(|| {
                dft.dft_algebra_batch_by_evals(RowMajorMatrix::new(
                    evals_repeated,
                    1 << folding_factor_next,
                ))
            })
        });

        let mmcs = MerkleTreeMmcs::<F::Packing, F::Packing, H, C, DIGEST_ELEMS>::new(
            self.merkle_hash.clone(),
            self.merkle_compress.clone(),
        );
        let extension_mmcs = ExtensionMmcs::new(mmcs.clone());
        let (root, prover_data) =
            info_span!("commit matrix").in_scope(|| extension_mmcs.commit_matrix(folded_matrix));

        prover_state.add_base_scalars(root.as_ref());

        // Handle OOD (Out-Of-Domain) samples
        let (ood_points, ood_answers) = sample_ood_points(
            prover_state,
            round_params.ood_samples,
            num_variables,
            |point| info_span!("ood evaluation").in_scope(|| folded_evaluations.evaluate(point)),
        );

        // STIR Queries
        let (stir_challenges, stir_challenges_indexes) = self.compute_stir_queries(
            round_index,
            prover_state,
            round_state,
            num_variables,
            round_params,
            ood_points,
        )?;

        // Collect Merkle proofs for stir queries
        let stir_evaluations = match &round_state.merkle_prover_data {
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

                // Evaluate answers in the folding randomness.
                let mut stir_evaluations = ood_answers;
                // Exactly one growth
                stir_evaluations.reserve_exact(answers.len());

                for answer in &answers {
                    stir_evaluations.push(
                        EvaluationsList::new(answer.clone())
                            .evaluate(&round_state.folding_randomness),
                    );
                }

                stir_evaluations
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

                // Evaluate answers in the folding randomness.
                let mut stir_evaluations = ood_answers;
                // Exactly one growth
                stir_evaluations.reserve_exact(answers.len());

                for answer in &answers {
                    stir_evaluations.push(
                        EvaluationsList::new(answer.clone())
                            .evaluate(&round_state.folding_randomness),
                    );
                }

                stir_evaluations
            }
        };

        prover_state.pow_grinding(round_params.pow_bits);

        // Randomness for combination
        let combination_randomness_gen: EF = prover_state.sample();
        let combination_randomness: Vec<_> = combination_randomness_gen
            .powers()
            .take(stir_challenges.len())
            .collect();

        round_state.sumcheck_prover.add_new_equality(
            &stir_challenges,
            &stir_evaluations,
            &combination_randomness,
        );

        let folding_randomness = round_state.sumcheck_prover.compute_sumcheck_polynomials(
            prover_state,
            folding_factor_next,
            round_params.folding_pow_bits,
        );

        let start_idx = self.folding_factor.total_number(round_index);
        let dst_randomness =
            &mut round_state.randomness_vec[start_idx..][..folding_randomness.len()];

        for (dst, src) in dst_randomness
            .iter_mut()
            .zip(folding_randomness.iter().rev())
        {
            *dst = *src;
        }

        // Update round state
        round_state.domain = new_domain;
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
    ) -> ProofResult<()>
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
        prover_state.add_extension_scalars(&round_state.sumcheck_prover.evals);

        // Final verifier queries and answers. The indices are over the folded domain.
        let final_challenge_indexes = get_challenge_stir_queries(
            // The size of the original domain before folding
            round_state.domain.size(),
            // The folding factor we used to fold the previous polynomial
            self.folding_factor.at_round(round_index),
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

        prover_state.pow_grinding(self.final_pow_bits);

        // Run final sumcheck if required
        if self.final_sumcheck_rounds > 0 {
            let final_folding_randomness =
                round_state.sumcheck_prover.compute_sumcheck_polynomials(
                    prover_state,
                    self.final_sumcheck_rounds,
                    self.final_folding_pow_bits,
                );
            let start_idx = self.folding_factor.total_number(round_index);
            let rand_dst = &mut round_state.randomness_vec
                [start_idx..start_idx + final_folding_randomness.len()];

            for (dst, src) in rand_dst
                .iter_mut()
                .zip(final_folding_randomness.iter().rev())
            {
                *dst = *src;
            }
        }

        Ok(())
    }

    #[instrument(skip_all, level = "debug")]
    fn compute_stir_queries<const DIGEST_ELEMS: usize>(
        &self,
        round_index: usize,
        prover_state: &mut ProverState<F, EF, Challenger>,
        round_state: &RoundState<EF, F, F, DenseMatrix<F>, DIGEST_ELEMS>,
        num_variables: usize,
        round_params: &RoundConfig<EF>,
        ood_points: Vec<EF>,
    ) -> ProofResult<(Vec<MultilinearPoint<EF>>, Vec<usize>)> {
        let stir_challenges_indexes = get_challenge_stir_queries(
            round_state.domain.size(),
            self.folding_factor.at_round(round_index),
            round_params.num_queries,
            prover_state,
        )?;

        // Compute the generator of the folded domain, in the extension field
        let domain_scaled_gen = round_state
            .domain
            .backing_domain
            .element(1 << self.folding_factor.at_round(round_index));
        let stir_challenges = ood_points
            .into_iter()
            .chain(
                stir_challenges_indexes
                    .iter()
                    .map(|i| domain_scaled_gen.exp_u64(*i as u64)),
            )
            .map(|univariate| MultilinearPoint::expand_from_univariate(univariate, num_variables))
            .collect();

        Ok((stir_challenges, stir_challenges_indexes))
    }
}
