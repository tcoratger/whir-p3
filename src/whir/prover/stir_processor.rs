use p3_commit::{ExtensionMmcs, Mmcs};
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_interpolation::interpolate_subgroup;
use p3_matrix::dense::DenseMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use serde::{Deserialize, Serialize};
use tracing::{info_span, instrument};

use crate::{
    constant::K_SKIP_SUMCHECK,
    fiat_shamir::{errors::ProofResult, prover::ProverState},
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    whir::{parameters::RoundConfig, prover::round::RoundState, utils::get_challenge_stir_queries},
};

/// Handles STIR (Succinct Transparent Interactive Randomness) query processing.
///
/// STIR queries test polynomial consistency by:
/// - Generating random challenge indices based on the current transcript
/// - Opening the Merkle tree at those locations to reveal polynomial data
/// - Evaluating the revealed data with folding randomness
/// - Handling special cases for univariate skip optimization
/// - Providing evaluation results for sumcheck combination
pub struct StirProcessor<'a, F, H, C, const DIGEST_ELEMS: usize> {
    /// Merkle tree hash function
    merkle_hash: &'a H,
    /// Merkle tree compression function
    merkle_compress: &'a C,
    /// Whether univariate skip optimization is enabled
    univariate_skip: bool,
    /// Whether this is the initial statement round
    initial_statement: bool,
    /// Phantom data for field type
    _phantom: std::marker::PhantomData<F>,
}

impl<'a, F, H, C, const DIGEST_ELEMS: usize> StirProcessor<'a, F, H, C, DIGEST_ELEMS>
where
    F: TwoAdicField,
    H: CryptographicHasher<F, [F; DIGEST_ELEMS]>
        + CryptographicHasher<F::Packing, [F::Packing; DIGEST_ELEMS]>
        + Sync,
    C: PseudoCompressionFunction<[F; DIGEST_ELEMS], 2>
        + PseudoCompressionFunction<[F::Packing; DIGEST_ELEMS], 2>
        + Sync,
    [F; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
{
    /// Creates a new STIR processor.
    pub fn new(
        merkle_hash: &'a H,
        merkle_compress: &'a C,
        univariate_skip: bool,
        initial_statement: bool,
    ) -> Self {
        Self {
            merkle_hash,
            merkle_compress,
            univariate_skip,
            initial_statement,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Computes STIR challenge points and indices for polynomial queries.
    ///
    /// Generates random challenge locations based on the current transcript state
    /// and expands them into multilinear evaluation points for both extension
    /// field (OOD) and base field (STIR) queries.
    #[instrument(skip_all, fields(num_variables, num_queries = round_config.num_queries))]
    pub fn compute_challenge_points<EF>(
        &self,
        prover_state: &mut ProverState<F, EF, impl p3_challenger::FieldChallenger<F> + p3_challenger::GrindingChallenger<Witness = F>>,
        domain_size: usize,
        domain_gen: F,
        folding_factor: usize,
        num_variables: usize,
        round_config: &RoundConfig<F>,
        ood_points: &[EF],
    ) -> ProofResult<(
        Vec<MultilinearPoint<EF>>,
        Vec<MultilinearPoint<F>>,
        Vec<usize>,
    )>
    where
        EF: ExtensionField<F> + TwoAdicField,
    {
        let stir_indices = get_challenge_stir_queries(
            domain_size,
            folding_factor,
            round_config.num_queries,
            prover_state,
        )?;

        let ood_challenges = ood_points
            .iter()
            .map(|univariate| MultilinearPoint::expand_from_univariate(*univariate, num_variables))
            .collect();

        let stir_challenges = stir_indices
            .iter()
            .map(|&i| {
                MultilinearPoint::expand_from_univariate(
                    domain_gen.exp_u64(i as u64),
                    num_variables,
                )
            })
            .collect();

        Ok((ood_challenges, stir_challenges, stir_indices))
    }

    /// Processes STIR queries by opening Merkle commitments and evaluating polynomials.
    ///
    /// This handles the complex evaluation logic including:
    /// - Opening Merkle tree proofs at challenge indices
    /// - Determining evaluation mode (skip vs standard)
    /// - Computing polynomial evaluations with folding randomness
    /// - Adding necessary hints to the prover transcript
    #[instrument(skip_all, fields(num_queries = stir_indices.len(), round_index))]
    pub fn process_stir_queries<EF>(
        &self,
        round_index: usize,
        prover_state: &mut ProverState<F, EF, impl p3_challenger::FieldChallenger<F> + p3_challenger::GrindingChallenger<Witness = F>>,
        round_state: &RoundState<EF, F, F, DenseMatrix<F>, DIGEST_ELEMS>,
        stir_indices: &[usize],
        folding_factor: usize,
    ) -> ProofResult<Vec<EF>>
    where
        EF: ExtensionField<F> + TwoAdicField,
    {
        let mmcs = MerkleTreeMmcs::<F::Packing, F::Packing, H, C, DIGEST_ELEMS>::new(
            self.merkle_hash.clone(),
            self.merkle_compress.clone(),
        );

        match &round_state.merkle_prover_data {
            None => self.process_base_field_queries(
                round_index,
                prover_state,
                round_state,
                stir_indices,
                folding_factor,
                &mmcs,
            ),
            Some(data) => self.process_extension_field_queries(
                prover_state,
                round_state,
                stir_indices,
                data,
                &mmcs,
            ),
        }
    }

    /// Processes base field STIR queries with skip-aware evaluation.
    fn process_base_field_queries<EF>(
        &self,
        round_index: usize,
        prover_state: &mut ProverState<F, EF, impl p3_challenger::FieldChallenger<F> + p3_challenger::GrindingChallenger<Witness = F>>,
        round_state: &RoundState<EF, F, F, DenseMatrix<F>, DIGEST_ELEMS>,
        stir_indices: &[usize],
        folding_factor: usize,
        mmcs: &MerkleTreeMmcs<F::Packing, F::Packing, H, C, DIGEST_ELEMS>,
    ) -> ProofResult<Vec<EF>>
    where
        EF: ExtensionField<F> + TwoAdicField,
    {
        let mut answers = Vec::with_capacity(stir_indices.len());
        let mut merkle_proofs = Vec::with_capacity(stir_indices.len());

        // Open Merkle tree at each challenge index
        for &index in stir_indices {
            let commitment = mmcs.open_batch(index, &round_state.commitment_merkle_prover_data);
            answers.push(commitment.opened_values[0].clone());
            merkle_proofs.push(commitment.opening_proof);
        }

        // Add hints for verification
        for answer in &answers {
            prover_state.hint_base_scalars(answer);
        }
        for merkle_proof in &merkle_proofs {
            for digest in merkle_proof {
                prover_state.hint_base_scalars(digest);
            }
        }

        // Evaluate polynomials with appropriate method
        let is_skip_round = self.should_use_skip_evaluation(round_index, folding_factor);
        let mut evaluations = Vec::with_capacity(answers.len());

        for answer in &answers {
            let eval = if is_skip_round {
                self.evaluate_with_skip(answer, round_state)?
            } else {
                self.evaluate_standard(answer, round_state)
            };
            evaluations.push(eval);
        }

        Ok(evaluations)
    }

    /// Processes extension field STIR queries with standard evaluation.
    fn process_extension_field_queries<EF>(
        &self,
        prover_state: &mut ProverState<F, EF, impl p3_challenger::FieldChallenger<F> + p3_challenger::GrindingChallenger<Witness = F>>,
        round_state: &RoundState<EF, F, F, DenseMatrix<F>, DIGEST_ELEMS>,
        stir_indices: &[usize],
        prover_data: &crate::whir::committer::RoundMerkleTree<F, EF, F, DIGEST_ELEMS>,
        mmcs: &MerkleTreeMmcs<F::Packing, F::Packing, H, C, DIGEST_ELEMS>,
    ) -> ProofResult<Vec<EF>>
    where
        EF: ExtensionField<F> + TwoAdicField,
    {
        let extension_mmcs = ExtensionMmcs::new(mmcs.clone());
        let mut answers = Vec::with_capacity(stir_indices.len());
        let mut merkle_proofs = Vec::with_capacity(stir_indices.len());

        // Open extension field Merkle tree at each challenge index
        for &index in stir_indices {
            let commitment = extension_mmcs.open_batch(index, prover_data);
            answers.push(commitment.opened_values[0].clone());
            merkle_proofs.push(commitment.opening_proof);
        }

        // Add hints for verification
        for answer in &answers {
            prover_state.hint_extension_scalars(answer);
        }
        for merkle_proof in &merkle_proofs {
            for digest in merkle_proof {
                prover_state.hint_base_scalars(digest);
            }
        }

        // Evaluate with standard method (extension field doesn't use skip)
        let mut evaluations = Vec::with_capacity(answers.len());
        for answer in &answers {
            evaluations.push(
                EvaluationsList::new(answer.clone()).evaluate(&round_state.folding_randomness),
            );
        }

        Ok(evaluations)
    }

    /// Determines if skip evaluation should be used for this round.
    fn should_use_skip_evaluation(&self, round_index: usize, folding_factor: usize) -> bool {
        self.initial_statement
            && round_index == 0
            && self.univariate_skip
            && folding_factor >= K_SKIP_SUMCHECK
    }

    /// Evaluates polynomial with univariate skip optimization.
    ///
    /// This implements the two-stage skip evaluation:
    /// 1. Reshape evaluations into a matrix based on skip structure
    /// 2. Interpolate over the skipped variables using the skip challenge
    /// 3. Evaluate the remaining polynomial at the rest of the challenges
    fn evaluate_with_skip<EF>(
        &self,
        answer: &[F],
        round_state: &RoundState<EF, F, F, DenseMatrix<F>, DIGEST_ELEMS>,
    ) -> ProofResult<EF>
    where
        EF: ExtensionField<F> + TwoAdicField,
    {
        let evals = EvaluationsList::new(answer.to_vec());
        let num_remaining_vars = evals.num_variables() - K_SKIP_SUMCHECK;
        let width = 1 << num_remaining_vars;

        // Reshape into matrix for skip evaluation
        let mat = evals.into_mat(width);

        // Extract challenges from the folding randomness
        let r_all = &round_state.folding_randomness;
        let r_skip = *r_all
            .last_variable()
            .expect("skip challenge must be present");
        let r_rest = r_all.get_subpoint_over_range(0..num_remaining_vars);

        // Two-stage evaluation
        let folded_row = interpolate_subgroup(&mat, r_skip);
        Ok(EvaluationsList::new(folded_row).evaluate(&r_rest))
    }

    /// Evaluates polynomial with standard multilinear evaluation.
    fn evaluate_standard<EF>(
        &self,
        answer: &[F],
        round_state: &RoundState<EF, F, F, DenseMatrix<F>, DIGEST_ELEMS>,
    ) -> EF
    where
        EF: ExtensionField<F> + TwoAdicField,
    {
        EvaluationsList::new(answer.to_vec()).evaluate(&round_state.folding_randomness)
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_field::extension::BinomialExtensionField;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type Hash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type Compress = TruncatedPermutation<Perm, 2, 8, 16>;

    #[test]
    fn test_stir_processor_creation() {
        use rand::{SeedableRng, rngs::SmallRng};

        let mut rng = SmallRng::seed_from_u64(42);
        let perm = Perm::new_from_rng(8, 22, &mut rng);
        let hash = Hash::new(perm.clone());
        let compress = Compress::new(perm);

        let _processor = StirProcessor::<_, _, _, 8>::new(&hash, &compress, true, true);
    }

    #[test]
    #[ignore] // TODO: Fix trait bound issues
    fn test_skip_evaluation_detection() {
        use rand::{SeedableRng, rngs::SmallRng};

        let mut rng = SmallRng::seed_from_u64(42);
        let perm = Perm::new_from_rng(8, 22, &mut rng);
        let hash = Hash::new(perm.clone());
        let compress = Compress::new(perm);

        let processor = StirProcessor::<_, _, _, 8>::new(&hash, &compress, true, true);

        // Should use skip for round 0 with sufficient folding factor
        assert!(processor.should_use_skip_evaluation(0, K_SKIP_SUMCHECK));
        assert!(processor.should_use_skip_evaluation(0, K_SKIP_SUMCHECK + 1));

        // Should not use skip for subsequent rounds
        assert!(!processor.should_use_skip_evaluation(1, K_SKIP_SUMCHECK));

        // Should not use skip with insufficient folding factor
        assert!(!processor.should_use_skip_evaluation(0, K_SKIP_SUMCHECK - 1));
    }

    #[test]
    #[ignore] // TODO: Fix trait bound issues
    fn test_skip_disabled_processor() {
        use rand::{SeedableRng, rngs::SmallRng};

        let mut rng = SmallRng::seed_from_u64(42);
        let perm = Perm::new_from_rng(8, 22, &mut rng);
        let hash = Hash::new(perm.clone());
        let compress = Compress::new(perm);

        let processor = StirProcessor::new(&hash, &compress, false, true);

        // Should never use skip when disabled
        assert!(!processor.should_use_skip_evaluation(0, K_SKIP_SUMCHECK));
        assert!(!processor.should_use_skip_evaluation(0, K_SKIP_SUMCHECK + 5));
    }
}
