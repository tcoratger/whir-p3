use p3_commit::{BatchOpening, ExtensionMmcs, Mmcs};
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use serde::{Deserialize, Serialize};
use tracing::{info_span, instrument};

use crate::{
    dft::EvalsDft,
    fiat_shamir::{errors::ProofResult, prover::ProverState},
    poly::evals::EvaluationsList,
    utils::parallel_repeat,
    whir::{
        parameters::RoundConfig,
        prover::round::RoundState,
        utils::get_challenge_stir_queries,
    },
};

/// Processes a single WHIR round including folding, commitment, and STIR operations.
///
/// The round processor encapsulates the complex logic of:
/// - Folding polynomial evaluations to reduce problem size
/// - Building Merkle tree commitments for the folded polynomials
/// - Handling out-of-domain evaluation requests
/// - Processing STIR challenge queries
/// - Managing proof-of-work grinding for security
pub struct RoundProcessor<'a, EF, F, H, C> {
    /// Round configuration parameters
    round_config: &'a RoundConfig<F>,
    /// Merkle tree hash function
    merkle_hash: &'a H,
    /// Merkle tree compression function
    merkle_compress: &'a C,
    /// Round index (0-based)
    round_index: usize,
    /// Number of variables before this round
    num_variables: usize,
    /// Folding factor for this round
    folding_factor: usize,
}

impl<'a, EF, F, H, C> RoundProcessor<'a, EF, F, H, C>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    H: CryptographicHasher<F, [F; 8]> + CryptographicHasher<F::Packing, [F::Packing; 8]> + Sync,
    C: PseudoCompressionFunction<[F; 8], 2> + PseudoCompressionFunction<[F::Packing; 8], 2> + Sync,
    [F; 8]: Serialize + for<'de> Deserialize<'de>,
{
    /// Creates a new round processor.
    pub fn new(
        round_config: &'a RoundConfig<F>,
        merkle_hash: &'a H,
        merkle_compress: &'a C,
        round_index: usize,
        num_variables: usize,
        folding_factor: usize,
    ) -> Self {
        Self {
            round_config,
            merkle_hash,
            merkle_compress,
            round_index,
            num_variables,
            folding_factor,
        }
    }

    /// Processes a complete WHIR round.
    ///
    /// This orchestrates all the steps needed for a single folding round:
    /// 1. Fold polynomial evaluations to reduce dimensionality
    /// 2. Build and commit to Merkle tree over folded evaluations
    /// 3. Handle out-of-domain evaluation queries
    /// 4. Process STIR challenge queries with proofs
    /// 5. Perform proof-of-work grinding for transcript security
    #[instrument(skip_all, fields(round = self.round_index, vars = self.num_variables))]
    pub fn process_round(
        &self,
        dft: &EvalsDft<F>,
        prover_state: &mut ProverState<F, EF, impl p3_challenger::FieldChallenger<F>>,
        round_state: &mut RoundState<EF, F, F, DenseMatrix<F>, 8>,
    ) -> ProofResult<()> {
        let folded_evaluations = &round_state.sumcheck_prover.evals;

        // Verify the polynomial has expected dimensions
        assert_eq!(self.num_variables, folded_evaluations.num_variables());

        // Step 1: Fold polynomial evaluations to reduce problem size
        let folded_polynomial = self.fold_polynomial_evaluations(dft, folded_evaluations)?;

        // Step 2: Build Merkle tree commitment over the folded polynomial
        let (commitment_data, tree_prover_data) =
            self.build_merkle_commitment(&folded_polynomial)?;

        // Step 3: Sample out-of-domain evaluation points
        let ood_points = self.sample_ood_points(prover_state, round_state)?;

        // Step 4: Perform proof-of-work grinding before revealing queries
        self.perform_pow_grinding(prover_state)?;

        // Step 5: Handle STIR challenge queries
        let stir_openings = self.process_stir_queries(
            prover_state,
            round_state,
            &folded_polynomial,
            &tree_prover_data,
            &ood_points,
        )?;

        // Step 6: Update round state for next iteration
        self.update_round_state(round_state, folded_polynomial, commitment_data, stir_openings)?;

        Ok(())
    }

    /// Folds polynomial evaluations to reduce the number of variables.
    ///
    /// This applies the folding transformation that reduces the polynomial
    /// from `num_variables` to `num_variables - folding_factor` dimensions.
    fn fold_polynomial_evaluations(
        &self,
        dft: &EvalsDft<F>,
        evaluations: &EvaluationsList<EF>,
    ) -> ProofResult<EvaluationsList<EF>> {
        info_span!("fold_polynomial").in_scope(|| {
            // The actual folding implementation would go here
            // For now, we'll return the evaluations as-is to maintain compatibility
            Ok(evaluations.clone())
        })
    }

    /// Builds a Merkle tree commitment over the folded polynomial evaluations.
    fn build_merkle_commitment(
        &self,
        folded_polynomial: &EvaluationsList<EF>,
    ) -> ProofResult<(Vec<u8>, Vec<u8>)> {
        info_span!("build_commitment").in_scope(|| {
            // Placeholder implementation
            Ok((vec![], vec![]))
        })
    }

    /// Samples out-of-domain evaluation points for uniqueness enforcement.
    fn sample_ood_points(
        &self,
        prover_state: &mut ProverState<F, EF, impl p3_challenger::FieldChallenger<F>>,
        round_state: &mut RoundState<EF, F, F, DenseMatrix<F>, 8>,
    ) -> ProofResult<Vec<EF>> {
        info_span!("sample_ood").in_scope(|| {
            // Placeholder implementation
            Ok(vec![])
        })
    }

    /// Performs proof-of-work grinding to secure the transcript.
    ///
    /// This prevents the prover from selectively choosing commitments that lead
    /// to favorable challenge queries by requiring expensive computation.
    fn perform_pow_grinding(
        &self,
        prover_state: &mut ProverState<F, EF, impl p3_challenger::FieldChallenger<F>>,
    ) -> ProofResult<()> {
        info_span!("pow_grinding").in_scope(|| {
            prover_state.pow_grinding(self.round_config.pow_bits);
            Ok(())
        })
    }

    /// Processes STIR challenge queries and generates the required proofs.
    ///
    /// STIR queries test that the folded polynomial was constructed honestly
    /// by querying specific positions and verifying their consistency.
    fn process_stir_queries(
        &self,
        prover_state: &mut ProverState<F, EF, impl p3_challenger::FieldChallenger<F>>,
        round_state: &mut RoundState<EF, F, F, DenseMatrix<F>, 8>,
        folded_polynomial: &EvaluationsList<EF>,
        tree_prover_data: &[u8],
        ood_points: &[EF],
    ) -> ProofResult<Vec<u8>> {
        info_span!("stir_queries").in_scope(|| {
            // Placeholder implementation
            Ok(vec![])
        })
    }

    /// Updates the round state with results from this round.
    fn update_round_state(
        &self,
        round_state: &mut RoundState<EF, F, F, DenseMatrix<F>, 8>,
        folded_polynomial: EvaluationsList<EF>,
        commitment_data: Vec<u8>,
        stir_openings: Vec<u8>,
    ) -> ProofResult<()> {
        // Update the round state for the next iteration
        round_state.sumcheck_prover.evals = folded_polynomial;
        Ok(())
    }
}

impl<'a, EF, F, H, C> RoundProcessor<'a, EF, F, H, C> {
    /// Calculates the number of variables after folding.
    pub fn variables_after_folding(&self) -> usize {
        self.num_variables - self.folding_factor
    }

    /// Determines if this is the final round (no more folding needed).
    pub fn is_final_round(&self, total_rounds: usize) -> bool {
        self.round_index >= total_rounds
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variables_after_folding() {
        use crate::whir::parameters::RoundConfig;
        use p3_baby_bear::BabyBear;
        use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
        use p3_baby_bear::Poseidon2BabyBear;

        type F = BabyBear;
        type Perm = Poseidon2BabyBear<16>;
        type Hash = PaddingFreeSponge<Perm, 16, 8, 8>;
        type Compress = TruncatedPermutation<Perm, 2, 8, 16>;

        let config = RoundConfig {
            num_variables: 6,
            folding_factor: 3,
            num_queries: 10,
            pow_bits: 8,
            domain_size: 64,
            folded_domain_gen: F::ONE,
            ood_samples: 1,
            folding_pow_bits: 4,
            log_inv_rate: 1,
        };

        let hash = Hash::new(Perm::new_from_rng(8, 22, &mut rand::rng()));
        let compress = Compress::new(Perm::new_from_rng(8, 22, &mut rand::rng()));

        let processor = RoundProcessor::new(&config, &hash, &compress, 0, 6, 3);

        assert_eq!(processor.variables_after_folding(), 3);
        assert!(!processor.is_final_round(2));
        assert!(processor.is_final_round(0));
    }
}