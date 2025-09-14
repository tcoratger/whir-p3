use p3_commit::{ExtensionMmcs, Mmcs};
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::dense::DenseMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::{
    fiat_shamir::{errors::ProofResult, prover::ProverState},
    parameters::FoldingFactor,
    whir::{prover::round::RoundState, utils::get_challenge_stir_queries},
};

/// Handles the final round of the WHIR protocol.
///
/// The final round processor manages the terminal stage where:
/// - Polynomial coefficients are sent directly (no more folding)
/// - Final proof-of-work grinding secures the transcript
/// - Challenge queries are computed and answered
/// - Merkle tree openings provide the final consistency proofs
/// - Optional final sumcheck handles remaining variables
pub struct FinalRoundProcessor<'a, F, H, C, const DIGEST_ELEMS: usize> {
    /// Merkle tree hash function
    merkle_hash: &'a H,
    /// Merkle tree compression function
    merkle_compress: &'a C,
    /// Folding factor configuration for computing final challenges
    folding_factor: FoldingFactor,
    /// Number of final queries to generate
    final_queries: usize,
    /// Proof-of-work bits for final grinding
    final_pow_bits: usize,
    /// Number of final sumcheck rounds
    final_sumcheck_rounds: usize,
    /// Proof-of-work bits for final sumcheck
    final_folding_pow_bits: usize,
    /// Phantom data for field type
    _phantom: std::marker::PhantomData<F>,
}

impl<'a, F, H, C, const DIGEST_ELEMS: usize> FinalRoundProcessor<'a, F, H, C, DIGEST_ELEMS>
where
    F: TwoAdicField,
    H: CryptographicHasher<F, [F; DIGEST_ELEMS]> + CryptographicHasher<F::Packing, [F::Packing; DIGEST_ELEMS]> + Sync,
    C: PseudoCompressionFunction<[F; DIGEST_ELEMS], 2> + PseudoCompressionFunction<[F::Packing; DIGEST_ELEMS], 2> + Sync,
    [F; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
{
    /// Creates a new final round processor.
    pub fn new(
        merkle_hash: &'a H,
        merkle_compress: &'a C,
        folding_factor: FoldingFactor,
        final_queries: usize,
        final_pow_bits: usize,
        final_sumcheck_rounds: usize,
        final_folding_pow_bits: usize,
    ) -> Self {
        Self {
            merkle_hash,
            merkle_compress,
            folding_factor,
            final_queries,
            final_pow_bits,
            final_sumcheck_rounds,
            final_folding_pow_bits,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Processes the final round of the WHIR protocol.
    ///
    /// This method handles the complete final round workflow:
    /// 1. Sends polynomial coefficients directly to the verifier
    /// 2. Performs proof-of-work grinding for transcript security
    /// 3. Generates and answers final challenge queries
    /// 4. Runs final sumcheck if additional variables remain
    ///
    /// # Arguments
    /// - `round_index`: The index of this final round
    /// - `prover_state`: Mutable prover state for transcript management
    /// - `round_state`: Current round state containing polynomial and commitment data
    ///
    /// # Returns
    /// `ProofResult<()>` indicating success or failure of the final round processing
    #[instrument(skip_all, fields(round_index, final_evals = round_state.sumcheck_prover.evals.num_evals()))]
    pub fn process_final_round<EF>(
        &self,
        round_index: usize,
        prover_state: &mut ProverState<F, EF, impl p3_challenger::FieldChallenger<F> + p3_challenger::GrindingChallenger<Witness = F>>,
        round_state: &mut RoundState<EF, F, F, DenseMatrix<F>, DIGEST_ELEMS>,
    ) -> ProofResult<()>
    where
        EF: ExtensionField<F> + TwoAdicField,
    {
        // Step 1: Send polynomial coefficients directly to the verifier
        self.send_final_coefficients(prover_state, round_state);

        // Step 2: Perform proof-of-work grinding to secure the transcript
        self.perform_final_pow_grinding(prover_state);

        // Step 3: Generate and answer final challenge queries
        self.process_final_queries(round_index, prover_state, round_state)?;

        // Step 4: Run final sumcheck if needed
        self.run_final_sumcheck_if_needed(round_index, prover_state, round_state)?;

        Ok(())
    }

    /// Sends polynomial coefficients directly to the verifier.
    ///
    /// At the final round, the polynomial is small enough that it's more efficient
    /// to send coefficients directly rather than continue with Merkle commitments.
    fn send_final_coefficients<EF>(
        &self,
        prover_state: &mut ProverState<F, EF, impl p3_challenger::FieldChallenger<F> + p3_challenger::GrindingChallenger<Witness = F>>,
        round_state: &RoundState<EF, F, F, DenseMatrix<F>, DIGEST_ELEMS>,
    ) where
        EF: ExtensionField<F> + TwoAdicField,
    {
        prover_state.add_extension_scalars(round_state.sumcheck_prover.evals.as_slice());
    }

    /// Performs final proof-of-work grinding.
    ///
    /// This critical security step prevents the prover from influencing
    /// the verifier's final challenge queries by making it computationally
    /// expensive to "shop" for favorable challenges.
    fn perform_final_pow_grinding<EF>(
        &self,
        prover_state: &mut ProverState<F, EF, impl p3_challenger::FieldChallenger<F> + p3_challenger::GrindingChallenger<Witness = F>>,
    ) where
        EF: ExtensionField<F> + TwoAdicField,
    {
        prover_state.pow_grinding(self.final_pow_bits);
    }

    /// Processes final challenge queries and opens the corresponding Merkle proofs.
    ///
    /// Generates final challenge indices, opens the Merkle tree at those positions,
    /// and provides the authentication proofs needed for verification.
    fn process_final_queries<EF>(
        &self,
        round_index: usize,
        prover_state: &mut ProverState<F, EF, impl p3_challenger::FieldChallenger<F> + p3_challenger::GrindingChallenger<Witness = F>>,
        round_state: &RoundState<EF, F, F, DenseMatrix<F>, DIGEST_ELEMS>,
    ) -> ProofResult<()>
    where
        EF: ExtensionField<F> + TwoAdicField,
    {
        // Generate final challenge indices
        let final_challenge_indexes = get_challenge_stir_queries(
            round_state.domain_size,
            self.folding_factor.at_round(round_index),
            self.final_queries,
            prover_state,
        )?;

        let mmcs = MerkleTreeMmcs::<F::Packing, F::Packing, H, C, DIGEST_ELEMS>::new(
            self.merkle_hash.clone(),
            self.merkle_compress.clone(),
        );

        // Process queries based on commitment type
        match &round_state.merkle_prover_data {
            None => self.process_base_final_queries(
                prover_state,
                round_state,
                final_challenge_indexes,
                &mmcs,
            ),
            Some(data) => {
                self.process_extension_final_queries(prover_state, data, final_challenge_indexes, &mmcs)
            }
        }
    }

    /// Processes final queries for base field commitments.
    fn process_base_final_queries<EF>(
        &self,
        prover_state: &mut ProverState<F, EF, impl p3_challenger::FieldChallenger<F> + p3_challenger::GrindingChallenger<Witness = F>>,
        round_state: &RoundState<EF, F, F, DenseMatrix<F>, DIGEST_ELEMS>,
        challenge_indexes: Vec<usize>,
        mmcs: &MerkleTreeMmcs<F::Packing, F::Packing, H, C, DIGEST_ELEMS>,
    ) where
        EF: ExtensionField<F> + TwoAdicField,
    {
        let mut answers = Vec::with_capacity(challenge_indexes.len());
        let mut merkle_proofs = Vec::with_capacity(challenge_indexes.len());

        // Open Merkle tree at each final challenge index
        for challenge in challenge_indexes {
            let commitment = mmcs.open_batch(challenge, &round_state.commitment_merkle_prover_data);
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
    }

    /// Processes final queries for extension field commitments.
    fn process_extension_final_queries<EF>(
        &self,
        prover_state: &mut ProverState<F, EF, impl p3_challenger::FieldChallenger<F> + p3_challenger::GrindingChallenger<Witness = F>>,
        prover_data: &crate::whir::committer::RoundMerkleTree<F, EF, F, DIGEST_ELEMS>,
        challenge_indexes: Vec<usize>,
        mmcs: &MerkleTreeMmcs<F::Packing, F::Packing, H, C, DIGEST_ELEMS>,
    ) where
        EF: ExtensionField<F> + TwoAdicField,
    {
        let extension_mmcs = ExtensionMmcs::new(mmcs.clone());
        let mut answers = Vec::with_capacity(challenge_indexes.len());
        let mut merkle_proofs = Vec::with_capacity(challenge_indexes.len());

        // Open extension Merkle tree at each final challenge index
        for challenge in challenge_indexes {
            let commitment = extension_mmcs.open_batch(challenge, prover_data);
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
    }

    /// Runs the final sumcheck protocol if additional variables remain.
    ///
    /// When the folding process doesn't reduce the polynomial to a constant,
    /// a final sumcheck handles the remaining variables to complete the protocol.
    fn run_final_sumcheck_if_needed<EF>(
        &self,
        round_index: usize,
        prover_state: &mut ProverState<F, EF, impl p3_challenger::FieldChallenger<F> + p3_challenger::GrindingChallenger<Witness = F>>,
        round_state: &mut RoundState<EF, F, F, DenseMatrix<F>, DIGEST_ELEMS>,
    ) -> ProofResult<()>
    where
        EF: ExtensionField<F> + TwoAdicField,
    {
        if self.final_sumcheck_rounds > 0 {
            let final_folding_randomness = round_state.sumcheck_prover.compute_sumcheck_polynomials(
                prover_state,
                self.final_sumcheck_rounds,
                self.final_folding_pow_bits,
            );

            // Update the randomness vector with final sumcheck challenges
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

#[cfg(test)]
mod tests {
    use super::*;
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_field::extension::BinomialExtensionField;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::{SeedableRng, rngs::SmallRng};

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type Hash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type Compress = TruncatedPermutation<Perm, 2, 8, 16>;

    #[test]
    fn test_final_round_processor_creation() {
        let mut rng = SmallRng::seed_from_u64(42);
        let perm = Perm::new_from_rng(8, 22, &mut rng);
        let hash = Hash::new(perm.clone());
        let compress = Compress::new(perm);

        let _processor = FinalRoundProcessor::<_, _, _, 8>::new(
            &hash,
            &compress,
            FoldingFactor::Constant(2),
            10,    // final_queries
            8,     // final_pow_bits
            3,     // final_sumcheck_rounds
            4,     // final_folding_pow_bits
        );
    }

    #[test]
    fn test_final_round_processor_configuration() {
        let mut rng = SmallRng::seed_from_u64(42);
        let perm = Perm::new_from_rng(8, 22, &mut rng);
        let hash = Hash::new(perm.clone());
        let compress = Compress::new(perm);

        let processor = FinalRoundProcessor::<_, _, _, 8>::new(
            &hash,
            &compress,
            FoldingFactor::ConstantFromSecondRound(5, 2),
            15,   // final_queries
            10,   // final_pow_bits
            0,    // final_sumcheck_rounds (no final sumcheck)
            0,    // final_folding_pow_bits
        );

        assert_eq!(processor.final_queries, 15);
        assert_eq!(processor.final_pow_bits, 10);
        assert_eq!(processor.final_sumcheck_rounds, 0);
    }
}