use p3_commit::{ExtensionMmcs, Mmcs};
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use serde::{Deserialize, Serialize};
use tracing::{info_span, instrument};

use crate::{
    dft::EvalsDft,
    fiat_shamir::errors::ProofResult,
    poly::evals::EvaluationsList,
    utils::parallel_repeat,
};

/// Handles polynomial commitment operations including Merkle tree construction and opening.
///
/// The commitment handler encapsulates:
/// - Polynomial folding and DFT transformations
/// - Merkle tree commitment generation
/// - Batch opening and proof generation for query responses
/// - Extension field handling for both base and extension commitments
pub struct CommitmentHandler<'a, F, H, C, const DIGEST_ELEMS: usize> {
    /// Merkle tree hash function
    merkle_hash: &'a H,
    /// Merkle tree compression function
    merkle_compress: &'a C,
    /// Phantom data for field type
    _phantom: std::marker::PhantomData<F>,
}

impl<'a, F, H, C, const DIGEST_ELEMS: usize> CommitmentHandler<'a, F, H, C, DIGEST_ELEMS>
where
    F: TwoAdicField,
    H: CryptographicHasher<F, [F; DIGEST_ELEMS]> + CryptographicHasher<F::Packing, [F::Packing; DIGEST_ELEMS]> + Sync,
    C: PseudoCompressionFunction<[F; DIGEST_ELEMS], 2> + PseudoCompressionFunction<[F::Packing; DIGEST_ELEMS], 2> + Sync,
    [F; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
{
    /// Creates a new commitment handler.
    pub fn new(merkle_hash: &'a H, merkle_compress: &'a C) -> Self {
        Self {
            merkle_hash,
            merkle_compress,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Builds the folded matrix for commitment.
    ///
    /// This performs the DFT transformation and matrix preparation steps
    /// needed before calling commit_matrix.
    #[instrument(skip_all, fields(evals = evaluations.num_evals(), inv_rate, folding_factor))]
    pub fn build_folded_matrix<EF>(
        &self,
        dft: &EvalsDft<F>,
        evaluations: &EvaluationsList<EF>,
        inv_rate: usize,
        folding_factor: usize,
    ) -> RowMajorMatrix<EF>
    where
        EF: ExtensionField<F>,
    {
        info_span!("fold matrix").in_scope(|| {
            let evals_repeated = info_span!("repeating evals")
                .in_scope(|| parallel_repeat(evaluations.as_slice(), inv_rate));

            // Apply DFT on interleaved polynomials to prepare for folding
            info_span!(
                "dft",
                height = evals_repeated.len() >> folding_factor,
                width = 1 << folding_factor
            )
            .in_scope(|| {
                dft.dft_algebra_batch_by_evals(RowMajorMatrix::new(
                    evals_repeated,
                    1 << folding_factor,
                ))
            })
        })
    }

    /// Opens the Merkle tree at specific query indices for base field commitments.
    ///
    /// Generates authentication paths and leaf values for the requested indices,
    /// allowing the verifier to check that specific evaluations are consistent
    /// with the committed polynomial.
    pub fn open_base_commitments(
        &self,
        query_indices: &[usize],
        prover_data: &[Vec<[F; DIGEST_ELEMS]>],
    ) -> ProofResult<(Vec<Vec<F>>, Vec<Vec<[F; DIGEST_ELEMS]>>)> {
        let mmcs = MerkleTreeMmcs::<F::Packing, F::Packing, H, C, DIGEST_ELEMS>::new(
            self.merkle_hash.clone(),
            self.merkle_compress.clone(),
        );

        let mut answers = Vec::with_capacity(query_indices.len());
        let mut merkle_proofs = Vec::with_capacity(query_indices.len());

        for &index in query_indices {
            let commitment = mmcs.open_batch(index, prover_data);
            answers.push(commitment.opened_values[0].clone());
            merkle_proofs.push(commitment.opening_proof);
        }

        Ok((answers, merkle_proofs))
    }

    /// Opens the Merkle tree at specific query indices for extension field commitments.
    ///
    /// Similar to base field opening but handles extension field elements,
    /// which require different serialization and commitment handling.
    pub fn open_extension_commitments<EF>(
        &self,
        query_indices: &[usize],
        prover_data: &[Vec<[F; DIGEST_ELEMS]>],
    ) -> ProofResult<(Vec<Vec<EF>>, Vec<Vec<[F; DIGEST_ELEMS]>>)>
    where
        EF: ExtensionField<F>,
    {
        let mmcs = MerkleTreeMmcs::<F::Packing, F::Packing, H, C, DIGEST_ELEMS>::new(
            self.merkle_hash.clone(),
            self.merkle_compress.clone(),
        );
        let extension_mmcs = ExtensionMmcs::new(mmcs);

        let mut answers = Vec::with_capacity(query_indices.len());
        let mut merkle_proofs = Vec::with_capacity(query_indices.len());

        for &index in query_indices {
            let commitment = extension_mmcs.open_batch(index, prover_data);
            answers.push(commitment.opened_values[0].clone());
            merkle_proofs.push(commitment.opening_proof);
        }

        Ok((answers, merkle_proofs))
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
    fn test_commitment_handler_creation() {
        let mut rng = SmallRng::seed_from_u64(42);
        let perm = Perm::new_from_rng(8, 22, &mut rng);
        let hash = Hash::new(perm.clone());
        let compress = Compress::new(perm);

        let _handler = CommitmentHandler::<_, _, _, 8>::new(&hash, &compress);
    }

    #[test]
    fn test_folded_matrix_construction() {
        let mut rng = SmallRng::seed_from_u64(42);
        let perm = Perm::new_from_rng(8, 22, &mut rng);
        let hash = Hash::new(perm.clone());
        let compress = Compress::new(perm);

        let handler = CommitmentHandler::<_, _, _, 8>::new(&hash, &compress);
        let dft = EvalsDft::<F>::default();

        // Create a simple polynomial with 8 evaluations (3 variables)
        let evaluations = EvaluationsList::new(vec![EF::ONE; 8]);

        let folded_matrix = handler.build_folded_matrix(&dft, &evaluations, 2, 1);
        assert_eq!(folded_matrix.width(), 2); // Should have width = 2^1
        assert!(folded_matrix.height() > 0); // Should have some height
    }
}