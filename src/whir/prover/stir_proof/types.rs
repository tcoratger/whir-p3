use p3_commit::{ExtensionMmcs, Mmcs};
use p3_matrix::dense::DenseMatrix;
use p3_merkle_tree::MerkleTreeMmcs;

/// Base field Merkle tree commitment scheme.
pub type BaseMmcs<F, H, C, const DIGEST_ELEMS: usize> = MerkleTreeMmcs<F, F, H, C, DIGEST_ELEMS>;

/// Extension field Merkle tree commitment scheme.
pub type ExtMmcs<F, EF, H, C, const DIGEST_ELEMS: usize> =
    ExtensionMmcs<F, EF, BaseMmcs<F, H, C, DIGEST_ELEMS>>;

/// Prover data type for extension field operations.
pub type ExtProverData<F, EF, H, C, const DIGEST_ELEMS: usize> =
    <ExtMmcs<F, EF, H, C, DIGEST_ELEMS> as Mmcs<EF>>::ProverData<DenseMatrix<EF>>;
