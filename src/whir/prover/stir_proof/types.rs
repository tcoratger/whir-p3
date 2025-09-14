use p3_commit::{ExtensionMmcs, Mmcs};
use p3_field::Field;
use p3_matrix::dense::DenseMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use serde::{Deserialize, Serialize};

/// Base field Merkle tree commitment scheme.
pub type BaseMmcs<F, H, C, const DIGEST_ELEMS: usize> = MerkleTreeMmcs<F, F, H, C, DIGEST_ELEMS>;

/// Extension field Merkle tree commitment scheme.
pub type ExtMmcs<F, EF, H, C, const DIGEST_ELEMS: usize> =
    ExtensionMmcs<F, EF, BaseMmcs<F, H, C, DIGEST_ELEMS>>;

/// Prover data type for extension field operations.
pub type ExtProverData<F, EF, H, C, const DIGEST_ELEMS: usize> =
    <ExtMmcs<F, EF, H, C, DIGEST_ELEMS> as Mmcs<EF>>::ProverData<DenseMatrix<EF>>;

/// Prover data type for base field operations.
pub type BaseProverData<F, H, C, const DIGEST_ELEMS: usize> =
    <BaseMmcs<F, H, C, DIGEST_ELEMS> as Mmcs<F>>::ProverData<DenseMatrix<F>>;

/// Proof type for base field operations.
pub type BaseProof<F, H, C, const DIGEST_ELEMS: usize> =
    <BaseMmcs<F, H, C, DIGEST_ELEMS> as Mmcs<F>>::Proof;

/// Proof type for extension field operations.
pub type ExtProof<F, EF, H, C, const DIGEST_ELEMS: usize> =
    <ExtMmcs<F, EF, H, C, DIGEST_ELEMS> as Mmcs<EF>>::Proof;

/// Constraint helper for cryptographic hash functions.
pub trait StirHasher<F, const DIGEST_ELEMS: usize>:
    CryptographicHasher<F, [F; DIGEST_ELEMS]>
    + CryptographicHasher<F::Packing, [F::Packing; DIGEST_ELEMS]>
    + Sync
    + Clone
where
    F: Field,
{
}

impl<F, H, const DIGEST_ELEMS: usize> StirHasher<F, DIGEST_ELEMS> for H
where
    F: Field,
    H: CryptographicHasher<F, [F; DIGEST_ELEMS]>
        + CryptographicHasher<F::Packing, [F::Packing; DIGEST_ELEMS]>
        + Sync
        + Clone,
{
}

/// Constraint helper for compression functions.
pub trait StirCompressor<F, const DIGEST_ELEMS: usize>:
    PseudoCompressionFunction<[F; DIGEST_ELEMS], 2>
    + PseudoCompressionFunction<[F::Packing; DIGEST_ELEMS], 2>
    + Sync
    + Clone
where
    F: Field,
{
}

impl<F, C, const DIGEST_ELEMS: usize> StirCompressor<F, DIGEST_ELEMS> for C
where
    F: Field,
    C: PseudoCompressionFunction<[F; DIGEST_ELEMS], 2>
        + PseudoCompressionFunction<[F::Packing; DIGEST_ELEMS], 2>
        + Sync
        + Clone,
{
}

/// Constraint helper for serializable digests.
pub trait SerializableDigest<F, const DIGEST_ELEMS: usize>:
    Serialize + for<'de> Deserialize<'de>
where
    F: Field,
{
}

impl<F, const DIGEST_ELEMS: usize> SerializableDigest<F, DIGEST_ELEMS> for [F; DIGEST_ELEMS]
where
    F: Field,
    [F; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
{
}
