use p3_field::{ExtensionField, Field};
use p3_matrix::{dense::DenseMatrix, extension::FlatMatrixView};
use p3_merkle_tree::MerkleTree;

use crate::poly::coeffs::CoefficientList;

pub mod reader;
pub mod writer;

pub type CommitmentMerkleTree<F, const DIGEST_ELEMS: usize> =
    MerkleTree<F, u8, DenseMatrix<F>, DIGEST_ELEMS>;
pub type RoundMerkleTree<F, EF, const DIGEST_ELEMS: usize> =
    MerkleTree<F, u8, FlatMatrixView<F, EF, DenseMatrix<EF>>, DIGEST_ELEMS>;

/// Represents the commitment and evaluation data for a polynomial.
///
/// This structure holds all necessary components to verify a commitment,
/// including the polynomial itself, the Merkle tree used for commitment,
/// and out-of-domain (OOD) evaluations.
#[derive(Debug)]
pub struct Witness<EF: ExtensionField<F>, F: Field, const DIGEST_ELEMS: usize> {
    /// The committed polynomial in coefficient form.
    pub polynomial: CoefficientList<F>,
    /// Prover data of the Merkle tree.
    pub prover_data: CommitmentMerkleTree<F, DIGEST_ELEMS>,
    /// Out-of-domain challenge points used for polynomial verification.
    pub ood_points: Vec<EF>,
    /// The corresponding polynomial evaluations at the OOD challenge points.
    pub ood_answers: Vec<EF>,
}
