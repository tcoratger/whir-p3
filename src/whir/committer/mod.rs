use p3_field::Field;
use p3_matrix::dense::DenseMatrix;
use p3_merkle_tree::{MerkleTree, MerkleTreeMmcs};

use crate::poly::coeffs::CoefficientList;

pub mod reader;
pub mod writer;

/// Represents the commitment and evaluation data for a polynomial.
///
/// This structure holds all necessary components to verify a commitment,
/// including the polynomial itself, the Merkle tree used for commitment,
/// and out-of-domain (OOD) evaluations.
#[derive(Debug)]
pub struct Witness<F: Field, H, C, const DIGEST_ELEMS: usize> {
    /// The committed polynomial in coefficient form.
    pub(crate) polynomial: CoefficientList<F>,
    /// The Merkle tree constructed from the polynomial evaluations.
    pub(crate) merkle_tree: MerkleTreeMmcs<F, u8, H, C, DIGEST_ELEMS>,
    /// Prover data of the Merkle tree.
    pub(crate) prover_data: MerkleTree<F, u8, DenseMatrix<F>, DIGEST_ELEMS>,
    /// The leaves of the Merkle tree, derived from folded polynomial evaluations.
    pub(crate) merkle_leaves: Vec<F>,
    /// Out-of-domain challenge points used for polynomial verification.
    pub(crate) ood_points: Vec<F>,
    /// The corresponding polynomial evaluations at the OOD challenge points.
    pub(crate) ood_answers: Vec<F>,
}
