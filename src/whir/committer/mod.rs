use p3_field::{ExtensionField, Field};
use p3_matrix::dense::DenseMatrix;
use p3_merkle_tree::MerkleTree;

use crate::poly::coeffs::CoefficientList;

pub mod reader;
pub mod writer;

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
    pub prover_data: MerkleTree<F, u8, DenseMatrix<F>, DIGEST_ELEMS>,
    /// Out-of-domain challenge points used for polynomial verification.
    pub ood_points: Vec<EF>,
    /// The corresponding polynomial evaluations at the OOD challenge points.
    pub ood_answers: Vec<EF>,
}
