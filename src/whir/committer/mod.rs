use p3_matrix::{dense::DenseMatrix, extension::FlatMatrixView};
use p3_merkle_tree::MerkleTree;

pub mod reader;
pub mod writer;

pub type ProverDataExt<F, EF, const DIGEST_ELEMS: usize> =
    MerkleTree<F, F, FlatMatrixView<F, EF, DenseMatrix<EF>>, DIGEST_ELEMS>;

pub type ProverData<F, const DIGEST_ELEMS: usize> = MerkleTree<F, F, DenseMatrix<F>, DIGEST_ELEMS>;
