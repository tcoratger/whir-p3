use p3_matrix::{dense::DenseMatrix, extension::FlatMatrixView};
use p3_merkle_tree::MerkleTree;

pub mod reader;
pub mod writer;

pub type ProverDataView<F, EF, W, const DIGEST_ELEMS: usize> =
    MerkleTree<F, W, FlatMatrixView<F, EF, DenseMatrix<EF>>, DIGEST_ELEMS>;
