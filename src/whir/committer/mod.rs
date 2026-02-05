use p3_matrix::{dense::DenseMatrix, extension::FlatMatrixView};
use p3_merkle_tree::MerkleTree;

pub mod reader;
pub mod writer;

pub type ProverDataView<F, EF, W, const DIGEST_ELEMS: usize> =
    ProverData<F, FlatMatrixView<F, EF, DenseMatrix<EF>>, W, DIGEST_ELEMS>;

// TODO: replace with normal MerkleTree everywhere
pub type ProverData<F, M, W, const DIGEST_ELEMS: usize> = MerkleTree<F, W, M, DIGEST_ELEMS>;
