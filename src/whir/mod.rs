pub mod committer;
pub mod fs_utils;
pub mod iopattern;
pub mod parameters;
pub mod parsed_proof;
pub mod prover;
pub mod statement;

// Only includes the authentication paths
#[derive(Default, Clone)]
pub struct WhirProof<F> {
    // pub merkle_paths: Vec<(MultiPath<MerkleConfig>, Vec<Vec<F>>)>,
    pub statement_values_at_random_point: Vec<F>,
}
