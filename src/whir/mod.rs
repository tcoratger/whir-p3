pub mod committer;
pub mod fs_utils;
pub mod iopattern;
pub mod parameters;
pub mod parsed_proof;
pub mod prover;
pub mod statement;
pub mod verifier;

// Only includes the authentication paths
#[derive(Debug, Default, Clone)]
pub struct WhirProof<F> {
    // pub merkle_paths: Vec<(MultiPath<MerkleConfig>, Vec<Vec<F>>)>,
    pub statement_values_at_random_point: Vec<F>,
}
