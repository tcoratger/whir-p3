pub mod committer;
pub mod fs_utils;
pub mod iopattern;
pub mod parameters;
pub mod parsed_proof;
pub mod prover;
pub mod statement;
pub mod stir_evaluations;
pub mod utils;
pub mod verifier;

// Only includes the authentication paths
#[derive(Debug, Default, Clone)]
pub struct WhirProof<F, const DIGEST_ELEMS: usize> {
    pub merkle_paths: Vec<(Vec<[F; DIGEST_ELEMS]>, Vec<Vec<F>>)>,
    pub statement_values_at_random_point: Vec<F>,
}

/// Signals an invalid IO pattern.
///
/// This error indicates a wrong IO Pattern declared
/// upon instantiation of the SAFE sponge.
#[derive(Debug, Clone)]
pub struct IOPatternError(String);

/// An error happened when creating or verifying a proof.
#[derive(Debug, Clone)]
pub enum ProofError {
    /// Signals the verification equation has failed.
    InvalidProof,
    /// The IO pattern specified mismatches the IO pattern used during the protocol execution.
    InvalidIO(IOPatternError),
    /// Serialization/Deserialization led to errors.
    SerializationError,
}

/// The result type when trying to prove or verify a proof using Fiat-Shamir.
pub type ProofResult<T> = Result<T, ProofError>;
