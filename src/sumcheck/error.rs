use alloc::string::String;

use thiserror::Error;

/// Errors from sumcheck protocol verification.
#[derive(Error, Debug)]
pub enum SumcheckError {
    /// Sumcheck verification failed due to inconsistent polynomial evaluations.
    #[error("Sumcheck verification failed at round {round}: expected {expected}, got {actual}")]
    SumcheckFailed {
        round: usize,
        expected: String,
        actual: String,
    },

    /// Proof-of-work witness verification failed.
    #[error("Invalid proof-of-work witness")]
    InvalidPowWitness,
}
