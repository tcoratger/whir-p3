//! Verifier error types for WHIR protocol validation.

use thiserror::Error;

use crate::fiat_shamir::errors::FiatShamirError;

/// Errors for WHIR protocol verification.
#[derive(Error, Debug)]
pub enum VerifierError {
    /// Merkle tree proof verification failed for polynomial commitment.
    #[error("Merkle proof verification failed at position {position}: {reason}")]
    MerkleProofInvalid { position: usize, reason: String },

    /// Sumcheck verification failed due to inconsistent polynomial evaluations.
    #[error("Sumcheck verification failed at round {round}: expected {expected}, got {actual}")]
    SumcheckFailed {
        round: usize,
        expected: String,
        actual: String,
    },

    /// STIR challenge responses are inconsistent or invalid.
    #[error("STIR challenge {challenge_id} verification failed: {details}")]
    StirChallengeFailed {
        challenge_id: usize,
        details: String,
    },

    /// Fiat-Shamir transcript error during verification.
    #[error(transparent)]
    FiatShamir(#[from] FiatShamirError),
}
