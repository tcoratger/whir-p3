//! Fiat-Shamir transcript errors for WHIR protocol challenges.

use thiserror::Error;

/// Granular error types for Fiat-Shamir transcript operations.
///
/// Each variant represents a specific failure mode in the Fiat-Shamir transform.
#[derive(Error, Debug, Clone)]
pub enum FiatShamirError {
    /// Transcript data exhausted during challenge sampling.
    #[error("Transcript exceeded: verifier requested more data than available")]
    ExceededTranscript,

    /// Proof-of-work witness fails difficulty requirement.
    #[error("Invalid grinding witness: proof-of-work verification failed")]
    InvalidGrindingWitness,
}
