//! Main error types for the WHIR protocol implementation.

use thiserror::Error;

use crate::{fiat_shamir::errors::FiatShamirError, whir::verifier::errors::VerifierError};

/// Top-level error type for WHIR protocol operations.
#[derive(Error, Debug)]
pub enum WhirError {
    /// Verification process failed during proof validation.
    #[error(transparent)]
    Verifier(#[from] VerifierError),

    /// Fiat-Shamir transcript generation or processing failed.
    #[error(transparent)]
    FiatShamir(#[from] FiatShamirError),
}
