//! Proof-of-work grinding for WHIR protocol.
//!
//! Grinding is a critical security mechanism that prevents prover from grinding challenges.
//! After committing to polynomials, the prover must perform computationally expensive
//! proof-of-work before receiving verifier challenges. This makes it infeasible to selectively
//! choose commitments that lead to favorable challenges.
//! Fiat-Shamir transcript errors for WHIR protocol challenges.

use p3_challenger::GrindingChallenger;
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

/// Perform proof-of-work grinding and return the witness.
///
/// This function forces the prover to perform expensive computation after committing
/// to a polynomial but before receiving verifier challenges. This prevents challenge
/// grinding attacks.
///
/// # Arguments
/// - `challenger`: The Fiat-Shamir challenger maintaining the transcript state
/// - `bits`: Number of bits of grinding difficulty. If zero, no grinding is performed.
///
/// # Returns
/// - `Some(witness)` if bits > 0, containing the witness that satisfies the PoW requirement
/// - `None` if bits == 0 (no grinding required)
///
/// # Example
/// ```ignore
/// let witness = pow_grinding(&mut challenger, 20);
/// if let Some(w) = witness {
///     // Store witness in proof structure
///     proof.pow_witness = w;
/// }
/// ```
pub fn pow_grinding<C, F>(challenger: &mut C, bits: usize) -> Option<F>
where
    C: GrindingChallenger<Witness = F>,
{
    if bits == 0 {
        return None;
    }

    // Perform grinding and obtain a witness element in the base field
    Some(challenger.grind(bits))
}

/// Verify a proof-of-work grinding witness.
///
/// Checks that the provided witness satisfies the required difficulty level according
/// to the current transcript state. The witness must have been produced by grinding
/// on the same transcript state.
///
/// # Arguments
/// - `challenger`: The Fiat-Shamir challenger maintaining the transcript state
/// - `witness`: The witness value to verify
/// - `bits`: Number of bits of grinding difficulty. If zero, no check is performed.
///
/// # Returns
/// - `Ok(())` if the witness is valid or bits == 0
/// - `Err(FiatShamirError::InvalidGrindingWitness)` if the witness doesn't satisfy the difficulty
///
/// # Errors
/// Returns `FiatShamirError::InvalidGrindingWitness` if the witness does not satisfy
/// the required difficulty level.
///
/// # Example
/// ```ignore
/// let witness = proof.pow_witness;
/// check_pow_grinding(&challenger, witness, 20)?;
/// ```
pub fn check_pow_grinding<C, F>(
    challenger: &mut C,
    witness: Option<F>,
    bits: usize,
) -> Result<(), FiatShamirError>
where
    C: GrindingChallenger<Witness = F>,
{
    // If no grinding is required or no witness is provided, succeed immediately
    if bits == 0 {
        return Ok(());
    }

    // If witness is None, succeed immediately
    let witness = match witness {
        Some(w) => w,
        None => return Ok(()),
    };

    // Verify the witness using the challenger
    if challenger.check_witness(bits, witness) {
        Ok(())
    } else {
        Err(FiatShamirError::InvalidGrindingWitness)
    }
}
