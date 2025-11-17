use p3_challenger::GrindingChallenger;

use crate::fiat_shamir::errors::FiatShamirError;

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
#[allow(unreachable_pub)]
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
#[allow(unreachable_pub)]
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
    let Some(witness) = witness else {
        return Ok(());
    };

    // Verify the witness using the challenger
    if challenger.check_witness(bits, witness) {
        Ok(())
    } else {
        Err(FiatShamirError::InvalidGrindingWitness)
    }
}
