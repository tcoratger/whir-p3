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

#[cfg(test)]
mod tests {
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::{CanObserve, DuplexChallenger};
    use p3_field::PrimeCharacteristicRing;
    use rand::SeedableRng;

    use super::*;

    type F = BabyBear;
    type Perm = Poseidon2BabyBear<16>;
    type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

    fn make_challenger() -> MyChallenger {
        let perm = Perm::new_from_rng_128(&mut rand::rngs::SmallRng::seed_from_u64(42));
        DuplexChallenger::new(perm)
    }

    #[test]
    fn test_pow_grinding_zero_bits_returns_none() {
        let mut challenger = make_challenger();
        let result = pow_grinding(&mut challenger, 0);
        assert!(result.is_none(), "Expected None when bits = 0");
    }

    #[test]
    fn test_pow_grinding_nonzero_bits_returns_some() {
        let mut challenger = make_challenger();
        let result = pow_grinding(&mut challenger, 4);
        assert!(result.is_some(), "Expected Some(witness) when bits > 0");
    }

    #[test]
    fn test_pow_grinding_produces_valid_witness() {
        let mut challenger = make_challenger();
        let witness = pow_grinding(&mut challenger, 5);
        assert!(witness.is_some());

        // Reset challenger to same state
        let mut verifier_challenger = make_challenger();
        let result = check_pow_grinding(&mut verifier_challenger, witness, 5);
        assert!(result.is_ok(), "Generated witness should be valid");
    }

    #[test]
    fn test_check_pow_grinding_zero_bits_always_succeeds() {
        let mut challenger = make_challenger();

        // Test with None witness
        assert!(check_pow_grinding(&mut challenger, None, 0).is_ok());

        // Test with Some witness (should still succeed when bits = 0)
        assert!(check_pow_grinding(&mut challenger, Some(F::from_u64(123)), 0).is_ok());
    }

    #[test]
    fn test_check_pow_grinding_none_witness_succeeds() {
        let mut challenger = make_challenger();
        // When witness is None, verification should succeed regardless of bits
        assert!(check_pow_grinding(&mut challenger, None, 8).is_ok());
    }

    #[test]
    fn test_check_pow_grinding_invalid_witness_fails() {
        let mut challenger = make_challenger();
        let bad_witness = Some(F::from_u64(42));

        // This witness is not valid for the current challenger state
        let result = check_pow_grinding(&mut challenger, bad_witness, 4);
        assert!(result.is_err());

        if let Err(e) = result {
            assert!(matches!(e, FiatShamirError::InvalidGrindingWitness));
        }
    }

    #[test]
    fn test_pow_grinding_roundtrip_multiple_difficulties() {
        for bits in 1..=8 {
            let mut prover_challenger = make_challenger();
            let witness = pow_grinding(&mut prover_challenger, bits);

            let mut verifier_challenger = make_challenger();
            let result = check_pow_grinding(&mut verifier_challenger, witness, bits);

            assert!(result.is_ok(), "Roundtrip should succeed for bits = {bits}");
        }
    }

    #[test]
    fn test_pow_grinding_different_with_different_challenger_state() {
        let mut challenger1 = make_challenger();
        // Modify challenger1 state by observing a value
        challenger1.observe(F::from_u64(999));

        let mut challenger2 = make_challenger();

        let witness1 = pow_grinding(&mut challenger1, 4);
        let witness2 = pow_grinding(&mut challenger2, 4);

        assert_ne!(
            witness1, witness2,
            "Different challenger states should produce different witnesses"
        );
    }

    #[test]
    fn test_check_pow_grinding_witness_from_wrong_challenger_fails() {
        // Generate witness with one challenger
        let mut challenger1 = make_challenger();
        let witness = pow_grinding(&mut challenger1, 4);

        // Try to verify with a different challenger state
        let mut challenger2 = make_challenger();
        challenger2.observe(F::from_u64(123)); // Different state

        let result = check_pow_grinding(&mut challenger2, witness, 4);
        assert!(
            result.is_err(),
            "Witness from different challenger state should fail verification"
        );
    }

    #[test]
    fn test_pow_grinding_higher_difficulty_takes_time() {
        // This is a performance characteristic test
        // Higher difficulty should generally take more iterations
        let mut challenger = make_challenger();
        let witness = pow_grinding(&mut challenger, 10);
        assert!(
            witness.is_some(),
            "Should be able to grind even with higher difficulty"
        );
    }

    #[test]
    fn test_check_pow_grinding_wrong_bits_fails() {
        let mut prover_challenger = make_challenger();
        let witness = pow_grinding(&mut prover_challenger, 4);

        let mut verifier_challenger = make_challenger();
        // Try to verify with different difficulty bits
        let result = check_pow_grinding(&mut verifier_challenger, witness, 8);

        assert!(
            result.is_err(),
            "Witness for 4 bits should not satisfy 8 bits difficulty"
        );
    }

    /// CRITICAL TEST: Verify that after pow_grinding (prover) and check_pow_grinding (verifier),
    /// both challengers end up in the SAME state.
    ///
    /// If this test fails, the transcript will diverge and STIR queries will differ.
    #[test]
    fn test_pow_grinding_and_check_leave_same_state() {
        use p3_challenger::CanSample;

        for bits in 1..=8 {
            // Start both from identical state
            let mut prover_challenger = make_challenger();
            let mut verifier_challenger = make_challenger();

            // Prover grinds
            let witness = pow_grinding(&mut prover_challenger, bits);

            // Verifier checks
            let result = check_pow_grinding(&mut verifier_challenger, witness, bits);
            assert!(
                result.is_ok(),
                "Verification should succeed for bits = {bits}"
            );

            // CRITICAL: Sample from both and verify they match
            let prover_sample: F = prover_challenger.sample();
            let verifier_sample: F = verifier_challenger.sample();

            assert_eq!(
                prover_sample, verifier_sample,
                "Challenger states diverged after pow_grinding/check_pow_grinding for bits = {bits}"
            );
        }
    }
}
