use alloc::vec::Vec;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field};
use serde::{Deserialize, Serialize};

/// Sumcheck polynomial data
///
/// Stores the polynomial evaluations for sumcheck rounds in a compact format.
/// Each round stores `[h(0), h(2)]` where `h(1)` is derived as `claimed_sum - h(0)`.
#[derive(Default, Serialize, Deserialize, Clone, Debug)]
pub struct SumcheckData<F, EF> {
    /// Polynomial evaluations for each sumcheck round
    ///
    /// Each entry is `[h(0), h(2)]` - the evaluations at 0 and 2
    ///
    /// `h(1)` is derived as `claimed_sum - h(0)` by the verifier
    ///
    /// Length: folding_factor
    pub polynomial_evaluations: Vec<[EF; 2]>,

    /// PoW witnesses for each sumcheck round
    /// Length: folding_factor
    pub pow_witnesses: Vec<F>,
}

impl<F, EF> SumcheckData<F, EF> {
    /// Commits polynomial coefficients to the transcript and returns a challenge.
    ///
    /// This helper function handles the Fiat-Shamir interaction for a sumcheck round.
    ///
    /// # Arguments
    ///
    /// * `challenger` - Fiat-Shamir transcript.
    /// * `c0` - Constant coefficient `h(0)`.
    /// * `c2` - Quadratic coefficient.
    /// * `pow_bits` - PoW difficulty (0 to skip grinding).
    ///
    /// # Returns
    ///
    /// The sampled challenge `r \in EF`.
    pub fn observe_and_sample<Challenger, BF>(
        &mut self,
        challenger: &mut Challenger,
        c0: EF,
        c2: EF,
        pow_bits: usize,
    ) -> EF
    where
        BF: Field,
        EF: ExtensionField<BF>,
        F: Clone,
        Challenger: FieldChallenger<BF> + GrindingChallenger<Witness = F>,
    {
        // Record the polynomial coefficients in the proof.
        self.polynomial_evaluations.push([c0, c2]);

        // Absorb coefficients into the transcript.
        //
        // Note: We only send (c_0, c_2). The verifier derives c_1 from the sum constraint.
        challenger.observe_algebra_slice(&[c0, c2]);

        // Optional proof-of-work to increase prover cost.
        //
        // This makes it expensive for a malicious prover to "mine" favorable challenges.
        if pow_bits > 0 {
            self.pow_witnesses.push(challenger.grind(pow_bits));
        }

        // Sample the verifier's challenge for this round.
        challenger.sample_algebra_element()
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};
    use rand::SeedableRng;

    use super::*;

    /// Type alias for the base field used in tests
    type F = BabyBear;

    /// Type alias for the extension field used in tests
    type EF = BinomialExtensionField<F, 4>;

    /// Type alias for the permutation used in tests
    type Perm = Poseidon2BabyBear<16>;

    /// Type alias for the challenger used in observe_and_sample tests.
    type TestChallenger = DuplexChallenger<F, Perm, 16, 8>;

    /// Creates a fresh challenger for testing.
    ///
    /// The challenger is seeded deterministically so tests are reproducible.
    fn create_test_challenger() -> TestChallenger {
        let perm = Perm::new_from_rng_128(&mut rand::rngs::SmallRng::seed_from_u64(42));
        DuplexChallenger::new(perm)
    }

    #[test]
    fn test_sumcheck_data_default() {
        // Create a default SumcheckData
        let sumcheck: SumcheckData<F, EF> = SumcheckData::default();

        // Verify polynomial_evaluations is empty
        assert_eq!(sumcheck.polynomial_evaluations.len(), 0);

        // Verify pow_witnesses is empty
        assert!(sumcheck.pow_witnesses.is_empty());
    }

    #[test]
    fn test_push_pow_witness() {
        let mut sumcheck: SumcheckData<F, EF> = SumcheckData::default();

        // First push
        let witness1 = F::from_u64(42);
        sumcheck.pow_witnesses.push(witness1);

        assert_eq!(sumcheck.pow_witnesses.len(), 1);
        assert_eq!(sumcheck.pow_witnesses[0], witness1);

        // Second push should append to existing vector
        let witness2 = F::from_u64(123);
        sumcheck.pow_witnesses.push(witness2);

        assert_eq!(sumcheck.pow_witnesses.len(), 2);
        assert_eq!(sumcheck.pow_witnesses[1], witness2);
    }

    #[test]
    fn test_observe_and_sample_records_coefficients() {
        // The method should push [c0, c2] to polynomial_evaluations.
        //
        // polynomial_evaluations stores the sumcheck polynomial coefficients
        // for each round: [h(0), h(2)] where h(1) is derived by the verifier.
        let c0 = EF::from_u64(5);
        let c2 = EF::from_u64(7);

        let mut sumcheck: SumcheckData<F, EF> = SumcheckData::default();
        let mut challenger = create_test_challenger();

        // polynomial_evaluations should be empty initially
        assert!(sumcheck.polynomial_evaluations.is_empty());

        // Call observe_and_sample with pow_bits = 0 (no grinding)
        let pow_bits = 0;
        let _r = sumcheck.observe_and_sample::<_, F>(&mut challenger, c0, c2, pow_bits);

        // polynomial_evaluations should now have one entry: [c0, c2]
        assert_eq!(sumcheck.polynomial_evaluations.len(), 1);
        assert_eq!(sumcheck.polynomial_evaluations[0][0], c0);
        assert_eq!(sumcheck.polynomial_evaluations[0][1], c2);
    }

    #[test]
    fn test_observe_and_sample_multiple_rounds() {
        // Multiple calls should accumulate coefficients in order.
        //
        // Round 0: push [c0_0, c2_0]
        // Round 1: push [c0_1, c2_1]
        // Round 2: push [c0_2, c2_2]
        let c0_0 = EF::from_u64(1);
        let c2_0 = EF::from_u64(2);
        let c0_1 = EF::from_u64(3);
        let c2_1 = EF::from_u64(4);
        let c0_2 = EF::from_u64(5);
        let c2_2 = EF::from_u64(6);

        let mut sumcheck: SumcheckData<F, EF> = SumcheckData::default();
        let mut challenger = create_test_challenger();
        let pow_bits = 0;

        // Round 0
        let _r0 = sumcheck.observe_and_sample::<_, F>(&mut challenger, c0_0, c2_0, pow_bits);
        assert_eq!(sumcheck.polynomial_evaluations.len(), 1);

        // Round 1
        let _r1 = sumcheck.observe_and_sample::<_, F>(&mut challenger, c0_1, c2_1, pow_bits);
        assert_eq!(sumcheck.polynomial_evaluations.len(), 2);

        // Round 2
        let _r2 = sumcheck.observe_and_sample::<_, F>(&mut challenger, c0_2, c2_2, pow_bits);
        assert_eq!(sumcheck.polynomial_evaluations.len(), 3);

        // Verify all stored coefficients match input order
        assert_eq!(sumcheck.polynomial_evaluations[0], [c0_0, c2_0]);
        assert_eq!(sumcheck.polynomial_evaluations[1], [c0_1, c2_1]);
        assert_eq!(sumcheck.polynomial_evaluations[2], [c0_2, c2_2]);
    }

    #[test]
    fn test_observe_and_sample_without_pow() {
        // When pow_bits = 0, no PoW witness should be recorded.
        //
        // The method skips the grinding step when pow_bits is zero,
        // so pow_witnesses should remain empty.
        let c0 = EF::from_u64(10);
        let c2 = EF::from_u64(20);

        let mut sumcheck: SumcheckData<F, EF> = SumcheckData::default();
        let mut challenger = create_test_challenger();

        // pow_witnesses should be empty initially
        assert!(sumcheck.pow_witnesses.is_empty());

        // Call with pow_bits = 0
        let pow_bits = 0;
        let _r = sumcheck.observe_and_sample::<_, F>(&mut challenger, c0, c2, pow_bits);

        // pow_witnesses should still be empty (no grinding performed)
        assert!(sumcheck.pow_witnesses.is_empty());
    }

    #[test]
    fn test_observe_and_sample_with_pow() {
        // When pow_bits > 0, a PoW witness should be recorded.
        //
        // The method calls challenger.grind(pow_bits) and pushes
        // the resulting witness to pow_witnesses.
        let c0 = EF::from_u64(10);
        let c2 = EF::from_u64(20);

        let mut sumcheck: SumcheckData<F, EF> = SumcheckData::default();
        let mut challenger = create_test_challenger();

        // pow_witnesses should be empty initially
        assert!(sumcheck.pow_witnesses.is_empty());

        // Call with pow_bits = 1 (minimal PoW)
        let pow_bits = 1;
        let _r = sumcheck.observe_and_sample::<_, F>(&mut challenger, c0, c2, pow_bits);

        // pow_witnesses should now have one entry
        assert_eq!(sumcheck.pow_witnesses.len(), 1);
    }

    #[test]
    fn test_observe_and_sample_pow_accumulates() {
        // Multiple rounds with PoW should accumulate witnesses.
        //
        // Each call with pow_bits > 0 should add one witness.
        let c0 = EF::from_u64(1);
        let c2 = EF::from_u64(2);

        let mut sumcheck: SumcheckData<F, EF> = SumcheckData::default();
        let mut challenger = create_test_challenger();
        let pow_bits = 1;

        // Three rounds with PoW
        let _r0 = sumcheck.observe_and_sample::<_, F>(&mut challenger, c0, c2, pow_bits);
        let _r1 = sumcheck.observe_and_sample::<_, F>(&mut challenger, c0, c2, pow_bits);
        let _r2 = sumcheck.observe_and_sample::<_, F>(&mut challenger, c0, c2, pow_bits);

        // Should have 3 witnesses
        assert_eq!(sumcheck.pow_witnesses.len(), 3);
        // And 3 polynomial evaluations
        assert_eq!(sumcheck.polynomial_evaluations.len(), 3);
    }

    #[test]
    fn test_observe_and_sample_deterministic_challenge() {
        // Fiat-Shamir property: same inputs produce same challenge.
        //
        // Two challengers with the same initial state, observing the same
        // coefficients, should sample the same challenge.
        let c0 = EF::from_u64(42);
        let c2 = EF::from_u64(99);
        let pow_bits = 0;

        // First run
        let mut sumcheck1: SumcheckData<F, EF> = SumcheckData::default();
        let mut challenger1 = create_test_challenger();
        let r1 = sumcheck1.observe_and_sample::<_, F>(&mut challenger1, c0, c2, pow_bits);

        // Second run with fresh but identically-seeded challenger
        let mut sumcheck2: SumcheckData<F, EF> = SumcheckData::default();
        let mut challenger2 = create_test_challenger();
        let r2 = sumcheck2.observe_and_sample::<_, F>(&mut challenger2, c0, c2, pow_bits);

        // Challenges should be identical
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_observe_and_sample_challenge_depends_on_history() {
        // The challenge at round i depends on all previous observations.
        //
        // Two sequences with different history should produce different
        // challenges even if the final round has the same coefficients.
        let c0 = EF::from_u64(100);
        let c2 = EF::from_u64(200);
        let pow_bits = 0;

        // Sequence A: observe once, then observe (c0, c2)
        let mut sumcheck_a: SumcheckData<F, EF> = SumcheckData::default();
        let mut challenger_a = create_test_challenger();
        let _r0_a =
            sumcheck_a.observe_and_sample::<_, F>(&mut challenger_a, EF::ONE, EF::TWO, pow_bits);
        let r1_a = sumcheck_a.observe_and_sample::<_, F>(&mut challenger_a, c0, c2, pow_bits);

        // Sequence B: directly observe (c0, c2) without prior round
        let mut sumcheck_b: SumcheckData<F, EF> = SumcheckData::default();
        let mut challenger_b = create_test_challenger();
        let r_b = sumcheck_b.observe_and_sample::<_, F>(&mut challenger_b, c0, c2, pow_bits);

        // Challenges should differ due to different transcript history
        assert_ne!(r1_a, r_b);
    }

    #[test]
    fn test_observe_and_sample_returns_extension_field_element() {
        // The returned challenge should be a valid extension field element.
        //
        // This is verified implicitly by the type system, but we can also
        // check that it's not trivially zero (with high probability).
        let c0 = EF::from_u64(7);
        let c2 = EF::from_u64(11);
        let pow_bits = 0;

        let mut sumcheck: SumcheckData<F, EF> = SumcheckData::default();
        let mut challenger = create_test_challenger();

        let r: EF = sumcheck.observe_and_sample::<_, F>(&mut challenger, c0, c2, pow_bits);

        // The challenge should (with overwhelming probability) be non-zero
        assert_ne!(r, EF::ZERO);
    }
}
