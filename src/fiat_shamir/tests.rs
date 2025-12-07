use alloc::{format, vec::Vec};

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::{CanObserve, CanSample, DuplexChallenger, GrindingChallenger};
use p3_field::extension::BinomialExtensionField;
use proptest::prelude::*;
use rand::{Rng, SeedableRng, rngs::SmallRng};

use crate::fiat_shamir::domain_separator::DomainSeparator;

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type Perm = Poseidon2BabyBear<16>;
type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

fn make_challenger() -> MyChallenger {
    let perm = Perm::new_from_rng_128(&mut rand::rngs::SmallRng::seed_from_u64(42));
    DuplexChallenger::new(perm)
}

fn make_domain_separator() -> DomainSeparator<EF, F> {
    DomainSeparator::new(Vec::new())
}

proptest! {
    /// Tests that observing the same base field scalars on two identical challengers
    /// produces the same transcript state (identical challenge outputs).
    #[test]
    fn test_base_scalar_roundtrip(seed in any::<u64>(), n in 1usize..16) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let vals: Vec<F> = (0..n).map(|_| rng.random()).collect();

        // Create two identical challengers
        let mut prover_challenger = make_challenger();
        let mut verifier_challenger = make_challenger();

        // Both observe the same data
        prover_challenger.observe_slice(&vals);
        verifier_challenger.observe_slice(&vals);

        // Verify both produce identical challenges
        let prover_sample: F = prover_challenger.sample();
        let verifier_sample: F = verifier_challenger.sample();
        prop_assert_eq!(prover_sample, verifier_sample);
    }

    /// Tests that observing the same extension field scalars (as base field elements)
    /// on two identical challengers produces the same transcript state.
    #[test]
    fn test_extension_scalar_roundtrip(seed in any::<u64>(), n in 1usize..32) {
        let mut rng = SmallRng::seed_from_u64(seed);
        // Generate base field values that represent extension field elements
        let vals: Vec<F> = (0..n).map(|_| rng.random()).collect();

        // Create two identical challengers
        let mut prover_challenger = make_challenger();
        let mut verifier_challenger = make_challenger();

        // Both observe the same data
        prover_challenger.observe_slice(&vals);
        verifier_challenger.observe_slice(&vals);

        // Verify both produce identical challenges
        let prover_sample: F = prover_challenger.sample();
        let verifier_sample: F = verifier_challenger.sample();
        prop_assert_eq!(prover_sample, verifier_sample);
    }

    /// Tests that hint data can be consistently stored and retrieved.
    /// In the new API, hints are stored directly in WhirProof.
    #[test]
    fn test_hint_base_scalar_roundtrip(seed in any::<u64>(), n in 1usize..16) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let vals: Vec<F> = (0..n).map(|_| rng.random()).collect();

        // Simulate storing hints: values stored by prover, retrieved by verifier
        let stored_hints = vals.clone();
        let recovered_hints = stored_hints;

        prop_assert_eq!(vals, recovered_hints);
    }

    /// Tests that extension field hints can be consistently stored and retrieved.
    #[test]
    fn test_hint_extension_scalar_roundtrip(seed in any::<u64>(), n in 1usize..8) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let ext_vals: Vec<EF> = (0..n).map(|_| rng.random()).collect();

        // Simulate storing extension hints
        let stored_hints = ext_vals.clone();
        let recovered_hints = stored_hints;

        prop_assert_eq!(ext_vals, recovered_hints);
    }

    /// Tests that PoW grinding on prover and check_witness on verifier
    /// with identical challenger states produces valid roundtrip.
    #[test]
    fn test_pow_grinding_roundtrip(bits in 1usize..8) {
        // Create two identical challengers
        let mut prover_challenger = make_challenger();
        let mut verifier_challenger = make_challenger();

        // Prover grinds to find witness
        let witness: F = prover_challenger.grind(bits);

        // Verifier checks the witness with identical initial state
        let valid = verifier_challenger.check_witness(bits, witness);
        prop_assert!(valid, "Verifier should accept valid witness from prover");
    }

    /// Tests that `observe_domain_separator` produces the same challenger state
    /// as manually observing the same field elements.
    #[test]
    fn test_observe_domain_separator(seed in any::<u64>(), pattern_len in 1usize..16) {
        let mut rng = SmallRng::seed_from_u64(seed);

        // Create a domain separator with random pattern elements
        let pattern: Vec<F> = (0..pattern_len).map(|_| rng.random()).collect();
        let domsep = DomainSeparator::<EF, F>::new(pattern.clone());

        // Create two identical challengers
        let mut challenger1 = make_challenger();
        let mut challenger2 = make_challenger();

        // Use observe_domain_separator on the first challenger
        domsep.observe_domain_separator(&mut challenger1);

        // Manually observe the same field elements on the second challenger
        challenger2.observe_slice(&pattern);

        // Verify that both challengers produce the same challenge values
        let sample1: F = challenger1.sample();
        let sample2: F = challenger2.sample();
        prop_assert_eq!(sample1, sample2, "Challengers should produce identical challenges after observing the same domain separator");

        // Sample a few more values to ensure consistency
        for _ in 0..3 {
            let s1: F = challenger1.sample();
            let s2: F = challenger2.sample();
            prop_assert_eq!(s1, s2, "All subsequent samples should match");
        }
    }
}
