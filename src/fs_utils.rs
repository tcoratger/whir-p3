use p3_challenger::{
    CanObserve, CanSample, GrindingChallenger, HashChallenger, SerializingChallenger32,
};
use p3_field::PrimeField32;
use p3_symmetric::CryptographicHasher;

/// Trait for adding out-of-domain (OOD) queries and responses to a challenger.
///
/// OOD queries are used in polynomial commitment schemes and proof systems,
/// allowing the challenger to sample challenge points outside the domain.
/// The prover responds with corresponding evaluations, which are later verified.
///
/// This trait modifies the challenger state by:
/// - Sampling `num_samples` OOD challenge points.
/// - Sampling `num_samples` OOD responses.
/// - Observing these values in the challenger for deterministic randomness.
///
/// If `num_samples == 0`, the challenger remains unchanged.
pub trait OODIOPattern<F: PrimeField32> {
    /// Adds `num_samples` OOD queries and their corresponding responses.
    ///
    /// - If `num_samples > 0`, it:
    ///   - Samples `num_samples` challenge points.
    ///   - Samples `num_samples` responses.
    ///   - Observes both the queries and responses for future deterministic challenges.
    fn add_ood(&mut self, num_samples: usize);
}

/// Implementation using `SerializingChallenger32`
impl<F, H> OODIOPattern<F> for SerializingChallenger32<F, HashChallenger<u8, H, 32>>
where
    F: PrimeField32,
    H: Clone + CryptographicHasher<u8, [u8; 32]>,
{
    fn add_ood(&mut self, num_samples: usize) {
        if num_samples > 0 {
            for _ in 0..num_samples {
                // Sample an OOD query
                let query: F = self.sample();
                // Sample corresponding answer
                let answer: F = self.sample();

                // Observe the query
                self.observe(query);
                // Observe the answer
                self.observe(answer);
            }
        }
    }
}

/// Trait for incorporating Proof-of-Work (PoW) challenges into a challenger.
///
/// This trait allows the challenger to generate a PoW challenge by computing a
/// proof that satisfies a difficulty condition (grinding). The generated proof
/// is then observed by the challenger, ensuring it is incorporated into future
/// challenge randomness.
///
/// This is useful in interactive proofs or protocols requiring computational
/// effort to deter spam or enforce security guarantees.
pub trait WhirPoWIOPattern {
    /// Generates a Proof-of-Work (PoW) challenge with `bits` difficulty.
    ///
    /// - If `bits > 0`, it:
    ///   - Computes a PoW witness by iterating over possible values until a valid one is found.
    ///   - Observes the witness, ensuring its influence on future randomness.
    /// - If `bits == 0`, no PoW is performed, and the challenger remains unchanged.
    fn pow(&mut self, bits: usize);
}

impl<C> WhirPoWIOPattern for C
where
    C: GrindingChallenger,
{
    fn pow(&mut self, bits: usize) {
        if bits > 0 {
            let proof = self.grind(bits);
            self.observe(proof);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_baby_bear::BabyBear;
    use p3_challenger::{CanObserve, CanSample, HashChallenger, SerializingChallenger32};
    use p3_field::PrimeCharacteristicRing;
    use p3_keccak::Keccak256Hash;

    #[test]
    fn test_add_ood() {
        let hasher = Keccak256Hash {}; // Replace with the hash function you're using
        let mut challenger = SerializingChallenger32::<
            BabyBear,
            HashChallenger<u8, Keccak256Hash, 32>,
        >::from_hasher(vec![], hasher);

        // Add 3 OOD queries
        challenger.add_ood(3);

        // Ensure that exactly 3 queries and responses were sampled
        let query: BabyBear = challenger.sample();
        let answer: BabyBear = challenger.sample();
        challenger.observe(query);
        challenger.observe(answer);

        // Ensure that query and answer are nonzero
        assert_ne!(query, BabyBear::ZERO);
        assert_ne!(answer, BabyBear::ZERO);
    }

    #[test]
    fn test_add_ood_no_samples() {
        let hasher = Keccak256Hash {};
        let mut challenger = SerializingChallenger32::<
            BabyBear,
            HashChallenger<u8, Keccak256Hash, 32>,
        >::from_hasher(vec![], hasher);

        // Add 0 OOD queries (should not change anything)
        challenger.add_ood(0);

        // Since no queries were added, sampling should not produce new values
        let query: BabyBear = challenger.sample();
        let answer: BabyBear = challenger.sample();

        assert_ne!(query, BabyBear::ZERO);
        assert_ne!(answer, BabyBear::ZERO);
    }

    #[test]
    fn test_pow_challenge() {
        let hasher = Keccak256Hash {};
        let mut challenger = SerializingChallenger32::<
            BabyBear,
            HashChallenger<u8, Keccak256Hash, 32>,
        >::from_hasher(vec![], hasher);

        // Perform PoW with 5-bit difficulty
        challenger.pow(5);

        // Ensure that a witness was generated and observed
        let proof: BabyBear = challenger.sample();
        challenger.observe(proof);

        assert_ne!(proof, BabyBear::ZERO); // The proof must be a nonzero value
    }

    #[test]
    fn test_pow_no_work() {
        let hasher = Keccak256Hash {};
        let mut challenger = SerializingChallenger32::<
            BabyBear,
            HashChallenger<u8, Keccak256Hash, 32>,
        >::from_hasher(vec![], hasher);

        // Perform PoW with 0-bit difficulty (should do nothing)
        challenger.pow(0);

        // No new values should have been observed
        let proof: BabyBear = challenger.sample();

        // The proof should not be modified
        assert_ne!(proof, BabyBear::ZERO);
    }
}
