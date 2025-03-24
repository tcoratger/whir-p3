use p3_challenger::{CanSample, GrindingChallenger};
use p3_field::{Field, PrimeField32};

use crate::merkle_tree::WhirChallenger;

/// Adds simulated sumcheck rounds to a Fiat-Shamir transcript.
pub trait SumcheckSingleChallenger<F: Field> {
    /// Performs `folding_factor` rounds of sumcheck interaction with the transcript.
    ///
    /// In each round:
    /// - Samples 3 scalars for the sumcheck polynomial.
    /// - Samples 1 scalar for folding randomness.
    /// - Optionally performs a PoW challenge if `pow_bits > 0`.
    fn add_sumcheck(&mut self, folding_factor: usize, pow_bits: usize);
}

impl<F> SumcheckSingleChallenger<F> for WhirChallenger<F>
where
    F: PrimeField32,
{
    fn add_sumcheck(&mut self, folding_factor: usize, pow_bits: usize) {
        for _ in 0..folding_factor {
            // Sample 3 polynomial coefficients
            let _coeffs: [F; 3] = [self.sample(), self.sample(), self.sample()];

            // Sample 1 folding randomness
            let _folding_rand: F = self.sample();

            // Apply proof-of-work if required
            if pow_bits > 0 {
                self.grind(pow_bits);
            }
        }
    }
}
