use p3_field::Field;

use crate::{fiat_shamir::codecs::traits::FieldDomainSeparator, fs_utils::WhirPoWDomainSeparator};

/// Adds simulated sumcheck rounds to a Fiat-Shamir transcript.
pub trait SumcheckSingleDomainSeparator<F: Field> {
    /// Performs `folding_factor` rounds of sumcheck interaction with the transcript.
    ///
    /// In each round:
    /// - Samples 3 scalars for the sumcheck polynomial.
    /// - Samples 1 scalar for folding randomness.
    /// - Optionally performs a PoW challenge if `pow_bits > 0`.
    fn add_sumcheck(self, folding_factor: usize, pow_bits: f64) -> Self;
}

impl<F, DomainSeparator> SumcheckSingleDomainSeparator<F> for DomainSeparator
where
    F: Field,
    DomainSeparator: FieldDomainSeparator<F> + WhirPoWDomainSeparator,
{
    fn add_sumcheck(mut self, folding_factor: usize, pow_bits: f64) -> Self {
        for _ in 0..folding_factor {
            self = self
                .add_scalars(3, "sumcheck_poly")
                .challenge_scalars(1, "folding_randomness")
                .pow(pow_bits);
        }
        self
    }
}
