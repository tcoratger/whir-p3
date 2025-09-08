pub mod domain_separator;
pub mod errors;
pub mod pattern;
pub mod prover;
#[cfg(test)]
mod tests;
pub mod verifier;

/// A trait for types that can sample challenges in a Fiat-Shamir-based protocol.
///
/// This trait abstracts over objects (such as prover or verifier states) that can
/// deterministically generate random challenges from a transcript using a cryptographic
/// challenger. The challenges are used to drive non-interactive proofs or interactive
/// proof reductions.
pub trait ChallengeSampler<F> {
    /// Sample a new random element from the extension field `F`.
    ///
    /// # Returns
    /// A freshly sampled challenge element in `F`, derived from the current transcript state.
    fn sample(&mut self) -> F;

    /// Sample a uniformly random integer consisting of a given number of bits.
    ///
    /// # Arguments
    /// - `bits`: Number of bits in the output integer.
    ///
    /// # Returns
    /// A `usize` value uniformly sampled from the range `0..2^bits`, derived from the transcript state.
    fn sample_bits(&mut self, bits: usize) -> usize;
}
