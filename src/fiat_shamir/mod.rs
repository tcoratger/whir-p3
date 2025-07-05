pub mod domain_separator;
pub mod errors;
pub mod pattern;
pub mod prover;
#[cfg(test)]
mod tests;
pub mod verifier;

pub trait ChallengSampler<EF> {
    fn sample(&mut self) -> EF;
    fn sample_bits(&mut self, bits: usize) -> usize;
}
