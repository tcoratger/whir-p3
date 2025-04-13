use crate::fiat_shamir::{prover::ProverState, verifier::VerifierState};

pub mod domain_separator;
pub mod duplex_sponge;
pub mod errors;
pub mod keccak;
pub mod pow;
pub mod prover;
pub mod sho;
pub mod utils;
pub mod verifier;

/// Default hash function used ([`keccak::Keccak`]).
pub type DefaultHash = keccak::Keccak;

#[derive(Debug)]
pub enum FiatShamir<'a> {
    Prover(ProverState),
    Verifier(VerifierState<'a>),
}
