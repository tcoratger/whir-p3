pub mod codecs;
pub mod domain_separator;
pub mod duplex_sponge;
pub mod errors;
pub mod keccak;
pub mod pow;
pub mod prover;
pub mod sho;
pub mod traits;
pub mod verifier;

/// Default random number generator used ([`rand::rngs::OsRng`]).
pub type DefaultRng = rand::rngs::OsRng;

/// Default hash function used ([`keccak::Keccak`]).
pub type DefaultHash = keccak::Keccak;
