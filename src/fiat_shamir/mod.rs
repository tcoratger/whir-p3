use rand::{CryptoRng, RngCore, TryRngCore};

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

/// Compatibility wrapper around [`rand::rngs::OsRng`] to adapt it to the
/// [`RngCore`] and [`CryptoRng`] traits from older versions of `rand_core`.
///
/// In `rand_core` 0.9, [`OsRng`] only implements the fallible [`TryRngCore`] and
/// no longer implements [`RngCore`] directly. This breaks compatibility with
/// code expecting an infallible RNG, such as previous implementations using
/// [`RngCore`] trait bounds.
///
/// This wrapper forwards `try_*` methods and unconditionally unwraps the result,
/// panicking if an RNG failure occurs. This mirrors the behavior of older
/// versions where failures were considered unrecoverable and rare.
///
/// Use this type if you want to retain a familiar [`RngCore`] interface
/// without migrating to `TryRngCore`.
#[derive(Debug, Default)]
pub struct CompatOsRng(rand::rngs::OsRng);

impl RngCore for CompatOsRng {
    fn next_u32(&mut self) -> u32 {
        self.0.try_next_u32().unwrap()
    }

    fn next_u64(&mut self) -> u64 {
        self.0.try_next_u64().unwrap()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.0.try_fill_bytes(dest).unwrap();
    }
}

impl CryptoRng for CompatOsRng {}

/// Default random number generator used ([`rand::rngs::OsRng`]).
pub type DefaultRng = CompatOsRng;

/// Default hash function used ([`keccak::Keccak`]).
pub type DefaultHash = keccak::Keccak;
