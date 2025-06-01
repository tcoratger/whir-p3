use duplex_sponge::interface::Unit;
use errors::DomainSeparatorMismatch;
use p3_keccak::KeccakF;

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

/// Default permutation used ([`KeccakF`]).
pub type DefaultPerm = KeccakF;

/// Squeezing bytes from the sponge.
///
/// While this trait is trivial for byte-oriented sponges, it is non-trivial for algebraic hashes.
/// In particular, the implementation of this trait is expected to provide different guarantees
/// between units `u8` and $\mathbb{F}_p$ elements:
/// - `u8` implementations are assumed to be streaming-friendly, that is:
///   `implementor.fill_challenge_units(&mut out[..1]); implementor.fill_challenge_units(&mut
///   out[1..]);` is expected to be equivalent to `implementor.fill_challenge_units(&mut out);`.
/// - $\mathbb{F}_p$ implementations are expected to provide no such guarantee. In addition, we expect the implementation to return bytes that are uniformly distributed. In particular, note that the most significant bytes of a $\mod p$ element are not uniformly distributed. The number of bytes good to be used can be discovered playing with [our scripts](https://github.com/arkworks-rs/spongefish/blob/main/scripts/useful_bits_modp.py).
pub trait UnitToBytes<U>
where
    U: Unit,
{
    /// Fill `input` with units sampled uniformly at random.
    fn fill_challenge_units(&mut self, input: &mut [U]) -> Result<(), DomainSeparatorMismatch>;
}
