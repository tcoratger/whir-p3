pub mod deserialize;
pub mod domain_separator;
pub mod prover_messages;
pub mod traits;
pub mod verifier_messages;

/// Bytes needed in order to encode an element of F.
#[allow(unused)]
pub(super) const fn bytes_modp(modulus_bits: u32) -> usize {
    (modulus_bits as usize).div_ceil(8)
}

/// Bytes needed in order to obtain a uniformly distributed random element of `modulus_bits`
#[allow(unused)]
pub(super) const fn bytes_uniform_modp(modulus_bits: u32) -> usize {
    (modulus_bits as usize + 128) / 8
}
