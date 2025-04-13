use num_bigint::BigUint;
use p3_field::PrimeField64;

/// Takes a big-endian byte slice and reduces it modulo the field order.
pub(crate) fn from_be_bytes_mod_order<F: PrimeField64>(bytes: &[u8]) -> F {
    // Step 1: Interpret input as a BigUint
    let x = BigUint::from_bytes_be(bytes);

    // Step 2: Compute x mod field modulus
    let modulus = BigUint::from(F::ORDER_U64);
    let reduced = x % modulus;

    // Step 3: Convert to field element
    F::from_u64(reduced.try_into().unwrap())
}

/// Takes a little-endian byte slice and reduces it modulo the field order.
pub(crate) fn from_le_bytes_mod_order<F: PrimeField64>(bytes: &[u8]) -> F {
    // Step 1: Interpret input as a BigUint
    let x = BigUint::from_bytes_le(bytes);

    // Step 2: Compute x mod field modulus
    let modulus = BigUint::from(F::ORDER_U64);
    let reduced = x % modulus;

    // Step 3: Convert to field element
    F::from_u64(reduced.try_into().unwrap())
}

/// Bytes needed in order to encode an element of F.
pub(crate) const fn bytes_modp(modulus_bits: u32) -> usize {
    (modulus_bits as usize).div_ceil(8)
}

/// Bytes needed in order to obtain a uniformly distributed random element of `modulus_bits`
pub(crate) const fn bytes_uniform_modp(modulus_bits: u32) -> usize {
    (modulus_bits as usize + 128) / 8
}
