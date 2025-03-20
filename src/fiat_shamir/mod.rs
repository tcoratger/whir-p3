use p3_field::{Field, PrimeField};
use sha3::{Digest, Keccak256, Sha3_256};

pub mod prover;
pub mod verifier;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FsError;

pub trait FsParticipant {
    fn challenge_bytes(&mut self, len: usize) -> Vec<u8>;
    fn challenge_scalars<F: Field>(&mut self, len: usize) -> Vec<F>;
}

pub(crate) fn hash_sha3(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut output = [0u8; 32];
    output.copy_from_slice(&result);
    output
}

pub fn generate_pseudo_random(seed: &[u8; 32], len: usize) -> Vec<u8> {
    if len == 0 {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(len);
    let mut counter = 0u32;

    while result.len() < len {
        let mut hasher = Keccak256::default();
        hasher.update(seed);
        hasher.update(&counter.to_be_bytes());
        let hash_result = hasher.finalize();
        let bytes_to_take = std::cmp::min(hash_result.len(), len - result.len());
        result.extend_from_slice(&hash_result[..bytes_to_take]);
        counter += 1;
    }

    result
}

pub fn serialize_field<F: Field>(f: F) -> Vec<u8> {
    let prime_bytes = F::PrimeSubfield::bits().div_ceil(8);
    assert!(prime_bytes == 4 || prime_bytes == 8);
    assert!(size_of::<F>() % prime_bytes == 0);
    let ext_dim = size_of::<F>() / prime_bytes;

    let subfields = unsafe {
        let ptr = &f as *const F as *const F::PrimeSubfield;
        std::slice::from_raw_parts(ptr, ext_dim)
    };

    let mut bytes = Vec::new();
    for i in 0..ext_dim {
        bytes.extend_from_slice(&prime_field_to_bytes(subfields[i]));
    }
    bytes
}

fn prime_field_to_bytes<F: PrimeField>(f: F) -> Vec<u8> {
    if F::bits() <= 16 {
        unimplemented!()
    } else if F::bits() <= 32 {
        if f.is_zero() {
            return vec![0; 4];
        }
        f.as_canonical_biguint().to_u32_digits()[0].to_le_bytes().to_vec()
    } else if F::bits() <= 64 {
        if f.is_zero() {
            return vec![0; 8];
        }
        f.as_canonical_biguint().to_u64_digits()[0].to_le_bytes().to_vec()
    } else {
        unimplemented!()
    }
}

pub fn field_bytes_in_memory<F: Field>() -> usize {
    let ext_degree: usize = F::bits().div_ceil(F::PrimeSubfield::bits()); // TODO very bad
    (F::PrimeSubfield::bits().div_ceil(8)) * ext_degree
}

pub fn deserialize_field<F: Field>(bytes: &[u8]) -> Option<F> {
    let prime_bytes = F::PrimeSubfield::bits().div_ceil(8);
    assert!(prime_bytes == 4 || prime_bytes == 8);
    assert!(size_of::<F>() % prime_bytes == 0);
    let ext_dim = size_of::<F>() / prime_bytes;
    if bytes.len() != ext_dim * prime_bytes {
        return None;
    }

    let mut subfields = Vec::new();
    for i in 0..ext_dim {
        subfields.push(
            prime_field_from_bytes::<F::PrimeSubfield>(
                &bytes[i * prime_bytes..(i + 1) * prime_bytes],
            )
            .unwrap(),
        );
    }

    unsafe {
        let ptr = subfields.as_ptr() as *const F;
        Some(std::ptr::read(ptr))
    }
}

fn prime_field_from_bytes<F: PrimeField>(bytes: &[u8]) -> Option<F> {
    if F::bits() <= 16 {
        unimplemented!()
    } else if F::bits() <= 32 {
        if bytes.len() != 4 {
            return None;
        }
        let mut arr = [0u8; 4];
        arr.copy_from_slice(bytes);
        Some(F::from_u32(u32::from_le_bytes(arr)))
    } else if F::bits() <= 64 {
        if bytes.len() != 8 {
            return None;
        }
        let mut arr = [0u8; 8];
        arr.copy_from_slice(bytes);
        Some(F::from_u64(u64::from_le_bytes(arr)))
    } else {
        unimplemented!()
    }
}

pub(crate) fn count_ending_zero_bits(buff: &[u8]) -> usize {
    let mut count = 0;
    'outer: for byte in buff {
        for i in 0..8 {
            if byte & (1 << i) == 0 {
                count += 1;
            } else {
                break 'outer;
            }
        }
    }
    count
}
