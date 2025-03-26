//! **Warning**: this function is not SHA3.
//! Despite internally we use the same permutation function,
//! we build a duplex sponge in overwrite mode
//! on the top of it using the `DuplexSponge` trait.
use zeroize::{Zeroize, ZeroizeOnDrop};

use super::duplex_sponge::{DuplexSponge, Permutation};

/// A duplex sponge based on the permutation [`keccak::f1600`]
/// using [`DuplexSponge`].
pub type Keccak = DuplexSponge<AlignedKeccakF1600>;

fn transmute_state(st: &mut AlignedKeccakF1600) -> &mut [u64; 25] {
    unsafe { &mut *std::ptr::from_mut::<AlignedKeccakF1600>(st).cast::<[u64; 25]>() }
}

/// Width of the Keccak-f1600 sponge (in bytes)
const KECCAK_WIDTH_BYTES: usize = 200;
/// Rate of the sponge (bytes): 136
const KECCAK_RATE_BYTES: usize = 136;
/// Capacity of the sponge: 64
const KECCAK_CAPACITY_BYTES: usize = KECCAK_WIDTH_BYTES - KECCAK_RATE_BYTES;

/// This is a wrapper around 200-byte buffer that's always 8-byte aligned
/// to make pointers to it safely convertible to pointers to [u64; 25]
/// (since u64 words must be 8-byte aligned)
#[derive(Debug, Clone, Zeroize, ZeroizeOnDrop)]
#[repr(align(8))]
pub struct AlignedKeccakF1600([u8; 200]);

impl Permutation for AlignedKeccakF1600 {
    type U = u8;
    const N: usize = KECCAK_WIDTH_BYTES;
    const R: usize = KECCAK_RATE_BYTES;

    fn new(tag: [u8; 32]) -> Self {
        let mut state = Self::default();
        state.0[Self::R..Self::R + 32].copy_from_slice(&tag);
        state
    }

    fn permute(&mut self) {
        keccak::f1600(transmute_state(self));
    }
}

impl Default for AlignedKeccakF1600 {
    fn default() -> Self {
        Self([0u8; Self::N])
    }
}

impl AsRef<[u8]> for AlignedKeccakF1600 {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl AsMut<[u8]> for AlignedKeccakF1600 {
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}
