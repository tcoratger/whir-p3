use p3_symmetric::Permutation;
use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::fiat_shamir::duplex_sponge::interface::DuplexSpongeInterface;

pub mod interface;

/// Width of the Keccak-f1600 sponge (in bytes)
const KECCAK_WIDTH_BYTES: usize = 200;
/// Rate of the sponge (bytes): 136
const KECCAK_RATE_BYTES: usize = 136;

/// Basic units over which a sponge operates.
///
/// We require the units to have a precise size in memory, to be cloneable,
/// and that we can zeroize them.
pub trait Unit: Clone + Sized {
    /// Write a bunch of units in the wire.
    fn write(bunch: &[Self], w: &mut impl std::io::Write) -> Result<(), std::io::Error>;
    /// Read a bunch of units from the wire
    fn read(r: &mut impl std::io::Read, bunch: &mut [Self]) -> Result<(), std::io::Error>;
}

/// A cryptographic sponge.
#[derive(Debug, Clone)]
pub struct DuplexSponge<C: Permutation<[u8; KECCAK_WIDTH_BYTES]>> {
    permutation: C,
    state: [u8; KECCAK_WIDTH_BYTES],
    absorb_pos: usize,
    squeeze_pos: usize,
}

impl<C: Permutation<[u8; KECCAK_WIDTH_BYTES]> + Clone> DuplexSponge<C> {
    pub const N: usize = KECCAK_WIDTH_BYTES;
    pub const R: usize = KECCAK_RATE_BYTES;
}

impl<C: Permutation<[u8; KECCAK_WIDTH_BYTES]> + Clone> Zeroize for DuplexSponge<C> {
    fn zeroize(&mut self) {
        self.state.zeroize();
    }
}

impl<C: Permutation<[u8; KECCAK_WIDTH_BYTES]> + Clone> ZeroizeOnDrop for DuplexSponge<C> {}

impl<C: Permutation<[u8; KECCAK_WIDTH_BYTES]> + Clone> DuplexSpongeInterface<C>
    for DuplexSponge<C>
{
    fn new(permutation: C, iv: [u8; 32]) -> Self {
        let mut state = [0u8; KECCAK_WIDTH_BYTES];
        state[Self::R..Self::R + iv.len()].copy_from_slice(&iv);

        Self {
            permutation,
            state,
            absorb_pos: 0,
            squeeze_pos: Self::R,
        }
    }

    fn absorb_unchecked(&mut self, mut input: &[u8]) -> &mut Self {
        while !input.is_empty() {
            if self.absorb_pos == Self::R {
                self.permutation.permute_mut(&mut self.state);
                self.absorb_pos = 0;
            } else {
                assert!(self.absorb_pos < Self::R);
                let chunk_len = usize::min(input.len(), Self::R - self.absorb_pos);
                let (chunk, rest) = input.split_at(chunk_len);

                self.state[self.absorb_pos..self.absorb_pos + chunk_len].clone_from_slice(chunk);
                self.absorb_pos += chunk_len;
                input = rest;
            }
        }
        self.squeeze_pos = Self::R;
        self
    }

    fn squeeze_unchecked(&mut self, output: &mut [u8]) -> &mut Self {
        let mut remaining = output;

        while !remaining.is_empty() {
            if self.squeeze_pos == Self::R {
                self.permutation.permute_mut(&mut self.state);
                self.squeeze_pos = 0;
                self.absorb_pos = 0;
            }
            let chunk_len = usize::min(remaining.len(), Self::R - self.squeeze_pos);
            let (out_chunk, rest) = remaining.split_at_mut(chunk_len);

            out_chunk.clone_from_slice(&self.state[self.squeeze_pos..self.squeeze_pos + chunk_len]);
            self.squeeze_pos += chunk_len;
            remaining = rest;
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A dummy permutation
    #[derive(Clone, Debug)]
    pub(super) struct DummyPermutation;

    impl Zeroize for DummyPermutation {
        fn zeroize(&mut self) {}
    }

    impl Permutation<[u8; KECCAK_WIDTH_BYTES]> for DummyPermutation {
        fn permute_mut(&self, state: &mut [u8; KECCAK_WIDTH_BYTES]) {
            for b in state.iter_mut() {
                *b = b.wrapping_add(1);
            }
        }
    }

    type Sponge = DuplexSponge<DummyPermutation>;

    #[test]
    fn test_new_sponge_initializes_state() {
        // Use IV filled with 42 to check initialization
        let iv = [42u8; 32];
        let sponge = Sponge::new(DummyPermutation, iv);
        assert_eq!(sponge.absorb_pos, 0);
        assert_eq!(sponge.squeeze_pos, Sponge::R);
        // The last N - R elements should store the IV
        for i in 0..iv.len() {
            assert_eq!(
                sponge.state[Sponge::R + i],
                42,
                "Expected sponge.state[{}] to be 42",
                Sponge::R + i
            );
        }
    }

    #[test]
    fn test_absorb_less_than_rate() {
        let mut sponge = Sponge::new(DummyPermutation, [0u8; 32]);
        // Absorb just 1 byte (less than R)
        sponge.absorb_unchecked(&[5u8]);
        assert_eq!(sponge.state[0], 5);
        // One position consumed
        assert_eq!(sponge.absorb_pos, 1);
        // Reset after absorb
        assert_eq!(sponge.squeeze_pos, Sponge::R);
    }

    #[test]
    fn test_absorb_across_permutation_boundary() {
        let mut sponge = Sponge::new(DummyPermutation, [0u8; 32]);

        // Absorb enough to cross the rate boundary (R + 1 bytes), triggering one permutation
        let big_input = vec![1u8; Sponge::R + 1];
        sponge.absorb_unchecked(&big_input);

        // Last absorbed byte should be at position 0 after permutation
        assert_eq!(sponge.state[0], 1);
        // One position consumed after reset
        assert_eq!(sponge.absorb_pos, 1);
    }

    #[test]
    fn test_squeeze_output_and_position() {
        let mut sponge = Sponge::new(DummyPermutation, [0u8; 32]);
        // Preload the state with known values
        sponge.state[0] = 10;
        sponge.state[1] = 20;

        let mut out = [0u8; 2];
        sponge.squeeze_unchecked(&mut out);

        // Output is the current state values (no permutation yet)
        assert_eq!(out, [11, 21]);

        // Position advanced by 2
        assert_eq!(sponge.squeeze_pos, 2);
    }

    #[test]
    fn test_squeeze_triggers_permute_when_full() {
        let mut sponge = Sponge::new(DummyPermutation, [0u8; 32]);

        // Force squeeze position to full → triggers permutation on next squeeze
        sponge.squeeze_pos = Sponge::R;

        // Snapshot state before permutation
        let old_state = sponge.state;

        let mut out = [0u8; 1];
        sponge.squeeze_unchecked(&mut out);

        // Check that permutation ran: all bytes incremented by 1
        for (before, after) in old_state.iter().zip(&sponge.state) {
            assert_eq!(*after, before.wrapping_add(1));
        }

        // Squeeze pulled one byte from the refreshed state
        assert_eq!(out[0], sponge.state[0]);

        // Squeeze position advanced by 1
        assert_eq!(sponge.squeeze_pos, 1);
    }
}
