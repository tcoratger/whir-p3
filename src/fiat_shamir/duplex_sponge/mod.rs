use interface::Unit;
use p3_symmetric::Permutation;
use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::fiat_shamir::duplex_sponge::interface::DuplexSpongeInterface;

pub mod interface;

/// A cryptographic sponge.
#[derive(Debug, Clone)]
pub struct DuplexSponge<U, C, const WIDTH: usize, const RATE: usize>
where
    U: Unit,
    C: Permutation<[U; WIDTH]>,
{
    permutation: C,
    state: [U; WIDTH],
    absorb_pos: usize,
    squeeze_pos: usize,
}

impl<U, C, const WIDTH: usize, const RATE: usize> Zeroize for DuplexSponge<U, C, WIDTH, RATE>
where
    U: Unit,
    C: Permutation<[U; WIDTH]>,
{
    fn zeroize(&mut self) {
        self.state.zeroize();
    }
}

impl<U, C, const WIDTH: usize, const RATE: usize> ZeroizeOnDrop for DuplexSponge<U, C, WIDTH, RATE>
where
    U: Unit,
    C: Permutation<[U; WIDTH]>,
{
}

impl<U, C, const WIDTH: usize, const RATE: usize> DuplexSpongeInterface<C, U, WIDTH>
    for DuplexSponge<U, C, WIDTH, RATE>
where
    U: Unit + Default + Copy,
    C: Permutation<[U; WIDTH]>,
{
    fn new(permutation: C, iv: [u8; 32]) -> Self {
        let mut state = [U::default(); WIDTH];
        for (i, &b) in iv.iter().enumerate() {
            state[RATE + i] = U::from_u8(b);
        }

        Self {
            permutation,
            state,
            absorb_pos: 0,
            squeeze_pos: RATE,
        }
    }

    fn absorb_unchecked(&mut self, mut input: &[U]) -> &mut Self {
        while !input.is_empty() {
            if self.absorb_pos == RATE {
                self.permutation.permute_mut(&mut self.state);
                self.absorb_pos = 0;
            } else {
                assert!(self.absorb_pos < RATE);
                let chunk_len = usize::min(input.len(), RATE - self.absorb_pos);
                let (chunk, rest) = input.split_at(chunk_len);

                self.state[self.absorb_pos..self.absorb_pos + chunk_len].clone_from_slice(chunk);
                self.absorb_pos += chunk_len;
                input = rest;
            }
        }
        self.squeeze_pos = RATE;
        self
    }

    fn squeeze_unchecked(&mut self, output: &mut [U]) -> &mut Self {
        if output.is_empty() {
            return self;
        }

        if self.squeeze_pos == RATE {
            self.squeeze_pos = 0;
            self.absorb_pos = 0;
            self.permutation.permute_mut(&mut self.state);
        }

        assert!(self.squeeze_pos < RATE);
        let chunk_len = usize::min(output.len(), RATE - self.squeeze_pos);
        let (output, rest) = output.split_at_mut(chunk_len);
        output.clone_from_slice(&self.state[self.squeeze_pos..self.squeeze_pos + chunk_len]);
        self.squeeze_pos += chunk_len;
        self.squeeze_unchecked(rest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fiat_shamir::keccak::{KECCAK_RATE_BYTES, KECCAK_WIDTH_BYTES};

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

    type Sponge = DuplexSponge<u8, DummyPermutation, KECCAK_WIDTH_BYTES, KECCAK_RATE_BYTES>;

    #[test]
    fn test_new_sponge_initializes_state() {
        // Use IV filled with 42 to check initialization
        let iv = [42u8; 32];
        let sponge = Sponge::new(DummyPermutation, iv);
        assert_eq!(sponge.absorb_pos, 0);
        assert_eq!(sponge.squeeze_pos, KECCAK_RATE_BYTES);
        // The last N - R elements should store the IV
        for i in 0..iv.len() {
            assert_eq!(
                sponge.state[KECCAK_RATE_BYTES + i],
                42,
                "Expected sponge.state[{}] to be 42",
                KECCAK_RATE_BYTES + i
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
        assert_eq!(sponge.squeeze_pos, KECCAK_RATE_BYTES);
    }

    #[test]
    fn test_absorb_across_permutation_boundary() {
        let mut sponge = Sponge::new(DummyPermutation, [0u8; 32]);

        // Absorb enough to cross the rate boundary (R + 1 bytes), triggering one permutation
        let big_input = vec![1u8; KECCAK_RATE_BYTES + 1];
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
        sponge.squeeze_pos = KECCAK_RATE_BYTES;

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
