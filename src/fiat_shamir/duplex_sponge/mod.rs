use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::fiat_shamir::duplex_sponge::interface::DuplexSpongeInterface;

pub mod interface;

/// Basic units over which a sponge operates.
///
/// We require the units to have a precise size in memory, to be cloneable,
/// and that we can zeroize them.
pub trait Unit: Clone + Sized + zeroize::Zeroize {
    /// Write a bunch of units in the wire.
    fn write(bunch: &[Self], w: &mut impl std::io::Write) -> Result<(), std::io::Error>;
    /// Read a bunch of units from the wire
    fn read(r: &mut impl std::io::Read, bunch: &mut [Self]) -> Result<(), std::io::Error>;
}

/// The basic state of a cryptographic sponge.
///
/// A cryptographic sponge operates over some domain [`Permutation::U`] units.
/// It has a width [`Permutation::N`] and can process elements at rate [`Permutation::R`],
/// using the permutation function [`Permutation::permute`].
///
/// For implementors:
///
/// - State is written in *the first* [`Permutation::R`] (rate) bytes of the state. The last
///   [`Permutation::N`]-[`Permutation::R`] bytes are never touched directly except during
///   initialization.
/// - The duplex sponge is in *overwrite mode*. This mode is not known to affect the security levels
///   and removes assumptions on [`Permutation::U`] as well as constraints in the final
///   zero-knowledge proof implementing the hash function.
/// - The [`std::default::Default`] implementation *MUST* initialize the state to zero.
/// - The [`Permutation::new`] method should initialize the sponge writing the entropy provided in
///   the `iv` in the last [`Permutation::N`]-[`Permutation::R`] elements of the state.
pub trait Permutation: Zeroize + Default + Clone + AsRef<[Self::U]> + AsMut<[Self::U]> {
    /// The basic unit over which the sponge operates.
    type U: Unit;

    /// The width of the sponge, equal to rate [`Permutation::R`] plus capacity.
    /// Cannot be less than 1. Cannot be less than [`Permutation::R`].
    const N: usize;

    /// The rate of the sponge.
    const R: usize;

    /// Initialize the state of the sponge using 32 bytes of seed.
    fn new(iv: [u8; 32]) -> Self;

    /// Permute the state of the sponge.
    fn permute(&mut self);
}

/// A cryptographic sponge.
#[derive(Debug, Clone, Default, Zeroize, ZeroizeOnDrop)]
pub struct DuplexSponge<C: Permutation> {
    permutation: C,
    absorb_pos: usize,
    squeeze_pos: usize,
}

impl<U: Unit, C: Permutation<U = U>> DuplexSpongeInterface<U> for DuplexSponge<C> {
    fn new(iv: [u8; 32]) -> Self {
        assert!(C::N > C::R, "Capacity of the sponge should be > 0.");
        Self {
            permutation: C::new(iv),
            absorb_pos: 0,
            squeeze_pos: C::R,
        }
    }

    fn absorb_unchecked(&mut self, mut input: &[U]) -> &mut Self {
        while !input.is_empty() {
            if self.absorb_pos == C::R {
                self.permutation.permute();
                self.absorb_pos = 0;
            } else {
                assert!(self.absorb_pos < C::R);
                let chunk_len = usize::min(input.len(), C::R - self.absorb_pos);
                let (chunk, rest) = input.split_at(chunk_len);

                self.permutation.as_mut()[self.absorb_pos..self.absorb_pos + chunk_len]
                    .clone_from_slice(chunk);
                self.absorb_pos += chunk_len;
                input = rest;
            }
        }
        self.squeeze_pos = C::R;
        self
    }

    fn squeeze_unchecked(&mut self, output: &mut [U]) -> &mut Self {
        if output.is_empty() {
            return self;
        }

        if self.squeeze_pos == C::R {
            self.squeeze_pos = 0;
            self.absorb_pos = 0;
            self.permutation.permute();
        }

        assert!(self.squeeze_pos < C::R);
        let chunk_len = usize::min(output.len(), C::R - self.squeeze_pos);
        let (output, rest) = output.split_at_mut(chunk_len);
        output.clone_from_slice(
            &self.permutation.as_ref()[self.squeeze_pos..self.squeeze_pos + chunk_len],
        );
        self.squeeze_pos += chunk_len;
        self.squeeze_unchecked(rest)
    }

    fn ratchet_unchecked(&mut self) -> &mut Self {
        self.permutation.permute();
        // set to zero the state up to rate
        // XXX. is the compiler really going to do this?
        self.permutation.as_mut()[..C::R]
            .iter_mut()
            .for_each(Zeroize::zeroize);
        self.squeeze_pos = C::R;
        self
    }
}

#[cfg(test)]
mod tests {
    use std::{
        cell::RefCell,
        io::{Read, Write},
    };

    use zeroize::Zeroize;

    use super::{DuplexSponge, Permutation, Unit};
    use crate::fiat_shamir::duplex_sponge::interface::DuplexSpongeInterface;

    // Dummy unit that wraps a single byte
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Zeroize)]
    pub(super) struct DummyUnit(pub u8);

    impl Unit for DummyUnit {
        fn write(bunch: &[Self], w: &mut impl Write) -> Result<(), std::io::Error> {
            w.write_all(&bunch.iter().map(|u| u.0).collect::<Vec<_>>())
        }

        fn read(r: &mut impl Read, bunch: &mut [Self]) -> Result<(), std::io::Error> {
            // Read raw bytes, then convert to DummyUnit
            let mut buf = vec![0u8; bunch.len()];
            r.read_exact(&mut buf)?;
            for (i, b) in buf.into_iter().enumerate() {
                bunch[i] = Self(b);
            }
            Ok(())
        }
    }

    // A dummy permutation that tracks how often it's permuted
    #[derive(Clone, Debug)]
    pub(super) struct DummyPermutation {
        pub state: [DummyUnit; Self::N],
        pub permuted: RefCell<usize>, // not part of cryptographic state
    }

    impl Zeroize for DummyPermutation {
        fn zeroize(&mut self) {
            self.state.zeroize();
        }
    }

    impl Default for DummyPermutation {
        fn default() -> Self {
            Self {
                state: [DummyUnit(0); Self::N],
                permuted: RefCell::new(0),
            }
        }
    }

    impl Permutation for DummyPermutation {
        type U = DummyUnit;
        const N: usize = 4;
        const R: usize = 2;

        fn new(iv: [u8; 32]) -> Self {
            let mut state = [DummyUnit(0); Self::N];
            for (i, byte) in iv.iter().take(Self::N - Self::R).enumerate() {
                state[Self::R + i] = DummyUnit(*byte);
            }
            Self {
                state,
                permuted: RefCell::new(0),
            }
        }

        fn permute(&mut self) {
            // Increment every byte in the state and count permutations
            *self.permuted.borrow_mut() += 1;
            for i in 0..Self::N {
                self.state[i].0 = self.state[i].0.wrapping_add(1);
            }
        }
    }

    impl AsRef<[DummyUnit]> for DummyPermutation {
        fn as_ref(&self) -> &[DummyUnit] {
            &self.state
        }
    }

    impl AsMut<[DummyUnit]> for DummyPermutation {
        fn as_mut(&mut self) -> &mut [DummyUnit] {
            &mut self.state
        }
    }

    type Sponge = DuplexSponge<DummyPermutation>;

    #[test]
    fn test_new_sponge_initializes_state() {
        // Use IV filled with 42 to check initialization
        let iv = [42u8; 32];
        let sponge = Sponge::new(iv);
        assert_eq!(sponge.absorb_pos, 0);
        assert_eq!(sponge.squeeze_pos, DummyPermutation::R);
        // The last N - R elements should store the IV
        for i in 0..(DummyPermutation::N - DummyPermutation::R) {
            assert_eq!(
                sponge.permutation.state[DummyPermutation::R + i],
                DummyUnit(42)
            );
        }
    }

    #[test]
    fn test_absorb_less_than_rate() {
        let mut sponge = Sponge::new([0u8; 32]);
        // Absorb just 1 element (less than R = 2)
        sponge.absorb_unchecked(&[DummyUnit(5)]);
        assert_eq!(sponge.permutation.state[0], DummyUnit(5));
        // One position consumed
        assert_eq!(sponge.absorb_pos, 1);
        // Reset after absorb
        assert_eq!(sponge.squeeze_pos, DummyPermutation::R);
    }

    #[test]
    fn test_absorb_across_permutation_boundary() {
        let mut sponge = Sponge::new([0u8; 32]);

        // Permutation not called yet
        assert_eq!(*sponge.permutation.permuted.borrow(), 0);

        // Absorb 3 elements (more than R = 2), will trigger permutation
        sponge.absorb_unchecked(&[DummyUnit(1), DummyUnit(2), DummyUnit(3)]);

        // Permutation triggered once
        assert_eq!(*sponge.permutation.permuted.borrow(), 1);
        // Last item is in position 0
        assert_eq!(sponge.permutation.state[0], DummyUnit(3));
        // Position reset after permute
        assert_eq!(sponge.absorb_pos, 1);
    }

    #[test]
    fn test_squeeze_output_and_position() {
        let mut sponge = Sponge::new([0u8; 32]);
        // Preload the state with known values
        sponge.permutation.state[0] = DummyUnit(10);
        sponge.permutation.state[1] = DummyUnit(20);
        let mut out = [DummyUnit(0); 2];
        sponge.squeeze_unchecked(&mut out);
        // Output is correct: incremented by 1 and wrapped around
        assert_eq!(out, [DummyUnit(11), DummyUnit(21)]);
        // Position advanced
        assert_eq!(sponge.squeeze_pos, DummyPermutation::R);
    }

    #[test]
    fn test_squeeze_triggers_permute_when_full() {
        let mut sponge = Sponge::new([0u8; 32]);
        // Force permute
        sponge.squeeze_pos = DummyPermutation::R;
        let mut out = [DummyUnit(0); 1];
        // No permutation triggered yet
        assert_eq!(*sponge.permutation.permuted.borrow(), 0);
        // Squeeze position hardcoded to R
        assert_eq!(sponge.squeeze_pos, 2);
        sponge.squeeze_unchecked(&mut out);
        // Permutation triggered
        assert_eq!(*sponge.permutation.permuted.borrow(), 1);
        // Resumed from beginning
        assert_eq!(sponge.squeeze_pos, 1);
    }

    #[test]
    fn test_ratchet_zeros_state_and_permute() {
        let mut sponge = Sponge::new([0u8; 32]);
        // Set up dummy state
        sponge.permutation.state[0] = DummyUnit(99);
        sponge.permutation.state[1] = DummyUnit(100);

        // Permute not called yet
        assert_eq!(*sponge.permutation.permuted.borrow(), 0);

        sponge.ratchet_unchecked();

        // Permute called
        assert_eq!(*sponge.permutation.permuted.borrow(), 1);
        // State zeroed
        assert_eq!(sponge.permutation.state[0], DummyUnit(0));
        assert_eq!(sponge.permutation.state[1], DummyUnit(0));
        // Reset squeeze_pos
        assert_eq!(sponge.squeeze_pos, DummyPermutation::R);
    }
}
