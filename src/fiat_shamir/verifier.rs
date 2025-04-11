use super::{
    DefaultHash,
    domain_separator::DomainSeparator,
    duplex_sponge::{Unit, interface::DuplexSpongeInterface},
    errors::DomainSeparatorMismatch,
    sho::HashStateWithInstructions,
    traits::{BytesToUnitDeserialize, UnitTranscript},
};

/// [`VerifierState`] is the verifier state.
///
/// Internally, it simply contains a stateful hash.
/// Given as input an [`DomainSeparator`] and a NARG string, it allows to
/// de-serialize elements from the NARG string and make them available to the zero-knowledge
/// verifier.
#[derive(Debug)]
pub struct VerifierState<'a, H = DefaultHash, U = u8>
where
    H: DuplexSpongeInterface<U>,
    U: Unit,
{
    pub(crate) hash_state: HashStateWithInstructions<H, U>,
    pub(crate) narg_string: &'a [u8],
}

impl<'a, U: Unit, H: DuplexSpongeInterface<U>> VerifierState<'a, H, U> {
    /// Creates a new [`VerifierState`] instance with the given sponge and IO Pattern.
    ///
    /// The resulting object will act as the verifier in a zero-knowledge protocol.
    ///
    /// ```ignore
    /// # use spongefish::*;
    ///
    /// let domsep = DomainSeparator::<DefaultHash>::new("📝").absorb(1, "inhale 🫁").squeeze(32, "exhale 🎏");
    /// // A silly NARG string for the example.
    /// let narg_string = &[0x42];
    /// let mut verifier_state = domsep.to_verifier_state(narg_string);
    /// assert_eq!(verifier_state.next_bytes().unwrap(), [0x42]);
    /// let challenge = verifier_state.challenge_bytes::<32>();
    /// assert!(challenge.is_ok());
    /// assert_ne!(challenge.unwrap(), [0; 32]);
    /// ```
    pub fn new(domain_separator: &DomainSeparator<H, U>, narg_string: &'a [u8]) -> Self {
        let hash_state = HashStateWithInstructions::new(domain_separator);
        Self {
            hash_state,
            narg_string,
        }
    }

    /// Read `input.len()` elements from the NARG string.
    #[inline]
    pub fn fill_next_units(&mut self, input: &mut [U]) -> Result<(), DomainSeparatorMismatch> {
        U::read(&mut self.narg_string, input)?;
        self.hash_state.absorb(input)?;
        Ok(())
    }
}

impl<H: DuplexSpongeInterface<U>, U: Unit> UnitTranscript<U> for VerifierState<'_, H, U> {
    /// Add native elements to the sponge without writing them to the NARG string.
    #[inline]
    fn public_units(&mut self, input: &[U]) -> Result<(), DomainSeparatorMismatch> {
        self.hash_state.absorb(input)
    }

    /// Fill `input` with units sampled uniformly at random.
    #[inline]
    fn fill_challenge_units(&mut self, input: &mut [U]) -> Result<(), DomainSeparatorMismatch> {
        self.hash_state.squeeze(input)
    }
}

impl<H: DuplexSpongeInterface<u8>> BytesToUnitDeserialize for VerifierState<'_, H, u8> {
    /// Read the next `input.len()` bytes from the NARG string and return them.
    #[inline]
    fn fill_next_bytes(&mut self, input: &mut [u8]) -> Result<(), DomainSeparatorMismatch> {
        self.fill_next_units(input)
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use super::*;

    #[derive(Default, Clone)]
    struct DummySponge {
        pub absorbed: Rc<RefCell<Vec<u8>>>,
        pub squeezed: Rc<RefCell<Vec<u8>>>,
        pub ratcheted: Rc<RefCell<bool>>,
    }

    impl zeroize::Zeroize for DummySponge {
        fn zeroize(&mut self) {
            self.absorbed.borrow_mut().clear();
            self.squeezed.borrow_mut().clear();
            *self.ratcheted.borrow_mut() = false;
        }
    }

    impl DummySponge {
        fn new_inner() -> Self {
            Self {
                absorbed: Rc::new(RefCell::new(Vec::new())),
                squeezed: Rc::new(RefCell::new(Vec::new())),
                ratcheted: Rc::new(RefCell::new(false)),
            }
        }
    }

    impl DuplexSpongeInterface<u8> for DummySponge {
        fn new(_iv: [u8; 32]) -> Self {
            Self::new_inner()
        }

        fn absorb_unchecked(&mut self, input: &[u8]) -> &mut Self {
            self.absorbed.borrow_mut().extend_from_slice(input);
            self
        }

        fn squeeze_unchecked(&mut self, output: &mut [u8]) -> &mut Self {
            for (i, byte) in output.iter_mut().enumerate() {
                *byte = i as u8;
            }
            self.squeezed.borrow_mut().extend_from_slice(output);
            self
        }

        fn ratchet_unchecked(&mut self) -> &mut Self {
            *self.ratcheted.borrow_mut() = true;
            self
        }
    }

    #[test]
    fn test_new_verifier_state_constructs_correctly() {
        let ds = DomainSeparator::<DummySponge>::new("test");
        let transcript = b"abc";
        let vs = VerifierState::<DummySponge>::new(&ds, transcript);
        assert_eq!(vs.narg_string, b"abc");
    }

    #[test]
    fn test_fill_next_units_reads_and_absorbs() {
        let ds = DomainSeparator::<DummySponge>::new("x").absorb(3, "input");
        let mut vs = VerifierState::<DummySponge>::new(&ds, b"abc");
        let mut buf = [0u8; 3];
        let res = vs.fill_next_units(&mut buf);
        assert!(res.is_ok());
        assert_eq!(buf, *b"abc");
        assert_eq!(*vs.hash_state.ds().absorbed.borrow(), b"abc");
    }

    #[test]
    fn test_fill_next_units_with_insufficient_data_errors() {
        let ds = DomainSeparator::<DummySponge>::new("x").absorb(4, "fail");
        let mut vs = VerifierState::<DummySponge>::new(&ds, b"xy");
        let mut buf = [0u8; 4];
        let res = vs.fill_next_units(&mut buf);
        assert!(res.is_err());
    }

    #[test]
    fn test_unit_transcript_public_units() {
        let ds = DomainSeparator::<DummySponge>::new("x").absorb(2, "public");
        let mut vs = VerifierState::<DummySponge>::new(&ds, b"..");
        assert!(vs.public_units(&[1, 2]).is_ok());
        assert_eq!(*vs.hash_state.ds().absorbed.borrow(), &[1, 2]);
    }

    #[test]
    fn test_unit_transcript_fill_challenge_units() {
        let ds = DomainSeparator::<DummySponge>::new("x").squeeze(4, "c");
        let mut vs = VerifierState::<DummySponge>::new(&ds, b"abcd");
        let mut out = [0u8; 4];
        assert!(vs.fill_challenge_units(&mut out).is_ok());
        assert_eq!(out, [0, 1, 2, 3]);
    }

    #[test]
    fn test_fill_next_bytes_impl() {
        let ds = DomainSeparator::<DummySponge>::new("x").absorb(3, "byte");
        let mut vs = VerifierState::<DummySponge>::new(&ds, b"xyz");
        let mut out = [0u8; 3];
        assert!(vs.fill_next_bytes(&mut out).is_ok());
        assert_eq!(out, *b"xyz");
    }
}
