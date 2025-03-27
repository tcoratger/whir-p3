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
