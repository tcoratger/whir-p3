use super::{
    DefaultHash,
    domain_separator::DomainSeparator,
    duplex_sponge::{Unit, interface::DuplexSpongeInterface},
    errors::DomainSeparatorMismatch,
    keccak::Keccak,
    sho::HashStateWithInstructions,
    traits::UnitTranscript,
};

/// [`ProverState`] is the prover state of an interactive proof (IP) system.
///
/// It internally holds the **secret coins** of the prover for zero-knowledge, and
/// has the hash function state for the verifier state.
///
/// Unless otherwise specified,
/// [`ProverState`] is set to work over bytes with [`DefaultHash`] and
/// rely on the default random number generator [`DefaultRng`].
///
///
/// # Safety
///
/// The prover state is meant to be private in contexts where zero-knowledge is desired.
/// Leaking the prover state *will* leak the prover's private coins and as such it will compromise
/// the zero-knowledge property. [`ProverState`] does not implement [`Clone`] or [`Copy`] to prevent
/// accidental leaks.
#[derive(Debug)]
pub struct ProverState<H = DefaultHash>
where
    H: DuplexSpongeInterface<u8>,
{
    /// The duplex sponge that is used to generate the random coins.
    pub(crate) ds: Keccak,
    /// The public coins for the protocol
    pub(crate) hash_state: HashStateWithInstructions<H>,
    /// The encoded data.
    pub(crate) narg_string: Vec<u8>,
}

impl<H> ProverState<H>
where
    H: DuplexSpongeInterface<u8>,
{
    pub fn new(domain_separator: &DomainSeparator<H>) -> Self {
        let hash_state = HashStateWithInstructions::new(domain_separator);

        let mut duplex_sponge = Keccak::default();
        duplex_sponge.absorb_unchecked(domain_separator.as_bytes());

        Self {
            ds: duplex_sponge,
            hash_state,
            narg_string: Vec::new(),
        }
    }

    /// Add a slice `[U]` to the protocol transcript.
    /// The messages are also internally encoded in the protocol transcript,
    /// and used to re-seed the prover's random number generator.
    pub fn add_units(&mut self, input: &[u8]) -> Result<(), DomainSeparatorMismatch> {
        let old_len = self.narg_string.len();
        self.hash_state.absorb(input)?;
        // write never fails on Vec<u8>
        u8::write(input, &mut self.narg_string).unwrap();
        self.ds.absorb_unchecked(&self.narg_string[old_len..]);

        Ok(())
    }

    /// Return the current protocol transcript.
    /// The protocol transcript does not have any information about the length or the type of the
    /// messages being read. This is because the information is considered pre-shared within the
    /// [`DomainSeparator`]. Additionally, since the verifier challenges are deterministically
    /// generated from the prover's messages, the transcript does not hold any of the verifier's
    /// messages.
    pub fn narg_string(&self) -> &[u8] {
        self.narg_string.as_slice()
    }

    pub fn add_bytes(&mut self, input: &[u8]) -> Result<(), DomainSeparatorMismatch> {
        self.add_units(input)
    }
}

impl<H> UnitTranscript<u8> for ProverState<H>
where
    H: DuplexSpongeInterface<u8>,
{
    /// Add public messages to the protocol transcript.
    /// Messages input to this function are not added to the protocol transcript.
    /// They are however absorbed into the verifier's sponge for Fiat-Shamir, and used to re-seed
    /// the prover state.
    fn public_units(&mut self, input: &[u8]) -> Result<(), DomainSeparatorMismatch> {
        let len = self.narg_string.len();
        self.add_units(input)?;
        self.narg_string.truncate(len);
        Ok(())
    }

    /// Fill a slice with uniformly-distributed challenges from the verifier.
    fn fill_challenge_units(&mut self, output: &mut [u8]) -> Result<(), DomainSeparatorMismatch> {
        self.hash_state.squeeze(output)
    }
}

impl<H> From<&DomainSeparator<H>> for ProverState<H>
where
    H: DuplexSpongeInterface<u8>,
{
    fn from(domain_separator: &DomainSeparator<H>) -> Self {
        Self::new(domain_separator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prover_state_public_units_does_not_affect_narg() {
        let domsep = DomainSeparator::<DefaultHash>::new("test").absorb(4, "data");
        let mut pstate = ProverState::from(&domsep);

        pstate.public_units(&[1, 2, 3, 4]).unwrap();
        assert_eq!(pstate.narg_string(), b"");
    }

    #[test]
    fn test_add_units_appends_to_narg_string() {
        let domsep = DomainSeparator::<DefaultHash>::new("test").absorb(3, "msg");
        let mut pstate = ProverState::from(&domsep);
        let input = [42, 43, 44];

        assert!(pstate.add_units(&input).is_ok());
        assert_eq!(pstate.narg_string(), &input);
    }

    #[test]
    fn test_add_units_too_many_elements_should_error() {
        let domsep = DomainSeparator::<DefaultHash>::new("test").absorb(2, "short");
        let mut pstate = ProverState::from(&domsep);

        let result = pstate.add_units(&[1, 2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_public_units_does_not_update_transcript() {
        let domsep = DomainSeparator::<DefaultHash>::new("test").absorb(2, "p");
        let mut pstate = ProverState::from(&domsep);
        let _ = pstate.public_units(&[0xaa, 0xbb]);

        assert_eq!(pstate.narg_string(), b"");
    }

    #[test]
    fn test_fill_challenge_units() {
        let domsep = DomainSeparator::<DefaultHash>::new("test").squeeze(8, "ch");
        let mut pstate = ProverState::from(&domsep);

        let mut out = [0u8; 8];
        let _ = pstate.fill_challenge_units(&mut out);
        assert_eq!(out, [77, 249, 17, 180, 176, 109, 121, 62]);
    }

    #[test]
    fn test_add_units_multiple_accumulates() {
        let domsep = DomainSeparator::<DefaultHash>::new("t")
            .absorb(2, "a")
            .absorb(3, "b");
        let mut p = ProverState::from(&domsep);

        p.add_units(&[10, 11]).unwrap();
        p.add_units(&[20, 21, 22]).unwrap();

        assert_eq!(p.narg_string(), &[10, 11, 20, 21, 22]);
    }

    #[test]
    fn test_narg_string_round_trip_check() {
        let domsep = DomainSeparator::<DefaultHash>::new("t").absorb(5, "data");
        let mut p = ProverState::from(&domsep);

        let msg = b"zkp42";
        p.add_units(msg).unwrap();

        let encoded = p.narg_string();
        assert_eq!(encoded, msg);
    }
}
