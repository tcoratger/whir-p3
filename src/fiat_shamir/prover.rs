use p3_field::Field;
use rand::{TryCryptoRng, TryRngCore};

use super::{
    DefaultHash, DefaultRng,
    domain_separator::DomainSeparator,
    duplex_sponge::{Unit, interface::DuplexSpongeInterface},
    errors::DomainSeparatorMismatch,
    keccak::Keccak,
    sho::HashStateWithInstructions,
    traits::UnitTranscript,
};
use crate::fiat_shamir::traits::{ByteWriter, CommonUnitToBytes};

/// [`ProverState`] is the prover state of an interactive proof (IP) system.
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
pub struct ProverState<H = DefaultHash, U = u8, R = DefaultRng>
where
    U: Unit,
    H: DuplexSpongeInterface<U>,
    R: TryRngCore + TryCryptoRng,
{
    /// The randomness state of the prover.
    pub(crate) rng: ProverPrivateRng<R>,
    /// The public coins for the protocol
    pub(crate) hash_state: HashStateWithInstructions<H, U>,
    /// The encoded data.
    pub(crate) narg_string: Vec<u8>,
}

impl<H, U, R> ProverState<H, U, R>
where
    H: DuplexSpongeInterface<U>,
    R: TryRngCore + TryCryptoRng,
    U: Unit,
{
    pub fn new(domain_separator: &DomainSeparator<H, U>, csrng: R) -> Self {
        let hash_state = HashStateWithInstructions::new(domain_separator);

        let mut duplex_sponge = Keccak::default();
        duplex_sponge.absorb_unchecked(domain_separator.as_bytes());
        let rng = ProverPrivateRng { ds: duplex_sponge, csrng };

        Self { rng, hash_state, narg_string: Vec::new() }
    }

    /// Add a slice `[U]` to the protocol transcript.
    /// The messages are also internally encoded in the protocol transcript,
    /// and used to re-seed the prover's random number generator.
    #[inline(always)]
    pub fn add_units(&mut self, input: &[U]) -> Result<(), DomainSeparatorMismatch> {
        let old_len = self.narg_string.len();
        self.hash_state.absorb(input)?;
        // write never fails on Vec<u8>
        U::write(input, &mut self.narg_string).unwrap();
        self.rng.ds.absorb_unchecked(&self.narg_string[old_len..]);

        Ok(())
    }

    /// Ratchet the verifier's state.
    #[inline(always)]
    pub fn ratchet(&mut self) -> Result<(), DomainSeparatorMismatch> {
        self.hash_state.ratchet()
    }

    /// Return a reference to the random number generator associated to the protocol transcript.
    #[inline(always)]
    pub fn rng(&mut self) -> &mut (impl TryCryptoRng + TryRngCore) {
        &mut self.rng
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
}

impl<H, U, R> UnitTranscript<U> for ProverState<H, U, R>
where
    U: Unit,
    H: DuplexSpongeInterface<U>,
    R: TryRngCore + TryCryptoRng,
{
    /// Add public messages to the protocol transcript.
    /// Messages input to this function are not added to the protocol transcript.
    /// They are however absorbed into the verifier's sponge for Fiat-Shamir, and used to re-seed
    /// the prover state.
    fn public_units(&mut self, input: &[U]) -> Result<(), DomainSeparatorMismatch> {
        let len = self.narg_string.len();
        self.add_units(input)?;
        self.narg_string.truncate(len);
        Ok(())
    }

    /// Fill a slice with uniformly-distributed challenges from the verifier.
    fn fill_challenge_units(&mut self, output: &mut [U]) -> Result<(), DomainSeparatorMismatch> {
        self.hash_state.squeeze(output)
    }
}

impl<H, U, R> core::fmt::Debug for ProverState<H, U, R>
where
    U: Unit,
    H: DuplexSpongeInterface<U>,
    R: TryRngCore + TryCryptoRng,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.hash_state.fmt(f)
    }
}

impl<U, H> From<&DomainSeparator<H, U>> for ProverState<H, U, DefaultRng>
where
    U: Unit,
    H: DuplexSpongeInterface<U>,
{
    fn from(domain_separator: &DomainSeparator<H, U>) -> Self {
        ProverState::new(domain_separator, DefaultRng::default())
    }
}

// impl<H, R> ByteWriter for ProverState<H, u8, R>
// where
//     H: DuplexSpongeInterface<u8>,
//     R: TryRngCore + TryCryptoRng,
// {
//     #[inline(always)]
//     fn add_bytes(&mut self, input: &[u8]) -> Result<(), DomainSeparatorMismatch> {
//         self.add_units(input)
//     }
// }

/// A cryptographically-secure random number generator that is bound to the protocol transcript.
///
/// For most public-coin protocols it is *vital* not to have two different verifier messages for the
/// same prover message. For this reason, we construct a Rng that will absorb whatever the verifier
/// absorbs, and that in addition it is seeded by a cryptographic random number generator (by
/// default, [`rand::rngs::OsRng`]).
///
/// Every time a challenge is being generated, the private prover sponge is ratcheted, so that it
/// can't be inverted and the randomness recovered.
#[derive(Debug)]
pub(crate) struct ProverPrivateRng<R: TryRngCore + TryCryptoRng> {
    /// The duplex sponge that is used to generate the random coins.
    pub(crate) ds: Keccak,
    /// The cryptographic random number generator that seeds the sponge.
    pub(crate) csrng: R,
}

impl<R: TryRngCore + TryCryptoRng> TryRngCore for ProverPrivateRng<R> {
    type Error = std::fmt::Error;

    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        let mut buf = [0u8; 4];
        self.try_fill_bytes(buf.as_mut())?;
        Ok(u32::from_le_bytes(buf))
    }

    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        let mut buf = [0u8; 8];
        self.try_fill_bytes(buf.as_mut())?;
        Ok(u64::from_le_bytes(buf))
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Self::Error> {
        self.ds.squeeze_unchecked(dest);
        Ok(())
    }
}

impl<R: TryRngCore + TryCryptoRng> TryCryptoRng for ProverPrivateRng<R> {}
