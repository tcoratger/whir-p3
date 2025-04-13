use p3_field::{BasedVectorSpace, Field, PrimeField64};
use p3_symmetric::Hash;

use super::{
    DefaultHash,
    codecs::utils::{bytes_uniform_modp, from_be_bytes_mod_order},
    domain_separator::DomainSeparator,
    duplex_sponge::{Unit, interface::DuplexSpongeInterface},
    errors::{DomainSeparatorMismatch, ProofError, ProofResult},
    keccak::Keccak,
    pow::traits::PowStrategy,
    sho::HashStateWithInstructions,
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

    pub fn add_scalars<F>(&mut self, input: &[F]) -> ProofResult<()>
    where
        F: Field + BasedVectorSpace<F::PrimeSubfield>,
        F::PrimeSubfield: PrimeField64,
    {
        // Serialize the input scalars to bytes using the CommonFieldToUnit trait
        let serialized = self.public_scalars(input)?;

        // Append the serialized bytes to the internal transcript byte buffer
        self.narg_string.extend(serialized);

        // Return success
        Ok(())
    }

    pub fn public_scalars<F>(&mut self, input: &[F]) -> ProofResult<Vec<u8>>
    where
        F: Field + BasedVectorSpace<F::PrimeSubfield>,
        F::PrimeSubfield: PrimeField64,
    {
        // Initialize a buffer to store the final serialized byte output
        let mut buf = Vec::new();

        // How many bytes are needed to sample a single base field element
        let num_bytes = F::PrimeSubfield::bits().div_ceil(8);

        // Loop over each scalar field element (could be base or extension field)
        for scalar in input {
            // Decompose the field element into its basis coefficients over the base field
            //
            // For base fields, this is just [scalar]; for extensions, it's length-D array
            for coeff in scalar.as_basis_coefficients_slice() {
                // Serialize each base field coefficient to 4 bytes (LE canonical form)
                let bytes = coeff.as_canonical_u64().to_le_bytes();

                // Append the serialized bytes to the output buffer
                buf.extend_from_slice(&bytes[..num_bytes]);
            }
        }

        // Absorb the serialized bytes into the Fiat-Shamir transcript
        self.public_bytes(&buf)?;

        // Return the serialized byte representation
        Ok(buf)
    }

    #[inline]
    pub fn public_bytes(&mut self, input: &[u8]) -> Result<(), DomainSeparatorMismatch> {
        self.public_units(input)
    }

    /// Add public messages to the protocol transcript.
    /// Messages input to this function are not added to the protocol transcript.
    /// They are however absorbed into the verifier's sponge for Fiat-Shamir, and used to re-seed
    /// the prover state.
    pub fn public_units(&mut self, input: &[u8]) -> Result<(), DomainSeparatorMismatch> {
        let len = self.narg_string.len();
        self.add_units(input)?;
        self.narg_string.truncate(len);
        Ok(())
    }

    /// Fill a slice with uniformly-distributed challenges from the verifier.
    pub fn fill_challenge_units(
        &mut self,
        output: &mut [u8],
    ) -> Result<(), DomainSeparatorMismatch> {
        self.hash_state.squeeze(output)
    }

    pub fn challenge_pow<S: PowStrategy>(&mut self, bits: f64) -> ProofResult<()> {
        let challenge = self.challenge_bytes()?;
        let nonce = S::new(challenge, bits)
            .solve()
            .ok_or(ProofError::InvalidProof)?;
        self.add_bytes(&nonce.to_be_bytes())?;
        Ok(())
    }

    #[inline]
    pub fn fill_challenge_bytes(
        &mut self,
        output: &mut [u8],
    ) -> Result<(), DomainSeparatorMismatch> {
        self.fill_challenge_units(output)
    }

    pub fn challenge_bytes<const N: usize>(&mut self) -> Result<[u8; N], DomainSeparatorMismatch> {
        let mut output = [0u8; N];
        self.fill_challenge_bytes(&mut output)?;
        Ok(output)
    }

    pub fn add_digest<F, const DIGEST_ELEMS: usize>(
        &mut self,
        digest: Hash<F, u8, DIGEST_ELEMS>,
    ) -> ProofResult<()>
    where
        F: Field,
    {
        self.add_bytes(digest.as_ref())
            .map_err(ProofError::InvalidDomainSeparator)
    }

    pub fn fill_challenge_scalars<F>(&mut self, output: &mut [F]) -> ProofResult<()>
    where
        F: Field + BasedVectorSpace<F::PrimeSubfield>,
        F::PrimeSubfield: PrimeField64,
    {
        // How many bytes are needed to sample a single base field element
        let base_field_size = bytes_uniform_modp(F::PrimeSubfield::bits() as u32);

        // Total bytes needed for one F element = extension degree × base field size
        let field_byte_len = F::DIMENSION * F::PrimeSubfield::DIMENSION * base_field_size;

        // Temporary buffer to hold bytes for each field element
        let mut buf = vec![0u8; field_byte_len];

        // Fill each output element from fresh transcript randomness
        for o in output.iter_mut() {
            // Draw uniform bytes from the transcript
            self.fill_challenge_bytes(&mut buf)?;

            // For each chunk, convert to base field element via modular reduction
            let base_coeffs = buf.chunks(base_field_size).map(from_be_bytes_mod_order);

            // Reconstruct the full field element using canonical basis
            *o = F::from_basis_coefficients_iter(base_coeffs);
        }

        Ok(())
    }

    pub fn challenge_scalars<F, const N: usize>(&mut self) -> ProofResult<[F; N]>
    where
        F: Field + BasedVectorSpace<F::PrimeSubfield>,
        F::PrimeSubfield: PrimeField64,
    {
        let mut output = [F::default(); N];
        self.fill_challenge_scalars(&mut output)?;
        Ok(output)
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
    use p3_baby_bear::BabyBear;
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};
    use p3_goldilocks::Goldilocks;

    use super::*;

    type H = DefaultHash;

    type F = BabyBear;
    type EF4 = BinomialExtensionField<F, 4>;

    type G = Goldilocks;
    type EG2 = BinomialExtensionField<G, 2>;

    #[test]
    fn test_prover_state_public_units_does_not_affect_narg() {
        let mut domsep = DomainSeparator::<DefaultHash>::new("test");
        domsep.absorb(4, "data");
        let mut pstate = ProverState::from(&domsep);

        pstate.public_units(&[1, 2, 3, 4]).unwrap();
        assert_eq!(pstate.narg_string(), b"");
    }

    #[test]
    fn test_add_units_appends_to_narg_string() {
        let mut domsep = DomainSeparator::<DefaultHash>::new("test");
        domsep.absorb(3, "msg");
        let mut pstate = ProverState::from(&domsep);
        let input = [42, 43, 44];

        assert!(pstate.add_units(&input).is_ok());
        assert_eq!(pstate.narg_string(), &input);
    }

    #[test]
    fn test_add_units_too_many_elements_should_error() {
        let mut domsep = DomainSeparator::<DefaultHash>::new("test");
        domsep.absorb(2, "short");
        let mut pstate = ProverState::from(&domsep);

        let result = pstate.add_units(&[1, 2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_public_units_does_not_update_transcript() {
        let mut domsep = DomainSeparator::<DefaultHash>::new("test");
        domsep.absorb(2, "p");
        let mut pstate = ProverState::from(&domsep);
        let _ = pstate.public_units(&[0xaa, 0xbb]);

        assert_eq!(pstate.narg_string(), b"");
    }

    #[test]
    fn test_fill_challenge_units() {
        let mut domsep = DomainSeparator::<DefaultHash>::new("test");
        domsep.squeeze(8, "ch");
        let mut pstate = ProverState::from(&domsep);

        let mut out = [0u8; 8];
        let _ = pstate.fill_challenge_units(&mut out);
        assert_eq!(out, [77, 249, 17, 180, 176, 109, 121, 62]);
    }

    #[test]
    fn test_add_units_multiple_accumulates() {
        let mut domsep = DomainSeparator::<DefaultHash>::new("t");
        domsep.absorb(2, "a");
        domsep.absorb(3, "b");
        let mut p = ProverState::from(&domsep);

        p.add_units(&[10, 11]).unwrap();
        p.add_units(&[20, 21, 22]).unwrap();

        assert_eq!(p.narg_string(), &[10, 11, 20, 21, 22]);
    }

    #[test]
    fn test_narg_string_round_trip_check() {
        let mut domsep = DomainSeparator::<DefaultHash>::new("t");
        domsep.absorb(5, "data");
        let mut p = ProverState::from(&domsep);

        let msg = b"zkp42";
        p.add_units(msg).unwrap();

        let encoded = p.narg_string();
        assert_eq!(encoded, msg);
    }

    #[test]
    fn test_add_scalars_babybear() {
        // Step 1: Create a domain separator with the label "test"
        let mut domsep: DomainSeparator<H> = DomainSeparator::new("test");

        // Step 2: Add an "absorb scalars" tag for 3 scalars, with label "com"
        // This ensures deterministic transcript layout
        domsep.add_scalars::<F>(3, "com");

        // Step 3: Initialize the prover state from the domain separator
        let mut prover_state = domsep.to_prover_state();

        // Step 4: Create 3 BabyBear field elements
        let f0 = F::from_u64(111);
        let f1 = F::from_u64(222);
        let f2 = F::from_u64(333);

        // Step 5: Add the scalars to the transcript
        prover_state.add_scalars(&[f0, f1, f2]).unwrap();

        // Step 6: Expected serialized bytes (little-endian u32 encoding)
        let expected_bytes = vec![
            111, 0, 0, 0, // 111
            222, 0, 0, 0, // 222
            77, 1, 0, 0, // 333 = 0x014D = [77, 1, 0, 0]
        ];

        // Step 7: Check that transcript matches expected encoding
        assert_eq!(
            prover_state.narg_string, expected_bytes,
            "Transcript serialization mismatch"
        );

        // Step 8: Verify determinism by repeating with a new prover
        let mut prover_state2 = domsep.to_prover_state();
        prover_state2.add_scalars(&[f0, f1, f2]).unwrap();

        assert_eq!(
            prover_state.narg_string, prover_state2.narg_string,
            "Transcript encoding should be deterministic for same inputs"
        );
    }

    #[test]
    fn test_add_scalars_goldilocks() {
        // Step 1: Create a domain separator with the label "test"
        let mut domsep: DomainSeparator<H> = DomainSeparator::new("test");

        // Step 2: Add an "absorb scalars" tag for 3 scalars, with label "com"
        // This ensures deterministic transcript layout
        domsep.add_scalars::<G>(3, "com");

        // Step 3: Initialize the prover state from the domain separator
        let mut prover_state = domsep.to_prover_state();

        // Step 4: Create 3 Goldilocks field elements
        let f0 = G::from_u64(111);
        let f1 = G::from_u64(222);
        let f2 = G::from_u64(333);

        // Step 5: Add the scalars to the transcript
        prover_state.add_scalars(&[f0, f1, f2]).unwrap();

        // Step 6: Expected serialized bytes (little-endian u32 encoding)
        let expected_bytes = vec![
            111, 0, 0, 0, 0, 0, 0, 0, // 111
            222, 0, 0, 0, 0, 0, 0, 0, // 222
            77, 1, 0, 0, 0, 0, 0, 0, // 333 = 0x014D = [77, 1, 0, 0, 0, 0, 0, 0]
        ];

        // Step 7: Check that transcript matches expected encoding
        assert_eq!(
            prover_state.narg_string, expected_bytes,
            "Transcript serialization mismatch"
        );

        // Step 8: Verify determinism by repeating with a new prover
        let mut prover_state2 = domsep.to_prover_state();
        prover_state2.add_scalars(&[f0, f1, f2]).unwrap();

        assert_eq!(
            prover_state.narg_string, prover_state2.narg_string,
            "Transcript encoding should be deterministic for same inputs"
        );
    }

    #[test]
    fn test_add_scalars_extension_babybear() {
        // Step 1: Create a domain separator with the label "test"
        let mut domsep: DomainSeparator<H> = DomainSeparator::new("test");

        // Step 2: Add absorb-scalar tag for EF4 type and 3 values
        domsep.add_scalars::<EF4>(3, "com");

        // Step 3: Initialize the prover state from the domain separator
        let mut prover_state = domsep.to_prover_state();

        // Step 4: Construct 3 extension field elements
        // - One large (MAX) value to ensure all 4 limbs are filled
        // - Two small values (fit in low limb only)
        let f0 = EF4::from_u64(u64::MAX);
        let f1 = EF4::from_u64(222);
        let f2 = EF4::from_u64(333);

        // Step 5: Add the scalars to the transcript
        prover_state.add_scalars(&[f0, f1, f2]).unwrap();

        // Step 6: Expected bytes from 3 extension field elements, each with 4 limbs
        // - 4 * 3 = 12 BabyBear limbs = 12 * 4 = 48 bytes
        let expected_bytes = vec![
            // f0 = u64::MAX → nontrivial encoded limb
            226, 221, 221, 69, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            // f1 = 222 → only first limb has value
            222, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            // f2 = 333 → only first limb has value
            77, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];

        // Step 7: Validate that the transcript encoding matches the expected bytes
        assert_eq!(
            prover_state.narg_string, expected_bytes,
            "Transcript serialization mismatch"
        );

        // Step 8: Repeat with a second prover to confirm determinism
        let mut prover_state2 = domsep.to_prover_state();
        prover_state2.add_scalars(&[f0, f1, f2]).unwrap();

        assert_eq!(
            prover_state.narg_string, prover_state2.narg_string,
            "Transcript encoding should be deterministic for same inputs"
        );
    }

    #[test]
    fn test_add_scalars_extension_goldilocks() {
        // Step 1: Create a domain separator with the label "test"
        let mut domsep: DomainSeparator<H> = DomainSeparator::new("test");

        // Step 2: Add absorb-scalar tag for EG2 type and 3 values
        domsep.add_scalars::<EG2>(3, "com");

        // Step 3: Initialize the prover state from the domain separator
        let mut prover_state = domsep.to_prover_state();

        // Step 4: Construct 3 extension field elements
        // - One large (MAX) value to ensure all 4 limbs are filled
        // - Two small values (fit in low limb only)
        let f0 = EG2::from_u64(u64::MAX);
        let f1 = EG2::from_u64(222);
        let f2 = EG2::from_u64(333);

        // Step 5: Add the scalars to the transcript
        prover_state.add_scalars(&[f0, f1, f2]).unwrap();

        // Step 6: Expected bytes from 3 extension field elements, each with 8 limbs
        let expected_bytes = vec![
            // f0 = u64::MAX → nontrivial encoded limb
            254, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            // f1 = 222 → only first limb has value
            222, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            // f2 = 333 → only first limb has value
            77, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];

        // Step 7: Validate that the transcript encoding matches the expected bytes
        assert_eq!(
            prover_state.narg_string, expected_bytes,
            "Transcript serialization mismatch"
        );

        // Step 8: Repeat with a second prover to confirm determinism
        let mut prover_state2 = domsep.to_prover_state();
        prover_state2.add_scalars(&[f0, f1, f2]).unwrap();

        assert_eq!(
            prover_state.narg_string, prover_state2.narg_string,
            "Transcript encoding should be deterministic for same inputs"
        );
    }
}
