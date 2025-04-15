use std::marker::PhantomData;

use p3_field::{
    BasedVectorSpace, ExtensionField, Field, PrimeCharacteristicRing, PrimeField64, TwoAdicField,
};
use p3_symmetric::Hash;

use super::{
    DefaultHash, UnitToBytes,
    domain_separator::DomainSeparator,
    duplex_sponge::{Unit, interface::DuplexSpongeInterface},
    errors::{DomainSeparatorMismatch, ProofError, ProofResult},
    keccak::Keccak,
    pow::traits::PowStrategy,
    sho::HashStateWithInstructions,
    utils::{bytes_uniform_modp, from_be_bytes_mod_order},
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
pub struct ProverState<EF, F, H = DefaultHash>
where
    H: DuplexSpongeInterface<u8>,
    EF: ExtensionField<F> + ExtensionField<<EF as PrimeCharacteristicRing>::PrimeSubfield>,
    F: Field + TwoAdicField,
{
    /// The duplex sponge that is used to generate the random coins.
    pub(crate) ds: Keccak,
    /// The public coins for the protocol
    pub(crate) hash_state: HashStateWithInstructions<H>,
    /// The encoded data.
    pub(crate) narg_string: Vec<u8>,
    /// Marker for the field.
    _field: PhantomData<F>,
    /// Marker for the extension field.
    _extension_field: PhantomData<EF>,
}

impl<EF, F, H> ProverState<EF, F, H>
where
    H: DuplexSpongeInterface<u8>,
    EF: ExtensionField<F>
        + TwoAdicField<PrimeSubfield = F>
        + ExtensionField<<EF as PrimeCharacteristicRing>::PrimeSubfield>,
    EF::PrimeSubfield: PrimeField64,
    F: Field + TwoAdicField,
{
    pub fn new(domain_separator: &DomainSeparator<EF, F, H>) -> Self {
        let hash_state = HashStateWithInstructions::new(domain_separator);

        let mut duplex_sponge = Keccak::default();
        duplex_sponge.absorb_unchecked(domain_separator.as_bytes());

        Self {
            ds: duplex_sponge,
            hash_state,
            narg_string: Vec::new(),
            _field: PhantomData,
            _extension_field: PhantomData,
        }
    }

    /// Add a slice `[U]` to the protocol transcript.
    /// The messages are also internally encoded in the protocol transcript,
    /// and used to re-seed the prover's random number generator.
    pub fn add_bytes(&mut self, input: &[u8]) -> Result<(), DomainSeparatorMismatch> {
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

    pub fn add_scalars(&mut self, input: &[EF]) -> ProofResult<()> {
        // Serialize the input scalars to bytes
        let serialized = self.public_scalars(input)?;

        // Append the serialized bytes to the internal transcript byte buffer
        self.narg_string.extend(serialized);

        // Return success
        Ok(())
    }

    pub fn public_scalars(&mut self, input: &[EF]) -> ProofResult<Vec<u8>> {
        // Initialize a buffer to store the final serialized byte output
        let mut buf = Vec::new();

        // How many bytes are needed to sample a single base field element
        let num_bytes = EF::PrimeSubfield::bits().div_ceil(8);

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

    /// Add public messages to the protocol transcript.
    /// Messages input to this function are not added to the protocol transcript.
    /// They are however absorbed into the verifier's sponge for Fiat-Shamir, and used to re-seed
    /// the prover state.
    pub fn public_bytes(&mut self, input: &[u8]) -> Result<(), DomainSeparatorMismatch> {
        let len = self.narg_string.len();
        self.add_bytes(input)?;
        self.narg_string.truncate(len);
        Ok(())
    }

    pub fn challenge_pow<S: PowStrategy>(&mut self, bits: f64) -> ProofResult<()> {
        let challenge = self.challenge_bytes()?;
        let nonce = S::new(challenge, bits)
            .solve()
            .ok_or(ProofError::InvalidProof)?;
        self.add_bytes(&nonce.to_be_bytes())?;
        Ok(())
    }

    pub fn challenge_bytes<const N: usize>(&mut self) -> Result<[u8; N], DomainSeparatorMismatch> {
        let mut output = [0u8; N];
        self.fill_challenge_bytes(&mut output)?;
        Ok(output)
    }

    pub fn add_digest<const DIGEST_ELEMS: usize>(
        &mut self,
        digest: Hash<F, u8, DIGEST_ELEMS>,
    ) -> ProofResult<()> {
        self.add_bytes(digest.as_ref())
            .map_err(ProofError::InvalidDomainSeparator)
    }

    pub fn fill_challenge_scalars(&mut self, output: &mut [EF]) -> ProofResult<()> {
        // How many bytes are needed to sample a single base field element
        let base_field_size = bytes_uniform_modp(EF::PrimeSubfield::bits() as u32);

        // Total bytes needed for one EF element = extension degree × base field size
        let field_byte_len = EF::DIMENSION * EF::PrimeSubfield::DIMENSION * base_field_size;

        // Temporary buffer to hold bytes for each field element
        let mut buf = vec![0u8; field_byte_len];

        // Fill each output element from fresh transcript randomness
        for o in output.iter_mut() {
            // Draw uniform bytes from the transcript
            self.fill_challenge_bytes(&mut buf)?;

            // For each chunk, convert to base field element via modular reduction
            let base_coeffs = buf.chunks(base_field_size).map(from_be_bytes_mod_order);

            // Reconstruct the full field element using canonical basis
            *o = EF::from_basis_coefficients_iter(base_coeffs);
        }

        Ok(())
    }

    pub fn challenge_scalars<const N: usize>(&mut self) -> ProofResult<[EF; N]> {
        let mut output = [EF::default(); N];
        self.fill_challenge_scalars(&mut output)?;
        Ok(output)
    }
}

impl<EF, F, H> UnitToBytes for ProverState<EF, F, H>
where
    H: DuplexSpongeInterface<u8>,
    EF: ExtensionField<F> + ExtensionField<<EF as PrimeCharacteristicRing>::PrimeSubfield>,
    F: Field + TwoAdicField,
{
    fn fill_challenge_bytes(&mut self, output: &mut [u8]) -> Result<(), DomainSeparatorMismatch> {
        self.hash_state.squeeze(output)
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
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
        let mut domsep = DomainSeparator::<F, F, DefaultHash>::new("test");
        domsep.absorb(4, "data");
        let mut pstate = domsep.to_prover_state();

        pstate.public_bytes(&[1, 2, 3, 4]).unwrap();
        assert_eq!(pstate.narg_string(), b"");
    }

    #[test]
    fn test_add_units_appends_to_narg_string() {
        let mut domsep = DomainSeparator::<F, F, DefaultHash>::new("test");
        domsep.absorb(3, "msg");
        let mut pstate = domsep.to_prover_state();
        let input = [42, 43, 44];

        assert!(pstate.add_bytes(&input).is_ok());
        assert_eq!(pstate.narg_string(), &input);
    }

    #[test]
    fn test_add_units_too_many_elements_should_error() {
        let mut domsep = DomainSeparator::<F, F, DefaultHash>::new("test");
        domsep.absorb(2, "short");
        let mut pstate = domsep.to_prover_state();

        let result = pstate.add_bytes(&[1, 2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_public_units_does_not_update_transcript() {
        let mut domsep = DomainSeparator::<F, F, DefaultHash>::new("test");
        domsep.absorb(2, "p");
        let mut pstate = domsep.to_prover_state();
        let _ = pstate.public_bytes(&[0xaa, 0xbb]);

        assert_eq!(pstate.narg_string(), b"");
    }

    #[test]
    fn test_fill_challenge_units() {
        let mut domsep = DomainSeparator::<F, F, DefaultHash>::new("test");
        domsep.squeeze(8, "ch");
        let mut pstate = domsep.to_prover_state();

        let mut out = [0u8; 8];
        let _ = pstate.fill_challenge_bytes(&mut out);
        assert_eq!(out, [77, 249, 17, 180, 176, 109, 121, 62]);
    }

    #[test]
    fn test_add_units_multiple_accumulates() {
        let mut domsep = DomainSeparator::<F, F, DefaultHash>::new("t");
        domsep.absorb(2, "a");
        domsep.absorb(3, "b");
        let mut p = domsep.to_prover_state();

        p.add_bytes(&[10, 11]).unwrap();
        p.add_bytes(&[20, 21, 22]).unwrap();

        assert_eq!(p.narg_string(), &[10, 11, 20, 21, 22]);
    }

    #[test]
    fn test_narg_string_round_trip_check() {
        let mut domsep = DomainSeparator::<F, F, DefaultHash>::new("t");
        domsep.absorb(5, "data");
        let mut p = domsep.to_prover_state();

        let msg = b"zkp42";
        p.add_bytes(msg).unwrap();

        let encoded = p.narg_string();
        assert_eq!(encoded, msg);
    }

    #[test]
    fn test_add_scalars_babybear() {
        // Step 1: Create a domain separator with the label "test"
        let mut domsep: DomainSeparator<F, F, H> = DomainSeparator::new("test");

        // Step 2: Add an "absorb scalars" tag for 3 scalars, with label "com"
        // This ensures deterministic transcript layout
        domsep.add_scalars(3, "com");

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
        let mut domsep: DomainSeparator<G, G, H> = DomainSeparator::new("test");

        // Step 2: Add an "absorb scalars" tag for 3 scalars, with label "com"
        // This ensures deterministic transcript layout
        domsep.add_scalars(3, "com");

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
        let mut domsep: DomainSeparator<EF4, F, H> = DomainSeparator::new("test");

        // Step 2: Add absorb-scalar tag for EF4 type and 3 values
        domsep.add_scalars(3, "com");

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
        let mut domsep: DomainSeparator<EG2, G, H> = DomainSeparator::new("test");

        // Step 2: Add absorb-scalar tag for EG2 type and 3 values
        domsep.add_scalars(3, "com");

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

    #[test]
    fn scalar_challenge_single_basefield_case_1() {
        // Generate a domain separator with known tag and one challenge scalar
        let mut domsep: DomainSeparator<F, F, H> = DomainSeparator::new("chal");
        domsep.challenge_scalars(1, "tag");
        let mut prover = domsep.to_prover_state();

        // Sample a single scalar
        let mut out = [F::ZERO; 1];
        prover.fill_challenge_scalars(&mut out).unwrap();

        // Expected value checked with reference implementation
        assert_eq!(out, [F::from_u64(941695054)]);
    }

    #[test]
    fn scalar_challenge_single_basefield_case_2() {
        let mut domsep: DomainSeparator<F, F, H> = DomainSeparator::new("chal2");
        domsep.challenge_scalars(1, "tag");
        let mut prover = domsep.to_prover_state();

        let mut out = [F::ZERO; 1];
        prover.fill_challenge_scalars(&mut out).unwrap();

        assert_eq!(out, [F::from_u64(1368000007)]);
    }

    #[test]
    fn scalar_challenge_multiple_basefield_scalars() {
        let mut domsep: DomainSeparator<F, F, H> = DomainSeparator::new("chal");
        domsep.challenge_scalars(10, "tag");
        let mut prover = domsep.to_prover_state();

        let mut out = [F::ZERO; 10];
        prover.fill_challenge_scalars(&mut out).unwrap();

        assert_eq!(
            out,
            [
                F::from_u64(1339394730),
                F::from_u64(299253387),
                F::from_u64(639309475),
                F::from_u64(291978),
                F::from_u64(693273190),
                F::from_u64(79969777),
                F::from_u64(1282539175),
                F::from_u64(1950046278),
                F::from_u64(1245120766),
                F::from_u64(1108619098)
            ]
        );
    }

    #[test]
    fn scalar_challenge_single_extension_scalar() {
        let mut domsep: DomainSeparator<EF4, F, H> = DomainSeparator::new("chal");
        domsep.challenge_scalars(1, "tag");
        let mut prover = domsep.to_prover_state();

        let mut out = [EF4::ZERO; 1];
        prover.fill_challenge_scalars(&mut out).unwrap();

        let expected = EF4::from_basis_coefficients_iter(
            [
                BabyBear::new(766723793),
                BabyBear::new(142148826),
                BabyBear::new(1747592655),
                BabyBear::new(1079604003),
            ]
            .into_iter(),
        );

        assert_eq!(out, [expected]);
    }

    #[test]
    fn scalar_challenge_multiple_extension_scalars() {
        let mut domsep: DomainSeparator<EF4, F, H> = DomainSeparator::new("chal");
        domsep.challenge_scalars(5, "tag");
        let mut prover = domsep.to_prover_state();

        let mut out = [EF4::ZERO; 5];
        prover.fill_challenge_scalars(&mut out).unwrap();

        let ef0 = EF4::from_basis_coefficients_iter(
            [
                BabyBear::new(221219480),
                BabyBear::new(1982332342),
                BabyBear::new(625475973),
                BabyBear::new(421782538),
            ]
            .into_iter(),
        );

        let ef1 = EF4::from_basis_coefficients_iter(
            [
                BabyBear::new(1967478349),
                BabyBear::new(966593806),
                BabyBear::new(1839663095),
                BabyBear::new(878608238),
            ]
            .into_iter(),
        );

        let ef2 = EF4::from_basis_coefficients_iter(
            [
                BabyBear::new(1330039744),
                BabyBear::new(410562161),
                BabyBear::new(825994336),
                BabyBear::new(1112934023),
            ]
            .into_iter(),
        );

        let ef3 = EF4::from_basis_coefficients_iter(
            [
                BabyBear::new(111882429),
                BabyBear::new(1246071646),
                BabyBear::new(1718768295),
                BabyBear::new(1127778746),
            ]
            .into_iter(),
        );

        let ef4 = EF4::from_basis_coefficients_iter(
            [
                BabyBear::new(1533982496),
                BabyBear::new(1606406037),
                BabyBear::new(1075981915),
                BabyBear::new(1199951082),
            ]
            .into_iter(),
        );

        // Result obtained via a script to double check the result
        assert_eq!(out, [ef0, ef1, ef2, ef3, ef4]);
    }

    #[test]
    fn test_common_field_to_unit_bytes_babybear() {
        // Generate some random BabyBear values
        let values = [BabyBear::from_u64(111), BabyBear::from_u64(222)];

        // Create a domain separator indicating we will absorb 2 public scalars
        let mut domsep: DomainSeparator<F, F, H> = DomainSeparator::new("field");
        domsep.add_scalars(2, "test");

        // Create prover and serialize expected values manually
        let expected_bytes = [111, 0, 0, 0, 222, 0, 0, 0];

        let mut prover = domsep.to_prover_state();
        let actual = prover.public_scalars(&values).unwrap();

        assert_eq!(
            actual, expected_bytes,
            "Public scalars should serialize to expected bytes"
        );

        // Determinism: same input, same transcript = same output
        let mut prover2 = domsep.to_prover_state();
        let actual2 = prover2.public_scalars(&values).unwrap();

        assert_eq!(
            actual, actual2,
            "Transcript serialization should be deterministic"
        );
    }

    #[test]
    fn test_common_field_to_unit_bytes_goldilocks() {
        // Generate some random Goldilocks values
        let values = [G::from_u64(111), G::from_u64(222)];

        // Create a domain separator indicating we will absorb 2 public scalars
        let mut domsep: DomainSeparator<G, G, H> = DomainSeparator::new("field");
        domsep.add_scalars(2, "test");

        // Create prover and serialize expected values manually
        let expected_bytes = [111, 0, 0, 0, 0, 0, 0, 0, 222, 0, 0, 0, 0, 0, 0, 0];

        let mut prover = domsep.to_prover_state();
        let actual = prover.public_scalars(&values).unwrap();

        assert_eq!(
            actual, expected_bytes,
            "Public scalars should serialize to expected bytes"
        );

        // Determinism: same input, same transcript = same output
        let mut prover2 = domsep.to_prover_state();
        let actual2 = prover2.public_scalars(&values).unwrap();

        assert_eq!(
            actual, actual2,
            "Transcript serialization should be deterministic"
        );
    }

    #[test]
    fn test_common_field_to_unit_bytes_babybear_extension() {
        // Construct two extension field elements using known u64 inputs
        let values = [EF4::from_u64(111), EF4::from_u64(222)];

        // Create a domain separator committing to 2 public scalars
        let mut domsep: DomainSeparator<EF4, F, H> = DomainSeparator::new("field");
        domsep.add_scalars(2, "test");

        // Compute expected bytes manually: serialize each coefficient of EF4
        let expected_bytes = [
            111, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 222, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0,
        ];

        // Serialize the values through the transcript
        let mut prover = domsep.to_prover_state();
        let actual = prover.public_scalars(&values).unwrap();

        // Check that the actual bytes match expected ones
        assert_eq!(
            actual, expected_bytes,
            "Public scalars should serialize to expected bytes"
        );

        // Check determinism: same input = same output
        let mut prover2 = domsep.to_prover_state();
        let actual2 = prover2.public_scalars(&values).unwrap();

        assert_eq!(
            actual, actual2,
            "Transcript serialization should be deterministic"
        );
    }

    #[test]
    fn test_common_field_to_unit_bytes_goldilocks_extension() {
        // Construct two extension field elements using known u64 inputs
        let values = [EG2::from_u64(111), EG2::from_u64(222)];

        // Create a domain separator committing to 2 public scalars
        let mut domsep: DomainSeparator<EG2, G, H> = DomainSeparator::new("field");
        domsep.add_scalars(2, "test");

        // Compute expected bytes manually: serialize each coefficient of EF4
        let expected_bytes = [
            111, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 222, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0,
        ];

        // Serialize the values through the transcript
        let mut prover = domsep.to_prover_state();
        let actual = prover.public_scalars(&values).unwrap();

        // Check that the actual bytes match expected ones
        assert_eq!(
            actual, expected_bytes,
            "Public scalars should serialize to expected bytes"
        );

        // Check determinism: same input = same output
        let mut prover2 = domsep.to_prover_state();
        let actual2 = prover2.public_scalars(&values).unwrap();

        assert_eq!(
            actual, actual2,
            "Transcript serialization should be deterministic"
        );
    }

    #[test]
    fn test_common_field_to_unit_mixed_values() {
        let values = [
            BabyBear::ZERO,
            BabyBear::ONE,
            BabyBear::from_u64(123456),
            BabyBear::from_u64(7891011),
        ];

        let mut domsep: DomainSeparator<F, F, H> = DomainSeparator::new("mixed");
        domsep.add_scalars(values.len(), "mix");

        let mut prover = domsep.to_prover_state();
        let actual = prover.public_scalars(&values).unwrap();

        let expected = vec![0, 0, 0, 0, 1, 0, 0, 0, 64, 226, 1, 0, 67, 104, 120, 0];

        assert_eq!(actual, expected, "Mixed values should serialize correctly");

        let mut prover2 = domsep.to_prover_state();
        assert_eq!(
            actual,
            prover2.public_scalars(&values).unwrap(),
            "Serialization must be deterministic"
        );
    }
}
