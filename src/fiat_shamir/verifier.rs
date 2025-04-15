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
    pow::traits::PowStrategy,
    sho::HashStateWithInstructions,
    utils::{bytes_modp, bytes_uniform_modp, from_be_bytes_mod_order, from_le_bytes_mod_order},
};

/// [`VerifierState`] is the verifier state.
///
/// Internally, it simply contains a stateful hash.
/// Given as input an [`DomainSeparator`] and a NARG string, it allows to
/// de-serialize elements from the NARG string and make them available to the zero-knowledge
/// verifier.
#[derive(Debug)]
pub struct VerifierState<'a, EF, F, H = DefaultHash>
where
    H: DuplexSpongeInterface<u8>,
    EF: ExtensionField<F>
        + TwoAdicField<PrimeSubfield = F>
        + ExtensionField<<EF as PrimeCharacteristicRing>::PrimeSubfield>,
    EF::PrimeSubfield: PrimeField64,
    F: Field + TwoAdicField,
{
    /// Internal sponge transcript that tracks the domain separator state and absorbs values.
    ///
    /// This manages the full Fiat-Shamir interaction logic, such as absorbing inputs and
    /// squeezing challenges. It also stores the domain separator instructions to enforce
    /// consistency between prover and verifier.
    pub(crate) hash_state: HashStateWithInstructions<H>,

    /// The "NARG" string: raw serialized input provided by the prover.
    ///
    /// This byte slice contains encoded values (scalars, digests, etc.) that are deserialized
    /// during verification. Each call to `next_bytes`, `fill_next_scalars`, etc., reads from this.
    pub(crate) narg_string: &'a [u8],

    /// Marker for the base field `F`.
    ///
    /// This field is never read or written; it ensures type correctness for field-level operations.
    _field: PhantomData<F>,

    /// Marker for the extension field `EF`.
    ///
    /// Like `_field`, this is only for type-level bookkeeping. The extension field is used
    /// to deserialize and operate on scalars in high-dimensional protocols.
    _extension_field: PhantomData<EF>,
}

impl<'a, EF, F, H> VerifierState<'a, EF, F, H>
where
    H: DuplexSpongeInterface<u8>,
    EF: ExtensionField<F>
        + TwoAdicField<PrimeSubfield = F>
        + ExtensionField<<EF as PrimeCharacteristicRing>::PrimeSubfield>,
    EF::PrimeSubfield: PrimeField64,
    F: Field + TwoAdicField,
{
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
    pub fn new(domain_separator: &DomainSeparator<EF, F, H>, narg_string: &'a [u8]) -> Self {
        let hash_state = HashStateWithInstructions::new(domain_separator);
        Self {
            hash_state,
            narg_string,
            _field: PhantomData,
            _extension_field: PhantomData,
        }
    }

    /// Read `input.len()` elements from the NARG string.
    #[inline]
    pub fn fill_next_bytes(&mut self, input: &mut [u8]) -> Result<(), DomainSeparatorMismatch> {
        u8::read(&mut self.narg_string, input)?;
        self.hash_state.absorb(input)?;
        Ok(())
    }

    pub fn next_bytes<const N: usize>(&mut self) -> Result<[u8; N], DomainSeparatorMismatch> {
        let mut input = [0u8; N];
        self.fill_next_bytes(&mut input)?;
        Ok(input)
    }

    /// Add native elements to the sponge without writing them to the NARG string.
    #[inline]
    pub fn public_units(&mut self, input: &[u8]) -> Result<(), DomainSeparatorMismatch> {
        self.hash_state.absorb(input)
    }

    pub fn fill_next_scalars(&mut self, output: &mut [EF]) -> ProofResult<()> {
        // Size of one base field element in bytes
        let base_bytes = bytes_modp(EF::PrimeSubfield::bits() as u32);

        // Number of coefficients (1 for base field, >1 for extension field)
        let ext_degree = EF::DIMENSION * EF::PrimeSubfield::DIMENSION;

        // Size of full F element = D * base field size
        let scalar_size = ext_degree * base_bytes;

        // Temporary buffer to deserialize each F element
        let mut buf = vec![0u8; scalar_size];

        for out in output.iter_mut() {
            // Fetch the next group of bytes from the transcript
            self.fill_next_bytes(&mut buf)?;

            // Interpret each chunk as a base field coefficient
            let coeffs = buf.chunks(base_bytes).map(from_le_bytes_mod_order);

            // Reconstruct the field element from its base field coefficients
            *out = EF::from_basis_coefficients_iter(coeffs);
        }

        Ok(())
    }

    pub fn next_scalars<const N: usize>(&mut self) -> ProofResult<[EF; N]> {
        let mut output = [EF::default(); N];
        self.fill_next_scalars(&mut output)?;
        Ok(output)
    }

    pub fn challenge_pow<S: PowStrategy>(&mut self, bits: f64) -> ProofResult<()> {
        let challenge = self.challenge_bytes()?;
        let nonce = u64::from_be_bytes(self.next_bytes()?);
        if S::new(challenge, bits).check(nonce) {
            Ok(())
        } else {
            Err(ProofError::InvalidProof)
        }
    }

    pub fn read_digest<const DIGEST_ELEMS: usize>(
        &mut self,
    ) -> ProofResult<Hash<F, u8, DIGEST_ELEMS>> {
        let mut digest = [0u8; DIGEST_ELEMS];
        self.fill_next_bytes(&mut digest)?;
        Ok(digest.into())
    }

    pub fn challenge_bytes<const N: usize>(&mut self) -> Result<[u8; N], DomainSeparatorMismatch> {
        let mut output = [0u8; N];
        self.fill_challenge_bytes(&mut output)?;
        Ok(output)
    }

    #[inline]
    pub fn public_bytes(&mut self, input: &[u8]) -> Result<(), DomainSeparatorMismatch> {
        self.public_units(input)
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
}

impl<EF, F, H> UnitToBytes for VerifierState<'_, EF, F, H>
where
    H: DuplexSpongeInterface<u8>,
    EF: ExtensionField<F>
        + TwoAdicField<PrimeSubfield = F>
        + ExtensionField<<EF as PrimeCharacteristicRing>::PrimeSubfield>,
    EF::PrimeSubfield: PrimeField64,
    F: Field + TwoAdicField,
{
    #[inline]
    fn fill_challenge_bytes(&mut self, input: &mut [u8]) -> Result<(), DomainSeparatorMismatch> {
        self.hash_state.squeeze(input)
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use p3_baby_bear::BabyBear;
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};
    use p3_goldilocks::Goldilocks;

    use super::*;

    type H = DefaultHash;

    type F = BabyBear;
    type EF4 = BinomialExtensionField<F, 4>;

    type G = Goldilocks;
    type EG2 = BinomialExtensionField<G, 2>;

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
    }

    #[test]
    fn test_new_verifier_state_constructs_correctly() {
        let ds = DomainSeparator::<F, F, DummySponge>::new("test");
        let transcript = b"abc";
        let vs = VerifierState::<F, F, DummySponge>::new(&ds, transcript);
        assert_eq!(vs.narg_string, b"abc");
    }

    #[test]
    fn test_fill_next_units_reads_and_absorbs() {
        let mut ds = DomainSeparator::<F, F, DummySponge>::new("x");
        ds.absorb(3, "input");
        let mut vs = VerifierState::<F, F, DummySponge>::new(&ds, b"abc");
        let mut buf = [0u8; 3];
        let res = vs.fill_next_bytes(&mut buf);
        assert!(res.is_ok());
        assert_eq!(buf, *b"abc");
        assert_eq!(*vs.hash_state.ds.absorbed.borrow(), b"abc");
    }

    #[test]
    fn test_fill_next_units_with_insufficient_data_errors() {
        let mut ds = DomainSeparator::<F, F, DummySponge>::new("x");
        ds.absorb(4, "fail");
        let mut vs = VerifierState::<F, F, DummySponge>::new(&ds, b"xy");
        let mut buf = [0u8; 4];
        let res = vs.fill_next_bytes(&mut buf);
        assert!(res.is_err());
    }

    #[test]
    fn test_unit_transcript_public_units() {
        let mut ds = DomainSeparator::<F, F, DummySponge>::new("x");
        ds.absorb(2, "public");
        let mut vs = VerifierState::<F, F, DummySponge>::new(&ds, b"..");
        assert!(vs.public_units(&[1, 2]).is_ok());
        assert_eq!(*vs.hash_state.ds.absorbed.borrow(), &[1, 2]);
    }

    #[test]
    fn test_unit_transcript_fill_challenge_bytes() {
        let mut ds = DomainSeparator::<F, F, DummySponge>::new("x");
        ds.squeeze(4, "c");
        let mut vs = VerifierState::<F, F, DummySponge>::new(&ds, b"abcd");
        let mut out = [0u8; 4];
        assert!(vs.fill_challenge_bytes(&mut out).is_ok());
        assert_eq!(out, [0, 1, 2, 3]);
    }

    #[test]
    fn test_fill_next_bytes_impl() {
        let mut ds = DomainSeparator::<F, F, DummySponge>::new("x");
        ds.absorb(3, "byte");
        let mut vs = VerifierState::<F, F, DummySponge>::new(&ds, b"xyz");
        let mut out = [0u8; 3];
        assert!(vs.fill_next_bytes(&mut out).is_ok());
        assert_eq!(out, *b"xyz");
    }

    #[test]
    fn test_fill_next_scalars_babybear() {
        // Step 1: Define two known BabyBear scalars to test deserialization
        let values = [F::from_u64(123), F::from_u64(456)];

        // Step 2: Manually serialize the scalars to raw bytes in little-endian u32 format
        // This matches the encoding done in `public_scalars`

        // How many bytes are needed to sample a single base field element
        let num_bytes = F::bits().div_ceil(8);

        let mut raw_bytes = vec![];
        for x in &values {
            let bytes = x.as_canonical_u64().to_le_bytes();
            raw_bytes.extend_from_slice(&bytes[..num_bytes]);
        }

        // Step 3: Create a domain separator that commits to absorbing 2 scalars
        // The label "scalars" is just metadata to distinguish this absorb phase
        let mut domsep: DomainSeparator<F, F, H> = DomainSeparator::new("test");
        domsep.add_scalars(values.len(), "scalars");

        // Step 4: Create a verifier from the domain separator, loaded with the raw bytes
        let mut verifier = domsep.to_verifier_state(&raw_bytes);

        // Step 5: Allocate output buffer and deserialize scalars from transcript
        let mut out = [F::ZERO; 2];
        verifier.fill_next_scalars(&mut out).unwrap();

        // Step 6: Check that deserialized scalars exactly match original input
        assert_eq!(out, values);
    }

    #[test]
    fn test_fill_next_scalars_ef4() {
        // Step 1: Construct two known EF4 extension field elements from explicit basis coefficients
        // These test that field elements composed of multiple base field limbs are correctly parsed
        let ef0 = EF4::from_basis_coefficients_iter(
            [
                F::from_u64(16231546437525696111),
                F::from_u64(3260480306969229290),
                F::from_u64(16069356457323344778),
                F::from_u64(18093877879687808447),
            ]
            .into_iter(),
        );

        let ef1 = EF4::from_basis_coefficients_iter(
            [
                F::from_u64(4745602262162961622),
                F::from_u64(7823278364822041281),
                F::from_u64(2045790219489339023),
                F::from_u64(9614754510566682848),
            ]
            .into_iter(),
        );

        // Step 2: Store the known expected values into a slice
        let values = [ef0, ef1];

        // Step 3: Precomputed raw bytes matching the encoding of `public_scalars`
        // Each EF4 element has 4 BabyBear limbs, each limb serialized as 4 LE bytes
        // Total = 2 elements * 4 limbs * 4 bytes = 32 bytes
        let raw_bytes = vec![
            106, 13, 109, 83, // limb 0 of ef0
            132, 35, 135, 77, // limb 1 of ef0
            127, 148, 35, 12, // limb 2 of ef0
            40, 78, 103, 12, // limb 3 of ef0
            153, 244, 3, 21, // limb 0 of ef1
            244, 220, 153, 42, // limb 1 of ef1
            30, 27, 16, 97, // limb 2 of ef1
            224, 9, 66, 40, // limb 3 of ef1
        ];

        // Step 4: Create a domain separator for absorbing 2 EF4 values
        let mut domsep: DomainSeparator<EF4, F, H> = DomainSeparator::new("ext");
        domsep.add_scalars(values.len(), "ext-scalars");

        // Step 5: Construct a verifier state from the domain separator and raw byte input
        let mut verifier = domsep.to_verifier_state(&raw_bytes);

        // Step 6: Allocate an output array and deserialize into it from the verifier
        let mut out = [EF4::ZERO; 2];
        verifier.fill_next_scalars(&mut out).unwrap();

        // Step 7: Ensure the decoded extension field elements match the original values
        assert_eq!(out, values);
    }

    #[test]
    fn scalar_challenge_single_basefield_case_1() {
        // Generate a domain separator with known tag and one challenge scalar
        let mut domsep: DomainSeparator<F, F, H> = DomainSeparator::new("chal");
        domsep.challenge_scalars(1, "tag");
        let mut prover = domsep.to_verifier_state(&[]);

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
        let mut prover = domsep.to_verifier_state(&[]);

        let mut out = [F::ZERO; 1];
        prover.fill_challenge_scalars(&mut out).unwrap();

        assert_eq!(out, [F::from_u64(1368000007)]);
    }

    #[test]
    fn scalar_challenge_multiple_basefield_scalars() {
        let mut domsep: DomainSeparator<F, F, H> = DomainSeparator::new("chal");
        domsep.challenge_scalars(10, "tag");
        let mut prover = domsep.to_verifier_state(&[]);

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
        let mut prover = domsep.to_verifier_state(&[]);

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
        let mut prover = domsep.to_verifier_state(&[]);

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

        let mut prover = domsep.to_verifier_state(&[]);
        let actual = prover.public_scalars(&values).unwrap();

        assert_eq!(
            actual, expected_bytes,
            "Public scalars should serialize to expected bytes"
        );

        // Determinism: same input, same transcript = same output
        let mut prover2 = domsep.to_verifier_state(&[]);
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

        let mut prover = domsep.to_verifier_state(&[]);
        let actual = prover.public_scalars(&values).unwrap();

        assert_eq!(
            actual, expected_bytes,
            "Public scalars should serialize to expected bytes"
        );

        // Determinism: same input, same transcript = same output
        let mut prover2 = domsep.to_verifier_state(&[]);
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
        let mut prover = domsep.to_verifier_state(&[]);
        let actual = prover.public_scalars(&values).unwrap();

        // Check that the actual bytes match expected ones
        assert_eq!(
            actual, expected_bytes,
            "Public scalars should serialize to expected bytes"
        );

        // Check determinism: same input = same output
        let mut prover2 = domsep.to_verifier_state(&[]);
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
        let mut prover = domsep.to_verifier_state(&[]);
        let actual = prover.public_scalars(&values).unwrap();

        // Check that the actual bytes match expected ones
        assert_eq!(
            actual, expected_bytes,
            "Public scalars should serialize to expected bytes"
        );

        // Check determinism: same input = same output
        let mut prover2 = domsep.to_verifier_state(&[]);
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

        let mut domsep: DomainSeparator<BabyBear, BabyBear, H> = DomainSeparator::new("mixed");
        domsep.add_scalars(values.len(), "mix");

        let mut prover = domsep.to_verifier_state(&[]);
        let actual = prover.public_scalars(&values).unwrap();

        let expected = vec![0, 0, 0, 0, 1, 0, 0, 0, 64, 226, 1, 0, 67, 104, 120, 0];

        assert_eq!(actual, expected, "Mixed values should serialize correctly");

        let mut prover2 = domsep.to_verifier_state(&[]);
        assert_eq!(
            actual,
            prover2.public_scalars(&values).unwrap(),
            "Serialization must be deterministic"
        );
    }
}
