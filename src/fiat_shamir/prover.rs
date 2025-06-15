use std::{fmt::Debug, marker::PhantomData};

use p3_challenger::{CanObserve, CanSample};
use p3_field::{ExtensionField, Field, PrimeField64, TwoAdicField};
use p3_symmetric::Hash;
use serde::Serialize;

use super::{
    domain_separator::DomainSeparator,
    errors::{DomainSeparatorMismatch, ProofError, ProofResult},
    pow::traits::PowStrategy,
    sho::HashStateWithInstructions,
    unit::UnitToBytes,
    utils::{bytes_uniform_modp, from_be_bytes_mod_order},
};
use crate::fiat_shamir::unit::Unit;

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
pub struct ProverState<EF, F, Challenger, U>
where
    U: Unit,
    Challenger: CanObserve<U> + CanSample<U>,
{
    /// The internal challenger.
    pub(crate) challenger: Challenger,
    /// The public coins for the protocol
    pub(crate) hash_state: HashStateWithInstructions<Challenger, U>,
    /// The encoded data.
    pub(crate) narg_string: Vec<u8>,
    /// Marker for the field.
    _field: PhantomData<F>,
    /// Marker for the extension field.
    _extension_field: PhantomData<EF>,
}

impl<EF, F, Challenger, U> ProverState<EF, F, Challenger, U>
where
    U: Unit + Default + Copy,
    Challenger: CanObserve<U> + CanSample<U>,
    EF: ExtensionField<F> + TwoAdicField,
    F: PrimeField64 + TwoAdicField,
{
    /// Initialize a new `ProverState` from the given domain separator.
    ///
    /// Seeds the internal sponge with the domain separator.
    #[must_use]
    pub fn new(domain_separator: &DomainSeparator<EF, F, U>, mut challenger: Challenger) -> Self
    where
        Challenger: Clone,
    {
        let hash_state = HashStateWithInstructions::new(domain_separator, challenger.clone());

        challenger.observe_slice(&domain_separator.as_units());

        Self {
            challenger,
            hash_state,
            narg_string: Vec::new(),
            _field: PhantomData,
            _extension_field: PhantomData,
        }
    }

    /// Add a slice `[U]` to the protocol transcript.
    /// The messages are also internally encoded in the protocol transcript,
    /// and used to re-seed the prover's random number generator.
    pub fn add_units(&mut self, input: &[U]) -> Result<(), DomainSeparatorMismatch> {
        self.hash_state.absorb(input)?;
        U::write(input, &mut self.narg_string).unwrap();
        self.challenger.observe_slice(input);
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

    /// Absorb a sequence of extension field scalars into the prover transcript.
    ///
    /// Serializes the scalars to bytes and appends them to the internal buffer.
    pub fn add_scalars(&mut self, input: &[EF]) -> ProofResult<()> {
        // Serialize the input scalars to bytes
        let serialized = self.public_scalars(input)?;

        // Append the serialized bytes to the internal transcript byte buffer
        self.narg_string.extend(serialized);

        // Return success
        Ok(())
    }

    /// Serialize public extension field scalars to bytes and absorb into sponge.
    ///
    /// Returns the serialized byte representation.
    pub fn public_scalars(&mut self, input: &[EF]) -> ProofResult<Vec<u8>> {
        // Build the byte vector by flattening all basis coefficients.
        //
        // For each extension field element:
        // - Decompose it into its canonical basis over the base field (returns a slice of F coefficients).
        //
        // For each base field coefficient:
        // - Convert it to a canonical little-endian u64 byte array (8 bytes).
        // - Truncate the byte array to `num_bytes` (only keep the low significant part).
        // - Collect all these truncated bytes into a flat vector.
        //
        // Example:
        // - BabyBear: one limb → 4 bytes.
        // - EF4 over BabyBear: 4 limbs → 16 bytes.
        let bytes: Vec<u8> = input
            .iter()
            .flat_map(p3_field::BasedVectorSpace::as_basis_coefficients_slice)
            .flat_map(|c| c.as_canonical_u64().to_le_bytes()[..F::NUM_BYTES].to_vec())
            .collect();

        // Absorb the serialized bytes into the Fiat-Shamir transcript
        self.public_units(&U::slice_from_u8_slice(&bytes))?;

        // Return the serialized byte representation
        Ok(bytes)
    }

    /// Add public messages to the protocol transcript.
    /// Messages input to this function are not added to the protocol transcript.
    /// They are however absorbed into the verifier's sponge for Fiat-Shamir, and used to re-seed
    /// the prover state.
    pub fn public_units(&mut self, input: &[U]) -> Result<(), DomainSeparatorMismatch> {
        let len = self.narg_string.len();
        self.add_units(input)?;
        self.narg_string.truncate(len);
        Ok(())
    }

    /// Perform a Fiat-Shamir proof-of-work challenge and append nonce to the transcript.
    ///
    /// Requires specifying number of PoW bits and a strategy for solving it.
    pub fn challenge_pow<S: PowStrategy>(&mut self, bits: f64) -> ProofResult<()> {
        let challenge = self.challenge_units()?;
        let nonce = S::new(U::array_to_u8_array(&challenge), bits)
            .solve()
            .ok_or(ProofError::InvalidProof)?;
        self.add_units(&U::slice_from_u8_slice(&nonce.to_be_bytes()))?;
        Ok(())
    }

    /// Sample N fresh challenge bytes from the Fiat-Shamir sponge.
    ///
    /// Used for sampling scalar field elements or general randomness.
    pub fn challenge_units<const N: usize>(&mut self) -> Result<[U; N], DomainSeparatorMismatch> {
        let mut output = [U::default(); N];
        self.fill_challenge_units(&mut output)?;
        Ok(output)
    }

    /// Absorb a digest object (e.g. Merkle root) into the transcript.
    pub fn add_digest<const DIGEST_ELEMS: usize>(
        &mut self,
        digest: Hash<F, U, DIGEST_ELEMS>,
    ) -> ProofResult<()> {
        self.add_units(digest.as_ref())
            .map_err(ProofError::InvalidDomainSeparator)
    }

    /// Fill a mutable slice with uniformly sampled extension field elements.
    ///
    /// Each element is sampled using Fiat-Shamir from the internal sponge.
    pub fn fill_challenge_scalars(&mut self, output: &mut [EF]) -> ProofResult<()> {
        // How many bytes are needed to sample a single base field element
        let base_field_size = bytes_uniform_modp(F::bits() as u32);

        // Total bytes needed for one EF element = extension degree × base field size
        let field_unit_len = EF::DIMENSION * base_field_size;

        // Temporary buffer to hold bytes for each field element
        let mut u_buf = vec![U::default(); field_unit_len];

        // Fill each output element from fresh transcript randomness
        for o in output.iter_mut() {
            // Draw uniform bytes from the transcript
            self.fill_challenge_units(&mut u_buf)?;

            // Reinterpret as bytes (safe because U must be 1-byte width)
            let byte_buf = U::slice_to_u8_slice(&u_buf);

            // For each chunk, convert to base field element via modular reduction
            let base_coeffs = byte_buf
                .chunks(base_field_size)
                .map(from_be_bytes_mod_order);

            // Reconstruct the full field element using canonical basis
            *o = EF::from_basis_coefficients_iter(base_coeffs).unwrap();
        }

        Ok(())
    }

    /// Sample N extension field elements as Fiat-Shamir challenges.
    pub fn challenge_scalars<const N: usize>(&mut self) -> ProofResult<[EF; N]> {
        let mut output = [EF::default(); N];
        self.fill_challenge_scalars(&mut output)?;
        Ok(output)
    }

    /// Absorb a hint message into the prover transcript.
    ///
    /// Encodes the hint as a 4-byte little-endian length prefix followed by raw bytes.
    pub fn hint_bytes(&mut self, hint: &[u8]) -> Result<(), DomainSeparatorMismatch> {
        self.hash_state.hint()?;
        let len = u32::try_from(hint.len()).expect("Hint size out of bounds");
        self.narg_string.extend_from_slice(&len.to_le_bytes());
        self.narg_string.extend_from_slice(hint);
        Ok(())
    }

    /// Serialize and absorb a structured hint into the prover transcript.
    ///
    /// This is used to insert auxiliary (non-binding) data into the proof transcript,
    /// such as evaluations or precomputed commitments. These hints are not derived from
    /// public input but are necessary for verification.
    ///
    /// The hint is encoded as:
    /// - A 4-byte little-endian length prefix (indicating the byte length of the payload),
    /// - Followed by the bincode-encoded value.
    ///
    /// # Type Parameters
    /// - `T`: Any type that implements `serde::Serialize` and `Debug`.
    ///
    /// # Errors
    /// - Returns `ProofError::SerializationError` if encoding fails.
    /// - Returns `DomainSeparatorMismatch` if no `.hint("label")` instruction was registered.
    pub fn hint<T: Serialize>(&mut self, hint: &T) -> ProofResult<()> {
        // Serialize the input object to a byte vector using bincode.
        // This encodes the object in a compact, deterministic binary format.
        let bytes = bincode::serde::encode_to_vec(hint, bincode::config::standard())
            .map_err(|_| ProofError::SerializationError)?;

        // Register the hint with the internal domain separator and append it
        // to the `narg_string`. This checks that:
        // - the domain separator allows a hint here,
        // - writes `[len, ...bytes]` into the transcript.
        self.hint_bytes(&bytes)?;

        // Successfully written.
        Ok(())
    }
}

impl<EF, F, MyChallenger, U> UnitToBytes<U> for ProverState<EF, F, MyChallenger, U>
where
    U: Unit + Default + Copy,
    MyChallenger: CanObserve<U> + CanSample<U>,
    EF: ExtensionField<F>,
    F: Field,
{
    fn fill_challenge_units(&mut self, output: &mut [U]) -> Result<(), DomainSeparatorMismatch> {
        self.hash_state.squeeze(output)
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_challenger::HashChallenger;
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};
    use p3_goldilocks::Goldilocks;
    use p3_keccak::Keccak256Hash;

    use super::*;

    type F = BabyBear;
    type EF4 = BinomialExtensionField<F, 4>;

    type G = Goldilocks;
    type EG2 = BinomialExtensionField<G, 2>;

    type MyChallenger = HashChallenger<u8, Keccak256Hash, 32>;

    #[test]
    fn test_public_units_does_not_affect_narg() {
        // Test with different data sizes to ensure robustness
        let test_cases = [
            (2, "small", vec![0xaa, 0xbb]),
            (4, "medium", vec![1, 2, 3, 4]),
        ];

        for (size, label, data) in test_cases {
            let mut domsep = DomainSeparator::<F, F, u8>::new("test");
            domsep.absorb(size, label);

            let challenger = MyChallenger::new(vec![], Keccak256Hash);
            let mut pstate = domsep.to_prover_state(challenger);

            pstate.public_units(&data).unwrap();
            assert_eq!(pstate.narg_string(), b"", "Failed for case: {}", label);
        }
    }

    #[test]
    fn test_add_units_appends_to_narg_string() {
        let mut domsep = DomainSeparator::<F, F, u8>::new("test");
        domsep.absorb(3, "msg");

        let challenger = MyChallenger::new(vec![], Keccak256Hash);
        let mut pstate = domsep.to_prover_state(challenger);
        let input = [42, 43, 44];

        assert!(pstate.add_units(&input).is_ok());
        assert_eq!(pstate.narg_string(), &input);
    }

    #[test]
    fn test_add_units_too_many_elements_should_error() {
        let mut domsep = DomainSeparator::<F, F, u8>::new("test");
        domsep.absorb(2, "short");

        let challenger = MyChallenger::new(vec![], Keccak256Hash);
        let mut pstate = domsep.to_prover_state(challenger);

        let result = pstate.add_units(&[1, 2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_add_units_multiple_accumulates() {
        let mut domsep = DomainSeparator::<F, F, u8>::new("t");
        domsep.absorb(2, "a");
        domsep.absorb(3, "b");

        let challenger = MyChallenger::new(vec![], Keccak256Hash);
        let mut p = domsep.to_prover_state(challenger);

        p.add_units(&[10, 11]).unwrap();
        p.add_units(&[20, 21, 22]).unwrap();

        assert_eq!(p.narg_string(), &[10, 11, 20, 21, 22]);
    }

    #[test]
    fn test_narg_string_round_trip_check() {
        let mut domsep = DomainSeparator::<F, F, u8>::new("t");
        domsep.absorb(5, "data");

        let challenger = MyChallenger::new(vec![], Keccak256Hash);
        let mut p = domsep.to_prover_state(challenger);

        let msg = b"zkp42";
        p.add_units(msg).unwrap();

        let encoded = p.narg_string();
        assert_eq!(encoded, msg);
    }

    #[test]
    fn test_add_scalars_babybear() {
        // Step 1: Create a domain separator with the label "test"
        let mut domsep: DomainSeparator<F, F, u8> = DomainSeparator::new("test");

        // Step 2: Add an "absorb scalars" tag for 3 scalars, with label "com"
        // This ensures deterministic transcript layout
        domsep.add_scalars(3, "com");

        // Step 3: Initialize the prover state from the domain separator
        let challenger = MyChallenger::new(vec![], Keccak256Hash);
        let mut prover_state = domsep.to_prover_state(challenger);

        // Step 4: Create 3 F field elements
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
        let challenger = MyChallenger::new(vec![], Keccak256Hash);
        let mut prover_state2 = domsep.to_prover_state(challenger);
        prover_state2.add_scalars(&[f0, f1, f2]).unwrap();

        assert_eq!(
            prover_state.narg_string, prover_state2.narg_string,
            "Transcript encoding should be deterministic for same inputs"
        );
    }

    #[test]
    fn test_add_scalars_goldilocks() {
        // Step 1: Create a domain separator with the label "test"
        let mut domsep: DomainSeparator<G, G, u8> = DomainSeparator::new("test");

        // Step 2: Add an "absorb scalars" tag for 3 scalars, with label "com"
        // This ensures deterministic transcript layout
        domsep.add_scalars(3, "com");

        // Step 3: Initialize the prover state from the domain separator
        let challenger = MyChallenger::new(vec![], Keccak256Hash);
        let mut prover_state = domsep.to_prover_state(challenger);

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
        let challenger = MyChallenger::new(vec![], Keccak256Hash);
        let mut prover_state2 = domsep.to_prover_state(challenger);
        prover_state2.add_scalars(&[f0, f1, f2]).unwrap();

        assert_eq!(
            prover_state.narg_string, prover_state2.narg_string,
            "Transcript encoding should be deterministic for same inputs"
        );
    }

    #[test]
    fn test_add_scalars_extension_babybear() {
        // Step 1: Create a domain separator with the label "test"
        let mut domsep: DomainSeparator<EF4, F, u8> = DomainSeparator::new("test");

        // Step 2: Add absorb-scalar tag for EF4 type and 3 values
        domsep.add_scalars(3, "com");

        // Step 3: Initialize the prover state from the domain separator
        let challenger = MyChallenger::new(vec![], Keccak256Hash);
        let mut prover_state = domsep.to_prover_state(challenger);

        // Step 4: Construct 3 extension field elements
        // - One large (MAX) value to ensure all 4 limbs are filled
        // - Two small values (fit in low limb only)
        let f0 = EF4::from_u64(u64::MAX);
        let f1 = EF4::from_u64(222);
        let f2 = EF4::from_u64(333);

        // Step 5: Add the scalars to the transcript
        prover_state.add_scalars(&[f0, f1, f2]).unwrap();

        // Step 6: Expected bytes from 3 extension field elements, each with 4 limbs
        // - 4 * 3 = 12 F limbs = 12 * 4 = 48 bytes
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
        let challenger = MyChallenger::new(vec![], Keccak256Hash);
        let mut prover_state2 = domsep.to_prover_state(challenger);
        prover_state2.add_scalars(&[f0, f1, f2]).unwrap();

        assert_eq!(
            prover_state.narg_string, prover_state2.narg_string,
            "Transcript encoding should be deterministic for same inputs"
        );
    }

    #[test]
    fn test_add_scalars_extension_goldilocks() {
        // Step 1: Create a domain separator with the label "test"
        let mut domsep: DomainSeparator<EG2, G, u8> = DomainSeparator::new("test");

        // Step 2: Add absorb-scalar tag for EG2 type and 3 values
        domsep.add_scalars(3, "com");

        // Step 3: Initialize the prover state from the domain separator
        let challenger = MyChallenger::new(vec![], Keccak256Hash);
        let mut prover_state = domsep.to_prover_state(challenger);

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
        let challenger = MyChallenger::new(vec![], Keccak256Hash);
        let mut prover_state2 = domsep.to_prover_state(challenger);
        prover_state2.add_scalars(&[f0, f1, f2]).unwrap();

        assert_eq!(
            prover_state.narg_string, prover_state2.narg_string,
            "Transcript encoding should be deterministic for same inputs"
        );
    }

    #[test]
    fn test_common_field_to_unit_bytes_babybear() {
        // Generate some random F values
        let values = [F::from_u64(111), F::from_u64(222)];

        // Create a domain separator indicating we will absorb 2 public scalars
        let mut domsep: DomainSeparator<F, F, u8> = DomainSeparator::new("field");
        domsep.add_scalars(2, "test");

        // Create prover and serialize expected values manually
        let expected_bytes = [111, 0, 0, 0, 222, 0, 0, 0];

        let challenger = MyChallenger::new(vec![], Keccak256Hash);
        let mut prover = domsep.to_prover_state(challenger);
        let actual = prover.public_scalars(&values).unwrap();

        assert_eq!(
            actual, expected_bytes,
            "Public scalars should serialize to expected bytes"
        );

        // Determinism: same input, same transcript = same output
        let challenger = MyChallenger::new(vec![], Keccak256Hash);
        let mut prover2 = domsep.to_prover_state(challenger);
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
        let mut domsep: DomainSeparator<G, G, u8> = DomainSeparator::new("field");
        domsep.add_scalars(2, "test");

        // Create prover and serialize expected values manually
        let expected_bytes = [111, 0, 0, 0, 0, 0, 0, 0, 222, 0, 0, 0, 0, 0, 0, 0];

        let challenger = MyChallenger::new(vec![], Keccak256Hash);
        let mut prover = domsep.to_prover_state(challenger);
        let actual = prover.public_scalars(&values).unwrap();

        assert_eq!(
            actual, expected_bytes,
            "Public scalars should serialize to expected bytes"
        );

        // Determinism: same input, same transcript = same output
        let challenger = MyChallenger::new(vec![], Keccak256Hash);
        let mut prover2 = domsep.to_prover_state(challenger);
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
        let mut domsep: DomainSeparator<EF4, F, u8> = DomainSeparator::new("field");
        domsep.add_scalars(2, "test");

        // Compute expected bytes manually: serialize each coefficient of EF4
        let expected_bytes = [
            111, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 222, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0,
        ];

        // Serialize the values through the transcript
        let challenger = MyChallenger::new(vec![], Keccak256Hash);
        let mut prover = domsep.to_prover_state(challenger);
        let actual = prover.public_scalars(&values).unwrap();

        // Check that the actual bytes match expected ones
        assert_eq!(
            actual, expected_bytes,
            "Public scalars should serialize to expected bytes"
        );

        // Check determinism: same input = same output
        let challenger = MyChallenger::new(vec![], Keccak256Hash);
        let mut prover2 = domsep.to_prover_state(challenger);
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
        let mut domsep: DomainSeparator<EG2, G, u8> = DomainSeparator::new("field");
        domsep.add_scalars(2, "test");

        // Compute expected bytes manually: serialize each coefficient of EF4
        let expected_bytes = [
            111, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 222, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0,
        ];

        // Serialize the values through the transcript
        let challenger = MyChallenger::new(vec![], Keccak256Hash);
        let mut prover = domsep.to_prover_state(challenger);
        let actual = prover.public_scalars(&values).unwrap();

        // Check that the actual bytes match expected ones
        assert_eq!(
            actual, expected_bytes,
            "Public scalars should serialize to expected bytes"
        );

        // Check determinism: same input = same output
        let challenger = MyChallenger::new(vec![], Keccak256Hash);
        let mut prover2 = domsep.to_prover_state(challenger);
        let actual2 = prover2.public_scalars(&values).unwrap();

        assert_eq!(
            actual, actual2,
            "Transcript serialization should be deterministic"
        );
    }

    #[test]
    fn test_common_field_to_unit_mixed_values() {
        let values = [F::ZERO, F::ONE, F::from_u64(123456), F::from_u64(7891011)];

        let mut domsep: DomainSeparator<F, F, u8> = DomainSeparator::new("mixed");
        domsep.add_scalars(values.len(), "mix");

        let challenger = MyChallenger::new(vec![], Keccak256Hash);
        let mut prover = domsep.to_prover_state(challenger);
        let actual = prover.public_scalars(&values).unwrap();

        let expected = vec![0, 0, 0, 0, 1, 0, 0, 0, 64, 226, 1, 0, 67, 104, 120, 0];

        assert_eq!(actual, expected, "Mixed values should serialize correctly");

        let challenger = MyChallenger::new(vec![], Keccak256Hash);
        let mut prover2 = domsep.to_prover_state(challenger);
        assert_eq!(
            actual,
            prover2.public_scalars(&values).unwrap(),
            "Serialization must be deterministic"
        );
    }

    #[test]
    fn test_hint_bytes_appends_hint_length_and_data() {
        let mut domsep: DomainSeparator<F, F, u8> = DomainSeparator::new("hint_test");
        domsep.hint("proof_hint");

        let challenger = MyChallenger::new(vec![], Keccak256Hash);
        let mut prover = domsep.to_prover_state(challenger);

        let hint = b"abc123";
        prover.hint_bytes(hint).unwrap();

        // Explanation:
        // - `hint` is "abc123", which has 6 bytes.
        // - The protocol encodes this as a 4-byte *little-endian* length prefix: 6 = 0x00000006 → [6, 0, 0, 0]
        // - Then it appends the hint bytes: b"abc123"
        // - So the full expected value is:
        let expected = [6, 0, 0, 0, b'a', b'b', b'c', b'1', b'2', b'3'];

        assert_eq!(prover.narg_string(), &expected);
    }

    #[test]
    fn test_hint_bytes_empty_hint_is_encoded_correctly() {
        let mut domsep: DomainSeparator<F, F, u8> = DomainSeparator::new("empty_hint");
        domsep.hint("empty");

        let challenger = MyChallenger::new(vec![], Keccak256Hash);
        let mut prover = domsep.to_prover_state(challenger);

        prover.hint_bytes(b"").unwrap();

        // Length = 0 encoded as 4 zero bytes
        assert_eq!(prover.narg_string(), &[0, 0, 0, 0]);
    }

    #[test]
    fn test_hint_bytes_fails_if_hint_op_missing() {
        let domsep: DomainSeparator<F, F, u8> = DomainSeparator::new("no_hint");

        let challenger = MyChallenger::new(vec![], Keccak256Hash);
        let mut prover = domsep.to_prover_state(challenger);

        // DomainSeparator contains no hint operation
        let result = prover.hint_bytes(b"some_hint");
        assert!(
            result.is_err(),
            "Should error if no hint op in domain separator"
        );
    }

    #[test]
    fn test_hint_bytes_is_deterministic() {
        let mut domsep: DomainSeparator<F, F, u8> = DomainSeparator::new("det_hint");
        domsep.hint("same");

        let hint = b"zkproof_hint";
        let challenger = MyChallenger::new(vec![], Keccak256Hash);
        let mut prover1 = domsep.to_prover_state(challenger.clone());
        let mut prover2 = domsep.to_prover_state(challenger);

        prover1.hint_bytes(hint).unwrap();
        prover2.hint_bytes(hint).unwrap();

        assert_eq!(
            prover1.narg_string(),
            prover2.narg_string(),
            "Encoding should be deterministic"
        );
    }

    #[test]
    fn test_hint_multiple_sequential() {
        let mut domsep: DomainSeparator<F, F, u8> = DomainSeparator::new("multi_hint");
        domsep.hint("hint1");
        domsep.hint("hint2");
        domsep.hint("hint3");

        let challenger = MyChallenger::new(vec![], Keccak256Hash);
        let mut prover = domsep.to_prover_state(challenger);

        let hint1 = F::from_u64(111);
        let hint2 = F::from_u64(222);
        let hint3 = F::from_u64(333);

        prover.hint(&hint1).unwrap();
        prover.hint(&hint2).unwrap();
        prover.hint(&hint3).unwrap();

        let transcript = prover.narg_string();
        assert!(
            transcript.len() > 12,
            "Should contain multiple encoded hints"
        );
    }

    #[test]
    fn test_challenge_pow() {
        use crate::fiat_shamir::pow::blake3::Blake3PoW;

        let mut domsep: DomainSeparator<F, F, u8> = DomainSeparator::new("pow_test");
        domsep.challenge_pow("challenge");

        let challenger = MyChallenger::new(vec![], Keccak256Hash);
        let mut prover = domsep.to_prover_state(challenger);

        let result = prover.challenge_pow::<Blake3PoW>(8.0);
        assert!(result.is_ok(), "Low difficulty PoW should succeed");

        assert_eq!(
            prover.narg_string().len(),
            8,
            "Nonce should be 8 bytes in transcript"
        );

        let transcript = prover.narg_string();
        assert!(
            !transcript.iter().all(|&b| b == 0),
            "Nonce should not be all zeros"
        );
    }
}
