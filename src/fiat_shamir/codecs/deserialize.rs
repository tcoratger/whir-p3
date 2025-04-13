use p3_field::{BasedVectorSpace, Field, PrimeField64};

use super::{traits::FieldToUnitDeserialize, utils::bytes_modp};
use crate::fiat_shamir::{
    codecs::utils::from_le_bytes_mod_order, duplex_sponge::interface::DuplexSpongeInterface,
    errors::ProofResult, verifier::VerifierState,
};

impl<F, H> FieldToUnitDeserialize<F> for VerifierState<'_, H>
where
    F: Field + BasedVectorSpace<F::PrimeSubfield>,
    F::PrimeSubfield: PrimeField64,
    H: DuplexSpongeInterface,
{
    fn fill_next_scalars(&mut self, output: &mut [F]) -> ProofResult<()> {
        // Size of one base field element in bytes
        let base_bytes = bytes_modp(F::PrimeSubfield::bits() as u32);

        // Number of coefficients (1 for base field, >1 for extension field)
        let ext_degree = F::DIMENSION * F::PrimeSubfield::DIMENSION;

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
            *out = F::from_basis_coefficients_iter(coeffs);
        }

        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};

    use super::*;
    use crate::fiat_shamir::{DefaultHash, domain_separator::DomainSeparator};

    type H = DefaultHash;

    /// Base field: BabyBear
    type F = BabyBear;

    /// Extension field: EF4 = BabyBear^4
    type EF4 = BinomialExtensionField<F, 4>;

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
        let mut domsep: DomainSeparator<H> = DomainSeparator::new("test");
        domsep.add_scalars::<F>(values.len(), "scalars");

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
        let mut domsep: DomainSeparator<H> = DomainSeparator::new("ext");
        domsep.add_scalars::<EF4>(values.len(), "ext-scalars");

        // Step 5: Construct a verifier state from the domain separator and raw byte input
        let mut verifier = domsep.to_verifier_state(&raw_bytes);

        // Step 6: Allocate an output array and deserialize into it from the verifier
        let mut out = [EF4::ZERO; 2];
        verifier.fill_next_scalars(&mut out).unwrap();

        // Step 7: Ensure the decoded extension field elements match the original values
        assert_eq!(out, values);
    }
}
