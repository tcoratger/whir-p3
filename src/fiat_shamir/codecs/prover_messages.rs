use p3_field::{BasedVectorSpace, Field, PrimeField64};

use super::traits::FieldToUnitSerialize;
use crate::fiat_shamir::{
    codecs::traits::CommonFieldToUnit, duplex_sponge::interface::DuplexSpongeInterface,
    errors::ProofResult, prover::ProverState,
};

impl<F, H> FieldToUnitSerialize<F> for ProverState<H>
where
    F: Field + BasedVectorSpace<F::PrimeSubfield>,
    F::PrimeSubfield: PrimeField64,
    H: DuplexSpongeInterface,
{
    fn add_scalars(&mut self, input: &[F]) -> ProofResult<()> {
        // Serialize the input scalars to bytes using the CommonFieldToUnit trait
        let serialized = self.public_scalars(input)?;

        // Append the serialized bytes to the internal transcript byte buffer
        self.narg_string.extend(serialized);

        // Return success
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};
    use p3_goldilocks::Goldilocks;

    use super::*;
    use crate::fiat_shamir::{DefaultHash, domain_separator::DomainSeparator};

    type H = DefaultHash;

    type F = BabyBear;
    type EF4 = BinomialExtensionField<F, 4>;

    type G = Goldilocks;
    type EG2 = BinomialExtensionField<G, 2>;

    #[test]
    fn test_add_scalars_babybear() {
        // Step 1: Create a domain separator with the label "test"
        let domsep: DomainSeparator<H> = DomainSeparator::new("test");

        // Step 2: Add an "absorb scalars" tag for 3 scalars, with label "com"
        // This ensures deterministic transcript layout
        let domsep = domsep.add_scalars::<F>(3, "com");

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
        let domsep: DomainSeparator<H> = DomainSeparator::new("test");

        // Step 2: Add an "absorb scalars" tag for 3 scalars, with label "com"
        // This ensures deterministic transcript layout
        let domsep = domsep.add_scalars::<G>(3, "com");

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
        let domsep: DomainSeparator<H> = DomainSeparator::new("test");

        // Step 2: Add absorb-scalar tag for EF4 type and 3 values
        let domsep = domsep.add_scalars::<EF4>(3, "com");

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
        let domsep: DomainSeparator<H> = DomainSeparator::new("test");

        // Step 2: Add absorb-scalar tag for EG2 type and 3 values
        let domsep = domsep.add_scalars::<EG2>(3, "com");

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
