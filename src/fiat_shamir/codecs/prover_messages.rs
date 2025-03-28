use p3_field::{BasedVectorSpace, Field, PrimeField32};
use rand::{CryptoRng, RngCore};

use super::traits::FieldToUnitSerialize;
use crate::fiat_shamir::{
    codecs::traits::CommonFieldToUnit, duplex_sponge::interface::DuplexSpongeInterface,
    errors::ProofResult, prover::ProverState,
};

impl<F, H, R> FieldToUnitSerialize<F> for ProverState<H, u8, R>
where
    F: Field + BasedVectorSpace<F::PrimeSubfield>,
    F::PrimeSubfield: PrimeField32,
    H: DuplexSpongeInterface,
    R: RngCore + CryptoRng,
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

// impl<H, R, F> BytesToUnitSerialize for ProverState<H, F, R>
// where
//     F: Field + Unit,
//     H: DuplexSpongeInterface<F>,
//     R: RngCore + CryptoRng,
// {
//     fn add_bytes(&mut self, input: &[u8]) -> Result<(), DomainSeparatorMismatch> {
//         self.public_bytes(input)?;
//         self.narg_string.extend(input);
//         Ok(())
//     }
// }

// impl<F, H, R> CommonFieldToUnit<F> for ProverState<H, u8, R>
// where
//     F: Field + Unit,
//     H: DuplexSpongeInterface<F>,
//     R: RngCore + CryptoRng,
// {
//     type Repr = Vec<u8>;

//     fn public_scalars(&mut self, input: &[F]) -> ProofResult<Self::Repr> {
//         let mut buf = Vec::new();
//         // for i in input {
//         //     i.serialize_compressed(&mut buf)?;
//         // }
//         self.public_bytes(&buf)?;
//         Ok(buf)
//     }
// }

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};

    use super::*;
    use crate::fiat_shamir::{
        codecs::traits::FieldDomainSeparator, domain_separator::DomainSeparator,
    };

    type F = BabyBear;
    type EF4 = BinomialExtensionField<F, 4>;

    #[test]
    fn test_add_scalars() {
        // Step 1: Create a domain separator with the label "test"
        let domsep = DomainSeparator::new("test");

        // Step 2: Add an "absorb scalars" tag for 3 scalars, with label "com"
        // This ensures deterministic transcript layout
        let domsep = <DomainSeparator as FieldDomainSeparator<F>>::add_scalars(domsep, 3, "com");

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
    fn test_add_scalars_extension() {
        // Step 1: Create a domain separator with the label "test"
        let domsep = DomainSeparator::new("test");

        // Step 2: Add absorb-scalar tag for EF4 type and 3 values
        let domsep = <DomainSeparator as FieldDomainSeparator<EF4>>::add_scalars(domsep, 3, "com");

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
}
