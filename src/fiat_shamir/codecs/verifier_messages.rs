use p3_field::{BasedVectorSpace, Field, PrimeField64};

use super::{
    traits::{CommonFieldToUnit, UnitToField},
    utils::from_be_bytes_mod_order,
};
use crate::fiat_shamir::{
    codecs::utils::bytes_uniform_modp,
    errors::ProofResult,
    traits::{CommonUnitToBytes, UnitToBytes, UnitTranscript},
};

impl<F, T> UnitToField<F> for T
where
    F: Field + BasedVectorSpace<F::PrimeSubfield>,
    F::PrimeSubfield: PrimeField64,
    T: UnitTranscript<u8>,
{
    fn fill_challenge_scalars(&mut self, output: &mut [F]) -> ProofResult<()> {
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
}

impl<T, F> CommonFieldToUnit<F> for T
where
    F: Field + BasedVectorSpace<F::PrimeSubfield>,
    F::PrimeSubfield: PrimeField64,
    T: UnitTranscript<u8>,
{
    type Repr = Vec<u8>;

    fn public_scalars(&mut self, input: &[F]) -> ProofResult<Self::Repr> {
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
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
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
    fn scalar_challenge_single_basefield_case_1() {
        // Generate a domain separator with known tag and one challenge scalar
        let domsep: DomainSeparator<H> = DomainSeparator::new("chal");
        let domsep = domsep.challenge_scalars::<F>(1, "tag");
        let mut prover = domsep.to_prover_state();

        // Sample a single scalar
        let mut out = [F::ZERO; 1];
        prover.fill_challenge_scalars(&mut out).unwrap();

        // Expected value checked with reference implementation
        assert_eq!(out, [F::from_u64(941695054)]);
    }

    #[test]
    fn scalar_challenge_single_basefield_case_2() {
        let domsep: DomainSeparator<H> = DomainSeparator::new("chal2");
        let domsep = domsep.challenge_scalars::<F>(1, "tag");
        let mut prover = domsep.to_prover_state();

        let mut out = [F::ZERO; 1];
        prover.fill_challenge_scalars(&mut out).unwrap();

        assert_eq!(out, [F::from_u64(1368000007)]);
    }

    #[test]
    fn scalar_challenge_multiple_basefield_scalars() {
        let domsep: DomainSeparator<H> = DomainSeparator::new("chal");
        let domsep = domsep.challenge_scalars::<F>(10, "tag");
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
        let domsep: DomainSeparator<H> = DomainSeparator::new("chal");
        let domsep = domsep.challenge_scalars::<EF4>(1, "tag");
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
        let domsep: DomainSeparator<H> = DomainSeparator::new("chal");
        let domsep = domsep.challenge_scalars::<EF4>(5, "tag");
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
        let domsep: DomainSeparator<H> = DomainSeparator::new("field");
        let domsep = domsep.add_scalars::<BabyBear>(2, "test");

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
        let domsep: DomainSeparator<H> = DomainSeparator::new("field");
        let domsep = domsep.add_scalars::<G>(2, "test");

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
        let domsep: DomainSeparator<H> = DomainSeparator::new("field");
        let domsep = domsep.add_scalars::<EF4>(2, "test");

        // Compute expected bytes manually: serialize each coefficient of EF4
        let expected_bytes = [
            111, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 222, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0,
        ];

        // Use CommonFieldToUnit to serialize the values through the transcript
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
        let domsep: DomainSeparator<H> = DomainSeparator::new("field");
        let domsep = domsep.add_scalars::<EG2>(2, "test");

        // Compute expected bytes manually: serialize each coefficient of EF4
        let expected_bytes = [
            111, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 222, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0,
        ];

        // Use CommonFieldToUnit to serialize the values through the transcript
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

        let domsep: DomainSeparator<H> = DomainSeparator::new("mixed");
        let domsep = domsep.add_scalars::<BabyBear>(values.len(), "mix");

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
