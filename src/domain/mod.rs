use p3_field::TwoAdicField;
use radix2::Radix2EvaluationDomain;

use crate::domain::general::GeneralEvaluationDomain;

pub mod general;
pub mod radix2;

/// Represents an evaluation domain used in FFT-based polynomial arithmetic.
///
/// This domain is constructed over a multiplicative subgroup of a finite field, enabling
/// efficient Fast Fourier Transforms (FFTs).
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Domain<EF: TwoAdicField> {
    pub backing_domain: GeneralEvaluationDomain<EF>,
}

impl<EF: TwoAdicField> Domain<EF> {
    /// Constructs a new evaluation domain for a polynomial of given `degree`.
    ///
    /// The domain size is computed as:
    /// ```ignore
    /// N = degree * 2 ^ log_rho_inv
    /// ```
    /// where `log_rho_inv` determines additional scaling.
    ///
    /// If the domain cannot be constructed, it returns `None`.
    #[must_use]
    pub fn new(degree: usize, log_rho_inv: usize) -> Option<Self> {
        let size = degree * (1 << log_rho_inv);
        let backing_domain = GeneralEvaluationDomain::new(size)?;
        Some(Self { backing_domain })
    }

    /// Returns the domain size after `folding_factor` applications of folding.
    ///
    /// Folding reduces the domain size by a factor of `2^folding_factor`, ensuring that
    /// `size` remains divisible by `2^folding_factor`. The resulting size is:
    /// ```ignore
    /// folded_size = size / 2 ^ folding_factor
    /// ```
    pub const fn folded_size(&self, folding_factor: usize) -> usize {
        assert!(self.backing_domain.size() % (1 << folding_factor) == 0);
        self.backing_domain.size() >> folding_factor
    }

    /// Returns the total size of the domain.
    #[inline]
    pub const fn size(&self) -> usize {
        self.backing_domain.size()
    }

    /// Scales the domain generator by a given power, reducing its size.
    ///
    /// Scaling transforms the domain `<w>` into `<w^power>`, where:
    /// ```ignore
    /// new_size = size / power
    /// ```
    /// The base domain is set to `None` since scaling only affects the extended field.
    #[must_use]
    pub fn scale(&self, power: usize) -> Self {
        Self {
            backing_domain: self.scale_generator_by(power),
        }
    }

    /// Scales the domain generator by a given power.
    ///
    /// Given a domain `<w>`, this computes `<w^power>`, reducing the size:
    /// ```ignore
    /// new_size = size / power
    /// ```
    /// It ensures `size % power == 0` for a valid transformation.
    fn scale_generator_by(&self, power: usize) -> GeneralEvaluationDomain<EF> {
        let starting_size = self.size();
        assert_eq!(starting_size % power, 0);
        let new_size = starting_size / power;
        let log_size_of_group = new_size.trailing_zeros();
        let size_as_field_element = EF::from_u64(new_size as u64);

        match self.backing_domain {
            GeneralEvaluationDomain::Radix2(r2) => {
                let group_gen = r2.group_gen.exp_u64(power as u64);
                let offset = r2.offset.exp_u64(power as u64);

                GeneralEvaluationDomain::Radix2(Radix2EvaluationDomain {
                    size: new_size as u64,
                    log_size_of_group,
                    size_as_field_element,
                    size_inv: size_as_field_element.inverse(),
                    group_gen,
                    group_gen_inv: group_gen.inverse(),
                    offset,
                    offset_inv: r2.offset_inv.exp_u64(power as u64),
                    offset_pow_size: offset.exp_u64(new_size as u64),
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_baby_bear::BabyBear;
    use p3_field::Field;
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};

    type F = BabyBear;
    type EF4 = BinomialExtensionField<F, 4>;

    #[test]
    fn test_domain_creation_valid() {
        // We choose degree = 8 and log_rho_inv = 0, so the expected size is:
        // size = degree * 2^log_rho_inv = 8 * 2^0 = 8
        let domain = Domain::<F>::new(8, 0).unwrap();
        assert_eq!(domain.backing_domain.size(), 8);
    }

    #[test]
    fn test_domain_creation_invalid() {
        // We try to create a domain with size larger than F's TWO_ADICITY limit.
        // F::TWO_ADICITY = 27, so we pick a size beyond 2^27.
        let invalid_size = 1 << (F::TWO_ADICITY + 1);
        assert!(Domain::<F>::new(invalid_size, 0).is_none());
    }

    #[test]
    fn test_coset_offsets() {
        let domain = Domain::<F>::new(16, 0).unwrap();
        let backing_domain = &domain.backing_domain;

        // Coset offset should be 1 in default case
        assert_eq!(backing_domain.coset_offset(), F::ONE);
        assert_eq!(backing_domain.coset_offset_inv(), F::ONE);

        // Offset raised to size should be 1: offset^size = 1
        let offset = backing_domain.coset_offset();
        assert_eq!(offset.exp_u64(backing_domain.size() as u64), F::ONE);
    }

    #[test]
    fn test_size_as_field_element() {
        let domain = Domain::<F>::new(16, 0).unwrap();
        let backing_domain = &domain.backing_domain;

        // Check if size_as_field_element correctly converts the size to field representation
        assert_eq!(backing_domain.size_as_field_element(), F::from_u64(16));
    }

    #[test]
    fn test_size_inv() {
        let domain = Domain::<F>::new(16, 0).unwrap();
        let backing_domain = &domain.backing_domain;

        // size_inv should be the multiplicative inverse of size in the field
        let size_inv = backing_domain.size_inv();
        assert_eq!(size_inv * F::from_u64(16), F::ONE);
    }

    #[test]
    fn test_folded_size_valid() {
        let domain = Domain::<F>::new(16, 0).unwrap();

        // Folding factor = 2 → New size = size / (2^2) = 16 / 4 = 4
        assert_eq!(domain.folded_size(2), 4);
    }

    #[test]
    #[should_panic]
    fn test_folded_size_invalid() {
        let domain = Domain::<F>::new(10, 0).unwrap();
        // This should panic since 16 is not divisible by 5^2
        domain.folded_size(5);
    }

    #[test]
    fn test_scaling_preserves_structure() {
        let domain = Domain::<F>::new(16, 0).unwrap();
        let scaled_domain = domain.scale(2);

        // The scaled domain should have the size divided by 2.
        assert_eq!(scaled_domain.size(), 8);

        // The generator of the scaled domain should be `g^2`.
        let expected_group_gen = domain.backing_domain.group_gen().exp_u64(2);
        assert_eq!(scaled_domain.backing_domain.group_gen(), expected_group_gen);

        // The inverse generator should be `g^-2`.
        let expected_group_gen_inv = expected_group_gen.try_inverse().unwrap();
        assert_eq!(
            scaled_domain.backing_domain.group_gen_inv(),
            expected_group_gen_inv
        );
    }

    #[test]
    fn test_scale_generator_by_valid() {
        let domain = Domain::<F>::new(16, 0).unwrap();
        let scaled_domain = domain.scale_generator_by(2);

        // New size = size / power = 16 / 2 = 8
        assert_eq!(scaled_domain.size(), 8);

        // New generator should be g^2
        let expected_group_gen = domain.backing_domain.group_gen().exp_u64(2);
        assert_eq!(scaled_domain.group_gen(), expected_group_gen);

        // New inverse generator should be (g^2)^-1
        let expected_group_gen_inv = expected_group_gen.inverse();
        assert_eq!(scaled_domain.group_gen_inv(), expected_group_gen_inv);
    }

    #[test]
    #[should_panic]
    fn test_scale_generator_by_invalid() {
        let domain = Domain::<F>::new(10, 0).unwrap();
        // This should panic since size is not divisible by 3
        domain.scale_generator_by(3);
    }

    #[test]
    fn test_offsets_after_scaling() {
        let domain = Domain::<F>::new(16, 0).unwrap();
        let scaled_domain = domain.scale_generator_by(2);

        // New domain size should be 16 / 2 = 8
        assert_eq!(scaled_domain.size(), 8);

        // The offset should be raised to the power of `power`
        let expected_offset = domain.backing_domain.coset_offset().exp_u64(2);
        assert_eq!(scaled_domain.coset_offset(), expected_offset);

        // The inverse offset should be raised to the power of `power`
        let expected_offset_inv = domain.backing_domain.coset_offset_inv().exp_u64(2);
        assert_eq!(scaled_domain.coset_offset_inv(), expected_offset_inv);

        // The offset_pow_size should be offset^(new_size)
        let expected_offset_pow_size = expected_offset.exp_u64(8);
        assert_eq!(
            scaled_domain.coset_offset_pow_size(),
            expected_offset_pow_size
        );
    }

    #[test]
    fn test_size_as_field_element_after_scaling() {
        let domain = Domain::<F>::new(16, 0).unwrap();
        let scaled_domain = domain.scale_generator_by(2);

        // New domain size should be 16 / 2 = 8
        let expected_size_as_field_element = F::from_u64(8);

        // Check if size_as_field_element correctly represents the scaled size in the field
        assert_eq!(
            scaled_domain.size_as_field_element(),
            expected_size_as_field_element
        );
    }

    #[test]
    fn test_log_size_of_group_after_scaling() {
        let domain = Domain::<F>::new(16, 0).unwrap();
        let scaled_domain = domain.scale_generator_by(2);

        // The original size is 16, so log_size_of_group should be log2(16) = 4.
        assert_eq!(domain.backing_domain.log_size_of_group(), 4);

        // After scaling by 2, the new size is 16 / 2 = 8, so log_size_of_group should be log2(8) =
        // 3.
        assert_eq!(scaled_domain.log_size_of_group(), 3);
    }

    #[test]
    fn test_domain_creation_ef4_valid() {
        // Creates a domain over EF4, but constructs it by lifting from a base domain in F.
        // This ensures that to_extension_domain works properly.
        let domain = Domain::<EF4>::new(8, 0).unwrap();
        assert_eq!(domain.backing_domain.size(), 8);
    }

    #[test]
    fn test_domain_creation_ef4_invalid() {
        // If the base domain cannot be created due to TWO_ADICITY limits, the constructor should fail.
        let invalid_size = 1 << (EF4::TWO_ADICITY + 1);
        assert!(Domain::<EF4>::new(invalid_size, 0).is_none());
    }

    #[test]
    fn test_scaling_preserves_structure_ef4() {
        // Scaling should preserve the multiplicative structure of the group in EF4.
        let domain = Domain::<EF4>::new(16, 0).unwrap();
        let scaled_domain = domain.scale(2); // Halve the size, square the generator

        // Size should be halved
        assert_eq!(scaled_domain.size(), 8);

        // Generator should be squared
        let expected_gen = domain.backing_domain.group_gen().exp_u64(2);
        assert_eq!(scaled_domain.backing_domain.group_gen(), expected_gen);

        // Inverse generator should be inverse of g²
        assert_eq!(
            scaled_domain.backing_domain.group_gen_inv(),
            expected_gen.inverse()
        );
    }

    #[test]
    fn test_folded_size_ef4_valid() {
        // Verify that the folded size logic works over EF4.
        // 16 elements → fold by factor 4 → expect size 4
        let domain = Domain::<EF4>::new(16, 0).unwrap();
        assert_eq!(domain.folded_size(2), 4);
    }

    #[test]
    fn test_scale_generator_by_valid_ef4() {
        // We directly test the internal generator scaling function with EF4.
        let domain = Domain::<EF4>::new(16, 0).unwrap();
        let scaled = domain.scale_generator_by(2);

        // Should result in domain of size 8
        assert_eq!(scaled.size(), 8);

        // The generator must be g² and the inverse g⁻²
        let expected_gen = domain.backing_domain.group_gen().exp_u64(2);
        assert_eq!(scaled.group_gen(), expected_gen);
        assert_eq!(scaled.group_gen_inv(), expected_gen.inverse());
    }

    #[test]
    #[should_panic]
    fn test_scale_generator_by_invalid_ef4() {
        // If the domain size is not divisible by the scaling factor,
        // this should panic due to assertion.
        let domain = Domain::<EF4>::new(10, 0).unwrap();
        domain.scale_generator_by(3); // 10 not divisible by 3
    }

    #[test]
    fn test_offsets_after_scaling_ef4() {
        // This test ensures that the coset offset is scaled correctly along with the generator.
        let domain = Domain::<EF4>::new(16, 0).unwrap();
        let scaled = domain.scale_generator_by(2);

        // offset' = offset^2, offset_inv' = offset_inv^2
        let expected_offset = domain.backing_domain.coset_offset().exp_u64(2);
        let expected_inv = domain.backing_domain.coset_offset_inv().exp_u64(2);

        // offset_pow_size' = offset'^new_size
        let expected_pow = expected_offset.exp_u64(8);

        assert_eq!(scaled.coset_offset(), expected_offset);
        assert_eq!(scaled.coset_offset_inv(), expected_inv);
        assert_eq!(scaled.coset_offset_pow_size(), expected_pow);
    }

    #[test]
    fn test_size_as_field_element_ef4() {
        // The domain size as an EF4 field element should match the u64 conversion.
        let domain = Domain::<EF4>::new(16, 0).unwrap();
        assert_eq!(
            domain.backing_domain.size_as_field_element(),
            EF4::from_u64(16)
        );
    }

    #[test]
    fn test_size_inv_ef4() {
        // size_inv should be the true multiplicative inverse of size in EF4.
        let domain = Domain::<EF4>::new(16, 0).unwrap();
        let inv = domain.backing_domain.size_inv();
        assert_eq!(inv * EF4::from_u64(16), EF4::ONE);
    }

    #[test]
    fn test_log_size_of_group_after_scaling_ef4() {
        // The log of the domain size (in base 2) should decrease by 1 when scaling by 2.
        let domain = Domain::<EF4>::new(16, 0).unwrap();
        let scaled = domain.scale_generator_by(2);

        assert_eq!(domain.backing_domain.log_size_of_group(), 4); // log2(16)
        assert_eq!(scaled.log_size_of_group(), 3); // log2(8)
    }
}
