use p3_field::Field;

use super::{traits::FieldDomainSeparator, utils::bytes_modp};
use crate::{
    crypto::field::ExtensionDegree,
    fiat_shamir::{
        codecs::utils::bytes_uniform_modp, domain_separator::DomainSeparator,
        duplex_sponge::interface::DuplexSpongeInterface, traits::ByteDomainSeparator,
    },
};

impl<F, H> FieldDomainSeparator<F> for DomainSeparator<H>
where
    F: Field + ExtensionDegree,
    H: DuplexSpongeInterface,
{
    fn add_scalars(self, count: usize, label: &str) -> Self {
        self.add_bytes(
            count * F::extension_degree() * bytes_modp(F::PrimeSubfield::bits() as u32),
            label,
        )
    }

    fn challenge_scalars(self, count: usize, label: &str) -> Self {
        self.challenge_bytes(
            count * F::extension_degree() * bytes_uniform_modp(F::PrimeSubfield::bits() as u32),
            label,
        )
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;

    use super::*;

    type F = BabyBear;
    type EF4 = BinomialExtensionField<F, 4>;

    #[test]
    fn test_add_scalars_babybear() {
        // Test absorption of scalar field elements (BabyBear).
        // - BabyBear is a base field with extension degree = 1
        // - bits = 31 → bytes_modp(31) = 4
        // - 2 scalars * 1 * 4 = 8 bytes absorbed
        // - "A" indicates absorption in the domain separator
        let sep = <DomainSeparator as FieldDomainSeparator<F>>::add_scalars(
            DomainSeparator::new("babybear"),
            2,
            "foo",
        );
        let expected = b"babybear\0A8foo";
        assert_eq!(sep.as_bytes(), expected);
    }

    #[test]
    fn test_challenge_scalars_babybear() {
        // Test squeezing scalar field elements (BabyBear).
        // - BabyBear has extension degree = 1
        // - bits = 31 → bytes_uniform_modp(31) = 5
        // - 3 scalars * 1 * 5 = 15 bytes squeezed
        // - "S" indicates squeezing in the domain separator
        let sep = <DomainSeparator as FieldDomainSeparator<F>>::challenge_scalars(
            DomainSeparator::new("bb"),
            3,
            "bar",
        );
        let expected = b"bb\0S57bar";
        assert_eq!(sep.as_bytes(), expected);
    }

    #[test]
    fn test_add_scalars_quartic_ext_field() {
        // Test absorption of scalars from a quartic extension field EF4.
        // - EF4 has extension degree = 4
        // - Base field bits = 31 → bytes_modp(31) = 4
        // - 2 scalars * 4 * 4 = 32 bytes absorbed
        let sep = <DomainSeparator as FieldDomainSeparator<EF4>>::add_scalars(
            DomainSeparator::new("ext"),
            2,
            "a",
        );
        let expected = b"ext\0A32a";
        assert_eq!(sep.as_bytes(), expected);
    }

    #[test]
    fn test_challenge_scalars_quartic_ext_field() {
        // Test squeezing of scalars from a quartic extension field EF4.
        // - EF4 has extension degree = 4
        // - Base field bits = 31 → bytes_uniform_modp(31) = 19
        // - 1 scalar * 4 * 19 = 76 bytes squeezed
        // - "S" indicates squeezing in the domain separator
        let sep = <DomainSeparator as FieldDomainSeparator<EF4>>::challenge_scalars(
            DomainSeparator::new("ext2"),
            1,
            "b",
        );

        let expected = b"ext2\0S76b";
        assert_eq!(sep.as_bytes(), expected);
    }
}
