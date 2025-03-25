use p3_field::{BasedVectorSpace, Field};

use super::{bytes_modp, traits::FieldDomainSeparator};
use crate::fiat_shamir::{
    codecs::bytes_uniform_modp, domain_separator::DomainSeparator,
    duplex_sponge::interface::DuplexSpongeInterface, traits::ByteDomainSeparator,
};

impl<F, H> FieldDomainSeparator<F> for DomainSeparator<H>
where
    F: Field + BasedVectorSpace<F>,
    H: DuplexSpongeInterface,
{
    fn add_scalars(self, count: usize, label: &str) -> Self {
        self.add_bytes(count * F::DIMENSION * bytes_modp(F::PrimeSubfield::bits() as u32), label)
    }

    fn challenge_scalars(self, count: usize, label: &str) -> Self {
        self.challenge_bytes(
            count * F::DIMENSION * bytes_uniform_modp(F::PrimeSubfield::bits() as u32),
            label,
        )
    }
}
