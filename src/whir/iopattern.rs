use p3_field::{Field, PrimeCharacteristicRing, TwoAdicField};

use super::parameters::WhirConfig;

pub trait DigestIOPattern {
    #[must_use]
    fn add_digest(self, label: &str) -> Self;
}

pub trait WhirIOPattern<F>
where
    F: Field + TwoAdicField,
    <F as PrimeCharacteristicRing>::PrimeSubfield: TwoAdicField,
{
    #[must_use]
    fn commit_statement<PowStrategy, Perm16, Perm24>(
        self,
        params: &WhirConfig<F, PowStrategy, Perm16, Perm24>,
    ) -> Self;

    #[must_use]
    fn add_whir_proof<PowStrategy, Perm16, Perm24>(
        self,
        params: &WhirConfig<F, PowStrategy, Perm16, Perm24>,
    ) -> Self;
}
