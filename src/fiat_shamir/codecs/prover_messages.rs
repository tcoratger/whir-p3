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
        let serialized = self.public_scalars(input);
        self.narg_string.extend(serialized?);
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
