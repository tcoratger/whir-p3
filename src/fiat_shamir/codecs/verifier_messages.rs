use std::io;

use p3_field::{BasedVectorSpace, Field, PrimeField, PrimeField32, PrimeField64};
use rand::{TryCryptoRng, TryRngCore};

use super::{
    bytes_uniform_modp,
    traits::{CommonFieldToUnit, UnitToField},
};
use crate::fiat_shamir::{
    duplex_sponge::{Unit, interface::DuplexSpongeInterface},
    errors::{DomainSeparatorMismatch, ProofResult},
    prover::ProverState,
    traits::{CommonUnitToBytes, UnitToBytes, UnitTranscript},
};

impl<F, T> UnitToField<F> for T
where
    F: Field + BasedVectorSpace<F>,
    T: UnitTranscript<u8>,
{
    fn fill_challenge_scalars(&mut self, output: &mut [F]) -> ProofResult<()> {
        let base_field_size = bytes_uniform_modp(F::PrimeSubfield::bits() as u32);
        let mut buf = vec![0u8; F::DIMENSION * base_field_size];

        for o in output.iter_mut() {
            self.fill_challenge_bytes(&mut buf)?;
            // TODO
            // *o = F::from_basis_coefficients_slice(
            //     buf.chunks(base_field_size).map(|c|
            // F::PrimeSubfield::from_be_bytes_mod_order(c)), );
        }
        Ok(())
    }
}

impl<T, F> CommonFieldToUnit<F> for T
where
    F: Field,
    T: UnitTranscript<u8>,
{
    type Repr = Vec<u8>;

    fn public_scalars(&mut self, input: &[F]) -> ProofResult<Self::Repr> {
        let mut buf = Vec::new();
        // TODO
        // for i in input {
        //     i.serialize_compressed(&mut buf)?;
        // }
        self.public_bytes(&buf)?;
        Ok(buf)
    }
}

// impl<H, R, F> CommonUnitToBytes for ProverState<H, F, R>
// where
//     F: Field + Unit,
//     H: DuplexSpongeInterface<F>,
//     R: TryRngCore + TryCryptoRng,
// {
//     fn public_bytes(&mut self, input: &[u8]) -> Result<(), DomainSeparatorMismatch> {
//         for &byte in input {
//             self.public_units(&[F::from_u8(byte)])?;
//         }
//         Ok(())
//     }
// }
