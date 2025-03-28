use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing, PrimeField32};

use super::traits::FieldToUnitDeserialize;
use crate::fiat_shamir::{
    duplex_sponge::interface::DuplexSpongeInterface,
    errors::{ProofError, ProofResult},
    traits::BytesToUnitDeserialize,
    verifier::VerifierState,
};

impl<F, H> FieldToUnitDeserialize<F> for VerifierState<'_, H>
where
    F: Field + BasedVectorSpace<<F as PrimeCharacteristicRing>::PrimeSubfield>,
    F::PrimeSubfield: PrimeField32,
    H: DuplexSpongeInterface,
{
    fn fill_next_scalars(&mut self, output: &mut [F]) -> ProofResult<()> {
        let size = core::mem::size_of::<F>();

        for o in output.iter_mut() {
            let mut buf = vec![0u8; size];
            self.fill_next_bytes(&mut buf)?;
            *o = bincode::deserialize(&buf).map_err(|_| ProofError::SerializationError)?;
        }

        Ok(())
    }
}
