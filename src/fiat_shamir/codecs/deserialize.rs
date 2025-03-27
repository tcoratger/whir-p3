use p3_field::Field;

use super::traits::FieldToUnitDeserialize;
use crate::fiat_shamir::{
    duplex_sponge::interface::DuplexSpongeInterface,
    errors::{ProofError, ProofResult},
    traits::BytesToUnitDeserialize,
    verifier::VerifierState,
};

impl<F, H> FieldToUnitDeserialize<F> for VerifierState<'_, H>
where
    F: Field,
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

// impl<H, MP> FieldToUnitDeserialize<MontyField31<MP>> for VerifierState<'_, H, MontyField31<MP>>
// where
//     MP: MontyParameters + FieldParameters,
//     H: DuplexSpongeInterface<MontyField31<MP>>,
// {
//     fn fill_next_scalars(&mut self, output: &mut [MontyField31<MP>]) -> ProofResult<()> {
//         self.fill_next_units(output)?;
//         Ok(())
//     }
// }
