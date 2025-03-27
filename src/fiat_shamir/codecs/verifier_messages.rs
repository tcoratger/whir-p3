use p3_field::{BasedVectorSpace, Field};

use super::{
    bytes_uniform_modp,
    traits::{CommonFieldToUnit, UnitToField},
};
use crate::fiat_shamir::{
    errors::ProofResult,
    traits::{CommonUnitToBytes, UnitToBytes, UnitTranscript},
};

// impl<MP> Unit for MontyField31<MP>
// where
//     MP: MontyParameters,
// {
//     fn write(bunch: &[Self], mut _w: &mut impl io::Write) -> Result<(), io::Error> {
//         for _b in bunch {
//             // b.serialize_compressed(&mut w)
//             //     .map_err(|_| io::Error::other("oh no!"))?;
//         }
//         Ok(())
//     }

//     fn read(mut _r: &mut impl io::Read, bunch: &mut [Self]) -> Result<(), io::Error> {
//         for _b in bunch.iter_mut() {
//             // let b_result = Self::deserialize_compressed(&mut r);
//             // *b = b_result.map_err(|_| io::Error::other("Unable to deserialize into Field."))?;
//         }
//         Ok(())
//     }
// }

impl<F, T> UnitToField<F> for T
where
    F: Field + BasedVectorSpace<F>,
    T: UnitTranscript<u8>,
{
    fn fill_challenge_scalars(&mut self, output: &mut [F]) -> ProofResult<()> {
        let base_field_size = bytes_uniform_modp(F::PrimeSubfield::bits() as u32);
        let mut buf = vec![0u8; F::DIMENSION * base_field_size];

        for _o in output.iter_mut() {
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

    fn public_scalars(&mut self, _input: &[F]) -> ProofResult<Self::Repr> {
        let buf = Vec::new();
        // TODO
        // for i in input {
        //     i.serialize_compressed(&mut buf)?;
        // }
        self.public_bytes(&buf)?;
        Ok(buf)
    }
}

// impl<H, MP> CommonFieldToUnit<MontyField31<MP>> for VerifierState<'_, H, MontyField31<MP>>
// where
//     MP: MontyParameters + FieldParameters,
//     H: DuplexSpongeInterface<MontyField31<MP>>,
// {
//     type Repr = ();

//     fn public_scalars(&mut self, input: &[MontyField31<MP>]) -> ProofResult<Self::Repr> {
//         // let flattened: Vec<_> = input
//         //     .iter()
//         //     .flat_map(Field::to_base_prime_field_elements)
//         //     .collect();
//         // self.public_units(&flattened)?;
//         Ok(())
//     }
// }
