use p3_field::Field;

use crate::fiat_shamir::errors::ProofResult;

/// Absorb and squeeze field elements to the IO pattern.
pub trait FieldDomainSeparator<F: Field> {
    #[must_use]
    fn add_scalars(self, count: usize, label: &str) -> Self;
    #[must_use]
    fn challenge_scalars(self, count: usize, label: &str) -> Self;
}

/// Interpret verifier messages as uniformly distributed field elements.
///
/// The implementation of this trait **MUST** ensure that the field elements
/// are uniformly distributed and valid.
pub trait UnitToField<F: Field> {
    fn fill_challenge_scalars(&mut self, output: &mut [F]) -> ProofResult<()>;

    fn challenge_scalars<const N: usize>(&mut self) -> ProofResult<[F; N]> {
        let mut output = [F::default(); N];
        self.fill_challenge_scalars(&mut output).map(|()| output)
    }
}

/// Add field elements as shared public information.
pub trait CommonFieldToUnit<F: Field> {
    type Repr;
    fn public_scalars(&mut self, input: &[F]) -> ProofResult<Self::Repr>;
}

/// Add field elements to the protocol transcript.
pub trait FieldToUnitSerialize<F: Field>: CommonFieldToUnit<F> {
    fn add_scalars(&mut self, input: &[F]) -> ProofResult<()>;
}

/// Deserialize field elements from the protocol transcript.
///
/// The implementation of this trait **MUST** ensure that the field elements
/// are correct encodings.
pub trait DeserializeField<F: Field>: CommonFieldToUnit<F> {
    fn fill_next_scalars(&mut self, output: &mut [F]) -> ProofResult<()>;

    fn next_scalars<const N: usize>(&mut self) -> ProofResult<[F; N]> {
        let mut output = [F::default(); N];
        self.fill_next_scalars(&mut output).map(|()| output)
    }
}
