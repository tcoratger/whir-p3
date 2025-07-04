use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{BasedVectorSpace, ExtensionField, Field};

use super::domain_separator::DomainSeparator;
use crate::{
    fiat_shamir::{ChallengSampler, errors::ProofError},
    utils::pack_scalars_to_extension,
};

/// The verifier state for a Fiat-Shamir protocol.
#[derive(Debug)]
pub struct VerifierState<F, EF, Challenger>
where
    F: Field,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    challenger: Challenger,
    proof_data: Vec<F>,
    index: usize,
    _extension_field: std::marker::PhantomData<EF>,
}

impl<F, EF, Challenger> VerifierState<F, EF, Challenger>
where
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    EF: ExtensionField<F>,
    F: Field,
{
    /// Initialize a new verifier state.
    #[must_use]
    pub fn new(
        domain_separator: &DomainSeparator<EF, F>,
        proof_data: Vec<F>,
        mut challenger: Challenger,
    ) -> Self {
        let iop_units = domain_separator.as_field_elements();
        challenger.observe_slice(&iop_units);

        Self {
            challenger,
            proof_data,
            index: 0,
            _extension_field: std::marker::PhantomData,
        }
    }

    // Receiving data from the prover:

    pub fn next_base_scalars_vec(&mut self, n: usize) -> Result<Vec<F>, ProofError> {
        if n > self.proof_data.len() - self.index {
            return Err(ProofError::ExceededTranscript);
        }
        let scalars = self.proof_data[self.index..self.index + n].to_vec();
        self.index += n;
        self.challenger.observe_slice(&scalars);
        Ok(scalars)
    }

    pub fn next_base_scalars_const<const N: usize>(&mut self) -> Result<[F; N], ProofError> {
        Ok(self.next_base_scalars_vec(N)?.try_into().unwrap())
    }

    pub fn next_extension_scalars_vec(&mut self, n: usize) -> Result<Vec<EF>, ProofError> {
        let extension_size = <EF as BasedVectorSpace<F>>::DIMENSION;
        Ok(pack_scalars_to_extension(
            &self.next_base_scalars_vec(n * extension_size)?,
        ))
    }

    pub fn next_extension_scalars_const<const N: usize>(&mut self) -> Result<[EF; N], ProofError> {
        Ok(self.next_extension_scalars_vec(N)?.try_into().unwrap())
    }

    pub fn next_extension_scalar(&mut self) -> Result<EF, ProofError> {
        Ok(self.next_extension_scalars_vec(1)?[0])
    }

    // Receiving hints from the prover:

    pub fn receive_hint_base_scalars(&mut self, n: usize) -> Result<Vec<F>, ProofError> {
        if n > self.proof_data.len() - self.index {
            return Err(ProofError::ExceededTranscript);
        }
        let scalars = self.proof_data[self.index..self.index + n].to_vec();
        self.index += n;
        Ok(scalars)
    }

    pub fn receive_hint_extension_scalars(&mut self, n: usize) -> Result<Vec<EF>, ProofError> {
        let extension_size = <EF as BasedVectorSpace<F>>::DIMENSION;
        Ok(pack_scalars_to_extension(
            &self.receive_hint_base_scalars(n * extension_size)?,
        ))
    }

    // Generating pseudo-random values:

    pub fn sample(&mut self) -> EF {
        self.challenger.sample_algebra_element()
    }

    pub fn sample_bits(&mut self, bits: usize) -> usize {
        self.challenger.sample_bits(bits)
    }

    // Pow grinding

    pub fn check_pow_grinding(&mut self, bits: usize) -> Result<(), ProofError> {
        if bits == 0 {
            return Ok(());
        }
        if self.index >= self.proof_data.len() {
            return Err(ProofError::ExceededTranscript);
        }
        let witness = self.proof_data[self.index];
        self.index += 1;
        if self.challenger.check_witness(bits, witness) {
            Ok(())
        } else {
            Err(ProofError::InvalidGrindingWitness)
        }
    }
}

impl<F, EF, Challenger> ChallengSampler<EF> for VerifierState<F, EF, Challenger>
where
    EF: ExtensionField<F>,
    F: Field,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    fn sample(&mut self) -> EF {
        self.sample()
    }

    fn sample_bits(&mut self, bits: usize) -> usize {
        self.sample_bits(bits)
    }
}
