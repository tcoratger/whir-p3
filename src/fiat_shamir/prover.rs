use std::fmt::Debug;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field};

use super::domain_separator::DomainSeparator;
use crate::{fiat_shamir::ChallengSampler, utils::flatten_scalars_to_base};

/// Prover state for a Fiat-Shamir protocol.
#[derive(Debug)]
pub struct ProverState<F, EF, Challenger>
where
    F: Field,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    challenger: Challenger,
    proof_data: Vec<F>,
    _extension_field: std::marker::PhantomData<EF>,
}

impl<F, EF, Challenger> ProverState<F, EF, Challenger>
where
    EF: ExtensionField<F>,
    F: Field,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    /// Initialize a new prover state.
    #[must_use]
    pub fn new(domain_separator: &DomainSeparator<EF, F>, mut challenger: Challenger) -> Self
    where
        Challenger: Clone,
    {
        challenger.observe_slice(&domain_separator.as_field_elements());

        Self {
            challenger,
            proof_data: Vec::new(),
            _extension_field: std::marker::PhantomData,
        }
    }

    pub fn proof_data(&self) -> &[F] {
        &self.proof_data
    }

    // Passing data to the verifier:

    pub fn add_base_scalars(&mut self, scalars: &[F]) {
        self.proof_data.extend(scalars);
        self.challenger.observe_slice(scalars);
    }

    pub fn add_extension_scalars(&mut self, scalars: &[EF]) {
        self.add_base_scalars(&flatten_scalars_to_base(scalars));
    }

    pub fn add_extension_scalar(&mut self, scalar: EF) {
        self.add_extension_scalars(&[scalar]);
    }

    // Passing hints to the verifier:

    pub fn hint_base_scalars(&mut self, scalars: &[F]) {
        self.proof_data.extend(scalars);
    }

    pub fn hint_extension_scalars(&mut self, scalars: &[EF]) {
        self.proof_data.extend(flatten_scalars_to_base(scalars));
    }

    // Generating pseudo-random values:

    pub fn sample(&mut self) -> EF {
        self.challenger.sample_algebra_element()
    }

    pub fn sample_bits(&mut self, bits: usize) -> usize {
        self.challenger.sample_bits(bits)
    }

    // Pow grinding

    pub fn pow_grinding(&mut self, bits: usize) {
        if bits == 0 {
            return;
        }
        let witness = self.challenger.grind(bits);
        self.proof_data.push(witness);
    }
}

impl<F, EF, Challenger> ChallengSampler<EF> for ProverState<F, EF, Challenger>
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
