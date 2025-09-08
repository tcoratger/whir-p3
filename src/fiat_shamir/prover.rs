use std::fmt::Debug;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field};

use super::domain_separator::DomainSeparator;
use crate::fiat_shamir::ChallengeSampler;

/// State held by the prover in a Fiat-Shamir protocol.
///
/// This struct tracks the prover's transcript data and manages interaction
/// with a cryptographic challenger. It collects data to be sent to the verifier,
/// maintains the current transcript for challenge derivation, and supports
/// hints and proof-of-work (PoW) grinding mechanisms.
#[derive(Debug)]
pub struct ProverState<F, EF, Challenger>
where
    F: Field,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    /// Cryptographic challenger used to sample challenges and observe data.
    challenger: Challenger,

    /// Transcript data (proof data) accumulated during protocol execution,
    /// to be sent to the verifier.
    proof_data: Vec<F>,

    /// Marker to keep track of the extension field type without storing it explicitly.
    _extension_field: std::marker::PhantomData<EF>,
}

impl<F, EF, Challenger> ProverState<F, EF, Challenger>
where
    EF: ExtensionField<F>,
    F: Field,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    /// Create a new prover state with a given domain separator and challenger.
    ///
    /// # Arguments
    /// - `domain_separator`: Used to bind this transcript to a specific protocol context.
    /// - `challenger`: The initial cryptographic challenger state.
    ///
    /// # Returns
    /// A fresh `ProverState` ready to accumulate data.
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

    /// Access all proof data accumulated so far.
    ///
    /// This data will be sent to the verifier as part of the proof.
    pub fn proof_data(&self) -> &[F] {
        &self.proof_data
    }

    /// Append base field scalars to the transcript and observe them in the challenger.
    ///
    /// # Arguments
    /// - `scalars`: Slice of base field elements to append.
    pub fn add_base_scalars(&mut self, scalars: &[F]) {
        // Extend the proof data vector with these scalars.
        self.proof_data.extend(scalars);

        // Notify the challenger that these scalars have been committed.
        self.challenger.observe_slice(scalars);
    }

    /// Append extension field scalars to the transcript.
    ///
    /// Internally, these are flattened to base field scalars.
    ///
    /// # Arguments
    /// - `scalars`: Slice of extension field elements to append.
    pub fn add_extension_scalars(&mut self, scalars: &[EF]) {
        // Flatten each extension scalar into base scalars and delegate.
        self.add_base_scalars(&EF::flatten_to_base(scalars.to_vec()));
    }

    /// Append a single extension field scalar to the transcript.
    ///
    /// # Arguments
    /// - `scalar`: Extension field element to append.
    pub fn add_extension_scalar(&mut self, scalar: EF) {
        // Call the multi-scalar function with a one-element slice.
        self.add_extension_scalars(&[scalar]);
    }

    /// Append base field scalars to the transcript as hints.
    ///
    /// Unlike `add_base_scalars`, hints are not observed by the challenger.
    ///
    /// # Arguments
    /// - `scalars`: Slice of base field elements to append.
    pub fn hint_base_scalars(&mut self, scalars: &[F]) {
        // Only extend proof data, no challenger observation.
        self.proof_data.extend(scalars);
    }

    /// Append extension field scalars to the transcript as hints.
    ///
    /// # Arguments
    /// - `scalars`: Slice of extension field elements to append.
    pub fn hint_extension_scalars(&mut self, scalars: &[EF]) {
        // Flatten extension field scalars and append as base field scalars.
        self.proof_data
            .extend(EF::flatten_to_base(scalars.to_vec()));
    }

    /// Sample a new random extension field element from the challenger.
    ///
    /// # Returns
    /// A new challenge element in the extension field.
    pub fn sample(&mut self) -> EF {
        self.challenger.sample_algebra_element()
    }

    /// Sample random bits from the challenger.
    ///
    /// # Arguments
    /// - `bits`: Number of bits to sample.
    ///
    /// # Returns
    /// A uniformly random value with `bits` bits.
    pub fn sample_bits(&mut self, bits: usize) -> usize {
        self.challenger.sample_bits(bits)
    }

    /// Perform PoW grinding and append the witness to the transcript.
    ///
    /// # Arguments
    /// - `bits`: Number of bits of grinding difficulty. If zero, no grinding is performed.
    pub fn pow_grinding(&mut self, bits: usize) {
        // Skip grinding entirely if difficulty is zero.
        if bits == 0 {
            return;
        }

        // Perform grinding and obtain a witness element in the base field.
        let witness = self.challenger.grind(bits);

        // Append the witness to the proof data.
        self.proof_data.push(witness);
    }
}

impl<F, EF, Challenger> ChallengeSampler<EF> for ProverState<F, EF, Challenger>
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
