use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{BasedVectorSpace, ExtensionField, Field};

use super::domain_separator::DomainSeparator;
use crate::{
    fiat_shamir::{ChallengSampler, errors::ProofError},
    utils::pack_scalars_to_extension,
};

/// State held by the verifier in a Fiat-Shamir protocol.
///
/// This struct reconstructs the transcript provided by the prover, consumes proof data,
/// and manages a cryptographic challenger to derive challenges deterministically.
#[derive(Debug)]
pub struct VerifierState<F, EF, Challenger>
where
    F: Field,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    /// Cryptographic challenger used for sampling challenges and observing proof data.
    challenger: Challenger,

    /// Proof data buffer received from the prover, in base field elements.
    proof_data: Vec<F>,

    /// Current read index into `proof_data`.
    index: usize,

    /// Marker to track the extension field type without storing it explicitly.
    _extension_field: std::marker::PhantomData<EF>,
}

impl<F, EF, Challenger> VerifierState<F, EF, Challenger>
where
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    EF: ExtensionField<F>,
    F: Field,
{
    /// Create a new verifier state using the given domain separator and proof data.
    ///
    /// # Arguments
    /// - `domain_separator`: Domain separator binding the transcript to a specific protocol.
    /// - `proof_data`: All base field elements committed by the prover.
    /// - `challenger`: Initialized cryptographic challenger.
    ///
    /// # Returns
    /// A new `VerifierState` ready to consume proof data and derive challenges.
    #[must_use]
    pub fn new(
        domain_separator: &DomainSeparator<EF, F>,
        proof_data: Vec<F>,
        mut challenger: Challenger,
    ) -> Self {
        // Observe the domain separator elements to initialize the challenger consistently.
        let iop_units = domain_separator.as_field_elements();
        challenger.observe_slice(&iop_units);

        Self {
            challenger,
            proof_data,
            index: 0,
            _extension_field: std::marker::PhantomData,
        }
    }

    pub const fn challenger(&self) -> &Challenger {
        &self.challenger
    }

    /// Consume and return `n` base scalars from the proof data, observing them in the challenger.
    ///
    /// # Arguments
    /// - `n`: Number of base scalars to read.
    ///
    /// # Errors
    /// Returns `ProofError::ExceededTranscript` if insufficient data remains.
    pub fn next_base_scalars_vec(&mut self, n: usize) -> Result<Vec<F>, ProofError> {
        // Check that enough data remains to read `n` elements.
        if n > self.proof_data.len() - self.index {
            return Err(ProofError::ExceededTranscript);
        }

        // Slice out the next `n` scalars and copy them.
        let scalars = self.proof_data[self.index..self.index + n].to_vec();
        self.index += n;

        // Observe these scalars in the challenger to update its state.
        self.challenger.observe_slice(&scalars);

        Ok(scalars)
    }

    /// Consume and return `N` base scalars as a fixed-size array, observing them in the challenger.
    ///
    /// # Errors
    /// Returns `ProofError::ExceededTranscript` if insufficient data remains.
    pub fn next_base_scalars_const<const N: usize>(&mut self) -> Result<[F; N], ProofError> {
        // Delegate to vector-based reader, then convert to array.
        Ok(self.next_base_scalars_vec(N)?.try_into().unwrap())
    }

    /// Consume and return `n` extension scalars from the proof data, observing them in the challenger.
    ///
    /// # Arguments
    /// - `n`: Number of extension scalars to read.
    ///
    /// # Errors
    /// Returns `ProofError::ExceededTranscript` if insufficient data remains.
    pub fn next_extension_scalars_vec(&mut self, n: usize) -> Result<Vec<EF>, ProofError> {
        // Calculate number of base scalars per extension scalar.
        let extension_size = <EF as BasedVectorSpace<F>>::DIMENSION;

        // Read and pack into extension elements.
        Ok(pack_scalars_to_extension(
            &self.next_base_scalars_vec(n * extension_size)?,
        ))
    }

    /// Consume and return `N` extension scalars as a fixed-size array, observing them in the challenger.
    ///
    /// # Errors
    /// Returns `ProofError::ExceededTranscript` if insufficient data remains.
    pub fn next_extension_scalars_const<const N: usize>(&mut self) -> Result<[EF; N], ProofError> {
        Ok(self.next_extension_scalars_vec(N)?.try_into().unwrap())
    }

    /// Consume and return a single extension scalar, observing it in the challenger.
    ///
    /// # Errors
    /// Returns `ProofError::ExceededTranscript` if insufficient data remains.
    pub fn next_extension_scalar(&mut self) -> Result<EF, ProofError> {
        Ok(self.next_extension_scalars_vec(1)?[0])
    }

    /// Consume and return `n` base scalars as hints (not observed by the challenger).
    ///
    /// # Arguments
    /// - `n`: Number of base scalars to read.
    ///
    /// # Errors
    /// Returns `ProofError::ExceededTranscript` if insufficient data remains.
    pub fn receive_hint_base_scalars(&mut self, n: usize) -> Result<Vec<F>, ProofError> {
        // Check that enough data remains to read `n` elements.
        if n > self.proof_data.len() - self.index {
            return Err(ProofError::ExceededTranscript);
        }

        // Slice out the next `n` scalars and copy them.
        let scalars = self.proof_data[self.index..self.index + n].to_vec();
        self.index += n;

        Ok(scalars)
    }

    /// Consume and return `n` extension scalars as hints (not observed by the challenger).
    ///
    /// # Arguments
    /// - `n`: Number of extension scalars to read.
    ///
    /// # Errors
    /// Returns `ProofError::ExceededTranscript` if insufficient data remains.
    pub fn receive_hint_extension_scalars(&mut self, n: usize) -> Result<Vec<EF>, ProofError> {
        let extension_size = <EF as BasedVectorSpace<F>>::DIMENSION;

        // Read and pack into extension elements without challenger observation.
        Ok(pack_scalars_to_extension(
            &self.receive_hint_base_scalars(n * extension_size)?,
        ))
    }

    /// Sample a new random extension field element using the challenger.
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

    /// Verify PoW grinding witness correctness.
    ///
    /// # Arguments
    /// - `bits`: Number of bits of grinding difficulty. If zero, no check is performed.
    ///
    /// # Errors
    /// Returns `ProofError::ExceededTranscript` if no data remains,
    /// or `ProofError::InvalidGrindingWitness` if the witness does not satisfy the difficulty.
    pub fn check_pow_grinding(&mut self, bits: usize) -> Result<(), ProofError> {
        // If no grinding is required, succeed immediately.
        if bits == 0 {
            return Ok(());
        }

        // Ensure there is at least one witness element to consume.
        if self.index >= self.proof_data.len() {
            return Err(ProofError::ExceededTranscript);
        }

        // Consume the next witness element.
        let witness = self.proof_data[self.index];
        self.index += 1;

        // Verify the witness using the challenger.
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
