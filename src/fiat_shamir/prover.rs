use std::fmt::Debug;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field, TwoAdicField};

use super::domain_separator::DomainSeparator;
use crate::fiat_shamir::proof_data::ProofData;

/// Prover state for a Fiat-Shamir protocol.
#[derive(Debug)]
pub struct ProverState<EF, F, Challenger, const DIGEST_ELEMS: usize>
where
    F: Field,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    /// The internal challenger.
    pub(crate) challenger: Challenger,
    /// The proof data.
    pub proof_data: ProofData<EF, F, F, DIGEST_ELEMS>,
}

impl<EF, F, Challenger, const DIGEST_ELEMS: usize> ProverState<EF, F, Challenger, DIGEST_ELEMS>
where
    EF: ExtensionField<F> + TwoAdicField,
    F: TwoAdicField,
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
            proof_data: ProofData::default(),
        }
    }
}
