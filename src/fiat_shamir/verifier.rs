use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field, TwoAdicField};

use super::domain_separator::DomainSeparator;
use crate::fiat_shamir::proof_data::ProofData;

/// The verifier state for a Fiat-Shamir protocol.
#[derive(Debug)]
pub struct VerifierState<EF, F, Challenger, const DIGEST_ELEMS: usize>
where
    F: Field,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    /// Challenger for sampling randomness.
    pub challenger: Challenger,

    /// The proof data.
    pub proof_data: ProofData<EF, F, F, DIGEST_ELEMS>,
}

impl<EF, F, Challenger, const DIGEST_ELEMS: usize> VerifierState<EF, F, Challenger, DIGEST_ELEMS>
where
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    EF: ExtensionField<F> + TwoAdicField,
    F: TwoAdicField,
{
    /// Initialize a new verifier state.
    #[must_use]
    pub fn new(
        domain_separator: &DomainSeparator<EF, F>,
        proof_data: ProofData<EF, F, F, DIGEST_ELEMS>,
        mut challenger: Challenger,
    ) -> Self {
        let iop_units = domain_separator.as_field_elements();
        challenger.observe_slice(&iop_units);

        Self {
            challenger,
            proof_data,
        }
    }
}
