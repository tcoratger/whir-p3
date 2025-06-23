use std::fmt::Debug;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field, TwoAdicField};

use super::domain_separator::DomainSeparator;
use crate::fiat_shamir::proof_data::ProofData;

/// [`ProverState`] is the prover state of an interactive proof (IP) system.
///
/// It internally holds the **secret coins** of the prover for zero-knowledge, and
/// has the hash function state for the verifier state.
///
/// Unless otherwise specified,
/// [`ProverState`] is set to work over bytes with [`DefaultHash`] and
/// rely on the default random number generator [`DefaultRng`].
///
///
/// # Safety
///
/// The prover state is meant to be private in contexts where zero-knowledge is desired.
/// Leaking the prover state *will* leak the prover's private coins and as such it will compromise
/// the zero-knowledge property. [`ProverState`] does not implement [`Clone`] or [`Copy`] to prevent
/// accidental leaks.
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
    /// Initialize a new `ProverState` from the given domain separator.
    ///
    /// Seeds the internal sponge with the domain separator.
    /// `verify_operations` indicates whether Fiat-Shamir operations (observe, sample, hint)
    /// should be verified at runtime.
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
