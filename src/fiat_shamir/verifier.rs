use std::marker::PhantomData;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field, PrimeField64, TwoAdicField};

use super::domain_separator::DomainSeparator;
use crate::fiat_shamir::proof_data::ProofData;

/// [`VerifierState`] is the verifier state.
///
/// Internally, it simply contains a stateful hash.
/// Given as input an [`DomainSeparator`] and a NARG string, it allows to
/// de-serialize elements from the NARG string and make them available to the zero-knowledge
/// verifier.
#[derive(Debug)]
pub struct VerifierState<EF, F, Challenger, const DIGEST_ELEMS: usize>
where
    F: Field,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    /// Challenger for sampling randomness.
    pub(crate) challenger: Challenger,

    /// The proof data.
    pub(crate) proof_data: ProofData<EF, F, F, DIGEST_ELEMS>,

    /// Marker for the base field `F`.
    ///
    /// This field is never read or written; it ensures type correctness for field-level operations.
    _field: PhantomData<F>,

    /// Marker for the extension field `EF`.
    ///
    /// Like `_field`, this is only for type-level bookkeeping. The extension field is used
    /// to deserialize and operate on scalars in high-dimensional protocols.
    _extension_field: PhantomData<EF>,
}

impl<EF, F, Challenger, const DIGEST_ELEMS: usize> VerifierState<EF, F, Challenger, DIGEST_ELEMS>
where
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    EF: ExtensionField<F> + TwoAdicField,
    F: PrimeField64 + TwoAdicField,
{
    /// Creates a new [`VerifierState`] instance with the given sponge and IO Pattern.
    ///
    /// The resulting object will act as the verifier in a zero-knowledge protocol.
    /// `verify_operations` indicates whether Fiat-Shamir operations (observe, sample, hint)
    /// should be verified at runtime.
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
            _field: PhantomData,
            _extension_field: PhantomData,
        }
    }
}
