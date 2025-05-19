use std::ops::Deref;

use p3_field::{ExtensionField, Field, PrimeField64, TwoAdicField};
use p3_symmetric::Hash;

use crate::{
    fiat_shamir::{errors::ProofResult, verifier::VerifierState},
    whir::parameters::WhirConfig,
};

/// Represents a parsed commitment from the prover in the WHIR protocol.
///
/// This includes the Merkle root of the committed table and any out-of-domain (OOD)
/// query points and their corresponding answers, which are required for verifier checks.
#[derive(Debug, Clone)]
pub struct ParsedCommitment<F, D> {
    /// Merkle root of the committed evaluation table.
    ///
    /// This hash is used by the verifier to check Merkle proofs of queried evaluations.
    pub root: D,

    /// Points queried by the verifier outside the low-degree evaluation domain.
    ///
    /// These are chosen using Fiat-Shamir and used to test polynomial consistency.
    pub ood_points: Vec<F>,

    /// Answers (evaluations) of the committed polynomial at the corresponding `ood_points`.
    pub ood_answers: Vec<F>,
}

/// Helper for parsing commitment data during verification.
///
/// The `CommitmentReader` wraps the WHIR configuration and provides a convenient
/// method to extract a `ParsedCommitment` by reading values from the Fiat-Shamir transcript.
#[derive(Debug)]
pub struct CommitmentReader<'a, EF, F, H, C, PowStrategy>(
    /// Reference to the verifier’s configuration object.
    ///
    /// This contains all parameters needed to parse the commitment,
    /// including how many out-of-domain samples are expected.
    &'a WhirConfig<EF, F, H, C, PowStrategy>,
)
where
    F: Field + TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField;

impl<'a, EF, F, H, C, PS> CommitmentReader<'a, EF, F, H, C, PS>
where
    F: Field + TwoAdicField + PrimeField64,
    EF: ExtensionField<F> + TwoAdicField,
{
    pub const fn new(params: &'a WhirConfig<EF, F, H, C, PS>) -> Self {
        Self(params)
    }

    pub fn parse_commitment<const DIGEST_ELEMS: usize>(
        &self,
        verifier_state: &mut VerifierState<'_, EF, F>,
    ) -> ProofResult<ParsedCommitment<EF, Hash<F, u8, DIGEST_ELEMS>>> {
        let root = verifier_state.read_digest()?;

        let mut ood_points = vec![EF::ZERO; self.committment_ood_samples];
        let mut ood_answers = vec![EF::ZERO; self.committment_ood_samples];
        if self.committment_ood_samples > 0 {
            verifier_state.fill_challenge_scalars(&mut ood_points)?;
            verifier_state.fill_next_scalars(&mut ood_answers)?;
        }

        Ok(ParsedCommitment {
            root,
            ood_points,
            ood_answers,
        })
    }
}

impl<EF, F, H, C, PowStrategy> Deref for CommitmentReader<'_, EF, F, H, C, PowStrategy>
where
    F: Field + TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
{
    type Target = WhirConfig<EF, F, H, C, PowStrategy>;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}
