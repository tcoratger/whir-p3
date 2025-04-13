use p3_field::{ExtensionField, Field, PrimeField64, TwoAdicField};
use p3_symmetric::Hash;

use crate::{
    fiat_shamir::{errors::ProofResult, verifier::VerifierState},
    whir::parameters::WhirConfig,
};

#[derive(Debug, Clone)]
pub struct ParsedCommitment<F, D> {
    pub root: D,
    pub ood_points: Vec<F>,
    pub ood_answers: Vec<F>,
}

#[derive(Debug)]
pub struct CommitmentReader<'a, EF, F, H, C, PowStrategy>(&'a WhirConfig<EF, F, H, C, PowStrategy>)
where
    F: Field + TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField<PrimeSubfield = F>;

impl<'a, EF, F, H, C, PS> CommitmentReader<'a, EF, F, H, C, PS>
where
    F: Field + TwoAdicField + PrimeField64,
    EF: ExtensionField<F> + TwoAdicField<PrimeSubfield = F>,
{
    pub const fn new(params: &'a WhirConfig<EF, F, H, C, PS>) -> Self {
        Self(params)
    }

    pub fn parse_commitment<const DIGEST_ELEMS: usize>(
        &self,
        verifier_state: &mut VerifierState<'_>,
    ) -> ProofResult<ParsedCommitment<EF, Hash<F, u8, DIGEST_ELEMS>>> {
        let root = verifier_state.read_digest()?;

        let mut ood_points = vec![EF::ZERO; self.0.committment_ood_samples];
        let mut ood_answers = vec![EF::ZERO; self.0.committment_ood_samples];
        if self.0.committment_ood_samples > 0 {
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
