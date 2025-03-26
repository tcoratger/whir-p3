use p3_field::{Field, PrimeCharacteristicRing, PrimeField32, TwoAdicField};
use p3_symmetric::Hash;

use crate::{
    fiat_shamir::{
        codecs::traits::{DeserializeField, UnitToField},
        errors::ProofResult,
        traits::UnitToBytes,
    },
    whir::{parameters::WhirConfig, utils::DigestReader},
};

#[derive(Debug, Clone)]
pub struct ParsedCommitment<F, D> {
    pub root: D,
    pub ood_points: Vec<F>,
    pub ood_answers: Vec<F>,
}

#[derive(Debug)]
pub struct CommitmentReader<'a, F, H, C, PowStrategy>(&'a WhirConfig<F, H, C, PowStrategy>)
where
    F: Field + TwoAdicField,
    <F as PrimeCharacteristicRing>::PrimeSubfield: TwoAdicField;

impl<'a, F, H, C, PS> CommitmentReader<'a, F, H, C, PS>
where
    F: Field + TwoAdicField + PrimeField32 + Eq,
    <F as PrimeCharacteristicRing>::PrimeSubfield: TwoAdicField,
{
    pub const fn new(params: &'a WhirConfig<F, H, C, PS>) -> Self {
        Self(params)
    }

    fn parse_commitment<VerifierState, const DIGEST_ELEMS: usize>(
        &self,
        verifier_state: &mut VerifierState,
    ) -> ProofResult<ParsedCommitment<F, Hash<F, u8, DIGEST_ELEMS>>>
    where
        VerifierState: UnitToBytes
            + DeserializeField<F>
            + UnitToField<F>
            + DigestReader<Hash<F, u8, DIGEST_ELEMS>>,
    {
        let root = verifier_state.read_digest()?;

        let mut ood_points = vec![F::ZERO; self.0.committment_ood_samples];
        let mut ood_answers = vec![F::ZERO; self.0.committment_ood_samples];
        if self.0.committment_ood_samples > 0 {
            verifier_state.fill_challenge_scalars(&mut ood_points)?;
            verifier_state.fill_next_scalars(&mut ood_answers)?;
        }

        Ok(ParsedCommitment { root, ood_points, ood_answers })
    }
}
