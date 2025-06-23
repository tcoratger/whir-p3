type Rounds<R> = Vec<R>;
type SumcheckRounds<R> = Vec<R>;

#[derive(Debug, Clone)]
pub struct ProofData<EF, F, W, const DIGEST_ELEMS: usize> {
    pub(crate) deferred_constraints: Vec<EF>,

    pub(crate) base_field_merkle_answers: Vec<Vec<F>>,

    pub(crate) round_merkle_answers: Rounds<Vec<Vec<EF>>>,
    pub(crate) round_merkle_proof: Rounds<Vec<Vec<[W; DIGEST_ELEMS]>>>,
    pub(crate) round_merkle_root: Rounds<[W; DIGEST_ELEMS]>,
    pub(crate) round_ood_answers: Rounds<Vec<EF>>,

    pub(crate) sumcheck_evaluations: Rounds<SumcheckRounds<Vec<EF>>>,

    pub(crate) final_folded_evaluations: Vec<EF>,
}

impl<EF, F, W, const DIGEST_ELEMS: usize> Default for ProofData<EF, F, W, DIGEST_ELEMS>
where
    W: Default + Copy,
{
    fn default() -> Self {
        Self {
            deferred_constraints: Vec::new(),

            base_field_merkle_answers: Vec::new(),

            round_merkle_answers: Vec::new(),
            round_merkle_proof: Vec::new(),
            round_merkle_root: Vec::new(),
            round_ood_answers: Vec::new(),

            sumcheck_evaluations: Vec::new(),

            final_folded_evaluations: Vec::new(),
        }
    }
}
