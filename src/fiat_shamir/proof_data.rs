type Rounds<R> = Vec<R>;
type SumcheckRounds<R> = Vec<R>;

#[derive(Debug, Clone, Default)]
pub struct ProofData<EF, F, W, const DIGEST_ELEMS: usize> {
    pub(crate) deferred_constraints: Vec<EF>,
    pub(crate) base_field_merkle_answers: Vec<Vec<F>>,
    pub(crate) round_merkle_answers: Rounds<Vec<Vec<EF>>>,
    pub(crate) round_merkle_proof: Rounds<Vec<Vec<[W; DIGEST_ELEMS]>>>,
    pub(crate) round_merkle_root: Rounds<[W; DIGEST_ELEMS]>,
    pub(crate) round_ood_answers: Rounds<Vec<EF>>,
    pub(crate) sumcheck_evaluations: Rounds<SumcheckRounds<Vec<EF>>>,
    pub(crate) final_folded_evaluations: Vec<EF>,
    pub(crate) pow_witnesses: Vec<F>,
}
