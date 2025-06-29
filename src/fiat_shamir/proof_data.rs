type Rounds<R> = Vec<R>;
type SumcheckRounds<R> = Vec<R>;

#[derive(Debug, Clone, Default)]
pub struct ProofData<EF, F, W, const DIGEST_ELEMS: usize> {
    pub deferred_constraints: Vec<EF>,
    pub base_field_merkle_answers: Vec<Vec<F>>,
    pub round_merkle_answers: Rounds<Vec<Vec<EF>>>,
    pub round_merkle_proof: Rounds<Vec<Vec<[W; DIGEST_ELEMS]>>>,
    pub round_merkle_root: Rounds<[W; DIGEST_ELEMS]>,
    pub round_ood_answers: Rounds<Vec<EF>>,
    pub sumcheck_evaluations: Rounds<SumcheckRounds<Vec<EF>>>,
    pub final_folded_evaluations: Vec<EF>,
    pub pow_witnesses: Vec<F>,
}
