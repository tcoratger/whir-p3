use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field, TwoAdicField};

use crate::{
    fiat_shamir::prover::ProverState,
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    sumcheck::{
        sumcheck_single::SumcheckSingle,
        sumcheck_small_value::{
            algorithm_5, fold_evals_with_challenges,
            svo_three_rounds,
        },
    },
    whir::statement::Statement,
};

impl<F, EF> SumcheckSingle<F, EF>
where
    F: Field + Ord,
    EF: ExtensionField<F>,
{
    // Compute a Sumcheck using the Small Value Optimization (SVO) for the first three rounds and
    // Algorithm 5 (page 18) for the remaining rounds.
    // See Algorithm 6 (page 19) in <https://eprint.iacr.org/2025/1117>.
    pub fn from_base_evals_svo<Challenger>(
        evals: &EvaluationsList<F>,
        statement: &Statement<EF>,
        combination_randomness: EF,
        prover_state: &mut ProverState<F, EF, Challenger>,
        folding_factor: usize,
        pow_bits: usize,
    ) -> (Self, MultilinearPoint<EF>)
    where
        F: TwoAdicField,
        EF: TwoAdicField,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        assert_ne!(folding_factor, 0);
        let mut challenges = Vec::with_capacity(folding_factor);

        // TODO: Since we are assuming that `statement` has only one constraint we wouldn't need this
        //`combine` nor the `combination_randomness` argument.
        let (_, mut sum) = statement.combine::<F>(combination_randomness);

        // Here we are assuming the the statemas has only one constraint.
        let w = &statement.constraints[0].point.0;

        let (r_1, r_2, r_3) =
            svo_three_rounds(prover_state, evals, &w, &mut sum);
        challenges.extend_from_slice(&[r_1, r_2, r_3]);

        prover_state.pow_grinding(pow_bits);


        // We fold to obtaind p(r1, r2, r3, x).
        let mut folded_evals = fold_evals_with_challenges(evals, &challenges);

        algorithm_5(
            prover_state,
            &mut folded_evals,
            &w,
            &mut challenges,
            &mut sum,
        );

        let challenge_point = MultilinearPoint::new(challenges);

        // Final weight: eq(w, r)
        let weights = EvaluationsList::new(vec![w.eq_poly(&challenge_point)]);

        let sumcheck = Self::new(folded_evals, weights, sum);

        (sumcheck, challenge_point)
    }
}
