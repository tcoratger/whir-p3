use alloc::{vec, vec::Vec};

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field, TwoAdicField};

use crate::{
    fiat_shamir::prover::ProverState,
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    sumcheck::{
        eq_state::SumcheckEqState,
        sumcheck_single::SumcheckSingle,
        sumcheck_small_value::{algorithm_5, fold_evals_with_challenges, svo_three_rounds},
    },
    whir::constraints::evaluator::Constraint,
};

/// Number of SVO rounds (first 3 rounds use special optimized algorithm).
///
/// This follows <https://eprint.iacr.org/2025/1117>.
pub(crate) const NUM_SVO_ROUNDS: usize = 3;

impl<F, EF> SumcheckSingle<F, EF>
where
    F: Field + Ord,
    EF: ExtensionField<F>,
{
    /// Compute a Sumcheck using the Small Value Optimization (SVO) for the first three rounds and
    /// Algorithm 5 (page 18) for the remaining rounds.
    /// See Algorithm 6 (page 19) in <https://eprint.iacr.org/2025/1117>.
    pub fn from_base_evals_svo<Challenger>(
        evals: &EvaluationsList<F>,
        prover_state: &mut ProverState<F, EF, Challenger>,
        folding_factor: usize,
        pow_bits: usize,
        constraint: &Constraint<F, EF>,
    ) -> (Self, MultilinearPoint<EF>)
    where
        F: TwoAdicField,
        EF: TwoAdicField,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        assert_ne!(folding_factor, 0);
        let mut challenges = Vec::with_capacity(folding_factor);

        // Here we are assuming the the equality statement has only one constraint.
        // TODO: Handle multiple constraints for a general WHIR implementation.
        let mut sum = constraint.eq_statement.evaluations[0];
        let w = &constraint.eq_statement.points[0];

        // Create the unified equality polynomial evaluator with precomputed tables
        let mut eq_poly = SumcheckEqState::<_, NUM_SVO_ROUNDS>::new(w);

        svo_three_rounds(
            prover_state,
            evals,
            w,
            &mut eq_poly,
            &mut challenges,
            &mut sum,
            pow_bits,
        );

        // We fold to obtain p(r1, r2, r3, x).
        let mut folded_evals = fold_evals_with_challenges(evals, &challenges);

        algorithm_5(
            prover_state,
            &mut folded_evals,
            &mut eq_poly,
            &mut challenges,
            &mut sum,
            pow_bits,
        );

        let challenge_point = MultilinearPoint::new(challenges);

        // Final weight: eq(w, r)
        let weights = EvaluationsList::new(vec![w.eq_poly(&challenge_point)]);

        let sumcheck = Self::new(folded_evals, weights, sum);

        (sumcheck, challenge_point)
    }
}
