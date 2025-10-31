use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_maybe_rayon::prelude::*;

use crate::{
    fiat_shamir::prover::ProverState,
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    sumcheck::{
        sumcheck_single::{SumcheckSingle, compute_sumcheck_polynomial},
        sumcheck_small_value::{
            NUM_OF_ROUNDS, algorithm_5, fold_evals_with_challenges,
            small_value_sumcheck_three_rounds_eq,
        },
    },
    whir::statement::Statement,
};

/// This function is the same as in sumcheck_singl.rs. We copy it here because it was private.
///
/// Executes a standard, intermediate round of the sumcheck protocol.
///
/// This function executes a standard, intermediate round of the sumcheck protocol. Unlike the initial round,
/// it operates entirely within the extension field `EF`. It computes the sumcheck polynomial from the
/// current evaluations and weights, adds it to the transcript, gets a new challenge from the verifier,
/// and then compresses both the polynomial and weight evaluations in-place.
///
/// ## Arguments
/// * `prover_state` - A mutable reference to the `ProverState`, managing the Fiat-Shamir transcript.
/// * `evals` - A mutable reference to the polynomial's evaluations in `EF`, which will be compressed.
/// * `weights` - A mutable reference to the weight evaluations in `EF`, which will also be compressed.
/// * `sum` - A mutable reference to the claimed sum, updated after folding.
/// * `pow_bits` - The number of proof-of-work bits for grinding.
///
/// ## Returns
/// The verifier's challenge `r` as an `EF` element.
fn round<Challenger, F: Field, EF: ExtensionField<F>>(
    prover_state: &mut ProverState<F, EF, Challenger>,
    evals: &mut EvaluationsList<EF>,
    weights: &mut EvaluationsList<EF>,
    sum: &mut EF,
    pow_bits: usize,
) -> EF
where
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    // Compute the quadratic sumcheck polynomial for the current variable.
    let sumcheck_poly = compute_sumcheck_polynomial(evals, weights, *sum);
    prover_state.add_extension_scalar(sumcheck_poly.evaluations()[0]);
    prover_state.add_extension_scalar(sumcheck_poly.evaluations()[2]);

    prover_state.pow_grinding(pow_bits);

    // Sample verifier challenge.
    let r: EF = prover_state.sample();

    // Compress polynomials and update the sum.
    join(|| evals.compress(r), || weights.compress(r));

    *sum = sumcheck_poly.evaluate_on_standard_domain(&MultilinearPoint::new(vec![r]));

    r
}

impl<F, EF> SumcheckSingle<F, EF>
where
    F: Field + Ord,
    EF: ExtensionField<F>,
{
    // - SVO for the first three rounds.
    // - Algorithm 5 for the remaining rounds.
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

        let (_, mut sum) = statement.combine::<F>(combination_randomness);

        // We assume the the statemas has only one constraint.
        let w = statement.constraints[0].point.0.clone();

        let (r_1, r_2, r_3) =
            small_value_sumcheck_three_rounds_eq(prover_state, evals, &w, &mut sum);
        challenges.push(r_1);
        challenges.push(r_2);
        challenges.push(r_3);
        prover_state.pow_grinding(pow_bits);



        let mut folded_evals = fold_evals_with_challenges(evals, &challenges);

        algorithm_5(
            prover_state,
            &mut folded_evals,
            &w,
            &mut challenges,
            &mut sum,
        );

        // Final weight: eq(w, r).
        let weights =
            EvaluationsList::new(vec![w.eq_poly(&MultilinearPoint::new(challenges.clone()))]);

        let sumcheck = Self::new(folded_evals, weights, sum);

        (sumcheck, MultilinearPoint::new(challenges))
    }
}
