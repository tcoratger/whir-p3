use alloc::{vec, vec::Vec};

use p3_field::{ExtensionField, Field, TwoAdicField};

use crate::{
    fiat_shamir::{
        errors::FiatShamirError,
        transcript::{Challenge, Pow, Writer},
    },
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    sumcheck::{
        eq_state::SumcheckEqState,
        product_polynomial::ProductPolynomial,
        sumcheck_single::SumcheckSingle,
        sumcheck_small_value::{algorithm_5, svo_first_rounds},
    },
    whir::constraints::Constraint,
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
    pub fn from_base_evals_svo<Transcript>(
        transcript: &mut Transcript,
        evals: &EvaluationsList<F>,
        folding_factor: usize,
        pow_bits: usize,
        constraint: &Constraint<F, EF>,
    ) -> Result<(Self, MultilinearPoint<EF>), FiatShamirError>
    where
        F: TwoAdicField,
        EF: TwoAdicField,
        Transcript: Writer<EF> + Challenge<EF> + Pow<F>,
    {
        assert_ne!(folding_factor, 0);
        let mut challenges = Vec::with_capacity(folding_factor);

        // Here we are assuming the the equality statement has only one constraint.
        // TODO: Handle multiple constraints for a general WHIR implementation.
        let mut sum = constraint.eq_statement.evaluations[0];
        let w = &constraint.eq_statement.points[0];

        // Create the unified equality polynomial evaluator with precomputed tables
        let mut eq_poly = SumcheckEqState::<_, NUM_SVO_ROUNDS>::new(w);

        svo_first_rounds(
            transcript,
            evals,
            w,
            &mut eq_poly,
            &mut challenges,
            &mut sum,
            pow_bits,
        )?;

        // We fold to obtain p(r1, r2, r3, x).
        let mut folded_evals = evals.fold_batch(&challenges);

        algorithm_5(
            transcript,
            &mut folded_evals,
            &mut eq_poly,
            &mut challenges,
            &mut sum,
            pow_bits,
        )?;

        let challenge_point = MultilinearPoint::new(challenges);

        // Final weight: eq(w, r)
        let weights = EvaluationsList::new(vec![w.eq_poly(&challenge_point)]);

        Ok((
            Self {
                poly: ProductPolynomial::new(folded_evals, weights),
                sum,
            },
            challenge_point,
        ))
    }
}
