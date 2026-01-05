use alloc::vec::Vec;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field, TwoAdicField};

use crate::{
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    sumcheck::{
        eq_state::SumcheckEqState,
        product_polynomial::ProductPolynomial,
        sumcheck_single::SumcheckSingle,
        sumcheck_small_value::{algorithm_5, svo_first_rounds, svo_first_rounds_batched},
    },
    whir::{constraints::Constraint, proof::SumcheckData},
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
        sumcheck_data: &mut SumcheckData<EF, F>,
        challenger: &mut Challenger,
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

        svo_first_rounds(
            sumcheck_data,
            challenger,
            evals,
            w,
            &mut eq_poly,
            &mut challenges,
            &mut sum,
            pow_bits,
        );

        // We fold to obtain p(r1, r2, r3, x).
        let mut folded_evals = evals.fold_batch(&challenges);

        algorithm_5(
            sumcheck_data,
            challenger,
            &mut folded_evals,
            &mut eq_poly,
            &mut challenges,
            &mut sum,
            pow_bits,
            folding_factor,
        );

        let challenge_point = MultilinearPoint::new(challenges);

        // Final weight: eq(w, r) split into scalar and remaining polynomial
        //
        // When num_variables > folding_factor, the eq polynomial splits as:
        //   eq(w, x) = eq(w[0..folding_factor], r) * eq(w[folding_factor..], x_remaining)
        //
        // The first part is a scalar (computed from challenges), the second part
        // is a polynomial over the remaining variables.
        let w_bound = &w.0[..folding_factor];
        let w_remaining = &w.0[folding_factor..];
        let scalar_part = MultilinearPoint::eval_eq(w_bound, challenge_point.as_slice());
        let weights = EvaluationsList::new_from_point(w_remaining, scalar_part);

        (
            Self {
                poly: ProductPolynomial::new(folded_evals, weights),
                sum,
            },
            challenge_point,
        )
    }

    /// Compute a sumcheck using the Batched SVO for multiple equality constraints.
    ///
    /// # Algorithm
    ///
    /// 1. Pre-computes combined weight polynomial W = Σ γ^i * eq(z_i, X)
    /// 2. **Round 1 only**: Uses SVO accumulators with M_BE multiplications
    /// 3. **Rounds 2-3**: Standard sumcheck on folded W and p (M_EE operations)
    /// 4. **Rounds 4+**: Standard sumcheck continues
    ///
    /// # Why only round 1 uses SVO accumulators?
    ///
    /// Unlike single-constraint SVO where W = eq(w, x) factors as l(x_i) * t(suffix),
    /// the batched weight W = Σ γ^i * eq(z_i, X) doesn't have this structure.
    ///
    /// After folding with challenge r_1, cross-terms appear:
    /// - W'(x) * p'(x) involves W(0,x)*p(1,x) and W(1,x)*p(0,x)
    /// - These cross-terms are NOT captured by pre-computed accumulators
    ///
    /// So we use accumulators for round 1, then fold W and p properly for rounds 2+.
    ///
    /// # Performance
    ///
    /// - Round 1: O(2^l) M_BE operations (base × extension) - cheaper than M_EE
    /// - Rounds 2+: O(2^{l-i}) M_EE operations per round
    ///
    /// # Arguments
    ///
    /// * `evals` - Base field polynomial evaluations p(x)
    /// * `sumcheck_data` - Output: polynomial evaluations and PoW witnesses
    /// * `challenger` - Fiat-Shamir challenger for randomness
    /// * `folding_factor` - Total number of sumcheck rounds
    /// * `pow_bits` - PoW difficulty (0 to skip grinding)
    /// * `constraint` - Constraint containing multiple equality constraints
    ///
    /// # Returns
    ///
    /// A tuple of (SumcheckSingle, MultilinearPoint) containing:
    /// - the final sumcheck state,
    /// - the challenge point for subsequent verification.
    pub fn from_base_evals_svo_batched<Challenger>(
        evals: &EvaluationsList<F>,
        sumcheck_data: &mut SumcheckData<EF, F>,
        challenger: &mut Challenger,
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
        assert!(
            folding_factor >= NUM_SVO_ROUNDS,
            "folding factor must be at least {NUM_SVO_ROUNDS} for batched SVO"
        );

        let mut challenges = Vec::with_capacity(folding_factor);

        // Compute combined weight polynomial W = Σ γ^i * eq(z_i, X) and expected sum
        let (weights, mut sum) = constraint.combine_new();

        // Run batched SVO for first NUM_SVO_ROUNDS rounds
        //
        // This uses M_BE multiplications for round 1 (since p is still in base field),
        // then switches to standard sumcheck for rounds 2-3 with properly folded polynomials.
        //
        // Returns the folded polynomials after all NUM_SVO_ROUNDS rounds.
        let (mut folded_evals, mut folded_weights) = svo_first_rounds_batched(
            sumcheck_data,
            challenger,
            evals,
            &weights,
            &mut challenges,
            &mut sum,
            pow_bits,
        );

        // Continue with standard sumcheck for remaining rounds
        //
        // This is similar to Algorithm 5 but without the eq_state structure
        for _ in NUM_SVO_ROUNDS..folding_factor {
            // Compute (c_0, c_2) coefficients using standard sumcheck
            let (c_0, c_2) = folded_evals.sumcheck_coefficients(&folded_weights);

            // Send to verifier and get challenge
            sumcheck_data.polynomial_evaluations.push([c_0, c_2]);
            challenger.observe_algebra_slice(&[c_0, c_2]);

            if pow_bits > 0 {
                sumcheck_data.push_pow_witness(challenger.grind(pow_bits));
            }

            let r: EF = challenger.sample_algebra_element();
            challenges.push(r);

            // Fold for next round
            folded_evals.compress(r);
            folded_weights.compress(r);

            // Update sum
            let eval_1 = sum - c_0;
            sum = c_2 * r.square() + (eval_1 - c_0 - c_2) * r + c_0;
        }

        let challenge_point = MultilinearPoint::new(challenges);

        (
            Self {
                poly: ProductPolynomial::new(folded_evals, folded_weights),
                sum,
            },
            challenge_point,
        )
    }
}
