use alloc::{vec, vec::Vec};
use core::ops::{Add, AddAssign};

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field};
use p3_maybe_rayon::prelude::*;

use crate::{
    fiat_shamir::grinding::pow_grinding,
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    sumcheck::{eq_state::SumcheckEqState, sumcheck_single_svo::NUM_SVO_ROUNDS},
    whir::proof::SumcheckData,
};

/// A container for the SVO accumulators for a specific number of rounds `N`.
///
/// Structure:
/// The accumulators are stored in a single flat vector for cache locality.
/// - Round 0 starts at index 0 with size 2^1.
/// - Round i starts at index (2^(i+1) - 2) with size 2^(i+1).
///
/// Total size = \sum_{i=0}^{N-1} 2^{i+1} = 2^{N+1} - 2.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SvoAccumulators<F: Field, const N: usize>(pub(crate) Vec<F>);

impl<F, const N: usize> Default for SvoAccumulators<F, N>
where
    F: Field,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F, const N: usize> SvoAccumulators<F, N>
where
    F: Field,
{
    /// Creates a new empty set of accumulators with pre-allocated capacity.
    #[must_use]
    pub fn new() -> Self {
        // Total size is 2 + 4 + ... + 2^N = 2^{N+1} - 2
        Self(F::zero_vec((1 << (N + 1)) - 2))
    }

    /// Adds a value to a specific accumulator.
    pub fn accumulate(&mut self, round: usize, index: usize, value: F) {
        let start = (1 << (round + 1)) - 2;
        self.0[start + index] += value;
    }

    /// Gets the slice of accumulators for a given round (safe version).
    #[must_use]
    pub fn at_round(&self, round: usize) -> &[F] {
        assert!(
            round < N,
            "round index out of bounds: round={round} but N={N}"
        );
        let start = (1 << (round + 1)) - 2;
        let len = 1 << (round + 1);
        &self.0[start..start + len]
    }

    /// Returns a mutable slice for a specific round.
    ///
    /// # Safety
    /// The caller must ensure `round < N`.
    pub unsafe fn at_round_unchecked_mut(&mut self, round: usize) -> &mut [F] {
        let start = (1 << (round + 1)) - 2;
        let len = 1 << (round + 1);
        unsafe { self.0.get_unchecked_mut(start..start + len) }
    }
}

impl<F: Field, const N: usize> Add for SvoAccumulators<F, N> {
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<F: Field, const N: usize> AddAssign for SvoAccumulators<F, N> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0
            .iter_mut()
            .zip(rhs.0.iter())
            .for_each(|(l, r)| *l += *r);
    }
}

/// Algorithm 6. Page 19.
/// Compute three sumcheck rounds using the small value optimization and split-eq accumulators.
#[allow(clippy::too_many_arguments)]
pub fn svo_first_rounds<Challenger, F: Field, EF: ExtensionField<F>>(
    sumcheck_data: &mut SumcheckData<EF, F>,
    challenger: &mut Challenger,
    poly: &EvaluationsList<F>,
    w: &MultilinearPoint<EF>,
    eq_poly: &mut SumcheckEqState<'_, EF, NUM_SVO_ROUNDS>,
    challenges: &mut Vec<EF>,
    sum: &mut EF,
    pow_bits: usize,
) where
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    let (e_in, e_out) = join(
        || w.svo_e_in_table::<NUM_SVO_ROUNDS>(),
        || w.svo_e_out_tables::<NUM_SVO_ROUNDS>(),
    );

    // We compute all the accumulators A_i(v, u).
    // let accumulators = compute_accumulators(poly, &e_in, &e_out);
    let accumulators = poly.compute_svo_accumulators(&e_in, &e_out);

    // ------------------   Round 1   ------------------

    // 1. For u in {0, 1} compute t_1(u)
    // Recall: In round 1, t_1(u) = A_1(u).
    let t_1_evals = accumulators.at_round(0);

    // 2. For u in {0, 1, inf} compute S_1(u) = t_1(u) * l_1(u).

    // We compute l_1(0) and l_1(1) using the cached state
    let linear_1_evals = eq_poly.current_linear_evals();

    // Compute S_1(u) = t_1(u) * l_1(u) for u in {0, inf}
    // - S_1(0) = t_1(0) * l_1(0)
    // - S_1(inf) = (t_1(1) - t_1(0)) * (l_1(1) - l_1(0))
    let s_0 = t_1_evals[0] * linear_1_evals[0];
    let s_inf = (t_1_evals[1] - t_1_evals[0]) * (linear_1_evals[1] - linear_1_evals[0]);

    // 3. Send S_1(u) to the verifier.
    sumcheck_data.polynomial_evaluations.push([s_0, s_inf]);
    challenger.observe_slice(&EF::flatten_to_base(vec![s_0, s_inf]));

    sumcheck_data.push_pow_witness(pow_grinding(challenger, pow_bits));

    // 4. Receive the challenge r_1 from the verifier.
    let r_1: EF = challenger.sample_algebra_element();
    challenges.push(r_1);
    eq_poly.bind(r_1);

    let s_1 = *sum - s_0;
    *sum = s_inf * r_1.square() + (s_1 - s_0 - s_inf) * r_1 + s_0;

    // 5. Compte R_2 = [L_0(r_1), L_1(r_1), L_inf(r_1)]
    // L_0 (x) = 1 - x
    // L_1 (x) = x
    let lagrange_evals_r_1 = [-r_1 + F::ONE, r_1];

    // ------------------  Round 2  ------------------

    // 1. For u in {0, 1} compute t_2(u).
    // First we take the accumulators A_2(v, u).
    // We computed 4 accumulators for round 2, since only need v in {0, 1} and u in {0, 1}.
    let accumulators_round_2 = accumulators.at_round(1);

    let t_2_evals = [
        // t_2(u=0) = L_0(r_1) * A(v=0, u=0) + L_1(r_1) * A(v=1, u=0)
        lagrange_evals_r_1[0] * accumulators_round_2[0]
            + lagrange_evals_r_1[1] * accumulators_round_2[2],
        // t_2(u=1) = L_0(r_1) * A(v=0, u=1) + L_1(r_1) * A(v=1, u=1)
        lagrange_evals_r_1[0] * accumulators_round_2[1]
            + lagrange_evals_r_1[1] * accumulators_round_2[3],
    ];

    // We compute l_2(0) and l_2(inf) using the cached state
    let linear_2_evals = eq_poly.current_linear_evals();

    // Compute S_2(u) = t_2(u) * l_2(u) for u in {0, inf}
    // - S_2(0) = t_2(0) * l_2(0)
    // - S_2(inf) = (t_2(1) - t_2(0)) * (l_2(1) - l_2(0))
    let s_0 = t_2_evals[0] * linear_2_evals[0];
    let s_inf = (t_2_evals[1] - t_2_evals[0]) * (linear_2_evals[1] - linear_2_evals[0]);

    // 3. Send S_2(u) to the verifier.
    sumcheck_data.polynomial_evaluations.push([s_0, s_inf]);
    challenger.observe_slice(&EF::flatten_to_base(vec![s_0, s_inf]));

    sumcheck_data.push_pow_witness(pow_grinding(challenger, pow_bits));

    // 4. Receive the challenge r_2 from the verifier.
    let r_2: EF = challenger.sample_algebra_element();
    challenges.push(r_2);
    eq_poly.bind(r_2);

    // 5. Compute R_3 = [L_00(r_1, r_2), L_01(r_1, r_2), ..., L_{inf inf}(r_1, r_2)]
    // L_00 (x1, x2) = (1 - x1) * (1 - x2)
    // L_01 (x1, x2) = (1 - x1) * x2
    // ...
    // L_{inf inf} (x1, x2) = (x1 - 1) * x1 * (x2 - 1) * x2

    let [l_0, l_1] = lagrange_evals_r_1;
    let one_minus_r_2 = -r_2 + F::ONE;

    let lagrange_evals_r_2 = [
        l_0 * one_minus_r_2, // L_0 0
        l_0 * r_2,           // L_0 1
        l_1 * one_minus_r_2, // L_1 0
        l_1 * r_2,           // L_1 1
    ];

    let s_1 = *sum - s_0;
    *sum = s_inf * r_2.square() + (s_1 - s_0 - s_inf) * r_2 + s_0;
    // ------------------  Round 3  ------------------

    // 1. For u in {0, 1} compute t_3(u).

    // First we take the accumulators A_3(v, u).
    // We computed 4 accumulators at the third round for v in {0, 1}^2 and u in {0, 1}.
    let accumulators_round_3 = accumulators.at_round(2);

    let t_3_evals: [EF; 2] = [
        // t_3(u=0) = Σ_{v} L_v(r_1, r_2) * A(v, u=0)
        lagrange_evals_r_2
            .iter()
            .zip(accumulators_round_3.iter().step_by(2))
            .map(|(l, a)| *l * *a)
            .sum(),
        // t_3(u=1) = Σ_{v} L_v(r_1, r_2) * A(v, u=1)
        lagrange_evals_r_2
            .iter()
            .zip(accumulators_round_3.iter().skip(1).step_by(2))
            .map(|(l, a)| *l * *a)
            .sum(),
    ];

    // 2. For u in {0, 1, inf} compute S_3(u) = t_3(u) * l_3(u).

    // We compute l_3(0) and l_3(inf) using the cached state
    let linear_3_evals = eq_poly.current_linear_evals();

    // Compute S_3(u) = t_3(u) * l_3(u) for u in {0, inf}
    // - S_3(0) = t_3(0) * l_3(0)
    // - S_3(inf) = (t_3(1) - t_3(0)) * (l_3(1) - l_3(0))
    let round_poly_evals = [
        t_3_evals[0] * linear_3_evals[0],
        (t_3_evals[1] - t_3_evals[0]) * (linear_3_evals[1] - linear_3_evals[0]),
    ];

    // 3. Send S_3(u) to the verifier.
    sumcheck_data.polynomial_evaluations.push(round_poly_evals);
    challenger.observe_slice(&EF::flatten_to_base(vec![
        round_poly_evals[0],
        round_poly_evals[1],
    ]));

    sumcheck_data.push_pow_witness(pow_grinding(challenger, pow_bits));

    // 4. Receive the challenge r_3 from the verifier.
    let r_3: EF = challenger.sample_algebra_element();
    challenges.push(r_3);
    eq_poly.bind(r_3);

    let eval_1 = *sum - round_poly_evals[0];
    *sum = round_poly_evals[1] * r_3.square()
        + (eval_1 - round_poly_evals[0] - round_poly_evals[1]) * r_3
        + round_poly_evals[0];
}

/// Computes the round polynomial evaluations `t_i(u)` for a single standard sumcheck round.
///
/// This is a pure function that computes `t_i(0)` and `t_i(1)` using precomputed
/// equality polynomial tables, without interacting with prover state.
///
/// ## Arguments
///
/// * `poly_slice`: The current polynomial evaluations
/// * `eq_l`: Left equality table (or tail table for rounds > l/2)
/// * `eq_r`: Right equality table (empty for rounds > l/2)
/// * `num_vars_x_r`: Number of variables in x_R
/// * `half_l`: l/2, used to determine which algorithm phase we're in
/// * `round`: Current round number (1-indexed)
///
/// ## Returns
///
/// `[t_i(0), t_i(1)]` - the evaluations of the round polynomial at 0 and 1.
pub fn compute_standard_round_poly_evals<EF>(
    poly: &EvaluationsList<EF>,
    eq_l: &[EF],
    eq_r: &[EF],
    num_vars_x_r: usize,
    half_l: usize,
    round: usize,
) -> [EF; 2]
where
    EF: ExtensionField<EF> + Send + Sync,
{
    let num_vars_poly_current = poly.num_variables();
    let poly_slice = poly.as_slice();

    let (t0, t1) = if round <= half_l {
        // Case: round <= l/2
        //
        // We are folding the polynomial P(u, x_L, x_R) against the precomputed
        // equality tables eq_L(x_L) and eq_R(x_R).
        //
        // The polynomial variables are split as: u || x_L || x_R
        //
        // Formula:
        // t_i(u) = ∑_{x_R} eq_R(x_R) · ( ∑_{x_L} eq_L(x_L) · P(u, x_L, x_R) )
        //
        // We define a closure to compute this sum for a specific 'u' (determined by `offset`).
        let compute_half = |offset: usize| -> EF {
            // Parallelize over the outer summation (x_R)
            (0..eq_r.len())
                .into_par_iter()
                .map(|x_r| {
                    // Inner summation (x_L) is sequential for cache locality
                    let sum_l: EF = (0..eq_l.len())
                        .map(|x_l| {
                            // Construct the flat index into the polynomial vector.
                            // The bits are arranged as: [ u_bit |  x_L_bits  |  x_R_bits ]
                            //                           ^       ^            ^
                            //                           |       |            |
                            // offset -------------------+       |            |
                            // (x_l << num_vars_x_r) ------------+            |
                            // x_r -------------------------------------------+
                            let idx = offset | (x_l << num_vars_x_r) | x_r;

                            eq_l[x_l] * poly_slice[idx]
                        })
                        .sum();

                    eq_r[x_r] * sum_l
                })
                .sum()
        };

        // Compute t_i(0) and t_i(1) in parallel
        join(
            || compute_half(0),                                // u = 0
            || compute_half(1 << (num_vars_poly_current - 1)), // u = 1
        )
    } else {
        // Case round > l/2: Use eq_tail (passed as eq_l)
        let half_size = 1 << (num_vars_poly_current - 1);

        let poly_slice_l = &poly_slice[..half_size];
        let poly_slice_r = &poly_slice[half_size..];

        debug_assert_eq!(eq_l.len(), poly_slice_l.len());
        debug_assert_eq!(eq_l.len(), poly_slice_r.len());

        join(
            || {
                // t_i(0): Dot product of eq_tail with the first half of poly
                eq_l.par_iter()
                    .zip(poly_slice_l.par_iter())
                    .map(|(&e, &p)| e * p)
                    .sum()
            },
            || {
                // t_i(1): Dot product of eq_tail with the second half of poly
                eq_l.par_iter()
                    .zip(poly_slice_r.par_iter())
                    .map(|(&e, &p)| e * p)
                    .sum()
            },
        )
    };

    [t0, t1]
}

/// Algorithm 5. Page 18.
/// Compute the remaining sumcheck rounds, from round l0 + 1 to round l.
pub fn algorithm_5<Challenger, F, EF>(
    sumcheck_data: &mut SumcheckData<EF, F>,
    challenger: &mut Challenger,
    poly: &mut EvaluationsList<EF>,
    eq_poly: &mut SumcheckEqState<'_, EF, NUM_SVO_ROUNDS>,
    challenges: &mut Vec<EF>,
    sum: &mut EF,
    pow_bits: usize,
) where
    F: Field,
    EF: ExtensionField<F> + Send + Sync,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    let num_vars = eq_poly.num_variables();
    // Current position in the sumcheck
    let start_round = eq_poly.bound_count();
    challenges.reserve(num_vars - start_round);

    // Main loop: compute rounds from start_round to num_vars-1
    for i in start_round..num_vars {
        let round = i + 1;

        // Get precomputed tables from unified struct
        let (eq_l, eq_r) = eq_poly.current_t_poly_tables();
        let num_vars_x_r = eq_poly.num_vars_x_r();
        let half_l = eq_poly.half_l();

        // Compute t_i(u) for u in {0, 1}
        let t_evals =
            compute_standard_round_poly_evals(poly, eq_l, eq_r, num_vars_x_r, half_l, round);

        // Compute S_i(u) = t_i(u) * l_i(u) for u in {0, inf}
        // - S_i(0) = t_i(0) * l_i(0)
        // - S_i(inf) = (t_i(1) - t_i(0)) * (l_i(1) - l_i(0))
        let linear_evals = eq_poly.current_linear_evals();
        let s_0 = t_evals[0] * linear_evals[0];
        let s_inf = (t_evals[1] - t_evals[0]) * (linear_evals[1] - linear_evals[0]);

        // Send S_i(u) to the verifier
        sumcheck_data.polynomial_evaluations.push([s_0, s_inf]);
        challenger.observe_slice(&EF::flatten_to_base(vec![s_0, s_inf]));
        sumcheck_data.push_pow_witness(pow_grinding(challenger, pow_bits));

        // Receive the challenge r_i from the verifier
        let r_i: EF = challenger.sample_algebra_element();
        challenges.push(r_i);

        // Update state for next round: binding updates scalar AND pops used table
        eq_poly.bind(r_i);
        poly.compress(r_i);

        // Update claimed sum
        let eval_1 = *sum - s_0;
        *sum = s_inf * r_i.square() + (eval_1 - s_0 - s_inf) * r_i + s_0;
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};

    use super::*;
    use crate::poly::evals::EvaluationsList;

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    #[test]
    fn test_evals_eq_in_hypercube_three_vars_matches_new_from_point() {
        let p = vec![F::from_u64(2), F::from_u64(3), F::from_u64(5)];
        let value = F::from_u64(1);

        let expected = EvaluationsList::new_from_point(&p, value)
            .into_iter()
            .collect::<Vec<_>>();
        let result = EvaluationsList::new_from_point(&p, value)
            .into_iter()
            .collect::<Vec<_>>();

        assert_eq!(expected, result);
    }

    #[test]
    fn test_eq_evals() {
        // r = [p0, p1, p2]
        let p0 = F::from_u64(2);
        let p1 = F::from_u64(3);
        let p2 = F::from_u64(5);

        // Indices: 000, 001, 010, 011, 100, 101, 110, 111
        let expected = vec![
            (F::ONE - p0) * (F::ONE - p1) * (F::ONE - p2), // 000 v[0]
            (F::ONE - p0) * (F::ONE - p1) * p2,            // 001 v[1]
            (F::ONE - p0) * p1 * (F::ONE - p2),            // 010 v[2]
            (F::ONE - p0) * p1 * p2,                       // 011 v[3]
            p0 * (F::ONE - p1) * (F::ONE - p2),            // 100 v[4]
            p0 * (F::ONE - p1) * p2,                       // 101 v[5]
            p0 * p1 * (F::ONE - p2),                       // 110 v[6]
            p0 * p1 * p2,                                  // 111 v[7]
        ];

        let out = EvaluationsList::new_from_point(&[p0, p1, p2], F::ONE).0;
        assert_eq!(out, expected);
    }

    #[test]
    fn test_accumulators_new_empty() {
        // SETUP:
        // Create a new empty accumulator structure for the SVO protocol.
        //
        // The SVO protocol uses NUM_SVO_ROUNDS rounds, where:
        // - Round 0 (i=0): 2^(0+1) = 2 accumulators for A_0(u) with u in {0, 1}
        // - Round 1 (i=1): 2^(1+1) = 4 accumulators for A_1(v, u) with v in {0, 1} and u in {0, 1}
        // - Round 2 (i=2): 2^(2+1) = 8 accumulators for A_2(v, u) with v in {0, 1}^2 and u in {0, 1}

        let accumulators = SvoAccumulators::<F, 3>::new();

        // VERIFY: Each round has the correct number of zero-initialized accumulators.

        // Round 0: 2 accumulators, all zero
        let round_0 = accumulators.at_round(0);
        assert_eq!(round_0.len(), 2, "Round 0 should have 2 accumulators");
        assert_eq!(round_0[0], F::ZERO, "Accumulator A_0(u=0) should be zero");
        assert_eq!(round_0[1], F::ZERO, "Accumulator A_0(u=1) should be zero");

        // Round 1: 4 accumulators, all zero
        let round_1 = accumulators.at_round(1);
        assert_eq!(round_1.len(), 4, "Round 1 should have 4 accumulators");
        for (i, &r1) in round_1.iter().enumerate().take(4) {
            assert_eq!(r1, F::ZERO, "Accumulator A_1[{i}] should be zero");
        }

        // Round 2: 8 accumulators, all zero
        let round_2 = accumulators.at_round(2);
        assert_eq!(round_2.len(), 8, "Round 2 should have 8 accumulators");
        for (i, &r2) in round_2.iter().enumerate().take(8) {
            assert_eq!(r2, F::ZERO, "Accumulator A_2[{i}] should be zero");
        }
    }

    #[test]
    fn test_accumulators_accumulate_single_round() {
        // SETUP:
        // Test accumulation on a single round to verify that values are correctly added.
        //
        // We focus on Round 1, which has 4 accumulators corresponding to:
        // - Index 0: A_1(v=0, u=0)
        // - Index 1: A_1(v=0, u=1)
        // - Index 2: A_1(v=1, u=0)
        // - Index 3: A_1(v=1, u=1)

        let mut accumulators = SvoAccumulators::<F, 3>::new();

        // Accumulate values at different indices in round 1
        let value_1 = F::from_u64(10); // Add 10 to A_1(v=0, u=0)
        let value_2 = F::from_u64(25); // Add 25 to A_1(v=1, u=1)
        let value_3 = F::from_u64(7); // Add 7 more to A_1(v=0, u=0)

        accumulators.accumulate(1, 0, value_1);
        accumulators.accumulate(1, 3, value_2);
        accumulators.accumulate(1, 0, value_3);

        // VERIFY: The accumulators contain the correct accumulated values.

        let round_1 = accumulators.at_round(1);

        // Index 0 should have 10 + 7 = 17
        assert_eq!(
            round_1[0],
            F::from_u64(17),
            "A_1(v=0, u=0) should accumulate to 17"
        );

        // Index 1 should remain zero (no accumulation)
        assert_eq!(round_1[1], F::ZERO, "A_1(v=0, u=1) should remain zero");

        // Index 2 should remain zero (no accumulation)
        assert_eq!(round_1[2], F::ZERO, "A_1(v=1, u=0) should remain zero");

        // Index 3 should have 25
        assert_eq!(
            round_1[3],
            F::from_u64(25),
            "A_1(v=1, u=1) should accumulate to 25"
        );
    }

    #[test]
    fn test_accumulators_accumulate_multiple_rounds() {
        // SETUP:
        // Test accumulation across multiple rounds to verify that each round maintains
        // its own independent set of accumulators.
        //
        // We accumulate values in all three rounds and verify they don't interfere.

        let mut accumulators = SvoAccumulators::<F, 3>::new();

        // Round 0: Accumulate at both indices
        // A_0(u=0) += 5, then += 3 → total 8
        // A_0(u=1) += 12
        accumulators.accumulate(0, 0, F::from_u64(5));
        accumulators.accumulate(0, 0, F::from_u64(3));
        accumulators.accumulate(0, 1, F::from_u64(12));

        // Round 1: Accumulate at specific indices
        // A_1(v=0, u=1) += 20
        // A_1(v=1, u=0) += 15
        accumulators.accumulate(1, 1, F::from_u64(20));
        accumulators.accumulate(1, 2, F::from_u64(15));

        // Round 2: Accumulate at the first and last indices
        // A_2[0] += 100
        // A_2[7] += 200
        accumulators.accumulate(2, 0, F::from_u64(100));
        accumulators.accumulate(2, 7, F::from_u64(200));

        // VERIFY: Each round's accumulators contain the expected values.

        // Check Round 0
        let round_0 = accumulators.at_round(0);
        assert_eq!(round_0[0], F::from_u64(8), "Round 0: A_0(u=0) should be 8");
        assert_eq!(
            round_0[1],
            F::from_u64(12),
            "Round 0: A_0(u=1) should be 12"
        );

        // Check Round 1
        let round_1 = accumulators.at_round(1);
        assert_eq!(round_1[0], F::ZERO, "Round 1: A_1(v=0, u=0) should be zero");
        assert_eq!(
            round_1[1],
            F::from_u64(20),
            "Round 1: A_1(v=0, u=1) should be 20"
        );
        assert_eq!(
            round_1[2],
            F::from_u64(15),
            "Round 1: A_1(v=1, u=0) should be 15"
        );
        assert_eq!(round_1[3], F::ZERO, "Round 1: A_1(v=1, u=1) should be zero");

        // Check Round 2
        let round_2 = accumulators.at_round(2);
        assert_eq!(
            round_2[0],
            F::from_u64(100),
            "Round 2: A_2[0] should be 100"
        );
        for (i, &r2) in round_2.iter().enumerate().take(7).skip(1) {
            assert_eq!(r2, F::ZERO, "Round 2: A_2[{i}] should be zero");
        }
        assert_eq!(
            round_2[7],
            F::from_u64(200),
            "Round 2: A_2[7] should be 200"
        );
    }

    #[test]
    fn test_accumulators_add() {
        // Create the first accumulator with specific values
        let mut acc_1 = SvoAccumulators::<F, 3>::new();

        // Thread 1 computed these accumulators from its chunk of the polynomial:
        // Round 0:
        acc_1.accumulate(0, 0, F::from_u64(10)); // A_0(u=0) = 10
        acc_1.accumulate(0, 1, F::from_u64(20)); // A_0(u=1) = 20

        // Round 1:
        acc_1.accumulate(1, 0, F::from_u64(5)); // A_1(v=0, u=0) = 5
        acc_1.accumulate(1, 1, F::from_u64(15)); // A_1(v=0, u=1) = 15
        acc_1.accumulate(1, 2, F::from_u64(25)); // A_1(v=1, u=0) = 25
        acc_1.accumulate(1, 3, F::from_u64(35)); // A_1(v=1, u=1) = 35

        // Round 2:
        acc_1.accumulate(2, 0, F::from_u64(100)); // A_2[0] = 100
        acc_1.accumulate(2, 3, F::from_u64(200)); // A_2[3] = 200
        acc_1.accumulate(2, 7, F::from_u64(300)); // A_2[7] = 300

        // Create the second accumulator with different values
        let mut acc_2 = SvoAccumulators::<F, 3>::new();

        // Thread 2 computed these accumulators from its chunk of the polynomial:
        // Round 0:
        acc_2.accumulate(0, 0, F::from_u64(3)); // A_0(u=0) = 3
        acc_2.accumulate(0, 1, F::from_u64(7)); // A_0(u=1) = 7

        // Round 1:
        acc_2.accumulate(1, 0, F::from_u64(2)); // A_1(v=0, u=0) = 2
        acc_2.accumulate(1, 1, F::from_u64(4)); // A_1(v=0, u=1) = 4
        acc_2.accumulate(1, 2, F::from_u64(6)); // A_1(v=1, u=0) = 6
        acc_2.accumulate(1, 3, F::from_u64(8)); // A_1(v=1, u=1) = 8

        // Round 2:
        acc_2.accumulate(2, 0, F::from_u64(50)); // A_2[0] = 50
        acc_2.accumulate(2, 1, F::from_u64(60)); // A_2[1] = 60
        acc_2.accumulate(2, 7, F::from_u64(70)); // A_2[7] = 70

        // COMBINE: Add the two accumulator structures using the `+` operator
        let result = acc_1 + acc_2;

        // VERIFY: Each element in the result is the sum of corresponding elements.

        // Check Round 0:
        // - result.A_0(u=0) = 10 + 3 = 13
        // - result.A_0(u=1) = 20 + 7 = 27
        let round_0 = result.at_round(0);
        assert_eq!(
            round_0[0],
            F::from_u64(13),
            "Round 0: A_0(u=0) = 10 + 3 = 13"
        );
        assert_eq!(
            round_0[1],
            F::from_u64(27),
            "Round 0: A_0(u=1) = 20 + 7 = 27"
        );

        // Check Round 1:
        // - result.A_1(v=0, u=0) = 5 + 2 = 7
        // - result.A_1(v=0, u=1) = 15 + 4 = 19
        // - result.A_1(v=1, u=0) = 25 + 6 = 31
        // - result.A_1(v=1, u=1) = 35 + 8 = 43
        let round_1 = result.at_round(1);
        assert_eq!(
            round_1[0],
            F::from_u64(7),
            "Round 1: A_1(v=0, u=0) = 5 + 2 = 7"
        );
        assert_eq!(
            round_1[1],
            F::from_u64(19),
            "Round 1: A_1(v=0, u=1) = 15 + 4 = 19"
        );
        assert_eq!(
            round_1[2],
            F::from_u64(31),
            "Round 1: A_1(v=1, u=0) = 25 + 6 = 31"
        );
        assert_eq!(
            round_1[3],
            F::from_u64(43),
            "Round 1: A_1(v=1, u=1) = 35 + 8 = 43"
        );

        // Check Round 2:
        // - result.A_2[0] = 100 + 50 = 150
        // - result.A_2[1] = 0 + 60 = 60    (acc_1 had zero here)
        // - result.A_2[2] = 0 + 0 = 0      (both had zero)
        // - result.A_2[3] = 200 + 0 = 200  (acc_2 had zero here)
        // - result.A_2[4] = 0 + 0 = 0      (both had zero)
        // - result.A_2[5] = 0 + 0 = 0      (both had zero)
        // - result.A_2[6] = 0 + 0 = 0      (both had zero)
        // - result.A_2[7] = 300 + 70 = 370
        let round_2 = result.at_round(2);
        assert_eq!(
            round_2[0],
            F::from_u64(150),
            "Round 2: A_2[0] = 100 + 50 = 150"
        );
        assert_eq!(
            round_2[1],
            F::from_u64(60),
            "Round 2: A_2[1] = 0 + 60 = 60 (only acc_2 contributed)"
        );
        assert_eq!(
            round_2[2],
            F::ZERO,
            "Round 2: A_2[2] = 0 + 0 = 0 (neither contributed)"
        );
        assert_eq!(
            round_2[3],
            F::from_u64(200),
            "Round 2: A_2[3] = 200 + 0 = 200 (only acc_1 contributed)"
        );
        assert_eq!(
            round_2[4],
            F::ZERO,
            "Round 2: A_2[4] = 0 + 0 = 0 (neither contributed)"
        );
        assert_eq!(
            round_2[5],
            F::ZERO,
            "Round 2: A_2[5] = 0 + 0 = 0 (neither contributed)"
        );
        assert_eq!(
            round_2[6],
            F::ZERO,
            "Round 2: A_2[6] = 0 + 0 = 0 (neither contributed)"
        );
        assert_eq!(
            round_2[7],
            F::from_u64(370),
            "Round 2: A_2[7] = 300 + 70 = 370"
        );
    }

    #[test]
    #[should_panic(expected = "round index out of bounds: round=3 but N=3")]
    fn test_at_round_panic_on_invalid_round() {
        // For N=3, valid rounds are 0, 1, 2. Accessing round 3 should panic.
        let accumulators = SvoAccumulators::<F, 3>::new();

        // This should panic with the expected message
        let _ = accumulators.at_round(3);
    }

    #[test]
    fn test_at_round_unchecked_mut_basic() {
        // For N=3:
        // - Round 0: 2 accumulators at indices 0..2
        // - Round 1: 4 accumulators at indices 2..6
        // - Round 2: 8 accumulators at indices 6..14

        let mut accumulators = SvoAccumulators::<F, 3>::new();

        // Modify Round 0 through the mutable slice
        unsafe {
            let round_0_mut = accumulators.at_round_unchecked_mut(0);
            assert_eq!(round_0_mut.len(), 2, "Round 0 should have 2 accumulators");

            round_0_mut[0] = F::from_u64(10);
            round_0_mut[1] = F::from_u64(20);
        }

        // Modify Round 1 through the mutable slice
        unsafe {
            let round_1_mut = accumulators.at_round_unchecked_mut(1);
            assert_eq!(round_1_mut.len(), 4, "Round 1 should have 4 accumulators");

            round_1_mut[0] = F::from_u64(30);
            round_1_mut[1] = F::from_u64(40);
            round_1_mut[2] = F::from_u64(50);
            round_1_mut[3] = F::from_u64(60);
        }

        // Modify Round 2 through the mutable slice
        unsafe {
            let round_2_mut = accumulators.at_round_unchecked_mut(2);
            assert_eq!(round_2_mut.len(), 8, "Round 2 should have 8 accumulators");

            round_2_mut[0] = F::from_u64(100);
            round_2_mut[7] = F::from_u64(200);
        }

        // VERIFY: Check that modifications were correctly applied
        let round_0 = accumulators.at_round(0);
        assert_eq!(round_0[0], F::from_u64(10), "Round 0[0] should be 10");
        assert_eq!(round_0[1], F::from_u64(20), "Round 0[1] should be 20");
        assert_eq!(round_0.len(), 2, "Round 0 should have 2 accumulators");

        let round_1 = accumulators.at_round(1);
        assert_eq!(round_1[0], F::from_u64(30), "Round 1[0] should be 30");
        assert_eq!(round_1[1], F::from_u64(40), "Round 1[1] should be 40");
        assert_eq!(round_1[2], F::from_u64(50), "Round 1[2] should be 50");
        assert_eq!(round_1[3], F::from_u64(60), "Round 1[3] should be 60");
        assert_eq!(round_1.len(), 4, "Round 1 should have 4 accumulators");

        let round_2 = accumulators.at_round(2);
        assert_eq!(round_2[0], F::from_u64(100), "Round 2[0] should be 100");
        assert_eq!(round_2[7], F::from_u64(200), "Round 2[7] should be 200");
        assert_eq!(round_2.len(), 8, "Round 2 should have 8 accumulators");
    }

    #[test]
    fn test_at_round_unchecked_mut_different_n() {
        // Test with N=2:
        // - Round 0: 2 accumulators (indices 0..2)
        // - Round 1: 4 accumulators (indices 2..6)

        let mut accumulators = SvoAccumulators::<F, 2>::new();

        // Set values in Round 0
        unsafe {
            let round_0_mut = accumulators.at_round_unchecked_mut(0);
            assert_eq!(
                round_0_mut.len(),
                2,
                "N=2: Round 0 should have 2 accumulators"
            );
            round_0_mut[0] = F::from_u64(5);
            round_0_mut[1] = F::from_u64(15);
        }

        // Set values in Round 1
        unsafe {
            let round_1_mut = accumulators.at_round_unchecked_mut(1);
            assert_eq!(
                round_1_mut.len(),
                4,
                "N=2: Round 1 should have 4 accumulators"
            );
            round_1_mut[0] = F::from_u64(25);
            round_1_mut[1] = F::from_u64(35);
            round_1_mut[2] = F::from_u64(45);
            round_1_mut[3] = F::from_u64(55);
        }

        // VERIFY: Check the values
        let round_0 = accumulators.at_round(0);
        assert_eq!(round_0[0], F::from_u64(5), "N=2: Round 0[0] should be 5");
        assert_eq!(round_0[1], F::from_u64(15), "N=2: Round 0[1] should be 15");
        assert_eq!(round_0.len(), 2, "N=2: Round 0 should have 2 accumulators");

        let round_1 = accumulators.at_round(1);
        assert_eq!(round_1[0], F::from_u64(25), "N=2: Round 1[0] should be 25");
        assert_eq!(round_1[1], F::from_u64(35), "N=2: Round 1[1] should be 35");
        assert_eq!(round_1[2], F::from_u64(45), "N=2: Round 1[2] should be 45");
        assert_eq!(round_1[3], F::from_u64(55), "N=2: Round 1[3] should be 55");
        assert_eq!(round_1.len(), 4, "N=2: Round 1 should have 4 accumulators");
    }

    #[test]
    fn test_add_and_add_assign_consistency() {
        // Create first accumulator with specific values
        let mut acc1_for_add = SvoAccumulators::<F, 3>::new();
        acc1_for_add.accumulate(0, 0, F::from_u64(10));
        acc1_for_add.accumulate(0, 1, F::from_u64(20));
        acc1_for_add.accumulate(1, 2, F::from_u64(30));
        acc1_for_add.accumulate(2, 5, F::from_u64(40));

        // Clone for AddAssign test
        let mut acc1_for_add_assign = acc1_for_add.clone();

        // Create second accumulator
        let mut acc2 = SvoAccumulators::<F, 3>::new();
        acc2.accumulate(0, 0, F::from_u64(5));
        acc2.accumulate(0, 1, F::from_u64(15));
        acc2.accumulate(1, 1, F::from_u64(25));
        acc2.accumulate(2, 7, F::from_u64(35));

        // Test Add trait: result_add = acc1 + acc2
        let result_add = acc1_for_add.clone() + acc2.clone();

        // Test AddAssign trait: acc1 += acc2
        acc1_for_add_assign += acc2;

        // VERIFY: Both operations produce identical results
        assert_eq!(
            result_add.at_round(0),
            acc1_for_add_assign.at_round(0),
            "Add and AddAssign should produce identical Round 0 results"
        );
        assert_eq!(
            result_add.at_round(1),
            acc1_for_add_assign.at_round(1),
            "Add and AddAssign should produce identical Round 1 results"
        );
        assert_eq!(
            result_add.at_round(2),
            acc1_for_add_assign.at_round(2),
            "Add and AddAssign should produce identical Round 2 results"
        );

        // Verify the actual values are correct (element-wise addition)
        assert_eq!(
            result_add.at_round(0)[0],
            F::from_u64(15),
            "Round 0[0]: 10 + 5 = 15"
        );
        assert_eq!(
            result_add.at_round(0)[1],
            F::from_u64(35),
            "Round 0[1]: 20 + 15 = 35"
        );
        assert_eq!(
            result_add.at_round(1)[1],
            F::from_u64(25),
            "Round 1[1]: 0 + 25 = 25"
        );
        assert_eq!(
            result_add.at_round(1)[2],
            F::from_u64(30),
            "Round 1[2]: 30 + 0 = 30"
        );
    }

    #[test]
    fn test_add_multiple_accumulators() {
        // Thread 1 accumulator
        let mut thread1 = SvoAccumulators::<F, 2>::new();
        thread1.accumulate(0, 0, F::from_u64(100));
        thread1.accumulate(0, 1, F::from_u64(200));
        thread1.accumulate(1, 0, F::from_u64(10));

        // Thread 2 accumulator
        let mut thread2 = SvoAccumulators::<F, 2>::new();
        thread2.accumulate(0, 0, F::from_u64(50));
        thread2.accumulate(0, 1, F::from_u64(75));
        thread2.accumulate(1, 3, F::from_u64(20));

        // Thread 3 accumulator
        let mut thread3 = SvoAccumulators::<F, 2>::new();
        thread3.accumulate(0, 0, F::from_u64(25));
        thread3.accumulate(1, 0, F::from_u64(5));
        thread3.accumulate(1, 3, F::from_u64(15));

        // Combine all thread results using Add: result = thread1 + thread2 + thread3
        let result = thread1 + thread2 + thread3;

        // VERIFY: Check cumulative sums
        //
        // Round 0[0] = 100 + 50 + 25 = 175
        // Round 0[1] = 200 + 75 + 0 = 275
        // Round 1[0] = 10 + 0 + 5 = 15
        // Round 1[3] = 0 + 20 + 15 = 35

        let round_0 = result.at_round(0);
        assert_eq!(
            round_0[0],
            F::from_u64(175),
            "Round 0[0] should accumulate all thread contributions: 100+50+25"
        );
        assert_eq!(
            round_0[1],
            F::from_u64(275),
            "Round 0[1] should accumulate all thread contributions: 200+75+0"
        );

        let round_1 = result.at_round(1);
        assert_eq!(
            round_1[0],
            F::from_u64(15),
            "Round 1[0] should accumulate all thread contributions: 10+0+5"
        );
        assert_eq!(
            round_1[3],
            F::from_u64(35),
            "Round 1[3] should accumulate all thread contributions: 0+20+15"
        );

        // Verify other indices remain zero (no contributions from any thread)
        assert_eq!(round_1[1], F::ZERO, "Round 1[1] had no contributions");
        assert_eq!(round_1[2], F::ZERO, "Round 1[2] had no contributions");
    }
}
