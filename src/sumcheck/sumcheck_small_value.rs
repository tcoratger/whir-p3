use alloc::vec::Vec;
use core::ops::Add;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field};
use p3_maybe_rayon::prelude::*;

use crate::{
    fiat_shamir::prover::ProverState,
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    sumcheck::{eq_state::SumcheckEqState, sumcheck_single_svo::NUM_SVO_ROUNDS},
};

/// One accumulator vector per SVO round.
/// - `accumulators[i]` has 2^i accumulators for A_i(v, u)
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Accumulators<F: Field>([Vec<F>; NUM_SVO_ROUNDS]);

impl<F> Accumulators<F>
where
    F: Field,
{
    /// In round i, we have 2^i accumulators: A_i(v, u) with v in {0, 1}^i and u in {0, 1}.
    ///
    /// We won't need accumulators with any digit as infinity.
    #[must_use]
    pub fn new_empty() -> Self {
        Self(core::array::from_fn(|i| F::zero_vec(1 << (i + 1))))
    }

    /// Adds a value to a specific accumulator.
    pub fn accumulate(&mut self, round: usize, index: usize, value: F) {
        self.0[round][index] += value;
    }

    /// Gets the slice of accumulators for a given round.
    #[must_use]
    pub fn at_round(&self, round: usize) -> &[F] {
        &self.0[round]
    }
}

impl<F: Field> Add for Accumulators<F> {
    type Output = Self;

    fn add(mut self, other: Self) -> Self {
        for i in 0..NUM_SVO_ROUNDS {
            self.0[i]
                .iter_mut()
                .zip(other.0[i].iter())
                .for_each(|(a, b)| *a += *b);
        }
        self
    }
}

/// Procedure 9. Page 37.
/// We compute only the accumulators that we'll use, that is,
/// A_i(v, u) for i in {0, 1, 2}, v in {0, 1}^{i}, and u in {0, 1}.
fn compute_accumulators<F: Field, EF: ExtensionField<F>>(
    poly: &EvaluationsList<F>,
    e_in: &EvaluationsList<EF>,
    e_out: &[EvaluationsList<EF>; NUM_SVO_ROUNDS],
) -> Accumulators<EF> {
    let l = poly.num_variables();
    let half_l = l / 2;

    let x_out_num_vars = half_l - NUM_SVO_ROUNDS + (l % 2);
    let x_num_vars = l - NUM_SVO_ROUNDS;
    debug_assert_eq!(half_l + x_out_num_vars, x_num_vars);

    let poly_evals = poly.as_slice();

    (0..1 << x_out_num_vars)
        .into_par_iter()
        .map(|x_out| {
            // Each thread will compute its own set of local accumulators.
            // This avoids mutable state sharing and the need for locks.
            let mut local_accumulators = Accumulators::<EF>::new_empty();

            let mut temp_accumulators = [EF::ZERO; 1 << NUM_SVO_ROUNDS];

            let num_x_in = 1 << half_l;

            for (x_in, &e_in_value) in e_in.iter().enumerate().take(num_x_in) {
                // For each beta in {0,1}^3, we update tA(beta) += e_in[x_in] * p(beta, x_in, x_out)
                for (i, temp_accumulator) in temp_accumulators
                    .iter_mut()
                    .enumerate()
                    .take(1 << NUM_SVO_ROUNDS)
                {
                    let beta = i << x_num_vars;
                    let index = beta | (x_in << x_out_num_vars) | x_out; // beta | x_in | x_out
                    *temp_accumulator += e_in_value * poly_evals[index]; // += e_in[x_in] * p(beta, x_in, x_out)
                }
            }

            // Destructure things since we will access them many times later
            let [t0, t1, t2, t3, t4, t5, t6, t7] = temp_accumulators;
            // Get E_out(y, x_out) for this x_out
            // Round 0 (i=0) -> y=(b1,b2) -> 2 bits
            let e0_0 = e_out[0].0[x_out]; // y=00
            let e0_1 = e_out[0].0[(1 << x_out_num_vars) | x_out]; // y=01
            let e0_2 = e_out[0].0[(2 << x_out_num_vars) | x_out]; // y=10
            let e0_3 = e_out[0].0[(3 << x_out_num_vars) | x_out]; // y=11
            // Round 1 (i=1) -> y=(b2) -> 1 bit
            let e1_0 = e_out[1].0[x_out]; // y=0
            let e1_1 = e_out[1].0[(1 << x_out_num_vars) | x_out]; // y=1
            // Round 2 (i=2) -> y=() -> 0 bits
            let e2 = e_out[2].0[x_out]; // y=()
            // Round 0 (i=0)
            // A_0(u=0) = Σ_{y} E_out_0(y) * tA( (u=0, y), x_out )
            local_accumulators.accumulate(0, 0, e0_0 * t0 + e0_1 * t1 + e0_2 * t2 + e0_3 * t3);
            // A_0(u=1) = Σ_{y} E_out_0(y) * tA( (u=1, y), x_out )
            local_accumulators.accumulate(0, 1, e0_0 * t4 + e0_1 * t5 + e0_2 * t6 + e0_3 * t7);
            // Round 1 (i=1)
            // A_1(v, u) = Σ_{y} E_out_1(y) * tA( (v, u, y), x_out )
            // v=0, u=0
            local_accumulators.accumulate(1, 0, e1_0 * t0 + e1_1 * t1);
            // v=0, u=1
            local_accumulators.accumulate(1, 1, e1_0 * t2 + e1_1 * t3);
            // v=1, u=0
            local_accumulators.accumulate(1, 2, e1_0 * t4 + e1_1 * t5);
            // v=1, u=1
            local_accumulators.accumulate(1, 3, e1_0 * t6 + e1_1 * t7);
            // Round 2 (i=2)
            // A_2(v, u) = E_out_2() * tA( (v, u), x_out )
            for (i, &temp_accumulator) in temp_accumulators.iter().enumerate() {
                local_accumulators.accumulate(2, i, e2 * temp_accumulator);
            }
            local_accumulators
        })
        .par_fold_reduce(
            || Accumulators::<EF>::new_empty(),
            |a, b| a + b,
            |a, b| a + b,
        )
}

/// Algorithm 6. Page 19.
/// Compute three sumcheck rounds using the small value optimization and split-eq accumulators.
pub fn svo_first_rounds<Challenger, F: Field, EF: ExtensionField<F>>(
    prover_state: &mut ProverState<F, EF, Challenger>,
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
    let accumulators = compute_accumulators(poly, &e_in, &e_out);

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
    prover_state.add_extension_scalars(&[s_0, s_inf]);

    prover_state.pow_grinding(pow_bits);

    // 4. Receive the challenge r_1 from the verifier.
    let r_1: EF = prover_state.sample();
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
    prover_state.add_extension_scalars(&[s_0, s_inf]);

    prover_state.pow_grinding(pow_bits);

    // 4. Receive the challenge r_2 from the verifier.
    let r_2: EF = prover_state.sample();
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
    prover_state.add_extension_scalars(&round_poly_evals);

    prover_state.pow_grinding(pow_bits);

    // 4. Receive the challenge r_3 from the verifier.
    let r_3: EF = prover_state.sample();
    challenges.push(r_3);
    eq_poly.bind(r_3);

    let eval_1 = *sum - round_poly_evals[0];
    *sum = round_poly_evals[1] * r_3.square()
        + (eval_1 - round_poly_evals[0] - round_poly_evals[1]) * r_3
        + round_poly_evals[0];
}

/// This function takes a list of evaluations and "folds" or "compresses" them according
/// to the provided challenges `r_1, ..., r_{k}`.
///
/// The result is a new evaluation list representing p(r_1, ..., r_{k}, x) for all x in {0,1}^k.
/// This implementation is based on Algorithm 2 (page 13), optimized for our case of use.
pub fn fold_evals_with_challenges<F, EF>(
    evals: &EvaluationsList<F>,
    challenges: &[EF],
) -> EvaluationsList<EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let num_challenges = challenges.len();
    let remaining_vars = evals.num_variables() - num_challenges;
    let num_remaining_evals = 1 << remaining_vars;

    let eq_evals = EvaluationsList::new_from_point(challenges, EF::ONE);

    let folded_evals_flat = (0..num_remaining_evals)
        .into_par_iter()
        .map(|i| {
            // Use the multilinear extension formula: p(r, x') = Σ_{b} eq(r, b) * p(b, x')
            eq_evals
                .iter()
                .enumerate()
                .fold(EF::ZERO, |acc, (j, &eq_val)| {
                    let original_eval_index = (j * num_remaining_evals) + i;
                    let p_b_x = evals.as_slice()[original_eval_index];
                    acc + eq_val * p_b_x
                })
        })
        .collect();

    EvaluationsList::new(folded_evals_flat)
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

    if round <= half_l {
        // Case round <= l/2: Use eq_L and eq_R
        join(
            || compute_t_evals_first_half(eq_l, eq_r, poly_slice, num_vars_x_r, 0),
            || {
                compute_t_evals_first_half(
                    eq_l,
                    eq_r,
                    poly_slice,
                    num_vars_x_r,
                    1 << (num_vars_poly_current - 1),
                )
            },
        )
        .into()
    } else {
        // Case round > l/2: Use eq_tail (passed as eq_l)
        let half_size = 1 << (num_vars_poly_current - 1);

        join(
            || compute_t_evals_second_half(eq_l, &poly_slice[..half_size]),
            || compute_t_evals_second_half(eq_l, &poly_slice[half_size..]),
        )
        .into()
    }
}

/// Algorithm 5. Page 18.
/// Compute the remaining sumcheck rounds, from round l0 + 1 to round l.
pub fn algorithm_5<Challenger, F, EF>(
    prover_state: &mut ProverState<F, EF, Challenger>,
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
        prover_state.add_extension_scalars(&[s_0, s_inf]);
        prover_state.pow_grinding(pow_bits);

        // Receive the challenge r_i from the verifier
        let r_i: EF = prover_state.sample();
        challenges.push(r_i);

        // Update state for next round: binding updates scalar AND pops used table
        eq_poly.bind(r_i);
        poly.compress_svo(r_i);

        // Update claimed sum
        let eval_1 = *sum - s_0;
        *sum = s_inf * r_i.square() + (eval_1 - s_0 - s_inf) * r_i + s_0;
    }
}

/// Auxiliary function for Algorithm 5, case `round <= l/2`.
/// Computes `t_i(u) = Σ_{x_R} eq_R(x_R) * ( Σ_{x_L} eq_L(x_L) * p(u, x_L, x_R) )`
#[inline]
fn compute_t_evals_first_half<F: Field + Send + Sync>(
    eq_l: &[F],
    eq_r: &[F],
    poly_slice: &[F],
    num_vars_x_r: usize,
    offset: usize,
) -> F {
    (0..eq_r.len())
        .into_par_iter()
        .map(|x_r| {
            let sum_l: F = (0..eq_l.len())
                .map(|x_l| {
                    // idx = p(u, x_L, x_R)
                    let idx = offset | (x_l << num_vars_x_r) | x_r;
                    eq_l[x_l] * poly_slice[idx]
                })
                .sum();
            eq_r[x_r] * sum_l
        })
        .sum()
}

/// Auxiliary function for Algorithm 5, case `round > l/2`.
/// Computes `t_i(u) = Σ_{x_tail} eq_tail(x_tail) * p(u, x_tail)`
#[inline]
fn compute_t_evals_second_half<F: Field + Send + Sync>(eq_tail: &[F], poly_sub_slice: &[F]) -> F {
    debug_assert_eq!(eq_tail.len(), poly_sub_slice.len());
    // Parallel dot product
    eq_tail
        .par_iter()
        .zip(poly_sub_slice.par_iter())
        .map(|(&e, &p)| e * p)
        .sum()
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

        let accumulators = Accumulators::<F>::new_empty();

        // VERIFY: Each round has the correct number of zero-initialized accumulators.

        // Round 0: 2 accumulators, all zero
        let round_0 = accumulators.at_round(0);
        assert_eq!(round_0.len(), 2, "Round 0 should have 2 accumulators");
        assert_eq!(round_0[0], F::ZERO, "Accumulator A_0(u=0) should be zero");
        assert_eq!(round_0[1], F::ZERO, "Accumulator A_0(u=1) should be zero");

        // Round 1: 4 accumulators, all zero
        let round_1 = accumulators.at_round(1);
        assert_eq!(round_1.len(), 4, "Round 1 should have 4 accumulators");
        for i in 0..4 {
            assert_eq!(round_1[i], F::ZERO, "Accumulator A_1[{}] should be zero", i);
        }

        // Round 2: 8 accumulators, all zero
        let round_2 = accumulators.at_round(2);
        assert_eq!(round_2.len(), 8, "Round 2 should have 8 accumulators");
        for i in 0..8 {
            assert_eq!(round_2[i], F::ZERO, "Accumulator A_2[{}] should be zero", i);
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

        let mut accumulators = Accumulators::<F>::new_empty();

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

        let mut accumulators = Accumulators::<F>::new_empty();

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
        for i in 1..7 {
            assert_eq!(round_2[i], F::ZERO, "Round 2: A_2[{}] should be zero", i);
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
        let mut acc_1 = Accumulators::<F>::new_empty();

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
        let mut acc_2 = Accumulators::<F>::new_empty();

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
}
