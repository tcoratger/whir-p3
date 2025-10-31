use crate::{
    fiat_shamir::prover::ProverState,
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
};
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field};

use std::ops::Add;

use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::eq::eval_eq;

// We apply the small value optimization (SVO) for the first three sumcheck rounds,
// following this paper <https://eprint.iacr.org/2025/1117>.
pub const NUM_OF_ROUNDS: usize = 3;

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Accumulators<F: Field> {
    // One accumulator vector per SVO round (l_0 = 3).
    pub accumulators: [Vec<F>; 3],
}

impl<F> Accumulators<F>
where
    F: Field,
{
    pub fn new_empty() -> Self {
        Accumulators {
            // In round 0, we have 2 accumulators: A_0(u) with u in {0, 1}.
            // In round 1, we have 4 accumulators: A_1(v, u) with v in {0, 1} and u in {0, 1}.
            // In round 2, we have 8 accumulators: A_2(v, u) with v in {0, 1}^2 and u in {0, 1}.
            // We won't need accumulators with any digit as infinity.
            accumulators: [vec![F::ZERO; 2], vec![F::ZERO; 4], vec![F::ZERO; 8]],
        }
    }

    pub fn accumulate(&mut self, round: usize, index: usize, value: F) {
        self.accumulators[round][index] += value;
    }

    pub fn get_accumulators_for_round(&self, round: usize) -> &[F] {
        &self.accumulators[round]
    }
}

impl<F: Field> Add for Accumulators<F> {
    type Output = Self;

    fn add(mut self, other: Self) -> Self {
        for i in 0..NUM_OF_ROUNDS {
            // NUM_OF_ROUNDS is 3
            for j in 0..self.accumulators[i].len() {
                self.accumulators[i][j] += other.accumulators[i][j];
            }
        }
        self
    }
}
// For round i, RoundAccumulators has all the accumulators of the form A_i(u, v).
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RoundAccumlators<F: Field> {
    pub round: usize,
    pub accumulators: Vec<F>,
}

// Precomputation needed for Procedure 9 (compute_accumulators).
// Compute the evaluations eq(w_{l0 + 1}, ..., w_{l0 + l/2} ; x) for all x in {0,1}^l/2
fn precompute_e_in<F: Field>(w: &MultilinearPoint<F>) -> Vec<F> {
    let half_l = w.num_variables() / 2;
    let w_in = w.0[NUM_OF_ROUNDS..NUM_OF_ROUNDS + half_l].to_vec();
    eval_eq_in_hypercube(&w_in)
}

// Precomputation needed for Procedure 9 (compute_accumulators).
// Compute three E_out vectors, one per round i in {0, 1, 2}.
// For each i, E_out = eq(w_{i+1}, ..., l0, w_{l/2 + l0 + 1}, ..., w_l ; x) 
fn precompute_e_out<F: Field>(w: &MultilinearPoint<F>) -> [Vec<F>; NUM_OF_ROUNDS] {
    let half_l = w.num_variables() / 2;
    let w_out_len = w.num_variables() - half_l - 1;

    std::array::from_fn(|round| {
        let mut w_out = Vec::with_capacity(w_out_len);
        w_out.extend_from_slice(&w.0[round + 1..NUM_OF_ROUNDS]);
        w_out.extend_from_slice(&w.0[half_l + NUM_OF_ROUNDS..]);
        eval_eq_in_hypercube(&w_out)
    })
}

// Procedure 9. Page 37.
// We compute only the accumulators that we'll use, that is,
// A_i(v, u) for i in {0, 1, 2}, v in {0, 1}^{i}, and u in {0, 1}.
fn compute_accumulators<F: Field, EF: ExtensionField<F>>(
    poly: &EvaluationsList<F>,
    e_in: Vec<EF>,
    e_out: [Vec<EF>; NUM_OF_ROUNDS],
) -> Accumulators<EF> {
    let l = poly.num_variables();
    let half_l = l / 2;

    let x_out_num_vars = half_l - NUM_OF_ROUNDS + (l % 2);
    let x_num_vars = l - NUM_OF_ROUNDS;
    debug_assert_eq!(half_l + x_out_num_vars, x_num_vars);

    let poly_evals = poly.as_slice();

    (0..1 << x_out_num_vars)
        .into_par_iter()
        .map(|x_out| {
            // Each thread will compute its own set of local accumulators.
            // This avoids mutable state sharing and the need for locks.
            let mut local_accumulators = Accumulators::<EF>::new_empty();

            let mut temp_accumulators = [EF::ZERO; 8];

            let num_x_in = 1 << half_l;

            for x_in in 0..num_x_in {
                let e_in_value = e_in[x_in];

                // For each beta in {0,1}^3, we update tA(beta) += e_in[x_in] * p(beta, x_in, x_out)
                for i in 0..8 {
                    let beta = i << x_num_vars;
                    let index = beta | (x_in << x_out_num_vars) | x_out; // beta | x_in | x_out
                    temp_accumulators[i] += e_in_value * poly_evals[index]; // += e_in[x_in] * p(beta, x_in, x_out)
                }
            }

            let temp_acc = &temp_accumulators;


            // e_out for round i = 0: y in {0,1}^2.
            let e0_0 = e_out[0][x_out]; // e_out_0(0,0, x_out)
            let e0_1 = e_out[0][(1 << x_out_num_vars) | x_out]; // e_out_0(0,1, x_out)
            let e0_2 = e_out[0][(2 << x_out_num_vars) | x_out]; // e_out_0(1,0, x_out)
            let e0_3 = e_out[0][(3 << x_out_num_vars) | x_out]; // e_out_0(1,1, x_out)
            // e_out for round i = 1: y in {0,1}.
            let e1_0 = e_out[1][x_out]; // e_out_1(0, x_out)
            let e1_1 = e_out[1][(1 << x_out_num_vars) | x_out]; // e_out_1(1, x_out)
            // e_out for round i = 2: there is no y.
            let e2 = e_out[2][x_out]; // e_out_2(x_out)

            // We do not use the idx4 function since we are directly computing the indices for efficiency.
            // We go through each beta = (v, u, y) in {0, 1}^3; we don't compute the cases where the digits are
            // infinity because we won't need them.
            // Recall that in `v || u` determines the accumulator's index, `y` determines the e_out factor, and the
            // whole beta determines de temp_acc.

            // beta = (0,0,0)
            local_accumulators.accumulate(0, 0, e0_0 * temp_acc[0]); // u = 0, y = 00
            local_accumulators.accumulate(1, 0, e1_0 * temp_acc[0]); // v = 0, u = 0, y = 0,
            local_accumulators.accumulate(2, 0, e2 * temp_acc[0]); // v = 00, u = 0

            // beta = (0,0,1)
            local_accumulators.accumulate(0, 0, e0_1 * temp_acc[1]); // u = 0, y = 01
            local_accumulators.accumulate(1, 0, e1_1 * temp_acc[1]); //  v = 0, u = 0, y = 1
            local_accumulators.accumulate(2, 1, e2 * temp_acc[1]); // v = 00, u = 1

            // beta = (0,1,0)
            local_accumulators.accumulate(0, 0, e0_2 * temp_acc[2]); // u = 0, y = 10
            local_accumulators.accumulate(1, 1, e1_0 * temp_acc[2]); // v = 0, u = 1, y = 0
            local_accumulators.accumulate(2, 2, e2 * temp_acc[2]); // v = 01, u = 0

            // beta = (0,1,1)
            local_accumulators.accumulate(0, 0, e0_3 * temp_acc[3]); // u = 0, y = 11
            local_accumulators.accumulate(1, 1, e1_1 * temp_acc[3]); // v = 0, u = 1, y = 1
            local_accumulators.accumulate(2, 3, e2 * temp_acc[3]); // v = 01, u = 1

            // beta = (1,0,0)
            local_accumulators.accumulate(0, 1, e0_0 * temp_acc[4]); // u = 1, y = 00
            local_accumulators.accumulate(1, 2, e1_0 * temp_acc[4]); // v = 1, u = 0, y = 0
            local_accumulators.accumulate(2, 4, e2 * temp_acc[4]); // v = 10, u = 0

            // beta = (1,0,1)
            local_accumulators.accumulate(0, 1, e0_1 * temp_acc[5]); // u = 1, y = 01
            local_accumulators.accumulate(1, 2, e1_1 * temp_acc[5]); // v = 1, u = 0, y = 1
            local_accumulators.accumulate(2, 5, e2 * temp_acc[5]); // v = 10, u = 1

            // beta = (1,1,0)
            local_accumulators.accumulate(0, 1, e0_2 * temp_acc[6]); // u = 1, y = 10
            local_accumulators.accumulate(1, 3, e1_0 * temp_acc[6]); // v = 1, u = 1, y = 0
            local_accumulators.accumulate(2, 6, e2 * temp_acc[6]); // v = 11, u = 0

            // beta = (1,1,1)
            local_accumulators.accumulate(0, 1, e0_3 * temp_acc[7]); // u = 1, y = 11
            local_accumulators.accumulate(1, 3, e1_1 * temp_acc[7]); // v = 1, u = 1, y = 1
            local_accumulators.accumulate(2, 7, e2 * temp_acc[7]); // v = 11, u = 1

            local_accumulators
        })
        .reduce(|| Accumulators::<EF>::new_empty(), |a, b| a + b) 
}

// Given a point w = (w_1, ..., w_l), it returns the evaluations of eq(w, x) for all x in {0, 1}^l.
pub fn eval_eq_in_hypercube<F: Field>(point: &Vec<F>) -> Vec<F> {
    let n = point.len();
    let mut evals = F::zero_vec(1 << n);
    eval_eq::<_, _, false>(point, &mut evals, F::ONE);
    evals
}

// Given two multilinear points p and q, it evaluates eq(p, q).
pub fn eval_eq_in_point<F: Field>(p: &[F], q: &[F]) -> F {
    let mut acc = F::ONE;
    for (&l, &r) in p.into_iter().zip(q) {
        acc *= F::ONE + l * r.double() - l - r;
    }
    acc
}

// Given the points w and r, we compute the linear function 
// l(x) = eq( w_1,...w_{i-1} ; r_1,...r_{i-1} ) * eq(w_i, x)
// Section 5.1, page 17.
pub fn compute_linear_function<F: Field>(w: &[F], r: &[F]) -> [F; 2] {
    let round = w.len();
    debug_assert!(r.len() == round - 1);

    let mut const_eq: F = F::ONE;
    if round != 1 {
        const_eq = eval_eq_in_point(&w[..round - 1], r);
    }
    let w_i = w.last().unwrap();

    // Evaluation of eq(w,X) in [eq(w,0),eq(w,1)]
    [const_eq * (F::ONE - *w_i), const_eq * *w_i]
}

// Given the linear functions l and t, we compute S(0) and S(inf).
// See Procedure 8, Page 35.
fn get_evals_from_l_and_t<F: Field>(l: &[F; 2], t: &[F]) -> [F; 2] {
    [
        t[0] * l[0],                   // S(0)
        (t[1] - t[0]) * (l[1] - l[0]), // S(inf) -> l(inf) = l(1) - l(0)
    ]
}

// Algorithm 6. Page 19.
// Compute three sumcheck rounds using the small value optimizaition and split-eq accumulators.
// It returns the three challenges r_1, r_2, r_3.
pub fn svo_three_rounds<Challenger, F: Field, EF: ExtensionField<F>>(
    prover_state: &mut ProverState<F, EF, Challenger>,
    poly: &EvaluationsList<F>,
    w: &MultilinearPoint<EF>,
    sum: &mut EF,
) -> (EF, EF, EF)
where
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    let e_in = precompute_e_in(w);
    let e_out = precompute_e_out(w);

    // We compute all the accumulators A_i(v, u).
    let accumulators = compute_accumulators(poly, e_in, e_out);

    // ------------------   Round 1   ------------------

    // 1. For u in {0, 1} compute t_1(u)
    // Recall: In round 1, t_1(u) = A_1(u).
    let t_1_evals = accumulators.get_accumulators_for_round(0);

    // 2. For u in {0, 1, inf} compute S_1(u) = t_1(u) * l_1(u).

    // We compute l_1(0) and l_1(1)
    let linear_1_evals = compute_linear_function(&w.0[..1], &[]);

    // We compute S_1(0) and S_1(inf)
    let round_poly_evals = get_evals_from_l_and_t(&linear_1_evals, &t_1_evals);

    // 3. Send S_1(u) to the verifier.
    prover_state.add_extension_scalars(&round_poly_evals);

    // 4. Receive the challenge r_1 from the verifier.
    let r_1: EF = prover_state.sample();

    let eval_1 = *sum - round_poly_evals[0];
    *sum = round_poly_evals[1] * r_1.square()
        + (eval_1 - round_poly_evals[0] - round_poly_evals[1]) * r_1
        + round_poly_evals[0];

    // 5. Compte R_2 = [L_0(r_1), L_1(r_1), L_inf(r_1)]
    // L_0 (x) = 1 - x
    // L_1 (x) = x
    // L_inf (x) = (x - 1)x
    let lagrange_evals_r_1 = [-r_1 + F::ONE, r_1];

    // ------------------ Round 2 ------------------

    // 1. For u in {0, 1} compute t_2(u).
    // First we take the accumulators A_2(v, u).
    // We computed 4 accumulators for round 2, since only need v in {0, 1} and u in {0, 1}.
    let accumulators_round_2 = accumulators.get_accumulators_for_round(1);

    let mut t_2_evals = [EF::ZERO; 2];

    
    t_2_evals[0] += lagrange_evals_r_1[0] * accumulators_round_2[0];
    t_2_evals[0] += lagrange_evals_r_1[1] * accumulators_round_2[2];

    t_2_evals[1] += lagrange_evals_r_1[0] * accumulators_round_2[1];
    t_2_evals[1] += lagrange_evals_r_1[1] * accumulators_round_2[3];

    // We compute l_2(0) and l_2(inf)
    let linear_2_evals = compute_linear_function(&w.0[..2], &[r_1]);

    // We compute S_2(0) and S_2(inf).
    let round_poly_evals = get_evals_from_l_and_t(&linear_2_evals, &t_2_evals);

    // 3. Send S_2(u) to the verifier.
    prover_state.add_extension_scalars(&round_poly_evals);

    // 4. Receive the challenge r_2 from the verifier.
    let r_2: EF = prover_state.sample();

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

    let eval_1 = *sum - round_poly_evals[0];
    *sum = round_poly_evals[1] * r_2.square()
        + (eval_1 - round_poly_evals[0] - round_poly_evals[1]) * r_2
        + round_poly_evals[0];

    // Round 3

    // 1. For u in {0, 1} compute t_3(u).

    // First we take the accumulators A_3(v, u).
    // We computed 4 accumulators at the third round for v in {0, 1}^2 and u in {0, 1}.
    let accumulators_round_3 = accumulators.get_accumulators_for_round(2);

    let mut t_3_evals = [EF::ZERO; 2];

    t_3_evals[0] += lagrange_evals_r_2[0] * accumulators_round_3[0]; // (v=00 u=0)
    t_3_evals[0] += lagrange_evals_r_2[1] * accumulators_round_3[2]; // (10 0)
    t_3_evals[0] += lagrange_evals_r_2[2] * accumulators_round_3[4]; // (01 0)
    t_3_evals[0] += lagrange_evals_r_2[3] * accumulators_round_3[6]; // (11 0)

    t_3_evals[1] += lagrange_evals_r_2[0] * accumulators_round_3[1]; // (00 1)
    t_3_evals[1] += lagrange_evals_r_2[1] * accumulators_round_3[3]; // (01 1)
    t_3_evals[1] += lagrange_evals_r_2[2] * accumulators_round_3[5]; // (10 1)
    t_3_evals[1] += lagrange_evals_r_2[3] * accumulators_round_3[7]; // (11 1)

    // 2. For u in {0, 1, inf} compute S_3(u) = t_3(u) * l_3(u).

    // We compute l_3(0) and l_3(inf)
    let linear_3_evals = compute_linear_function(&w.0[..3], &[r_1, r_2]);

    // We compute S_3(0) and S_3(inf).
    let round_poly_evals = get_evals_from_l_and_t(&linear_3_evals, &t_3_evals);

    // 3. Send S_3(u) to the verifier.
    prover_state.add_extension_scalars(&round_poly_evals);

    let r_3: EF = prover_state.sample();

    let eval_1 = *sum - round_poly_evals[0];
    *sum = round_poly_evals[1] * r_3.square()
        + (eval_1 - round_poly_evals[0] - round_poly_evals[1]) * r_3
        + round_poly_evals[0];

    (r_1, r_2, r_3)
}

/// This function takes a list of evaluations and "folds" or "compresses" them according
/// to the provided challenges `r_1, ..., r_{k}`. The result is a new evaluation list
/// representing p(r_1, ..., r_{k}, x) for all x in {0,1}^k.
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

    let eq_evals: Vec<EF> = eval_eq_in_hypercube(&challenges.to_vec());

    let folded_evals_flat: Vec<EF> = (0..num_remaining_evals)
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

// Algorithm 5. Page 18.
// Compute the remaining sumcheck rounds, from round l0 + 1 to round l.
pub fn algorithm_5<Challenger, F: Field, EF: ExtensionField<F>>(
    prover_state: &mut ProverState<F, EF, Challenger>,
    poly: &mut EvaluationsList<EF>,
    w: &MultilinearPoint<EF>,
    challenges: &mut Vec<EF>,
    sum: &mut EF,
) where
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    EF: Send + Sync,
{
    let num_vars = w.num_variables();
    let half_l = num_vars / 2;

    // We compute eq(w_{l/2 + 1}, ...,  w_l ; x_R) for all x_R in {0, 1}^{l/2}
    // These evaluations don't depend on the round i, so they are computed outside the loop.
    let eq_r = eval_eq_in_hypercube(&w.0[half_l..].to_vec());
    let num_vars_x_r = eq_r.len().ilog2() as usize;

    // The number of variables of x_R is: l/2 if l is even and l/2 + 1 if l is odd.
    debug_assert!(num_vars_x_r == half_l + (num_vars % 2));

    let mut t = Vec::with_capacity(2);
    challenges.reserve(num_vars - 3);

    // Loop for the final rounds, from l_0+1 (in our case 4) to the end.
    let current_round = challenges.len() + 1;
    for i in current_round..num_vars + 1 {
        t.clear();

        // We get the number of variables of `poly` in the current round.
        let num_vars_poly_current = poly.num_variables();
        let poly_slice = poly.as_slice();

        // 1. We compute t_i(u) for u in {0, 1}.
        if i <= half_l {
            // We compute eq(w_{i + 1}, ...,  w_{l/2} ; x_L) for all x_L in {0, 1}^{l/2 - i}
            let eq_l = eval_eq_in_hypercube(&w.0[i..half_l].to_vec());
            let eq_l_len = eq_l.len();
            let num_x_r = 1 << num_vars_x_r;

            let (t_0, t_1) = rayon::join(
                || {
                    compute_t_evals_first_rounds(
                        &eq_l,
                        &eq_r,
                        poly_slice,
                        eq_l_len,
                        num_x_r,
                        num_vars_x_r,
                        0,
                    )
                },
                || {
                    compute_t_evals_first_rounds(
                        &eq_l,
                        &eq_r,
                        poly_slice,
                        eq_l_len,
                        num_x_r,
                        num_vars_x_r,
                        1 << (num_vars_poly_current - 1),
                    )
                },
            );

            t.push(t_0);
            t.push(t_1);

        // Case i > l/2: Only one part of the eq evaluations remains to be processed.
        } else {
            let eq = eval_eq_in_hypercube(&w.0[i..num_vars].to_vec());
            let half_size = 1 << (num_vars_poly_current - 1);

            let (t_0, t_1) = rayon::join(
                || compute_t_evals_last_rounds(&eq, poly_slice, 0, half_size),
                || compute_t_evals_last_rounds(&eq, poly_slice, half_size, half_size),
            );

            t.push(t_0);
            t.push(t_1);
        }

        // 2. We compute S_i(u) = t_i(u) * l_i(u) for u in {0, inf}:
        // Compute l_i(0) and l_i(inf).
        let linear_evals = compute_linear_function(&w.0[..i], challenges);
        // Compute S_i(u).
        let round_poly_evals = get_evals_from_l_and_t(&linear_evals, &t);

        // 3. Send S_i(u) to the verifier.
        prover_state.add_extension_scalars(&round_poly_evals);

        // 4. Receive the challenge r_i from the verifier.
        let r_i: EF = prover_state.sample();
        challenges.push(r_i);

        // 5. Fold and update the poly.
        poly.compress_svo(r_i);

        // Update claimed sum
        let eval_1 = *sum - round_poly_evals[0];
        *sum = round_poly_evals[1] * r_i.square()
            + (eval_1 - round_poly_evals[0] - round_poly_evals[1]) * r_i
            + round_poly_evals[0];
    }
}

// Auxiliary function for Algorithm 5.
// Compute t_i(u) for the case i <= l/2,
// where u can be 0 or 1, indicated by the offset parameter: 0 for `u = 0` and `1 << (num_vars - 1)` for `u = 1`.
#[inline]
fn compute_t_evals_first_rounds<F: Field>(
    eq_l: &[F],
    eq_r: &[F],
    poly_slice: &[F],
    eq_l_len: usize,
    num_x_r: usize,
    num_vars_x_r: usize,
    offset: usize,
) -> F
where
    F: Send + Sync,
{
    let chunk_size = (num_x_r / rayon::current_num_threads().max(1)).max(64);

    (0..num_x_r)
        .into_par_iter()
        .chunks(chunk_size)
        .map(|chunk| {
            chunk
                .into_iter()
                .map(|x_r| {
                    let sum_l: F = (0..eq_l_len)
                        .map(|x_l| {
                            // We need the idx for p(r_[<i-1], 0, x_L, x_R) in the case u = 0 and p(r_[<i-1], 1, x_L, x_R) in the case u = 1.
                            let idx = offset | (x_l << num_vars_x_r) | x_r;
                            eq_l[x_l] * poly_slice[idx]
                        })
                        .sum();
                    eq_r[x_r] * sum_l
                })
                .sum::<F>()
        })
        .sum()
}

// Auxiliary function for Algorithm 5.
// Compute t_i(u) for the case i > l/2,
// where u can be 0 or 1, indicated by the offset parameter: 0 for `u = 0` and `1 << (num_vars - 1)` for `u = 1`.
#[inline]
fn compute_t_evals_last_rounds<F: Field>(
    eq: &[F],
    poly_slice: &[F],
    offset: usize,
    size: usize,
) -> F
where
    F: Send + Sync,
{
    (0..size)
        .into_par_iter()
        .map(|x| eq[x] * poly_slice[offset + x])
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    };
    use p3_baby_bear::BabyBear;
    use p3_field::{
        BasedVectorSpace, PrimeCharacteristicRing, extension::BinomialExtensionField,
        integers::QuotientMap,
    };
    use rand::RngCore;

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    #[test]
    fn test_evals_serial_three_vars_matches_new_from_point() {
        let p = vec![F::from_u64(2), F::from_u64(3), F::from_u64(5)];
        let point = MultilinearPoint::new(p.to_vec());
        let value = F::from_u64(1);

        let via_method = EvaluationsList::new_from_point(&point, value)
            .into_iter()
            .collect::<Vec<_>>();
        let via_serial = eval_eq_in_hypercube(&p);

        assert_eq!(via_serial, via_method);
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

        let out = eval_eq_in_hypercube(&vec![p0, p1, p2]);
        assert_eq!(out, expected);
    }

    #[test]
    fn test_precompute_e_in() {
        let w: Vec<EF> = (0..10).map(|i| EF::from(F::from_int(i))).collect();
        let w = MultilinearPoint::new(w);

        let e_in = precompute_e_in(&w);

        let w_in = w.0[NUM_OF_ROUNDS..NUM_OF_ROUNDS + 5].to_vec();

        assert_eq!(
            w_in,
            vec![
                EF::from(F::from_int(3)),
                EF::from(F::from_int(4)),
                EF::from(F::from_int(5)),
                EF::from(F::from_int(6)),
                EF::from(F::from_int(7)),
            ]
        );

        // e_in should have length 2^5 = 32.
        assert_eq!(e_in.len(), 1 << 5);

        //  e_in[0] should be eq(w_in, 00000)
        let expected_0: EF = w_in.iter().map(|w_i| EF::ONE - *w_i).product();

        assert_eq!(expected_0, e_in[0]);
        assert_eq!(EF::from(-F::from_int(720)), e_in[0]);

        // e_in[5] should be  eq(w_in, 00101)
        let expected_5 =
            (EF::ONE - w_in[0]) * (EF::ONE - w_in[1]) * w_in[2] * (EF::ONE - w_in[3]) * w_in[4];
        assert_eq!(expected_5, e_in[5]);

        // e_in[15] should be eq(w_in, 10000)
        let expected_16: EF = w_in[1..].iter().map(|w_i| EF::ONE - *w_i).product::<EF>() * w_in[0];
        assert_eq!(expected_16, e_in[16]);

        // e_in[31] should be eq(w_in, 11111)
        let expected_31: EF = w_in.iter().map(|w_i| *w_i).product();
        assert_eq!(expected_31, e_in[31]);
    }

    #[test]
    fn test_precompute_e_out() {
        let mut rng = rand::rng();

        let w: Vec<EF> = (0..10)
            .map(|_| {
                let r1: u32 = rng.next_u32();
                let r2: u32 = rng.next_u32();
                let r3: u32 = rng.next_u32();
                let r4: u32 = rng.next_u32();

                EF::from_basis_coefficients_slice(&[
                    F::from_u32(r1),
                    F::from_u32(r2),
                    F::from_u32(r3),
                    F::from_u32(r4),
                ])
                .unwrap()
            })
            .collect();

        let w = MultilinearPoint::new(w);
        let e_out = precompute_e_out(&w);

        // Round 1:
        assert_eq!(e_out[0].len(), 16);

        assert_eq!(
            e_out[0][0],
            (EF::ONE - w[1]) * (EF::ONE - w[2]) * (EF::ONE - w[8]) * (EF::ONE - w[9])
        );
        assert_eq!(
            e_out[0][1],
            (EF::ONE - w[1]) * (EF::ONE - w[2]) * (EF::ONE - w[8]) * w[9]
        );
        assert_eq!(
            e_out[0][2],
            (EF::ONE - w[1]) * (EF::ONE - w[2]) * w[8] * (EF::ONE - w[9])
        );
        assert_eq!(
            e_out[0][3],
            (EF::ONE - w[1]) * (EF::ONE - w[2]) * w[8] * w[9]
        );
        assert_eq!(
            e_out[0][4],
            (EF::ONE - w[1]) * w[2] * (EF::ONE - w[8]) * (EF::ONE - w[9])
        );
        assert_eq!(
            e_out[0][5],
            (EF::ONE - w[1]) * w[2] * (EF::ONE - w[8]) * w[9]
        );
        assert_eq!(
            e_out[0][6],
            (EF::ONE - w[1]) * w[2] * w[8] * (EF::ONE - w[9])
        );
        assert_eq!(e_out[0][7], (EF::ONE - w[1]) * w[2] * w[8] * w[9]);

        assert_eq!(
            e_out[0][8],
            w[1] * (EF::ONE - w[2]) * (EF::ONE - w[8]) * (EF::ONE - w[9])
        );
        assert_eq!(
            e_out[0][9],
            w[1] * (EF::ONE - w[2]) * (EF::ONE - w[8]) * w[9]
        );
        assert_eq!(
            e_out[0][10],
            w[1] * (EF::ONE - w[2]) * w[8] * (EF::ONE - w[9])
        );
        assert_eq!(e_out[0][11], w[1] * (EF::ONE - w[2]) * w[8] * w[9]);
        assert_eq!(
            e_out[0][12],
            w[1] * w[2] * (EF::ONE - w[8]) * (EF::ONE - w[9])
        );
        assert_eq!(e_out[0][13], w[1] * w[2] * (EF::ONE - w[8]) * w[9]);
        assert_eq!(e_out[0][14], w[1] * w[2] * w[8] * (EF::ONE - w[9]));
        assert_eq!(e_out[0][15], w[1] * w[2] * w[8] * w[9]);

        // Round 2:
        assert_eq!(e_out[1].len(), 8);

        assert_eq!(
            e_out[1][0],
            (EF::ONE - w[2]) * (EF::ONE - w[8]) * (EF::ONE - w[9])
        );
        assert_eq!(e_out[1][1], (EF::ONE - w[2]) * (EF::ONE - w[8]) * w[9]);
        assert_eq!(e_out[1][2], (EF::ONE - w[2]) * w[8] * (EF::ONE - w[9]));
        assert_eq!(e_out[1][3], (EF::ONE - w[2]) * w[8] * w[9]);
        assert_eq!(e_out[1][4], w[2] * (EF::ONE - w[8]) * (EF::ONE - w[9]));
        assert_eq!(e_out[1][5], w[2] * (EF::ONE - w[8]) * w[9]);
        assert_eq!(e_out[1][6], w[2] * w[8] * (EF::ONE - w[9]));
        assert_eq!(e_out[1][7], w[2] * w[8] * w[9]);

        // Round 3:
        assert_eq!(e_out[2].len(), 4);

        assert_eq!(e_out[2][0], (EF::ONE - w[8]) * (EF::ONE - w[9]));
        assert_eq!(e_out[2][1], (EF::ONE - w[8]) * w[9]);
        assert_eq!(e_out[2][2], w[8] * (EF::ONE - w[9]));
        assert_eq!(e_out[2][3], w[8] * w[9]);
    }


    fn get_random_ef() -> EF {
        let mut rng = rand::rng();

        let r1: u32 = rng.next_u32();
        let r2: u32 = rng.next_u32();
        let r3: u32 = rng.next_u32();
        let r4: u32 = rng.next_u32();

        EF::from_basis_coefficients_slice(&[
            F::from_u32(r1),
            F::from_u32(r2),
            F::from_u32(r3),
            F::from_u32(r4),
        ])
        .unwrap()
    }

    #[test]
    fn test_compute_linear_function() {
        // w = [1]
        // r = []
        let w = [EF::from(F::from_int(1))];
        let r = [];
        // l(0) = 0
        // l(1) = 1
        let expected = [EF::from(F::from_int(0)), EF::from(F::from_int(1))];
        let result = compute_linear_function(&w, &r);
        assert_eq!(result, expected);

        // w = [1, 1]
        // r = [1]
        let w = [EF::from(F::from_int(1)), EF::from(F::from_int(1))];
        let r = [EF::from(F::from_int(1))];
        // l(0) = 0
        // l(1) = 1
        let expected = [EF::from(F::from_int(0)), EF::from(F::from_int(1))];
        let result = compute_linear_function(&w, &r);
        assert_eq!(result, expected);

        // w = [w0, w1, w2, w3]
        // r = [r0, r1, r2]
        let w: Vec<EF> = (0..4).map(|_| get_random_ef()).collect();
        let r: Vec<EF> = (0..3).map(|_| get_random_ef()).collect();

        let expected = [
            eval_eq_in_point(&w[..3], &r) * eval_eq_in_point(&w[3..], &[EF::ZERO]),
            eval_eq_in_point(&w[..3], &r) * eval_eq_in_point(&w[3..], &[EF::ONE]),
        ];
        let result = compute_linear_function(&w, &r);
        assert_eq!(result, expected);
    }


    fn get_evals_from_l_and_t<F: Field>(l: &[F; 2], t: &[F]) -> [F; 2] {
        [
            t[0] * l[0],                   // s(0)
            (t[1] - t[0]) * (l[1] - l[0]), // s(inf) -> l(inf) = l(1) - l(0)
        ]
    }
}
