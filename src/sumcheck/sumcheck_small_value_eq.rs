use crate::{
    fiat_shamir::prover::ProverState,
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    sumcheck::small_value_utils::{
        Accumulators, compute_p_beta, idx4, idx4_v2, to_base_three_coeff,
    },
    whir::verifier::sumcheck,
};
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field};

use super::sumcheck_polynomial::SumcheckPolynomial;
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::eq::eval_eq;

// WE ASSUME THE NUMBER OF ROUNDS WE ARE DOING WITH SMALL VALUES IS 3
const NUM_OF_ROUNDS: usize = 3;

fn precompute_e_in<F: Field>(w: &MultilinearPoint<F>) -> Vec<F> {
    let half_l = w.num_variables() / 2;
    let w_in = w.0[NUM_OF_ROUNDS..NUM_OF_ROUNDS + half_l].to_vec();
    eval_eq_in_hypercube(&w_in)
}

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

/// Reorders the polynomial evaluations to improve cache locality.
///
/// Instead of the original layout, this function groups the 8 values
/// needed for each `compute_p_beta` call into contiguous blocks.
fn transpose_poly_for_svo<F: Field>(
    poly: &EvaluationsList<F>,
    num_variables: usize,
    x_out_num_vars: usize,
    half_l: usize,
) -> Vec<F> {
    let num_x_in = 1 << half_l;
    let num_x_out = 1 << x_out_num_vars;
    let step_size = 1 << (num_variables - NUM_OF_ROUNDS);
    let block_size = 8;

    // Pre-allocate the full memory for the transposed data.
    let mut transposed_poly = vec![F::ZERO; 1 << num_variables];
    let x_out_block_size = num_x_in * block_size;

    // Parallelize the transposition work.
    transposed_poly
        .par_chunks_mut(x_out_block_size)
        .enumerate()
        .for_each(|(x_out, chunk)| {
            // Each thread works on a separate `x_out` chunk.
            for x_in in 0..num_x_in {
                let start_index = (x_in << x_out_num_vars) | x_out;

                // The destination index is relative to the start of the current chunk.
                let dest_base_index = x_in * block_size;

                let mut iter = poly.iter().skip(start_index).step_by(step_size);
                for i in 0..block_size {
                    chunk[dest_base_index + i] = *iter.next().unwrap();
                }
            }
        });

    transposed_poly
}

// OLD VERSION:
// Procedure 9. Page 37.
fn compute_accumulators_eq_old<F: Field, EF: ExtensionField<F>>(
    poly: &EvaluationsList<F>,
    e_in: Vec<EF>,
    e_out: [Vec<EF>; NUM_OF_ROUNDS],
) -> Accumulators<EF> {
    let l = poly.num_variables();
    let half_l = l / 2;

    let x_out_num_variables = half_l - NUM_OF_ROUNDS + (l % 2);
    debug_assert_eq!(half_l + x_out_num_variables, l - NUM_OF_ROUNDS);

    // Optimization number 3: Transpose the polynomial to improve cache locality.
    // 1 . Transpose the polynomial befoere entering the parallel loop.
    let transposed_poly = transpose_poly_for_svo(poly, l, x_out_num_variables, half_l);

    // Parallelize the outer loop over `x_out`
    (0..1 << x_out_num_variables)
        .into_par_iter()
        .map(|x_out| {
            // Each thread will compute its own set of local accumulators.
            // This avoids mutable state sharing and the need for locks.
            let mut local_accumulators = Accumulators::<EF>::new_empty();

            // This inner part remains the same, but operates on local variables.
            let mut temp_accumulators: Vec<EF> = vec![EF::ZERO; 27];
            let mut p_evals_buffer = [F::ZERO; 27];
            let num_x_in = 1 << half_l;

            for x_in in 0..num_x_in {
                // 2. Read a contiguous block instead of jumping through memory.
                let block_start = (x_out * num_x_in + x_in) * 8;
                let current_evals_arr: [F; 8] = transposed_poly[block_start..block_start + 8]
                    .try_into()
                    .unwrap();

                compute_p_beta(&current_evals_arr, &mut p_evals_buffer);
                let e_in_value = e_in[x_in];

                for (accumulator, &p_eval) in
                    temp_accumulators.iter_mut().zip(p_evals_buffer.iter())
                {
                    *accumulator += e_in_value * p_eval;
                }
            }

            // hardcoded accumulator distribution
            // This now populates the `local_accumulators` for this specific `x_out`.
            let temp_acc = &temp_accumulators;
            let e_out_2 = e_out[2][x_out];

            // Pre-fetch e_out values to avoid repeated indexing and allocations.
            let e0_0 = e_out[0][(0 << x_out_num_variables) | x_out];
            let e0_1 = e_out[0][(1 << x_out_num_variables) | x_out];
            let e0_2 = e_out[0][(2 << x_out_num_variables) | x_out];
            let e0_3 = e_out[0][(3 << x_out_num_variables) | x_out];
            let e1_0 = e_out[1][(0 << x_out_num_variables) | x_out];
            let e1_1 = e_out[1][(1 << x_out_num_variables) | x_out];

            // Now we do not use the idx4 function since we are directly computing the indices.

            // beta_index = 0; b=(0,0,0);
            local_accumulators.accumulate(0, 0, e0_0 * temp_acc[0]); // y=0<<1|0=0
            local_accumulators.accumulate(1, 0, e1_0 * temp_acc[0]); // y=0
            local_accumulators.accumulate(2, 0, e_out_2 * temp_acc[0]);

            // beta_index = 1; b=(0,0,1);
            local_accumulators.accumulate(0, 0, e0_1 * temp_acc[1]); // y=0<<1|1=1
            local_accumulators.accumulate(1, 0, e1_1 * temp_acc[1]); // y=1
            local_accumulators.accumulate(2, 1, e_out_2 * temp_acc[1]);

            // beta_index = 2; b=(0,0,2);
            local_accumulators.accumulate(2, 2, e_out_2 * temp_acc[2]);

            // beta_index = 3; b=(0,1,0);
            local_accumulators.accumulate(0, 0, e0_2 * temp_acc[3]); // y=1<<1|0=2
            local_accumulators.accumulate(1, 1, e1_0 * temp_acc[3]); // y=0
            local_accumulators.accumulate(2, 3, e_out_2 * temp_acc[3]);

            // beta_index = 4; b=(0,1,1);
            local_accumulators.accumulate(0, 0, e0_3 * temp_acc[4]); // y=1<<1|1=3
            local_accumulators.accumulate(1, 1, e1_1 * temp_acc[4]); // y=1
            local_accumulators.accumulate(2, 4, e_out_2 * temp_acc[4]);

            // beta_index = 5; b=(0,1,2);
            local_accumulators.accumulate(2, 5, e_out_2 * temp_acc[5]);

            // beta_index = 6; b=(0,2,0);
            local_accumulators.accumulate(1, 2, e1_0 * temp_acc[6]); // y=0
            local_accumulators.accumulate(2, 6, e_out_2 * temp_acc[6]);

            // beta_index = 7; b=(0,2,1);
            local_accumulators.accumulate(1, 2, e1_1 * temp_acc[7]); // y=1
            local_accumulators.accumulate(2, 7, e_out_2 * temp_acc[7]);

            // beta_index = 8; b=(0,2,2);
            local_accumulators.accumulate(2, 8, e_out_2 * temp_acc[8]);

            // beta_index = 9; b=(1,0,0);
            local_accumulators.accumulate(0, 1, e0_0 * temp_acc[9]); // y=0<<1|0=0
            local_accumulators.accumulate(1, 3, e1_0 * temp_acc[9]); // y=0
            local_accumulators.accumulate(2, 9, e_out_2 * temp_acc[9]);

            // beta_index = 10; b=(1,0,1);
            local_accumulators.accumulate(0, 1, e0_1 * temp_acc[10]); // y=0<<1|1=1
            local_accumulators.accumulate(1, 3, e1_1 * temp_acc[10]); // y=1
            local_accumulators.accumulate(2, 10, e_out_2 * temp_acc[10]);

            // beta_index = 11; b=(1,0,2);
            local_accumulators.accumulate(2, 11, e_out_2 * temp_acc[11]);

            // beta_index = 12; b=(1,1,0);
            local_accumulators.accumulate(0, 1, e0_2 * temp_acc[12]); // y=1<<1|0=2
            local_accumulators.accumulate(1, 4, e1_0 * temp_acc[12]); // y=0
            local_accumulators.accumulate(2, 12, e_out_2 * temp_acc[12]);

            // beta_index = 13; b=(1,1,1);
            local_accumulators.accumulate(0, 1, e0_3 * temp_acc[13]); // y=1<<1|1=3
            local_accumulators.accumulate(1, 4, e1_1 * temp_acc[13]); // y=1
            local_accumulators.accumulate(2, 13, e_out_2 * temp_acc[13]);

            // beta_index = 14; b=(1,1,2);
            local_accumulators.accumulate(2, 14, e_out_2 * temp_acc[14]);

            // beta_index = 15; b=(1,2,0);
            local_accumulators.accumulate(1, 5, e1_0 * temp_acc[15]); // y=0
            local_accumulators.accumulate(2, 15, e_out_2 * temp_acc[15]);

            // beta_index = 16; b=(1,2,1);
            local_accumulators.accumulate(1, 5, e1_1 * temp_acc[16]); // y=1
            local_accumulators.accumulate(2, 16, e_out_2 * temp_acc[16]);

            // beta_index = 17; b=(1,2,2);
            local_accumulators.accumulate(2, 17, e_out_2 * temp_acc[17]);

            // beta_index = 18; b=(2,0,0);
            local_accumulators.accumulate(1, 6, e1_0 * temp_acc[18]); // y=0
            local_accumulators.accumulate(2, 18, e_out_2 * temp_acc[18]);

            // beta_index = 19; b=(2,0,1);
            local_accumulators.accumulate(1, 6, e1_1 * temp_acc[19]); // y=1
            local_accumulators.accumulate(2, 19, e_out_2 * temp_acc[19]);

            // beta_index = 20; b=(2,0,2);
            local_accumulators.accumulate(2, 20, e_out_2 * temp_acc[20]);

            // beta_index = 21; b=(2,1,0);
            local_accumulators.accumulate(1, 7, e1_0 * temp_acc[21]); // y=0
            local_accumulators.accumulate(2, 21, e_out_2 * temp_acc[21]);

            // beta_index = 22; b=(2,1,1);
            local_accumulators.accumulate(1, 7, e1_1 * temp_acc[22]); // y=1
            local_accumulators.accumulate(2, 22, e_out_2 * temp_acc[22]);

            // beta_index = 23; b=(2,1,2);
            local_accumulators.accumulate(2, 23, e_out_2 * temp_acc[23]);

            // beta_index = 24; b=(2,2,0);
            local_accumulators.accumulate(1, 8, e1_0 * temp_acc[24]); // y=0
            local_accumulators.accumulate(2, 24, e_out_2 * temp_acc[24]);

            // beta_index = 25; b=(2,2,1);
            local_accumulators.accumulate(1, 8, e1_1 * temp_acc[25]); // y=1
            local_accumulators.accumulate(2, 25, e_out_2 * temp_acc[25]);

            // beta_index = 26; b=(2,2,2);
            local_accumulators.accumulate(2, 26, e_out_2 * temp_acc[26]);

            // Return the computed local accumulators for this thread.
            local_accumulators
        })
        // Reduce the results from all threads into a single Accumulators struct.
        .reduce(
            || Accumulators::<EF>::new_empty(),
            |mut a, b| {
                for (round_a, round_b) in a.accumulators.iter_mut().zip(b.accumulators.iter()) {
                    for (acc_a, acc_b) in round_a.iter_mut().zip(round_b.iter()) {
                        *acc_a += *acc_b;
                    }
                }
                a
            },
        )
}

// Procedure 9. Page 37.
fn compute_accumulators_eq<F: Field, EF: ExtensionField<F>>(
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

            // Hardcoded accumulator distribution
            // This populates the `local_accumulators` for this specific `x_out`.
            let temp_acc = &temp_accumulators;

            // Pre-fetch e_out values to avoid repeated indexing and allocations.

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

            // Now we do not use the idx4 function since we are directly computing the indices.
            // We go through each beta = (v, u, y) in {0, 1}^3. We don't compute the cases where the digits are
            // infinity because we won't need them.
            // Recall that in `v || u` determines the accumulator's index, `y` determines the e_out facor, and the
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
        // Reduce the results from all threads into a single Accumulators struct.
        .reduce(
            || Accumulators::<EF>::new_empty(),
            |mut a, b| {
                for (round_a, round_b) in a.accumulators.iter_mut().zip(b.accumulators.iter()) {
                    for (acc_a, acc_b) in round_a.iter_mut().zip(round_b.iter()) {
                        *acc_a += *acc_b;
                    }
                }
                a
            },
        )
}

// Given a point w = (w_1, ..., w_l), it returns the evaluations of eq(w, x) for all x in {0, 1}^l.
pub fn eval_eq_in_hypercube<F: Field>(point: &Vec<F>) -> Vec<F> {
    let n = point.len();
    let mut evals = F::zero_vec(1 << n);
    eval_eq::<_, _, false>(point, &mut evals, F::ONE);
    evals
}

// Esta funcion es una copia de eval_eq() en poly/multilinear.rs
// Dado p y q, devuelve eq(p, q).
pub fn eval_eq_in_point<F: Field>(p: &[F], q: &[F]) -> F {
    let mut acc = F::ONE;
    for (&l, &r) in p.into_iter().zip(q) {
        acc *= F::ONE + l * r.double() - l - r;
    }
    acc
}

pub fn compute_linear_function<F: Field>(w: &[F], r: &[F]) -> [F; 2] {
    let round = w.len();
    debug_assert!(r.len() == round - 1);

    let mut const_eq: F = F::ONE;
    if round != 1 {
        const_eq = eval_eq_in_point(&w[..round - 1], r);
    }
    let w_i = w.last().unwrap();

    // Evaluacion de eq(w,X) en [eq(w,0),eq(w,1)]
    [const_eq * (F::ONE - *w_i), const_eq * *w_i]
}

fn get_evals_from_l_and_t<F: Field>(l: &[F; 2], t: &[F]) -> [F; 2] {
    [
        t[0] * l[0],                   // s(0)
        (t[1] - t[0]) * (l[1] - l[0]), //s(inf) -> l(inf) = l(1) - l(0)
    ]
}

// Algorithm 6. Page 19.
// Compute three sumcheck rounds using the small value optimizaition and split-eq accumulators.
// It Returns the three challenges r_1, r_2, r_3(TODO: creo que debería devolver también los polys foldeados).
pub fn small_value_sumcheck_three_rounds_eq<Challenger, F: Field, EF: ExtensionField<F>>(
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
    let accumulators = compute_accumulators_eq(poly, e_in, e_out);
    // println!("Accumulators: {:?}", accumulators);

    // ------------------   Round 1   ------------------

    // 1. For u in {0, 1, inf} compute t_1(u)
    // Recall: In round 1, t_1(u) = A_1(u).
    let t_1_evals = accumulators.get_accumulators_for_round(0);

    // 2. For u in {0, 1, inf} compute S_1(u) = t_1(u) * l_1(u).

    // We compute l_1(0) and l_1(1)
    let linear_1_evals = compute_linear_function(&w.0[..1], &[]);

    // We compute S_1(0) and S_1(inf)
    let round_poly_evals = get_evals_from_l_and_t(&linear_1_evals, &t_1_evals);

    // 3. Send S_1(u) to the verifier.d
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

    // 1. For u in {0, 1, inf} compute t_2(u).
    // First we take the accumulators A_2(v, u).
    // There are 9 accumulators, since v in {0, 1, inf} and u in {0, 1, inf}.
    let accumulators_round_2 = accumulators.get_accumulators_for_round(1);

    let mut t_2_evals = [EF::ZERO; 2];

    // // OLD VERSION:
    // t_2_evals[0] += lagrange_evals_r_1[0] * accumulators_round_2[0];
    // t_2_evals[0] += lagrange_evals_r_1[1] * accumulators_round_2[3];

    // t_2_evals[1] += lagrange_evals_r_1[0] * accumulators_round_2[1];
    // t_2_evals[1] += lagrange_evals_r_1[1] * accumulators_round_2[4];

    t_2_evals[0] += lagrange_evals_r_1[0] * accumulators_round_2[0];
    t_2_evals[0] += lagrange_evals_r_1[1] * accumulators_round_2[2];

    t_2_evals[1] += lagrange_evals_r_1[0] * accumulators_round_2[1];
    t_2_evals[1] += lagrange_evals_r_1[1] * accumulators_round_2[3];

    // We compute l_2(0) and l_12inf)
    let linear_2_evals = compute_linear_function(&w.0[..2], &[r_1]);

    // We compute S_2(u)
    let round_poly_evals = get_evals_from_l_and_t(&linear_2_evals, &t_2_evals);

    // 3. Send S_2(u) to the verifier.
    // TODO: En realidad no hace falta mandar S_2(1) porque se deduce usando S_2(0).
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

    // 1. For u in {0, 1, inf} compute t_3(u).

    // First we take the accumulators A_2(v, u).
    // There are 27 accumulators, since v in {0, 1, inf}^2 and u in {0, 1, inf}.
    let accumulators_round_3 = accumulators.get_accumulators_for_round(2);

    let mut t_3_evals = [EF::ZERO; 2];

    // // OLD VERSION:
    // t_3_evals[0] += lagrange_evals_r_2[0] * accumulators_round_3[0]; // (0,0,u=0)
    // t_3_evals[0] += lagrange_evals_r_2[1] * accumulators_round_3[3]; // (1,0,u=0)
    // t_3_evals[0] += lagrange_evals_r_2[2] * accumulators_round_3[9]; // (0,1,u=0)
    // t_3_evals[0] += lagrange_evals_r_2[3] * accumulators_round_3[12]; // (1,1,u=0)

    // t_3_evals[1] += lagrange_evals_r_2[0] * accumulators_round_3[1]; // (0,0,u=1)
    // t_3_evals[1] += lagrange_evals_r_2[1] * accumulators_round_3[4]; // (1,0,u=1)
    // t_3_evals[1] += lagrange_evals_r_2[2] * accumulators_round_3[10]; // (0,1,u=1)
    // t_3_evals[1] += lagrange_evals_r_2[3] * accumulators_round_3[13]; // (1,1,u=1)

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

    // We compute S_3(u)
    let round_poly_evals = get_evals_from_l_and_t(&linear_3_evals, &t_3_evals);

    // 3. Send S_3(u) to the verifier.
    // TODO: En realidad no hace falta mandar S_3(1) porque se dedecue usando S_3(0).
    prover_state.add_extension_scalars(&round_poly_evals);

    let r_3: EF = prover_state.sample();

    let eval_1 = *sum - round_poly_evals[0];
    *sum = round_poly_evals[1] * r_3.square()
        + (eval_1 - round_poly_evals[0] - round_poly_evals[1]) * r_3
        + round_poly_evals[0];

    (r_1, r_2, r_3)
}

pub fn algorithm_2<Challenger, F: Field, EF: ExtensionField<F>>(
    prover_state: &mut ProverState<F, EF, Challenger>,
    poly: &EvaluationsList<F>,
    w: &MultilinearPoint<EF>,
    sum: &mut EF,
    challenges: &mut Vec<EF>,
) -> Vec<EF>
where
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    let l = poly.num_variables();
    let eq = eval_eq_in_hypercube(&w.0);
    let eq_r = eval_eq_in_hypercube(challenges);

    let chunk_size = 1 << (l - 4);
    let step = 1 << (l - 3);

    let mut folded_poly = Vec::with_capacity(chunk_size * 2);

    let eval_zero: EF = (0..chunk_size)
        .map(|x| {
            // Compute eq_sum and p_sum in a single pass
            let (eq_sum, p_sum) = eq
                .iter()
                .skip(x)
                .step_by(step)
                .zip(poly.iter().skip(x).step_by(step))
                .zip(eq_r.iter())
                .take(8)
                .fold(
                    (EF::ZERO, EF::ZERO),
                    |(eq_acc, p_acc), ((eq_val, poly_val), eq_r_val)| {
                        (eq_acc + *eq_val * *eq_r_val, p_acc + *eq_r_val * *poly_val)
                    },
                );

            folded_poly.push(p_sum);
            eq_sum * p_sum
        })
        .sum();

    let eval_inf: EF = (0..chunk_size)
        .map(|x| {
            let offset = x + chunk_size;

            let (eq_sum, p_one) = eq
                .iter()
                .skip(x)
                .step_by(step)
                .zip(eq.iter().skip(offset).step_by(step))
                .zip(poly.iter().skip(offset).step_by(step))
                .zip(eq_r.iter())
                .take(8)
                .fold(
                    (EF::ZERO, EF::ZERO),
                    |(eq_acc, p_acc), (((eq_0, eq_1), poly_val), eq_r_val)| {
                        let eq_diff = *eq_1 - *eq_0;
                        (eq_acc + eq_diff * *eq_r_val, p_acc + *eq_r_val * *poly_val)
                    },
                );

            folded_poly.push(p_one);
            let p_diff = p_one - folded_poly[x];
            eq_sum * p_diff
        })
        .sum();

    prover_state.add_extension_scalars(&[eval_zero, eval_inf]);

    // Update the claimed sum: S_4(r_4)
    let r_4 = prover_state.sample();
    challenges.push(r_4);

    let eval_1 = *sum - eval_zero;
    *sum = eval_inf * r_4.square() + (eval_1 - eval_zero - eval_inf) * r_4 + eval_zero;

    folded_poly
}

// Algorithm 5. Page 18.
pub fn algorithm_5<Challenger, F: Field, EF: ExtensionField<F>>(
    prover_state: &mut ProverState<F, EF, Challenger>,
    poly: &mut EvaluationsList<EF>,
    w: &MultilinearPoint<EF>,
    challenges: &mut Vec<EF>,
    sum: &mut EF,
) where
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    let num_vars = w.num_variables();
    let half_l = num_vars / 2;

    // We compute eq(w_{l/2 + 1}, ...,  w_l ; x_R) for all x_R in {0, 1}^{l/2}
    // These evaluations don't depend on the round i, so they are computed outside the loop.
    let eq_r = eval_eq_in_hypercube(&w.0[half_l..].to_vec());
    let num_vars_x_r = eq_r.len().ilog2() as usize;

    // The number of variables of x_R is: l/2 if l is even and l/2 + 1 if l is odd.
    debug_assert!(num_vars_x_r == half_l + (num_vars % 2));

    // Loop for the final rounds, from l_0+1 (in our case 4) to the end.
    let current_round = challenges.len() + 1;
    for i in current_round..num_vars + 1 {
        let mut t = Vec::new();

        // We get the number of variables of `poly` in the current round.
        let num_vars_poly_current = poly.num_variables();

        // 1. We compute t_i(u) for u in {0, 1}.
        if i <= half_l {
            // We compute eq(w_{i + 1}, ...,  w_{l/2} ; x_L) for all x_L in {0, 1}^{l/2 - i}
            let eq_l = eval_eq_in_hypercube(&w.0[i..half_l].to_vec());

            // For t_i(0), we need p(r_[<i-1], 0, x_L, x_R).
            let t_0: EF = (0..(1 << num_vars_x_r))
                .map(|x_r| {
                    let sum_l: EF = (0..eq_l.len())
                        .map(|x_l| eq_l[x_l] * (poly.as_slice()[(x_l << num_vars_x_r) | x_r]))
                        .sum();
                    eq_r[x_r] * sum_l
                })
                .sum();

            // For t_i(1), we need p(r_[<i-1], 1, x_L, x_R)
            let t_1: EF = (0..(1 << num_vars_x_r))
                .map(|x_r| {
                    let sum_l: EF = (0..eq_l.len())
                        .map(|x_l| {
                            eq_l[x_l]
                                * (poly.as_slice()[(1 << (num_vars_poly_current - 1))
                                    | (x_l << num_vars_x_r)
                                    | x_r])
                        })
                        .sum();
                    eq_r[x_r] * sum_l
                })
                .sum();
            t.push(t_0);
            t.push(t_1);
        } else {
            // Case i > l/2: Only one part of the eq evaluations remains to be processed.
            let eq = eval_eq_in_hypercube(&w.0[i..num_vars].to_vec());
            // For t_i(0), we need p(r_[<i-1], 0, x)
            let t_0: EF = (0..(1 << (num_vars_poly_current - 1)))
                .map(|x| eq[x] * (poly.as_slice()[x]))
                .sum();

            // For t_i(1), we need p(r_[<i-1], 1, x)
            let t_1: EF = (0..(1 << (num_vars_poly_current - 1)))
                .map(|x| eq[x] * (poly.as_slice()[(1 << (num_vars_poly_current - 1)) | x]))
                .sum();
            t.push(t_0);
            t.push(t_1);
        }

        // 2. We compute S_i(u) = t_i(u) * l_i(u) for u in {0, inf}.
        // We compute l_i(0) and l_i(inf)
        let linear_evals = compute_linear_function(&w.0[..i], &challenges);

        // We compute S_i(u)
        let round_poly_evals = get_evals_from_l_and_t(&linear_evals, &t);

        // 3. Send S_i(u) to the verifier.
        prover_state.add_extension_scalars(&round_poly_evals);

        // 4. Receive the challenge r_i from the verifier.
        let r_i: EF = prover_state.sample();
        challenges.push(r_i);

        // 5. Fold and update the poly.
        poly.compress_svo(r_i);

        // Update the claimed sum.
        let eval_1 = *sum - round_poly_evals[0];
        *sum = round_poly_evals[1] * r_i.square()
            + (eval_1 - round_poly_evals[0] - round_poly_evals[1]) * r_i
            + round_poly_evals[0];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
        sumcheck::sumcheck_small_value::compute_accumulators,
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

    // This tests is outdated and should be removed
    #[test]
    fn test_compute_accumulators_eq_l_10() {
        // We'll use polynomials of 10 variables.
        let l = 10;

        // w = [w0, w2, ..., w9]
        // Each w_i is a random extension field element built from 4 random field elements.
        let w: Vec<EF> = (0..l).map(|_| get_random_ef()).collect();
        let w = MultilinearPoint::new(w);
        // We build a random multilinear polynomial of 10 variables, using 2^10 evaluations in the hypercube {0,1}^10
        let poly = EvaluationsList::new((0..(1 << l)).map(|_| get_random_f()).collect());

        // We precompute E_in and E_out
        let e_in = precompute_e_in(&w);
        let e_out = precompute_e_out(&w);

        // We compute the accumulators.
        let accumulators = compute_accumulators_eq(&poly, e_in.clone(), e_out.clone());

        // A_3(0,0,0)
        let eq_w3_w4 = eval_eq_in_hypercube(&w.0[3..5].to_vec());
        let eq_w5_to_w9 = eval_eq_in_hypercube(&w.0[5..].to_vec());

        let expected_accumulator = (0..4)
            .map(|i| {
                eq_w5_to_w9
                    .iter()
                    .zip(poly.iter().skip(i * 32).take(32))
                    .map(|(eq, p)| *eq * *p)
                    .sum::<EF>()
            })
            .zip(eq_w3_w4.iter())
            .map(|(sum, eq)| sum * *eq)
            .sum::<EF>();

        assert_eq!(
            expected_accumulator,
            accumulators.get_accumulators_for_round(2)[0]
        );

        // A_3(1,0,1):
        let expected_accumulator = (0..4)
            .map(|i| {
                eq_w5_to_w9
                    .iter()
                    .zip(poly.iter().skip(640 + (i * 32)).take(32))
                    .map(|(eq, p)| *eq * *p)
                    .sum::<EF>()
            })
            .zip(eq_w3_w4.iter())
            .map(|(sum, eq)| sum * *eq)
            .sum::<EF>();

        assert_eq!(
            expected_accumulator,
            accumulators.get_accumulators_for_round(2)[10]
        );

        // We now compute A_3(1, inf, 0):

        // We compute p(1, inf, 0, b_L, b_R) for all b_L in {0, 1}^2 and b_R in {0, 1}^5.
        // Recall that p(1, inf, 0, b_L, b_R) = p(1, 1, 0, b_L, b_R) - p(1, 0, 0, b_L, b_R)
        let p_evals: [Vec<F>; 4] = [
            poly.iter()
                .skip(768)
                .take(32)
                .zip(poly.iter().skip(512).take(32))
                .map(|(p1, p0)| *p1 - *p0)
                .collect::<Vec<F>>(),
            poly.iter()
                .skip(800)
                .take(32)
                .zip(poly.iter().skip(544).take(32))
                .map(|(p1, p0)| *p1 - *p0)
                .collect::<Vec<F>>(),
            poly.iter()
                .skip(832)
                .take(32)
                .zip(poly.iter().skip(576).take(32))
                .map(|(p1, p0)| *p1 - *p0)
                .collect::<Vec<F>>(),
            poly.iter()
                .skip(864)
                .take(32)
                .zip(poly.iter().skip(608).take(32))
                .map(|(p1, p0)| *p1 - *p0)
                .collect::<Vec<F>>(),
        ];

        let expected_accumulator = (0..4)
            .map(|i| {
                eq_w5_to_w9
                    .iter()
                    .zip(p_evals[i].clone())
                    .map(|(eq, p)| *eq * p)
                    .sum::<EF>()
            })
            .zip(eq_w3_w4.iter())
            .map(|(sum, eq)| sum * *eq)
            .sum::<EF>();

        assert_eq!(
            expected_accumulator,
            accumulators.get_accumulators_for_round(2)[15]
        );

        // We compute now A_2(0, 0):
        let eq_w2_w3_w4 = eval_eq_in_hypercube(&w.0[2..5].to_vec());

        // We compute A_2(0, 0)
        let expected_accumulator = (0..8)
            .map(|i| {
                eq_w5_to_w9
                    .iter()
                    .zip(poly.iter().skip(i * 32).take(32))
                    .map(|(eq, p)| *eq * *p)
                    .sum::<EF>()
            })
            .zip(eq_w2_w3_w4.iter())
            .map(|(sum, eq)| sum * *eq)
            .sum::<EF>();

        assert_eq!(
            expected_accumulator,
            accumulators.get_accumulators_for_round(1)[0]
        );

        // A_2(inf, 0)
        let p_evals_inf_0: Vec<Vec<F>> = (0..8)
            .map(|i| {
                poly.iter()
                    .skip(512 + (i * 32))
                    .take(32)
                    .zip(poly.iter().skip(i * 32).take(32))
                    .map(|(p_1, p_0)| *p_1 - *p_0)
                    .collect()
            })
            .collect();

        let expected_accumulator = (0..8)
            .map(|i| {
                eq_w5_to_w9
                    .iter()
                    .zip(p_evals_inf_0[i].clone())
                    .map(|(eq, p)| *eq * p)
                    .sum::<EF>()
            })
            .zip(eq_w2_w3_w4.iter())
            .map(|(sum, eq)| sum * *eq)
            .sum::<EF>();

        assert_eq!(
            expected_accumulator,
            accumulators.get_accumulators_for_round(1)[6]
        );

        // A_1(u)
        let eq_w1_w2_w3_w4 = eval_eq_in_hypercube(&w.0[1..5].to_vec());

        // We compute A_1(0)
        let expected_accumulator = (0..16)
            .map(|i| {
                eq_w5_to_w9
                    .iter()
                    .zip(poly.iter().skip(i * 32).take(32))
                    .map(|(eq, p)| *eq * *p)
                    .sum::<EF>()
            })
            .zip(eq_w1_w2_w3_w4.iter())
            .map(|(sum, eq)| sum * *eq)
            .sum::<EF>();

        assert_eq!(
            expected_accumulator,
            accumulators.get_accumulators_for_round(0)[0]
        );

        // A_1(1)
        let expected_accumulator = (0..16)
            .map(|i| {
                eq_w5_to_w9
                    .iter()
                    .zip(poly.iter().skip(512 + (i * 32)).take(32))
                    .map(|(eq, p)| *eq * *p)
                    .sum::<EF>()
            })
            .zip(eq_w1_w2_w3_w4.iter())
            .map(|(sum, eq)| sum * *eq)
            .sum::<EF>();

        assert_eq!(
            expected_accumulator,
            accumulators.get_accumulators_for_round(0)[1]
        );

        // A_1(inf)
        let p_evals_inf: Vec<Vec<_>> = (0..16)
            .map(|i| {
                poly.iter()
                    .skip(512 + (i * 32))
                    .take(32)
                    .zip(poly.iter().skip(i * 32).take(32))
                    .map(|(p_1, p_0)| *p_1 - *p_0)
                    .collect()
            })
            .collect();

        let expected_accumulator = (0..16)
            .map(|i| {
                eq_w5_to_w9
                    .iter()
                    .zip(p_evals_inf[i].clone())
                    .map(|(eq, p)| *eq * p)
                    .sum::<EF>()
            })
            .zip(eq_w1_w2_w3_w4.iter())
            .map(|(sum, eq)| sum * *eq)
            .sum::<EF>();

        assert_eq!(
            expected_accumulator,
            accumulators.get_accumulators_for_round(0)[2]
        );

        // assert_eq!(expected_accumulator, accumulators[0].accumulators[0]);
    }

    // This tests is outdated and should be removed
    #[test]
    fn test_compute_accumulators_eq_l_11() {
        // We'll use polynomials of 10 variables.
        let l = 11;

        // w = [w0, w2, ..., w9]
        // Each w_i is a random extension field element built from 4 random field elements.
        let w: Vec<EF> = (0..l).map(|_| get_random_ef()).collect();
        let w = MultilinearPoint(w);

        // We build a random multilinear polynomial of 10 variables, using 2^10 evaluations in the hypercube {0,1}^10
        let poly = EvaluationsList::new((0..(1 << l)).map(|_| get_random_f()).collect());

        // We precompute E_in and E_out
        let e_in = precompute_e_in(&w);
        let e_out = precompute_e_out(&w);

        // We compute the accumulators.
        let accumulators = compute_accumulators_eq(&poly, e_in.clone(), e_out.clone());

        // A_3(0,0,0)
        let eq_w3_w4 = eval_eq_in_hypercube(&w.0[3..5].to_vec());
        let eq_w5_to_w10 = eval_eq_in_hypercube(&w.0[5..].to_vec());

        let expected_accumulator = (0..4)
            .map(|i| {
                eq_w5_to_w10
                    .iter()
                    .zip(poly.iter().skip(i * 64).take(64))
                    .map(|(eq, p)| *eq * *p)
                    .sum::<EF>()
            })
            .zip(eq_w3_w4.iter())
            .map(|(sum, eq)| sum * *eq)
            .sum::<EF>();

        assert_eq!(
            expected_accumulator,
            accumulators.get_accumulators_for_round(2)[0]
        );

        // A_3(1,0,1):
        let expected_accumulator = (0..4)
            .map(|i| {
                eq_w5_to_w10
                    .iter()
                    .zip(poly.iter().skip(1280 + (i * 64)).take(64))
                    .map(|(eq, p)| *eq * *p)
                    .sum::<EF>()
            })
            .zip(eq_w3_w4.iter())
            .map(|(sum, eq)| sum * *eq)
            .sum::<EF>();

        assert_eq!(
            expected_accumulator,
            accumulators.get_accumulators_for_round(2)[10]
        );

        // We now compute A_3(1, inf, 0):

        // We compute p(1, inf, 0, b_L, b_R) for all b_L in {0, 1}^2 and b_R in {0, 1}^5.
        // Recall that p(1, inf, 0, b_L, b_R) = p(1, 1, 0, b_L, b_R) - p(1, 0, 0, b_L, b_R)
        let p_evals: [Vec<F>; 4] = [
            poly.iter()
                .skip(1536)
                .take(64)
                .zip(poly.iter().skip(1024).take(64))
                .map(|(p1, p0)| *p1 - *p0)
                .collect::<Vec<F>>(),
            poly.iter()
                .skip(1600)
                .take(64)
                .zip(poly.iter().skip(1088).take(64))
                .map(|(p1, p0)| *p1 - *p0)
                .collect::<Vec<F>>(),
            poly.iter()
                .skip(1664)
                .take(64)
                .zip(poly.iter().skip(1152).take(64))
                .map(|(p1, p0)| *p1 - *p0)
                .collect::<Vec<F>>(),
            poly.iter()
                .skip(1728)
                .take(64)
                .zip(poly.iter().skip(1216).take(64))
                .map(|(p1, p0)| *p1 - *p0)
                .collect::<Vec<F>>(),
        ];

        let expected_accumulator = (0..4)
            .map(|i| {
                eq_w5_to_w10
                    .iter()
                    .zip(p_evals[i].clone())
                    .map(|(eq, p)| *eq * p)
                    .sum::<EF>()
            })
            .zip(eq_w3_w4.iter())
            .map(|(sum, eq)| sum * *eq)
            .sum::<EF>();

        assert_eq!(
            expected_accumulator,
            accumulators.get_accumulators_for_round(2)[15]
        );

        // Round 2:
        let eq_w2_w3_w4 = eval_eq_in_hypercube(&w.0[2..5].to_vec());

        // A_2(0, 0):
        let expected_accumulator = (0..8)
            .map(|i| {
                eq_w5_to_w10
                    .iter()
                    .zip(poly.iter().skip(i * 64).take(64))
                    .map(|(eq, p)| *eq * *p)
                    .sum::<EF>()
            })
            .zip(eq_w2_w3_w4.iter())
            .map(|(sum, eq)| sum * *eq)
            .sum::<EF>();

        assert_eq!(
            expected_accumulator,
            accumulators.get_accumulators_for_round(1)[0]
        );

        // A_2(inf, 0):
        let p_evals_inf_0: Vec<Vec<_>> = (0..8)
            .map(|i| {
                poly.iter()
                    .skip(1024 + (i * 64))
                    .take(64)
                    .zip(poly.iter().skip(i * 64).take(64))
                    .map(|(p_1, p_0)| *p_1 - *p_0)
                    .collect()
            })
            .collect();

        let expected_accumulator = (0..8)
            .map(|i| {
                eq_w5_to_w10
                    .iter()
                    .zip(p_evals_inf_0[i].clone())
                    .map(|(eq, p)| *eq * p)
                    .sum::<EF>()
            })
            .zip(eq_w2_w3_w4.iter())
            .map(|(sum, eq)| sum * *eq)
            .sum::<EF>();

        assert_eq!(
            expected_accumulator,
            accumulators.get_accumulators_for_round(1)[6]
        );

        // Round 3:
        let eq_w1_w2_w3_w4 = eval_eq_in_hypercube(&w.0[1..5].to_vec());

        // A_1(0)
        let expected_accumulator = (0..16)
            .map(|i| {
                eq_w5_to_w10
                    .iter()
                    .zip(poly.iter().skip(i * 64).take(64))
                    .map(|(eq, p)| *eq * *p)
                    .sum::<EF>()
            })
            .zip(eq_w1_w2_w3_w4.iter())
            .map(|(sum, eq)| sum * *eq)
            .sum::<EF>();

        assert_eq!(
            expected_accumulator,
            accumulators.get_accumulators_for_round(0)[0]
        );

        // A_1(1)
        let expected_accumulator = (0..16)
            .map(|i| {
                eq_w5_to_w10
                    .iter()
                    .zip(poly.iter().skip(1024 + (i * 64)).take(64))
                    .map(|(eq, p)| *eq * *p)
                    .sum::<EF>()
            })
            .zip(eq_w1_w2_w3_w4.iter())
            .map(|(sum, eq)| sum * *eq)
            .sum::<EF>();

        assert_eq!(
            expected_accumulator,
            accumulators.get_accumulators_for_round(0)[1]
        );

        // A_1(inf)
        let p_evals_inf: Vec<Vec<_>> = (0..16)
            .map(|i| {
                poly.iter()
                    .skip(1024 + (i * 64))
                    .take(64)
                    .zip(poly.iter().skip(i * 64).take(64))
                    .map(|(p_1, p_0)| *p_1 - *p_0)
                    .collect()
            })
            .collect();

        let expected_accumulator = (0..16)
            .map(|i| {
                eq_w5_to_w10
                    .iter()
                    .zip(p_evals_inf[i].clone())
                    .map(|(eq, p)| *eq * p)
                    .sum::<EF>()
            })
            .zip(eq_w1_w2_w3_w4.iter())
            .map(|(sum, eq)| sum * *eq)
            .sum::<EF>();

        assert_eq!(
            expected_accumulator,
            accumulators.get_accumulators_for_round(0)[2]
        );
    }

    fn get_random_f() -> F {
        let mut rng = rand::rng();
        F::from_u32(rng.next_u32())
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

    fn naive_sumcheck_verification<F: Field, EF: ExtensionField<F>>(
        w: Vec<EF>,
        poly: EvaluationsList<F>,
    ) -> EF {
        let eq = eval_eq_in_hypercube(&w);
        poly.iter().zip(eq.iter()).map(|(p, e)| *e * *p).sum()
    }

    fn get_evals_from_l_and_t<F: Field>(l: &[F; 2], t: &[F]) -> [F; 2] {
        [
            t[0] * l[0],                   // s(0)
            (t[1] - t[0]) * (l[1] - l[0]), // s(inf) -> l(inf) = l(1) - l(0)
        ]
    }

    #[test]
    fn compare_sv_vs_eq() {
        let poly = EvaluationsList::new((0..512).map(|_| get_random_f()).collect());
        let w: Vec<F> = (0..9).map(|_| get_random_f()).collect();
        let w = MultilinearPoint(w);

        let r_1 = get_random_f();
        let r_2 = get_random_f();

        // -------------  EQ  -------------
        let e_in = precompute_e_in(&w);
        let e_out = precompute_e_out(&w);

        // We compute all the accumulators A_i(v, u).
        let accumulators = compute_accumulators_eq(&poly, e_in, e_out);

        // ------------------   Round 1   ------------------

        // 1. For u in {0, 1, inf} compute t_1(u)
        // Recall: In round 1, t_1(u) = A_1(u).
        let t_1_evals = accumulators.get_accumulators_for_round(0);

        // 2. For u in {0, 1, inf} compute S_1(u) = t_1(u) * l_1(u).

        // We compute l_1(0) and l_1(1)
        let linear_1_evals = compute_linear_function(&w.0[..1], &[]);

        // We compute S_1(0) and S_1(inf)
        let round_poly_evals = get_evals_from_l_and_t(&linear_1_evals, &t_1_evals);

        println!("ROUND 1 EQ: {:?}", round_poly_evals);

        // 5. Compte R_2 = [L_0(r_1), L_1(r_1), L_inf(r_1)]
        // L_0 (x) = 1 - x
        // L_1 (x) = x
        let lagrange_evals_r_1 = [-r_1 + F::ONE, r_1];

        // ------------------ Round 2 ------------------

        // 1. For u in {0, 1, inf} compute t_2(u).
        // First we take the accumulators A_2(v, u).
        // There are 9 accumulators, since v in {0, 1, inf} and u in {0, 1, inf}.
        let accumulators_round_2 = accumulators.get_accumulators_for_round(1);

        let mut t_2_evals = [F::ZERO; 2];

        t_2_evals[0] += lagrange_evals_r_1[0] * accumulators_round_2[0];
        t_2_evals[0] += lagrange_evals_r_1[1] * accumulators_round_2[3];

        t_2_evals[1] += lagrange_evals_r_1[0] * accumulators_round_2[1];
        t_2_evals[1] += lagrange_evals_r_1[1] * accumulators_round_2[4];

        // 2. For u in {0, 1, inf} compute S_2(u) = t_2(u) * l_2(u).

        // We compute l_2(0) and l_2(inf)
        let linear_2_evals = compute_linear_function(&w.0[..2], &[r_1]);

        // We compute S_2(0) and S_2(inf)
        let round_poly_evals = get_evals_from_l_and_t(&linear_2_evals, &t_2_evals);

        println!("ROUND 2 EQ: {:?}", round_poly_evals);

        // 5. Compute R_3 = [L_00(r_1, r_2), L_01(r_1, r_2), ..., L_{inf inf}(r_1, r_2)]
        // L_00 (x1, x2) = (1 - x1) * (1 - x2)
        // L_01 (x1, x2) = (1 - x1) * x2
        // ...
        // L_{inf inf} (x1, x2) = (x1 - 1) * x1 * (x2 - 1) * x2

        let [l_0, l_1] = lagrange_evals_r_1;
        let one_minus_r_2 = -r_2 + F::ONE;
        // let mul_inf = (r_2 - F::ONE) * r_2;

        let lagrange_evals_r_2 = [
            l_0 * one_minus_r_2, // L_0 0
            l_0 * r_2,           // L_0 1
            l_1 * one_minus_r_2, // L_1 0
            l_1 * r_2,           // L_1 1
        ];

        // Round 3

        // 1. For u in {0, 1, inf} compute t_3(u).

        // First we take the accumulators A_2(v, u).
        // There are 27 accumulators, since v in {0, 1, inf}^2 and u in {0, 1, inf}.
        let accumulators_round_3 = accumulators.get_accumulators_for_round(2);

        let mut t_3_evals = [F::ZERO; 2];

        t_3_evals[0] += lagrange_evals_r_2[0] * accumulators_round_3[0]; // (0,0,u=0)
        t_3_evals[0] += lagrange_evals_r_2[1] * accumulators_round_3[3]; // (1,0,u=0)
        t_3_evals[0] += lagrange_evals_r_2[2] * accumulators_round_3[9]; // (0,1,u=0)
        t_3_evals[0] += lagrange_evals_r_2[3] * accumulators_round_3[12]; // (1,1,u=0)

        t_3_evals[1] += lagrange_evals_r_2[0] * accumulators_round_3[1]; // (0,0,u=1)
        t_3_evals[1] += lagrange_evals_r_2[1] * accumulators_round_3[4]; // (1,0,u=1)
        t_3_evals[1] += lagrange_evals_r_2[2] * accumulators_round_3[10]; // (0,1,u=1)
        t_3_evals[1] += lagrange_evals_r_2[3] * accumulators_round_3[13]; // (1,1,u=1)

        // 2. For u in {0, 1, inf} compute S_3(u) = t_3(u) * l_3(u).

        // We compute l_3(0) and l_3(inf)
        let linear_3_evals = compute_linear_function(&w.0[..3], &[r_1, r_2]);

        // We compute S_3(u)
        let round_poly_evals = get_evals_from_l_and_t(&linear_3_evals, &t_3_evals);

        println!("ROUND 3 EQ: {:?}", round_poly_evals);

        // -------------  P * Q  -------------
        let poly_2 = EvaluationsList::new(eval_eq_in_hypercube(&w.0));

        let round_accumulators = compute_accumulators(&poly, &poly_2);

        let round_poly_evals_1 = &round_accumulators[0].accumulators;

        println!("ROUND 1 PQ: {:?}", round_poly_evals_1);

        let lagrange_evals_r_1 = [-r_1 + F::ONE, r_1, (r_1 - F::ONE) * r_1];

        let accumulators_2 = &round_accumulators[1].accumulators;

        let mut round_poly_evals_2 = [F::ZERO; 3];

        for (lagrange_index, chunk) in accumulators_2.chunks_exact(3).enumerate() {
            round_poly_evals_2[0] += lagrange_evals_r_1[lagrange_index] * chunk[0];
            round_poly_evals_2[1] += lagrange_evals_r_1[lagrange_index] * chunk[1];
            round_poly_evals_2[2] += lagrange_evals_r_1[lagrange_index] * chunk[2];
        }

        let expected_eval = round_poly_evals_1[2] * r_1.square()
            + (round_poly_evals_1[1] - round_poly_evals_1[0] - round_poly_evals_1[2]) * r_1
            + round_poly_evals_1[0];

        assert_eq!(round_poly_evals_2[0] + round_poly_evals_2[1], expected_eval);
        println!("ROUND 2 PQ: {:?}", round_poly_evals_2);

        let l_0 = lagrange_evals_r_1[0];
        let l_1 = lagrange_evals_r_1[1];
        let l_inf = lagrange_evals_r_1[2];

        let lagrange_evals_r_2 = [
            l_0 * (-r_2 + F::ONE),        // L_0 0
            l_0 * r_2,                    // L_0 1
            l_0 * (r_2 - F::ONE) * r_2,   // L_0 inf
            l_1 * (-r_2 + F::ONE),        // L_1 0
            l_1 * r_2,                    // L_1 1
            l_1 * (r_2 - F::ONE) * r_2,   // L_1 inf
            l_inf * (-r_2 + F::ONE),      // L_inf 0
            l_inf * r_2,                  // L_inf 1
            l_inf * (r_2 - F::ONE) * r_2, // L_inf inf
        ];

        let accumulators_3 = &round_accumulators[2].accumulators;

        let mut round_poly_evals_3 = [F::ZERO; 3];

        for (lagrange_index, accumulators_chunk) in accumulators_3.chunks_exact(9).enumerate() {
            round_poly_evals_3[0] += lagrange_evals_r_2[lagrange_index * 3] * accumulators_chunk[0];
            round_poly_evals_3[1] += lagrange_evals_r_2[lagrange_index * 3] * accumulators_chunk[1];
            round_poly_evals_3[2] += lagrange_evals_r_2[lagrange_index * 3] * accumulators_chunk[2];

            round_poly_evals_3[0] +=
                lagrange_evals_r_2[lagrange_index * 3 + 1] * accumulators_chunk[3];
            round_poly_evals_3[1] +=
                lagrange_evals_r_2[lagrange_index * 3 + 1] * accumulators_chunk[4];
            round_poly_evals_3[2] +=
                lagrange_evals_r_2[lagrange_index * 3 + 1] * accumulators_chunk[5];

            round_poly_evals_3[0] +=
                lagrange_evals_r_2[lagrange_index * 3 + 2] * accumulators_chunk[6];
            round_poly_evals_3[1] +=
                lagrange_evals_r_2[lagrange_index * 3 + 2] * accumulators_chunk[7];
            round_poly_evals_3[2] +=
                lagrange_evals_r_2[lagrange_index * 3 + 2] * accumulators_chunk[8];
        }

        let expected_eval = round_poly_evals_2[2] * r_2.square()
            + (round_poly_evals_2[1] - round_poly_evals_2[0] - round_poly_evals_2[2]) * r_2
            + round_poly_evals_2[0];

        assert_eq!(round_poly_evals_3[0] + round_poly_evals_3[1], expected_eval);
        println!("ROUND 3 PQ: {:?}", round_poly_evals_3);
    }
}
