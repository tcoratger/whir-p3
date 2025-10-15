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

// Procedure 9. Page 37.
fn compute_accumulators_eq<F: Field, EF: ExtensionField<F>>(
    poly: &EvaluationsList<F>,
    e_in: Vec<EF>,
    e_out: [Vec<EF>; NUM_OF_ROUNDS],
) -> Accumulators<EF> {
    let l = poly.num_variables();
    let half_l = l / 2;

    let mut accumulators = Accumulators::<EF>::new_empty();

    let x_out_num_variables = half_l - NUM_OF_ROUNDS + (l % 2);
    debug_assert_eq!(half_l + x_out_num_variables, l - NUM_OF_ROUNDS);

    for x_out in 0..1 << (x_out_num_variables) {
        let mut temp_accumulators: Vec<EF> = vec![EF::ZERO; 27];

        for x_in in 0..1 << half_l {
            // We collect the evaluations of p(X_0, X_1, X_2, x_in, x_out) where
            // x_in and x_out are fixed and X_0, X_1, X_2 are variables.
            let start_index = (x_in << x_out_num_variables) | x_out;
            let step_size = 1 << (l - NUM_OF_ROUNDS);

            let current_evals: Vec<F> = poly
                .iter()
                .skip(start_index)
                .step_by(step_size)
                .copied()
                .collect();

            // We compute p(beta, x_in, x_out) for all beta in {0, 1, inf}^3
            let p_evals = compute_p_beta(current_evals);
            let e_in_value = e_in[x_in];

            for (accumulator, &p_eval) in temp_accumulators.iter_mut().zip(&p_evals) {
                *accumulator += e_in_value * p_eval;
            }
        }

        // TODO: This can be hardcoded for better performance.
        for beta_index in 0..27 {
            let [index_1, index_2, index_3] = idx4_v2(beta_index);
            let [_, beta_2, beta_3] = to_base_three_coeff(beta_index);
            let temp_acc = temp_accumulators[beta_index];

            // Accumulator 1: uses y = beta_2 || beta_3
            if let Some(index) = index_1 {
                let y = beta_2 << 1 | beta_3;
                let e_out_value = e_out[0][(y << x_out_num_variables) | x_out];
                accumulators.accumulate(0, index, e_out_value * temp_acc);
            }

            // Accumulator 2: uses y = beta_3
            if let Some(index) = index_2 {
                let y = beta_3;
                let e_out_value = e_out[1][(y << x_out_num_variables) | x_out];
                accumulators.accumulate(1, index, e_out_value * temp_acc);
            }

            // Accumulator 3: uses x_out directly
            if let Some(index) = index_3 {
                accumulators.accumulate(2, index, e_out[2][x_out] * temp_acc);
            }
        }
    }
    accumulators
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

    // ------------------   Round 1   ------------------

    // 1. For u in {0, 1, inf} compute t_1(u)
    // Recall: In round 1, t_1(u) = A_1(u).
    let t_1_evals = accumulators.get_accumulators_for_round(0);

    // 2. For u in {0, 1, inf} compute S_1(u) = t_1(u) * l_1(u).

    // We compute l_1(0) and l_1(1)
    let linear_1_evals = compute_linear_function(&w.0[..1], &[]);

    // We compute S_1(0) and S_1(inf)
    let round_poly_evals = get_evals_from_l_and_t(&linear_1_evals, &t_1_evals);
    println!("Round 1:");
    println!("S(0): {}", round_poly_evals[0]);
    println!("S(inf): {}", round_poly_evals[1]);
    println!("S(1): {}", linear_1_evals[1] * t_1_evals[1]);

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

    t_2_evals[0] += lagrange_evals_r_1[0] * accumulators_round_2[0];
    t_2_evals[0] += lagrange_evals_r_1[1] * accumulators_round_2[3];

    t_2_evals[1] += lagrange_evals_r_1[0] * accumulators_round_2[1];
    t_2_evals[1] += lagrange_evals_r_1[1] * accumulators_round_2[4];

    // We compute l_2(0) and l_12inf)
    let linear_2_evals = compute_linear_function(&w.0[..2], &[r_1]);

    // We compute S_2(u)
    let round_poly_evals = get_evals_from_l_and_t(&linear_2_evals, &t_2_evals);
    println!("Round 2:");
    println!("S(0): {}", round_poly_evals[0]);
    println!("S(inf): {}", round_poly_evals[1]);
    println!("S(1): {}", linear_2_evals[1] * t_2_evals[1]);

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
    println!("Round 3:");
    println!("S(0): {}", round_poly_evals[0]);
    println!("S(inf): {}", round_poly_evals[1]);
    println!("S(1): {}", linear_3_evals[1] * t_3_evals[1]);

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

    // Loop for the final rounds, from l_0+1 (in our case 4) to the end.
    for i in 4..num_vars + 1 {
        println!("Round {}", i);
        let mut t = Vec::new();

        // We get the number of variables of `poly` in the current round.
        let num_vars_poly_current = poly.num_variables();

        // 1. We compute t_i(u) for u in {0, 1}.
        if i <= half_l {
            // Split the eq evaluation into two parts
            let eq_r = eval_eq_in_hypercube(&w.0[half_l..].to_vec());
            let eq_l = eval_eq_in_hypercube(&w.0[i..half_l].to_vec());
            let num_vars_x_r = eq_r.len().ilog2() as usize;

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
            // Case i > l/2: Only one part of the eq evaluations remains to be processed
            let eq_l = eval_eq_in_hypercube(&w.0[i..num_vars].to_vec());
            // For t_i(0), we need p(r_[<i-1], 0, x)
            let t_0: EF = (0..(1 << (num_vars_poly_current - 1)))
                .map(|x| eq_l[x] * (poly.as_slice()[x]))
                .sum();

            // For t_i(1), we need p(r_[<i-1], 1, x)
            let t_1: EF = (0..(1 << (num_vars_poly_current - 1)))
                .map(|x| eq_l[x] * (poly.as_slice()[(1 << (num_vars_poly_current - 1)) | x]))
                .sum();
            t.push(t_0);
            t.push(t_1);
        }

        // 2. We compute S_i(u) = t_i(u) * l_i(u) for u in {0, inf}.
        // We compute l_i(0) and l_i(inf)
        let linear_evals = compute_linear_function(&w.0[..i], &challenges);

        // We compute S_i(u)
        let round_poly_evals = get_evals_from_l_and_t(&linear_evals, &t);

        // println!("S(0): {}", round_poly_evals[0]);
        // println!("S(inf): {}", round_poly_evals[1]);
        // println!("S(1): {}", linear_evals[1] * t[1]);

        // 3. Send S_2(u) to the verifier.
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
