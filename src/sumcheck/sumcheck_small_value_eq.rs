use crate::{
    fiat_shamir::prover::ProverState,
    poly::evals::EvaluationsList,
    sumcheck::small_value_utils::{
        NUM_OF_ROUNDS, RoundAccumlators, compute_p_beta, idx4, to_base_three_coeff,
    },
    whir::verifier::sumcheck,
};
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field};

use super::sumcheck_polynomial::SumcheckPolynomial;

// TODO: w could be a MultilinearPoitn?
fn precompute_e_in<F: Field>(w: &Vec<F>) -> Vec<F> {
    let half_l = w.len() / 2;
    let w_in = w[NUM_OF_ROUNDS..NUM_OF_ROUNDS + half_l].to_vec();
    eval_eq_in_hypercube(&w_in)
}

fn precompute_e_out<F: Field>(w: &Vec<F>) -> [Vec<F>; NUM_OF_ROUNDS] {
    let half_l = w.len() / 2;
    let mut res = [(); NUM_OF_ROUNDS].map(|_| Vec::new());
    for round in 0..NUM_OF_ROUNDS {
        let w_out: Vec<F> = w[round + 1..NUM_OF_ROUNDS]
            .iter()
            .chain(&w[half_l + NUM_OF_ROUNDS..])
            .cloned()
            .collect();
        res[round] = eval_eq_in_hypercube(&w_out)
    }
    res
}

// Procedure 9. Page 37.
fn compute_accumulators_eq<F: Field, EF: ExtensionField<F>>(
    poly: &EvaluationsList<F>,
    e_in: Vec<EF>,
    e_out: [Vec<EF>; NUM_OF_ROUNDS],
) -> [RoundAccumlators<EF>; NUM_OF_ROUNDS] {
    let l = poly.num_variables();
    let half_l = l / 2;

    // We initialize the accumulators for each round: A_1, A_2 and A_3.
    let mut round_1_accumulator = RoundAccumlators::<EF>::new_empty(1);
    let mut round_2_accumulator = RoundAccumlators::<EF>::new_empty(2);
    let mut round_3_accumulator = RoundAccumlators::<EF>::new_empty(3);

    let x_out_num_variables: usize;
    if l % 2 == 0 {
        x_out_num_variables = half_l - NUM_OF_ROUNDS;
    } else {
        x_out_num_variables = half_l - NUM_OF_ROUNDS + 1;
    }

    debug_assert_eq!(half_l + x_out_num_variables, l - NUM_OF_ROUNDS);

    for x_out in 0..1 << (x_out_num_variables) {
        let mut temp_accumulators: Vec<EF> = vec![EF::ZERO; 27];

        for x_in in 0..1 << half_l {
            // We collect the evaluations of p(X_0, X_1, X_2, x_in, x_out) where
            // x_in and x_out are fixed and X_0, X_1, X_2 are variables.
            let current_evals: Vec<F> = poly
                .iter()
                .skip((x_in << x_out_num_variables) | x_out) // x_in || x_out
                .step_by(1 << (l - NUM_OF_ROUNDS))
                .cloned()
                .collect();

            // We compute p(beta, x_in, x_out) for all beta in {0, 1, inf}^3
            let p_evals = compute_p_beta(current_evals);

            // TODO: change it to a map.
            for beta_index in 0..27 {
                temp_accumulators[beta_index] += e_in[x_in] * p_evals[beta_index];
            }
        }

        for beta_index in 0..27 {
            let [
                index_accumulator_1,
                index_accumulator_2,
                index_accumulator_3,
            ] = idx4(beta_index);

            let [_, beta_2, beta_3] = to_base_three_coeff(beta_index);

            if let Some(index) = index_accumulator_1 {
                // We need e_out[0][beta_2 || beta_3 || x_out] because:
                // y = beta_2 || beta_3.
                // Recall that in round 1, beta_2 and beta_3 are in {0, 1} since they represent y1 and y2 (if not, then index is None.)
                let y = beta_2 << 1 | beta_3;
                round_1_accumulator.accumulate_eval(
                    e_out[0][(y << x_out_num_variables) | x_out] * temp_accumulators[beta_index],
                    index,
                );
            }

            if let Some(index) = index_accumulator_2 {
                // We need e_out[1][beta_3 || x_out] because:
                // y = beta_3.
                // Recall beta_3 in {0, 1} since it represents y (if not, then index is None.).
                round_2_accumulator.accumulate_eval(
                    e_out[1][(beta_3 << x_out_num_variables) | x_out]
                        * temp_accumulators[beta_index],
                    index,
                );
            }

            if let Some(index) = index_accumulator_3 {
                round_3_accumulator
                    .accumulate_eval(e_out[2][x_out] * temp_accumulators[beta_index], index);
            }
        }
    }
    let result = [
        round_1_accumulator,
        round_2_accumulator,
        round_3_accumulator,
    ];
    result
}

// Implement Procedure 2 from the paper
// https://eprint.iacr.org/2025/1117.pdf Bagad, Dao, Thaler and Domb
// Compute eq(w,x) for all x∈{0,1}^l
// This is the same function as new_from_point (which is optimized for parallelization)
pub fn eval_eq_in_hypercube<F: Field>(w: &[F]) -> Vec<F> {
    let n = w.len();
    let mut evals: Vec<F> = vec![F::ONE; 1 << n];
    let mut size = 1usize;
    for i in 0..n {
        // in each iteration, we double the size
        size *= 2;
        for j in (0..size).rev().step_by(2) {
            let scalar = evals[j / 2];
            evals[j] = scalar * w[i]; // odd
            evals[j - 1] = scalar - evals[j]; // even
        }
    }
    evals
}

// Esta funcion es una copia de eval_eq() en poly/multilinear.rs
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

// Algorithm 6. Page 19.
// Compute three sumcheck rounds using the small value optimizaition and split-eq accumulators.
// It Returns the two challenges r_1 and r_2 (TODO: creo que debería devolver también los polys foldeados).
pub fn small_value_sumcheck_three_rounds_eq<Challenger, F: Field, EF: ExtensionField<F>>(
    prover_state: &mut ProverState<F, EF, Challenger>,
    poly: &EvaluationsList<F>,
    w: &Vec<EF>,
) -> ([EF; 2], SumcheckPolynomial<EF>)
where
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    let e_in = precompute_e_in(w);
    let e_out = precompute_e_out(w);

    // We compute all the accumulators A_i(v, u).
    let round_accumulators = compute_accumulators_eq(poly, e_in, e_out);

    // ------------------   Round 1   ------------------

    // 1. For u in {0, 1, inf} compute t_1(u)
    // Recall: In round 1, t_1(u) = A_1(u).

    let t_1_evals = &round_accumulators[0].accumulators;

    // 2. For u in {0, 1, inf} compute S_1(u) = t_1(u) * l_1(u).

    // We compute l_1(0) and l_1(1)
    let linear_1_evals = compute_linear_function(&w[..1], &[]);

    // We compute S_1(u)
    let round_poly_evals = [
        t_1_evals[0] * linear_1_evals[0],
        t_1_evals[1] * linear_1_evals[1],
        t_1_evals[2] * (linear_1_evals[1] - linear_1_evals[0]), // l_1(inf) = l_1(1) - l_1(0).
    ];

    // 3. Send S_1(u) to the verifier.
    // TODO: En realidad no hace falta mandar S_1(1) porque se deduce usando S_1(0).
    prover_state.add_extension_scalars(&round_poly_evals);

    // 4. Receive the challenge r_1 from the verifier.
    let r_1: EF = prover_state.sample();

    // 5. Compte R_2 = [L_0(r_1), L_1(r_1), L_inf(r_1)]
    // L_0 (x) = 1 - x
    // L_1 (x) = x
    // L_inf (x) = (x - 1)x
    let lagrange_evals_r_1 = [-r_1 + F::ONE, r_1, (r_1 - F::ONE) * r_1];

    // ------------------ Round 2 ------------------

    // 1. For u in {0, 1, inf} compute t_2(u).
    // First we take the accumulators A_2(v, u).
    // There are 9 accumulators, since v in {0, 1, inf} and u in {0, 1, inf}.
    let accumulators_round_2 = &round_accumulators[1].accumulators;

    let mut t_2_evals = [EF::ZERO; 3];

    t_2_evals[0] += lagrange_evals_r_1[0] * accumulators_round_2[0];
    t_2_evals[0] += lagrange_evals_r_1[1] * accumulators_round_2[3];

    t_2_evals[1] += lagrange_evals_r_1[0] * accumulators_round_2[1];
    t_2_evals[1] += lagrange_evals_r_1[1] * accumulators_round_2[4];

    t_2_evals[2] += lagrange_evals_r_1[0] * accumulators_round_2[2];
    t_2_evals[2] += lagrange_evals_r_1[1] * accumulators_round_2[5];

    // 2. For u in {0, 1, inf} compute S_2(u) = t_2(u) * l_2(u).

    // We compute l_2(0) and l_12inf)
    let linear_2_evals = compute_linear_function(&w[..2], &[r_1]);

    // We compute S_2(u)

    let round_poly_evals = [
        t_2_evals[0] * linear_2_evals[0],
        t_2_evals[1] * linear_2_evals[1],
        t_2_evals[2] * (linear_2_evals[1] - linear_2_evals[0]), // l_2(inf) = l_2(1) - l_2(0).
    ];

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

    let l_0 = lagrange_evals_r_1[0];
    let l_1 = lagrange_evals_r_1[1];
    let l_inf = lagrange_evals_r_1[2];

    let mul_inf = (r_2 - F::ONE) * r_2;

    // TODO: calcular `(r_2 - F::ONE) * r_2` una sola vez. Lo dejo por ahora así por claridad.
    let lagrange_evals_r_2 = [
        l_0 * (-r_2 + F::ONE),   // L_0 0
        l_0 * r_2,               // L_0 1
        l_0 * mul_inf,           // L_0 inf
        l_1 * (-r_2 + F::ONE),   // L_1 0
        l_1 * r_2,               // L_1 1
        l_1 * mul_inf,           // L_1 inf
        l_inf * (-r_2 + F::ONE), // L_inf 0
        l_inf * r_2,             // L_inf 1
        l_inf * mul_inf,         // L_inf inf
    ];

    // Round 3

    // 1. For u in {0, 1, inf} compute t_3(u).

    // First we take the accumulators A_2(v, u).
    // There are 27 accumulators, since v in {0, 1, inf}^2 and u in {0, 1, inf}.
    let accumulators_round_3 = &round_accumulators[2].accumulators;

    let mut t_3_evals = [EF::ZERO; 3];

    t_3_evals[0] += lagrange_evals_r_2[0] * accumulators_round_3[0]; // (0,0,u=0)
    t_3_evals[1] += lagrange_evals_r_2[0] * accumulators_round_3[1]; // (0,0,u=1)
    t_3_evals[2] += lagrange_evals_r_2[0] * accumulators_round_3[2]; // (0,0,u=∞)

    t_3_evals[0] += lagrange_evals_r_2[1] * accumulators_round_3[3]; // (1,0,u=0)
    t_3_evals[1] += lagrange_evals_r_2[1] * accumulators_round_3[4]; // (1,0,u=1)
    t_3_evals[2] += lagrange_evals_r_2[1] * accumulators_round_3[5]; // (1,0,u=∞)

    t_3_evals[0] += lagrange_evals_r_2[3] * accumulators_round_3[9]; // (0,1,u=0)
    t_3_evals[1] += lagrange_evals_r_2[3] * accumulators_round_3[10]; // (0,1,u=1)
    t_3_evals[2] += lagrange_evals_r_2[3] * accumulators_round_3[11]; // (0,1,u=∞)

    t_3_evals[0] += lagrange_evals_r_2[4] * accumulators_round_3[12]; // (1,1,u=0)
    t_3_evals[1] += lagrange_evals_r_2[4] * accumulators_round_3[13]; // (1,1,u=1)
    t_3_evals[2] += lagrange_evals_r_2[4] * accumulators_round_3[14]; // (1,1,u=∞)

    // 2. For u in {0, 1, inf} compute S_3(u) = t_3(u) * l_3(u).

    // We compute l_3(0) and l_3(inf)
    let linear_3_evals = compute_linear_function(&w[..3], &[r_1, r_2]);

    // We compute S_3(u)
    let round_poly_evals = [
        t_3_evals[0] * linear_3_evals[0],
        t_3_evals[1] * linear_3_evals[1],
        t_3_evals[2] * (linear_3_evals[1] - linear_3_evals[0]), // l_3(inf) = l_3(1) - l_3(0).
    ];

    // 3. Send S_3(u) to the verifier.
    // TODO: En realidad no hace falta mandar S_3(1) porque se dedecue usando S_3(0).
    prover_state.add_extension_scalars(&round_poly_evals);

    // TODO: Esto es muy ineficiente.
    let eval_in_two = round_poly_evals[2] * EF::from(F::TWO).square()
        + (round_poly_evals[1] - round_poly_evals[0] - round_poly_evals[2]) * EF::from(F::TWO)
        + round_poly_evals[0];

    let sumcheck_poly_evals = [round_poly_evals[0], round_poly_evals[1], eval_in_two];

    let sumcheck_poly = SumcheckPolynomial::new(sumcheck_poly_evals.to_vec());

    // TODO: Me parece que también va a haber que devolver poly_1 y poly_2 foldeados (con r_1 y r_2) para seguir con el sumcheck.
    ([r_1, r_2], sumcheck_poly)
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
        let p = [F::from_u64(2), F::from_u64(3), F::from_u64(5)];
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

        let out = eval_eq_in_hypercube(&[p0, p1, p2]);
        assert_eq!(out, expected);
    }

    #[test]
    fn test_precompute_e_in() {
        let w: Vec<EF> = (0..10).map(|i| EF::from(F::from_int(i))).collect();

        let e_in = precompute_e_in(&w);

        let w_in = w[NUM_OF_ROUNDS..NUM_OF_ROUNDS + 5].to_vec();

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

        // We build a random multilinear polynomial of 10 variables, using 2^10 evaluations in the hypercube {0,1}^10
        let poly = EvaluationsList::new((0..(1 << l)).map(|_| get_random_f()).collect());

        // We precompute E_in and E_out
        let e_in = precompute_e_in(&w);
        let e_out = precompute_e_out(&w);

        // We compute the accumulators.
        let accumulators = compute_accumulators_eq(&poly, e_in.clone(), e_out.clone());

        // A_3(0,0,0)
        let eq_w3_w4 = eval_eq_in_hypercube(&w[3..5]);
        let eq_w5_to_w9 = eval_eq_in_hypercube(&w[5..]);

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

        assert_eq!(expected_accumulator, accumulators[2].accumulators[0]);

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

        assert_eq!(expected_accumulator, accumulators[2].accumulators[10]);

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

        assert_eq!(expected_accumulator, accumulators[2].accumulators[15]);

        // We compute now A_2(0, 0):
        let eq_w2_w3_w4 = eval_eq_in_hypercube(&w[2..5]);

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

        assert_eq!(expected_accumulator, accumulators[1].accumulators[0]);

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

        assert_eq!(expected_accumulator, accumulators[1].accumulators[6]);

        // A_1(u)
        let eq_w1_w2_w3_w4 = eval_eq_in_hypercube(&w[1..5]);

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

        assert_eq!(expected_accumulator, accumulators[0].accumulators[0]);

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

        assert_eq!(expected_accumulator, accumulators[0].accumulators[1]);

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

        assert_eq!(expected_accumulator, accumulators[0].accumulators[2]);

        // assert_eq!(expected_accumulator, accumulators[0].accumulators[0]);
    }

    #[test]
    fn test_compute_accumulators_eq_l_11() {
        // We'll use polynomials of 10 variables.
        let l = 11;

        // w = [w0, w2, ..., w9]
        // Each w_i is a random extension field element built from 4 random field elements.
        let w: Vec<EF> = (0..l).map(|_| get_random_ef()).collect();

        // We build a random multilinear polynomial of 10 variables, using 2^10 evaluations in the hypercube {0,1}^10
        let poly = EvaluationsList::new((0..(1 << l)).map(|_| get_random_f()).collect());

        // We precompute E_in and E_out
        let e_in = precompute_e_in(&w);
        let e_out = precompute_e_out(&w);

        // We compute the accumulators.
        let accumulators = compute_accumulators_eq(&poly, e_in.clone(), e_out.clone());

        // A_3(0,0,0)
        let eq_w3_w4 = eval_eq_in_hypercube(&w[3..5]);
        let eq_w5_to_w10 = eval_eq_in_hypercube(&w[5..]);

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

        assert_eq!(expected_accumulator, accumulators[2].accumulators[0]);

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

        assert_eq!(expected_accumulator, accumulators[2].accumulators[10]);

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

        assert_eq!(expected_accumulator, accumulators[2].accumulators[15]);

        // Round 2:
        let eq_w2_w3_w4 = eval_eq_in_hypercube(&w[2..5]);

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

        assert_eq!(expected_accumulator, accumulators[1].accumulators[0]);

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

        assert_eq!(expected_accumulator, accumulators[1].accumulators[6]);

        // Round 3:
        let eq_w1_w2_w3_w4 = eval_eq_in_hypercube(&w[1..5]);

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

        assert_eq!(expected_accumulator, accumulators[0].accumulators[0]);

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

        assert_eq!(expected_accumulator, accumulators[0].accumulators[1]);

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

        assert_eq!(expected_accumulator, accumulators[0].accumulators[2]);
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

    fn get_evals_from_l_and_t<F: Field>(l: &[F; 2], t: &[F]) -> [F; 3] {
        [
            t[0] * l[0],          // s(0)
            t[1] * l[1],          // s(1)
            t[2] * (l[1] - l[0]), //s(inf) -> l(inf) = l(1) - l(0)
        ]
    }

    #[test]
    fn test_svo_sumcheck_rounds_eq_simulation() {
        let poly = EvaluationsList::new((0..512).map(|_| get_random_f()).collect());
        let w: Vec<EF> = (0..9).map(|_| get_random_ef()).collect();

        let expected_sum = naive_sumcheck_verification(w.clone(), poly.clone());

        let e_in = precompute_e_in(&w);
        let e_out = precompute_e_out(&w);

        // We compute all the accumulators A_i(v, u).
        let round_accumulators = compute_accumulators_eq(&poly, e_in, e_out);

        // ------------------   Round 1   ------------------

        // 1. For u in {0, 1, inf} compute t_1(u)
        // Recall: In round 1, t_1(u) = A_1(u).

        let t_1_evals = &round_accumulators[0].accumulators;

        // 2. For u in {0, 1, inf} compute S_1(u) = t_1(u) * l_1(u).

        // We compute l_1(0) and l_1(1)
        let linear_1_evals = compute_linear_function(&w[..1], &[]);

        // We compute S_1(u)
        let round_poly_evals_1 = get_evals_from_l_and_t(&linear_1_evals, &t_1_evals[..]);

        assert_eq!(round_poly_evals_1[0] + round_poly_evals_1[1], expected_sum);

        // 4. Receive the challenge r_1 from the verifier.
        let r_1 = get_random_ef();

        // 5. Compte R_2 = [L_0(r_1), L_1(r_1), L_inf(r_1)]
        // L_0 (x) = 1 - x
        // L_1 (x) = x
        // L_inf (x) = (x - 1)x
        let lagrange_evals_r_1 = [-r_1 + F::ONE, r_1, (r_1 - F::ONE) * r_1];

        // ------------------ Round 2 ------------------

        // 1. For u in {0, 1, inf} compute t_2(u).
        // First we take the accumulators A_2(v, u).
        // There are 9 accumulators, since v in {0, 1, inf} and u in {0, 1, inf}.
        let accumulators_round_2 = &round_accumulators[1].accumulators;

        let mut t_2_evals = [EF::ZERO; 3];

        t_2_evals[0] += lagrange_evals_r_1[0] * accumulators_round_2[0];
        t_2_evals[0] += lagrange_evals_r_1[1] * accumulators_round_2[3];

        t_2_evals[1] += lagrange_evals_r_1[0] * accumulators_round_2[1];
        t_2_evals[1] += lagrange_evals_r_1[1] * accumulators_round_2[4];

        t_2_evals[2] += lagrange_evals_r_1[0] * accumulators_round_2[2];
        t_2_evals[2] += lagrange_evals_r_1[1] * accumulators_round_2[5];

        assert_eq!(t_2_evals[2], t_2_evals[1] - t_2_evals[0]);

        // 2. For u in {0, 1, inf} compute S_2(u) = t_2(u) * l_2(u).

        // We compute l_2(0) and l_2(1)
        let linear_2_evals = compute_linear_function(&w[..2], &[r_1]);

        // We compute S_2(u)
        let round_poly_evals_2 = get_evals_from_l_and_t(&linear_2_evals, &t_2_evals);

        // S_1(r_1) = a * (r_1)^2 + b * r_1 + c where
        // a = S_1(inf),
        // b = S_1(1) - S_1(0) - S_1(inf),
        // c = S_1(0)
        let expected_eval = round_poly_evals_1[2] * r_1.square()
            + (round_poly_evals_1[1] - round_poly_evals_1[0] - round_poly_evals_1[2]) * r_1
            + round_poly_evals_1[0];

        assert_eq!(round_poly_evals_2[0] + round_poly_evals_2[1], expected_eval);

        // 4. Receive the challenge r_2 from the verifier.
        let r_2 = get_random_ef();

        // 5. Compute R_3 = [L_00(r_1, r_2), L_01(r_1, r_2), ..., L_{inf inf}(r_1, r_2)]
        // L_00 (x1, x2) = (1 - x1) * (1 - x2)
        // L_01 (x1, x2) = (1 - x1) * x2
        // ...
        // L_{inf inf} (x1, x2) = (x1 - 1) * x1 * (x2 - 1) * x2

        let l_0 = lagrange_evals_r_1[0];
        let l_1 = lagrange_evals_r_1[1];
        let l_inf = lagrange_evals_r_1[2];

        let mul_inf = (r_2 - F::ONE) * r_2;

        // TODO: calcular `(r_2 - F::ONE) * r_2` una sola vez. Lo dejo por ahora así por claridad.
        let lagrange_evals_r_2 = [
            l_0 * (-r_2 + F::ONE),   // L_0 0
            l_0 * r_2,               // L_0 1
            l_0 * mul_inf,           // L_0 inf
            l_1 * (-r_2 + F::ONE),   // L_1 0
            l_1 * r_2,               // L_1 1
            l_1 * mul_inf,           // L_1 inf
            l_inf * (-r_2 + F::ONE), // L_inf 0
            l_inf * r_2,             // L_inf 1
            l_inf * mul_inf,         // L_inf inf
        ];

        // Round 3

        // 1. For u in {0, 1, inf} compute t_3(u).

        // First we take the accumulators A_2(v, u).
        // There are 27 accumulators, since v in {0, 1, inf}^2 and u in {0, 1, inf}.
        let accumulators_round_3 = &round_accumulators[2].accumulators;

        let mut t_3_evals = [EF::ZERO; 3];

        t_3_evals[0] += lagrange_evals_r_2[0] * accumulators_round_3[0]; // (0,0,u=0)
        t_3_evals[1] += lagrange_evals_r_2[0] * accumulators_round_3[1]; // (0,0,u=1)
        t_3_evals[2] += lagrange_evals_r_2[0] * accumulators_round_3[2]; // (0,0,u=∞)

        t_3_evals[0] += lagrange_evals_r_2[1] * accumulators_round_3[3]; // (1,0,u=0)
        t_3_evals[1] += lagrange_evals_r_2[1] * accumulators_round_3[4]; // (1,0,u=1)
        t_3_evals[2] += lagrange_evals_r_2[1] * accumulators_round_3[5]; // (1,0,u=∞)

        t_3_evals[0] += lagrange_evals_r_2[3] * accumulators_round_3[9]; // (0,1,u=0)
        t_3_evals[1] += lagrange_evals_r_2[3] * accumulators_round_3[10]; // (0,1,u=1)
        t_3_evals[2] += lagrange_evals_r_2[3] * accumulators_round_3[11]; // (0,1,u=∞)

        t_3_evals[0] += lagrange_evals_r_2[4] * accumulators_round_3[12]; // (1,1,u=0)
        t_3_evals[1] += lagrange_evals_r_2[4] * accumulators_round_3[13]; // (1,1,u=1)
        t_3_evals[2] += lagrange_evals_r_2[4] * accumulators_round_3[14]; // (1,1,u=∞)

        // 2. For u in {0, 1, inf} compute S_3(u) = t_3(u) * l_3(u).

        // We compute l_3(0) and l_3(inf)
        let linear_3_evals = compute_linear_function(&w[..3], &[r_1, r_2]);

        // We compute S_3(u)
        let round_poly_evals_3 = get_evals_from_l_and_t(&linear_3_evals, &t_3_evals);

        // S_2(r_2) = a * (r_2)^2 + b * r_2 + c where
        // a = S_2(inf),
        // b = S_2(1) - S_2(0) - S_2(inf),
        // c = S_2(0)
        let expected_eval = round_poly_evals_2[2] * r_2.square()
            + (round_poly_evals_2[1] - round_poly_evals_2[0] - round_poly_evals_2[2]) * r_2
            + round_poly_evals_2[0];

        assert_eq!(round_poly_evals_3[0] + round_poly_evals_3[1], expected_eval);
    }
    #[test]
    fn compare_sv_vs_eq() {
        let poly = EvaluationsList::new((0..512).map(|_| get_random_f()).collect());
        let w: Vec<F> = (0..9).map(|_| get_random_f()).collect();

        let r_1 = get_random_f();
        let r_2 = get_random_f();

        // -------------  EQ  -------------
        let e_in = precompute_e_in(&w);
        let e_out = precompute_e_out(&w);

        let round_accumulators = compute_accumulators_eq(&poly, e_in, e_out);
        let t_1_evals = &round_accumulators[0].accumulators;
        let linear_1_evals = compute_linear_function(&w[..1], &[]);
        let round_poly_evals_1 = get_evals_from_l_and_t(&linear_1_evals, &t_1_evals[..]);

        println!("ROUND 1 EQ: {:?}", round_poly_evals_1);

        let lagrange_evals_r_1 = [-r_1 + F::ONE, r_1, (r_1 - F::ONE) * r_1];

        let accumulators_round_2 = &round_accumulators[1].accumulators;

        let mut t_2_evals = [F::ZERO; 3];

        t_2_evals[0] += lagrange_evals_r_1[0] * accumulators_round_2[0];
        t_2_evals[0] += lagrange_evals_r_1[1] * accumulators_round_2[3];

        t_2_evals[1] += lagrange_evals_r_1[0] * accumulators_round_2[1];
        t_2_evals[1] += lagrange_evals_r_1[1] * accumulators_round_2[4];

        t_2_evals[2] += lagrange_evals_r_1[0] * accumulators_round_2[2];
        t_2_evals[2] += lagrange_evals_r_1[1] * accumulators_round_2[5];

        let linear_2_evals = compute_linear_function(&w[..2], &[r_1]);
        let round_poly_evals_2 = get_evals_from_l_and_t(&linear_2_evals, &t_2_evals);

        println!("ROUND 2 EQ: {:?}", round_poly_evals_2);

        let l_0 = lagrange_evals_r_1[0];
        let l_1 = lagrange_evals_r_1[1];
        let l_inf = lagrange_evals_r_1[2];

        let mul_inf = (r_2 - F::ONE) * r_2;

        let lagrange_evals_r_2 = [
            l_0 * (-r_2 + F::ONE),   // L_0 0
            l_0 * r_2,               // L_0 1
            l_0 * mul_inf,           // L_0 inf
            l_1 * (-r_2 + F::ONE),   // L_1 0
            l_1 * r_2,               // L_1 1
            l_1 * mul_inf,           // L_1 inf
            l_inf * (-r_2 + F::ONE), // L_inf 0
            l_inf * r_2,             // L_inf 1
            l_inf * mul_inf,         // L_inf inf
        ];

        let accumulators_round_3 = &round_accumulators[2].accumulators;

        let mut t_3_evals = [F::ZERO; 3];

        t_3_evals[0] += lagrange_evals_r_2[0] * accumulators_round_3[0]; // (0,0,u=0)
        t_3_evals[1] += lagrange_evals_r_2[0] * accumulators_round_3[1]; // (0,0,u=1)
        t_3_evals[2] += lagrange_evals_r_2[0] * accumulators_round_3[2]; // (0,0,u=∞)

        t_3_evals[0] += lagrange_evals_r_2[1] * accumulators_round_3[3]; // (1,0,u=0)
        t_3_evals[1] += lagrange_evals_r_2[1] * accumulators_round_3[4]; // (1,0,u=1)
        t_3_evals[2] += lagrange_evals_r_2[1] * accumulators_round_3[5]; // (1,0,u=∞)

        t_3_evals[0] += lagrange_evals_r_2[3] * accumulators_round_3[9]; // (0,1,u=0)
        t_3_evals[1] += lagrange_evals_r_2[3] * accumulators_round_3[10]; // (0,1,u=1)
        t_3_evals[2] += lagrange_evals_r_2[3] * accumulators_round_3[11]; // (0,1,u=∞)

        t_3_evals[0] += lagrange_evals_r_2[4] * accumulators_round_3[12]; // (1,1,u=0)
        t_3_evals[1] += lagrange_evals_r_2[4] * accumulators_round_3[13]; // (1,1,u=1)
        t_3_evals[2] += lagrange_evals_r_2[4] * accumulators_round_3[14]; // (1,1,u=∞)

        let linear_3_evals = compute_linear_function(&w[..3], &[r_1, r_2]);

        let round_poly_evals_3 = get_evals_from_l_and_t(&linear_3_evals, &t_3_evals);

        println!("ROUND 3 EQ: {:?}", round_poly_evals_3);

        // -------------  P * Q  -------------
        let poly_2 = EvaluationsList::new(eval_eq_in_hypercube(&w));

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
