use crate::poly::multilinear::MultilinearPoint;
use crate::{
    dft::DitEvalsButterfly, fiat_shamir::prover::ProverState, poly::evals::EvaluationsList,
};
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field, BasedVectorSpace};

const NUM_OF_ROUNDS: usize = 3;

#[derive(Debug, Clone, Eq, PartialEq, Copy)]
enum EvaluationPoint {
    Infinity,
    Zero,
    One,
}
impl EvaluationPoint {
    fn to_usize_representation(&self) -> usize {
        match self {
            EvaluationPoint::Infinity => 2,
            EvaluationPoint::Zero => 0,
            EvaluationPoint::One => 1,
        }
    }

    fn from_usize_representation(n: usize) -> Self {
        match n {
            0 => EvaluationPoint::Zero,
            1 => EvaluationPoint::One,
            2 => EvaluationPoint::Infinity,
            _ => unreachable!(),
        }
    }
}

// For round i, RoundAccumulators has all the accumulators of the form A_i(u, v).
#[derive(Debug, Clone, Eq, PartialEq)]
struct RoundAccumlators<F: Field> {
    round: usize,
    accumulators: Vec<F>,
}

impl<F> RoundAccumlators<F>
where
    F: Field,
{
    fn new(round: usize, accumulators: Vec<F>) -> Self {
        RoundAccumlators {
            round,
            accumulators,
        }
    }

    // Given the round i, it returns the accumulators A_i(u, v) all initialized in zero.
    fn new_empty(round: usize) -> Self {
        match round {
            // In the round 1, there are 3 accumulators: A_1(0), A_1(1) and A_1(inf).
            1 => RoundAccumlators {
                round,
                accumulators: vec![F::ZERO; 3],
            },
            // In round 2, there are 3 * 3 = 9 accumulators,
            // since v in {0, 1, inf} and u in {0, 1, inf}.
            2 => RoundAccumlators {
                round,
                accumulators: vec![F::ZERO; 9],
            },
            // In round 3, there are 3^2 * 3 = 27 accumulators,
            // since v in {0, 1, inf}^2 and u in {0, 1, inf}.
            3 => RoundAccumlators {
                round,
                accumulators: vec![F::ZERO; 27],
            },
            _ => unreachable!(),
        }
    }

    fn accumulate_eval(&mut self, eval: F, index: usize) {
        self.accumulators[index] += eval;
    }
}
// ESTA FUNCIÓN YA NO SE USA
// la llamemos ahi una sola vez. El input debería ser solo beta (sin la round) y tiene que determinar para cada round qué devuelve.
// CAMBIAR COMENTARIO.
// Esta función mapea una tupla del tipo (i,v,u,y) que resulta de aplicar indx4(\beta) a un indice que correpende a un acumulador especifico.
// Recall: v in {0, 1, inf}^{i-1}, u in {0, 1, inf}, y in {0, 1}^{3-i}, where i is the number of round.
// Vamos a tomar u y v como coeficientes en base 3 (donde inf lo tomamos como 2), los representamos en binario y los concatenamos con y en su forma binaria.
// Observar que en cada ronda cambian los tamaños de v, u and y.
fn idx4(
    round: u8,
    beta_1: &EvaluationPoint,
    beta_2: &EvaluationPoint,
    beta_3: &EvaluationPoint,
) -> Option<usize> {
    let beta_1 = beta_1.to_usize_representation();
    let beta_2 = beta_2.to_usize_representation();
    let beta_3 = beta_3.to_usize_representation();

    match round {
        1 => {
            // (beta_1, beta_2, beta_3) = (u, y1_ y2) where:
            // u in {0, 1, inf}, y1 in {0, 1}, y2 in {0, 1}.
            if beta_2 == 2 || beta_3 == 2 {
                return None; // y1 and y2 must be in {0, 1}
            }
            // We have only three accumulators in round 1: A_1(0), A_1(1) and A_1(inf).
            // Index 0 -> A_1(0)
            // Index 1 -> A_1(1)
            // Index 2 -> A_1(inf)
            return Some(beta_1);
        }
        2 => {
            // (beta_1, beta_2, beta_3) = (v, u, y) where:
            // v in {0, 1, inf}, u in {0, 1, inf}, y in {0, 1}.
            if beta_3 == 2 {
                return None; // y must be in {0, 1}
            }
            // We think (v, u) as the coefficients of a base 3 number.
            // So, the index is v * 3 + u
            let index = beta_1 * 3 + beta_2;
            return Some(index);
        }
        3 => {
            // (beta_1, beta_2, beta_3) = (v1, v2, u) where:
            // v1 in {0, 1, inf}, v2 in {0, 1, inf}, u in {0, 1, inf}.
            // There are 3 * 3 * 3 = 27 tuples.
            // We order the tuples according to the number we get when using (v1, v2, u) as the coefficients in base 3.
            // So, the index is v1 * 3^2 + v2 * 3 + u.
            let index = beta_1 * 9 + beta_2 * 3 + beta_3;
            return Some(index);
        }
        _ => unreachable!(),
    }
}

// ESTA FUNCIÓN YA NO SE USA
fn idx4_v2(beta: [EvaluationPoint; 3]) -> [Option<usize>; 3] {
    let beta_1 = beta[0].to_usize_representation();
    let beta_2 = beta[1].to_usize_representation();
    let beta_3 = beta[2].to_usize_representation();

    match beta {
        [_, _, EvaluationPoint::Infinity] => {
            let index_3 = beta_1 * 9 + beta_2 * 3 + beta_3;
            return [None, None, Some(index_3)];
        }
        [_, EvaluationPoint::Infinity, _] => {
            let index_2 = beta_1 * 3 + beta_2;
            let index_3 = beta_1 * 9 + beta_2 * 3 + beta_3;
            return [None, Some(index_2), Some(index_3)];
        }
        _ => {
            let index_1 = beta_1;
            let index_2 = beta_1 * 3 + beta_2;
            let index_3 = beta_1 * 9 + beta_2 * 3 + beta_3;
            return [Some(index_1), Some(index_2), Some(index_3)];
        }
    }
}

// We asseume n in {0, ..., 26}
fn to_base_three_coeff(n: usize) -> [usize; 3] {
    let mut n = n;
    let mut coeffs = [0; 3];
    for i in (0..NUM_OF_ROUNDS).rev() {
        coeffs[i] = n % 3;
        n /= 3;
    }
    coeffs
}

fn idx4_v3(index_beta: usize) -> [Option<usize>; 3] {
    let [b1, b2, b3] = to_base_three_coeff(index_beta);

    match (b1, b2, b3) {
        (_, _, 2) => [None, None, Some(b1 * 9 + b2 * 3 + b3)],
        (_, 2, _) => [None, Some(b1 * 3 + b2), Some(b1 * 9 + b2 * 3 + b3)],
        _ => [Some(b1), Some(b1 * 3 + b2), Some(b1 * 9 + b2 * 3 + b3)],
    }
}

// ESTA FUNCIÓN NO SE USA.
// TODO: En vez de pasar de indice a beta y despues de beta a indice, hay que buscar la manera de hacerlo sin tener que pasar al beta.
// Ojo: no es la inversa de la función anterior. (No incluye el round)
// `index` in {0, ..., 27}
fn from_index_to_beta(index: usize) -> [EvaluationPoint; 3] {
    let mut index = index;
    let mut res = [EvaluationPoint::Zero; NUM_OF_ROUNDS];
    for i in (0..NUM_OF_ROUNDS).rev() {
        res[i] = EvaluationPoint::from_usize_representation(index % 3);
        index /= 3;
    }
    res
}

//     Ronda 1:        Ronda 2:        Ronda 3:

// 000-00         1               1               1
// 001-01         2               2               1
// 002-02                                        (1)
// 010-03         3               1               2
// 011-04         4               2               2
// 012-05                                        (2)
// 020-06                        (1)              3
// 021-07                        (2)              3
// 022-08                                        (3)
// 100-09         1               3               4
// 101-10         2               4               4
// 102-11                                        (4)
// 110-12         3               3               5
// 111-13         4               4               5
// 112-14                                        (5)
// 120-15                        (3)              6
// 121-16                        (4)              6
// 122-17                                        (6)
// 200-18        (1)              5               7
// 201-19        (2)              6               7
// 202-20                                        (7)
// 210-21        (3)              5               8
// 211-22        (4)              6               8
// 212-23                                        (8)
// 220-24                        (5)              9
// 221-25                        (6)              9
// 222-26                                        (9)
//
// Implement Procedure 6 (Page 34).
// Fijado x'' en {0, 1}^{l-3}, dadas las evaluaciones del multilineal q(x1, x2, x3) = p(x1, x2, x3, x'') en el booleano devuelve las
// evaluaciones de q en beta para todo beta in {0, 1, inf}^3.
fn compute_p_beta<F: Field>(current_evals: Vec<F>) -> Vec<F> {
    let mut next_evals = vec![F::ZERO; 27];

    next_evals[0] = current_evals[0]; // 000
    next_evals[1] = current_evals[1]; // 001
    next_evals[3] = current_evals[2]; // 010
    next_evals[4] = current_evals[3]; // 011
    next_evals[9] = current_evals[4]; // 100
    next_evals[10] = current_evals[5]; // 101
    next_evals[12] = current_evals[6]; // 110
    next_evals[13] = current_evals[7]; // 111

    // j = 1
    next_evals[18] = next_evals[9] - next_evals[0]; // 200
    next_evals[19] = next_evals[10] - next_evals[1]; // 201
    next_evals[21] = next_evals[12] - next_evals[3]; // 210
    next_evals[22] = next_evals[13] - next_evals[4]; // 211

    // j = 2
    next_evals[6] = next_evals[3] - next_evals[0]; // 020
    next_evals[7] = next_evals[4] - next_evals[1]; // 021
    next_evals[15] = next_evals[12] - next_evals[9]; // 120
    next_evals[16] = next_evals[13] - next_evals[10]; // 121
    next_evals[24] = next_evals[21] - next_evals[18]; // 220
    next_evals[25] = next_evals[22] - next_evals[19]; // 221

    // j = 3
    next_evals[2] = next_evals[1] - next_evals[0]; // 002
    next_evals[5] = next_evals[4] - next_evals[3]; // 012
    next_evals[8] = next_evals[7] - next_evals[6]; // 022
    next_evals[11] = next_evals[10] - next_evals[9]; // 102
    next_evals[14] = next_evals[13] - next_evals[12]; // 112
    next_evals[17] = next_evals[16] - next_evals[15]; // 122
    next_evals[20] = next_evals[19] - next_evals[18]; // 202
    next_evals[23] = next_evals[22] - next_evals[21]; // 212
    next_evals[26] = next_evals[25] - next_evals[24]; // 222

    next_evals
}

// Implements the Procedure 7 in https://eprint.iacr.org/2025/1117.pdf (page 34).
fn compute_accumulators<F: Field>(
    poly_1: &EvaluationsList<F>,
    poly_2: &EvaluationsList<F>,
) -> [RoundAccumlators<F>; NUM_OF_ROUNDS] {
    assert_eq!(poly_1.num_variables(), poly_2.num_variables());
    let l = poly_1.num_variables();

    // We initialize the accumulators for each round: A_1, A_2 and A_3.
    let mut round_1_accumulator = RoundAccumlators::<F>::new_empty(1);
    let mut round_2_accumulator = RoundAccumlators::<F>::new_empty(2);
    let mut round_3_accumulator = RoundAccumlators::<F>::new_empty(3);

    // For x'' in {0 .. 2^{l - 3}}:
    for x in 0..1 << (l - NUM_OF_ROUNDS) {
        // We compute p_1(beta, x'') for all beta in {0, 1, inf}^3
        let current_evals_1: Vec<F> = poly_1
            .iter()
            .skip(x)
            .step_by(1 << (l - NUM_OF_ROUNDS))
            .cloned()
            .collect();
        let evals_1 = compute_p_beta(current_evals_1);

        // We compute p_2(beta, x'') for all beta in {0, 1, inf}^3
        let current_evals_2: Vec<F> = poly_2
            .iter()
            .skip(x)
            .step_by(1 << (l - NUM_OF_ROUNDS))
            .cloned()
            .collect();
        let evals_2 = compute_p_beta(current_evals_2);

        // For each beta in {0, 1, inf}^3:
        // (We have 27 = 3 ^ NUM_OF_ROUNDS number of betas)
        for beta_index in 0..27 {
            let [
                index_accumulator_1,
                index_accumulator_2,
                index_accumulator_3,
            ] = idx4_v3(beta_index);

            for (index_opt, acc) in [
                (index_accumulator_1, &mut round_1_accumulator),
                (index_accumulator_2, &mut round_2_accumulator),
                (index_accumulator_3, &mut round_3_accumulator),
            ] {
                if let Some(index) = index_opt {
                    acc.accumulate_eval(evals_1[beta_index] * evals_2[beta_index], index);
                }
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

// Algorithm 4. Page 15.
// Compute three sumcheck rounds using the small value optimizaition.
// It Returns the two challenges r_1 and r_2 (TODO: creo que debería devolver también los polys foldeados).
fn small_value_sumcheck_three_rounds<Challenger, F: Field, EF: ExtensionField<F>>(
    prover_state: &mut ProverState<F, EF, Challenger>,
    poly_1: &EvaluationsList<F>,
    poly_2: &EvaluationsList<F>,
) -> [EF; 2]
where
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    // We compute all the accumulators A_i(v, u).
    let round_accumulators = compute_accumulators(&poly_1, &poly_2);

    // Round 1

    // 1. For u in {0, 1, inf} compute S_1(u).
    // Recall: S_1(u) = A_1(u).
    let round_poly_evals = &round_accumulators[0].accumulators;

    // 2. Send S_1(u) to the verifier.
    // TODO: En realidad no hace falta mandar S_1(1) porque se deduce usando S_1(0).
    prover_state.add_base_scalars(&round_poly_evals);

    // 3. Receive the challenge r_1 from the verifier.
    let r_1: EF = prover_state.sample();

    // 4. Compte R_2 = [L_0(r_1), L_1(r_1), L_inf(r_1)]
    // L_0 (x) = 1 - x
    // L_1 (x) = x
    // L_inf (x) = (x - 1)x
    let lagrange_evals_r_1 = [-r_1 + F::ONE, r_1, (r_1 - F::ONE) * r_1];

    // Round 2

    // 1. For u in {0, 1, inf} compute S_2(u).
    // First we take the accumulators A_2(v, u).
    // There are 9 accumulators, since v in {0, 1, inf} and u in {0, 1, inf}.
    let accumulators_round_2 = &round_accumulators[1].accumulators;

    let mut round_poly_evals = [EF::ZERO; 3];

    // We split accumulators_2 in three chunks of three elements each, where each chunk corresponds to
    // a fixed v and the three elements in the chunk correspond to the three possible values of u.
    for (lagrange_index, accumulators_chunk) in accumulators_round_2.chunks_exact(3).enumerate() {
        // S_2(0) = A_2(0, 0) * L_0(r_1) + A_2(1, 0) * L_1(r_1) + A_2(inf, 0) * L_inf(r_1)
        round_poly_evals[0] += lagrange_evals_r_1[lagrange_index] * accumulators_chunk[0];

        // S_2(1) = A_2(0, 1) * L_0(r_1) + A_2(1, 1) * L_1(r_1) + A_2(inf, 1) * L_inf(r_1)
        round_poly_evals[1] += lagrange_evals_r_1[lagrange_index] * accumulators_chunk[1];

        // S_2(inf) = A_2(0, inf) * L_0(r_1) + A_2(1, inf) * L_1(r_1) + A_2(inf, inf) * L_inf(r_1)
        round_poly_evals[2] += lagrange_evals_r_1[lagrange_index] * accumulators_chunk[2];
    }

    // 2. Send S_2(u) to the verifier.
    // TODO: En realidad no hace falta mandar S_2(1) porque se deduce usando S_2(0).
    prover_state.add_extension_scalars(&round_poly_evals);

    // 3. Receive the challenge r_2 from the verifier.
    let r_2: EF = prover_state.sample();

    // 4. Compute R_3 = [L_00(r_1, r_2), L_01(r_1, r_2), ..., L_{inf inf}(r_1, r_2)]
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

    // 1. For u in {0, 1, inf} compute S_3(u).

    // First we take the accumulators A_2(v, u).
    // There are 27 accumulators, since v in {0, 1, inf}^2 and u in {0, 1, inf}.
    let accumulators_round_3 = &round_accumulators[2].accumulators;

    round_poly_evals = [EF::ZERO; 3];

    // We split the accumulators in three chunks of 9 elements each, where each chunk corresponds
    // to fixed v1 and v2,
    for (lagrange_index, accumulators_chunk) in accumulators_round_3.chunks_exact(9).enumerate() {
        // Within each chunk of 9, we have 3 groups of 3 elements for each u2 value
        for (u_index, u_chunk) in accumulators_chunk.chunks_exact(3).enumerate() {
            // S_3(0) += L_{v1,v2}(r_1, r_2) * A_3(v1, v2, 0)
            round_poly_evals[0] += lagrange_evals_r_2[lagrange_index * 3 + u_index] * u_chunk[0];

            // S_3(1) += L_{v1,v2}(r_1, r_2) * A_3(v1, v2, 1)
            round_poly_evals[1] += lagrange_evals_r_2[lagrange_index * 3 + u_index] * u_chunk[1];

            // S_3(inf) += L_{v1,v2}(r_1, r_2) * A_3(v1, v2, inf)
            round_poly_evals[2] += lagrange_evals_r_2[lagrange_index * 3 + u_index] * u_chunk[2];
        }
    }

    // 2. Send S_3(u) to the verifier.
    // TODO: En realidad no hace falta mandar S_3(1) porque se dedecue usando S_3(0).
    prover_state.add_extension_scalars(&round_poly_evals);

    // TODO: Me parece que también va a haber que devolver poly_1 y poly_2 foldeados (con r_1 y r_2) para seguir con el sumcheck.
    [r_1, r_2]
}
// TODO: w could be a MultilinearPoitn?
fn precompute_e_in<F: Field>(w: &Vec<F>) -> Vec<F> {
    let half_l = w.len() / 2;
    let w_in = w[NUM_OF_ROUNDS..NUM_OF_ROUNDS + half_l].to_vec();
    eval_eq(&w_in)
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
        res[round] = eval_eq(&w_out)
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

    let x_out_num_variables = half_l - NUM_OF_ROUNDS;

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
            ] = idx4_v3(beta_index);

            let [_, beta_2, beta_3] = to_base_three_coeff(beta_index);

            if let Some(index) = index_accumulator_1 {
                // We need e_out[0][beta_2 || beta_3 || x_out] because:
                // y = beta_2 || beta_3.
                // Recall that in round 1, beta_2 and beta_3 are in {0, 1} since they represent y1 and y2 (if not, then index is None.)
                let y = beta_2 << 1 | beta_3;
                round_1_accumulator
                    .accumulate_eval(e_out[0][(y << x_out_num_variables) | x_out] * temp_accumulators[beta_index], index);
            }

            if let Some(index) = index_accumulator_2 {
                // We need e_out[1][beta_3 || x_out] because:
                // y = beta_3.
                // Recall beta_3 in {0, 1} since it represents y (if not, then index is None.).
                round_2_accumulator
                    .accumulate_eval(e_out[1][(beta_3 << x_out_num_variables) | x_out] * temp_accumulators[beta_index], index);
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
pub fn eval_eq<F: Field>(w: &[F]) -> Vec<F> {
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

fn eq_expected<F: Field>(w: &[F], idx: usize) -> F {
    let mut acc = F::ONE;
    for (j, wj) in w.iter().enumerate() {
        if (idx >> j) & 1 == 0 {
            acc *= F::ONE - *wj;
        } else {
            acc *= *wj;
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::{evals::EvaluationsList, multilinear::MultilinearPoint};
    use p3_baby_bear::BabyBear;
    use p3_field::{
        PrimeCharacteristicRing, extension::BinomialExtensionField, integers::QuotientMap, BasedVectorSpace
    };
    use rand::{Rng, RngCore};

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    // BORRAR
    // TEST NOT WORKING. We changed the function idx4.
    #[test]
    fn test_idx4() {
        let inf = EvaluationPoint::Infinity;
        let zero = EvaluationPoint::Zero;
        let one = EvaluationPoint::One;

        // Round 1
        // (beta_1, beta_2, beta_3) = (u, y1, y2) where
        // u in {0, 1, inf}, y1 in {0, 1}, y2 in {0, 1}.
        // We order the tuples by representing u in binary and concatenating it with y1 and y2.
        // There are 3 * 2^2 = 12 tuples.
        assert_eq!(idx4(1, &zero, &zero, &zero).unwrap(), 0);
        assert_eq!(idx4(1, &zero, &zero, &one).unwrap(), 1);
        assert_eq!(idx4(1, &zero, &one, &zero).unwrap(), 2);
        assert_eq!(idx4(1, &zero, &one, &one).unwrap(), 3);
        assert_eq!(idx4(1, &one, &zero, &zero).unwrap(), 4);
        assert_eq!(idx4(1, &one, &zero, &one).unwrap(), 5);
        assert_eq!(idx4(1, &one, &one, &zero).unwrap(), 6);
        assert_eq!(idx4(1, &one, &one, &one).unwrap(), 7);
        assert_eq!(idx4(1, &inf, &zero, &zero).unwrap(), 8);
        assert_eq!(idx4(1, &inf, &zero, &one).unwrap(), 9);
        assert_eq!(idx4(1, &inf, &one, &zero).unwrap(), 10);
        assert_eq!(idx4(1, &inf, &one, &one).unwrap(), 11);

        assert_eq!(idx4(1, &zero, &inf, &zero), None);
        assert_eq!(idx4(1, &zero, &zero, &inf), None);
        assert_eq!(idx4(1, &zero, &inf, &inf), None);

        // Round 2
        // (beta_1, beta_2, beta_3) = (v, u, y) where
        // v in {0, 1, inf}, u in {0, 1, inf}, y in {0, 1}.
        // We order the tuples in the following way:
        // We think (v, u) as the coefficients of a base 3 number, and then we concatenate its binary representation with y.
        // There are 3 * 3 * 2 = 18 tuples.
        assert_eq!(idx4(2, &zero, &zero, &zero).unwrap(), 0);
        assert_eq!(idx4(2, &zero, &zero, &one).unwrap(), 1);
        assert_eq!(idx4(2, &zero, &one, &zero).unwrap(), 2);
        assert_eq!(idx4(2, &zero, &one, &one).unwrap(), 3);
        assert_eq!(idx4(2, &zero, &inf, &zero).unwrap(), 4);
        assert_eq!(idx4(2, &zero, &inf, &one).unwrap(), 5);

        assert_eq!(idx4(2, &one, &zero, &zero).unwrap(), 6);
        assert_eq!(idx4(2, &one, &zero, &one).unwrap(), 7);
        assert_eq!(idx4(2, &one, &one, &zero).unwrap(), 8);
        assert_eq!(idx4(2, &one, &one, &one).unwrap(), 9);
        assert_eq!(idx4(2, &one, &inf, &zero).unwrap(), 10);
        assert_eq!(idx4(2, &one, &inf, &one).unwrap(), 11);

        assert_eq!(idx4(2, &inf, &zero, &zero).unwrap(), 12);
        assert_eq!(idx4(2, &inf, &zero, &one).unwrap(), 13);
        assert_eq!(idx4(2, &inf, &one, &zero).unwrap(), 14);
        assert_eq!(idx4(2, &inf, &one, &one).unwrap(), 15);
        assert_eq!(idx4(2, &inf, &inf, &zero).unwrap(), 16);
        assert_eq!(idx4(2, &inf, &inf, &one).unwrap(), 17);

        assert_eq!(idx4(1, &zero, &zero, &inf), None);
        assert_eq!(idx4(1, &zero, &inf, &inf), None);

        // Round 3
        // (beta_1, beta_2, beta_3) = (v1, v2, u) where
        // v1 in {0, 1, inf}, v2 in {0, 1, inf}, u in {0, 1, inf}.
        // We order the tuples according to the number we get when using (v1, v2, u) as the coefficients in base 3.
        // There are 3^2 * 3 = 27 tuples in total.
        assert_eq!(idx4(3, &zero, &zero, &zero).unwrap(), 0);
        assert_eq!(idx4(3, &zero, &zero, &one).unwrap(), 1);
        assert_eq!(idx4(3, &zero, &zero, &inf).unwrap(), 2);
        assert_eq!(idx4(3, &zero, &one, &zero).unwrap(), 3);
        assert_eq!(idx4(3, &zero, &one, &one).unwrap(), 4);
        assert_eq!(idx4(3, &zero, &one, &inf).unwrap(), 5);
        assert_eq!(idx4(3, &zero, &inf, &zero).unwrap(), 6);
        assert_eq!(idx4(3, &zero, &inf, &one).unwrap(), 7);
        assert_eq!(idx4(3, &zero, &inf, &inf).unwrap(), 8);

        assert_eq!(idx4(3, &one, &zero, &zero).unwrap(), 9);
        assert_eq!(idx4(3, &one, &zero, &one).unwrap(), 10);
        assert_eq!(idx4(3, &one, &zero, &inf).unwrap(), 11);
        assert_eq!(idx4(3, &one, &one, &zero).unwrap(), 12);
        assert_eq!(idx4(3, &one, &one, &one).unwrap(), 13);
        assert_eq!(idx4(3, &one, &one, &inf).unwrap(), 14);
        assert_eq!(idx4(3, &one, &inf, &zero).unwrap(), 15);
        assert_eq!(idx4(3, &one, &inf, &one).unwrap(), 16);
        assert_eq!(idx4(3, &one, &inf, &inf).unwrap(), 17);

        assert_eq!(idx4(3, &inf, &zero, &zero).unwrap(), 18);
        assert_eq!(idx4(3, &inf, &zero, &one).unwrap(), 19);
        assert_eq!(idx4(3, &inf, &zero, &inf).unwrap(), 20);
        assert_eq!(idx4(3, &inf, &one, &zero).unwrap(), 21);
        assert_eq!(idx4(3, &inf, &one, &one).unwrap(), 22);
        assert_eq!(idx4(3, &inf, &one, &inf).unwrap(), 23);
        assert_eq!(idx4(3, &inf, &inf, &zero).unwrap(), 24);
        assert_eq!(idx4(3, &inf, &inf, &one).unwrap(), 25);
        assert_eq!(idx4(3, &inf, &inf, &inf).unwrap(), 26);
    }

    // BORRAR.
    #[test]
    fn test_from_index_to_beta() {
        let inf = EvaluationPoint::Infinity;
        let zero = EvaluationPoint::Zero;
        let one = EvaluationPoint::One;

        assert_eq!(from_index_to_beta(0), [zero, zero, zero]);
        assert_eq!(from_index_to_beta(1), [zero, zero, one]);
        assert_eq!(from_index_to_beta(2), [zero, zero, inf]);
        assert_eq!(from_index_to_beta(3), [zero, one, zero]);
        assert_eq!(from_index_to_beta(4), [zero, one, one]);
        assert_eq!(from_index_to_beta(5), [zero, one, inf]);
        assert_eq!(from_index_to_beta(6), [zero, inf, zero]);
        assert_eq!(from_index_to_beta(7), [zero, inf, one]);
        assert_eq!(from_index_to_beta(8), [zero, inf, inf]);

        assert_eq!(from_index_to_beta(9), [one, zero, zero]);
        assert_eq!(from_index_to_beta(10), [one, zero, one]);
        assert_eq!(from_index_to_beta(11), [one, zero, inf]);
        assert_eq!(from_index_to_beta(12), [one, one, zero]);
        assert_eq!(from_index_to_beta(13), [one, one, one]);
        assert_eq!(from_index_to_beta(14), [one, one, inf]);
        assert_eq!(from_index_to_beta(15), [one, inf, zero]);
        assert_eq!(from_index_to_beta(16), [one, inf, one]);
        assert_eq!(from_index_to_beta(17), [one, inf, inf]);

        assert_eq!(from_index_to_beta(18), [inf, zero, zero]);
        assert_eq!(from_index_to_beta(19), [inf, zero, one]);
        assert_eq!(from_index_to_beta(20), [inf, zero, inf]);
        assert_eq!(from_index_to_beta(21), [inf, one, zero]);
        assert_eq!(from_index_to_beta(22), [inf, one, one]);
        assert_eq!(from_index_to_beta(23), [inf, one, inf]);
        assert_eq!(from_index_to_beta(24), [inf, inf, zero]);
        assert_eq!(from_index_to_beta(25), [inf, inf, one]);
        assert_eq!(from_index_to_beta(26), [inf, inf, inf]);
    }

    // #[test]
    // fn print_calculate_p_beta() {
    //     let current_evals: Vec<&F> = (1..9).map(|i| F::from_int(i)).collect();
    //     println!("Current_evals: {:?}", current_evals);
    //     let result = calculate_p_beta(current_evals);
    //     println!("Result: {:?}", result);
    // }

    #[test]
    fn test_compute_acumulators() {
        let poly_1: EvaluationsList<F> =
            EvaluationsList::new((0..16).map(|i| F::from_int(i)).collect());
        let poly_2: EvaluationsList<F> =
            EvaluationsList::new((0..16).map(|i| F::from_int(i)).collect());
        let [
            accumulator_round_1,
            accumulator_round_2,
            accumulator_round_3,
        ] = compute_accumulators(&poly_1, &poly_2);

        // p(x) = q(x) => p(x)q(x) = p(x)ˆ2

        // A3(0,0,0) = p(0,0,0,0)^2 + p(0,0,0,1)^2 = 0^2 + 1^2 = 1
        assert_eq!(accumulator_round_3.accumulators[0], F::from_int(1));

        // A3(0,0,1) = p(0,0,1,0)^2 + p(0,0,1,1)^2 = 2^2 + 3^2 = 4 + 9 = 13
        assert_eq!(accumulator_round_3.accumulators[1], F::from_int(13));

        // A3(0,0,inf) = p(0,0,inf,0)^2 + p(0,0,inf,1)^2
        //             = (p(0,0,1,0) - p(0,0,0,0))^2 + (p(0,0,1,1) - p(0,0,0,1))^2
        //             = (2 - 0)^2 + (3 - 1)^2 = 2^2 + 2^
        //             = 8
        assert_eq!(accumulator_round_3.accumulators[2], F::from_int(8));

        // A3(inf,1,inf) = p(inf,1,inf,0)^2 + p(inf,1,inf,1)^2
        //             = (p(1,1,inf,0) - p(0,1,inf,0))^2 + (p(1,1,inf,1) - p(0,1,inf,1)))^2
        //             = (p(1,1,1,0)- p(1,1,0,0) - (p(0,1,1,0) - p(0,1,0,0)))ˆ2 + (p(1,1,1,1)- p(1,1,0,1) - (p(0,1,1,1) - p(0,1,0,1)))ˆ2
        //             = ((14 - 12) - (6 - 4))ˆ2 + ((15 - 13) - (7 - 5))ˆ2
        //             = (2 - 2)^2 + (2 - 2)^2
        //             = 0
        assert_eq!(accumulator_round_3.accumulators[23], F::from_int(0));

        // A2(0,0) = p(0,0,0,0)ˆ2 + p(0,0,0,1)ˆ2 + p(0,0,1,0)ˆ2 + p(0,0,1,1)ˆ2
        //         = 0ˆ2 + 1ˆ2 + 2ˆ2 + 3ˆ2
        //         = 14
        assert_eq!(accumulator_round_2.accumulators[0], F::from_int(14));

        // A2(0,1) = p(0,1,0,0)ˆ2 + p(0,1,0,1)ˆ2 + p(0,1,1,0)ˆ2 + p(0,1,1,1)ˆ2
        //         = 4ˆ2 + 5ˆ2 + 6ˆ2 + 7ˆ2
        //         = 126
        assert_eq!(accumulator_round_2.accumulators[1], F::from_int(126));

        // A2(0,inf) = p(0,inf,0,0)ˆ2 + p(0,inf,0,1)ˆ2 + p(0,inf,1,0)ˆ2 + p(0,inf,1,1)ˆ2
        //           = (p(0,1,0,0)- p(0,0,0,0))ˆ2 + (p(0,1,0,1)- p(0,0,0,1))ˆ2 + (p(0,1,1,0)- p(0,0,1,0))ˆ2 + (p(0,1,1,1)- p(0,0,1,1))ˆ2
        //           = (4 - 0)ˆ2 + (5 - 1)ˆ2 + (6 - 2)ˆ2 + (7 - 3)ˆ2
        //           = 4 * 4ˆ2
        //           = 64
        assert_eq!(accumulator_round_2.accumulators[2], F::from_int(64));

        // A2(inf, 1) = = p(inf,1,0,0)ˆ2 + p(inf,1,0,1)ˆ2 + p(inf,1,1,0)ˆ2 + p(inf,1,1,1)ˆ2
        //         = 8^2 + 8^2 + 8^2 + 8^2 (haciendo la cuenta se ve que los 4 términos valen 8^2)
        //         = 4 * 8^2
        //         = 256
        assert_eq!(accumulator_round_2.accumulators[7], F::from_int(256));

        // A1(0) = p(0,0,0,0)^2 + p(0,0,0,1)^2 + p(0,0,1,0)^2 + p(0,0,1,1)^2
        //       + p(0,1,0,0)^2 + p(0,1,0,1)^2 + p(0,1,1,0)^2 + p(0,1,1,1)^2
        //       = 0^2 + 1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2 + 7^2
        //       = 140
        assert_eq!(accumulator_round_1.accumulators[0], F::from_int(140));

        // A1(1) = p(1,0,0,0)^2 + p(1,0,0,1)^2 + p(1,0,1,0)^2 + p(1,0,1,1)^2
        //       + p(1,1,0,0)^2 + p(1,1,0,1)^2 + p(1,1,1,0)^2 + p(1,1,1,1)^2
        //       = 8^2 + 9^2 + 10^2 + 11^2 + 12^2 + 13^2 + 14^2 + 15^2
        //       = 1100
        assert_eq!(accumulator_round_1.accumulators[1], F::from_int(1100));

        // A1(inf) = p(inf,0,0,0)^2 + p(inf,0,0,1)^2 + p(inf,0,1,0)^2 + p(inf,0,1,1)^2
        //         + p(inf,1,0,0)^2 + p(inf,1,0,1)^2 + p(inf,1,1,0)^2 + p(inf,1,1,1)^2
        //         = (8^2) * 8 (haciendo la cuenta se ve que los 8 términos valen 8^2)
        //         = 512
        assert_eq!(accumulator_round_1.accumulators[2], F::from_int(512));
    }

    #[test]
    fn test_svo_sumcheck_rounds_simulation() {
        // In this test, we simulate the function `small_value_sumcheck_three_rounds` using r1 = 2 and r2 = 1 as challenges,
        // instead of sampling them, so that we can test that the univariate polynomials S_1, S_2 and S_3 are
        // the same as the ones we computed by hand.

        let poly_1 = EvaluationsList::new((0..8).map(|i| F::from_int(i)).collect());
        let poly_2 = EvaluationsList::new((0..8).map(|i| F::from_int(i)).collect());

        let round_accumulators = compute_accumulators(&poly_1, &poly_2);

        // Round 1

        // 1. For u in {0, 1, inf} compute S_1(u).
        // Recall: S_1(u) = A_1(u).

        let round_poly_evals = &round_accumulators[0].accumulators;

        // 2. We check S_1(x) = 64 * x^2 + 48 * x + 14 (we computed it by hand).
        assert_eq!(round_poly_evals[0], F::from_int(14)); // S_1(0)
        assert_eq!(round_poly_evals[1], F::from_int(126)); // S_1(1)
        assert_eq!(round_poly_evals[2], F::from_int(64)); // S_1(inf)

        // 3. Receive the challenge r_1 from the verifier. We fix it as 2 for testing.
        let r_1: EF = EF::TWO;

        // 4. Compte R_2 = [L_0(r_1), L_1(r_1), L_inf(r_1)]
        // L_0 (x) = 1 - x
        // L_1 (x) = x
        // L_inf (x) = (x - 1)x
        let lagrange_evals_r_1 = [-r_1 + F::ONE, r_1, (r_1 - F::ONE) * r_1];

        // Round 2

        // 1. For u in {0, 1, inf} compute S_2(u).
        // First we take the accumulators A_2(v, u).
        // There are 9 accumulators, since v in {0, 1, inf} and u in {0, 1, inf}.
        let accumulators_2 = &round_accumulators[1].accumulators;

        let mut round_poly_evals = [EF::ZERO; 3];

        for (lagrange_index, chunk) in accumulators_2.chunks_exact(3).enumerate() {
            round_poly_evals[0] += lagrange_evals_r_1[lagrange_index] * chunk[0];
            round_poly_evals[1] += lagrange_evals_r_1[lagrange_index] * chunk[1];
            round_poly_evals[2] += lagrange_evals_r_1[lagrange_index] * chunk[2];
        }

        // 2. We check S_2(x) = 8 * x^2 + 68 * x + 145 (we computed it by hand).
        assert_eq!(round_poly_evals[0], EF::from(F::from_int(145))); // S_2(0)
        assert_eq!(round_poly_evals[1], EF::from(F::from_int(221))); // S_2(1)
        assert_eq!(round_poly_evals[2], EF::from(F::from_int(8))); // S_2(inf)

        // 3. Receive the challenge r_2 from the verifier. We fix it as 1 for testing.
        let r_2: EF = EF::ONE;

        // 4. Compute R_3 = [L_00(r_1, r_2), L_01(r_1, r_2), ..., L_{inf inf}(r_1, r_2)]
        // L_00 (x1, x2) = (1 - x1) * (1 - x2)
        // L_01 (x1, x2) = (1 - x1) * x2
        // ...
        // L_{inf inf} (x1, x2) = (x1 - 1) * x1 * (x2 - 1) * x2

        let l_0 = lagrange_evals_r_1[0];
        let l_1 = lagrange_evals_r_1[1];
        let l_inf = lagrange_evals_r_1[2];

        // TODO: calcular `(r_2 - F::ONE) * r_2` una sola vez. Lo dejo por ahora así por claridad.
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

        // Round 3

        // 1. For u in {0, 1, inf} compute S_3(u).

        // First we take the accumulators A_2(v, u).
        // There are 27 accumulators, since v in {0, 1, inf}^2 and u in {0, 1, inf}.
        let accumulators_3 = &round_accumulators[2].accumulators;

        let mut round_poly_evals = [EF::ZERO; 3];

        // We split
        for (lagrange_index, accumulators_chunk) in accumulators_3.chunks_exact(9).enumerate() {
            // Los primeros 3 elementos: u2 = 0
            round_poly_evals[0] += lagrange_evals_r_2[lagrange_index * 3] * accumulators_chunk[0];
            round_poly_evals[1] += lagrange_evals_r_2[lagrange_index * 3] * accumulators_chunk[1];
            round_poly_evals[2] += lagrange_evals_r_2[lagrange_index * 3] * accumulators_chunk[2];

            // Los siguientes 3 elementos: u2 = 1
            round_poly_evals[0] +=
                lagrange_evals_r_2[lagrange_index * 3 + 1] * accumulators_chunk[3];
            round_poly_evals[1] +=
                lagrange_evals_r_2[lagrange_index * 3 + 1] * accumulators_chunk[4];
            round_poly_evals[2] +=
                lagrange_evals_r_2[lagrange_index * 3 + 1] * accumulators_chunk[5];

            // Los últimos 3 elementos: u2 = inf
            round_poly_evals[0] +=
                lagrange_evals_r_2[lagrange_index * 3 + 2] * accumulators_chunk[6];
            round_poly_evals[1] +=
                lagrange_evals_r_2[lagrange_index * 3 + 2] * accumulators_chunk[7];
            round_poly_evals[2] +=
                lagrange_evals_r_2[lagrange_index * 3 + 2] * accumulators_chunk[8];
        }

        // 2. We check S_3(x) = x^2 + 20 * x + 100 (we computed it by hand).
        assert_eq!(round_poly_evals[0], EF::from(F::from_int(100))); // S_3(0)
        assert_eq!(round_poly_evals[1], EF::from(F::from_int(121))); // S_3(1)
        assert_eq!(round_poly_evals[2], EF::from(F::from_int(1))); // S_3(inf)
    }

    #[test]
    fn test_evals_serial_three_vars_matches_new_from_point() {
        let p = [F::from_u64(2), F::from_u64(3), F::from_u64(5)];
        let point = MultilinearPoint::new(p.to_vec());
        let value = F::from_u64(1);

        let via_method = EvaluationsList::new_from_point(&point, value)
            .into_iter()
            .collect::<Vec<_>>();
        let via_serial = eval_eq(&p);

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

        let out = eval_eq(&[p0, p1, p2]);
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

            EF::from_basis_coefficients_slice(&[F::from_u32(r1), F::from_u32(r2), F::from_u32(r3), F::from_u32(r4)]).unwrap()
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
        assert_eq!(
            e_out[0][7],
            (EF::ONE - w[1]) * w[2] * w[8] * w[9]);
            
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
    fn test_compute_accumulators_eq() {

        // We'll use polynomials of 10 variables.
        let l = 10;

        let mut rng = rand::rng();
        
        // w = [w0, w2, ..., w9]
        // Each w_i is a random extension field element built from 4 random field elements.
        let w: Vec<EF> = (0..l)
            .map(|_| {
                let r1: u32 = rng.next_u32();
                let r2: u32 = rng.next_u32();
                let r3: u32 = rng.next_u32();
                let r4: u32 = rng.next_u32();

                EF::from_basis_coefficients_slice(&[F::from_u32(r1), F::from_u32(r2), F::from_u32(r3), F::from_u32(r4)]).unwrap()
            })
            .collect();

        // We build a random multilinear polynomial of 10 variables, using 2^10 evaluations in the hypercube {0,1}^10
        let poly = EvaluationsList::new((0..(1 << l))
            .map(|_| {
                let r: u32 = rng.next_u32(); //
                F::from_u32(r)
            })
            .collect());
        
        // We precompute E_in and E_out
        let e_in = precompute_e_in(&w);
        let e_out = precompute_e_out(&w);

        // We compute the accumulators.
        let accumulators = compute_accumulators_eq(&poly, e_in, e_out);

        // We want to check that A_3(0,0,0) is correct.
        // We now compute it manually.

        // eq_w3_w4 = [eq(w3, w4; 0, 0), eq(w3, w4; 0, 1), eq(w3, w4; 1, 0), eq(w3, w4; 1, 1)]
        let eq_w3_w4 = [
            (EF::ONE - w[3]) * (EF::ONE - w[4]),
            (EF::ONE - w[3]) * w[4],
            w[3] * (EF::ONE - w[4]),
            w[3] * w[4],
        ];

        // We now compute eq(w5, w6, w7, w8, w9; b) for all b in {0,1}^5
        let w_input = [w[5], w[6], w[7], w[8], w[9]];
        let eq_w5_to_w9 = eval_eq(&w_input);

        // We compute A_3(0, 0, 0)
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

        // We compute A_3(1,0,1):
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
        let p_evals: [Vec<F>;4] = [poly.iter().skip(768).take(32).zip(poly.iter().skip(512).take(32)).map(|(p1, p0)| *p1 - *p0).collect::<Vec<F>>(), 
                                   poly.iter().skip(800).take(32).zip(poly.iter().skip(544).take(32)).map(|(p1, p0)| *p1 - *p0).collect::<Vec<F>>(),
                                   poly.iter().skip(832).take(32).zip(poly.iter().skip(576).take(32)).map(|(p1, p0)| *p1 - *p0).collect::<Vec<F>>(),
                                   poly.iter().skip(864).take(32).zip(poly.iter().skip(608).take(32)).map(|(p1, p0)| *p1 - *p0).collect::<Vec<F>>()];

        
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
        let eq_w2_w3_w4 = [
            (EF::ONE - w[2]) * (EF::ONE - w[3]) * (EF::ONE - w[4]),
            (EF::ONE - w[2]) * (EF::ONE - w[3]) * w[4],
            (EF::ONE - w[2]) * w[3] * (EF::ONE - w[4]),
            (EF::ONE - w[2]) * w[3] * w[4],
            w[2] * (EF::ONE - w[3]) * (EF::ONE - w[4]),
            w[2] * (EF::ONE - w[3]) * w[4],
            w[2] * w[3] * (EF::ONE - w[4]),
            w[2] * w[3] * w[4],
        ];

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
    

        // A_1(u)
        let eq_w1_w2_w3_w4 = [
            (EF::ONE - w[1]) * (EF::ONE - w[2]) * (EF::ONE - w[3]) * (EF::ONE - w[4]),
            (EF::ONE - w[1]) * (EF::ONE - w[2]) * (EF::ONE - w[3]) * w[4],
            (EF::ONE - w[1]) * (EF::ONE - w[2]) * w[3] * (EF::ONE - w[4]),
            (EF::ONE - w[1]) * (EF::ONE - w[2]) * w[3] * w[4],
            (EF::ONE - w[1]) * w[2] * (EF::ONE - w[3]) * (EF::ONE - w[4]),
            (EF::ONE - w[1]) * w[2] * (EF::ONE - w[3]) * w[4],
            (EF::ONE - w[1]) * w[2] * w[3] * (EF::ONE - w[4]),
            (EF::ONE - w[1]) * w[2] * w[3] * w[4],
            w[1] * (EF::ONE - w[2]) * (EF::ONE - w[3]) * (EF::ONE - w[4]),
            w[1] * (EF::ONE - w[2]) * (EF::ONE - w[3]) * w[4],
            w[1] * (EF::ONE - w[2]) * w[3] * (EF::ONE - w[4]),
            w[1] * (EF::ONE - w[2]) * w[3] * w[4],
            w[1] * w[2] * (EF::ONE - w[3]) * (EF::ONE - w[4]),
            w[1] * w[2] * (EF::ONE - w[3]) * w[4],
            w[1] * w[2] * w[3] * (EF::ONE - w[4]),
            w[1] * w[2] * w[3] * w[4],
        ]; 

        // We compute A_3(0)
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
    }
}
