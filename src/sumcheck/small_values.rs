use crate::{
    fiat_shamir::prover::ProverState,
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    whir::statement::Statement,
};
use p3_field::{ExtensionField, Field, TwoAdicField};

const NUM_OF_ROUNDS: usize = 3;
// Objetivo: Implementar el Procedure 7 en https://eprint.iacr.org/2025/1117.pdf (page 34)

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
struct RoundAccumlators<F: Field> {
    round: usize,
    accumulators: Vec<F>,
}

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
            if beta_2 == 2 || beta_3 == 2 {
                return None; // y1 and y2 must be in {0, 1}
            }
            // (beta_1, beta_2, beta_3) = (u, y1, y2) where:
            // u in {0, 1, inf}, y1 in {0, 1}, y2 in {0, 1}.
            // There are 3 * 2^2 = 12 tuples.
            // We order the tuples by representing u in binary and concatenating it with y1 and y2.
            // So, the index is the concatenation u || y
            let u = beta_1;
            let y = (beta_2 << 1) | beta_3;
            let index = (u << 2) | y;
            return Some(index);
        }
        2 => {
            if beta_3 == 2 {
                return None; // y must be in {0, 1}
            }
            // (beta_1, beta_2, beta_3) = (v, u, y) where:
            // v in {0, 1, inf}, u in {0, 1, inf}, y in {0, 1}.
            // There are 3 * 3 * 2 = 18 tuples.
            // We order the tuples in the following way:
            // We think (v, u) as the coefficients of a base 3 number, and then we concatenate its binary representation with y.
            // So, the index is the concatenation (v * 3 + u) || y.
            let v = beta_1;
            let u = beta_2;
            let y = beta_3;
            let index = ((v * 3 + u) << 1) | y;
            return Some(index);
        }
        3 => {
            // (beta_1, beta_2, beta_3) = (v1, v2, u) where:
            // v1 in {0, 1, inf}, v2 in {0, 1, inf}, u in {0, 1, inf}.
            // We order the tuples according to the number we get when using (v1, v2, u) as the coefficients in base 3.
            // So, the index is v1 * 3^2 + v2 * 3 + u.
            let v1 = beta_1;
            let v2 = beta_2;
            let u = beta_3;
            let index = v1 * 9 + v2 * 3 + u;
            return Some(index);
        }
        _ => unreachable!(),
    }
}

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

// Procedure 6. Page 34
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
// Fijado x'' en {0, 1}^{l-3}, dadas las evaluaciones del multilineal q(x1, x2, x3) = p(x1, x2, x3, x'') en el booleano devuelve las
// evaluaciones de q en beta para todo beta in {0, 1, inf}^3.
fn calculate_p_beta<F: Field>(current_evals: Vec<F>) -> Vec<F> {
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

fn compute_accumulators<F: Field>(
    poly_1: Vec<F>,
    poly_2: Vec<F>,
) -> [RoundAccumlators<F>; NUM_OF_ROUNDS] {
    assert_eq!(poly_1.num_variables(), poly_2.num_variables());
    let l = poly_1.num_variables();
    // x'' in {0 .. 2^{l - 3}}
    (0..1 << (l - NUM_OF_ROUNDS)).map(|x| {
        //for offset in 0..p {
        // let res: Vec<_> = poly_1
        //     .iter()
        //     .skip(x)
        //     .step_by(1 << (l - NUM_OF_ROUNDS))
        //     .map(|current_evals| calculate_p_beta(current_evals))
        //     .collect();

        let current_evals = poly_1.skip(x).step_by(1 << (l - NUM_OF_ROUNDS)).collect();
        calculate_p_beta(current_evals);

        // Procedure 6

        // 27 = 3 ^ NUM_OF_ROUNDS
        (0..27).map(|beta_index| {
            let beta = from_index_to_beta(beta_index);

            // We need to implement the evaluation p(beta, x'')

            let poly_1_eval = evaluate(poly_1);
            //let poly_2_eval = evaluate();
        })
    })
}

// for offset in 0..p {
//     let res: Vec<_> = v
//         .iter()
//         .skip(offset)
//         .step_by(p)
//         .map(|x| x * 2) // <-- uso el map directo
//         .collect();

#[cfg(test)]
mod tests {
    use super::*;
    use p3_baby_bear::BabyBear;
    use p3_field::integers::QuotientMap;
    type F = BabyBear;

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

    #[test]
    fn print_calculate_p_beta() {
        let current_evals: Vec<F> = (1..9).map(|i| F::from_int(i)).collect();
        println!("Current_evals: {:?}", current_evals);
        let result = calculate_p_beta(current_evals);
        println!("Result: {:?}", result);
    }
}
