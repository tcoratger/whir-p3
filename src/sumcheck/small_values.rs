use p3_field::{ExtensionField, Field, TwoAdicField};
use crate::{
    fiat_shamir::prover::ProverState,
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    whir::statement::Statement,
};


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
            _ => unreachable!()
        }

    }
}

// For round i, RoundAccumulators has all the accumulators of the form A_i(u, v).
struct RoundAccumlators<F: Field> {
    round: usize,
    accumulators: Vec<F>
}


// Esta función mapea una tupla del tipo (i,v,u,y) que resulta de aplicar indx4(\beta) a un indice que correpende a un acumulador especifico.
// Recall: v in {0, 1, inf}^{i-1}, u in {0, 1, inf}, y in {0, 1}^{3-i}, where i is the number of round.
// Vamos a tomar u y v como coeficientes en base 3 (donde inf lo tomamos como 2), los representamos en binario y los concatenamos con y en su forma binaria.
// Observar que en cada ronda cambian los tamaños de v, u and y.
fn from_beta_to_index(
    round: u8,
    beta_1: &EvaluationPoint,
    beta_2: &EvaluationPoint,
    beta_3: &EvaluationPoint,
) -> usize {
    let beta_1 = beta_1.to_usize_representation();
    let beta_2 = beta_2.to_usize_representation();
    let beta_3 = beta_3.to_usize_representation();

    match round {
        1 => {
            // (beta_1, beta_2, beta_3) = (u, y1, y2) where:
            // u in {0, 1, inf}, y1 in {0, 1}, y2 in {0, 1}.
            // There are 3 * 2^2 = 12 tuples.
            // We order the tuples by representing u in binary and concatenating it with y1 and y2.
            // So, the index is the concatenation u || y
            let u = beta_1;
            let y = (beta_2 << 1) | beta_3;
            let index = (u << 2) | y;
            return index;
        }
        2 => {
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
            return index;
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
            return index;
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

fn calculate_p_beta<F: Field>(current_evals: &mut Vec<F>) {
    
    // Round 1:
    // TODO: refactorizar esto!!
    let v: Vec<usize> = (0..8).collect();

    for (k0, k1) in v.iter()
        .take(v.len() / 2)
        .zip(v.iter().skip(v.len() / 2))
    {
        current_evals.push(current_evals[*k1] - current_evals[*k0]);        
    }
}

// L is the number of variables of the multilinear polynomials.
// fn compute_accumulators<F: Field>(poly_1: EvaluationsList<F>, poly_2: EvaluationsList<F>) -> [RoundAccumlators<F>; NUM_OF_ROUNDS] {
//     assert_eq!(poly_1.num_variables(), poly_2.num_variables());
//     let l = poly_1.num_variables();
//     // x'' in {0 .. 2^{l - 3}}
//     (0 .. 1 << (l-NUM_OF_ROUNDS)).map(|x|  {
//         // Procedure 6


//         // 27 = 3 ^ NUM_OF_ROUNDS
//         (0..27).map(|beta_index| {
//             let beta = from_index_to_beta(beta_index);

//             // We need to implement the evaluation p(beta, x'')

//             let poly_1_eval = evaluate(poly_1, ); 
//             let poly_2_eval = evaluate();

//         })
//     })
    
// }

#[cfg(test)]
mod tests {
    use super::*;
    use p3_baby_bear::BabyBear;
    use p3_field::integers::QuotientMap;
    type F = BabyBear;
    

    #[test]
    fn test_calculate_p_beta() {
        let mut current_evals: Vec<F> = (100..=107).map(|i| F::from_int(i)).collect();

        println!("{:?}", current_evals);
        
        calculate_p_beta(&mut current_evals);

        println!("{:?}", current_evals);
    }


    #[test]
    fn test_from_beta_to_index() {
        let inf = EvaluationPoint::Infinity;
        let zero = EvaluationPoint::Zero;
        let one = EvaluationPoint::One;

        // Round 1
        // (beta_1, beta_2, beta_3) = (u, y1, y2) where
        // u in {0, 1, inf}, y1 in {0, 1}, y2 in {0, 1}.
        // We order the tuples by representing u in binary and concatenating it with y1 and y2.
        // There are 3 * 2^2 = 12 tuples.
        assert_eq!(from_beta_to_index(1, &zero, &zero, &zero), 0);
        assert_eq!(from_beta_to_index(1, &zero, &zero, &one), 1);
        assert_eq!(from_beta_to_index(1, &zero, &one, &zero), 2);
        assert_eq!(from_beta_to_index(1, &zero, &one, &one), 3);
        assert_eq!(from_beta_to_index(1, &one, &zero, &zero), 4);
        assert_eq!(from_beta_to_index(1, &one, &zero, &one), 5);
        assert_eq!(from_beta_to_index(1, &one, &one, &zero), 6);
        assert_eq!(from_beta_to_index(1, &one, &one, &one), 7);
        assert_eq!(from_beta_to_index(1, &inf, &zero, &zero), 8);
        assert_eq!(from_beta_to_index(1, &inf, &zero, &one), 9);
        assert_eq!(from_beta_to_index(1, &inf, &one, &zero), 10);
        assert_eq!(from_beta_to_index(1, &inf, &one, &one), 11);

        // Round 2
        // (beta_1, beta_2, beta_3) = (v, u, y) where
        // v in {0, 1, inf}, u in {0, 1, inf}, y in {0, 1}.
        // We order the tuples in the following way:
        // We think (v, u) as the coefficients of a base 3 number, and then we concatenate its binary representation with y.
        // There are 3 * 3 * 2 = 18 tuples.
        assert_eq!(from_beta_to_index(2, &zero, &zero, &zero), 0);
        assert_eq!(from_beta_to_index(2, &zero, &zero, &one), 1);
        assert_eq!(from_beta_to_index(2, &zero, &one, &zero), 2);
        assert_eq!(from_beta_to_index(2, &zero, &one, &one), 3);
        assert_eq!(from_beta_to_index(2, &zero, &inf, &zero), 4);
        assert_eq!(from_beta_to_index(2, &zero, &inf, &one), 5);

        assert_eq!(from_beta_to_index(2, &one, &zero, &zero), 6);
        assert_eq!(from_beta_to_index(2, &one, &zero, &one), 7);
        assert_eq!(from_beta_to_index(2, &one, &one, &zero), 8);
        assert_eq!(from_beta_to_index(2, &one, &one, &one), 9);
        assert_eq!(from_beta_to_index(2, &one, &inf, &zero), 10);
        assert_eq!(from_beta_to_index(2, &one, &inf, &one), 11);

        assert_eq!(from_beta_to_index(2, &inf, &zero, &zero), 12);
        assert_eq!(from_beta_to_index(2, &inf, &zero, &one), 13);
        assert_eq!(from_beta_to_index(2, &inf, &one, &zero), 14);
        assert_eq!(from_beta_to_index(2, &inf, &one, &one), 15);
        assert_eq!(from_beta_to_index(2, &inf, &inf, &zero), 16);
        assert_eq!(from_beta_to_index(2, &inf, &inf, &one), 17);

        // Round 3
        // (beta_1, beta_2, beta_3) = (v1, v2, u) where
        // v1 in {0, 1, inf}, v2 in {0, 1, inf}, u in {0, 1, inf}.
        // We order the tuples according to the number we get when using (v1, v2, u) as the coefficients in base 3.
        // There are 3^2 * 3 = 27 tuples in total.
        assert_eq!(from_beta_to_index(3, &zero, &zero, &zero), 0);
        assert_eq!(from_beta_to_index(3, &zero, &zero, &one), 1);
        assert_eq!(from_beta_to_index(3, &zero, &zero, &inf), 2);
        assert_eq!(from_beta_to_index(3, &zero, &one, &zero), 3);
        assert_eq!(from_beta_to_index(3, &zero, &one, &one), 4);
        assert_eq!(from_beta_to_index(3, &zero, &one, &inf), 5);
        assert_eq!(from_beta_to_index(3, &zero, &inf, &zero), 6);
        assert_eq!(from_beta_to_index(3, &zero, &inf, &one), 7);
        assert_eq!(from_beta_to_index(3, &zero, &inf, &inf), 8);

        assert_eq!(from_beta_to_index(3, &one, &zero, &zero), 9);
        assert_eq!(from_beta_to_index(3, &one, &zero, &one), 10);
        assert_eq!(from_beta_to_index(3, &one, &zero, &inf), 11);
        assert_eq!(from_beta_to_index(3, &one, &one, &zero), 12);
        assert_eq!(from_beta_to_index(3, &one, &one, &one), 13);
        assert_eq!(from_beta_to_index(3, &one, &one, &inf), 14);
        assert_eq!(from_beta_to_index(3, &one, &inf, &zero), 15);
        assert_eq!(from_beta_to_index(3, &one, &inf, &one), 16);
        assert_eq!(from_beta_to_index(3, &one, &inf, &inf), 17);

        assert_eq!(from_beta_to_index(3, &inf, &zero, &zero), 18);
        assert_eq!(from_beta_to_index(3, &inf, &zero, &one), 19);
        assert_eq!(from_beta_to_index(3, &inf, &zero, &inf), 20);
        assert_eq!(from_beta_to_index(3, &inf, &one, &zero), 21);
        assert_eq!(from_beta_to_index(3, &inf, &one, &one), 22);
        assert_eq!(from_beta_to_index(3, &inf, &one, &inf), 23);
        assert_eq!(from_beta_to_index(3, &inf, &inf, &zero), 24);
        assert_eq!(from_beta_to_index(3, &inf, &inf, &one), 25);
        assert_eq!(from_beta_to_index(3, &inf, &inf, &inf), 26);
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
}

//     Ronda 1:        Ronda 2:        Ronda 3:

// 000         1               1               1
// 001         2               2               1
// 002                                        (1)
// 010         3               1               2
// 011         4               2               2
// 012                                        (2)
// 020                        (1)              3
// 021                        (2)              3
// 022                                        (3)
// 100         1               3               4
// 101         2               4               4
// 102                                        (4)
// 110         3               3               5
// 111         4               4               5
// 112                                        (5)
// 120                        (3)              6
// 121                        (4)              6
// 122                                        (6)
// 200        (1)              5               7
// 201        (2)              6               7
// 202                                        (7)
// 210        (3)              5               8
// 211        (4)              6               8
// 212                                        (8)
// 220                        (5)              9
// 221                        (6)              9
// 222                                        (9)