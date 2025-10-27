use p3_field::Field;
use std::ops::Add;

pub const NUM_OF_ROUNDS: usize = 3;

#[derive(Debug, Clone, Eq, PartialEq, Copy)]
pub enum EvaluationPoint {
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
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Accumulators<F: Field> {
    pub accumulators: [Vec<F>; 3],
}

impl<F> Accumulators<F>
where
    F: Field,
{
    pub fn new_empty() -> Self {
        Accumulators {
            accumulators: [vec![F::ZERO; 3], vec![F::ZERO; 9], vec![F::ZERO; 27]],
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

impl<F> RoundAccumlators<F>
where
    F: Field,
{
    pub fn new(round: usize, accumulators: Vec<F>) -> Self {
        RoundAccumlators {
            round,
            accumulators,
        }
    }

    // Given the round i, it returns the accumulators A_i(u, v) all initialized in zero.
    pub fn new_empty(round: usize) -> Self {
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

    pub fn accumulate_eval(&mut self, eval: F, index: usize) {
        self.accumulators[index] += eval;
    }
}

// We asseume n in {0, ..., 26}
pub fn to_base_three_coeff(n: usize) -> [usize; 3] {
    let mut n = n;
    let mut coeffs = [0; 3];
    for i in (0..NUM_OF_ROUNDS).rev() {
        coeffs[i] = n % 3;
        n /= 3;
    }
    coeffs
}

pub fn idx4(index_beta: usize) -> [Option<usize>; 3] {
    let [b1, b2, b3] = to_base_three_coeff(index_beta);

    match (b1, b2, b3) {
        (_, _, 2) => [None, None, Some(b1 * 9 + b2 * 3 + b3)],
        (_, 2, _) => [None, Some(b1 * 3 + b2), Some(b1 * 9 + b2 * 3 + b3)],
        _ => [Some(b1), Some(b1 * 3 + b2), Some(b1 * 9 + b2 * 3 + b3)],
    }
}

pub fn idx4_v2(index_beta: usize) -> [Option<usize>; 3] {
    let [b1, b2, b3] = to_base_three_coeff(index_beta);

    match (b1, b2, b3) {
        (_, _, 2) | (_, 2, _) | (2, _, _) => [None, None, None],
        _ => [Some(b1), Some(b1 * 3 + b2), Some(b1 * 9 + b2 * 3 + b3)],
    }
}

// Implement Procedure 6 (Page 34).
// Fijado x'' en {0, 1}^{l-3}, dadas las evaluaciones del multilineal q(x1, x2, x3) = p(x1, x2, x3, x'') en el booleano devuelve las
// evaluaciones de q en beta para todo beta in {0, 1, inf}^3.
pub fn compute_p_beta<F: Field>(current_evals: &[F; 8], next_evals: &mut [F; 27]) {
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
}
