use crate::{
    fiat_shamir::prover::ProverState,
    poly::evals::EvaluationsList,
    sumcheck::small_value_utils::{NUM_OF_ROUNDS, RoundAccumlators, compute_p_beta, idx4},
};
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field};

// Implements the Procedure 7 in https://eprint.iacr.org/2025/1117.pdf (page 34).
pub fn compute_accumulators<F: Field>(
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
            ] = idx4(beta_index);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::evals::EvaluationsList;
    use p3_baby_bear::BabyBear;
    use p3_field::{
        PrimeCharacteristicRing, extension::BinomialExtensionField, integers::QuotientMap,
    };

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

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
}
