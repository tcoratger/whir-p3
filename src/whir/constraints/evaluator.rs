use p3_field::{ExtensionField, Field, TwoAdicField};

use crate::{
    constant::K_SKIP_SUMCHECK,
    parameters::FoldingFactor,
    poly::multilinear::MultilinearPoint,
    whir::{constraints::statement::Statement, parameters::WhirConfig},
};

/// Lightweight evaluator for the combined constraint polynomial W(r).
#[derive(Clone, Debug)]
pub struct ConstraintPolyEvaluator {
    /// Number of variables in the multilinear polynomial space.
    pub num_variables: usize,
    /// The folding factor.
    pub folding_factor: FoldingFactor,
    /// Boolean indicating whether the univariate skip optimization is active.
    pub univariate_skip: bool,
}

impl ConstraintPolyEvaluator {
    /// Creates a new `ConstraintPolyEvaluator` with the given parameters.
    #[must_use]
    pub const fn new(
        num_variables: usize,
        folding_factor: FoldingFactor,
        univariate_skip: bool,
    ) -> Self {
        Self {
            num_variables,
            folding_factor,
            univariate_skip,
        }
    }

    /// Evaluate the combined constraint polynomial W(r).
    ///
    /// `constraints` is a vector with one entry per round; each entry is
    /// `(alpha_powers_for_that_round, constraints_in_that_round)`.
    ///
    /// `deferred` contains precomputed evaluations for any constraints that set
    /// `defer_evaluation = true` (they are consumed in order).
    ///
    /// `point` is the verifier's final challenge:
    /// - standard case: length = n
    /// - skip case:     length = (n - K_SKIP_SUMCHECK) + 1  (r_rest || r_skip)
    #[must_use]
    pub fn eval_constraints_poly<EF>(
        &self,
        constraints: &[(Vec<EF>, Statement<EF>)],
        point: &MultilinearPoint<EF>,
    ) -> EF
    where
        EF: TwoAdicField,
    {
        // Remaining variable count for the round we are evaluating.
        let mut vars_left = self.num_variables;

        let mut acc = EF::ZERO;

        for (round_idx, (alpha_pows, round_statement)) in constraints.iter().enumerate() {
            // Construct the point slice appropriate for this round.
            //
            // For round 0 we use the full `point`:
            //   - standard case: this is the full n-dimensional r
            //   - skip case:     this is r_rest || r_skip (special shape)
            // For round > 0 we shrink by the previous round's folding factor and
            // take a prefix of length `vars_left`.
            let point_for_round = if round_idx > 0 {
                vars_left -= self.folding_factor.at_round(round_idx - 1);
                point.get_subpoint_over_range(0..vars_left)
            } else {
                point.clone()
            };

            // Check if this is the first round and if the univariate skip optimization is active.
            let is_skip_round = round_idx == 0
                && self.univariate_skip
                && self.folding_factor.at_round(0) >= K_SKIP_SUMCHECK;

            let round_sum: EF = round_statement
                .points
                .iter()
                .zip(alpha_pows)
                .map(|(point, &alpha_i)| {
                    // Each constraint contributes either a deferred evaluation, a skip-aware
                    // evaluation, or a standard evaluation.
                    let val = if is_skip_round {
                        // Skip-aware evaluation over r_rest || r_skip.
                        debug_assert_eq!(point.num_variables(), self.num_variables);
                        point.eq_poly_with_skip(&point_for_round, K_SKIP_SUMCHECK)
                    } else {
                        // Standard multilinear evaluation on the current domain.
                        debug_assert_eq!(point.num_variables(), vars_left);
                        point.eq_poly(&point_for_round)
                    };

                    // Multiply by its random combination coefficient.
                    val * alpha_i
                })
                .sum();

            // Add this round's total contribution to the final value.
            acc += round_sum;
        }

        acc
    }
}

impl<EF, F, H, C, Challenger> From<WhirConfig<EF, F, H, C, Challenger>> for ConstraintPolyEvaluator
where
    F: Field,
    EF: ExtensionField<F>,
{
    fn from(cfg: WhirConfig<EF, F, H, C, Challenger>) -> Self {
        Self {
            num_variables: cfg.num_variables,
            folding_factor: cfg.folding_factor,
            univariate_skip: cfg.univariate_skip,
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};
    use p3_interpolation::interpolate_subgroup;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use proptest::prelude::*;
    use rand::{Rng, SeedableRng, rngs::SmallRng};

    use super::*;
    use crate::{
        parameters::{FoldingFactor, ProtocolParameters, errors::SecurityAssumption},
        poly::evals::EvaluationsList,
        whir::constraints::statement::Statement,
    };

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;
    type Perm = Poseidon2BabyBear<16>;

    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

    #[test]
    fn test_eval_constraints_poly_non_skip() {
        // -- Test Configuration --
        // We use 20 variables to ensure a non-trivial number of folding rounds.
        let num_vars = 20;
        // A constant folding factor of 5 is used.
        let folding_factor = FoldingFactor::Constant(5);
        // This configuration implies a 3-round folding schedule before the final polynomial:
        // Round 0: 20 vars -> 15 vars
        // Round 1: 15 vars -> 10 vars
        // Round 2: 10 vars ->  5 vars (final polynomial)

        // We will add a varying number of constraints in each round.
        let num_constraints_per_round = &[2, 3, 1];

        // Initialize a deterministic random number generator for reproducibility.
        let mut rng = SmallRng::seed_from_u64(0);

        // -- Cryptographic Primitives & Verifier Config --
        let perm = Perm::new_from_rng_128(&mut rng);
        let merkle_hash = MyHash::new(perm.clone());
        let merkle_compress = MyCompress::new(perm);
        let whir_params = ProtocolParameters {
            folding_factor,
            merkle_hash,
            merkle_compress,
            univariate_skip: false,
            initial_statement: true,
            security_level: 90,
            pow_bits: 0,
            soundness_type: SecurityAssumption::UniqueDecoding,
            starting_log_inv_rate: 1,
            rs_domain_initial_reduction_factor: 1,
        };
        let params =
            WhirConfig::<EF, F, MyHash, MyCompress, MyChallenger>::new(num_vars, whir_params);
        let evaluator: ConstraintPolyEvaluator = params.into();

        // -- Random Constraints and Challenges --
        // This block generates the inputs that the verifier would receive in a real proof.
        let mut statements = vec![];
        let mut alphas: Vec<EF> = vec![];
        let mut num_vars_at_round = num_vars;

        // Generate constraints and alpha challenges for each of the 3 rounds.
        for num_constraints in num_constraints_per_round {
            // Create a statement for the current domain size (20, then 15, then 10).
            let mut statement = Statement::initialize(num_vars_at_round);
            for _ in 0..*num_constraints {
                statement.add_evaluated_constraint(
                    MultilinearPoint::rand(&mut rng, num_vars_at_round),
                    rng.random(),
                );
            }
            statements.push(statement);
            // Generate a random combination scalar (alpha) for this round.
            alphas.push(rng.random());
            // Shrink the number of variables for the next round.
            num_vars_at_round -= folding_factor.at_round(0);
        }

        // Assemble the data in the format that `eval_constraints_poly` expects.
        let round_constraints: Vec<_> = statements
            .iter()
            .cloned()
            .zip(&alphas)
            .map(|(statement, &alpha)| (alpha.powers().collect_n(statement.len()), statement))
            .collect();

        // Generate the final, full 20-dimensional challenge point `r`.
        let final_point = MultilinearPoint::rand(&mut rng, num_vars);

        // Calculate W(r) using the function under test
        //
        // This is the recursive method we want to validate.
        let result_from_eval_poly =
            evaluator.eval_constraints_poly(&round_constraints, &final_point);

        // Manually compute W(r) with explicit recursive evaluation
        let mut expected_result = EF::ZERO;

        // --- Contribution from Round 0 ---
        //
        // Combine the constraints for the first round using its alpha.
        let (w0_combined, _) = statements[0].combine::<F>(alphas[0]);
        // The evaluation point for this round is the full, unsliced 20-variable challenge point.
        let point_round0 = final_point.clone();
        // Add the contribution from this round.
        expected_result += w0_combined.evaluate(&point_round0);

        // --- Contribution from Round 1 ---
        //
        // Combine the constraints for the second round using its alpha.
        let (w1_combined, _) = statements[1].combine::<F>(alphas[1]);
        // The domain has shrunk. The evaluation point is the first 15 variables of the full point.
        let point_round1 = final_point.get_subpoint_over_range(0..15);
        // Add the contribution from this round.
        expected_result += w1_combined.evaluate(&point_round1);

        // --- Contribution from Round 2 ---
        //
        // Combine the constraints for the third round.
        let (w2_combined, _) = statements[2].combine::<F>(alphas[2]);
        // The domain shrinks again. The evaluation point is the first 10 variables of the full point.
        let point_round2 = final_point.get_subpoint_over_range(0..10);
        // Add the contribution from this round.
        expected_result += w2_combined.evaluate(&point_round2);

        // The result from the recursive function must match the materialized ground truth.
        assert_eq!(result_from_eval_poly, expected_result);
    }

    proptest! {
        #[test]
        fn prop_eval_constraints_poly_non_skip(
            (n, folding_factor_val) in (10..=20usize)
                .prop_flat_map(|n| (
                    Just(n),
                    2..=(n / 2)
                ))
        ) {
            // `Tracks the number of variables remaining before each round.
            let mut num_vars_current = n;
            // The folding factor is constant for all rounds.
            let folding_factor = FoldingFactor::Constant(folding_factor_val);
            // Will store the number of variables folded in each specific round.
            let mut folding_factors_vec = vec![];
            // We simulate the folding process to build the schedule.
            //
            // The protocol folds variables until 0 remain.
            while num_vars_current > 0 {
                // In each round, we fold `folding_factor_val` variables.
                //
                // If this would leave fewer than 0 variables, we fold just enough to reach 0.
                let num_to_fold = std::cmp::min(folding_factor_val, num_vars_current);
                // This check avoids an infinite loop if `num_vars_current` gets stuck.
                if num_to_fold == 0 { break; }
                // Record the number of variables folded in this round.
                folding_factors_vec.push(num_to_fold);
                // Decrease the variable count for the next round.
                num_vars_current -= num_to_fold;
            }
            // The total number of folding rounds.
            let num_rounds = folding_factors_vec.len();

            // Use a seeded RNG for a reproducible test run.
            let mut rng = SmallRng::seed_from_u64(0);
            // For each round, generate a random number of constraints (from 0 to 8).
            let num_constraints_per_round: Vec<usize> = (0..num_rounds)
                .map(|_| rng.random_range(0..=8))
                .collect();

            // -- Cryptographic Primitives & Verifier Config --
            // Set up the necessary cryptographic components for the `WhirConfig`.
            let perm = Perm::new_from_rng_128(&mut rng);
            let merkle_hash = MyHash::new(perm.clone());
            let merkle_compress = MyCompress::new(perm);

            // Define the top-level parameters for the protocol.
            let whir_params = ProtocolParameters {
                folding_factor,
                merkle_hash,
                merkle_compress,
                // This test is for the standard, non-skip case.
                univariate_skip: false,
                initial_statement: true,
                security_level: 90,
                pow_bits: 0,
                soundness_type: SecurityAssumption::UniqueDecoding,
                starting_log_inv_rate: 1,
                rs_domain_initial_reduction_factor: 1,
            };
            // Create the complete verifier configuration object.
            let params =
                WhirConfig::<EF, F, MyHash, MyCompress, MyChallenger>::new(n, whir_params);
            let evaluator: ConstraintPolyEvaluator = params.into();

            // -- Random Constraints and Challenges --
            // This block generates the inputs that the verifier would receive in a real proof.
            let mut statements = vec![];
            let mut alphas: Vec<EF> = vec![];
            num_vars_current = n;

            // Generate the statements and alpha challenges for each round based on our dynamic schedule.
            for i in 0..num_rounds {
                // Create a statement for the current domain size (e.g., 20, then 15, then 10...).
                let mut statement = Statement::initialize(num_vars_current);
                // Add the random number of constraints for this round.
                for _ in 0..num_constraints_per_round[i] {
                    statement.add_evaluated_constraint(
                        MultilinearPoint::rand(&mut rng, num_vars_current),
                        rng.random(),
                    );
                }
                statements.push(statement);
                // Generate a random combination scalar (alpha) for this round.
                alphas.push(rng.random());
                // Shrink the number of variables for the next round.
                num_vars_current -= folding_factors_vec[i];
            }

            // Assemble the final data structure in the format required by `eval_constraints_poly`.
            let round_constraints: Vec<_> = statements
                .iter()
                .cloned()
                .zip(&alphas)
                .map(|(s, &a)| (a.powers().collect_n(s.len()), s))
                .collect();

            // Generate the final, full n-dimensional challenge point `r`.
            let final_point = MultilinearPoint::rand(&mut rng, n);


            // Calculate W(r) using the function under test
            //
            // This is the recursive method we want to validate.
            let result_from_eval_poly =
                evaluator.eval_constraints_poly(&round_constraints, &final_point);


            // Calculate W(r) by materializing and evaluating round-by-round
            //
            // This simpler, more direct method serves as our ground truth.
            let mut expected_result = EF::ZERO;
            let mut num_vars_for_round = n;

            // Loop through each round to calculate its contribution to the final evaluation.
            for i in 0..num_rounds {
                // Combine this round's constraints into a single polynomial `W_i(X)`.
                let (w_combined, _) = statements[i].combine::<F>(alphas[i]);

                // Create the challenge point for this round by taking a prefix slice of the full point `r`.
                let point_for_round = final_point.get_subpoint_over_range(0..num_vars_for_round);
                // Evaluate `W_i` at the correctly sliced point.
                let w_eval = w_combined.evaluate(&point_for_round);

                // Add this round's contribution to the total.
                expected_result += w_eval;

                // Shrink the number of variables for the next round's slice.
                num_vars_for_round -= folding_factors_vec[i];
            }

            // The result from the recursive function must match the materialized ground truth.
            prop_assert_eq!(result_from_eval_poly, expected_result);
        }
    }

    #[test]
    fn test_eval_constraints_poly_with_skip() {
        // -- Test Configuration --
        //
        // We use 20 variables to ensure a non-trivial number of folding rounds.
        let num_vars = 20;

        // We use a constant folding factor of `K_SKIP_SUMCHECK` to trigger the skip.
        let folding_factor = FoldingFactor::Constant(K_SKIP_SUMCHECK);

        // This configuration implies a folding schedule:
        // Round 0: 20 vars --(skip 5)--> 15 vars
        // Round 1: 15 vars --(fold 5)--> 10 vars
        // Round 2: 11 vars --(fold 5)-->  5 vars
        // Round 3:  7 vars --(fold 5)-->  0 vars (final polynomial)
        let num_constraints_per_round = &[2, 3, 1, 2];
        let num_rounds = num_constraints_per_round.len();

        // Initialize a deterministic RNG for reproducibility.
        let mut rng = SmallRng::seed_from_u64(0);

        // -- Cryptographic Primitives & Verifier Config --
        let perm = Perm::new_from_rng_128(&mut rng);
        let merkle_hash = MyHash::new(perm.clone());
        let merkle_compress = MyCompress::new(perm);
        let whir_params = ProtocolParameters {
            folding_factor,
            merkle_hash,
            merkle_compress,
            // This test is for the skip case.
            univariate_skip: true,
            initial_statement: true,
            security_level: 90,
            pow_bits: 0,
            soundness_type: SecurityAssumption::UniqueDecoding,
            starting_log_inv_rate: 1,
            rs_domain_initial_reduction_factor: 1,
        };
        let params =
            WhirConfig::<EF, F, MyHash, MyCompress, MyChallenger>::new(num_vars, whir_params);
        let evaluator: ConstraintPolyEvaluator = params.into();

        // -- Random Constraints and Challenges --
        let mut statements = vec![];
        let mut alphas: Vec<EF> = vec![];
        let mut num_vars_at_round = num_vars;

        // Generate constraints and alpha challenges for each round.
        for (i, &num_constraints) in num_constraints_per_round
            .iter()
            .enumerate()
            .take(num_rounds)
        {
            let mut statement = Statement::initialize(num_vars_at_round);
            for _ in 0..num_constraints {
                statement.add_evaluated_constraint(
                    MultilinearPoint::rand(&mut rng, num_vars_at_round),
                    rng.random(),
                );
            }
            statements.push(statement);
            alphas.push(rng.random());
            num_vars_at_round -= folding_factor.at_round(i);
        }

        // Assemble the data in the format that `eval_constraints_poly` expects.
        let round_constraints: Vec<_> = statements
            .iter()
            .cloned()
            .zip(&alphas)
            .map(|(s, &a)| (a.powers().collect_n(s.len()), s))
            .collect();

        // For a skip protocol, the verifier's final challenge object has a special
        // structure with (n - k_skip) + 1 elements, not n.
        let final_point = MultilinearPoint::<EF>::rand(&mut rng, (num_vars - K_SKIP_SUMCHECK) + 1);

        // Calculate W(r) using the function under test
        let result_from_eval_poly =
            evaluator.eval_constraints_poly(&round_constraints, &final_point);

        // Manually compute W(r) with explicit recursive evaluation
        let mut expected_result = EF::ZERO;

        // --- Contribution from Round 0 (Skip Round) ---
        //
        // Combine the constraints for the first round into a single polynomial, W_0(X).
        let (w0_combined, _) = statements[0].combine::<F>(alphas[0]);

        // To evaluate W_0(r) using skip semantics, we follow the same pipeline as the prover:
        // a) Deconstruct the special challenge object `r` into its components:
        // - `r_rest`,
        // - `r_skip`.
        let num_remaining = num_vars - K_SKIP_SUMCHECK;
        let r_rest = final_point.get_subpoint_over_range(0..num_remaining);
        let r_skip = *final_point
            .last_variable()
            .expect("skip challenge must be present");

        // b) Reshape the W_0(X) evaluation table into a matrix.
        let w0_mat = w0_combined.into_mat(1 << num_remaining);

        // c) "Fold" the skipped variables by interpolating the matrix at `r_skip`.
        let folded_row = interpolate_subgroup(&w0_mat, r_skip);

        // d) Evaluate the resulting smaller polynomial at the remaining challenges `r_rest`.
        let w0_eval = EvaluationsList::new(folded_row).evaluate(&r_rest);
        expected_result += w0_eval;

        // --- Contribution from Round 1 (Standard Round) ---
        let (w1_combined, _) = statements[1].combine::<F>(alphas[1]);
        // For subsequent rounds, the evaluation point is a prefix slice of the `r_rest` challenges.
        let point_round1 = r_rest.get_subpoint_over_range(0..statements[1].num_variables());
        expected_result += w1_combined.evaluate(&point_round1);

        // --- Contribution from Round 2 (Standard Round) ---
        let (w2_combined, _) = statements[2].combine::<F>(alphas[2]);
        let point_round2 = r_rest.get_subpoint_over_range(0..statements[2].num_variables());
        expected_result += w2_combined.evaluate(&point_round2);

        // --- Contribution from Round 3 (Standard Round) ---
        let (w3_combined, _) = statements[3].combine::<F>(alphas[3]);
        let point_round3 = r_rest.get_subpoint_over_range(0..statements[3].num_variables());
        expected_result += w3_combined.evaluate(&point_round3);

        // The result from the recursive function must match the materialized ground truth.
        assert_eq!(result_from_eval_poly, expected_result);
    }

    proptest! {
        #[test]
        fn prop_eval_constraints_poly_with_skip(
            (n, standard_folding_factor) in (10..=20usize)
                .prop_flat_map(|n| (
                    Just(n),
                    2..=((n - K_SKIP_SUMCHECK) / 2).max(2)
                ))
        ) {
            // Tracks the number of variables remaining before each round.
            let mut num_vars_current = n;
            // - The first round folds K_SKIP_SUMCHECK variables,
            // - Subsequent rounds use the random factor.
            let folding_factor = FoldingFactor::ConstantFromSecondRound(K_SKIP_SUMCHECK, standard_folding_factor);
            // Will store the number of variables folded in each specific round.
            let mut folding_factors_vec = vec![];

            // We simulate the folding process to build the schedule.
            while num_vars_current > 0 {
                let num_to_fold = folding_factor.at_round(folding_factors_vec.len());
                // Ensure we don't overshoot the target of 0 remaining variables.
                let effective_num_to_fold = std::cmp::min(num_to_fold, num_vars_current);
                if effective_num_to_fold == 0 { break; }
                folding_factors_vec.push(effective_num_to_fold);
                num_vars_current -= effective_num_to_fold;
            }
            let num_rounds = folding_factors_vec.len();

            // Use a seeded RNG for a reproducible test run.
            let mut rng = SmallRng::seed_from_u64(0);
            // For each round, generate a random number of constraints (from 0 to 2).
            let num_constraints_per_round: Vec<usize> = (0..num_rounds)
                .map(|_| rng.random_range(0..=2))
                .collect();

            // -- Cryptographic Primitives & Verifier Config --
            let perm = Perm::new_from_rng_128(&mut rng);
            let merkle_hash = MyHash::new(perm.clone());
            let merkle_compress = MyCompress::new(perm);
            let whir_params = ProtocolParameters {
                folding_factor,
                merkle_hash,
                merkle_compress,
                // This test is for the skip case.
                univariate_skip: true,
                initial_statement: true,
                security_level: 90,
                pow_bits: 0,
                soundness_type: SecurityAssumption::UniqueDecoding,
                starting_log_inv_rate: 1,
                rs_domain_initial_reduction_factor: 1,
            };
            let params =
                WhirConfig::<EF, F, MyHash, MyCompress, MyChallenger>::new(n, whir_params);
            let evaluator: ConstraintPolyEvaluator = params.into();

            // -- Random Constraints and Challenges --
            let mut statements = vec![];
            let mut alphas: Vec<EF> = vec![];
            num_vars_current = n;

            // Generate the statements and alphas for each round based on our dynamic schedule.
            for i in 0..num_rounds {
                let mut statement = Statement::initialize(num_vars_current);
                for _ in 0..num_constraints_per_round[i] {
                    statement.add_evaluated_constraint(
                        MultilinearPoint::rand(&mut rng, num_vars_current),
                        rng.random(),
                    );
                }
                statements.push(statement);
                alphas.push(rng.random());
                num_vars_current -= folding_factors_vec[i];
            }

            // Assemble the data for the function call.
            let round_constraints: Vec<_> = statements
                .iter()
                .cloned()
                .zip(&alphas)
                .map(|(s, &a)| (a.powers().collect_n(s.len()), s))
                .collect();

            // For a skip protocol, the verifier's final challenge object has a special
            // structure with `(n - k_skip) + 1` elements, not `n`.
            let final_point =
                MultilinearPoint::rand(&mut rng, (n - K_SKIP_SUMCHECK) + 1);


            // Calculate W(r) using the function under test
            let result_from_eval_poly =
                evaluator.eval_constraints_poly(&round_constraints, &final_point);


            // Calculate W(r) by materializing and evaluating round-by-round
            let mut expected_result = EF::ZERO;
            let mut num_vars_for_round = n;

            // Contribution from Round 0 (Skip Round)
            //
            // Combine the constraints for the first round using its alpha.
            let (w0_combined, _) = statements[0].combine::<F>(alphas[0]);
            // Evaluate W_0(r) using the manual skip evaluation pipeline.
            let num_remaining = n - K_SKIP_SUMCHECK;
            let r_rest = final_point.get_subpoint_over_range(0..num_remaining);
            let r_skip = *final_point.last_variable().expect("skip challenge must be present");

            let w0_mat = w0_combined.into_mat(1 << num_remaining);
            let folded_row = interpolate_subgroup(&w0_mat, r_skip);
            let w0_eval = EvaluationsList::new(folded_row).evaluate(&r_rest);
            expected_result += w0_eval;
            num_vars_for_round -= folding_factors_vec[0];

            // Contribution from Subsequent Rounds (Standard)
            //
            // Loop through the remaining rounds.
            for i in 1..num_rounds {
                // Combine this round's constraints into a single polynomial `W_i(X)`.
                let (w_combined, _) = statements[i].combine::<F>(alphas[i]);

                // Evaluate `W_i` at the correct prefix slice of the `r_rest` challenges.
                let point_for_round = r_rest.get_subpoint_over_range(0..num_vars_for_round);
                let w_eval = w_combined.evaluate(&point_for_round);

                // Add this round's contribution to the total.
                expected_result += w_eval;

                // Shrink the number of variables for the next round's slice.
                num_vars_for_round -= folding_factors_vec[i];
            }

            // The result from the recursive function must match the materialized ground truth.
            prop_assert_eq!(result_from_eval_poly, expected_result);
        }
    }
}
