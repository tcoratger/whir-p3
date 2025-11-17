use alloc::{vec, vec::Vec};

use p3_field::{ExtensionField, Field, TwoAdicField};

use crate::{
    constant::K_SKIP_SUMCHECK, parameters::FoldingFactor, poly::multilinear::MultilinearPoint,
    whir::constraints::Constraint,
};

/// Evaluate a single round's constraint.
fn eval_round<F: Field, EF: ExtensionField<F> + TwoAdicField>(
    round: usize,
    constraint: &Constraint<F, EF>,
    original_point: &MultilinearPoint<EF>,
    context: &PointContext<EF>,
) -> EF {
    let (eval_point, use_skip_eval) = match (round, context) {
        // Round 0 with skip: use full rotated point with skip evaluation
        (0, PointContext::Skip { rotated, .. }) => (rotated.clone(), true),
        // Round 0 without skip: reverse full point
        (0, PointContext::NonSkip) => (original_point.reversed(), false),
        // Round >0 with skip: slice from this round's offset to end
        (
            i,
            PointContext::Skip {
                rotated,
                prover_challenge_offsets,
            },
        ) => {
            let start = if i == 1 {
                0
            } else {
                prover_challenge_offsets[i - 1]
            };
            let challenges = rotated.get_subpoint_over_range(0..rotated.num_variables() - 1);
            (
                challenges.get_subpoint_over_range(start..challenges.num_variables()),
                false,
            )
        }
        // Round >0 without skip: take first num_vars and reverse
        (_, PointContext::NonSkip) => {
            let slice = original_point.get_subpoint_over_range(0..constraint.num_variables());
            (slice.reversed(), false)
        }
    };

    // Evaluate eq and sel constraints at the computed point
    let eq_contribution = constraint
        .iter_eqs()
        .map(|(pt, coeff)| {
            let val = if use_skip_eval {
                pt.eq_poly_with_skip(&eval_point, K_SKIP_SUMCHECK)
            } else {
                pt.eq_poly(&eval_point)
            };
            val * coeff
        })
        .sum::<EF>();

    let sel_contribution = constraint
        .iter_sels()
        .map(|(&var, coeff)| {
            let expanded =
                MultilinearPoint::expand_from_univariate(var, constraint.num_variables());
            coeff * expanded.select_poly(&eval_point)
        })
        .sum::<EF>();

    eq_contribution + sel_contribution
}

/// Lightweight evaluator for the combined constraint polynomial W(r).
#[derive(Clone, Debug)]
pub struct ConstraintPolyEvaluator {
    /// Number of variables in the multilinear polynomial space.
    pub num_variables: usize,
    /// The folding factor.
    pub folding_factor: FoldingFactor,
    /// Optional skip step indicating whether the univariate skip optimization is active.
    pub univariate_skip: Option<usize>,
}

impl ConstraintPolyEvaluator {
    /// Creates a new `ConstraintPolyEvaluator` with the given parameters.
    #[must_use]
    pub const fn new(
        num_variables: usize,
        folding_factor: FoldingFactor,
        univariate_skip: Option<usize>,
    ) -> Self {
        Self {
            num_variables,
            folding_factor,
            univariate_skip,
        }
    }

    /// Evaluate the combined constraint polynomial W(r).
    ///
    /// ## Input Structure
    /// - `constraints[i]`: constraint created after prover round i-1
    /// - `point`: all folding randomness (prover rounds + final sumcheck)
    ///   - Non-skip: [r_0, r_1, ..., r_final] in forward order
    ///   - Skip: [r_skip, r_0, r_1, ..., r_final] where r_0, r_1 are from prover rounds
    ///
    /// ## Key Insight
    /// Constraint i needs evaluation point matching its polynomial's remaining variables.
    /// This means using challenges from prover round i onwards + final sumcheck.
    #[must_use]
    pub fn eval_constraints_poly<F: Field, EF: ExtensionField<F> + TwoAdicField>(
        &self,
        constraints: &[Constraint<F, EF>],
        point: &MultilinearPoint<EF>,
    ) -> EF {
        let using_skip = self.univariate_skip.is_some();

        // Prepare point structure based on skip/non-skip case
        let context = if using_skip {
            self.prepare_skip_context(point, constraints.len() - 1)
        } else {
            PointContext::NonSkip
        };

        constraints
            .iter()
            .enumerate()
            .map(|(i, constraint)| eval_round(i, constraint, point, &context))
            .sum()
    }

    /// Prepare skip-specific context: rotate point and compute challenge offsets.
    fn prepare_skip_context<EF: ExtensionField<impl Field> + TwoAdicField>(
        &self,
        point: &MultilinearPoint<EF>,
        num_prover_rounds: usize,
    ) -> PointContext<EF> {
        // Rotate [r_skip, rest...] -> [rest..., r_skip] for easier slicing
        let mut rotated = point.as_slice()[1..].to_vec();
        rotated.push(point.as_slice()[0]);
        let rotated = MultilinearPoint::new(rotated);

        // Compute where each prover round's challenges start in the rotated point
        let mut offsets = vec![0];
        for round in 0..num_prover_rounds {
            offsets.push(offsets[round] + self.folding_factor.at_round(round + 1));
        }

        PointContext::Skip {
            rotated,
            prover_challenge_offsets: offsets,
        }
    }
}

/// Context for point slicing across constraint evaluations.
enum PointContext<EF> {
    /// Non-skip case: use original point with reversal.
    NonSkip,
    /// Skip case: precomputed rotated point and prover round challenge offsets.
    Skip {
        rotated: MultilinearPoint<EF>,
        prover_challenge_offsets: Vec<usize>,
    },
}

#[cfg(test)]
mod tests {
    use alloc::{vec, vec::Vec};

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};
    use p3_interpolation::interpolate_subgroup;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use proptest::prelude::*;
    use rand::{Rng, SeedableRng, rngs::SmallRng};

    use super::*;
    use crate::{
        parameters::FoldingFactor,
        poly::evals::EvaluationsList,
        whir::constraints::statement::{EqStatement, SelectStatement},
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
        let num_eq_constraints_per_round = &[2usize, 3, 1];
        let num_sel_constraints_per_round = &[31usize, 41, 51];

        // Initialize a deterministic random number generator for reproducibility.
        let mut rng = SmallRng::seed_from_u64(0);

        // -- Random Constraints and Challenges --
        // This block generates the inputs that the verifier would receive in a real proof.
        let mut num_vars_at_round = num_vars;
        let mut constraints = vec![];

        // Generate eq and select constraints and challenges for each of the 3 rounds.
        for (round_idx, (&num_eq, &num_sel)) in num_eq_constraints_per_round
            .iter()
            .zip(num_sel_constraints_per_round.iter())
            .enumerate()
        {
            // Generate a random combination challenge for this round.
            let gamma = rng.random();
            // Create eq statement for the current domain size (20, then 15, then 10).
            let mut eq_statement = EqStatement::initialize(num_vars_at_round);
            (0..num_eq).for_each(|_| {
                eq_statement.add_evaluated_constraint(
                    MultilinearPoint::rand(&mut rng, num_vars_at_round),
                    rng.random(),
                );
            });

            // Create select statement for the current domain size (20, then 15, then 10).
            let mut sel_statement = SelectStatement::<F, EF>::initialize(num_vars_at_round);
            (0..num_sel).for_each(|_| sel_statement.add_constraint(rng.random(), rng.random()));
            constraints.push(Constraint::new(gamma, eq_statement, sel_statement));

            // Shrink the number of variables for the next round.
            num_vars_at_round -= folding_factor.at_round(round_idx);
        }

        // Generate the final, full 20-dimensional challenge point `r`.
        let final_point = MultilinearPoint::rand(&mut rng, num_vars);

        // Calculate W(r) using the function under test
        let evaluator = ConstraintPolyEvaluator::new(num_vars, folding_factor, None);
        let result_from_eval_poly = evaluator.eval_constraints_poly(&constraints, &final_point);

        // Calculate W(r) by materializing and evaluating round-by-round
        // This simpler, more direct method serves as our ground truth.
        // Loop through each round to calculate its contribution to the final evaluation.
        let expected_result = constraints
            .iter()
            .map(|constraint| {
                let num_vars = constraint.num_variables();
                let mut combined = EvaluationsList::zero(num_vars);
                let mut eval = EF::ZERO;
                constraint.combine(&mut combined, &mut eval);
                let point = final_point.get_subpoint_over_range(0..num_vars).reversed();
                combined.evaluate_hypercube(&point)
            })
            .sum::<EF>();

        // The result from the recursive function must match the materialized ground truth.
        assert_eq!(result_from_eval_poly, expected_result);
    }

    proptest! {
        #[test]
        fn prop_eval_constraints_poly_non_skip(
            (num_vars, folding_factor_val) in (10..=20usize)
                .prop_flat_map(|n| (
                    Just(n),
                    2..=(n / 2)
                ))
        ) {
            // `Tracks the number of variables remaining before each round.
            let mut num_vars_current = num_vars;
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
                let num_to_fold = core::cmp::min(folding_factor_val, num_vars_current);
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
            let num_eq_constraints_per_round: Vec<usize> = (0..num_rounds)
                .map(|_| rng.random_range(0..=2))
                .collect();
            let num_sel_constraints_per_round: Vec<usize> = (0..num_rounds)
                .map(|_| rng.random_range(0..=2))
                .collect();

            // -- Random Constraints and Challenges --
            // This block generates the inputs that the verifier would receive in a real proof.
            let mut num_vars_current = num_vars;
            let mut constraints = vec![];

            // Generate eq and select constraints and alpha challenges for each of the 3 rounds.
            for (round_idx, (&num_eq, &num_sel)) in num_eq_constraints_per_round
                .iter()
                .zip(num_sel_constraints_per_round.iter())
                .enumerate()
            {
                // Generate a random combination scalar (alpha) for this round.
                let gamma = rng.random();
                // Create eq statement for the current domain size (20, then 15, then 10).
                let mut eq_statement = EqStatement::initialize(num_vars_current);
                (0..num_eq).for_each(|_| {
                    eq_statement.add_evaluated_constraint(
                        MultilinearPoint::rand(&mut rng, num_vars_current),
                        rng.random(),
                    );
                });

                // Create select statement for the current domain size (20, then 15, then 10).
                let mut sel_statement = SelectStatement::<F, EF>::initialize(num_vars_current);
                (0..num_sel).for_each(|_| sel_statement.add_constraint(rng.random(), rng.random()));
                constraints.push(Constraint::new(gamma, eq_statement, sel_statement));

                // Shrink the number of variables for the next round.
                num_vars_current -= folding_factors_vec[round_idx];
            }

            // Generate the final, full n-dimensional challenge point `r`.
            let final_point = MultilinearPoint::rand(&mut rng, num_vars);


            // Calculate W(r) using the function under test
            //
            // This is the recursive method we want to validate.
            let evaluator = ConstraintPolyEvaluator::new(num_vars, folding_factor, None);
            let result_from_eval_poly =
                evaluator.eval_constraints_poly(&constraints, &final_point);

            // Calculate W(r) by materializing and evaluating round-by-round
            //
            // This simpler, more direct method serves as our ground truth.
            let mut num_vars_at_round = num_vars;
            // Loop through each round to calculate its contribution to the final evaluation.
            let expected_result = constraints
                .iter()
                .enumerate()
                .map(|(round_idx, constraint)| {
                    let point = final_point.get_subpoint_over_range(0..num_vars_at_round).reversed();
                    let mut combined = EvaluationsList::zero(constraint.num_variables());
                    let mut eval = EF::ZERO;
                    constraint.combine(&mut combined, &mut eval);
                    num_vars_at_round -= folding_factors_vec[round_idx];
                    combined.evaluate_hypercube(&point)
                })
                .sum::<EF>();

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
        let num_eq_constraints_per_round = &[2usize, 3, 1, 2];
        let num_sel_constraints_per_round = &[0, 21, 31, 41];

        // Initialize a deterministic RNG for reproducibility.
        let mut rng = SmallRng::seed_from_u64(0);

        // -- Random Constraints and Challenges --
        let mut num_vars_at_round = num_vars;
        let mut constraints = vec![];

        // Generate eq and select constraints and alpha challenges for each rounds.
        for (round_idx, (&num_eq, &num_sel)) in num_eq_constraints_per_round
            .iter()
            .zip(num_sel_constraints_per_round.iter())
            .enumerate()
        {
            // Create eq statement for the current domain size (20, then 15, then 10).
            let mut eq_statement = EqStatement::initialize(num_vars_at_round);
            (0..num_eq).for_each(|_| {
                eq_statement.add_evaluated_constraint(
                    MultilinearPoint::rand(&mut rng, num_vars_at_round),
                    rng.random(),
                );
            });

            // Create select statement for the current domain size (20, then 15, then 10).
            let mut sel_statement = SelectStatement::<F, EF>::initialize(num_vars_at_round);
            (0..num_sel).for_each(|_| sel_statement.add_constraint(rng.random(), rng.random()));
            constraints.push(Constraint::new(rng.random(), eq_statement, sel_statement));

            // Shrink the number of variables for the next round.
            num_vars_at_round -= folding_factor.at_round(round_idx);
        }

        // For a skip protocol, the verifier's final challenge object has a special
        // structure with (n - k_skip) + 1 elements, not n.
        let final_point = MultilinearPoint::<EF>::rand(&mut rng, (num_vars - K_SKIP_SUMCHECK) + 1);

        // Calculate W(r) using the function under test
        let evaluator =
            ConstraintPolyEvaluator::new(num_vars, folding_factor, Some(K_SKIP_SUMCHECK));
        let result_from_eval_poly = evaluator.eval_constraints_poly(&constraints, &final_point);

        // Manually compute W(r) with explicit recursive evaluation
        let mut expected_result = EF::ZERO;

        // --- Contribution from Round 0 (Skip Round) ---
        //
        // Combine the constraints for the first round into a single polynomial, W_0(X).
        let mut w0_combined = EvaluationsList::zero(constraints[0].eq_statement.num_variables());
        let mut sum = EF::ZERO;
        constraints[0].eq_statement.combine_hypercube::<F, false>(
            &mut w0_combined,
            &mut sum,
            constraints[0].challenge,
        );

        // To evaluate W_0(r) using skip semantics, we follow the same pipeline as the prover:
        // a) Deconstruct the special challenge object `r` into its components:
        // - `r_rest`,
        // - `r_skip`.
        let num_remaining = num_vars - K_SKIP_SUMCHECK;

        let final_point = final_point.reversed();
        let r_rest = final_point.get_subpoint_over_range(0..num_remaining);
        let r_skip = *final_point
            .last_variable()
            .expect("skip challenge must be present");

        // b) Reshape the W_0(X) evaluation table into a matrix.
        let w0_mat = w0_combined.into_mat(1 << num_remaining);

        // c) "Fold" the skipped variables by interpolating the matrix at `r_skip`.
        let folded_row = interpolate_subgroup(&w0_mat, r_skip);

        // d) Evaluate the resulting smaller polynomial at the remaining challenges `r_rest`.
        let w0_eval = EvaluationsList::new(folded_row).evaluate_hypercube(&r_rest.reversed());
        expected_result += w0_eval;

        expected_result += constraints
            .iter()
            .skip(1)
            .map(|constraint| {
                let mut combined = EvaluationsList::zero(constraint.num_variables());
                let mut eval = EF::ZERO;
                constraint.combine(&mut combined, &mut eval);
                let point = r_rest.get_subpoint_over_range(0..constraint.num_variables());
                combined.evaluate_hypercube(&point.reversed())
            })
            .sum::<EF>();

        // The result from the recursive function must match the materialized ground truth.
        assert_eq!(result_from_eval_poly, expected_result);
    }

    proptest! {
        #[test]
        fn prop_eval_constraints_poly_with_skip(
            (num_vars, standard_folding_factor) in (10..=20usize)
                .prop_flat_map(|n| (
                    Just(n),
                    2..=((n - K_SKIP_SUMCHECK) / 2).max(2)
                ))
        ) {
            // Tracks the number of variables remaining before each round.
            let mut num_vars_current = num_vars;
            // - The first round folds K_SKIP_SUMCHECK variables,
            // - Subsequent rounds use the random factor.
            let folding_factor = FoldingFactor::ConstantFromSecondRound(K_SKIP_SUMCHECK, standard_folding_factor);
            // Will store the number of variables folded in each specific round.
            let mut folding_factors_vec = vec![];

            // We simulate the folding process to build the schedule.
            while num_vars_current > 0 {
                let num_to_fold = folding_factor.at_round(folding_factors_vec.len());
                // Ensure we don't overshoot the target of 0 remaining variables.
                let effective_num_to_fold = core::cmp::min(num_to_fold, num_vars_current);
                if effective_num_to_fold == 0 { break; }
                folding_factors_vec.push(effective_num_to_fold);
                num_vars_current -= effective_num_to_fold;
            }
            let num_rounds = folding_factors_vec.len();

            // Use a seeded RNG for a reproducible test run.
            let mut rng = SmallRng::seed_from_u64(0);
            // For each round, generate a random number of constraints (from 0 to 2).
            let num_eq_constraints_per_round: Vec<usize> = (0..num_rounds)
                .map(|_| rng.random_range(0..=2))
                .collect();
            let num_sel_constraints_per_round: Vec<usize> = (0..num_rounds)
                .map(|i| {
                    if i == 0 {
                        0
                    } else {
                        rng.random_range(0..=2)
                    }
                }).collect();

            // -- Cryptographic Primitives & Verifier Config --
            let evaluator =
                ConstraintPolyEvaluator::new(num_vars, folding_factor, Some(K_SKIP_SUMCHECK));

            // -- Random Constraints and Challenges --
            let mut num_vars_current = num_vars;
            let mut constraints = vec![];

            // Generate eq and select constraints and alpha challenges for each rounds.
            for (round_idx, (&num_eq, &num_sel)) in num_eq_constraints_per_round
                .iter()
                .zip(num_sel_constraints_per_round.iter())
                .enumerate()
            {
                // Create eq statement for the current domain size (20, then 15, then 10).
                let mut eq_statement = EqStatement::initialize(num_vars_current);
                (0..num_eq).for_each(|_| {
                    eq_statement.add_evaluated_constraint(
                        MultilinearPoint::rand(&mut rng, num_vars_current),
                        rng.random(),
                    );
                });

                // Create select statement for the current domain size (20, then 15, then 10).
                let mut sel_statement = SelectStatement::<F, EF>::initialize(num_vars_current);
                (0..num_sel).for_each(|_| sel_statement.add_constraint(rng.random(), rng.random()));
                constraints.push(Constraint::new(rng.random(), eq_statement, sel_statement));

                // Shrink the number of variables for the next round.
                num_vars_current -= folding_factors_vec[round_idx];
            }

            // For a skip protocol, the verifier's final challenge object has a special
            // structure with `(n - k_skip) + 1` elements, not `n`.
            let final_point =
                MultilinearPoint::rand(&mut rng, (num_vars - K_SKIP_SUMCHECK) + 1);


            // Calculate W(r) using the function under test
            let result_from_eval_poly =
                evaluator.eval_constraints_poly(&constraints, &final_point);


            // Calculate W(r) by materializing and evaluating round-by-round
            let mut expected_result = EF::ZERO;

            // Contribution from Round 0 (Skip Round)
            //
            // Combine the constraints for the first round into a single polynomial, W_0(X).
            let mut w0_combined = EvaluationsList::zero(constraints[0].eq_statement.num_variables());
            let mut sum = EF::ZERO;
            constraints[0]
                .eq_statement
                .combine_hypercube::<F, false>(&mut w0_combined, &mut sum, constraints[0].challenge);


            let num_remaining = num_vars - K_SKIP_SUMCHECK;
            let final_point = final_point.reversed();
            let r_rest = final_point.get_subpoint_over_range(0..num_remaining);
            let r_skip = *final_point
                .last_variable()
                .expect("skip challenge must be present");

            // b) Reshape the W_0(X) evaluation table into a matrix.
            let w0_mat = w0_combined.into_mat(1 << num_remaining);

            // c) "Fold" the skipped variables by interpolating the matrix at `r_skip`.
            let folded_row = interpolate_subgroup(&w0_mat, r_skip);

            // d) Evaluate the resulting smaller polynomial at the remaining challenges `r_rest`.
            let w0_eval = EvaluationsList::new(folded_row).evaluate_hypercube(&r_rest.reversed());
            expected_result += w0_eval;

            expected_result += constraints
                .iter()
                .skip(1)
                .map(|constraint| {
                    let mut combined = EvaluationsList::zero(constraint.num_variables());
                    let mut eval = EF::ZERO;
                    constraint.combine(&mut combined, &mut eval);
                    let point = r_rest.get_subpoint_over_range(0..constraint.num_variables());
                    combined.evaluate_hypercube(&point.reversed())
                })
                .sum::<EF>();

            // The result from the recursive function must match the materialized ground truth.
            prop_assert_eq!(result_from_eval_poly, expected_result);
        }
    }
}
