use p3_field::{ExtensionField, Field, TwoAdicField};

use crate::{
    constant::K_SKIP_SUMCHECK,
    parameters::FoldingFactor,
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    whir::{
        constraints::statement::{EqStatement, SelectStatement},
        parameters::WhirConfig,
    },
};

#[derive(Clone, Debug)]
pub struct Constraint<F: Field, EF: ExtensionField<F>> {
    pub eq_statement: EqStatement<EF>,
    pub sel_statement: SelectStatement<F, EF>,
    pub challenge: EF,
}

impl<F: Field, EF: ExtensionField<F>> Constraint<F, EF> {
    pub fn new(
        challenge: EF,
        eq_statement: EqStatement<EF>,
        sel_statement: SelectStatement<F, EF>,
    ) -> Self {
        assert_eq!(eq_statement.num_variables(), sel_statement.num_variables());
        Self {
            eq_statement,
            sel_statement,
            challenge,
        }
    }

    pub fn new_eq_only(challenge: EF, eq_statement: EqStatement<EF>) -> Self {
        let num_variables = eq_statement.num_variables();
        Self::new(
            challenge,
            eq_statement,
            SelectStatement::initialize(num_variables),
        )
    }

    pub const fn num_variables(&self) -> usize {
        self.eq_statement.num_variables()
    }

    pub fn combine_evals(&self, eval: &mut EF) {
        self.eq_statement.combine_evals(eval, self.challenge);
        self.sel_statement
            .combine_evals(eval, self.challenge, self.eq_statement.len());
    }

    pub fn combine(&self, combined: &mut EvaluationsList<EF>, eval: &mut EF) {
        self.eq_statement
            .combine_hypercube::<F, true>(combined, eval, self.challenge);
        self.sel_statement
            .combine(combined, eval, self.challenge, self.eq_statement.len());
    }

    pub fn combine_new(&self) -> (EvaluationsList<EF>, EF) {
        let mut combined = EvaluationsList::zero(self.num_variables());
        let mut eval = EF::ZERO;
        self.eq_statement
            .combine_hypercube::<F, false>(&mut combined, &mut eval, self.challenge);
        self.sel_statement.combine(
            &mut combined,
            &mut eval,
            self.challenge,
            self.eq_statement.len(),
        );
        (combined, eval)
    }

    pub fn validate_for_skip_case(&self) {
        assert!(
            self.sel_statement.is_empty(),
            "select constraints not supported in skip case"
        );
    }

    pub fn iter_eqs(&self) -> impl Iterator<Item = (&MultilinearPoint<EF>, EF)> {
        self.eq_statement.points.iter().zip(self.challenge.powers())
    }

    pub fn iter_sels(&self) -> impl Iterator<Item = (&F, EF)> {
        self.sel_statement
            .vars
            .iter()
            .zip(self.challenge.powers().skip(self.eq_statement.len()))
    }
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
    /// `constraints` are collection of eq and select statements per round
    /// `point` is the verifier's final challenge:
    /// - standard case: length = n
    /// - skip case:     length = (n - K_SKIP_SUMCHECK) + 1  (r_rest || r_skip)
    #[must_use]
    pub fn eval_constraints_poly<F: Field, EF: ExtensionField<F> + TwoAdicField>(
        &self,
        constraints: &[Constraint<F, EF>],
        point: &MultilinearPoint<EF>,
    ) -> EF {
        // Remaining variable count for the round we are evaluating.
        let mut vars_left = self.num_variables;

        let mut acc = EF::ZERO;
        for (round_idx, constraint) in constraints.iter().enumerate() {
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
                && self.univariate_skip.is_some_and(|skip_step| {
                    constraint.validate_for_skip_case();
                    self.folding_factor.at_round(0) >= skip_step
                });

            acc += constraint
                .iter_eqs()
                .map(|(point, alpha_i)| {
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
                .sum::<EF>();

            acc += constraint
                .iter_sels()
                .map(|(&var, alpha_i)| {
                    let point = MultilinearPoint::expand_from_univariate(var, vars_left);
                    alpha_i * point.select_poly(&point_for_round)
                })
                .sum::<EF>();
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
            univariate_skip: cfg.univariate_skip.then_some(K_SKIP_SUMCHECK),
        }
    }
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
        parameters::{FoldingFactor, ProtocolParameters, errors::SecurityAssumption},
        poly::evals::EvaluationsList,
        whir::constraints::statement::EqStatement,
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
        //
        // This is the recursive method we want to validate.
        let result_from_eval_poly = evaluator.eval_constraints_poly(&constraints, &final_point);

        // Calculate W(r) by materializing and evaluating round-by-round

        // This simpler, more direct method serves as our ground truth.
        let mut num_vars_at_round = num_vars;
        // Loop through each round to calculate its contribution to the final evaluation.
        let expected_result = constraints
            .iter()
            .enumerate()
            .map(|(i, constraint)| {
                let point = final_point.get_subpoint_over_range(0..num_vars_at_round);
                let mut combined = EvaluationsList::zero(constraint.num_variables());
                let mut eval = EF::ZERO;
                constraint.combine(&mut combined, &mut eval);
                num_vars_at_round -= folding_factor.at_round(i);
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
                WhirConfig::<EF, F, MyHash, MyCompress, MyChallenger>::new(num_vars, whir_params);
            let evaluator: ConstraintPolyEvaluator = params.into();

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
                    let point = final_point.get_subpoint_over_range(0..num_vars_at_round);
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
        let r_rest = final_point.get_subpoint_over_range(0..num_remaining);
        let r_skip = *final_point
            .last_variable()
            .expect("skip challenge must be present");

        // b) Reshape the W_0(X) evaluation table into a matrix.
        let w0_mat = w0_combined.into_mat(1 << num_remaining);

        // c) "Fold" the skipped variables by interpolating the matrix at `r_skip`.
        let folded_row = interpolate_subgroup(&w0_mat, r_skip);

        // d) Evaluate the resulting smaller polynomial at the remaining challenges `r_rest`.
        let w0_eval = EvaluationsList::new(folded_row).evaluate_hypercube(&r_rest);
        expected_result += w0_eval;

        expected_result += constraints
            .iter()
            .skip(1)
            .map(|constraint| {
                let mut combined = EvaluationsList::zero(constraint.num_variables());
                let mut eval = EF::ZERO;
                constraint.combine(&mut combined, &mut eval);
                let point = r_rest.get_subpoint_over_range(0..constraint.num_variables());
                combined.evaluate_hypercube(&point)
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
            let r_rest = final_point.get_subpoint_over_range(0..num_remaining);
            let r_skip = *final_point
                .last_variable()
                .expect("skip challenge must be present");

            // b) Reshape the W_0(X) evaluation table into a matrix.
            let w0_mat = w0_combined.into_mat(1 << num_remaining);

            // c) "Fold" the skipped variables by interpolating the matrix at `r_skip`.
            let folded_row = interpolate_subgroup(&w0_mat, r_skip);

            // d) Evaluate the resulting smaller polynomial at the remaining challenges `r_rest`.
            let w0_eval = EvaluationsList::new(folded_row).evaluate_hypercube(&r_rest);
            expected_result += w0_eval;

            expected_result += constraints
                .iter()
                .skip(1)
                .map(|constraint| {
                    let mut combined = EvaluationsList::zero(constraint.num_variables());
                    let mut eval = EF::ZERO;
                    constraint.combine(&mut combined, &mut eval);
                    let point = r_rest.get_subpoint_over_range(0..constraint.num_variables());
                    combined.evaluate_hypercube(&point)
                })
                .sum::<EF>();

            // The result from the recursive function must match the materialized ground truth.
            prop_assert_eq!(result_from_eval_poly, expected_result);
        }
    }
}
