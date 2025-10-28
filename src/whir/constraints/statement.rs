use p3_field::{ExtensionField, Field, TwoAdicField, dot_product};
use p3_matrix::dense::RowMajorMatrix;
use p3_multilinear_util::eq_batch::eval_eq_batch;
use tracing::instrument;

use crate::poly::{evals::EvaluationsList, multilinear::MultilinearPoint};

/// A batched system of evaluation constraints $p(z_i) = s_i$ on $\{0,1\}^m$.
///
/// Each entry ties a Boolean point `z_i` to an expected polynomial evaluation `s_i`.
///
/// Batching with a random challenge $\gamma$ produces a single combined weight
/// polynomial $W$ and a single scalar $S$ that summarize all constraints.
///
/// Invariants
/// ----------
/// - `points.len() == evaluations.len()`.
/// - Every `MultilinearPoint` in `points` has exactly `num_variables` coordinates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EqStatement<F> {
    /// Number of variables in the multilinear polynomial space.
    num_variables: usize,
    /// List of evaluation points.
    pub(crate) points: Vec<MultilinearPoint<F>>,
    /// List of target evaluations.
    pub(crate) evaluations: Vec<F>,
}

impl<F: Field> EqStatement<F> {
    /// Creates an empty `EqStatement<F>` for polynomials with `num_variables` variables.
    #[must_use]
    pub const fn initialize(num_variables: usize) -> Self {
        Self {
            num_variables,
            points: Vec::new(),
            evaluations: Vec::new(),
        }
    }

    /// Creates a filled `EqStatement<F>` for polynomials with `num_variables` variables.
    #[must_use]
    pub fn new(
        num_variables: usize,
        points: Vec<MultilinearPoint<F>>,
        evaluations: Vec<F>,
    ) -> Self {
        // Validate that we have one evaluation per point.
        assert_eq!(
            points.len(),
            evaluations.len(),
            "Number of points ({}) must match number of evaluations ({})",
            points.len(),
            evaluations.len()
        );

        // Validate that each point has the correct number of variables.
        for point in &points {
            assert_eq!(point.num_variables(), num_variables);
        }

        Self {
            num_variables,
            points,
            evaluations,
        }
    }

    /// Creates a filled `EqStatement<F>` for polynomials with `num_variables` variables
    /// using the univariate skip optimization.
    ///
    /// # Purpose
    ///
    /// This constructor validates constraint points that are structured for univariate skip..
    ///
    /// # Univariate Skip Structure
    ///
    /// When using univariate skip with `k_skip` variables:
    ///
    /// - The first `k_skip` variables are mapped to a multiplicative subgroup D
    /// - The remaining `j = num_variables - k_skip` variables live on the Boolean hypercube H^j
    /// - Each constraint point has `j + 1` coordinates structured as:
    ///   - `point[0]`: Evaluation point for the skipped variables (single coordinate in D)
    ///   - `point[1..j+1]`: Evaluation points for the hypercube variables (j coordinates)
    ///
    /// # Arguments
    ///
    /// - `num_variables`: Total number of variables in the original polynomial
    /// - `k_skip`: Number of variables to skip (must satisfy `0 < k_skip ≤ num_variables`)
    /// - `points`: Constraint points, each with `(num_variables - k_skip) + 1` coordinates
    /// - `evaluations`: Expected evaluations at the constraint points
    #[must_use]
    pub fn new_with_univariate_skip(
        num_variables: usize,
        k_skip: usize,
        points: Vec<MultilinearPoint<F>>,
        evaluations: Vec<F>,
    ) -> Self {
        // Validate that we have one evaluation per point.
        assert_eq!(
            points.len(),
            evaluations.len(),
            "Number of points ({}) must match number of evaluations ({})",
            points.len(),
            evaluations.len()
        );

        // Validate k_skip is in valid range.
        assert!(
            k_skip > 0,
            "k_skip must be greater than 0 (got {k_skip}). For k_skip=0, use the standard new() method."
        );
        assert!(
            k_skip <= num_variables,
            "k_skip ({k_skip}) must not exceed num_variables ({num_variables})"
        );

        // Calculate expected number of coordinates per point.
        //
        // Each point should have (j + 1) coordinates where j = n - k_skip.
        let expected_point_vars = num_variables - k_skip + 1;

        // Validate all points have the correct structure for univariate skip.
        for point in &points {
            assert_eq!(
                point.num_variables(),
                expected_point_vars,
                "Point must have {} coordinates for univariate skip with k_skip={} and num_variables={}, but has {}",
                expected_point_vars,
                k_skip,
                num_variables,
                point.num_variables()
            );
        }

        Self {
            num_variables,
            points,
            evaluations,
        }
    }

    /// Returns the number of variables defining the polynomial space.
    #[must_use]
    pub const fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// Returns true if the statement contains no constraints.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        debug_assert!(self.points.is_empty() == self.evaluations.is_empty());
        self.points.is_empty()
    }

    /// Returns an iterator over the evaluation constraints in the statement.
    pub fn iter(&self) -> impl Iterator<Item = (&MultilinearPoint<F>, &F)> {
        self.points.iter().zip(self.evaluations.iter())
    }

    /// Returns the number of constraints in the statement.
    #[must_use]
    pub const fn len(&self) -> usize {
        debug_assert!(self.points.len() == self.evaluations.len());
        self.points.len()
    }

    /// Verifies that a given polynomial satisfies all constraints in the statement.
    #[must_use]
    pub fn verify(&self, poly: &EvaluationsList<F>) -> bool {
        self.iter()
            .all(|(point, &expected_eval)| poly.evaluate(point) == expected_eval)
    }

    /// Concatenates another statement's constraints into this one.
    pub fn concatenate(&mut self, other: &Self) {
        assert_eq!(self.num_variables, other.num_variables);
        self.points.extend_from_slice(&other.points);
        self.evaluations.extend_from_slice(&other.evaluations);
    }

    /// Adds an evaluation constraint `p(z) = s` to the system.
    ///
    /// This method takes the polynomial `p` and uses it to compute the evaluation `s`.
    ///
    /// # Panics
    /// Panics if the number of variables in the `point` does not match the statement.
    pub fn add_unevaluated_constraint<BF>(
        &mut self,
        point: MultilinearPoint<F>,
        poly: &EvaluationsList<BF>,
    ) where
        BF: Field,
        F: ExtensionField<BF>,
    {
        assert_eq!(point.num_variables(), self.num_variables());
        let eval = poly.evaluate(&point);
        self.points.push(point);
        self.evaluations.push(eval);
    }

    /// Adds an evaluation constraint `p(z) = s` to the system.
    ///
    /// Assumes the evaluation `s` is already known.
    ///
    /// # Panics
    /// Panics if the number of variables in the `point` does not match the statement.
    pub fn add_evaluated_constraint(&mut self, point: MultilinearPoint<F>, eval: F) {
        assert_eq!(point.num_variables(), self.num_variables());
        self.points.push(point);
        self.evaluations.push(eval);
    }

    /// Inserts multiple constraints at the front of the system.
    ///
    /// Panics if any constraint's number of variables does not match the system.
    /// Combines all constraints into a single aggregated polynomial and expected sum using a challenge.
    #[instrument(skip_all, fields(num_constraints = self.len(), num_variables = self.num_variables()))]
    pub fn combine<Base, const INITIALIZED: bool>(
        &self,
        acc_weights: &mut EvaluationsList<F>,
        acc_sum: &mut F,
        challenge: F,
    ) where
        Base: Field,
        F: ExtensionField<Base>,
    {
        // If there are no constraints, the combination is:
        // - The combined polynomial W(X) is identically zero (all evaluations = 0).
        // - The combined expected sum S is zero.
        if self.points.is_empty() {
            return;
        }

        let num_constraints = self.len();
        let num_variables = self.num_variables();

        // Precompute challenge powers γ^i for i = 0..num_constraints-1.
        let challenges = challenge.powers().collect_n(num_constraints);

        // Create a matrix where each column is one evaluation point.
        //
        // Matrix layout:
        // - rows are variables,
        // - columns are evaluation points.
        let points_data = F::zero_vec(num_variables * num_constraints);
        let mut points_matrix = RowMajorMatrix::new(points_data, num_constraints);

        // Parallelize the transpose operation over rows (variables).
        //
        // Each thread writes to its own contiguous row, which is cache-friendly.
        points_matrix
            .rows_mut()
            .enumerate()
            .for_each(|(var_idx, row_slice)| {
                for (col_idx, point) in self.points.iter().enumerate() {
                    row_slice[col_idx] = point[var_idx];
                }
            });

        // Compute the batched equality polynomial evaluations.
        // This computes W(x) = ∑_i γ^i * eq(x, z_i) for all x ∈ {0,1}^k.
        eval_eq_batch::<Base, F, INITIALIZED>(
            points_matrix.as_view(),
            &mut acc_weights.0,
            &challenges,
        );

        // Combine expected evaluations: S = ∑_i γ^i * s_i
        *acc_sum +=
            dot_product::<F, _, _>(self.evaluations.iter().copied(), challenges.into_iter());
    }

    /// Combines a list of evals into a single linear combination using powers of `gamma`,
    /// and updates the running claimed_eval in place.
    ///
    /// # Arguments
    /// - `claimed_eval`: Mutable reference to the total accumulated claimed eval so far. Updated in place.
    /// - `gamma`: A random extension field element used to weight the evals.
    pub fn combine_evals(&self, claimed_eval: &mut F, gamma: F) {
        *claimed_eval += dot_product(self.evaluations.iter().copied(), gamma.powers());
    }

    /// Computes the equality polynomial for a point over a multiplicative subgroup domain.
    ///
    /// This function evaluates the unique polynomial `eq_D(X, y)` which is `1` if `X=y`
    /// and `0` at all other points `X` in the subgroup `D`. This implementation correctly
    /// interpolates the polynomial, allowing the constraint point `y` to be an arbitrary
    /// field element, not necessarily in `D`.
    ///
    /// The formula is:
    /// ```text
    /// eq_D(X, y) = (1/|D|) * Σ_{i=0}^{|D|-1} (X * y^{-1})^i
    /// ```
    ///
    /// # Arguments
    /// * `x`: The evaluation point, which is an element of the subgroup `D`.
    /// * `y`: The target constraint point, which is an arbitrary field element.
    /// * `subgroup_size`: The size of the multiplicative subgroup, `|D|`.
    ///
    /// # Returns
    /// The value of the equality polynomial `eq_D` evaluated at `x`.
    fn eq_d(x: F, y: F, subgroup_size: usize) -> F
    where
        F: TwoAdicField,
    {
        // Calculate the modular inverse of the subgroup's order, `1/|D|`.
        //
        // As `subgroup_size` is a power of two, this inversion is guaranteed to exist.
        let order = F::from_usize(subgroup_size);
        let order_inv = order
            .try_inverse()
            .expect("subgroup size must be invertible in the field");

        // Calculate the modular inverse of the target point `y`.
        let y_inv = y.inverse();

        // Calculate the base for the polynomial evaluation, `X * y^{-1}`.
        let base = x * y_inv;

        // Evaluate the polynomial `P(z) = Σ z^i` for `i` from 0 to `|D|-1` at the point `z = base`.
        let mut total_sum = F::ZERO;
        let mut power = F::ONE;
        for _ in 0..subgroup_size {
            total_sum += power;
            power *= base;
        }

        // Multiply by `1/|D|` to get the final evaluation.
        total_sum * order_inv
    }

    /// Combines constraints with the univariate skip optimization in mind.
    ///
    /// This function implements a protocol where constraint points are structured for a mixed
    /// domain. For an n-variable problem with a k-variable skip, constraint points must have
    /// `(n-k) + 1` coordinates.
    ///
    /// ## Protocol Point Structure
    ///
    /// - `point[0]` is a **single field element** `z_skip`, representing the constraint for the `k` skipped variables.
    /// - `point[1..]` is an **`(n-k)`-dimensional vector** `z_suffix` for the remaining hypercube variables.
    ///
    /// ## Mathematical Formulation
    ///
    /// The function computes the combined polynomial `W` over a `2^k x 2^j` grid (where `j=n-k`).
    /// The equality check for a constraint `z = (z_skip, z_suffix)` at a grid cell corresponding
    /// to the evaluation point `(x, y)` (where `x ∈ D` and `y ∈ H^j`) is:
    ///
    /// `eq(x, y) = eq_D(x, z_skip) * eq_H(y, z_suffix)`
    ///
    /// # Arguments
    /// * `challenge`: A random challenge `γ` for batching constraints.
    /// * `k_skip`: The number of variables folded into the subgroup `D`.
    ///
    /// # Returns
    /// A tuple containing:
    /// - An `EvaluationsList` with the `2^n` evaluations of the combined polynomial `W`.
    /// - The combined expected sum `S = Σ γ^i * s_i`.
    pub fn combine_univariate_skip<Base>(
        &self,
        challenge: F,
        k_skip: usize,
    ) -> (EvaluationsList<F>, F)
    where
        Base: Field,
        F: ExtensionField<Base> + TwoAdicField,
    {
        if self.is_empty() {
            return (
                EvaluationsList::new(F::zero_vec(1 << self.num_variables())),
                F::ZERO,
            );
        }

        // The number of constraints to be combined.
        let num_constraints = self.len();
        // The dimension of the points for this protocol, i.e., 1 + j.
        let num_point_vars = self.num_variables();
        // The number of variables for the hypercube part of the domain.
        let num_hypercube_vars = num_point_vars - 1;

        // The size of the multiplicative subgroup domain `D`.
        let subgroup_size = 1 << k_skip;
        // The size of the Boolean hypercube domain `H^j`.
        let hypercube_size = 1 << num_hypercube_vars;
        // The total size of the evaluation table for the mixed domain `D x H^j`.
        let total_domain_size = subgroup_size * hypercube_size;

        // Precompute powers of the random challenge `γ` for batching.
        let challenges = challenge.powers().collect_n(num_constraints);
        // Initialize the table that will hold all evaluations of the combined polynomial `W`.
        let mut final_evals = F::zero_vec(total_domain_size);

        // Get the generator for the subgroup `D` to map indices to subgroup elements.
        let subgroup_gen = F::two_adic_generator(k_skip);

        // Iterate over each constraint `z_i = (z_skip, z_suffix)` and its challenge `γ^i`.
        for (constraint_idx, (point, &_expected_eval)) in self.iter().enumerate() {
            // The challenge power for the current constraint.
            let gamma_i = challenges[constraint_idx];

            // Extract the single coordinate for the skipped part and the vector for the hypercube part.
            assert!(
                !point.as_slice().is_empty(),
                "Constraint point cannot be empty"
            );
            let z_skip = point[0];
            let z_suffix = &point.as_slice()[1..];
            assert_eq!(
                z_suffix.len(),
                num_hypercube_vars,
                "Point has an incorrect number of hypercube coordinates"
            );

            // Pre-compute the evaluations for the "column" (hypercube) part of the equality check.
            //
            // This result is constant for all `2^k` rows, so we compute it once per constraint.
            let suffix_point = MultilinearPoint::new(z_suffix.to_vec());
            let suffix_evals = EvaluationsList::<F>::new_from_point(&suffix_point, F::ONE);

            // Pre-compute the evaluations for the "row" (subgroup) part of the equality check.
            //
            // - We iterate through subgroup elements `x` in natural order (g^0, g^1, ...),
            // - We place their evaluations into the bit-reversed position in the `prefix_evals` array
            // to match the lexicographical layout of the evaluation table.
            //
            // TODO: we need to check this ordering is correct when spread in the code.
            // But with some trials it seems we need to do bit-reversed indexing.
            let mut prefix_evals = F::zero_vec(subgroup_size);
            let mut x = F::ONE;
            for i in 0..subgroup_size {
                // The bit-reversed index determines where the evaluation for x = g^i goes.
                let bit_rev_i = i.reverse_bits() >> (usize::BITS - k_skip as u32);
                prefix_evals[bit_rev_i] = Self::eq_d(x, z_skip, subgroup_size);
                x *= subgroup_gen; // Next subgroup element is g^(i+1)
            }

            // Combine the pre-computed parts using a tensor product to build the final table.
            for (row_idx, &prefix_eq_val) in prefix_evals.iter().enumerate().take(subgroup_size) {
                // Combine with the challenge to get the scaling factor for this row.
                let scalar_for_row = gamma_i * prefix_eq_val;

                // Apply the scaling factor to each column's pre-computed hypercube evaluation.
                let row_start = row_idx * hypercube_size;
                for col_idx in 0..hypercube_size {
                    final_evals[row_start + col_idx] +=
                        scalar_for_row * suffix_evals.as_slice()[col_idx];
                }
            }
        }

        // Compute the combined expected sum `S = Σ γ^i * s_i`.
        let sum = dot_product(self.evaluations.iter().copied(), challenges.into_iter());

        // Return the fully computed evaluation table and the combined sum.
        (EvaluationsList::new(final_evals), sum)
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use proptest::prelude::*;

    use super::*;

    type F = BabyBear;

    #[test]
    fn test_statement_combine_single_constraint() {
        let mut statement = EqStatement::initialize(1);
        let point = MultilinearPoint::new(vec![F::ONE]);
        let expected_eval = F::from_u64(7);
        statement.add_evaluated_constraint(point.clone(), expected_eval);

        let challenge = F::from_u64(2); // This is unused with one constraint.
        let mut combined_evals = EvaluationsList::zero(statement.num_variables());
        let mut combined_sum = F::ZERO;
        statement.combine::<_, false>(&mut combined_evals, &mut combined_sum, challenge);

        // Expected evals for eq_z(X) where z = (1).
        // For x=0, eq=0. For x=1, eq=1.
        let expected_combined_evals_vec = EvaluationsList::new_from_point(&point, F::ONE);

        assert_eq!(combined_evals, expected_combined_evals_vec);
        assert_eq!(combined_sum, expected_eval);
    }

    #[test]
    fn test_statement_with_multiple_constraints() {
        let mut statement = EqStatement::initialize(2);

        // Constraint 1: evaluate at z1 = (1,0), expected value 5
        let point1 = MultilinearPoint::new(vec![F::ONE, F::ZERO]);
        let eval1 = F::from_u64(5);
        statement.add_evaluated_constraint(point1.clone(), eval1);

        // Constraint 2: evaluate at z2 = (0,1), expected value 7
        let point2 = MultilinearPoint::new(vec![F::ZERO, F::ONE]);
        let eval2 = F::from_u64(7);
        statement.add_evaluated_constraint(point2.clone(), eval2);

        let challenge = F::from_u64(2);
        let mut combined_evals = EvaluationsList::zero(statement.num_variables());
        let mut combined_sum = F::ZERO;
        statement.combine::<_, false>(&mut combined_evals, &mut combined_sum, challenge);

        // Expected evals: W(X) = eq_z1(X) + challenge * eq_z2(X)
        let mut expected_combined_evals_vec = EvaluationsList::new_from_point(&point1, F::ONE);
        expected_combined_evals_vec.accumulate_batch(&[point2], &[challenge]);

        // Expected sum: S = s1 + challenge * s2
        let expected_combined_sum = eval1 + challenge * eval2;

        assert_eq!(combined_evals, expected_combined_evals_vec);
        assert_eq!(combined_sum, expected_combined_sum);
    }

    #[test]
    fn test_compute_evaluation_weight() {
        // Define an evaluation weight at a specific point
        let point = MultilinearPoint::new(vec![F::from_u64(3)]);

        // Define a randomness point for folding
        let folding_randomness = MultilinearPoint::new(vec![F::from_u64(2)]);

        // Expected result is the evaluation of eq_poly at the given randomness
        let expected = point.eq_poly(&folding_randomness);

        assert_eq!(point.eq_poly(&folding_randomness), expected);
    }

    #[test]
    fn test_constructors_and_basic_properties() {
        // Test new constructor
        let point = MultilinearPoint::new(vec![F::ONE]);
        let eval = F::from_u64(42);
        let statement = EqStatement::new(1, vec![point], vec![eval]);

        assert_eq!(statement.num_variables(), 1);
        assert_eq!(statement.len(), 1);
        assert!(!statement.is_empty());

        // Test initialize constructor
        let empty_statement = EqStatement::<F>::initialize(2);
        assert_eq!(empty_statement.num_variables(), 2);
        assert_eq!(empty_statement.len(), 0);
        assert!(empty_statement.is_empty());
    }

    #[test]
    fn test_verify_constraints() {
        // Create polynomial with evaluations [1, 2]
        let poly = EvaluationsList::new(vec![F::from_u64(1), F::from_u64(2)]);
        let mut statement = EqStatement::<F>::initialize(1);

        // Test matching constraint: f(0) = 1
        statement.add_evaluated_constraint(MultilinearPoint::new(vec![F::ZERO]), F::from_u64(1));
        assert!(statement.verify(&poly));

        // Test mismatched constraint: f(1) = 5 (but poly has f(1) = 2)
        statement.add_evaluated_constraint(MultilinearPoint::new(vec![F::ONE]), F::from_u64(5));
        assert!(!statement.verify(&poly));
    }

    #[test]
    fn test_concatenate() {
        // Test successful concatenation
        let mut statement1 = EqStatement::<F>::initialize(1);
        let mut statement2 = EqStatement::<F>::initialize(1);
        statement1.add_evaluated_constraint(MultilinearPoint::new(vec![F::ZERO]), F::from_u64(10));
        statement2.add_evaluated_constraint(MultilinearPoint::new(vec![F::ONE]), F::from_u64(20));

        statement1.concatenate(&statement2);
        assert_eq!(statement1.len(), 2);
    }

    #[test]
    #[should_panic(expected = "assertion `left == right` failed")]
    fn test_concatenate_mismatched_variables() {
        let mut statement1 = EqStatement::<F>::initialize(2);
        let statement2 = EqStatement::<F>::initialize(3);
        statement1.concatenate(&statement2); // Should panic
    }

    #[test]
    fn test_add_constraints() {
        // Test that add_unevaluated_constraint behaves identically to add_evaluated_constraint
        let poly = EvaluationsList::new(vec![F::from_u64(1), F::from_u64(2)]);
        let point = MultilinearPoint::new(vec![F::ZERO]);

        // Create two identical statements
        let mut statement1 = EqStatement::<F>::initialize(1);
        let mut statement2 = EqStatement::<F>::initialize(1);

        // Add same constraint using both methods
        let eval = poly.evaluate(&point);
        statement1.add_evaluated_constraint(point.clone(), eval);
        statement2.add_unevaluated_constraint(point, &poly);

        // Both statements should be identical
        assert_eq!(statement1.len(), statement2.len());
        assert_eq!(statement1.len(), 1);

        // Both should verify against the polynomial
        assert!(statement1.verify(&poly));
        assert!(statement2.verify(&poly));

        // Both should have same constraints
        let constraints1: Vec<_> = statement1.iter().collect();
        let constraints2: Vec<_> = statement2.iter().collect();
        assert_eq!(constraints1, constraints2);

        // Test get_points consumes statement
        assert_eq!(statement1.points.len(), 1);
    }

    #[test]
    #[should_panic(expected = "assertion `left == right` failed")]
    fn test_wrong_variable_count() {
        let mut statement = EqStatement::<F>::initialize(1);
        let wrong_point = MultilinearPoint::new(vec![F::ONE, F::ZERO]); // 2 vars for 1-var statement
        statement.add_evaluated_constraint(wrong_point, F::from_u64(5));
    }

    #[test]
    fn test_combine_operations() {
        // Test empty statement combine
        let empty_statement = EqStatement::<F>::initialize(1);

        let mut combined_evals = EvaluationsList::zero(empty_statement.num_variables());
        let mut combined_sum = F::ZERO;
        empty_statement.combine::<_, false>(
            &mut combined_evals,
            &mut combined_sum,
            F::from_u64(42),
        );
        assert_eq!(combined_sum, F::ZERO);

        // Test combine_evals with constraints
        let mut statement = EqStatement::<F>::initialize(1);
        statement.add_evaluated_constraint(MultilinearPoint::new(vec![F::ZERO]), F::from_u64(3));
        statement.add_evaluated_constraint(MultilinearPoint::new(vec![F::ONE]), F::from_u64(7));

        let mut claimed_eval = F::ZERO;
        statement.combine_evals(&mut claimed_eval, F::from_u64(2));

        // Verify: 3*1 + 7*2 = 17
        assert_eq!(claimed_eval, F::from_u64(17));
    }

    #[test]
    fn test_eq_d_basic() {
        // Test the eq_D function with a small subgroup of size 4
        let subgroup_size = 4;

        // Generate the subgroup: {1, g, g^2, g^3} where g is the 4th root of unity
        let generator = F::two_adic_generator(2); // 2^2 = 4
        let subgroup: Vec<F> = (0..subgroup_size)
            .map(|i| generator.exp_u64(i as u64))
            .collect();

        // Test that eq_D(x, y) = 1 when x == y
        for &y in &subgroup {
            for &x in &subgroup {
                let result = EqStatement::<F>::eq_d(x, y, subgroup_size);
                if x == y {
                    assert_eq!(result, F::ONE, "eq_D({x:?}, {y:?}) should be 1");
                } else {
                    assert_eq!(result, F::ZERO, "eq_D({x:?}, {y:?}) should be 0");
                }
            }
        }
    }

    proptest! {
        #[test]
        fn proptest_eq_d(k_skip in 1usize..=5usize) {
            // The index for the evaluation point `x` in the subgroup.
            let x_idx = 0u64;
            // The index for the constraint point `y` in the subgroup.
            let y_idx = 0u64;
            // Calculate the size of the subgroup, |D| = 2^k.
            let subgroup_size = 1 << k_skip;
            // Get the generator `g` for the 2^k-th roots of unity.
            let generator = F::two_adic_generator(k_skip);

            // Determine the evaluation point `x = g^x_idx`.
            let x = generator.exp_u64(x_idx);
            // Determine the constraint point `y = g^y_idx`.
            let y = generator.exp_u64(y_idx);

            // Compute the equality polynomial evaluation eq_D(x, y).
            let result = EqStatement::<F>::eq_d(x, y, subgroup_size);

            // The equality polynomial must be 1 if the points are the same, and 0 otherwise.
            if x_idx == y_idx {
                // If x == y, the result must be 1.
                prop_assert_eq!(result, F::ONE, "eq_D(x, y) should be 1 for x == y");
            } else {
                // If x != y, the result must be 0.
                prop_assert_eq!(result, F::ZERO, "eq_D(x, y) should be 0 for x != y");
            }
        }
    }

    #[test]
    fn test_combine_univariate_skip_two_constraints() {
        // We test with:
        // - k=2 skipped variables (subgroup D of size 4). Let the generator be `g`.
        // - j=1 hypercube variable (H^1 of size 2).
        // The domain is a 4x2 grid (8 total points).
        let k_skip = 2;
        let num_hypercube_vars = 1;
        let num_point_vars = num_hypercube_vars + 1;

        // Get the subgroup generator `g`.
        let subgroup_gen = F::two_adic_generator(k_skip);
        let num_cols = 1 << num_hypercube_vars;

        // Constraints
        let mut statement = EqStatement::initialize(num_point_vars);
        // Constraint 0: z0 = (g^1, [1]), with evaluation s0 = 5.
        let point0 = MultilinearPoint::new([vec![subgroup_gen.exp_u64(1)], vec![F::ONE]].concat());
        let eval0 = F::from_u64(5);
        statement.add_evaluated_constraint(point0, eval0);

        // Constraint 1: z1 = (g^3, [0]), with evaluation s1 = 8.
        let point1 = MultilinearPoint::new([vec![subgroup_gen.exp_u64(3)], vec![F::ZERO]].concat());
        let eval1 = F::from_u64(8);
        statement.add_evaluated_constraint(point1, eval1);

        // Use a random challenge for batching.
        let challenge = F::from_u64(2);

        // Execution
        let (combined_evals, combined_sum) =
            statement.combine_univariate_skip::<F>(challenge, k_skip);

        // Verification

        // Manually compute the expected combined sum: S = s0 * γ^0 + s1 * γ^1.
        let expected_sum = eval0 * challenge.exp_u64(0) + eval1 * challenge.exp_u64(1);
        assert_eq!(combined_sum, expected_sum);

        // Manually derive the expected evaluation table W(x, y) = eq(z0) * γ^0 + eq(z1) * γ^1.
        // This table should have exactly two non-zero entries.

        // Location of z0 = (g^1, [1]):
        // - The row for g^1 has a natural index of 1. Its bit-reversed index is 2.
        let row0 = 1usize.reverse_bits() >> (usize::BITS - k_skip as u32); // 2
        // - The column for [1] is index 1.
        let col0 = 1;
        // - The 1D memory index is row * num_cols + col.
        let idx0 = row0 * num_cols + col0;
        // - The value at this index should be the weight γ^0 = 1.
        // This corresponds to the first constraint (i=0),
        // which contributes its weight as eq(z0) is 1 at this location.
        let val0 = challenge.exp_u64(0);

        // Location of z1 = (g^3, [0]):
        // - The row for g^3 has a natural index of 3. Its bit-reversed index is 3.
        let row1 = 3usize.reverse_bits() >> (usize::BITS - k_skip as u32); // 3
        // - The column for [0] is index 0.
        let col1 = 0;
        // - The 1D memory index is row * num_cols + col.
        let idx1 = row1 * num_cols + col1;
        // - The value at this index should be the weight γ^1 = 2.
        // This corresponds to the second constraint (i=1),
        // which contributes its weight as eq(z1) is 1 at this location.
        let val1 = challenge.exp_u64(1);

        // Construct the full expected table from our manual calculations.
        let mut expected_evals_vec = F::zero_vec(8);
        expected_evals_vec[idx0] = val0;
        expected_evals_vec[idx1] = val1;

        // The computed evaluations must match our manually derived table.
        assert_eq!(combined_evals.as_slice(), expected_evals_vec.as_slice());
    }

    #[test]
    fn test_combine_univariate_skip_with_collision() {
        // This test uses four constraints, two of which are at the *same point*.
        //
        // This validates that the combination logic correctly *accumulates* weights.
        // Domain:
        // - k=2 skip vars (4 rows),
        // - j=1 hypercube var (2 cols).
        let k_skip = 2;
        let num_hypercube_vars = 1;

        let subgroup_gen = F::two_adic_generator(k_skip);

        // Constraints
        let mut statement = EqStatement::initialize(num_hypercube_vars + 1);
        // Constraint 0: z0 = (g^1, [1]), eval s0 = 5.
        statement.add_evaluated_constraint(
            MultilinearPoint::new([vec![subgroup_gen.exp_u64(1)], vec![F::ONE]].concat()),
            F::from_u64(5),
        );
        // Constraint 1: z1 = (g^3, [0]), eval s1 = 8.
        statement.add_evaluated_constraint(
            MultilinearPoint::new([vec![subgroup_gen.exp_u64(3)], vec![F::ZERO]].concat()),
            F::from_u64(8),
        );
        // Constraint 2: z2 = (g^0, [1]), eval s2 = 3.
        statement.add_evaluated_constraint(
            MultilinearPoint::new([vec![subgroup_gen.exp_u64(0)], vec![F::ONE]].concat()),
            F::from_u64(3),
        );
        // Constraint 3: z3 = (g^1, [1]), eval s3 = 1. *** This collides with z0 ***
        statement.add_evaluated_constraint(
            MultilinearPoint::new([vec![subgroup_gen.exp_u64(1)], vec![F::ONE]].concat()),
            F::from_u64(1),
        );

        let challenge = F::from_u64(2);

        // Execution
        let (combined_evals, combined_sum) =
            statement.combine_univariate_skip::<F>(challenge, k_skip);

        // Verification

        // Manually compute the expected sum: S = Σ s_i * γ^i.
        let s = &statement.evaluations;
        let expected_sum = s[0] * challenge.exp_u64(0)
            + s[1] * challenge.exp_u64(1)
            + s[2] * challenge.exp_u64(2)
            + s[3] * challenge.exp_u64(3);
        assert_eq!(combined_sum, expected_sum);

        // Manually derive the expected evaluation table.
        //
        // It will have 3 non-zero entries, one of which is an accumulation of two weights.
        let mut expected_evals_vec = F::zero_vec(8);

        // Contribution from z0=(g^1, [1]) is γ^0 = 1.
        // Location: row=bit_rev(1)=2, col=1 -> index=5.
        expected_evals_vec[5] += challenge.exp_u64(0);

        // Contribution from z1=(g^3, [0]) is γ^1 = 2.
        // Location: row=bit_rev(3)=3, col=0 -> index=6.
        expected_evals_vec[6] += challenge.exp_u64(1); // val = 2

        // Contribution from z2=(g^0, [1]) is γ^2 = 4.
        // Location: row=bit_rev(0)=0, col=1 -> index=1.
        expected_evals_vec[1] += challenge.exp_u64(2); // val = 4

        // Contribution from z3=(g^1, [1]) is γ^3 = 8.
        // Location: same as z0, index=5.
        expected_evals_vec[5] += challenge.exp_u64(3); // val at index 5 is now 1 + 8 = 9

        // Final expected table:
        // Index 0: 0
        // Index 1: 4 (from z2)
        // Index 2: 0
        // Index 3: 0
        // Index 4: 0
        // Index 5: 9 (from z0 + z3)
        // Index 6: 2 (from z1)
        // Index 7: 0
        assert_eq!(combined_evals.as_slice(), expected_evals_vec.as_slice());
    }

    proptest! {
        #[test]
        fn prop_statement_workflow(
            // Random 4-var polynomial: 16 evaluations (2^4)
            poly_evals in prop::collection::vec(0u32..100, 16),
            // Random challenge value
            challenge in 1u32..50,
            // Random constraint points (4 coords × 2 points)
            point_coords in prop::collection::vec(0u32..10, 8),
        ) {
            // Create a 4-variable polynomial from random evaluations
            let poly = EvaluationsList::new(poly_evals.into_iter().map(F::from_u32).collect());

            // Create statement with random constraints that match the polynomial
            let mut statement = EqStatement::<F>::initialize(4);
            let point1 = MultilinearPoint::new(vec![
                F::from_u32(point_coords[0]), F::from_u32(point_coords[1]),
                F::from_u32(point_coords[2]), F::from_u32(point_coords[3])
            ]);
            let point2 = MultilinearPoint::new(vec![
                F::from_u32(point_coords[4]), F::from_u32(point_coords[5]),
                F::from_u32(point_coords[6]), F::from_u32(point_coords[7])
            ]);

            // Add constraints: poly(point1) = actual_eval1, poly(point2) = actual_eval2
            let eval1 = poly.evaluate(&point1);
            let eval2 = poly.evaluate(&point2);
            statement.add_evaluated_constraint(point1, eval1);
            statement.add_evaluated_constraint(point2, eval2);

            // Statement should verify against polynomial (consistent constraints)
            prop_assert!(statement.verify(&poly));

            // Combine constraints with challenge
            let gamma = F::from_u32(challenge);
            let mut combined_poly = EvaluationsList::zero(statement.num_variables());
            let mut combined_sum = F::ZERO;
            statement.combine::<_, false>(&mut combined_poly, &mut combined_sum, gamma);

            // Combined polynomial should have same number of variables
            prop_assert_eq!(combined_poly.num_variables(), 4);

            // Combined evaluations should match combine result
            let mut claimed_eval = F::ZERO;
            statement.combine_evals(&mut claimed_eval, gamma);
            // Both methods should give same sum
            prop_assert_eq!(combined_sum, claimed_eval);

            // Adding wrong constraint should break verification
            let wrong_point = MultilinearPoint::new(vec![F::ZERO, F::ZERO, F::ZERO, F::ZERO]);
            // Obviously wrong evaluation
            let wrong_eval = F::from_u32(999);
            let actual_eval = poly.evaluate(&wrong_point);
            // Only test if actually different
            if wrong_eval != actual_eval {
                statement.add_evaluated_constraint(wrong_point, wrong_eval);
                // Should fail verification
                prop_assert!(!statement.verify(&poly));
            }
        }
    }

    #[test]
    fn test_new_with_univariate_skip_valid() {
        // Test with n=5, k_skip=2
        // Points should have (5-2)+1 = 4 coordinates
        let points = vec![
            MultilinearPoint::new(vec![F::from_u64(10), F::ONE, F::ZERO, F::ONE]),
            MultilinearPoint::new(vec![F::from_u64(20), F::ZERO, F::ONE, F::ZERO]),
        ];
        let evaluations = vec![F::from_u64(100), F::from_u64(200)];

        let statement = EqStatement::new_with_univariate_skip(5, 2, points, evaluations);

        assert_eq!(statement.num_variables(), 5);
        assert_eq!(statement.len(), 2);
    }

    #[test]
    #[should_panic(expected = "k_skip must be greater than 0")]
    fn test_new_with_univariate_skip_zero_k() {
        // Should panic when k_skip = 0
        let points = vec![MultilinearPoint::new(vec![F::ONE, F::ZERO])];
        let evaluations = vec![F::from_u64(100)];

        let _ = EqStatement::new_with_univariate_skip(2, 0, points, evaluations);
    }

    #[test]
    #[should_panic(expected = "must not exceed num_variables")]
    fn test_new_with_univariate_skip_k_too_large() {
        // Should panic when k_skip > num_variables
        let points = vec![MultilinearPoint::new(vec![F::ONE])];
        let evaluations = vec![F::from_u64(100)];

        let _ = EqStatement::new_with_univariate_skip(2, 3, points, evaluations);
    }

    #[test]
    #[should_panic(expected = "Point must have 4 coordinates")]
    fn test_new_with_univariate_skip_wrong_point_size() {
        // For n=5, k_skip=2, points need (5-2)+1 = 4 coordinates
        // This point only has 3 coordinates, should panic
        let points = vec![MultilinearPoint::new(vec![F::ONE, F::ZERO, F::ONE])];
        let evaluations = vec![F::from_u64(100)];

        let _ = EqStatement::new_with_univariate_skip(5, 2, points, evaluations);
    }

    #[test]
    fn test_new_with_univariate_skip_edge_case_k_equals_n_minus_1() {
        // Test edge case: k_skip = n-1
        // For n=5, k_skip=4, points should have (5-4)+1 = 2 coordinates
        let points = vec![MultilinearPoint::new(vec![F::from_u64(10), F::ONE])];
        let evaluations = vec![F::from_u64(100)];

        let statement = EqStatement::new_with_univariate_skip(5, 4, points, evaluations);

        assert_eq!(statement.num_variables(), 5);
        assert_eq!(statement.len(), 1);
        assert_eq!(statement.points[0].num_variables(), 2);
    }

    #[test]
    fn test_new_with_univariate_skip_k_equals_n() {
        // Test edge case: k_skip = n
        // For n=3, k_skip=3, points should have (3-3)+1 = 1 coordinate
        let points = vec![MultilinearPoint::new(vec![F::from_u64(42)])];
        let evaluations = vec![F::from_u64(100)];

        let statement = EqStatement::new_with_univariate_skip(3, 3, points, evaluations);

        assert_eq!(statement.num_variables(), 3);
        assert_eq!(statement.len(), 1);
        assert_eq!(statement.points[0].num_variables(), 1);
    }

    #[test]
    #[should_panic(expected = "Number of points (2) must match number of evaluations (1)")]
    fn test_new_mismatched_lengths() {
        // Should panic when points.len() != evaluations.len()
        let points = vec![
            MultilinearPoint::new(vec![F::ONE]),
            MultilinearPoint::new(vec![F::ZERO]),
        ];
        let evaluations = vec![F::from_u64(100)];

        let _ = EqStatement::new(1, points, evaluations);
    }

    #[test]
    #[should_panic(expected = "Number of points (1) must match number of evaluations (2)")]
    fn test_new_with_univariate_skip_mismatched_lengths() {
        // Should panic when points.len() != evaluations.len()
        let points = vec![MultilinearPoint::new(vec![F::from_u64(10), F::ONE])];
        let evaluations = vec![F::from_u64(100), F::from_u64(200)];

        let _ = EqStatement::new_with_univariate_skip(3, 2, points, evaluations);
    }
}
