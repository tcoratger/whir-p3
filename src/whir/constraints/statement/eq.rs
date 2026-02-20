use alloc::vec::Vec;

use itertools::Itertools;
use p3_field::{
    ExtensionField, Field, PackedFieldExtension, PackedValue, PrimeCharacteristicRing, dot_product,
};
use p3_matrix::{
    Matrix,
    dense::{RowMajorMatrix, RowMajorMatrixView},
};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::eq_batch::eval_eq_batch;
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::poly::{evals::EvaluationsList, multilinear::MultilinearPoint};

/// Given multiple points in matrix form where each column is a point `P_i`,
/// builds individual eq coefficients `eq(P_i, X)` for all points `X` in the boolean hypercube
/// Returns a matrix where each columns is the eq coefficients of `P_i`.
fn batch_eqs<F: Field, EF: ExtensionField<F>>(
    points: RowMajorMatrixView<'_, EF>,
    alpha: EF,
) -> RowMajorMatrix<EF> {
    let k = points.height();
    let n = points.width();
    assert_ne!(n, 0);

    let mut mat = RowMajorMatrix::new(EF::zero_vec(n * (1 << k)), n);
    mat.row_mut(0).copy_from_slice(&alpha.powers().collect_n(n));
    points.row_slices().enumerate().for_each(|(i, vars)| {
        let (mut lo, mut hi) = mat.split_rows_mut(1 << i);
        lo.rows_mut().zip(hi.rows_mut()).for_each(|(lo, hi)| {
            vars.iter()
                .zip(lo.iter_mut().zip(hi.iter_mut()))
                .for_each(|(&var, (lo, hi))| {
                    *hi = *lo * var;
                    *lo -= *hi;
                });
        });
    });
    mat
}

/// Given multiple points in matrix form where each column is a point `P_i`,
/// builds individual eq coefficients `eq(P_i, X)` for all points `X` in the boolean hypercube
/// Returns a matrix where each columns is the eq coefficients of `P_i` in packed form.
fn packed_batch_eqs<F: Field, EF: ExtensionField<F>>(
    points: RowMajorMatrixView<'_, EF>,
) -> RowMajorMatrix<EF::ExtensionPacking> {
    let k = points.height();
    let n = points.width();
    assert_ne!(n, 0);
    let k_pack = log2_strict_usize(F::Packing::WIDTH);
    assert!(k >= k_pack);

    let (init_vars, rest_vars) = points.split_rows(k_pack);
    let mut mat = RowMajorMatrix::new(EF::ExtensionPacking::zero_vec(n * (1 << (k - k_pack))), n);
    if k_pack > 0 {
        init_vars
            .transpose()
            .row_slices()
            .zip(mat.values.iter_mut())
            .for_each(|(vars, packed)| {
                let point = vars.iter().rev().copied().collect::<Vec<_>>();
                *packed = EF::ExtensionPacking::from_ext_slice(
                    EvaluationsList::new_from_point(&point, EF::ONE).as_slice(),
                );
            });
    } else {
        mat.row_mut(0).fill(EF::ExtensionPacking::ONE);
    }

    rest_vars.row_slices().enumerate().for_each(|(i, vars)| {
        let (mut lo, mut hi) = mat.split_rows_mut(1 << i);
        lo.rows_mut().zip(hi.rows_mut()).for_each(|(lo, hi)| {
            vars.iter()
                .zip(lo.iter_mut().zip(hi.iter_mut()))
                .for_each(|(&var, (lo, hi))| {
                    *hi = *lo * var;
                    *lo -= *hi;
                });
        });
    });
    mat
}

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
            .all(|(point, &expected_eval)| poly.evaluate_hypercube_base(point) == expected_eval)
    }

    /// Concatenates another statement's constraints into this one.
    pub fn concatenate(&mut self, other: &Self) {
        assert_eq!(self.num_variables, other.num_variables);
        self.points.extend_from_slice(&other.points);
        self.evaluations.extend_from_slice(&other.evaluations);
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
    ///
    /// # Standard Hypercube Representation
    ///
    /// This method is for the standard case where the equality polynomial is computed over the
    /// Boolean hypercube `{0,1}^num_variables`. It evaluates W(x) = ∑_i γ^i * eq(x, z_i) for all
    /// x ∈ {0,1}^k, where the z_i are arbitrary constraint points.
    #[instrument(skip_all, fields(num_constraints = self.len(), num_variables = self.num_variables()))]
    pub fn combine_hypercube<Base, const INITIALIZED: bool>(
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

    /// Inserts multiple constraints at the front of the system.
    ///
    /// Panics if any constraint's number of variables does not match the system.
    /// Combines all constraints into a single aggregated polynomial and expected sum using a challenge.
    ///
    /// # Standard Hypercube Representation
    ///
    /// This method is for the standard case where the equality polynomial is computed over the
    /// Boolean hypercube `{0,1}^num_variables`. It evaluates W(x) = ∑_i γ^i * eq(x, z_i) for all
    /// x ∈ {0,1}^k, where the z_i are arbitrary constraint points.
    #[instrument(skip_all, fields(num_constraints = self.len(), num_variables = self.num_variables()))]
    pub fn combine_hypercube_packed<Base, const INITIALIZED: bool>(
        &self,
        weights: &mut EvaluationsList<F::ExtensionPacking>,
        sum: &mut F,
        challenge: F,
    ) where
        Base: Field,
        F: ExtensionField<Base>,
    {
        if self.points.is_empty() {
            return;
        }

        let k = self.num_variables();
        let k_pack = log2_strict_usize(Base::Packing::WIDTH);
        assert!(k >= k_pack);
        assert_eq!(weights.num_variables() + k_pack, k);

        // Combine expected evaluations: S = ∑_i γ^i * s_i
        self.combine_evals(sum, challenge);

        // Apply naive method if number of variables is too small for packed split method
        if k_pack * 2 > k {
            self.points
                .iter()
                .zip(challenge.powers())
                .enumerate()
                .for_each(|(i, (point, challenge))| {
                    let eq = EvaluationsList::new_from_point(point.as_slice(), challenge);
                    weights
                        .0
                        .iter_mut()
                        .zip_eq(eq.0.chunks(Base::Packing::WIDTH))
                        .for_each(|(out, chunk)| {
                            let packed = F::ExtensionPacking::from_ext_slice(chunk);
                            if INITIALIZED || i > 0 {
                                *out += packed;
                            } else {
                                *out = packed;
                            }
                        });
                });
            return;
        }

        let points = MultilinearPoint::transpose(&self.points, true);
        let (left, right) = points.split_rows(k / 2);
        let left = packed_batch_eqs::<Base, F>(left);
        let right = batch_eqs::<Base, F>(right, challenge);

        weights
            .0
            .par_chunks_mut(left.height())
            .zip_eq(right.par_row_slices())
            .for_each(|(out, right)| {
                out.iter_mut().zip(left.rows()).for_each(|(out, left)| {
                    if INITIALIZED {
                        *out +=
                            dot_product::<F::ExtensionPacking, _, _>(left, right.iter().copied());
                    } else {
                        *out = dot_product(left, right.iter().copied());
                    }
                });
            });
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
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};
    use proptest::prelude::*;
    use rand::{RngExt, SeedableRng, rngs::SmallRng};

    use super::*;

    impl<F: Field> EqStatement<F> {
        /// Creates a filled `EqStatement<F>` for polynomials with `num_variables` variables.
        ///
        /// # Standard Hypercube Representation
        ///
        /// This constructor is for the standard case where the polynomial is represented as
        /// evaluations over the Boolean hypercube `{0,1}^num_variables`, and will be evaluated
        /// at arbitrary constraint points using standard multilinear interpolation. Each point
        /// has exactly `num_variables` coordinates.
        #[must_use]
        pub fn new_hypercube(points: Vec<MultilinearPoint<F>>, evaluations: Vec<F>) -> Self {
            // Validate that we have one evaluation per point.
            assert_eq!(
                points.len(),
                evaluations.len(),
                "Number of points ({}) must match number of evaluations ({})",
                points.len(),
                evaluations.len()
            );

            // Validate that each point has the correct number of variables.
            let num_variables = points
                .iter()
                .map(MultilinearPoint::num_variables)
                .all_equal_value()
                .unwrap();
            Self {
                num_variables,
                points,
                evaluations,
            }
        }
    }

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    #[test]
    fn test_statement_combine_single_constraint() {
        let mut statement = EqStatement::initialize(1);
        let point = MultilinearPoint::new(vec![F::ONE]);
        let expected_eval = F::from_u64(7);
        statement.add_evaluated_constraint(point.clone(), expected_eval);

        let challenge = F::from_u64(2); // This is unused with one constraint.
        let mut combined_evals = EvaluationsList::zero(statement.num_variables());
        let mut combined_sum = F::ZERO;
        statement.combine_hypercube::<_, false>(&mut combined_evals, &mut combined_sum, challenge);

        // Expected evals for eq_z(X) where z = (1).
        // For x=0, eq=0. For x=1, eq=1.
        let expected_combined_evals_vec = EvaluationsList::new_from_point(point.as_slice(), F::ONE);

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
        statement.combine_hypercube::<_, false>(&mut combined_evals, &mut combined_sum, challenge);

        // Expected evals: W(X) = eq_z1(X) + challenge * eq_z2(X)
        let expected_eq1 = EvaluationsList::new_from_point(point1.as_slice(), F::ONE);
        let expected_eq2 = EvaluationsList::new_from_point(point2.as_slice(), challenge);
        let expected_combined_evals_vec = EvaluationsList::new(
            expected_eq1
                .iter()
                .zip(expected_eq2.iter())
                .map(|(&a, &b)| a + b)
                .collect(),
        );

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
        // Test new_hypercube constructor
        let point = MultilinearPoint::new(vec![F::ONE]);
        let eval = F::from_u64(42);
        let statement = EqStatement::new_hypercube(vec![point], vec![eval]);

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
    fn test_add_evaluated_constraint() {
        let poly = EvaluationsList::new(vec![F::from_u64(1), F::from_u64(2)]);
        let point = MultilinearPoint::new(vec![F::ZERO]);

        let mut statement = EqStatement::<F>::initialize(1);

        // Add constraint with pre-computed evaluation
        let eval = poly.evaluate_hypercube_base(&point);
        statement.add_evaluated_constraint(point, eval);

        // Statement should have one constraint
        assert_eq!(statement.len(), 1);

        // Should verify against the polynomial
        assert!(statement.verify(&poly));

        // Points should be stored
        assert_eq!(statement.points.len(), 1);
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
        empty_statement.combine_hypercube::<_, false>(
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
            let eval1 = poly.evaluate_hypercube_base(&point1);
            let eval2 = poly.evaluate_hypercube_base(&point2);
            statement.add_evaluated_constraint(point1, eval1);
            statement.add_evaluated_constraint(point2, eval2);

            // Statement should verify against polynomial (consistent constraints)
            prop_assert!(statement.verify(&poly));

            // Combine constraints with challenge
            let gamma = F::from_u32(challenge);
            let mut combined_poly = EvaluationsList::zero(statement.num_variables());
            let mut combined_sum = F::ZERO;
            statement.combine_hypercube::<_, false>(&mut combined_poly, &mut combined_sum, gamma);

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
            let actual_eval = poly.evaluate_hypercube_base(&wrong_point);
            // Only test if actually different
            if wrong_eval != actual_eval {
                statement.add_evaluated_constraint(wrong_point, wrong_eval);
                // Should fail verification
                prop_assert!(!statement.verify(&poly));
            }
        }
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

        let _ = EqStatement::new_hypercube(points, evaluations);
    }

    #[test]
    fn test_packed_combine() {
        let mut rng = SmallRng::seed_from_u64(1);
        let challenge: EF = rng.random();
        let k_pack = log2_strict_usize(<F as Field>::Packing::WIDTH);

        for k in k_pack..10 {
            let mut out0 = EvaluationsList::zero(k);
            let mut out1 =
                EvaluationsList::<<EF as ExtensionField<F>>::ExtensionPacking>::zero(k - k_pack);
            let mut sum0 = EF::ZERO;
            let mut sum1 = EF::ZERO;
            let mut init = false;
            for n in [1, 2, 10, 11] {
                let points = (0..n)
                    .map(|_| MultilinearPoint::rand(&mut rng, k))
                    .collect::<Vec<_>>();
                let evals = (0..n).map(|_| rng.random()).collect::<Vec<EF>>();

                let statement = EqStatement::<EF>::new_hypercube(points, evals);

                if init {
                    statement.combine_hypercube::<F, true>(&mut out0, &mut sum0, challenge);
                    statement.combine_hypercube_packed::<F, true>(&mut out1, &mut sum1, challenge);
                } else {
                    statement.combine_hypercube::<F, false>(&mut out0, &mut sum0, challenge);
                    statement.combine_hypercube_packed::<F, false>(&mut out1, &mut sum1, challenge);
                    init = true;
                }

                assert_eq!(out0.0,<<EF as ExtensionField<F>>::ExtensionPacking as PackedFieldExtension<F, EF>>::to_ext_iter(
                    out1.as_slice().iter().copied(),
                )
                .collect::<Vec<_>>());
                assert_eq!(sum0, sum1);
            }
        }
    }
}
