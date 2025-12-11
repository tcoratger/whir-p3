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
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::poly::{evals::EvaluationsList, multilinear::MultilinearPoint};

/// Expands powers-of-two of table into full power table.
/// Each column in `points` is powers-of-two of a variable that should be layouted in reverse order as:
/// `[v_i^{2^{k-1}}, v_i^{2^{k-2}}, …, v_i^{2^0}]` where `k` is the height of table ie. number of variables.
/// Returns a matrix where each column is powers of `v_i`.
fn batch_pows<F: Field>(points: RowMajorMatrixView<'_, F>) -> RowMajorMatrix<F> {
    let k = points.height();
    let n = points.width();

    let mut mat = RowMajorMatrix::new(F::zero_vec(n * (1 << k)), n);
    mat.row_mut(0).fill(F::ONE);

    points.row_slices().enumerate().for_each(|(i, vars)| {
        let (lo, mut hi) = mat.split_rows_mut(1 << i);
        lo.rows().zip(hi.rows_mut()).for_each(|(lo, hi)| {
            vars.iter()
                .zip(lo.zip(hi.iter_mut()))
                .for_each(|(&var, (lo, hi))| *hi = lo * var);
        });
    });
    mat
}

/// Expands powers-of-two of table into full power table.
/// Each column in `points` is powers-of-two of a variable that should be layouted in reverse order as:
/// `[v_i^{2^{k-1}}, v_i^{2^{k-2}}, …, v_i^{2^0}]` where `k` is the height of table ie. number of variables.
/// Returns a matrix where each column is powers of `v_i` in packed form.
fn packed_batch_pows<F: Field>(points: RowMajorMatrixView<'_, F>) -> RowMajorMatrix<F::Packing> {
    let k = points.height();
    let n = points.width();
    assert_ne!(n, 0);
    let k_pack = log2_strict_usize(F::Packing::WIDTH);
    assert!(k >= k_pack);

    let (init_vars, rest_vars) = points.split_rows(k_pack);
    let mut mat = RowMajorMatrix::new(F::Packing::zero_vec(n * (1 << (k - k_pack))), n);
    if k_pack > 0 {
        init_vars
            .transpose()
            .row_slices()
            .zip(mat.values.iter_mut())
            .for_each(|(vars, packed)| {
                let point = RowMajorMatrixView::new(vars, 1);
                *packed = *F::Packing::from_slice(&batch_pows(point).values);
            });
    } else {
        mat.row_mut(0).fill(F::Packing::ONE);
    }

    rest_vars.row_slices().enumerate().for_each(|(i, vars)| {
        let (lo, mut hi) = mat.split_rows_mut(1 << i);
        lo.rows().zip(hi.rows_mut()).for_each(|(lo, hi)| {
            vars.iter()
                .zip(lo.zip(hi.iter_mut()))
                .for_each(|(&var, (lo, hi))| *hi = lo * var);
        });
    });
    mat
}

/// A batched system of `select`-based evaluation constraints for multilinear polynomials.
///
/// This struct represents a collection of evaluation constraints of the form `p(z_i) = s_i`
/// for a multilinear polynomial `p` over the Boolean hypercube `{0,1}^k`.
///
/// # The Select Function
///
/// For vectors `X, Y ∈ F^k`, the select function is defined as:
///
/// ```text
/// select(X, Y) = ∏_i (X_i · Y_i + (1 - Y_i))
/// ```
///
/// **Key Property:** When `Y ∈ {0,1}^k` is a Boolean vector and `X = pow(z)`:
///
/// ```text
/// select(pow(z), b) = z^{int(b)}
/// ```
///
/// where `pow(z) = (z, z^2, z^4, ..., z^{2^{k-1}})` and `int(b)` interprets the Boolean
/// vector `b` as an integer in binary.
///
/// **Derivation:**
/// ```text
/// select(pow(z), b) = ∏_i (z^{2^i} · b_i + (1 - b_i))
///                   = ∏_{i: b_i=1} (z^{2^i})     [since b_i ∈ {0,1}]
///                   = z^{Σ_{i: b_i=1} 2^i}
///                   = z^{int(b)}
/// ```
///
/// # Verification Claims
///
/// Each constraint `(z_i, s_i)` in this statement asserts:
///
/// ```text
/// Σ_{b ∈ {0,1}^k} P(b) · select(pow(z_i), b) = s_i
/// ```
///
/// where `P(b)` are the evaluations of the polynomial over the Boolean hypercube.
///
/// # Batching
///
/// Multiple constraints are batched using random challenge `γ` to produce:
///
/// - **Weight polynomial**: `W(b) = Σ_i γ^i · select(pow(z_i), b)`
/// - **Target sum**: `S = Σ_i γ^i · s_i`
///
/// This reduces `n` separate verification claims to a single sumcheck:
///
/// ```text
/// Σ_{b ∈ {0,1}^k} P(b) · W(b) = S
/// ```
#[derive(Clone, Debug)]
pub struct SelectStatement<F, EF> {
    /// Number of variables `k` defining the Boolean hypercube `{0,1}^k`.
    ///
    /// This determines the dimension of the multilinear polynomial space and the size
    /// of the evaluation domain (2^k points).
    num_variables: usize,

    /// Evaluation points `[z_1, z_2, ..., z_n]` where each constraint checks `p(z_i) = s_i`.
    ///
    /// Each `z_i ∈ F` is a base field element. The `pow` map will expand it to
    /// `pow(z_i) = (z_i, z_i^2, z_i^4, ..., z_i^{2^{k-1}})` for the select function.
    pub(crate) vars: Vec<F>,

    /// Expected evaluation values `[s_1, s_2, ..., s_n]` corresponding to each constraint.
    ///
    /// Each `s_i ∈ EF` is an extension field element representing the claimed evaluation
    /// of the polynomial at point `z_i`.
    evaluations: Vec<EF>,
}

impl<F: Field, EF: ExtensionField<F>> SelectStatement<F, EF> {
    /// Creates an empty select statement for polynomials over `{0,1}^k`.
    ///
    /// # Parameters
    ///
    /// - `num_variables`: The dimension `k` of the Boolean hypercube
    ///
    /// # Returns
    ///
    /// An initialized statement with no constraints, ready to accept constraints.
    #[must_use]
    pub const fn initialize(num_variables: usize) -> Self {
        Self {
            num_variables,
            vars: Vec::new(),
            evaluations: Vec::new(),
        }
    }

    /// Creates a select statement pre-populated with constraints.
    ///
    /// # Parameters
    ///
    /// - `num_variables`: The dimension `k` of the Boolean hypercube
    /// - `vars`: Evaluation points `[z_1, ..., z_n]`
    /// - `evaluations`: Expected values `[s_1, ..., s_n]`
    ///
    /// # Panics
    ///
    /// Panics if the nu
    #[must_use]
    pub const fn new(num_variables: usize, vars: Vec<F>, evaluations: Vec<EF>) -> Self {
        assert!(vars.len() == evaluations.len());
        Self {
            num_variables,
            vars,
            evaluations,
        }
    }

    /// Returns the number of variables `k` defining the polynomial space dimension.
    ///
    /// This is the dimension of the Boolean hypercube `{0,1}^k` over which polynomials
    /// are defined, containing `2^k` evaluation points.
    #[must_use]
    pub const fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// Returns `true` if no constraints have been added to this statement.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        debug_assert!(self.vars.is_empty() == self.evaluations.is_empty());
        self.vars.is_empty()
    }

    /// Returns an iterator over constraint pairs `(z_i, s_i)`.
    ///
    /// Each pair represents one evaluation constraint: `p(z_i) = s_i`.
    pub fn iter(&self) -> impl Iterator<Item = (&F, &EF)> {
        self.vars.iter().zip(self.evaluations.iter())
    }

    /// Returns the number of evaluation constraints `n` in this statement.
    #[must_use]
    pub const fn len(&self) -> usize {
        debug_assert!(self.vars.len() == self.evaluations.len());
        self.vars.len()
    }

    /// Verifies that a given polynomial satisfies all constraints in the statement.
    ///
    /// For each constraint `(z_i, s_i)`, this method interprets the evaluation table as
    /// coefficients of a univariate polynomial, evaluates it at `z_i` using Horner's method,
    /// and checks if the result equals the expected value `s_i`.
    ///
    /// For a polynomial represented by evaluations `[c_0, c_1, ..., c_{2^k-1}]`:
    ///
    /// ```text
    /// p(z) = c_0 + z(c_1 + z(c_2 + z(...)))
    /// ```
    ///
    /// This is computed right-to-left as:
    /// ```text
    /// acc = 0
    /// for i = 2^k-1 down to 0:
    ///     acc = acc * z + c_i
    /// ```
    ///
    /// # Parameters
    ///
    /// - `poly`: Evaluation table treated as univariate polynomial coefficients
    ///
    /// # Returns
    ///
    /// `true` if all constraints are satisfied, `false` otherwise.
    #[must_use]
    pub fn verify(&self, poly: &EvaluationsList<EF>) -> bool {
        self.iter().all(|(&var, &expected_eval)| {
            // Evaluate the polynomial at `var` using Horner's method.
            // This computes: p(var) = c_0 + var(c_1 + var(c_2 + ...))
            poly.iter()
                .rfold(EF::ZERO, |result, coeff| result * var + *coeff)
                == expected_eval
        })
    }

    /// Adds a single evaluation constraint `p(z) = s` to the statement.
    ///
    /// # Parameters
    ///
    /// - `var`: Evaluation point `z ∈ F`
    /// - `eval`: Expected evaluation value `s ∈ EF`
    pub fn add_constraint(&mut self, var: F, eval: EF) {
        self.vars.push(var);
        self.evaluations.push(eval);
    }

    /// Batches all constraints into a single weighted polynomial and target sum for sumcheck.
    ///
    /// Given constraints `p(z_1) = s_1, ..., p(z_n) = s_n`, this method transforms them into
    /// a single sumcheck claim using random challenge `γ`:
    ///
    /// ```text
    /// Σ_{b ∈ {0,1}^k} P(b) · W(b) = S
    /// ```
    ///
    /// where:
    /// - **Weight polynomial**: `W(b) = Σ_i γ^{i+shift} · select(pow(z_i), b)`
    /// - **Target sum**: `S = Σ_i γ^{i+shift} · s_i`
    ///
    /// The method computes `W(b)` for all `b ∈ {0,1}^k` and `S`, adding them to the
    /// provided accumulators.
    ///
    /// # Parameters
    ///
    /// - `acc_weights`: Accumulator for the weight polynomial `W(b)`. Must have `2^k` entries.
    ///   This method **adds** the batched weights to existing values.
    ///
    /// - `acc_sum`: Accumulator for the target sum `S`. This method **adds** the batched
    ///   evaluations to the existing value.
    ///
    /// - `challenge`: Random challenge `γ ∈ EF` used for batching.
    ///
    /// - `shift`: Power offset for challenge. Constraint `i` uses weight `γ^{i+shift}`.
    ///   Allows multiple statement types to use non-overlapping challenge powers.
    #[instrument(skip_all, fields(num_constraints = self.len(), num_variables = self.num_variables()))]
    pub fn combine(
        &self,
        acc_weights: &mut EvaluationsList<EF>,
        acc_sum: &mut EF,
        challenge: EF,
        shift: usize,
    ) {
        // Early return for empty statement:
        //
        // No constraints means no contribution to the batched claim.
        if self.vars.is_empty() {
            return;
        }

        // Extract dimensions for clarity.
        //
        // Number of constraints
        let n = self.len();
        // Dimension of Boolean hypercube
        let k = self.num_variables();

        // STAGE 1: Compute Power Maps
        //
        // For each evaluation point z_i, compute pow(z_i) = [z_i, z_i^2, z_i^4, ..., z_i^{2^{k-1}}]
        // by repeated squaring.
        //
        // Result: n × k matrix stored as Vec<Vec<F>>:
        //   - Outer vector: one entry per constraint (length n)
        //   - Inner vector: power sequence for that constraint (length k)
        let bin_powers = self
            .vars
            .par_iter()
            .copied()
            .map(|mut var| {
                // Generate [var, var^2, var^4, var^8, ...]
                (0..k)
                    .map(|_| {
                        let current = var;
                        // Prepare next power
                        var = var.square();
                        current
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        // STAGE 2: Build Select Matrix via Binary Tree Expansion
        //
        // We build a 2^k × n matrix where entry [b, i] = select(pow(z_i), b).
        //
        // The matrix is stored in row-major order as a flat vector of length 2^k · n.

        // Allocate the matrix: 2^k rows × n columns.
        let mut acc = F::zero_vec((1 << k) * n);

        // Initialize the first row to all ones.
        //
        // This represents the base case: select(X, 0...0) = 1 for any X.
        acc[..n].fill(F::ONE);

        // Expand the matrix one bit at a time using binary tree structure.
        //
        // After iteration i, we have 2^{i+1} rows filled.
        for i in 0..k {
            // At the start of iteration i, we have 2^i rows already computed.
            //
            // We now split the buffer into:
            //   - `lo`: the first 2^i rows (already computed)
            //   - `hi`: the next 2^i rows (to be computed as copies of `lo`)
            let num_existing_rows = 1 << i;
            let (lo, hi) = acc.split_at_mut(num_existing_rows * n);

            // Extract the i-th column of the power matrix: [pow(z_1)[i], ..., pow(z_n)[i]]
            //
            // These are the values z_1^{2^i}, z_2^{2^i}, ..., z_n^{2^i}.
            let bin_powers_col = bin_powers
                .iter()
                .map(|powers| powers[i])
                .collect::<Vec<_>>();

            // Process each row pair in parallel.
            // Each chunk represents one row of n elements.
            lo.par_chunks_mut(n)
                .zip(hi.par_chunks_mut(n))
                .for_each(|(lo_row, hi_row)| {
                    // For each column (constraint) in this row:
                    // - lo_row[j] already contains select(pow(z_j), b | bit_i = 0)
                    // - hi_row[j] should contain select(pow(z_j), b | bit_i = 1)
                    //
                    // Since select(X, Y) multiplies X[i] when Y[i] = 1, we have:
                    //   hi_row[j] = lo_row[j] * pow(z_j)[i]
                    bin_powers_col
                        .iter()
                        .zip(lo_row.iter_mut())
                        .zip(hi_row.iter_mut())
                        .for_each(|((&z_pow, lo_val), hi_val)| {
                            *hi_val = *lo_val * z_pow;
                        });
                });
        }

        // At this point, `acc` is a 2^k × n matrix where:
        //   acc[b * n + i] = select(pow(z_i), b)

        // STAGE 3: Batch with Random Challenge
        //
        // Combine the n columns of the select matrix using powers of the challenge γ.

        // Precompute the challenge powers: [γ^shift, γ^{shift+1}, ..., γ^{shift+n-1}]
        let challenges = challenge.powers().skip(shift).take(n).collect::<Vec<_>>();

        // For each hypercube point b (each row of the select matrix):
        //   W(b) += Σ_i γ^{i+shift} · select(pow(z_i), b)
        acc.par_chunks(n)
            .zip(acc_weights.0.par_iter_mut())
            .for_each(|(row, weight_out)| {
                // Compute the linear combination of this row using challenge powers.
                *weight_out += row.iter().zip(challenges.iter()).fold(
                    EF::ZERO,
                    |acc_val, (&select_val, &challenge_power)| {
                        acc_val + challenge_power * select_val
                    },
                );
            });

        // Compute the target sum: S = Σ_i γ^{i+shift} · s_i
        *acc_sum +=
            dot_product::<EF, _, _>(challenges.into_iter(), self.evaluations.iter().copied());
    }

    #[instrument(skip_all, fields(num_constraints = self.len(), num_variables = self.num_variables()))]
    pub fn combine_packed(
        &self,
        weights: &mut EvaluationsList<EF::ExtensionPacking>,
        sum: &mut EF,
        challenge: EF,
        shift: usize,
    ) {
        if self.vars.is_empty() {
            return;
        }

        let n = self.len();
        let k = self.num_variables();
        let k_pack = log2_strict_usize(F::Packing::WIDTH);
        assert!(k >= k_pack);
        assert_eq!(weights.num_variables() + k_pack, k);

        // Combine expected evaluations: S = ∑_i γ^i * s_i
        self.combine_evals(sum, challenge, shift);

        // Apply naive method if number of variables is too small for packed split method
        if k_pack * 2 > k {
            self.vars
                .iter()
                .zip(challenge.powers().skip(shift))
                .for_each(|(&var, challenge)| {
                    let pow = EF::from(var).shifted_powers(challenge).collect_n(1 << k);
                    weights
                        .0
                        .iter_mut()
                        .zip_eq(pow.chunks(F::Packing::WIDTH))
                        .for_each(|(out, chunk)| {
                            let packed = EF::ExtensionPacking::from_ext_slice(chunk);
                            *out += packed;
                        });
                });
            return;
        }

        let points = self
            .vars
            .iter()
            .map(|&var| MultilinearPoint::expand_from_univariate(var, k))
            .collect::<Vec<_>>();
        let points = MultilinearPoint::transpose(&points, true);
        let (left, right) = points.split_rows(k / 2);
        let left = packed_batch_pows(left);
        let right = batch_pows(right);

        let alphas = challenge
            .powers()
            .skip(shift)
            .take(n)
            .map(EF::ExtensionPacking::from)
            .collect::<Vec<_>>();

        weights
            .0
            .par_chunks_mut(left.height())
            .zip(right.par_row_slices())
            .for_each(|(out, right)| {
                out.iter_mut().zip(left.rows()).for_each(|(out, left)| {
                    *out += left
                        .zip(right.iter())
                        .zip(alphas.iter())
                        .map(|((left, &right), &alpha)| alpha * (left * right))
                        .sum::<EF::ExtensionPacking>();
                });
            });
    }

    /// Batches expected evaluation values into a single target sum using challenge powers.
    ///
    /// Computes and adds to `claimed_eval`:
    ///
    /// ```text
    /// S = Σ_i γ^{i+shift} · s_i
    /// ```
    ///
    /// where `s_i` are the expected evaluation values in `self.evaluations`.
    ///
    /// # Parameters
    ///
    /// - `claimed_eval`: Accumulator for the target sum. This method **adds** the batched
    ///   evaluations to the existing value.
    ///
    /// - `challenge`: Random challenge `γ ∈ EF` used for batching.
    ///
    /// - `shift`: Power offset. Constraint `i` uses weight `γ^{i+shift}`.
    pub fn combine_evals(&self, claimed_eval: &mut EF, challenge: EF, shift: usize) {
        // Compute: Σ_i γ^{i+shift} · s_i
        // This is equivalent to dot_product(evaluations, [γ^shift, γ^{shift+1}, ...])
        *claimed_eval += dot_product::<EF, _, _>(
            self.evaluations.iter().copied(),
            challenge.powers().skip(shift).take(self.len()),
        );
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::{
        PackedFieldExtension, PrimeCharacteristicRing, extension::BinomialExtensionField,
    };
    use proptest::prelude::*;
    use rand::{SeedableRng, rngs::SmallRng};

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    #[test]
    fn test_select_statement_initialize() {
        // Test that initialize creates an empty statement with correct num_variables.
        let statement = SelectStatement::<F, F>::initialize(3);

        // The statement should have 3 variables.
        assert_eq!(statement.num_variables(), 3);
        // The statement should be empty (no constraints).
        assert!(statement.is_empty());
        // The length should be 0.
        assert_eq!(statement.len(), 0);
    }

    #[test]
    fn test_select_statement_new() {
        // Test that new creates a statement with pre-populated constraints.
        let vars = vec![F::from_u64(5), F::from_u64(7)];
        let evaluations = vec![F::from_u64(10), F::from_u64(20)];

        let statement = SelectStatement::new(2, vars.clone(), evaluations.clone());

        // The statement should have 2 variables.
        assert_eq!(statement.num_variables(), 2);
        // The statement should not be empty.
        assert!(!statement.is_empty());
        // The statement should have 2 constraints.
        assert_eq!(statement.len(), 2);
        // The vars and evaluations should match.
        assert_eq!(statement.vars, vars);
        assert_eq!(statement.evaluations, evaluations);
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn test_select_statement_new_mismatched_lengths() {
        // Test that new panics when vars.len() != evaluations.len().
        let vars = vec![F::from_u64(5)];
        let evaluations = vec![F::from_u64(10), F::from_u64(20)];

        // This should panic due to length mismatch.
        let _ = SelectStatement::new(2, vars, evaluations);
    }

    #[test]
    fn test_select_statement_add_constraint() {
        // Test adding constraints one at a time.
        let mut statement = SelectStatement::<F, F>::initialize(2);

        // Initially empty.
        assert!(statement.is_empty());
        assert_eq!(statement.len(), 0);

        // Add first constraint: p(5) = 10.
        statement.add_constraint(F::from_u64(5), F::from_u64(10));
        assert!(!statement.is_empty());
        assert_eq!(statement.len(), 1);

        // Add second constraint: p(7) = 20.
        statement.add_constraint(F::from_u64(7), F::from_u64(20));
        assert_eq!(statement.len(), 2);

        // Verify the constraints were added correctly.
        let constraints: Vec<_> = statement.iter().collect();
        assert_eq!(constraints.len(), 2);
        assert_eq!(*constraints[0].0, F::from_u64(5));
        assert_eq!(*constraints[0].1, F::from_u64(10));
        assert_eq!(*constraints[1].0, F::from_u64(7));
        assert_eq!(*constraints[1].1, F::from_u64(20));
    }

    #[test]
    fn test_select_statement_verify_basic() {
        // Test the verify method with a simple polynomial.
        //
        // Create a polynomial with evaluations [c0, c1, c2, c3] over {0,1}^2.
        let c0 = F::from_u64(1);
        let c1 = F::from_u64(2);
        let c2 = F::from_u64(3);
        let c3 = F::from_u64(4);
        let poly = EvaluationsList::new(vec![c0, c1, c2, c3]);

        // Create a statement with k=2 variables.
        let k = 2;
        let mut statement = SelectStatement::<F, F>::initialize(k);

        // The polynomial evaluations [c0, c1, c2, c3] can be interpreted as a univariate polynomial:
        // p(z) = c0 + c1*z + c2*z^2 + c3*z^3
        //
        // Test p(0) = c0 = 1.
        let z0 = F::ZERO;
        let eval0 = c0;
        statement.add_constraint(z0, eval0);
        assert!(statement.verify(&poly));

        // Test p(1) = c0 + c1 + c2 + c3
        let mut statement2 = SelectStatement::<F, F>::initialize(k);
        let z1 = F::ONE;
        let eval1 = c0 + c1 + c2 + c3;
        statement2.add_constraint(z1, eval1);
        assert!(statement2.verify(&poly));

        // Test p(2) = c0 + c1*2 + c2*4 + c3*8
        let mut statement3 = SelectStatement::<F, F>::initialize(k);
        let z2 = F::from_u64(2);
        let eval2 = c0 + c1 * z2 + c2 * z2 * z2 + c3 * z2 * z2 * z2;
        statement3.add_constraint(z2, eval2);
        assert!(statement3.verify(&poly));

        // Test a failing verification: p(1) = wrong_eval
        let mut statement4 = SelectStatement::<F, F>::initialize(k);
        let wrong_eval = F::from_u64(56765);
        statement4.add_constraint(z1, wrong_eval);
        assert!(!statement4.verify(&poly));
    }

    #[test]
    fn test_select_statement_combine_single_constraint() {
        // Test combining a single constraint.
        //
        // For k=2 variables, we have a 2^2 = 4-point domain.
        let k = 2;
        let domain_size = 1 << k;

        // Create a statement with one constraint: p(z) = s.
        let mut statement = SelectStatement::<F, F>::initialize(k);
        let z = F::from_u64(5);
        let s = F::from_u64(100);
        statement.add_constraint(z, s);

        // The challenge γ is unused for a single constraint (it would multiply by γ^0 = 1).
        let gamma = F::from_u64(2);
        let shift = 0;

        // Initialize accumulators.
        let mut acc_weights = EvaluationsList::zero(k);
        let mut acc_sum = F::ZERO;

        // Combine the constraints.
        statement.combine(&mut acc_weights, &mut acc_sum, gamma, shift);

        // The target sum should be S = γ^0 · s = 1 · s = s.
        let expected_sum = s;
        assert_eq!(acc_sum, expected_sum);

        // The weight polynomial should be W(b) = select(pow(z), b) for all b ∈ {0,1}^k.
        //
        // Verify each entry manually using the property: select(pow(z), b) = z^b.
        for b in 0..domain_size {
            let expected_weight = z.exp_u64(b as u64);
            assert_eq!(
                acc_weights.as_slice()[b],
                expected_weight,
                "Weight mismatch at index {b}"
            );
        }
    }

    #[test]
    fn test_select_statement_combine_multiple_constraints() {
        // Test combining multiple constraints with batching.
        //
        // For k=2 variables, we have a 2^2 = 4-point domain.
        let k = 2;
        let domain_size = 1 << k;

        // Create a statement with two constraints:
        // - Constraint 0: p(z0) = s0
        // - Constraint 1: p(z1) = s1
        let mut statement = SelectStatement::<F, F>::initialize(k);
        let z0 = F::from_u64(3);
        let s0 = F::from_u64(10);
        let z1 = F::from_u64(7);
        let s1 = F::from_u64(20);
        statement.add_constraint(z0, s0);
        statement.add_constraint(z1, s1);

        // Use challenge γ for batching.
        let gamma = F::from_u64(2);
        let shift = 0;

        // Initialize accumulators.
        let mut acc_weights = EvaluationsList::zero(k);
        let mut acc_sum = F::ZERO;

        // Combine the constraints.
        statement.combine(&mut acc_weights, &mut acc_sum, gamma, shift);

        // The target sum should be:
        // S = γ^0 · s0 + γ^1 · s1 = 1·s0 + γ·s1 = s0 + gamma*s1.
        let expected_sum = s0 + gamma * s1;
        assert_eq!(acc_sum, expected_sum);

        // The weight polynomial should be:
        // W(b) = γ^0 · select(pow(z0), b) + γ^1 · select(pow(z1), b)
        //      = select(pow(z0), b) + gamma · select(pow(z1), b)
        // Using the property: select(pow(z), b) = z^b.
        for b in 0..domain_size {
            let weight0 = z0.exp_u64(b as u64);
            let weight1 = z1.exp_u64(b as u64);
            let expected_weight = weight0 + gamma * weight1;
            assert_eq!(
                acc_weights.as_slice()[b],
                expected_weight,
                "Weight mismatch at index {b}"
            );
        }
    }

    #[test]
    fn test_select_statement_combine_with_shift() {
        // Test combining constraints with a non-zero shift parameter.
        //
        // The shift parameter allows multiple statement types to use non-overlapping
        // challenge powers for batching.
        let k = 1;
        let domain_size = 1 << k;

        // Create a statement with one constraint: p(z) = s.
        let mut statement = SelectStatement::<F, F>::initialize(k);
        let z = F::from_u64(5);
        let s = F::from_u64(100);
        statement.add_constraint(z, s);

        // Use challenge γ with shift.
        // This means the constraint will be weighted by γ^{0+shift} = γ^shift.
        let gamma = F::from_u64(2);
        let shift = 3;

        // Initialize accumulators.
        let mut acc_weights = EvaluationsList::zero(k);
        let mut acc_sum = F::ZERO;

        // Combine the constraints.
        statement.combine(&mut acc_weights, &mut acc_sum, gamma, shift);

        // The target sum should be S = γ^shift · s.
        let gamma_to_shift = gamma.exp_u64(shift as u64);
        let expected_sum = gamma_to_shift * s;
        assert_eq!(acc_sum, expected_sum);

        // The weight polynomial should be W(b) = γ^shift · select(pow(z), b).
        // Using the property: select(pow(z), b) = z^b.
        for b in 0..domain_size {
            let select_val = z.exp_u64(b as u64);
            let expected_weight = gamma_to_shift * select_val;
            assert_eq!(
                acc_weights.as_slice()[b],
                expected_weight,
                "Weight mismatch at index {b}"
            );
        }
    }

    #[test]
    fn test_select_statement_combine_empty() {
        // Test that combining an empty statement does nothing.
        let k = 2;
        let statement = SelectStatement::<F, F>::initialize(k);

        // Initialize accumulators with non-zero values.
        let w0 = F::from_u64(1);
        let w1 = F::from_u64(2);
        let w2 = F::from_u64(3);
        let w3 = F::from_u64(4);
        let mut acc_weights = EvaluationsList::new(vec![w0, w1, w2, w3]);
        let initial_sum = F::from_u64(99);
        let mut acc_sum = initial_sum;

        // Store original values.
        let original_weights = acc_weights.clone();
        let original_sum = acc_sum;

        // Combine the empty statement.
        let gamma = F::from_u64(2);
        let shift = 0;
        statement.combine(&mut acc_weights, &mut acc_sum, gamma, shift);

        // The accumulators should remain unchanged.
        assert_eq!(acc_weights, original_weights);
        assert_eq!(acc_sum, original_sum);
    }

    #[test]
    fn test_select_statement_combine_accumulation() {
        // Test that combine properly accumulates (adds to) existing values.
        //
        // This is important for batching multiple statements together.
        let k = 1;
        let domain_size = 1 << k;

        // Create first statement with constraint p(z1) = s1.
        let mut statement1 = SelectStatement::<F, F>::initialize(k);
        let z1 = F::from_u64(2);
        let s1 = F::from_u64(5);
        statement1.add_constraint(z1, s1);

        // Create second statement with constraint p(z2) = s2.
        let mut statement2 = SelectStatement::<F, F>::initialize(k);
        let z2 = F::from_u64(3);
        let s2 = F::from_u64(7);
        statement2.add_constraint(z2, s2);

        let gamma = F::from_u64(2);
        let shift = 0;

        // Initialize accumulators.
        let mut acc_weights = EvaluationsList::zero(k);
        let mut acc_sum = F::ZERO;

        // Combine first statement.
        statement1.combine(&mut acc_weights, &mut acc_sum, gamma, shift);

        // Store intermediate values.
        let intermediate_weights = acc_weights.clone();
        let intermediate_sum = acc_sum;

        // Combine second statement (should add to existing values).
        statement2.combine(&mut acc_weights, &mut acc_sum, gamma, shift);

        // The accumulated sum should be intermediate_sum + s2.
        let expected_sum = intermediate_sum + s2;
        assert_eq!(acc_sum, expected_sum);

        // The accumulated weights should be the sum of both select functions.
        // Using the property: select(pow(z), b) = z^b.
        for b in 0..domain_size {
            let weight2 = z2.exp_u64(b as u64);
            let expected_weight = intermediate_weights.as_slice()[b] + weight2;
            assert_eq!(
                acc_weights.as_slice()[b],
                expected_weight,
                "Accumulated weight mismatch at index {b}"
            );
        }
    }

    #[test]
    fn test_select_statement_combine_evals() {
        // Test the combine_evals method.
        let k = 2;

        // Create a statement with two constraints.
        let mut statement = SelectStatement::<F, F>::initialize(k);
        let s0 = F::from_u64(10);
        let s1 = F::from_u64(20);
        statement.add_constraint(F::from_u64(3), s0);
        statement.add_constraint(F::from_u64(7), s1);

        let gamma = F::from_u64(2);
        let shift = 1;

        // Test combine_evals.
        let mut claimed_eval = F::ZERO;
        statement.combine_evals(&mut claimed_eval, gamma, shift);

        // Expected: S = γ^{shift} · s0 + γ^{shift+1} · s1 = γ^1·s0 + γ^2·s1.
        let gamma_1 = gamma.exp_u64(shift as u64);
        let gamma_2 = gamma.exp_u64((shift + 1) as u64);
        let expected = gamma_1 * s0 + gamma_2 * s1;
        assert_eq!(claimed_eval, expected);
    }

    #[test]
    fn test_select_statement_combine_evals_accumulation() {
        // Test that combine_evals properly accumulates.
        let k = 1;

        let mut statement = SelectStatement::<F, F>::initialize(k);
        let s = F::from_u64(10);
        statement.add_constraint(F::from_u64(5), s);

        let gamma = F::from_u64(3);
        let shift = 0;

        // Start with a non-zero claimed_eval.
        let initial_eval = F::from_u64(42);
        let mut claimed_eval = initial_eval;

        // Combine evals should add to the existing value.
        statement.combine_evals(&mut claimed_eval, gamma, shift);

        // Expected: initial_eval + γ^0 · s = initial_eval + 1·s = initial_eval + s.
        let expected = initial_eval + s;
        assert_eq!(claimed_eval, expected);
    }

    #[test]
    fn test_select_combine_consistency_with_verify() {
        // Test that combine and verify are consistent.
        //
        // If we create a polynomial that satisfies the constraints, then:
        // 1. verify() should return true
        // 2. The combined weights should correctly compute the polynomial evaluations
        let k = 2;
        let domain_size = 1 << k;

        // Create a simple polynomial: evaluations [c0, c1, c2, c3].
        let c0 = F::from_u64(1);
        let c1 = F::from_u64(2);
        let c2 = F::from_u64(3);
        let c3 = F::from_u64(4);
        let poly = EvaluationsList::new(vec![c0, c1, c2, c3]);

        // Create constraints that match the polynomial.
        // Using Horner evaluation: p(z) = c0 + c1*z + c2*z^2 + c3*z^3.
        let mut statement = SelectStatement::<F, F>::initialize(k);

        // Evaluate p(z) at z using Horner's method.
        let z = F::from_u64(2);
        let expected_eval = poly
            .iter()
            .rfold(F::ZERO, |result, &coeff| result * z + coeff);
        statement.add_constraint(z, expected_eval);

        // Verify should pass.
        assert!(statement.verify(&poly));

        // Now combine and check that the weight polynomial correctly represents
        // the select function.
        let gamma = F::from_u64(3);
        let shift = 0;
        let mut acc_weights = EvaluationsList::zero(k);
        let mut acc_sum = F::ZERO;
        statement.combine(&mut acc_weights, &mut acc_sum, gamma, shift);

        // The sum should match the expected evaluation.
        assert_eq!(acc_sum, expected_eval);

        // The weight polynomial should satisfy:
        // Σ_{b ∈ {0,1}^k} poly(b) · W(b) = expected_eval
        let mut computed_sum = F::ZERO;
        for b in 0..domain_size {
            computed_sum += poly.as_slice()[b] * acc_weights.as_slice()[b];
        }
        assert_eq!(computed_sum, expected_eval);
    }

    proptest! {
        #[test]
        fn prop_select_statement_combine_sum(
            // Number of variables (1 to 4 for reasonable test size).
            k in 1usize..=4,
            // Number of constraints (1 to 5).
            num_constraints in 1usize..=5,
            // Random evaluation points (avoiding 0 for better coverage).
            // Generate exactly num_constraints values.
            z_values in prop::collection::vec(1u32..100, 1..=5),
            // Random expected evaluations.
            s_values in prop::collection::vec(0u32..100, 1..=5),
            // Random challenge.
            challenge in 1u32..50,
        ) {
            // Ensure we have enough values for the test.
            let actual_num_constraints = num_constraints.min(z_values.len()).min(s_values.len());
            if actual_num_constraints == 0 {
                return Ok(());
            }

            let z_values = &z_values[..actual_num_constraints];
            let s_values = &s_values[..actual_num_constraints];

            // Create statement with random constraints.
            let mut statement = SelectStatement::<F, F>::initialize(k);
            for (&z, &s) in z_values.iter().zip(s_values.iter()) {
                statement.add_constraint(F::from_u32(z), F::from_u32(s));
            }

            let gamma = F::from_u32(challenge);

            // Combine with shift=0.
            let mut acc_weights = EvaluationsList::zero(k);
            let mut acc_sum = F::ZERO;
            statement.combine(&mut acc_weights, &mut acc_sum, gamma, 0);

            // Compute expected sum manually: S = Σ_i γ^i · s_i.
            let mut expected_sum = F::ZERO;
            for (i, &s) in s_values.iter().enumerate() {
                expected_sum += gamma.exp_u64(i as u64) * F::from_u32(s);
            }

            prop_assert_eq!(acc_sum, expected_sum);
        }
    }

    proptest! {
        #[test]
        fn prop_select_statement_verify(
            // Polynomial evaluations (2^k values for k=3).
            poly_evals in prop::collection::vec(0u32..100, 8),
            // Evaluation point (avoiding 0 for better coverage).
            z in 1u32..50,
        ) {
            let k = 3; // Fixed k=3 gives 2^3 = 8 evaluations.
            let poly = EvaluationsList::new(poly_evals.into_iter().map(F::from_u32).collect());

            // Compute expected evaluation using Horner's method.
            let z_field = F::from_u32(z);
            let expected_eval = poly
                .iter()
                .rfold(F::ZERO, |result, &coeff| result * z_field + coeff);

            // Create statement with correct constraint.
            let mut statement = SelectStatement::<F, F>::initialize(k);
            statement.add_constraint(z_field, expected_eval);

            // Verify should pass.
            prop_assert!(statement.verify(&poly));

            // Add a wrong constraint (off by 1, unless it wraps to same value).
            let wrong_eval = expected_eval + F::ONE;
            if wrong_eval != expected_eval {
                statement.add_constraint(z_field, wrong_eval);
                // Verify should fail now.
                prop_assert!(!statement.verify(&poly));
            }
        }
    }

    proptest! {
        #[test]
        fn prop_combine_evals_consistency(
            // Number of constraints.
            num_constraints in 1usize..=5,
            // Random evaluations.
            s_values in prop::collection::vec(0u32..100, 1..=5),
            // Random challenge.
            challenge in 1u32..50,
            // Random shift.
            shift in 0usize..3,
        ) {
            let s_values = &s_values[..num_constraints.min(s_values.len())];

            // Create statement with arbitrary z values (they don't matter for this test).
            let mut statement = SelectStatement::<F, F>::initialize(2);
            for &s in s_values {
                statement.add_constraint(F::from_u32(1), F::from_u32(s));
            }

            let gamma = F::from_u32(challenge);

            // Method 1: Use combine_evals.
            let mut claimed_eval1 = F::ZERO;
            statement.combine_evals(&mut claimed_eval1, gamma, shift);

            // Method 2: Compute manually.
            let mut claimed_eval2 = F::ZERO;
            for (i, &s) in s_values.iter().enumerate() {
                claimed_eval2 += gamma.exp_u64((i + shift) as u64) * F::from_u32(s);
            }

            prop_assert_eq!(claimed_eval1, claimed_eval2);
        }
    }

    #[test]
    fn test_packed_combine() {
        type PackedExt = <EF as ExtensionField<F>>::ExtensionPacking;

        let mut rng = SmallRng::seed_from_u64(1);
        let challenge: EF = rng.random();
        let k_pack = log2_strict_usize(<F as Field>::Packing::WIDTH);

        let mut shift = 0;
        for k in k_pack..10 {
            let mut out0 = EvaluationsList::zero(k);
            let mut out1 = EvaluationsList::<PackedExt>::zero(k - k_pack);
            let mut sum0 = EF::ZERO;
            let mut sum1 = EF::ZERO;
            for n in [1, 2, 10, 11] {
                let vars = (0..n).map(|_| rng.random()).collect::<Vec<F>>();
                let evals = (0..n).map(|_| rng.random()).collect::<Vec<EF>>();

                let statement = SelectStatement::<F, EF>::new(k, vars, evals);

                statement.combine(&mut out0, &mut sum0, challenge, shift);
                statement.combine_packed(&mut out1, &mut sum1, challenge, shift);
                shift += statement.len();

                assert_eq!(out0.0,<<EF as ExtensionField<F>>::ExtensionPacking as PackedFieldExtension<F, EF>>::to_ext_iter(
                    out1.as_slice().iter().copied(),
                )
                .collect::<Vec<_>>());
                assert_eq!(sum0, sum1);
            }
        }
    }
}
