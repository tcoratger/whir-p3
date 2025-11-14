use alloc::{vec, vec::Vec};

use p3_field::{ExtensionField, Field, dot_product};
use p3_maybe_rayon::prelude::*;
use tracing::instrument;

use crate::poly::evals::EvaluationsList;

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
    pub fn combine<Base>(
        &self,
        acc_weights: &mut EvaluationsList<EF>,
        acc_sum: &mut EF,
        challenge: EF,
        shift: usize,
    ) where
        Base: Field,
        F: ExtensionField<Base>,
    {
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
        acc[..n].copy_from_slice(&vec![F::ONE; n]);

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
