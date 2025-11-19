use alloc::{vec, vec::Vec};

use itertools::Itertools;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_interpolation::interpolate_subgroup;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::eq_batch::{eval_eq_base_batch, eval_eq_batch};
use serde::{Deserialize, Serialize};
use tracing::instrument;

use super::{coeffs::CoefficientList, multilinear::MultilinearPoint, wavelet::Radix2WaveletKernel};
use crate::{
    constant::MLE_RECURSION_THRESHOLD, sumcheck::sumcheck_small_value::SvoAccumulators,
    utils::uninitialized_vec,
};

const PARALLEL_THRESHOLD: usize = 4096;

/// Represents a multilinear polynomial `f` in `n` variables, stored by its evaluations
/// over the boolean hypercube `{0,1}^n`.
///
/// The inner vector stores function evaluations at points of the hypercube in lexicographic
/// order. The number of variables `n` is inferred from the length of this vector, where
/// `self.len() = 2^n`.
#[allow(clippy::unsafe_derive_deserialize)]
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
#[must_use]
pub struct EvaluationsList<F>(pub(crate) Vec<F>);

impl<F> EvaluationsList<F>
where
    F: Field,
{
    /// Constructs an `EvaluationsList` from a vector of evaluations.
    ///
    /// The `evals` vector must adhere to the following constraints:
    /// - Its length must be a power of two, as it represents evaluations over a
    ///   binary hypercube of some dimension `n`.
    /// - The evaluations must be ordered lexicographically corresponding to the points
    ///   on the hypercube.
    ///
    /// # Panics
    /// Panics if `evals.len()` is not a power of two.
    #[inline]
    pub const fn new(evals: Vec<F>) -> Self {
        assert!(
            evals.len().is_power_of_two(),
            "Evaluation list length must be a power of two."
        );

        Self(evals)
    }

    /// Given a number of points initializes a new zero polynomial
    pub fn zero(num_evals: usize) -> Self {
        Self(F::zero_vec(1 << num_evals))
    }

    /// Given a point `P` (as a slice), compute the evaluation vector of the equality
    /// function `eq(P, X)` for all points `X` in the boolean hypercube, scaled by a value.
    ///
    /// ## Arguments
    /// * `point`: A slice of field elements representing the point.
    /// * `value`: A scalar value to multiply all evaluations by.
    ///
    /// ## Returns
    /// An `EvaluationsList` containing `value * eq(point, X)` for all `X` in `{0,1}^n`.
    #[inline]
    pub fn new_from_point(point: &[F], value: F) -> Self {
        let n = point.len();
        if n == 0 {
            return Self(vec![value]);
        }
        let mut evals = F::zero_vec(1 << n);
        eval_eq_batch::<_, _, false>(RowMajorMatrixView::new_col(point), &mut evals, &[value]);
        Self(evals)
    }

    /// Evaluates the polynomial as a constant.
    /// This is only valid for constant polynomials (i.e., when `num_variables` is 0).
    ///
    /// Returns None in other cases.
    ///
    /// # Panics
    /// Panics if `num_variables` is not 0.
    #[must_use]
    #[inline]
    pub fn as_constant(&self) -> Option<F> {
        (self.num_evals() == 1).then_some(self.0[0])
    }

    /// Given multiple multilinear points, compute the evaluation vectors of the equality functions
    /// and add them to the current evaluation vector in a batch.
    #[inline]
    pub fn accumulate_batch(&mut self, points: &[MultilinearPoint<F>], values: &[F]) {
        assert_eq!(points.len(), values.len());
        if points.is_empty() {
            return;
        }
        let num_vars = self.num_variables();
        assert_eq!(num_vars, points[0].num_variables());

        // Convert points to a matrix where each column is a point
        let point_data: Vec<_> = (0..num_vars)
            .flat_map(|var_idx| points.iter().map(move |p| p.as_slice()[var_idx]))
            .collect();
        let points_matrix = RowMajorMatrixView::new(&point_data, points.len());

        eval_eq_batch::<_, _, true>(points_matrix, &mut self.0, values);
    }

    /// Given multiple multilinear points in a base field, compute the evaluation vectors of the equality functions
    /// and add them to the current evaluation vector in a batch.
    #[inline]
    pub fn accumulate_base_batch<BF: Field>(
        &mut self,
        points: &[MultilinearPoint<BF>],
        values: &[F],
    ) where
        F: ExtensionField<BF>,
    {
        assert_eq!(points.len(), values.len());
        if points.is_empty() {
            return;
        }
        let num_vars = self.num_variables();
        assert_eq!(num_vars, points[0].num_variables());

        // Convert points to a matrix where each column is a point
        let point_data: Vec<_> = (0..num_vars)
            .flat_map(|var_idx| points.iter().map(move |p| p.as_slice()[var_idx]))
            .collect();
        let points_matrix = RowMajorMatrixView::new(&point_data, points.len());

        eval_eq_base_batch::<_, _, true>(points_matrix, &mut self.0, values);
    }

    /// Returns the total number of stored evaluations.
    #[must_use]
    #[inline]
    pub const fn num_evals(&self) -> usize {
        self.0.len()
    }

    /// Returns the number of variables in the multilinear polynomial.
    #[must_use]
    #[inline]
    pub const fn num_variables(&self) -> usize {
        // Safety: The length is guaranteed to be a power of two.
        self.0.len().ilog2() as usize
    }

    /// Evaluates the multilinear polynomial at `point ∈ EF^n`.
    ///
    /// Computes
    /// ```text
    ///     f(point) = \sum_{x ∈ {0,1}^n} eq(x, point) * f(x),
    /// ```
    /// where
    /// ```text
    ///     eq(x, point) = \prod_{i=1}^{n} (1 - p_i + 2 p_i x_i).
    /// ```
    #[must_use]
    #[inline]
    pub fn evaluate_hypercube<EF: ExtensionField<F>>(&self, point: &MultilinearPoint<EF>) -> EF {
        eval_multilinear(&self.0, point)
    }

    /// Folds a multilinear polynomial stored in evaluation form along the last `k` variables.
    ///
    /// Given evaluations `f: {0,1}^n → F`, this method returns a new evaluation list `g` such that:
    ///
    /// ```text
    ///     g(x_0, ..., x_{n-k-1}) = f(x_0, ..., x_{n-k-1}, r_0, ..., r_{k-1})
    /// ```
    ///
    /// # Arguments
    /// - `folding_randomness`: The extension-field values to substitute for the last `k` variables.
    ///
    /// # Returns
    /// - A new `EvaluationsList<EF>` representing the folded function over the remaining `n - k` variables.
    ///
    /// # Panics
    /// - If the evaluation list is not sized `2^n` for some `n`.
    #[instrument(skip_all)]
    #[inline]
    pub fn fold<EF>(&self, folding_randomness: &MultilinearPoint<EF>) -> EvaluationsList<EF>
    where
        EF: ExtensionField<F>,
    {
        assert_ne!(folding_randomness.num_variables(), 0);
        assert!(folding_randomness.num_variables() <= self.num_variables());
        let mut poly = self.compress_ext(folding_randomness.as_slice()[0]);
        for &r in &folding_randomness.as_slice()[1..] {
            poly.compress(r);
        }
        poly
    }

    /// Create a matrix representation of the evaluation list.
    #[inline]
    #[must_use]
    pub fn into_mat(self, width: usize) -> RowMajorMatrix<F> {
        RowMajorMatrix::new(self.0, width)
    }

    /// Returns a reference to the underlying slice of evaluations.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[F] {
        &self.0
    }

    /// Returns an iterator over the evaluations.
    #[inline]
    pub fn iter(&self) -> core::slice::Iter<'_, F> {
        self.0.iter()
    }

    /// Convert from a list of evaluations to a list of multilinear coefficients.
    #[must_use]
    #[inline]
    #[instrument(skip_all)]
    pub fn to_coefficients<B: Field>(self) -> CoefficientList<F>
    where
        F: ExtensionField<B>,
    {
        let kernel = Radix2WaveletKernel::<B>::default();
        let evals = kernel.inverse_wavelet_transform_algebra(self.0);
        CoefficientList::new(evals)
    }

    /// Compresses the evaluation list by folding the **first** variable ($X_1$) with a challenge.
    ///
    /// This function is the core operation for the standard rounds of a sumcheck prover,
    /// where variables are folded one by one in lexicographical order.
    ///
    /// ## Mathematical Formula
    ///
    /// Given a polynomial $p(X_1, \ldots, X_n)$ represented by its evaluations, this
    /// function computes the evaluations of the folded
    /// polynomial $p'(X_2, \ldots, X_n) = p(r, X_2, \ldots, X_n)$.
    ///
    /// It uses the multilinear extension formula for the first variable:
    ///
    /// ```text
    /// p(r, x') = p(0, x') + r \cdot (p(1, x') - p(0, x'))
    /// ```
    ///
    /// where $x' = (x_2, \ldots, x_n)$ represents all other variables.
    ///
    /// ## Memory Access Pattern
    ///
    /// This function relies on the **lexicographical order** of the evaluation list.
    /// - The first half of the slice contains all evaluations where $X_1 = 0$,
    /// - The second half contains all evaluations where $X_1 = 1$.
    ///
    /// ```text
    /// Before:
    /// [ p(0, 0..0), p(0, 0..1), ..., p(0, 1..1) | p(1, 0..0), p(1, 0..1), ..., p(1, 1..1) ]
    ///  └────────── Left Half (p(0, x')) ──────┘   └────────── Right Half (p(1, x')) ────┘
    ///
    /// After: (Computed in-place into the left half)
    /// [ p(r, 0..0), p(r, 0..1), ..., p(r, 1..1) ]
    ///   └───────── Folded result ─────────────┘
    /// ```
    ///
    /// The function computes `result[i] = left[i] + r * (right[i] - left[i])` for
    /// all `i` in the first half, and then truncates the list.
    #[inline]
    pub fn compress(&mut self, r: F) {
        assert_ne!(self.num_variables(), 0);
        let num_evals = self.num_evals();
        let mid = num_evals / 2;

        // Evaluations at `a_i` and `a_{i + n/2}` slots are folded with `r` into `a_i` slot
        let (p0, p1) = self.0.split_at_mut(mid);
        if num_evals >= PARALLEL_THRESHOLD {
            p0.par_iter_mut()
                .zip(p1.par_iter())
                .for_each(|(a0, &a1)| *a0 += r * (a1 - *a0));
        } else {
            p0.iter_mut()
                .zip(p1.iter())
                .for_each(|(a0, &a1)| *a0 += r * (a1 - *a0));
        }
        // Free higher part of the evaluations
        self.0.truncate(mid);
    }

    /// Folds a list of evaluations from a base field `F` into an extension field `EF`.
    ///
    /// ## Arguments
    /// * `r`: A value `r` from the extension field `EF`, used as the random challenge for folding.
    ///
    /// ## Returns
    /// A new `EvaluationsList<EF>` containing the compressed evaluations in the extension field.
    ///
    /// The compression is achieved by applying the following formula to pairs of evaluations:
    /// ```text
    ///     p'(X_2, ..., X_n) = (p(1, X_2, ..., X_n) - p(0, X_2, ..., X_n)) \cdot r + p(0, X_2, ..., X_n)
    /// ```
    #[inline]
    #[instrument(skip_all)]
    pub fn compress_ext<EF: ExtensionField<F>>(&self, r: EF) -> EvaluationsList<EF> {
        assert_ne!(self.num_variables(), 0);
        let num_evals = self.num_evals();
        let mid = num_evals / 2;
        // Evaluations at `a_i` and `a_{i + n/2}` slots are folded with `r`
        let (p0, p1) = self.0.split_at(mid);
        // Create new EvaluationsList in the extension field
        EvaluationsList(if num_evals >= PARALLEL_THRESHOLD {
            p0.par_iter()
                .zip(p1.par_iter())
                .map(|(&a0, &a1)| r * (a1 - a0) + a0)
                .collect()
        } else {
            p0.iter()
                .zip(p1.iter())
                .map(|(&a0, &a1)| r * (a1 - a0) + a0)
                .collect()
        })
    }

    /// Evaluates a polynomial over the mixed domain `D × H^j` at an arbitrary point.
    ///
    /// # Purpose
    ///
    /// This method supports the **univariate skip** optimization in sumcheck protocol.
    ///
    /// Instead of evaluating over the full Boolean hypercube `{0,1}^n`, we use a mixed domain
    /// where:
    /// - `k` variables live on a multiplicative subgroup `D`,
    /// - `j = n - k` variables remain on the Boolean hypercube `H^j`.
    ///
    /// # How It Works
    ///
    /// The evaluation proceeds in two stages:
    ///
    /// **Stage 1: Univariate Interpolation**
    ///
    /// The `k` skipped variables correspond to a multiplicative subgroup `D` of size `2^k`.
    ///
    /// We treat the evaluation table as a `2^k × 2^j` grid with `2^k` rows and `2^j` columns.
    ///
    /// Each column represents evaluations of a univariate polynomial over the subgroup `D`.
    ///
    /// We interpolate each column polynomial and evaluate it at the given point `x`.
    ///
    /// This collapses all `2^k` rows into a single row of `2^j` values.
    ///
    /// **Stage 2: Multilinear Interpolation**
    ///
    /// The folded row from Stage 1 represents evaluations of a multilinear polynomial
    /// in the remaining `j` variables.
    ///
    /// We evaluate this polynomial at the remaining coordinates `y` using standard
    /// multilinear interpolation.
    ///
    /// # Input Structure
    ///
    /// The evaluation point `point` must have exactly `j + 1` coordinates:
    ///
    /// - `point[0]` = `x`: Evaluation point for the `k` skipped variables,
    /// - `point[1..]` = `y`: Evaluation points for the `j` hypercube variables.
    ///
    /// # Memory Layout
    ///
    /// The evaluation table is stored as a `2^k × 2^j` grid in row-major order.
    ///
    /// - Rows correspond to subgroup elements.
    /// - Columns correspond to hypercube corners.
    ///
    /// This is lexicographical (natural) order, not bit-reversed order.
    /// TODO: we should check in the future if we need bit reversed order.
    ///
    /// # Parameters
    ///
    /// - `point`: The evaluation point with `j + 1` coordinates as described above
    /// - `log_skip_size`: Number of variables to skip (must satisfy `0 < log_skip_size ≤ n`)
    #[must_use]
    #[inline]
    pub fn evaluate_with_univariate_skip<EF>(
        &self,
        point: &MultilinearPoint<EF>,
        log_skip_size: usize,
    ) -> EF
    where
        F: TwoAdicField,
        EF: TwoAdicField + ExtensionField<F>,
    {
        // Get the total number of variables n in the polynomial.
        let n = self.num_variables();

        // - Validate that log_skip_size is in the valid range (0, n].
        // - Using log_skip_size = 0 is meaningless; use standard evaluate() instead.
        assert!(
            log_skip_size > 0,
            "log_skip_size must be greater than 0 (got {log_skip_size}). For log_skip_size=0, use the standard evaluate() method."
        );
        assert!(
            log_skip_size <= n,
            "log_skip_size ({log_skip_size}) must not exceed num_variables ({n})"
        );

        // Compute j = n - k, the number of remaining hypercube variables.
        let num_hypercube_vars = n - log_skip_size;

        // Verify the evaluation point has the correct structure: j + 1 coordinates.
        // - The first coordinate is for the skipped variables,
        // - The rest for the hypercube.
        assert_eq!(
            point.num_variables(),
            num_hypercube_vars + 1,
            "Point must have {} coordinates for univariate skip with log_skip_size={} and n={}, but has {}",
            num_hypercube_vars + 1,
            log_skip_size,
            n,
            point.num_variables()
        );

        // Split the evaluation point into two parts:
        //   - x  = point[0]     : evaluation point for the k skipped variables (univariate)
        //   - y  = point[1..]   : evaluation points for the j hypercube variables (multilinear)
        let x = point[0];
        let y_coords = &point.as_slice()[1..];
        let y_point = MultilinearPoint::new(y_coords.to_vec());

        // STAGE 1: Univariate Interpolation
        //
        // We have 2^n evaluations organized as a 2^k × 2^j grid.
        // - Rows index the subgroup D (size 2^k).
        // - Columns index the hypercube H^j (size 2^j).

        // Compute the number of columns in the grid: 2^j.
        let hypercube_size = 1 << num_hypercube_vars;

        // Create a matrix view of the evaluation table.
        // This interprets the flat array as a 2^k × 2^j matrix in row-major order.
        let mat = RowMajorMatrixView::new(self.as_slice(), hypercube_size);

        // Interpolate each of the 2^j columns over the subgroup D and evaluate at x.
        //
        // Each column is a univariate polynomial of degree < 2^k evaluated at the subgroup.
        // After interpolation at x, we get a single row of 2^j values.
        //
        // TODO: Depending on what is slower, it may be worth investigating whether it's better
        // to perform the univariate interpolation first (current approach) or last (after the
        // hypercube evaluation). The optimal order may depend on the relative sizes of k and j.
        let folded_evals: Vec<EF> = interpolate_subgroup(&mat, x);

        // STAGE 2: Multilinear Interpolation
        //
        // The folded row represents evaluations of a j-variate multilinear polynomial
        // over the Boolean hypercube {0,1}^j.
        //
        // We now evaluate this polynomial at the coordinates y.

        // Wrap the folded evaluations as a new list of evaluations.
        let final_poly = EvaluationsList::new(folded_evals);

        // Perform standard multilinear interpolation at y and return the result.
        final_poly.evaluate_hypercube(&y_point)
    }

    /// Computes precomputation accumulators for the Small-Value Optimization (SVO).
    ///
    /// Speeding Up Sum-Check Proving: https://eprint.iacr.org/2025/1117.pdf
    ///
    /// This is the core of the SVO prover strategy (Procedure 9, Page 37).
    ///
    /// It precomputes all sum-check round polynomials for the first `N` rounds simultaneously,
    /// exploiting the fact that polynomial values are "small".
    ///
    /// # What This Does
    ///
    /// In standard sum-check, proving each round costs `O(2^n)` large-field multiplications.
    /// This function precomputes `N` rounds at once, replacing most large-field operations
    /// with cheaper small-field operations.
    ///
    /// When `g(x) = E(x) · P(x)` where:
    /// - `E(x)` is the equality polynomial (large field values)
    /// - `P(x)` evaluates to small integers or base field elements
    ///
    /// We can delay the expensive `E` multiplications until the very end.
    ///
    /// # Why This Works
    ///
    /// ## Variable Splitting
    ///
    /// We split the `n` variables into three groups:
    ///
    /// ```text
    /// ┌─────┬────────┬─────────┐
    /// │  β  │  x_in  │  x_out  │  ← n total variables
    /// └─────┴────────┴─────────┘
    ///    N     n/2    remaining
    /// ```
    ///
    /// - **β** (N vars): The "prefix" variables being sum-checked across rounds 0..N
    /// - **x_in** (n/2 vars): Inner variables summed in tight loop (cache-friendly)
    /// - **x_out** (rest): Outer variables parallelized across threads
    ///
    /// ## Equality Polynomial Factorization
    ///
    /// The equality polynomial factors nicely:
    ///
    /// ```text
    /// E(x) = E_in(x_in) · E_out(x_out)
    /// ```
    ///
    /// This lets us compute the accumulator for round `i` as:
    ///
    /// ```text
    /// A_i(v,u) = Σ_{x_out} E_out(x_out) · [ Σ_{x_in} E_in(x_in) · P(v,u,x_in,x_out) ]
    ///                                       └──────────────────────────────────────┘
    ///                                                "small" inner product
    /// ```
    ///
    /// The bracketed inner sum uses only small values, so it's cheap!
    ///
    /// # Algorithm Steps
    ///
    /// ## 1. Parallel Outer Loop (x_out)
    ///
    /// We parallelize over `x_out` chunks. Each thread gets its own:
    /// - Accumulator storage (no contention)
    /// - Temporary buffer (no allocations in hot path)
    ///
    /// ## 2. Inner Product (x_in) - The Fast Part
    ///
    /// For each fixed `x_out`, we iterate linearly over `x_in`:
    ///
    /// ```text
    /// temp[β] += E_in[x_in] · P(β, x_in, x_out)
    ///            └────────┘   └───────────────┘
    ///              large           small
    /// ```
    ///
    /// ## 3. Distribution (E_out) - The Large Part
    ///
    /// After computing all `temp[β]` for this `x_out`, we multiply by the
    /// expensive `E_out` values and distribute to the round accumulators:
    ///
    /// ```text
    /// A_r[acc_idx] += Σ_k E_out[...] · temp[...]
    /// ```
    ///
    /// This is the only place we do large-field × large-field multiplication.
    pub fn compute_svo_accumulators<EF, const N: usize>(
        &self,
        e_in: &EvaluationsList<EF>,
        e_out: &[EvaluationsList<EF>; N],
    ) -> SvoAccumulators<EF, N>
    where
        EF: ExtensionField<F> + Send + Sync,
    {
        let l = self.num_variables();
        let half_l = l / 2;

        // Calculate variable splits: β (N vars) || x_in (half_l vars) || x_out (remaining)
        let x_out_num_vars = half_l - N + (l % 2);
        let x_num_vars = l - N;

        debug_assert_eq!(half_l + x_out_num_vars, x_num_vars);

        let poly_evals = self.as_slice();
        let num_x_in = 1 << half_l;

        // Cache raw pointers to E_out slices to avoid bounds checks in hot loop.
        //
        // Safe: slices outlive the parallel iterator below.
        let e_out_slices: [&[EF]; N] = core::array::from_fn(|i| e_out[i].as_slice());

        //  PARALLEL FOLD-REDUCE
        //
        // Strategy: Split work over x_out chunks across threads.
        // Each thread maintains its own accumulator + temp buffer.
        //
        (0..1 << x_out_num_vars)
            .into_par_iter()
            .fold(
                || {
                    //  THREAD INITIALIZATION (once per thread)
                    // - The thread-local accumulator
                    // - A reusable scratch space for the inner product
                    (SvoAccumulators::<EF, N>::new(), EF::zero_vec(1 << N))
                },
                |(mut local_accs, mut temp_buffer), x_out| {
                    // STEP 1: Reset temp buffer
                    temp_buffer.fill(EF::ZERO);
                    let temp_ptr = temp_buffer.as_mut_ptr();

                    // STEP 2: Inner product (small multiplications)
                    //
                    // For each x_in, accumulate:
                    //   temp[β] += E_in[x_in] · P(β, x_in, x_out)
                    for (x_in, &e_in_val) in e_in.0.iter().enumerate().take(num_x_in) {
                        // Base index encodes: (β=0, x_in, x_out)
                        let base_index = (x_in << x_out_num_vars) | x_out;

                        // Iterate over all 2^N possible β prefixes
                        for i in 0..(1 << N) {
                            let beta = i << x_num_vars;
                            let index = beta | base_index;

                            // SAFETY: index < 2^l by construction
                            unsafe {
                                let p_val = *poly_evals.get_unchecked(index);
                                *temp_ptr.add(i) += e_in_val * p_val;
                            }
                        }
                    }

                    // STEP 3: Distribution (large multiplications)
                    //
                    // Multiply temp[β] by E_out and add to round accumulators
                    for (r, e_out_slice) in e_out_slices.iter().enumerate().take(N) {
                        let block_size = 1 << (N - 1 - r);
                        let e_out_ptr = e_out_slice.as_ptr();

                        // SAFETY: r < N by loop bound
                        let round_accs = unsafe { local_accs.at_round_unchecked_mut(r) };

                        for (acc_idx, acc) in round_accs.iter_mut().enumerate() {
                            let start_idx = acc_idx * block_size;
                            let mut sum = EF::ZERO;

                            // Dot product: sum over block variations
                            for k in 0..block_size {
                                let e_idx = (k << x_out_num_vars) | x_out;

                                unsafe {
                                    let e_val = *e_out_ptr.add(e_idx);
                                    let t_val = *temp_ptr.add(start_idx + k);
                                    sum += e_val * t_val;
                                }
                            }

                            *acc += sum;
                        }
                    }

                    (local_accs, temp_buffer)
                },
            )
            // REDUCTION: Merge all thread-local accumulators
            .map(|(accs, _)| accs) // Drop temp buffers
            .reduce(SvoAccumulators::new, |a, b| a + b)
    }
}

impl<'a, F> IntoIterator for &'a EvaluationsList<F> {
    type Item = &'a F;
    type IntoIter = core::slice::Iter<'a, F>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<F> IntoIterator for EvaluationsList<F> {
    type Item = F;
    type IntoIter = alloc::vec::IntoIter<F>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

/// Evaluates a multilinear polynomial at an arbitrary point using fast interpolation.
///
/// It's given the polynomial's evaluations over all corners of the boolean hypercube `{0,1}^n` and
/// can find the value at any other point `p` (even with coordinates outside of `{0,1}`).
///
/// ## Algorithm
///
/// The core idea is to recursively reduce the number of variables by one at each step.
/// Imagine a 3D cube where we know the value at its 8 corners. To find the value at some
/// point `p` inside the cube, we can:
/// 1.  Find the values at the midpoints of the 4 edges along the x-axis.
/// 2.  Use those 4 points to find the values at the midpoints of the 2 "ribs" along the y-axis.
/// 3.  Finally, use those 2 points to find the single value at `p` along the z-axis.
///
/// This function implements this idea using the recurrence relation:
/// ```text
///     f(x_0, ..., x_{n-1}) = f_0(x_1, ..., x_{n-1}) * (1 - x_0) + f_1(x_1, ..., x_{n-1}) * x_0,
/// ```
/// where `f_0` is the polynomial with `x_0` fixed to `0` and `f_1` is with `x_0` fixed to `1`.
///
/// ## Implementation Strategies
///
/// To maximize performance, this function uses several strategies:
/// - **Hardcoded Paths:** For polynomials with 0 to 4 variables, the recursion is fully unrolled
///   into highly efficient, direct calculations.
/// - **Recursive Method:** For 5 to 19 variables, a standard recursive approach with a `rayon::join`
///   is used for parallelism on sufficiently large subproblems.
/// - **Non-Recursive Method:** For 20 or more variables, the algorithm switches to a non-recursive,
///   chunk-based method. This avoids deep recursion stacks and uses a memory access pattern
///   that is more friendly to parallelization.
///
/// ## Arguments
///
/// - `evals`: A slice containing the `2^n` evaluations of the polynomial over the boolean
///   hypercube, ordered lexicographically. For `n=2`, the order is `f(0,0), f(0,1), f(1,0), f(1,1)`.
/// - `point`: A slice containing the `n` coordinates of the point `p` at which to evaluate.
fn eval_multilinear<F, EF>(evals: &[F], point: &MultilinearPoint<EF>) -> EF
where
    F: Field,
    EF: ExtensionField<F>,
{
    // Ensure that the number of evaluations matches the number of variables in the point.
    //
    // This is a critical invariant: `evals.len()` must be exactly `2^point.len()`.
    debug_assert_eq!(evals.len(), 1 << point.num_variables());

    // Select the optimal evaluation strategy based on the number of variables.
    match point.as_slice() {
        // Case: 0 Variables (Constant Polynomial)
        //
        // A polynomial with zero variables is just a constant.
        [] => evals[0].into(),

        // Case: 1 Variable (Linear Interpolation)
        //
        // This is the base case for the recursion: f(x) = f(0) * (1-x) + f(1) * x.
        // The expression is an optimized form: f(0) + x * (f(1) - f(0)).
        [x] => *x * (evals[1] - evals[0]) + evals[0],

        // Case: 2 Variables (Bilinear Interpolation)
        //
        // This is a fully unrolled version for 2 variables, avoiding recursive calls.
        [x0, x1] => {
            // Interpolate along the x1-axis for x0=0 to get `a0`.
            let a0 = *x1 * (evals[1] - evals[0]) + evals[0];
            // Interpolate along the x1-axis for x0=1 to get `a1`.
            let a1 = *x1 * (evals[3] - evals[2]) + evals[2];
            // Finally, interpolate between `a0` and `a1` along the x0-axis.
            a0 + (a1 - a0) * *x0
        }

        // Cases: 3 and 4 Variables
        //
        // These are further unrolled versions for 3 and 4 variables for maximum speed.
        // The logic is the same as the 2-variable case, just with more steps.
        [x0, x1, x2] => {
            let a00 = *x2 * (evals[1] - evals[0]) + evals[0];
            let a01 = *x2 * (evals[3] - evals[2]) + evals[2];
            let a10 = *x2 * (evals[5] - evals[4]) + evals[4];
            let a11 = *x2 * (evals[7] - evals[6]) + evals[6];
            let a0 = a00 + *x1 * (a01 - a00);
            let a1 = a10 + *x1 * (a11 - a10);
            a0 + (a1 - a0) * *x0
        }
        [x0, x1, x2, x3] => {
            let a000 = *x3 * (evals[1] - evals[0]) + evals[0];
            let a001 = *x3 * (evals[3] - evals[2]) + evals[2];
            let a010 = *x3 * (evals[5] - evals[4]) + evals[4];
            let a011 = *x3 * (evals[7] - evals[6]) + evals[6];
            let a100 = *x3 * (evals[9] - evals[8]) + evals[8];
            let a101 = *x3 * (evals[11] - evals[10]) + evals[10];
            let a110 = *x3 * (evals[13] - evals[12]) + evals[12];
            let a111 = *x3 * (evals[15] - evals[14]) + evals[14];
            let a00 = a000 + *x2 * (a001 - a000);
            let a01 = a010 + *x2 * (a011 - a010);
            let a10 = a100 + *x2 * (a101 - a100);
            let a11 = a110 + *x2 * (a111 - a110);
            let a0 = a00 + *x1 * (a01 - a00);
            let a1 = a10 + *x1 * (a11 - a10);
            a0 + (a1 - a0) * *x0
        }

        // General Case (5+ Variables)
        //
        // This handles all other cases, using one of two different strategies.
        [x, tail @ ..] => {
            // For a very large number of variables we use a non-recursive algorithm better suited for wide parallelization.
            if point.num_variables() >= MLE_RECURSION_THRESHOLD {
                let mid = point.num_variables() / 2;
                let (hi, lo) = point.as_slice().split_at(mid);

                // Precomputation of Basis Polynomials
                //
                // The basis polynomial eq(v, p) can be split: eq(v, p) = eq(v_low, p_low) * eq(v_high, p_high).
                //
                // We precompute all `2^|lo|` values of eq(v_low, p_low) and store them in `left`.
                // We precompute all `2^|hi|` values of eq(v_high, p_high) and store them in `right`.

                // Allocate uninitialized memory for the polynomial evaluations.
                let mut left = unsafe { uninitialized_vec(1 << lo.len()) };
                let mut right = unsafe { uninitialized_vec(1 << hi.len()) };

                // Compute all eq(v_low, p_low) values and fill the `left` and `right` vectors.
                eval_eq_batch::<_, _, false>(
                    RowMajorMatrixView::new_col(lo),
                    &mut left,
                    &[EF::ONE],
                );
                eval_eq_batch::<_, _, false>(
                    RowMajorMatrixView::new_col(hi),
                    &mut right,
                    &[EF::ONE],
                );

                // Parallelized Final Summation
                //
                // This chain of operations computes the regrouped sum:
                // Σ_{v_high} eq(v_high, p_high) * (Σ_{v_low} f(v_high, v_low) * eq(v_low, p_low))
                evals
                    .par_chunks(left.len())
                    .zip_eq(right.par_iter())
                    .map(|(part, &c)| {
                        // This is the inner sum: a dot product between the evaluation chunk and the `left` basis values.
                        part.iter()
                            .zip_eq(left.iter())
                            .map(|(&a, &b)| b * a)
                            .sum::<EF>()
                            * c
                    })
                    .sum()
            } else {
                // Create a new point with the remaining coordinates.
                let sub_point = MultilinearPoint::new(tail.to_vec());

                // For moderately sized inputs (5 to 19 variables), use the recursive strategy.
                //
                // Split the evaluations into two halves, corresponding to the first variable being 0 or 1.
                let (f0, f1) = evals.split_at(evals.len() / 2);

                // Recursively evaluate on the two smaller hypercubes.
                let (f0_eval, f1_eval) = {
                    // Only spawn parallel tasks if the subproblem is large enough to overcome
                    // the overhead of threading.
                    let work_size: usize = (1 << 15) / core::mem::size_of::<F>();
                    if evals.len() > work_size {
                        join(
                            || eval_multilinear(f0, &sub_point),
                            || eval_multilinear(f1, &sub_point),
                        )
                    } else {
                        // For smaller subproblems, execute sequentially.
                        (
                            eval_multilinear(f0, &sub_point),
                            eval_multilinear(f1, &sub_point),
                        )
                    }
                };
                // Perform the final linear interpolation for the first variable `x`.
                f0_eval + (f1_eval - f0_eval) * *x
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::{
        PrimeCharacteristicRing, PrimeField64, TwoAdicField, extension::BinomialExtensionField,
    };
    use proptest::prelude::*;
    use rand::{Rng, SeedableRng, rngs::SmallRng};

    use super::*;
    use crate::poly::coeffs::CoefficientList;

    type F = BabyBear;
    type EF4 = BinomialExtensionField<F, 4>;

    #[test]
    fn test_new_evaluations_list() {
        let evals = vec![F::ZERO, F::ONE, F::ZERO, F::ONE];
        let evaluations_list = EvaluationsList::new(evals.clone());

        assert_eq!(evaluations_list.num_evals(), evals.len());
        assert_eq!(evaluations_list.num_variables(), 2);
        assert_eq!(evaluations_list.as_slice(), &evals);
    }

    #[test]
    #[should_panic]
    fn test_new_evaluations_list_invalid_length() {
        // Length is not a power of two, should panic
        let _ = EvaluationsList::new(vec![F::ONE, F::ZERO, F::ONE]);
    }

    #[test]
    fn test_indexing() {
        let evals = vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ];
        let evaluations_list = EvaluationsList::new(evals.clone());

        assert_eq!(evaluations_list.0[0], evals[0]);
        assert_eq!(evaluations_list.0[1], evals[1]);
        assert_eq!(evaluations_list.0[2], evals[2]);
        assert_eq!(evaluations_list.0[3], evals[3]);
    }

    #[test]
    #[should_panic]
    fn test_index_out_of_bounds() {
        let evals = vec![F::ZERO, F::ONE, F::ZERO, F::ONE];
        let evaluations_list = EvaluationsList::new(evals);

        let _ = evaluations_list.as_slice()[4]; // Index out of range, should panic
    }

    #[test]
    fn test_mutability_of_evals() {
        let mut evals = EvaluationsList::new(vec![F::ZERO, F::ONE, F::ZERO, F::ONE]);

        assert_eq!(evals.0[1], F::ONE);

        evals.0[1] = F::from_u64(5);

        assert_eq!(evals.0[1], F::from_u64(5));
    }

    #[test]
    fn test_evaluate_edge_cases() {
        let e1 = F::from_u64(7);
        let e2 = F::from_u64(8);
        let e3 = F::from_u64(9);
        let e4 = F::from_u64(10);

        let evals = EvaluationsList::new(vec![e1, e2, e3, e4]);

        // Evaluating at a binary hypercube point should return the direct value
        assert_eq!(
            evals.evaluate_hypercube(&MultilinearPoint::new(vec![F::ZERO, F::ZERO])),
            e1
        );
        assert_eq!(
            evals.evaluate_hypercube(&MultilinearPoint::new(vec![F::ZERO, F::ONE])),
            e2
        );
        assert_eq!(
            evals.evaluate_hypercube(&MultilinearPoint::new(vec![F::ONE, F::ZERO])),
            e3
        );
        assert_eq!(
            evals.evaluate_hypercube(&MultilinearPoint::new(vec![F::ONE, F::ONE])),
            e4
        );
    }

    #[test]
    fn test_num_evals() {
        let evals = EvaluationsList::new(vec![F::ONE, F::ZERO, F::ONE, F::ZERO]);
        assert_eq!(evals.num_evals(), 4);
    }

    #[test]
    fn test_num_variables() {
        let evals = EvaluationsList::new(vec![F::ONE, F::ZERO, F::ONE, F::ZERO]);
        assert_eq!(evals.num_variables(), 2);
    }

    #[test]
    fn test_eval_extension_on_non_hypercube_points() {
        let evals = EvaluationsList::new(vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ]);

        let point = MultilinearPoint::new(vec![F::from_u64(2), F::from_u64(3)]);

        let result = evals.evaluate_hypercube(&point);

        // Expected result using `eval_multilinear`
        let expected = eval_multilinear(evals.as_slice(), &point);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_eval_multilinear_1d() {
        let a = F::from_u64(5);
        let b = F::from_u64(10);
        let evals = vec![a, b];

        // Evaluate at midpoint `x = 1/2`
        let x = F::from_u64(1) / F::from_u64(2);
        let expected = a + (b - a) * x;

        assert_eq!(
            eval_multilinear(&evals, &MultilinearPoint::new(vec![x])),
            expected
        );
    }

    #[test]
    fn test_eval_multilinear_2d() {
        let a = F::from_u64(1);
        let b = F::from_u64(2);
        let c = F::from_u64(3);
        let d = F::from_u64(4);

        // The evaluations are stored in lexicographic order for (x, y)
        // f(0,0) = a, f(0,1) = c, f(1,0) = b, f(1,1) = d
        let evals = vec![a, b, c, d];

        // Evaluate at `(x, y) = (1/2, 1/2)`
        let x = F::from_u64(1) / F::from_u64(2);
        let y = F::from_u64(1) / F::from_u64(2);

        // Interpolation formula:
        // f(x, y) = (1-x)(1-y) * f(0,0) + (1-x)y * f(0,1) + x(1-y) * f(1,0) + xy * f(1,1)
        let expected = (F::ONE - x) * (F::ONE - y) * a
            + (F::ONE - x) * y * c
            + x * (F::ONE - y) * b
            + x * y * d;

        assert_eq!(
            eval_multilinear(&evals, &MultilinearPoint::new(vec![x, y])),
            expected
        );
    }

    #[test]
    fn test_eval_multilinear_3d() {
        let a = F::from_u64(1);
        let b = F::from_u64(2);
        let c = F::from_u64(3);
        let d = F::from_u64(4);
        let e = F::from_u64(5);
        let f = F::from_u64(6);
        let g = F::from_u64(7);
        let h = F::from_u64(8);

        // The evaluations are stored in lexicographic order for (x, y, z)
        // f(0,0,0) = a, f(0,0,1) = c, f(0,1,0) = b, f(0,1,1) = e
        // f(1,0,0) = d, f(1,0,1) = f, f(1,1,0) = g, f(1,1,1) = h
        let evals = vec![a, b, c, e, d, f, g, h];

        let x = F::from_u64(1) / F::from_u64(3);
        let y = F::from_u64(1) / F::from_u64(3);
        let z = F::from_u64(1) / F::from_u64(3);

        // Using trilinear interpolation formula:
        let expected = (F::ONE - x) * (F::ONE - y) * (F::ONE - z) * a
            + (F::ONE - x) * (F::ONE - y) * z * c
            + (F::ONE - x) * y * (F::ONE - z) * b
            + (F::ONE - x) * y * z * e
            + x * (F::ONE - y) * (F::ONE - z) * d
            + x * (F::ONE - y) * z * f
            + x * y * (F::ONE - z) * g
            + x * y * z * h;

        assert_eq!(
            eval_multilinear(&evals, &MultilinearPoint::new(vec![x, y, z])),
            expected
        );
    }

    #[test]
    fn test_eval_multilinear_4d() {
        let a = F::from_u64(1);
        let b = F::from_u64(2);
        let c = F::from_u64(3);
        let d = F::from_u64(4);
        let e = F::from_u64(5);
        let f = F::from_u64(6);
        let g = F::from_u64(7);
        let h = F::from_u64(8);
        let i = F::from_u64(9);
        let j = F::from_u64(10);
        let k = F::from_u64(11);
        let l = F::from_u64(12);
        let m = F::from_u64(13);
        let n = F::from_u64(14);
        let o = F::from_u64(15);
        let p = F::from_u64(16);

        // Evaluations stored in lexicographic order for (x, y, z, w)
        let evals = vec![a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p];

        let x = F::from_u64(1) / F::from_u64(2);
        let y = F::from_u64(2) / F::from_u64(3);
        let z = F::from_u64(1) / F::from_u64(4);
        let w = F::from_u64(3) / F::from_u64(5);

        // Quadlinear interpolation formula
        let expected = (F::ONE - x) * (F::ONE - y) * (F::ONE - z) * (F::ONE - w) * a
            + (F::ONE - x) * (F::ONE - y) * (F::ONE - z) * w * b
            + (F::ONE - x) * (F::ONE - y) * z * (F::ONE - w) * c
            + (F::ONE - x) * (F::ONE - y) * z * w * d
            + (F::ONE - x) * y * (F::ONE - z) * (F::ONE - w) * e
            + (F::ONE - x) * y * (F::ONE - z) * w * f
            + (F::ONE - x) * y * z * (F::ONE - w) * g
            + (F::ONE - x) * y * z * w * h
            + x * (F::ONE - y) * (F::ONE - z) * (F::ONE - w) * i
            + x * (F::ONE - y) * (F::ONE - z) * w * j
            + x * (F::ONE - y) * z * (F::ONE - w) * k
            + x * (F::ONE - y) * z * w * l
            + x * y * (F::ONE - z) * (F::ONE - w) * m
            + x * y * (F::ONE - z) * w * n
            + x * y * z * (F::ONE - w) * o
            + x * y * z * w * p;

        // Validate against the function output
        assert_eq!(
            eval_multilinear(&evals, &MultilinearPoint::new(vec![x, y, z, w])),
            expected
        );
    }

    proptest! {
        #[test]
        fn prop_eval_multilinear_equiv_between_f_and_ef4(
            values in prop::collection::vec(0u64..100, 8),
            x0 in 0u64..100,
            x1 in 0u64..100,
            x2 in 0u64..100,
        ) {
            // Base field evaluations
            let coeffs_f: Vec<F> = values.iter().copied().map(F::from_u64).collect();
            let poly_f = EvaluationsList::new(coeffs_f);

            // Lift to extension field EF4
            let coeffs_ef: Vec<EF4> = values.iter().copied().map(EF4::from_u64).collect();
            let poly_ef = EvaluationsList::new(coeffs_ef);

            // Evaluation point in EF4
            let point_ef = MultilinearPoint::new(vec![
                EF4::from_u64(x0),
                EF4::from_u64(x1),
                EF4::from_u64(x2),
            ]);

            // Evaluate using both base and extension representations
            let eval_f = poly_f.evaluate_hypercube(&point_ef);
            let eval_ef = poly_ef.evaluate_hypercube(&point_ef);

            prop_assert_eq!(eval_f, eval_ef);
        }
    }

    #[test]
    fn test_multilinear_eval_two_vars() {
        // Define a simple 2-variable multilinear polynomial:
        //
        // Variables: x_1, x_2
        // Coefficients ordered in lexicographic order: (x_1, x_2)
        //
        // - coeffs[0] → constant term
        // - coeffs[1] → x_2 term
        // - coeffs[2] → x_1 term
        // - coeffs[3] → x_1·x_2 term
        //
        // Thus, the polynomial is:
        //
        //   f(x_1, x_2) = c0 + c1·x_2 + c2·x_1 + c3·x_1·x_2
        //
        // where:
        let c0 = F::from_u64(5); // constant
        let c1 = F::from_u64(6); // x_2 coefficient
        let c2 = F::from_u64(7); // x_1 coefficient
        let c3 = F::from_u64(8); // x_1·x_2 coefficient
        //
        // So concretely:
        //
        //   f(x_1, x_2) = 5 + 6·x_2 + 7·x_1 + 8·x_1·x_2
        let coeffs = CoefficientList::new(vec![c0, c1, c2, c3]);

        // Convert coefficients to evaluations via the wavelet transform
        let evals = coeffs.to_evaluations();

        // Choose evaluation point:
        //
        // Let's pick (x_1, x_2) = (2, 1)
        let x1 = F::from_u64(2);
        let x2 = F::from_u64(1);
        let coords = MultilinearPoint::new(vec![x1, x2]);

        // Manually compute the expected value step-by-step:
        //
        // Reminder:
        //   f(x_1, x_2) = 5 + 6·x_2 + 7·x_1 + 8·x_1·x_2
        //
        // Substituting (x_1, x_2):
        let expected = c0 + c1 * x2 + c2 * x1 + c3 * x1 * x2;

        // Now evaluate using the function under test
        let result = evals.evaluate_hypercube(&coords);

        // Check that it matches the manual computation
        assert_eq!(result, expected);
    }

    #[test]
    fn test_evaluate_3_variables() {
        // Define a multilinear polynomial in 3 variables: x_0, x_1, x_2
        //
        // Coefficients ordered lex in index order:
        //
        // - coeffs[0] → constant term
        // - coeffs[1] → x_2
        // - coeffs[2] → x_1
        // - coeffs[3] → x_1·x_2
        // - coeffs[4] → x_0
        // - coeffs[5] → x_0·x_2
        // - coeffs[6] → x_0·x_1
        // - coeffs[7] → x_0·x_1·x_2
        //
        // Thus:
        //    f(x_0,x_1,x_2) = c0 + c1·x_2 + c2·x_1 + c3·x_1·x_2
        //                + c4·x_0 + c5·x_0·x_2 + c6·x_0·x_1 + c7·x_0·x_1·x_2
        let c0 = F::from_u64(1);
        let c1 = F::from_u64(2);
        let c2 = F::from_u64(3);
        let c3 = F::from_u64(4);
        let c4 = F::from_u64(5);
        let c5 = F::from_u64(6);
        let c6 = F::from_u64(7);
        let c7 = F::from_u64(8);

        let coeffs = CoefficientList::new(vec![c0, c1, c2, c3, c4, c5, c6, c7]);
        let evals = coeffs.to_evaluations();

        // Pick point: (x_0,x_1,x_2) = (2, 3, 4)
        let x0 = F::from_u64(2);
        let x1 = F::from_u64(3);
        let x2 = F::from_u64(4);

        let point = MultilinearPoint::new(vec![x0, x1, x2]);

        // Manually compute:
        //
        // expected = 1
        //          + 2·4
        //          + 3·3
        //          + 4·3·4
        //          + 5·2
        //          + 6·2·4
        //          + 7·2·3
        //          + 8·2·3·4
        let expected = c0
            + c1 * x2
            + c2 * x1
            + c3 * x1 * x2
            + c4 * x0
            + c5 * x0 * x2
            + c6 * x0 * x1
            + c7 * x0 * x1 * x2;

        let result = evals.evaluate_hypercube(&point);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_eval_extension_3_variables() {
        // Define a multilinear polynomial in 3 variables: x_0, x_1, x_2
        //
        // Coefficients ordered lex in index order:
        //
        // - coeffs[0] → constant term
        // - coeffs[1] → x_2 term
        // - coeffs[2] → x_1 term
        // - coeffs[3] → x_1·x_2 term
        // - coeffs[4] → x_0 term
        // - coeffs[5] → x_0·x_2 term
        // - coeffs[6] → x_0·x_1 term
        // - coeffs[7] → x_0·x_1·x_2 term
        //
        // Thus:
        //    f(x_0,x_1,x_2) = c0 + c1·x_2 + c2·x_1 + c3·x_1·x_2
        //                + c4·x_0 + c5·x_0·x_2 + c6·x_0·x_1 + c7·x_0·x_1·x_2
        let c0 = F::from_u64(1);
        let c1 = F::from_u64(2);
        let c2 = F::from_u64(3);
        let c3 = F::from_u64(4);
        let c4 = F::from_u64(5);
        let c5 = F::from_u64(6);
        let c6 = F::from_u64(7);
        let c7 = F::from_u64(8);

        let coeffs = CoefficientList::new(vec![c0, c1, c2, c3, c4, c5, c6, c7]);
        let evals = coeffs.to_evaluations();

        // Choose evaluation point: (x_0, x_1, x_2) = (2, 3, 4)
        //
        // Here we lift into the extension field EF4
        let x0 = EF4::from_u64(2);
        let x1 = EF4::from_u64(3);
        let x2 = EF4::from_u64(4);

        let point = MultilinearPoint::new(vec![x0, x1, x2]);

        // Manually compute expected value
        //
        // Substituting (x_0,x_1,x_2) = (2,3,4) into:
        //
        //   f(x_0,x_1,x_2) = 1
        //               + 2·4
        //               + 3·3
        //               + 4·3·4
        //               + 5·2
        //               + 6·2·4
        //               + 7·2·3
        //               + 8·2·3·4
        //
        // and lifting each constant into EF4 for correct typing
        let expected = EF4::from(c0)
            + EF4::from(c1) * x2
            + EF4::from(c2) * x1
            + EF4::from(c3) * x1 * x2
            + EF4::from(c4) * x0
            + EF4::from(c5) * x0 * x2
            + EF4::from(c6) * x0 * x1
            + EF4::from(c7) * x0 * x1 * x2;

        // Evaluate via `evaluate_hypercube` method
        let result = evals.evaluate_hypercube(&point);

        // Verify that result matches manual computation
        assert_eq!(result, expected);
    }

    #[test]
    fn test_folding_and_evaluation() {
        // Set number of Boolean input variables n = 10.
        let num_variables = 10;

        // Build a multilinear polynomial f(x) = x with coefficients 0, 1, ..., 1023 in F
        let coeffs = (0..(1 << num_variables)).map(F::from_u64).collect();

        // Wrap into CoefficientList to access polynomial logic
        let coeffs_list = CoefficientList::new(coeffs);

        // Convert to EvaluationsList using a wavelet transform
        let evals_list: EvaluationsList<F> = coeffs_list.clone().to_evaluations();

        // Define a fixed evaluation point in F^n: [0, 35, 70, ..., 35*(n-1)]
        let randomness: Vec<_> = (0..num_variables)
            .map(|i| F::from_u64(35 * i as u64))
            .collect();

        // Try folding at every possible prefix of the randomness vector: k = 1 to n-1
        for k in 1..num_variables {
            // Use the first k values as the fold coordinates (we will substitute those)
            let fold_part = randomness[0..k].to_vec();

            // The remaining coordinates are used as the evaluation input into the folded poly
            let eval_part = randomness[k..randomness.len()].to_vec();

            // Convert to a MultilinearPoint (in EF) for folding
            let fold_random = MultilinearPoint::new(fold_part.clone());

            // Reconstruct the full point (x_0, ..., xₙ₋₁) = [eval_part || fold_part]
            // Used to evaluate the original uncompressed polynomial
            let eval_point1 =
                MultilinearPoint::new([fold_part.clone(), eval_part.clone()].concat());
            let eval_point2 = MultilinearPoint::new([eval_part.clone(), fold_part].concat());

            // Fold the evaluation list over the last `k` variables
            let folded_evals = evals_list.fold(&fold_random);

            // Verify that the number of variables has been folded correctly
            assert_eq!(folded_evals.num_variables(), num_variables - k);

            // Fold the coefficients list over the last `k` variables
            let folded_coeffs = coeffs_list.fold(&fold_random);

            // Verify that the number of variables has been folded correctly
            assert_eq!(folded_coeffs.num_variables(), num_variables - k);

            // Verify correctness:
            // folded(e) == original([e, r]) for all k
            assert_eq!(
                folded_evals.evaluate_hypercube(&MultilinearPoint::new(eval_part.clone())),
                evals_list.evaluate_hypercube(&eval_point1)
            );

            // Compare with the coefficient list equivalent
            assert_eq!(
                folded_coeffs.evaluate(&MultilinearPoint::new(eval_part)),
                evals_list.evaluate_hypercube(&eval_point2)
            );
        }
    }

    #[test]
    fn test_fold_with_extension_one_var() {
        // Define a 2-variable polynomial:
        // f(x_0, x_1) = 1 + 2·x_1 + 3·x_0 + 4·x_0·x_1
        let coeffs = vec![
            F::from_u64(1), // constant
            F::from_u64(2), // x_1
            F::from_u64(3), // x_0
            F::from_u64(4), // x_0·x_1
        ];
        let poly = CoefficientList::new(coeffs);

        // Convert coefficients into an EvaluationsList (for testing the fold on evals)
        let evals_list: EvaluationsList<F> = poly.clone().to_evaluations();

        // We fold over the last variable (x_1) by setting x_1 = 5 in EF4
        let r1 = EF4::from_u64(5);

        // Perform the fold: f(x_0, 5) becomes a new function g(x_0)
        let folded = evals_list.fold(&MultilinearPoint::new(vec![r1]));

        // For 10 test points x_0 = 0, 1, ..., 9
        for x0_f in 0..10 {
            // Lift to EF4 for extension-field evaluation
            let x0 = EF4::from_u64(x0_f);

            // Construct the full point (x_0, x_1 = 5)
            let full_point = MultilinearPoint::new(vec![r1, x0]);

            // Construct folded point (x_0)
            let folded_point = MultilinearPoint::new(vec![x0]);

            // Evaluate original poly at (x_0, 5)
            let expected = poly.evaluate(&full_point);

            // Evaluate folded poly at x_0
            let actual = folded.evaluate_hypercube(&folded_point);

            // Ensure the results agree
            assert_eq!(expected, actual);
        }
    }

    proptest! {
        #[test]
        fn prop_eval_eq_matches_naive_for_eval_list(
            // number of variables (keep small to avoid blowup)
            n in 1usize..5,
             // always at least 5 elements
            evals_raw in prop::collection::vec(0u64..F::ORDER_U64, 5),
        ) {
            // Slice out exactly n elements, guaranteed present
            let evals: Vec<F> = evals_raw[..n].iter().map(|&x| F::from_u64(x)).collect();

            // Allocate output buffer of size 2^n
            let mut out = vec![F::ZERO; 1 << n];

            // Run eval_eq with scalar = 1
            eval_eq_batch::<F, F, false>(
                RowMajorMatrixView::new_col(&evals),
                &mut out,
                &[F::ONE],
            );

            // Naively compute expected values for each binary assignment
            let mut expected = vec![F::ZERO; 1 << n];
            for (i, e) in expected.iter_mut().enumerate().take(1 << n) {
                let mut weight = F::ONE;
                for (j, &val) in evals.iter().enumerate() {
                    let bit = (i >> (n - 1 - j)) & 1;
                    if bit == 1 {
                        weight *= val;
                    } else {
                        weight *= F::ONE - val;
                    }
                }
                *e = weight;
            }

            prop_assert_eq!(out, expected);
        }
    }

    #[test]
    fn test_eval_multilinear_large_input_brute_force() {
        // Define the number of variables.
        //
        // We use 20 to trigger the case where the recursive algorithm is not optimal.
        const NUM_VARS: usize = 20;

        // Use a seeded random number generator for a reproducible test case.
        let mut rng = SmallRng::seed_from_u64(42);

        // The number of evaluations on the boolean hypercube is 2^n.
        let num_evals = 1 << NUM_VARS;

        // Create a vector of random evaluations for our polynomial `f`.
        let evals_vec: Vec<F> = (0..num_evals).map(|_| rng.random()).collect();
        let evals_list = EvaluationsList::new(evals_vec);

        // Create a random point `p` where we will evaluate the polynomial.
        let point_vec: Vec<EF4> = (0..NUM_VARS).map(|_| rng.random()).collect();
        let point = MultilinearPoint::new(point_vec);

        // BRUTE-FORCE CALCULATION (GROUND TRUTH)
        //
        // We will now calculate the expected result using the fundamental formula:
        // f(p) = Σ_{v ∈ {0,1}^n} f(v) * eq(v, p)
        // where eq(v, p) = Π_{i=0..n-1} (v_i*p_i + (1-v_i)*(1-p_i))

        // This variable will accumulate the sum. It must be in the extension field.
        let mut expected_sum = EF4::ZERO;

        // Iterate through every point `v` on the boolean hypercube {0,1}^20.
        //
        // The loop counter `i` represents the integer value of the bit-string for `v`.
        for i in 0..num_evals {
            // This will hold the eq(v, p) value for the current hypercube point `v`.
            let mut eq_term = EF4::ONE;

            // To build eq(v, p), we iterate through each dimension of the hypercube.
            for j in 0..NUM_VARS {
                // Get the j-th bit of `i`. This corresponds to the coordinate v_j.
                // We read bits from most-significant to least-significant to match the
                // lexicographical ordering of the `evals_list`.
                let v_j = (i >> (NUM_VARS - 1 - j)) & 1;

                // Get the corresponding j-th coordinate of our evaluation point `p`.
                let p_j = point.as_slice()[j];

                if v_j == 1 {
                    // If the hypercube coordinate v_j is 1, the factor is p_j.
                    eq_term *= p_j;
                } else {
                    // If the hypercube coordinate v_j is 0, the factor is (1 - p_j).
                    eq_term *= EF4::ONE - p_j;
                }
            }

            // Get the pre-computed evaluation f(v) from our list. The index `i`
            // directly corresponds to the lexicographically ordered point `v`.
            let f_v = evals_list.as_slice()[i];

            // Add the term f(v) * eq(v, p) to the total sum. We must lift `f_v` from the
            // base field `F` to the extension field `EF4` for the multiplication.
            expected_sum += eq_term * f_v;
        }

        // Now, run the optimized function that we want to test.
        let actual_result = evals_list.evaluate_hypercube(&point);

        // Finally, assert that the results are equal.
        assert_eq!(actual_result, expected_sum);
    }

    #[test]
    fn test_new_from_point_zero_vars() {
        let point = MultilinearPoint::<F>::new(vec![]);
        let value = F::from_u64(42);
        let evals_list = EvaluationsList::new_from_point(point.as_slice(), value);

        // For n=0, the hypercube has one point, and the `eq` polynomial is the constant 1.
        // The result should be a list with a single element: `value`.
        assert_eq!(evals_list.num_variables(), 0);
        assert_eq!(evals_list.as_slice(), &[value]);
    }

    #[test]
    fn test_new_from_point_one_var() {
        let p0 = F::from_u64(7);
        let point = MultilinearPoint::new(vec![p0]);
        let value = F::from_u64(3);
        let evals_list = EvaluationsList::new_from_point(point.as_slice(), value);

        // For a point `p = [p0]`, the `eq` evaluations over `X={0,1}` are:
        // - eq(p, 0) = 1 - p0
        // - eq(p, 1) = p0
        // These are then scaled by `value`.
        let expected = vec![value * (F::ONE - p0), value * p0];

        assert_eq!(evals_list.num_variables(), 1);
        assert_eq!(evals_list.as_slice(), &expected);
    }

    #[test]
    #[allow(clippy::identity_op)]
    fn test_new_from_point_three_vars() {
        let p = [F::from_u64(2), F::from_u64(3), F::from_u64(5)];
        let value = F::from_u64(10);
        let evals_list = EvaluationsList::new_from_point(&p, value);

        // Manually compute the expected result for eq(p, b) * value for all 8 points `b`.
        // The implementation's lexicographical order means the index `i` is formed as
        // i = 4*b0 + 2*b1 + 1*b2, where `b` is the hypercube point (b0, b1, b2).
        let mut expected = Vec::with_capacity(8);
        for i in 0..8 {
            // We extract the bits of `i` to determine the coordinates of the hypercube point `b`.
            //
            // MSB of `i` corresponds to the first variable, p[0].
            let b0 = (i >> 2) & 1;
            // Middle bit of `i` corresponds to the second variable, p[1].
            let b1 = (i >> 1) & 1;
            // LSB of `i` corresponds to the last variable, p[2].
            let b2 = (i >> 0) & 1;

            // Calculate the eq(p, b) term for this specific point `b`.
            let term0 = if b0 == 1 { p[0] } else { F::ONE - p[0] };
            let term1 = if b1 == 1 { p[1] } else { F::ONE - p[1] };
            let term2 = if b2 == 1 { p[2] } else { F::ONE - p[2] };

            expected.push(value * term0 * term1 * term2);
        }

        assert_eq!(evals_list.num_variables(), 3);
        assert_eq!(evals_list.as_slice(), &expected);
    }

    #[test]
    fn test_as_constant_for_constant_poly() {
        // A polynomial with 0 variables is a constant. Its evaluation
        // list contains a single value.
        let constant_value = F::from_u64(42);
        let evals = EvaluationsList::new(vec![constant_value]);

        // `as_constant` should return the value wrapped in `Some`.
        assert_eq!(evals.num_variables(), 0);
        assert_eq!(evals.as_constant(), Some(constant_value));
    }

    #[test]
    fn test_as_constant_for_non_constant_poly() {
        // A polynomial with 2 variables is not a constant. Its evaluation
        // list has 2^2 = 4 entries.
        let evals = EvaluationsList::new(vec![F::ONE, F::ZERO, F::ONE, F::ZERO]);

        // For any non-constant polynomial, `as_constant` should return `None`.
        assert_ne!(evals.num_variables(), 0);
        assert_eq!(evals.as_constant(), None);
    }

    #[test]
    #[allow(clippy::identity_op)]
    fn test_accumulate_batch_single_point() {
        // Set up an initial list of evaluations.
        let n = 2;
        let initial_values = vec![
            F::from_u64(10),
            F::from_u64(20),
            F::from_u64(30),
            F::from_u64(40),
        ];
        let mut evals_list = EvaluationsList::new(initial_values.clone());

        // Define the point and value to accumulate.
        let p = [F::from_u64(2), F::from_u64(3)];
        let point = MultilinearPoint::new(p.to_vec());
        let value = F::from_u64(5);

        // Manually compute the `eq` evaluations that should be added.
        let mut eq_evals_to_add = Vec::with_capacity(1 << n);
        for i in 0..(1 << n) {
            let b0 = (i >> 1) & 1; // MSB for p[0]
            let b1 = (i >> 0) & 1; // LSB for p[1]
            let term0 = if b0 == 1 { p[0] } else { F::ONE - p[0] };
            let term1 = if b1 == 1 { p[1] } else { F::ONE - p[1] };
            eq_evals_to_add.push(value * term0 * term1);
        }

        // Calculate the final expected evaluations after addition.
        let expected: Vec<F> = initial_values
            .iter()
            .zip(eq_evals_to_add.iter())
            .map(|(&initial, &to_add)| initial + to_add)
            .collect();

        // Call accumulate_batch with a single point and assert that the result matches the expected sum.
        evals_list.accumulate_batch(&[point], &[value]);
        assert_eq!(evals_list.as_slice(), &expected);
    }

    #[test]
    #[allow(clippy::identity_op)]
    fn test_accumulate_batch_multiple_points() {
        // Set up an initial list of evaluations.
        let n = 2;
        let initial_values = vec![
            F::from_u64(10),
            F::from_u64(20),
            F::from_u64(30),
            F::from_u64(40),
        ];
        let mut evals_list = EvaluationsList::new(initial_values.clone());

        // Define two points and values to accumulate.
        let p1 = [F::from_u64(2), F::from_u64(3)];
        let p2 = [F::from_u64(4), F::from_u64(5)];
        let point1 = MultilinearPoint::new(p1.to_vec());
        let point2 = MultilinearPoint::new(p2.to_vec());
        let value1 = F::from_u64(5);
        let value2 = F::from_u64(7);

        // Manually compute the `eq` evaluations for both points.
        let mut eq_evals_sum = vec![F::ZERO; 1 << n];
        for (p, val) in [(p1, value1), (p2, value2)] {
            for (i, eq_eval_sum) in eq_evals_sum.iter_mut().enumerate().take(1 << n) {
                let b0 = (i >> 1) & 1;
                let b1 = (i >> 0) & 1;
                let term0 = if b0 == 1 { p[0] } else { F::ONE - p[0] };
                let term1 = if b1 == 1 { p[1] } else { F::ONE - p[1] };
                *eq_eval_sum += val * term0 * term1;
            }
        }

        // Calculate the final expected evaluations after addition.
        let expected: Vec<F> = initial_values
            .iter()
            .zip(eq_evals_sum.iter())
            .map(|(&initial, &to_add)| initial + to_add)
            .collect();

        // Call accumulate_batch with multiple points.
        evals_list.accumulate_batch(&[point1, point2], &[value1, value2]);
        assert_eq!(evals_list.as_slice(), &expected);
    }

    #[test]
    #[allow(clippy::identity_op)]
    fn test_accumulate_base_batch() {
        // Set up an initial list of evaluations.
        let n = 2;
        // Initial evaluations in the extension field.
        let initial_values: Vec<EF4> = vec![
            EF4::from_u64(10),
            EF4::from_u64(20),
            EF4::from_u64(30),
            EF4::from_u64(40),
        ];
        let mut evals_list = EvaluationsList::new(initial_values.clone());

        // Point in the base field `F`, value in the extension field `EF4`.
        let p_base = [F::from_u64(2), F::from_u64(3)];
        let point_base = MultilinearPoint::new(p_base.to_vec());
        let value_ext = EF4::from_u64(5);

        // Manually compute `eq` evals,
        // lifting base field elements to extension field.
        let mut eq_evals_to_add = Vec::with_capacity(1 << n);
        for i in 0..(1 << n) {
            let b0 = (i >> 1) & 1; // MSB
            let b1 = (i >> 0) & 1; // LSB
            let term0 = if b0 == 1 {
                EF4::from(p_base[0])
            } else {
                EF4::ONE - EF4::from(p_base[0])
            };
            let term1 = if b1 == 1 {
                EF4::from(p_base[1])
            } else {
                EF4::ONE - EF4::from(p_base[1])
            };
            eq_evals_to_add.push(value_ext * term0 * term1);
        }

        // Calculate the final expected sum in the extension field.
        let expected: Vec<EF4> = initial_values
            .iter()
            .zip(eq_evals_to_add.iter())
            .map(|(&initial, &to_add)| initial + to_add)
            .collect();

        // Accumulate and assert the result.
        evals_list.accumulate_base_batch(&[point_base], &[value_ext]);
        assert_eq!(evals_list.as_slice(), &expected);
    }

    #[test]
    #[should_panic]
    fn test_compress_panics_on_constant() {
        // A constant polynomial has 0 variables and cannot be compressed.
        let mut evals_list = EvaluationsList::new(vec![F::from_u64(42)]);
        evals_list.compress(F::ONE); // This should panic.
    }

    #[test]
    fn test_compress_basic() {
        // Initial layout (n=3 variables, 8 evaluations):
        //
        // [p(0,0,0), p(0,0,1), p(0,1,0), p(0,1,1), p(1,0,0), p(1,0,1), p(1,1,0), p(1,1,1)]
        let p_000 = F::from_u64(1);
        let p_001 = F::from_u64(2);
        let p_010 = F::from_u64(3);
        let p_011 = F::from_u64(4);
        let p_100 = F::from_u64(5);
        let p_101 = F::from_u64(6);
        let p_110 = F::from_u64(7);
        let p_111 = F::from_u64(8);
        let initial_evals = vec![p_000, p_001, p_010, p_011, p_100, p_101, p_110, p_111];
        let mut evals_list = EvaluationsList::new(initial_evals);
        let r = F::from_u64(10);

        // Compute the expected folded values using the half-split pattern:
        //
        // result[i] = r * (right[i] - left[i]) + left[i]
        let expected = vec![
            // p(r, 0, 0) = r * (p(1,0,0) - p(0,0,0)) + p(0,0,0)
            r * (p_100 - p_000) + p_000,
            // p(r, 0, 1) = r * (p(1,0,1) - p(0,0,1)) + p(0,0,1)
            r * (p_101 - p_001) + p_001,
            // p(r, 1, 0) = r * (p(1,1,0) - p(0,1,0)) + p(0,1,0)
            r * (p_110 - p_010) + p_010,
            // p(r, 1, 1) = r * (p(1,1,1) - p(0,1,1)) + p(0,1,1)
            r * (p_111 - p_011) + p_011,
        ];

        // Before compression, the dimensions are:
        // - n = 3 variables
        // - num_evals = 8 evaluations
        assert_eq!(evals_list.num_variables(), 3);
        assert_eq!(evals_list.num_evals(), 8);

        evals_list.compress(r);

        // After compression, the dimensions are:
        // - n = 2 variables
        // - num_evals = 4 evaluations
        assert_eq!(evals_list.num_variables(), 2);
        assert_eq!(evals_list.num_evals(), 4);
        assert_eq!(evals_list.as_slice(), &expected);
    }

    #[test]
    fn test_compress_parallel_path() {
        // Test the parallel code path.
        let num_evals = PARALLEL_THRESHOLD;
        let mid = num_evals / 2;

        let p_left_0 = F::from_u64(1);
        let p_right_0 = F::from_usize(mid + 1);
        let p_left_1 = F::from_u64(2);
        let p_right_1 = F::from_usize(mid + 2);

        let initial_evals: Vec<F> = (0..num_evals).map(|i| F::from_usize(i + 1)).collect();
        let mut evals_list = EvaluationsList::new(initial_evals);
        let r = F::from_u64(3);

        let num_variables_before = evals_list.num_variables();

        evals_list.compress(r);

        // Verify dimensions.
        assert_eq!(evals_list.num_variables(), num_variables_before - 1);
        assert_eq!(evals_list.num_evals(), mid);

        // Verify first few elements manually to ensure correctness.
        //
        // result[i] = r * (right[i] - left[i]) + left[i]
        assert_eq!(
            evals_list.as_slice()[0],
            r * (p_right_0 - p_left_0) + p_left_0
        );
        assert_eq!(
            evals_list.as_slice()[1],
            r * (p_right_1 - p_left_1) + p_left_1
        );
    }

    #[test]
    fn test_compress_multiple_rounds() {
        // Test multiple rounds of compression, as would happen in reality.
        //
        // Each round folds the first variable of the current polynomial.
        let initial_evals: Vec<F> = (1..=16).map(F::from_u64).collect();
        let mut evals_list = EvaluationsList::new(initial_evals);

        let challenges = vec![F::from_u64(3), F::from_u64(7), F::from_u64(11)];

        // Apply three rounds of compression.
        for &r in &challenges {
            evals_list.compress(r);
        }

        // After 3 rounds, we should go from 4 variables to 1 variable.
        assert_eq!(evals_list.num_variables(), 1);
        assert_eq!(evals_list.num_evals(), 2);
    }

    #[test]
    fn test_compress_single_variable() {
        // Test compression of a polynomial with just 1 variable (edge case).
        //
        // Initial layout (n=1 variable, 2 evaluations):
        //
        // [p(0), p(1)]
        let p_0 = F::from_u64(5);
        let p_1 = F::from_u64(9);
        let mut evals_list = EvaluationsList::new(vec![p_0, p_1]);
        let r = F::from_u64(7);

        evals_list.compress(r);

        // After compression, we should have a constant polynomial.
        assert_eq!(evals_list.num_variables(), 0);
        assert_eq!(evals_list.num_evals(), 1);

        // Result: p(r) = r * (p(1) - p(0)) + p(0)
        let expected = r * (p_1 - p_0) + p_0;
        assert_eq!(evals_list.as_slice(), vec![expected]);
    }

    #[test]
    fn test_compress_with_zero_challenge() {
        // Test with challenge r = 0 (should select the left half).
        //
        // Initial layout (n=3 variables, 8 evaluations):
        //
        // [p(0,0,0), p(0,0,1), p(0,1,0), p(0,1,1), p(1,0,0), p(1,0,1), p(1,1,0), p(1,1,1)]
        let p_000 = F::from_u64(1);
        let p_001 = F::from_u64(2);
        let p_010 = F::from_u64(3);
        let p_011 = F::from_u64(4);
        let p_100 = F::from_u64(5);
        let p_101 = F::from_u64(6);
        let p_110 = F::from_u64(7);
        let p_111 = F::from_u64(8);
        let mut evals_list =
            EvaluationsList::new(vec![p_000, p_001, p_010, p_011, p_100, p_101, p_110, p_111]);
        let r = F::ZERO;

        evals_list.compress(r);

        // With r = 0, result[i] = 0 * (right[i] - left[i]) + left[i] = left[i]
        //
        // So we should get the first half (left half where X_1 = 0):
        let expected = vec![p_000, p_001, p_010, p_011];

        assert_eq!(evals_list.as_slice(), &expected);
    }

    #[test]
    fn test_compress_with_one_challenge() {
        // Test with challenge r = 1 (should select the right half).
        //
        // Initial layout (n=3 variables, 8 evaluations):
        //
        // [p(0,0,0), p(0,0,1), p(0,1,0), p(0,1,1), p(1,0,0), p(1,0,1), p(1,1,0), p(1,1,1)]
        let p_000 = F::from_u64(1);
        let p_001 = F::from_u64(2);
        let p_010 = F::from_u64(3);
        let p_011 = F::from_u64(4);
        let p_100 = F::from_u64(5);
        let p_101 = F::from_u64(6);
        let p_110 = F::from_u64(7);
        let p_111 = F::from_u64(8);
        let mut evals_list =
            EvaluationsList::new(vec![p_000, p_001, p_010, p_011, p_100, p_101, p_110, p_111]);
        let r = F::ONE;

        evals_list.compress(r);

        // With r = 1, result[i] = 1 * (right[i] - left[i]) + left[i] = right[i]
        //
        // So we should get the second half (right half where X_1 = 1):
        let expected = vec![p_100, p_101, p_110, p_111];

        assert_eq!(evals_list.as_slice(), &expected);
    }

    #[test]
    fn test_compress_ext() {
        // This test verifies the out-of-place compression into an extension field.
        let initial_evals: Vec<F> = [1u64, 3, 5, 7, 2, 4, 6, 8]
            .into_iter()
            .map(F::from_u64)
            .collect();
        let evals_list = EvaluationsList::new(initial_evals);
        let r_ext = EF4::from_u64(10);

        // The expected result is the same as `test_compress`, but with elements
        // lifted into the extension field EF4.
        let expected: Vec<EF4> = vec![
            r_ext * (EF4::from_u64(2) - EF4::from_u64(1)) + EF4::from_u64(1),
            r_ext * (EF4::from_u64(4) - EF4::from_u64(3)) + EF4::from_u64(3),
            r_ext * (EF4::from_u64(6) - EF4::from_u64(5)) + EF4::from_u64(5),
            r_ext * (EF4::from_u64(8) - EF4::from_u64(7)) + EF4::from_u64(7),
        ];

        // The method returns a new list and does not modify the original.
        let compressed_ext_list = evals_list.compress_ext(r_ext);

        assert_eq!(compressed_ext_list.num_variables(), 2);
        assert_eq!(compressed_ext_list.num_evals(), 4);
        assert_eq!(compressed_ext_list.as_slice(), &expected);
    }

    #[test]
    #[should_panic]
    fn test_compress_ext_panics_on_constant() {
        // A constant polynomial has 0 variables and cannot be compressed.
        let evals_list = EvaluationsList::new(vec![F::from_u64(42)]);
        let _ = evals_list.compress_ext(EF4::ONE); // This should panic.
    }

    proptest! {
        #[test]
        fn prop_compress_and_compress_ext_agree(
            n in 1..=6,
            seed in any::<u64>(),
        ) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let num_evals = 1 << n;
            let evals: Vec<F> = (0..num_evals).map(|_| rng.random()).collect();
            let r_base: F = rng.random();

            // Path A: Use the in-place `compress` method.
            let mut list_a = EvaluationsList::new(evals.clone());
            list_a.compress(r_base);
            // Lift the result into the extension field for comparison.
            let result_a_lifted: Vec<EF4> = list_a.as_slice().iter().map(|&x| EF4::from(x)).collect();

            // Path B: Use the `compress_ext` method with the same challenge, lifted.
            let list_b = EvaluationsList::new(evals);
            let r_ext = EF4::from(r_base);
            let result_b_ext = list_b.compress_ext(r_ext);

            // The results should be identical.
            prop_assert_eq!(result_a_lifted, result_b_ext.as_slice());
        }

        #[test]
        fn prop_compress_dimensions(
            n in 1usize..=10,
            seed in any::<u64>(),
        ) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let num_evals = 1 << n;
            let evals: Vec<F> = (0..num_evals).map(|_| rng.random()).collect();
            let r: F = rng.random();

            let mut list = EvaluationsList::new(evals);
            list.compress(r);

            // After one round of compression, we should have n-1 variables.
            prop_assert_eq!(list.num_variables(), n - 1);
            prop_assert_eq!(list.num_evals(), num_evals / 2);
        }

        #[test]
        fn prop_compress_boundary_challenges(
            n in 2usize..=8,
            seed in any::<u64>(),
        ) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let num_evals = 1 << n;
            let evals: Vec<F> = (0..num_evals).map(|_| rng.random()).collect();

            // Test with r = 0: should select left half
            let mut list_zero = EvaluationsList::new(evals.clone());
            list_zero.compress(F::ZERO);
            prop_assert_eq!(list_zero.num_evals(), num_evals / 2);

            // Test with r = 1: should select right half
            let mut list_one = EvaluationsList::new(evals);
            list_one.compress(F::ONE);
            prop_assert_eq!(list_one.num_evals(), num_evals / 2);

            // Results should be different (unless all evals are the same)
            if list_zero.as_slice() != list_one.as_slice() {
                prop_assert_ne!(list_zero.as_slice(), list_one.as_slice());
            }
        }

        #[test]
        fn prop_compress_multiple_rounds(
            n in 2usize..=8,
            num_rounds in 1usize..=5,
            seed in any::<u64>(),
        ) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let num_evals = 1 << n;
            let evals: Vec<F> = (0..num_evals).map(|_| rng.random()).collect();

            // Generate random challenges, but not more than the number of variables.
            let actual_rounds = num_rounds.min(n);
            let challenges: Vec<F> = (0..actual_rounds).map(|_| rng.random()).collect();

            // Apply multiple rounds of compress.
            let mut list = EvaluationsList::new(evals);
            for &r in &challenges {
                list.compress(r);
            }

            // Verify final dimensions are correct.
            prop_assert_eq!(list.num_variables(), n - actual_rounds);
            prop_assert_eq!(list.num_evals(), 1 << (n - actual_rounds));
        }
    }

    #[test]
    fn test_skip_k1_j1_arbitrary_coords() {
        // SETUP:
        // - n = 2 total variables.
        // - k_skip = 1 (folds 1 variable into a subgroup D of size 2).
        // - j = n - k = 1 (1 remaining hypercube variable).
        // - The polynomial is defined over the domain D × H^1, where D = {g^0, g^1} and H^1 = {0, 1}.

        let k_skip = 1;

        // The evaluations are laid out in natural (lexicographical) row-major order.
        //
        // Conceptual Grid:
        //         y=0     y=1
        //       ┌───────┬───────┐
        // x=g^0 │   7   │  13   │  (Row 0)
        //       ├───────┼───────┤
        // x=g^1 │  19   │  29   │  (Row 1)
        //       └───────┴───────┘
        let evals = vec![
            F::from_u64(7),  // f(g^0, 0)
            F::from_u64(13), // f(g^0, 1)
            F::from_u64(19), // f(g^1, 0)
            F::from_u64(29), // f(g^1, 1)
        ];
        let evals_list = EvaluationsList::new(evals.clone());

        // Test at an arbitrary point not on the grid: (x, y) = (2, 3).
        let x = F::from_u64(2);
        let y = F::from_u64(3);
        let point = MultilinearPoint::new(vec![x, y]);

        // Evaluate the polynomial at the arbitrary point.
        let result = evals_list.evaluate_with_univariate_skip(&point, k_skip);

        // STAGE 1: Univariate Interpolation (Collapsing the Rows)
        //
        // Manually compute the interpolation for each column at the point `x`.
        // The subgroup D for k=1 is {g^0, g^1} which is {1, -1}.
        //
        // For a linear function through points (1, y0) and (-1, y1), the value at x is:
        // f(x) = (y0 + y1)/2 + x*(y0 - y1)/2
        //
        // Column 0 (y=0): Linear interpolation through (1, 7) and (-1, 19)
        let folded_0 = (evals[0] + evals[2]).halve() + (evals[0] - evals[2]).halve() * x;

        // Column 1 (y=1): Linear interpolation through (1, 13) and (-1, 29)
        let folded_1 = (evals[1] + evals[3]).halve() + (evals[1] - evals[3]).halve() * x;

        let folded_evals_manual = vec![folded_0, folded_1];
        assert_eq!(
            folded_evals_manual,
            vec![F::from_u64(1), F::from_u64(5)],
            "Stage 1 manual calculation does not match expected folded values"
        );

        // STAGE 2: Multilinear Interpolation (Evaluating the Result)
        //
        // The `folded_evals` are [p(0), p(1)] for a new 1-variable polynomial p(y).
        // We now evaluate this at y=3 using the formula: p(y) = p(0)*(1-y) + p(1)*y
        let p_at_0 = folded_evals_manual[0];
        let p_at_1 = folded_evals_manual[1];
        let expected = p_at_0 * (F::ONE - y) + p_at_1 * y;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_skip_k2_j1_arbitrary_coords_extension_field() {
        // SETUP:
        // - n = 3 total variables.
        // - k_skip = 2 (folds 2 variables into a subgroup D of size 4).
        // - j = n - k = 1 (1 remaining hypercube variable).
        // - Point is in an extension field EF4.

        let k_skip = 2;

        // The evaluations are laid out in lexicographical (natural) row-major order.
        // The underlying interpolation function expects this simple, intuitive layout.
        //
        // Conceptual Grid (Lexicographical Order):
        //         y=0     y=1
        //       ┌───────┬───────┐
        // x=g^0 │   1   │   2   │
        //       ├───────┼───────┤
        // x=g^1 │   5   │   6   │
        //       ├───────┼───────┤
        // x=g^2 │   3   │   4   │
        //       ├───────┼───────┤
        // x=g^3 │   7   │   8   │
        //       └───────┴───────┘
        let evals = vec![
            F::from_u64(1),
            F::from_u64(2), // evals for g^0
            F::from_u64(5),
            F::from_u64(6), // evals for g^1
            F::from_u64(3),
            F::from_u64(4), // evals for g^2
            F::from_u64(7),
            F::from_u64(8), // evals for g^3
        ];
        let evals_list = EvaluationsList::new(evals.clone());

        // Test at an arbitrary point (x, y) = (5, 7) in the extension field.
        let x = EF4::from_u64(5);
        let y = EF4::from_u64(7);
        let point = MultilinearPoint::new(vec![x, y]);

        // Evaluate the polynomial.
        let result = evals_list.evaluate_with_univariate_skip(&point, k_skip);

        // STAGE 1: Univariate Interpolation
        // The subgroup D for k=2 is {g^0, g^1, g^2, g^3}.
        let g = F::two_adic_generator(k_skip);
        let domain = [F::ONE, g, g.square(), g.square() * g]; // {g^0, g^1, g^2, g^3}

        // Lagrange basis polynomials L_i(x) = product_{j!=i} (x - domain[j]) / (domain[i] - domain[j])
        let mut lagrange_at_x = Vec::with_capacity(4);
        for i in 0..4 {
            let mut l_i = EF4::ONE;
            for j in 0..4 {
                if i != j {
                    l_i *= (x - EF4::from(domain[j]))
                        * (EF4::from(domain[i]) - EF4::from(domain[j])).inverse();
                }
            }
            lagrange_at_x.push(l_i);
        }

        // Get evaluations for each column from the lexicographically ordered `evals` vector.
        //
        // Column 0 (y=0): values are {1, 5, 3, 7} for {g^0, g^1, g^2, g^3}
        let p0_evals = [evals[0], evals[2], evals[4], evals[6]];
        // Column 1 (y=1): values are {2, 6, 4, 8} for {g^0, g^1, g^2, g^3}
        let p1_evals = [evals[1], evals[3], evals[5], evals[7]];

        // Interpolate each column polynomial: p(x) = sum(p(g^i) * L_i(x))
        let folded_0: EF4 = (0..4)
            .map(|i| EF4::from(p0_evals[i]) * lagrange_at_x[i])
            .sum();
        let folded_1: EF4 = (0..4)
            .map(|i| EF4::from(p1_evals[i]) * lagrange_at_x[i])
            .sum();

        // STAGE 2: Multilinear Interpolation
        //
        // Manually evaluate the final 1-variable polynomial at y=7.
        let expected = folded_0 * (EF4::ONE - y) + folded_1 * y;

        assert_eq!(result, expected, "Manual two-stage interpolation failed");
    }

    #[test]
    fn test_skip_all_vars_k2_j0() {
        // SETUP:
        // - n = 2 total variables.
        // - k_skip = 2 (all variables are folded).
        // - j = n - k = 0 (no remaining hypercube variables).
        // - The polynomial is defined only over the subgroup D of size 4.

        let k_skip = 2;

        // Since j=0, the domain is just the subgroup D. The evaluations are a flat list
        // in the natural (lexicographical) order of the subgroup elements.
        //
        // Evaluation Points: { g^0,   g^1,   g^2,   g^3 }
        // Values:            {  10,    20,    30,    40 }
        let evals = vec![
            F::from_u64(10), // f(g^0)
            F::from_u64(20), // f(g^1)
            F::from_u64(30), // f(g^2)
            F::from_u64(40), // f(g^3)
        ];
        let evals_list = EvaluationsList::new(evals.clone());

        // The evaluation point only has one coordinate, `x`, since j=0.
        let x = F::from_u64(5);
        let point = MultilinearPoint::new(vec![x]);

        // Evaluate the polynomial.
        let result = evals_list.evaluate_with_univariate_skip(&point, k_skip);

        // STAGE 1: Univariate Interpolation
        //
        // The logic is identical to the k=2 test above, but for a single column.
        let g = F::two_adic_generator(k_skip);
        let domain = [F::ONE, g, g.square(), g.square() * g];

        // We compute the Lagrange basis polynomials at the evaluation point `x`.
        // L_i(x) = product_{m!=i} (x - domain[m]) / (domain[i] - domain[m])
        let mut lagrange_at_x = Vec::with_capacity(4);
        for i in 0..4 {
            let mut l_i = F::ONE;
            for j in 0..4 {
                if i != j {
                    l_i *= (x - domain[j]) * (domain[i] - domain[j]).inverse();
                }
            }
            lagrange_at_x.push(l_i);
        }

        // The interpolated value is the sum of each evaluation multiplied by its corresponding
        // Lagrange basis polynomial: p(x) = sum( p(g^i) * L_i(x) )
        let expected: F = (0..4).map(|i| evals[i] * lagrange_at_x[i]).sum();

        // STAGE 2: Multilinear Interpolation
        // This stage is trivial because there are no remaining variables (`j=0`).
        //
        // The result from Stage 1 is the final answer.

        assert_eq!(result, expected);
    }

    #[test]
    #[should_panic(expected = "log_skip_size must be greater than 0")]
    fn test_evaluate_with_univariate_skip_panics_on_k0() {
        // Verify that log_skip_size=0 is invalid, as it should use the standard `evaluate` method.
        let evals_list = EvaluationsList::new(vec![F::ONE, F::ZERO]);
        let point = MultilinearPoint::new(vec![F::ONE, F::ZERO]);
        let _ = evals_list.evaluate_with_univariate_skip(&point, 0);
    }

    #[test]
    #[should_panic(expected = "log_skip_size")]
    fn test_evaluate_with_univariate_skip_invalid_k() {
        // Verify that log_skip_size > n causes a panic
        let evals_list = EvaluationsList::new(vec![F::ONE, F::ZERO, F::ONE, F::ZERO]);
        let point = MultilinearPoint::new(vec![F::ONE]);
        let _ = evals_list.evaluate_with_univariate_skip(&point, 3); // n=2, log_skip_size=3 is invalid
    }

    #[test]
    fn test_compute_svo_accumulators_n2_manual() {
        //  TEST SETUP: N=2, n=4
        //
        // This test computes the SVO accumulators for N=2 rounds over a
        // 4-variable polynomial to verify the algorithm's correctness.
        //
        // Variable Split:
        //   - β (prefix, 2 vars): Variables being sum-checked in rounds 0-1
        //   - x_in (2 vars): Inner loop variables (cache-friendly iteration)
        //   - x_out (0 vars): Outer loop variables (parallelized) - NONE in this case
        //
        // Memory Layout:
        //   Variables: β_1 β_0 x_1 x_0  (lexicographic order, LSB on right)
        //              └──┬──┘ └──┬──┘
        //                 β      x_in
        //
        // Polynomial evaluations P(β_1, β_0, x_1, x_0):
        //
        // NOTE: In lexicographic order, the indices map as:
        //   index = β_1·8 + β_0·4 + x_1·2 + x_0
        const N: usize = 2;

        // Define polynomial with simple values for easy manual verification:
        //
        //   Index | β_1 β_0 x_1 x_0 | P(index)
        //   ------|-----------------|----------
        //     0   |  0   0   0   0  |    1
        //     1   |  0   0   0   1  |    2
        //     2   |  0   0   1   0  |    3
        //     3   |  0   0   1   1  |    4
        //     4   |  0   1   0   0  |    5
        //     5   |  0   1   0   1  |    6
        //     6   |  0   1   1   0  |    7
        //     7   |  0   1   1   1  |    8
        //     8   |  1   0   0   0  |    9
        //     9   |  1   0   0   1  |   10
        //    10   |  1   0   1   0  |   11
        //    11   |  1   0   1   1  |   12
        //    12   |  1   1   0   0  |   13
        //    13   |  1   1   0   1  |   14
        //    14   |  1   1   1   0  |   15
        //    15   |  1   1   1   1  |   16
        let poly = EvaluationsList::new((1..=16).map(|i| EF4::from_u64(i)).collect());

        // Define E_in (equality polynomial for x_in = x_1 x_0):
        //
        // For a random point (r_1, r_0), E_in evaluates the equality polynomial
        // at each Boolean assignment:
        //
        //   E_in[00] = (1-r_1)(1-r_0)
        //   E_in[01] = (1-r_1)·r_0
        //   E_in[10] = r_1·(1-r_0)
        //   E_in[11] = r_1·r_0
        //
        // We use simple values for manual computation:
        let r1 = EF4::from_u64(2);
        let r0 = EF4::from_u64(3);

        let e_in = EvaluationsList::new(vec![
            (EF4::ONE - r1) * (EF4::ONE - r0),
            (EF4::ONE - r1) * r0,
            r1 * (EF4::ONE - r0),
            r1 * r0,
        ]);

        // For N=2, we need E_out for rounds 0 and 1.
        //
        // Round 0: E_out[0] corresponds to β_0 (the last prefix variable)
        //   E_out[0][0] = (1-s_0)
        //   E_out[0][1] = s_0
        //
        // Round 1: E_out[1] corresponds to β_1 (the first prefix variable)
        //   E_out[1][0] = (1-s_1)
        //   E_out[1][1] = s_1
        let s0 = EF4::from_u64(5);
        let s1 = EF4::from_u64(7);

        let e_out = [
            EvaluationsList::new(vec![EF4::ONE - s0, s0]), // Round 0: [(1-s_0), s_0]
            EvaluationsList::new(vec![EF4::ONE - s1, s1]), // Round 1: [(1-s_1), s_1]
        ];

        //  CALL FUNCTION UNDER TEST
        let result = poly.compute_svo_accumulators::<EF4, N>(&e_in, &e_out);

        //  MANUAL COMPUTATION: Round 0 Accumulators
        //
        // Round 0 computes accumulators for the first variable β_0.
        //
        // For each value of u (the next variable to be evaluated):
        //   A_0[u] = Σ_{x_in} E_in[x_in] · P(u, x_in)
        //
        // Since β has 2 bits and we're in round 0, u ranges over {00, 01}.
        //
        // A_0[00] = Σ_{x_1 x_0} E_in[x_1 x_0] · P(0,0,x_1,x_0)
        //         = E_in[00]·P(0,0,0,0) + E_in[01]·P(0,0,0,1) + E_in[10]·P(0,0,1,0) + E_in[11]·P(0,0,1,1)
        let a0_00 = e_in.0[0] * poly.0[0]
            + e_in.0[1] * poly.0[1]
            + e_in.0[2] * poly.0[2]
            + e_in.0[3] * poly.0[3];

        // A_0[01] = Σ_{x_1 x_0} E_in[x_1 x_0] · P(0,1,x_1,x_0)
        //         = E_in[00]·P(0,1,0,0) + E_in[01]·P(0,1,0,1) + E_in[10]·P(0,1,1,0) + E_in[11]·P(0,1,1,1)
        let a0_01 = e_in.0[0] * poly.0[4]
            + e_in.0[1] * poly.0[5]
            + e_in.0[2] * poly.0[6]
            + e_in.0[3] * poly.0[7];

        // A_0[10] = Σ_{x_1 x_0} E_in[x_1 x_0] · P(1,0,x_1,x_0)
        //         = E_in[00]·P(1,0,0,0) + E_in[01]·P(1,0,0,1) + E_in[10]·P(1,0,1,0) + E_in[11]·P(1,0,1,1)
        let a0_10 = e_in.0[0] * poly.0[8]
            + e_in.0[1] * poly.0[9]
            + e_in.0[2] * poly.0[10]
            + e_in.0[3] * poly.0[11];

        // A_0[11] = Σ_{x_1 x_0} E_in[x_1 x_0] · P(1,1,x_1,x_0)
        //         = E_in[00]·P(1,1,0,0) + E_in[01]·P(1,1,0,1) + E_in[10]·P(1,1,1,0) + E_in[11]·P(1,1,1,1)
        let a0_11 = e_in.0[0] * poly.0[12]
            + e_in.0[1] * poly.0[13]
            + e_in.0[2] * poly.0[14]
            + e_in.0[3] * poly.0[15];

        // The round 0 polynomial is obtained by multiplying by E_out[0]:
        //
        // Round 0 evaluates over β_1 (the MSB of β):
        //   Round0[0] (for β_1=0) = A_0[00]·E_out[0][0] + A_0[01]·E_out[0][1]
        //   Round0[1] (for β_1=1) = A_0[10]·E_out[0][0] + A_0[11]·E_out[0][1]
        let round0_0 = a0_00 * e_out[0].0[0] + a0_01 * e_out[0].0[1];
        let round0_1 = a0_10 * e_out[0].0[0] + a0_11 * e_out[0].0[1];

        // Verify against computed result (Round 0 is at index 0-1):
        let computed_round0 = result.at_round(0);
        assert_eq!(computed_round0[0], round0_0);
        assert_eq!(computed_round0[1], round0_1);

        // MANUAL COMPUTATION: Round 1 Accumulators
        //
        // Round 1 evaluates over β_0 (the LSB of β).
        //
        // With block_size=1, each accumulator uses a single temp value:
        //
        //   Round1[0] = A_0[00]·E_out[1][0]
        //   Round1[1] = A_0[01]·E_out[1][0]
        //   Round1[2] = A_0[10]·E_out[1][0]
        //   Round1[3] = A_0[11]·E_out[1][0]
        let round1_0 = a0_00 * e_out[1].0[0];
        let round1_1 = a0_01 * e_out[1].0[0];
        let round1_2 = a0_10 * e_out[1].0[0];
        let round1_3 = a0_11 * e_out[1].0[0];

        // Verify against computed result (Round 1 is at index 2-5):
        let computed_round1 = result.at_round(1);
        assert_eq!(computed_round1[0], round1_0);
        assert_eq!(computed_round1[1], round1_1);
        assert_eq!(computed_round1[2], round1_2);
        assert_eq!(computed_round1[3], round1_3);
    }

    #[test]
    fn test_compute_svo_accumulators_n3_six_variables() {
        // ═══════════════════════════════════════════════════════════════════════════
        //  TEST SETUP: N=3, n=6
        // ═══════════════════════════════════════════════════════════════════════════
        //
        // This test verifies N=3 (three-round precomputation) on a 6-variable polynomial.
        //
        // Variable Split:
        //   - β (3 vars): β_2 β_1 β_0 - prefix variables for rounds 0-2
        //   - x_in (3 vars): x_2 x_1 x_0 - inner loop variables
        //   - x_out (0 vars): None - no parallelization in this small example
        //
        // We use a simple polynomial where P(i) = i + 1 for manual verification.
        const N: usize = 3;

        let poly = EvaluationsList::new((1..=64).map(|i| EF4::from_u64(i)).collect());

        // Define E_in (equality polynomial for x_in = x_2 x_1 x_0):
        //
        // For a random point (r_2, r_1, r_0), E_in evaluates the equality polynomial
        // at each Boolean assignment:
        //
        //   E_in[000] = (1-r_2)(1-r_1)(1-r_0)
        //   E_in[001] = (1-r_2)(1-r_1)·r_0
        //   E_in[010] = (1-r_2)·r_1·(1-r_0)
        //   E_in[011] = (1-r_2)·r_1·r_0
        //   E_in[100] = r_2·(1-r_1)(1-r_0)
        //   E_in[101] = r_2·(1-r_1)·r_0
        //   E_in[110] = r_2·r_1·(1-r_0)
        //   E_in[111] = r_2·r_1·r_0
        //
        // We use simple values for manual computation:
        let r2 = EF4::from_u64(2);
        let r1 = EF4::from_u64(3);
        let r0 = EF4::from_u64(5);

        let e_in = EvaluationsList::new(vec![
            (EF4::ONE - r2) * (EF4::ONE - r1) * (EF4::ONE - r0),
            (EF4::ONE - r2) * (EF4::ONE - r1) * r0,
            (EF4::ONE - r2) * r1 * (EF4::ONE - r0),
            (EF4::ONE - r2) * r1 * r0,
            r2 * (EF4::ONE - r1) * (EF4::ONE - r0),
            r2 * (EF4::ONE - r1) * r0,
            r2 * r1 * (EF4::ONE - r0),
            r2 * r1 * r0,
        ]);

        // For N=3, we need E_out for rounds 0, 1, and 2.
        //
        // Round 0: E_out[0] corresponds to β_0 (the last prefix variable)
        //   E_out[0][0] = (1-s_0)
        //   E_out[0][1] = s_0
        //
        // Round 1: E_out[1] corresponds to β_1 (the middle prefix variable)
        //   E_out[1][0] = (1-s_1)
        //   E_out[1][1] = s_1
        //
        // Round 2: E_out[2] corresponds to β_2 (the first prefix variable)
        //   E_out[2][0] = (1-s_2)
        //   E_out[2][1] = s_2
        let s0 = EF4::from_u64(7);
        let s1 = EF4::from_u64(11);
        let s2 = EF4::from_u64(13);

        let e_out = [
            EvaluationsList::new(vec![EF4::ONE - s0, s0]), // Round 0
            EvaluationsList::new(vec![EF4::ONE - s1, s1]), // Round 1
            EvaluationsList::new(vec![EF4::ONE - s2, s2]), // Round 2
        ];

        //  CALL FUNCTION UNDER TEST
        let result = poly.compute_svo_accumulators::<EF4, N>(&e_in, &e_out);

        //  MANUAL COMPUTATION: Verify Structure
        //
        // For N=3:
        //   - Round 0: 2^(0+1) = 2 elements
        //   - Round 1: 2^(1+1) = 4 elements
        //   - Round 2: 2^(2+1) = 8 elements
        //   - Total: 2 + 4 + 8 = 14 elements

        assert_eq!(result.at_round(0).len(), 2);
        assert_eq!(result.at_round(1).len(), 4);
        assert_eq!(result.at_round(2).len(), 8);

        //  MANUAL COMPUTATION: Verify Round 2 Accumulators
        //
        // Round 2 evaluates over β_0 (the LSB of β). With block_size=1, each
        // accumulator uses a single temp value.
        //
        // First, compute intermediate accumulators A_0[β] for all 8 β values:
        //
        // A_0[000] = Σ_{x_2 x_1 x_0} E_in[x_2 x_1 x_0] · P(000, x_2 x_1 x_0)
        //          = E_in[000]·P(0) + E_in[001]·P(1) + ... + E_in[111]·P(7)
        let a0_000: EF4 = (0..8).map(|i| e_in.0[i] * poly.0[i]).sum();

        // A_0[001] = Σ_{x_2 x_1 x_0} E_in[x_2 x_1 x_0] · P(001, x_2 x_1 x_0)
        //          = E_in[000]·P(8) + E_in[001]·P(9) + ... + E_in[111]·P(15)
        let a0_001: EF4 = (0..8).map(|i| e_in.0[i] * poly.0[8 + i]).sum();
        assert_eq!(a0_001, EF4::from_u64(28));

        // Similarly for the other β values (we'll compute all for full verification):
        let a0_010: EF4 = (0..8).map(|i| e_in.0[i] * poly.0[16 + i]).sum();
        let a0_011: EF4 = (0..8).map(|i| e_in.0[i] * poly.0[24 + i]).sum();
        let a0_100: EF4 = (0..8).map(|i| e_in.0[i] * poly.0[32 + i]).sum();
        let a0_101: EF4 = (0..8).map(|i| e_in.0[i] * poly.0[40 + i]).sum();
        let a0_110: EF4 = (0..8).map(|i| e_in.0[i] * poly.0[48 + i]).sum();
        let a0_111: EF4 = (0..8).map(|i| e_in.0[i] * poly.0[56 + i]).sum();

        // Round 2 uses block_size=1, so each accumulator is:
        //   Round2[i] = A_0[i] · E_out[2][0]  (only uses E_out[2][0])
        //
        // Round2[0] = A_0[000] · E_out[2][0] = 20 · (-12) = -240
        // Round2[1] = A_0[001] · E_out[2][0] = 28 · (-12) = -336
        // ... and so on
        let expected_round2_0 = a0_000 * e_out[2].0[0];
        let expected_round2_1 = a0_001 * e_out[2].0[0];
        let expected_round2_2 = a0_010 * e_out[2].0[0];
        let expected_round2_3 = a0_011 * e_out[2].0[0];
        let expected_round2_4 = a0_100 * e_out[2].0[0];
        let expected_round2_5 = a0_101 * e_out[2].0[0];
        let expected_round2_6 = a0_110 * e_out[2].0[0];
        let expected_round2_7 = a0_111 * e_out[2].0[0];

        let computed_round2 = result.at_round(2);
        assert_eq!(computed_round2[0], expected_round2_0);
        assert_eq!(computed_round2[1], expected_round2_1);
        assert_eq!(computed_round2[2], expected_round2_2);
        assert_eq!(computed_round2[3], expected_round2_3);
        assert_eq!(computed_round2[4], expected_round2_4);
        assert_eq!(computed_round2[5], expected_round2_5);
        assert_eq!(computed_round2[6], expected_round2_6);
        assert_eq!(computed_round2[7], expected_round2_7);

        assert_eq!(computed_round2.len(), 8);
    }

    #[test]
    fn test_compute_svo_accumulators_extension_field() {
        //  TEST: Extension Field Correctness
        //
        // This test verifies that compute_svo_accumulators works correctly when:
        // - the polynomial is over the base field
        // - the equality polynomials are over the extension field
        //
        // Setup: N=2, n=4
        //
        // Variable ordering: β_1 β_0 x_1 x_0
        //
        // Polynomial table (base field):
        //
        //   Index | β_1 β_0 x_1 x_0 | P(index)
        //   ------|-----------------|----------
        //     0   |  0   0   0   0  |    1
        //     1   |  0   0   0   1  |    2
        //     2   |  0   0   1   0  |    3
        //     3   |  0   0   1   1  |    4
        //     4   |  0   1   0   0  |    5
        //     5   |  0   1   0   1  |    6
        //     6   |  0   1   1   0  |    7
        //     7   |  0   1   1   1  |    8
        //     8   |  1   0   0   0  |    9
        //     9   |  1   0   0   1  |   10
        //    10   |  1   0   1   0  |   11
        //    11   |  1   0   1   1  |   12
        //    12   |  1   1   0   0  |   13
        //    13   |  1   1   0   1  |   14
        //    14   |  1   1   1   0  |   15
        //    15   |  1   1   1   1  |   16
        const N: usize = 2;

        // Polynomial with base field values
        let poly_f = EvaluationsList::new((1..=16).map(|i| F::from_u64(i)).collect());

        //  E_IN SETUP
        //
        // We use r_1, r_0 (in extension field)
        //
        // E_in factorizes over x_in = (x_1, x_0):
        //
        //   E_in[x_1 x_0] = E_in[x_1, x_0] = (1-x_1·r_1 - (1-x_1)·(1-r_1)) · (1-x_0·r_0 - (1-x_0)·(1-r_0))
        //
        // Expanded form:
        //   E_in[0 0] = (1-r_1) · (1-r_0)
        //   E_in[0 1] = (1-r_1) ·    r_0
        //   E_in[1 0] =    r_1  · (1-r_0)
        //   E_in[1 1] =    r_1  ·    r_0
        let r1 = EF4::from_u64(2);
        let r0 = EF4::from_u64(3);

        let e_in = EvaluationsList::new(vec![
            (EF4::ONE - r1) * (EF4::ONE - r0),
            (EF4::ONE - r1) * r0,
            r1 * (EF4::ONE - r0),
            r1 * r0,
        ]);

        //  E_OUT SETUP
        //
        // We use s_0, s_1 (in extension field)
        //
        // E_out[0] factorizes over x_0 (the first outer variable after β):
        //   E_out[0][0] = 1 - s_0
        //   E_out[0][1] =     s_0
        //
        // E_out[1] factorizes over β_0 (the last β variable):
        //   E_out[1][0] = 1 - s_1
        //   E_out[1][1] =     s_1
        let s0 = EF4::from_u64(5);
        let s1 = EF4::from_u64(7);

        let e_out = [
            EvaluationsList::new(vec![EF4::ONE - s0, s0]),
            EvaluationsList::new(vec![EF4::ONE - s1, s1]),
        ];

        //  CALL FUNCTION
        let result = poly_f.compute_svo_accumulators::<EF4, N>(&e_in, &e_out);

        //  MANUAL COMPUTATION
        //
        // Step 1: Compute intermediate accumulators A_0[β_1 β_0]
        //
        // A_0[β_1 β_0] = Σ_{x_in} E_in[x_in] · P[β || x_in || 0...0]
        //
        // For each β value, we sum over x_in ∈ {00, 01, 10, 11}:
        //
        // A_0[0 0] = E_in[00]·P[0] + E_in[01]·P[1] + E_in[10]·P[2] + E_in[11]·P[3]
        // A_0[0 1] = E_in[00]·P[4] + E_in[01]·P[5] + E_in[10]·P[6] + E_in[11]·P[7]
        // A_0[1 0] = E_in[00]·P[8] + E_in[01]·P[9] + E_in[10]·P[10] + E_in[11]·P[11]
        // A_0[1 1] = E_in[00]·P[12] + E_in[01]·P[13] + E_in[10]·P[14] + E_in[11]·P[15]

        //  ROUND 0 COMPUTATION
        //
        // Round 0 (r=0) has block_size = 2^(N-1-0) = 2^1 = 2.
        // We group pairs of A_0 accumulators and sum over β_0.
        //
        // Round_0[β_1] = Σ_{β_0 ∈ {0,1}} A_0[β_1, β_0] · E_out[0][β_0]
        //
        // Round0[0] = A_0[0,0]·E_out[0][0] + A_0[0,1]·E_out[0][1]
        // Round0[1] = A_0[1,0]·E_out[0][0] + A_0[1,1]·E_out[0][1]
        assert_eq!(result.at_round(0)[0], EF4::from_u64(28));
        assert_eq!(result.at_round(0)[1], EF4::from_u64(36));

        //  ROUND 1 COMPUTATION
        //
        // Round 1 (r=1) has block_size = 2^(N-1-1) = 2^0 = 1.
        //
        // With block_size=1, each accumulator is evaluated individually with only the first
        // element of E_out[1] (since we're evaluating at a specific point, not summing).
        //
        // Round1[0] = A_0[0,0] · E_out[1][0]
        // Round1[1] = A_0[0,1] · E_out[1][0]
        // Round1[2] = A_0[1,0] · E_out[1][0]
        // Round1[3] = A_0[1,1] · E_out[1][0]
        assert_eq!(result.at_round(1)[0], EF4::from_u64(8) * (EF4::ONE - s1));
        assert_eq!(result.at_round(1)[1], EF4::from_u64(12) * (EF4::ONE - s1));
        assert_eq!(result.at_round(1)[2], EF4::from_u64(16) * (EF4::ONE - s1));
        assert_eq!(result.at_round(1)[3], EF4::from_u64(20) * (EF4::ONE - s1));
    }

    #[test]
    fn test_compute_svo_accumulators_edge_case_all_ones() {
        //  TEST: Edge Case - All Ones
        //
        // This test verifies behavior when:
        //   - Polynomial evaluations are all 1
        //   - Equality polynomials have specific uniform structure
        //
        // This is an edge case that stresses the accumulation logic.
        const N: usize = 2;

        // Polynomial: all evaluations = 1
        let poly = EvaluationsList::new(vec![EF4::ONE; 16]);

        // E_in: all equal values
        let e_in = EvaluationsList::new(vec![EF4::from_u64(2); 4]);

        // E_out: simple alternating pattern
        let e_out = [
            EvaluationsList::new(vec![EF4::from_u64(3), EF4::from_u64(5)]),
            EvaluationsList::new(vec![EF4::from_u64(7), EF4::from_u64(11)]),
        ];

        //  CALL FUNCTION
        let result = poly.compute_svo_accumulators::<EF4, N>(&e_in, &e_out);

        //  VERIFY: Manual computation
        //
        // For round 0:
        //   Each accumulator A_0[β] = Σ_{x_in} E_in[x_in] · 1
        //
        // So all four intermediate accumulators are 8.
        //
        // Round 0 polynomial:
        //   Round0[0] = 8·3 + 8·5 = 24 + 40 = 64
        //   Round0[1] = 8·3 + 8·5 = 24 + 40 = 64
        let expected_round0_0 = EF4::from_u64(8) * e_out[0].0[0] + EF4::from_u64(8) * e_out[0].0[1];
        let expected_round0_1 = EF4::from_u64(8) * e_out[0].0[0] + EF4::from_u64(8) * e_out[0].0[1];

        assert_eq!(result.at_round(0)[0], expected_round0_0);
        assert_eq!(result.at_round(0)[1], expected_round0_1);
        assert_eq!(result.at_round(0)[0], EF4::from_u64(64));
        assert_eq!(result.at_round(0)[1], EF4::from_u64(64));

        // Round 1 (with block_size=1, each uses only e_out[1][0]):
        //   Round1[0] = 8·7 = 56
        //   Round1[1] = 8·7 = 56
        //   Round1[2] = 8·7 = 56
        //   Round1[3] = 8·7 = 56
        assert_eq!(result.at_round(1)[0], EF4::from_u64(56));
        assert_eq!(result.at_round(1)[1], EF4::from_u64(56));
        assert_eq!(result.at_round(1)[2], EF4::from_u64(56));
        assert_eq!(result.at_round(1)[3], EF4::from_u64(56));
    }
}
