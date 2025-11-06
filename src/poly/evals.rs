use itertools::Itertools;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_interpolation::interpolate_subgroup;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::eq_batch::{eval_eq_base_batch, eval_eq_batch};
use tracing::instrument;

use super::{coeffs::CoefficientList, multilinear::MultilinearPoint, wavelet::Radix2WaveletKernel};
use crate::{constant::MLE_RECURSION_THRESHOLD, utils::uninitialized_vec};

const PARALLEL_THRESHOLD: usize = 4096;

/// Represents a multilinear polynomial `f` in `n` variables, stored by its evaluations
/// over the boolean hypercube `{0,1}^n`.
///
/// The inner vector stores function evaluations at points of the hypercube in lexicographic
/// order. The number of variables `n` is inferred from the length of this vector, where
/// `self.len() = 2^n`.
#[derive(Debug, Clone, Eq, PartialEq)]
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
        let folding_factor = folding_randomness.num_variables();
        let evals = self
            .0
            .par_chunks_exact(1 << folding_factor)
            .map(|ev| eval_multilinear(ev, folding_randomness))
            .collect();

        EvaluationsList(evals)
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
    pub fn iter(&self) -> std::slice::Iter<'_, F> {
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

    /// Compresses a list of evaluations in-place using a random challenge.
    ///
    /// ## Arguments
    /// * `evals`: A mutable reference to an `EvaluationsList<F>`, which will be modified in-place.
    /// * `r`: A value from the field `F`, used as the random folding challenge.
    ///
    /// ## Mathematical Formula
    /// The compression is achieved by applying the following formula to pairs of evaluations:
    /// ```text
    ///     p'(X_2, ..., X_n) = (p(1, X_2, ..., X_n) - p(0, X_2, ..., X_n)) \cdot r + p(0, X_2, ..., X_n)
    /// ```
    #[inline]
    pub fn compress(&mut self, r: F) {
        // Ensure the polynomial is not a constant (i.e., has variables to fold).
        assert_ne!(self.num_variables(), 0);

        // For large inputs, we use a parallel, out-of-place strategy.
        if self.num_evals() >= PARALLEL_THRESHOLD {
            // Define the folding operation for a pair of elements.
            let fold = |slice: &[F]| -> F { r * (slice[1] - slice[0]) + slice[0] };
            // Execute the fold in parallel and collect into a new vector.
            let folded = self.0.par_chunks_exact(2).map(fold).collect();
            // Replace the old evaluations with the new, folded evaluations.
            self.0 = folded;
        } else {
            // For smaller inputs, we use a sequential, in-place strategy.
            let mid = self.num_evals() / 2;
            for i in 0..mid {
                let p0 = self.0[2 * i];
                let p1 = self.0[2 * i + 1];
                self.0[i] = r * (p1 - p0) + p0;
            }
            self.0.truncate(mid);
        }
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
    ///
    /// > **Note: `compress_svo` vs. `compress`**
    /// >
    /// > - **`compress_svo` (this):** Folds the **first** variable ($X_1$).
    /// >   Accesses `[i]` and `[mid + i]`.
    /// >
    /// > - **`compress`:** Folds the **last** variable ($X_n$).
    /// >   Accesses `[2i]` and `[2i + 1]`.
    ///
    /// ## Context: Sumcheck with Small-Value Optimization (SVO)
    ///
    /// In the SVO paper, the sumcheck prover has two distinct phases:
    ///
    /// 1.  **SVO Rounds (Rounds 1 to $l_0$):**
    ///     The prover uses precomputed "accumulators" and interpolation.
    ///     The evaluation list is not folded.
    ///     **`compress_svo` is NOT used here.**
    ///
    /// 2.  **Standard Rounds (Rounds $l_0+1$ to $l$):**
    ///     After the SVO rounds, the prover switches over to the classical
    ///     sumcheck protocol. For every subsequent round $i$, it must fold
    ///     the *next* variable $X_i$ using the challenge $r_i$.
    ///     **`compress_svo` IS used for every round in this phase.**
    #[inline]
    pub fn compress_svo(&mut self, r: F) {
        assert_ne!(self.num_variables(), 0);
        let mid = self.num_evals() / 2;
        // For large inputs, we use a parallel strategy.
        if self.num_evals() >= PARALLEL_THRESHOLD {
            // Split into mutable left half and immutable right half.
            let (left, right) = self.0.split_at_mut(mid);
            // Process each pair (left[i], right[i]) in parallel.
            left.par_iter_mut()
                .zip(right.par_iter())
                .for_each(|(p0_mut, &p1)| {
                    // Compute the folded value and write it back to the left half.
                    //
                    // Formula: p(r, x') = r * (p(1, x') - p(0, x')) + p(0, x')
                    *p0_mut += r * (p1 - *p0_mut);
                });
        } else {
            // For smaller inputs, we use a sequential strategy.
            for i in 0..mid {
                // p(0, x'_i) from left half
                let p0 = self.0[i];
                // p(1, x'_i) from right half
                let p1 = self.0[mid + i];
                // Compute the folded value in-place.
                self.0[i] += r * (p1 - p0);
            }
        }
        // Remove the right half, which is no longer needed.
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

        // Fold between base and extension field elements
        let fold = |slice: &[F]| -> EF { r * (slice[1] - slice[0]) + slice[0] };

        // Threshold below which sequential computation is faster
        //
        // This was chosen based on experiments with the `compress` function.
        // It is possible that the threshold can be tuned further.
        let folded = if self.num_evals() >= PARALLEL_THRESHOLD {
            self.0.par_chunks_exact(2).map(fold).collect()
        } else {
            self.0.chunks_exact(2).map(fold).collect()
        };

        EvaluationsList::new(folded)
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
}

impl<'a, F> IntoIterator for &'a EvaluationsList<F> {
    type Item = &'a F;
    type IntoIter = std::slice::Iter<'a, F>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<F> IntoIterator for EvaluationsList<F> {
    type Item = F;
    type IntoIter = std::vec::IntoIter<F>;

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
                    let work_size: usize = (1 << 15) / std::mem::size_of::<F>();
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
    use p3_baby_bear::BabyBear;
    use p3_field::{
        PrimeCharacteristicRing, PrimeField64, TwoAdicField, extension::BinomialExtensionField,
    };
    use proptest::prelude::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};

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
        // Variables: X₁, X₂
        // Coefficients ordered in lexicographic order: (X₁, X₂)
        //
        // - coeffs[0] → constant term
        // - coeffs[1] → X₂ term
        // - coeffs[2] → X₁ term
        // - coeffs[3] → X₁·X₂ term
        //
        // Thus, the polynomial is:
        //
        //   f(X₁, X₂) = c0 + c1·X₂ + c2·X₁ + c3·X₁·X₂
        //
        // where:
        let c0 = F::from_u64(5); // constant
        let c1 = F::from_u64(6); // X₂ coefficient
        let c2 = F::from_u64(7); // X₁ coefficient
        let c3 = F::from_u64(8); // X₁·X₂ coefficient
        //
        // So concretely:
        //
        //   f(X₁, X₂) = 5 + 6·X₂ + 7·X₁ + 8·X₁·X₂
        let coeffs = CoefficientList::new(vec![c0, c1, c2, c3]);

        // Convert coefficients to evaluations via the wavelet transform
        let evals = coeffs.to_evaluations();

        // Choose evaluation point:
        //
        // Let's pick (x₁, x₂) = (2, 1)
        let x1 = F::from_u64(2);
        let x2 = F::from_u64(1);
        let coords = MultilinearPoint::new(vec![x1, x2]);

        // Manually compute the expected value step-by-step:
        //
        // Reminder:
        //   f(X₁, X₂) = 5 + 6·X₂ + 7·X₁ + 8·X₁·X₂
        //
        // Substituting (X₁, X₂):
        let expected = c0 + c1 * x2 + c2 * x1 + c3 * x1 * x2;

        // Now evaluate using the function under test
        let result = evals.evaluate_hypercube(&coords);

        // Check that it matches the manual computation
        assert_eq!(result, expected);
    }

    #[test]
    fn test_evaluate_3_variables() {
        // Define a multilinear polynomial in 3 variables: X₀, X₁, X₂
        //
        // Coefficients ordered lex in index order:
        //
        // - coeffs[0] → constant term
        // - coeffs[1] → X₂
        // - coeffs[2] → X₁
        // - coeffs[3] → X₁·X₂
        // - coeffs[4] → X₀
        // - coeffs[5] → X₀·X₂
        // - coeffs[6] → X₀·X₁
        // - coeffs[7] → X₀·X₁·X₂
        //
        // Thus:
        //    f(X₀,X₁,X₂) = c0 + c1·X₂ + c2·X₁ + c3·X₁·X₂
        //                + c4·X₀ + c5·X₀·X₂ + c6·X₀·X₁ + c7·X₀·X₁·X₂
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

        // Pick point: (x₀,x₁,x₂) = (2, 3, 4)
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
        // Define a multilinear polynomial in 3 variables: X₀, X₁, X₂
        //
        // Coefficients ordered lex in index order:
        //
        // - coeffs[0] → constant term
        // - coeffs[1] → X₂ term
        // - coeffs[2] → X₁ term
        // - coeffs[3] → X₁·X₂ term
        // - coeffs[4] → X₀ term
        // - coeffs[5] → X₀·X₂ term
        // - coeffs[6] → X₀·X₁ term
        // - coeffs[7] → X₀·X₁·X₂ term
        //
        // Thus:
        //    f(X₀,X₁,X₂) = c0 + c1·X₂ + c2·X₁ + c3·X₁·X₂
        //                + c4·X₀ + c5·X₀·X₂ + c6·X₀·X₁ + c7·X₀·X₁·X₂
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

        // Choose evaluation point: (x₀, x₁, x₂) = (2, 3, 4)
        //
        // Here we lift into the extension field EF4
        let x0 = EF4::from_u64(2);
        let x1 = EF4::from_u64(3);
        let x2 = EF4::from_u64(4);

        let point = MultilinearPoint::new(vec![x0, x1, x2]);

        // Manually compute expected value
        //
        // Substituting (X₀,X₁,X₂) = (2,3,4) into:
        //
        //   f(X₀,X₁,X₂) = 1
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

        // Try folding at every possible prefix of the randomness vector: k = 0 to n-1
        for k in 0..num_variables {
            // Use the first k values as the fold coordinates (we will substitute those)
            let fold_part = randomness[0..k].to_vec();

            // The remaining coordinates are used as the evaluation input into the folded poly
            let eval_part = randomness[k..randomness.len()].to_vec();

            // Convert to a MultilinearPoint (in EF) for folding
            let fold_random = MultilinearPoint::new(fold_part.clone());

            // Reconstruct the full point (x₀, ..., xₙ₋₁) = [eval_part || fold_part]
            // Used to evaluate the original uncompressed polynomial
            let eval_point = MultilinearPoint::new([eval_part.clone(), fold_part].concat());

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
                evals_list.evaluate_hypercube(&eval_point)
            );

            // Compare with the coefficient list equivalent
            assert_eq!(
                folded_coeffs.evaluate(&MultilinearPoint::new(eval_part)),
                evals_list.evaluate_hypercube(&eval_point)
            );
        }
    }

    #[test]
    fn test_fold_with_extension_one_var() {
        // Define a 2-variable polynomial:
        // f(X₀, X₁) = 1 + 2·X₁ + 3·X₀ + 4·X₀·X₁
        let coeffs = vec![
            F::from_u64(1), // constant
            F::from_u64(2), // X₁
            F::from_u64(3), // X₀
            F::from_u64(4), // X₀·X₁
        ];
        let poly = CoefficientList::new(coeffs);

        // Convert coefficients into an EvaluationsList (for testing the fold on evals)
        let evals_list: EvaluationsList<F> = poly.clone().to_evaluations();

        // We fold over the last variable (X₁) by setting X₁ = 5 in EF4
        let r1 = EF4::from_u64(5);

        // Perform the fold: f(X₀, 5) becomes a new function g(X₀)
        let folded = evals_list.fold(&MultilinearPoint::new(vec![r1]));

        // For 10 test points x₀ = 0, 1, ..., 9
        for x0_f in 0..10 {
            // Lift to EF4 for extension-field evaluation
            let x0 = EF4::from_u64(x0_f);

            // Construct the full point (x₀, X₁ = 5)
            let full_point = MultilinearPoint::new(vec![x0, r1]);

            // Construct folded point (x₀)
            let folded_point = MultilinearPoint::new(vec![x0]);

            // Evaluate original poly at (x₀, 5)
            let expected = poly.evaluate(&full_point);

            // Evaluate folded poly at x₀
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
        let mut rng = StdRng::seed_from_u64(42);

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
        let point: &[F] = &[];
        let value = F::from_u64(42);
        let evals_list = EvaluationsList::new_from_point(point, value);

        // For n=0, the hypercube has one point, and the `eq` polynomial is the constant 1.
        // The result should be a list with a single element: `value`.
        assert_eq!(evals_list.num_variables(), 0);
        assert_eq!(evals_list.as_slice(), &[value]);
    }

    #[test]
    fn test_new_from_point_one_var() {
        let p0 = F::from_u64(7);
        let point = &[p0];
        let value = F::from_u64(3);
        let evals_list = EvaluationsList::new_from_point(point, value);

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
    fn test_compress() {
        let initial_evals: Vec<F> = (1..=8).map(F::from_u64).collect();
        let mut evals_list = EvaluationsList::new(initial_evals);
        let r = F::from_u64(10);

        // Manually compute the expected folded values using the formula:
        // p' = r * (p1 - p0) + p0
        let expected = vec![
            r * (F::from_u64(2) - F::from_u64(1)) + F::from_u64(1),
            r * (F::from_u64(4) - F::from_u64(3)) + F::from_u64(3),
            r * (F::from_u64(6) - F::from_u64(5)) + F::from_u64(5),
            r * (F::from_u64(8) - F::from_u64(7)) + F::from_u64(7),
        ];

        // The method modifies the list in-place.
        evals_list.compress(r);

        assert_eq!(evals_list.num_variables(), 2);
        assert_eq!(evals_list.num_evals(), 4);
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
    fn test_compress_svo_basic() {
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

        evals_list.compress_svo(r);

        // After compression, the dimensions are:
        // - n = 2 variables
        // - num_evals = 4 evaluations
        assert_eq!(evals_list.num_variables(), 2);
        assert_eq!(evals_list.num_evals(), 4);
        assert_eq!(evals_list.as_slice(), &expected);
    }

    #[test]
    fn test_compress_svo_folds_first_variable() {
        // Verify that compress_svo folds the FIRST variable (not the last like compress).
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
        let initial_evals = vec![p_000, p_001, p_010, p_011, p_100, p_101, p_110, p_111];
        let mut evals_svo = EvaluationsList::new(initial_evals.clone());
        let r = F::from_u64(10);

        evals_svo.compress_svo(r);

        // Expected: folding first variable X_1
        //
        // - result[0] = p(r, 0, 0) = r * (p(1,0,0) - p(0,0,0)) + p(0,0,0)
        // - result[1] = p(r, 0, 1) = r * (p(1,0,1) - p(0,0,1)) + p(0,0,1)
        // - result[2] = p(r, 1, 0) = r * (p(1,1,0) - p(0,1,0)) + p(0,1,0)
        // - result[3] = p(r, 1, 1) = r * (p(1,1,1) - p(0,1,1)) + p(0,1,1)
        let expected = vec![
            r * (p_100 - p_000) + p_000,
            r * (p_101 - p_001) + p_001,
            r * (p_110 - p_010) + p_010,
            r * (p_111 - p_011) + p_011,
        ];

        assert_eq!(evals_svo.as_slice(), &expected);

        // Verify this is different from compress (which folds last variable)
        let mut evals_compress = EvaluationsList::new(initial_evals);
        evals_compress.compress(r);
        assert_ne!(evals_svo.as_slice(), evals_compress.as_slice());
    }

    #[test]
    fn test_compress_svo_parallel_path() {
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

        evals_list.compress_svo(r);

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
    #[should_panic]
    fn test_compress_svo_panics_on_constant() {
        // A constant polynomial has 0 variables and cannot be compressed.
        let mut evals_list = EvaluationsList::new(vec![F::from_u64(42)]);
        evals_list.compress_svo(F::ONE); // This should panic.
    }

    #[test]
    fn test_compress_svo_multiple_rounds() {
        // Test multiple rounds of compression, as would happen in reality.
        //
        // Each round folds the first variable of the current polynomial.
        let initial_evals: Vec<F> = (1..=16).map(F::from_u64).collect();
        let mut evals_list = EvaluationsList::new(initial_evals);

        let challenges = vec![F::from_u64(3), F::from_u64(7), F::from_u64(11)];

        // Apply three rounds of compression.
        for &r in &challenges {
            evals_list.compress_svo(r);
        }

        // After 3 rounds, we should go from 4 variables to 1 variable.
        assert_eq!(evals_list.num_variables(), 1);
        assert_eq!(evals_list.num_evals(), 2);
    }

    #[test]
    fn test_compress_svo_single_variable() {
        // Test compression of a polynomial with just 1 variable (edge case).
        //
        // Initial layout (n=1 variable, 2 evaluations):
        //
        // [p(0), p(1)]
        let p_0 = F::from_u64(5);
        let p_1 = F::from_u64(9);
        let mut evals_list = EvaluationsList::new(vec![p_0, p_1]);
        let r = F::from_u64(7);

        evals_list.compress_svo(r);

        // After compression, we should have a constant polynomial.
        assert_eq!(evals_list.num_variables(), 0);
        assert_eq!(evals_list.num_evals(), 1);

        // Result: p(r) = r * (p(1) - p(0)) + p(0)
        let expected = r * (p_1 - p_0) + p_0;
        assert_eq!(evals_list.as_slice(), vec![expected]);
    }

    #[test]
    fn test_compress_svo_with_zero_challenge() {
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

        evals_list.compress_svo(r);

        // With r = 0, result[i] = 0 * (right[i] - left[i]) + left[i] = left[i]
        //
        // So we should get the first half (left half where X_1 = 0):
        let expected = vec![p_000, p_001, p_010, p_011];

        assert_eq!(evals_list.as_slice(), &expected);
    }

    #[test]
    fn test_compress_svo_with_one_challenge() {
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

        evals_list.compress_svo(r);

        // With r = 1, result[i] = 1 * (right[i] - left[i]) + left[i] = right[i]
        //
        // So we should get the second half (right half where X_1 = 1):
        let expected = vec![p_100, p_101, p_110, p_111];

        assert_eq!(evals_list.as_slice(), &expected);
    }

    #[test]
    fn test_compress_ext() {
        // This test verifies the out-of-place compression into an extension field.
        let initial_evals: Vec<F> = (1..=8).map(F::from_u64).collect();
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
            let mut rng = StdRng::seed_from_u64(seed);
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
        fn prop_compress_svo_dimensions(
            n in 1usize..=10,
            seed in any::<u64>(),
        ) {
            let mut rng = StdRng::seed_from_u64(seed);
            let num_evals = 1 << n;
            let evals: Vec<F> = (0..num_evals).map(|_| rng.random()).collect();
            let r: F = rng.random();

            let mut list = EvaluationsList::new(evals);
            list.compress_svo(r);

            // After one round of compression, we should have n-1 variables.
            prop_assert_eq!(list.num_variables(), n - 1);
            prop_assert_eq!(list.num_evals(), num_evals / 2);
        }

        #[test]
        fn prop_compress_svo_boundary_challenges(
            n in 2usize..=8,
            seed in any::<u64>(),
        ) {
            let mut rng = StdRng::seed_from_u64(seed);
            let num_evals = 1 << n;
            let evals: Vec<F> = (0..num_evals).map(|_| rng.random()).collect();

            // Test with r = 0: should select left half
            let mut list_zero = EvaluationsList::new(evals.clone());
            list_zero.compress_svo(F::ZERO);
            prop_assert_eq!(list_zero.num_evals(), num_evals / 2);

            // Test with r = 1: should select right half
            let mut list_one = EvaluationsList::new(evals);
            list_one.compress_svo(F::ONE);
            prop_assert_eq!(list_one.num_evals(), num_evals / 2);

            // Results should be different (unless all evals are the same)
            if list_zero.as_slice() != list_one.as_slice() {
                prop_assert_ne!(list_zero.as_slice(), list_one.as_slice());
            }
        }

        #[test]
        fn prop_compress_svo_multiple_rounds(
            n in 2usize..=8,
            num_rounds in 1usize..=5,
            seed in any::<u64>(),
        ) {
            let mut rng = StdRng::seed_from_u64(seed);
            let num_evals = 1 << n;
            let evals: Vec<F> = (0..num_evals).map(|_| rng.random()).collect();

            // Generate random challenges, but not more than the number of variables.
            let actual_rounds = num_rounds.min(n);
            let challenges: Vec<F> = (0..actual_rounds).map(|_| rng.random()).collect();

            // Apply multiple rounds of compress_svo.
            let mut list = EvaluationsList::new(evals);
            for &r in &challenges {
                list.compress_svo(r);
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
}
