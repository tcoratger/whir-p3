use alloc::{vec, vec::Vec};

use itertools::Itertools;
use p3_field::{
    Algebra, ExtensionField, Field, PackedFieldExtension, PackedValue, PrimeCharacteristicRing,
    TwoAdicField,
};
use p3_interpolation::interpolate_subgroup;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::eq_batch::eval_eq_batch;
use p3_util::log2_strict_usize;
use serde::{Deserialize, Serialize};
use tracing::instrument;

use super::multilinear::MultilinearPoint;
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

impl<F: Copy + Clone + Send + Sync> EvaluationsList<F> {
    /// Given a number of points initializes a new zero polynomial
    #[inline]
    pub fn zero(num_variables: usize) -> Self
    where
        F: PrimeCharacteristicRing,
    {
        Self(F::zero_vec(1 << num_variables))
    }

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

    /// Evaluates the polynomial as a constant.
    ///
    /// This is only valid for constant polynomials (i.e., when `num_variables` is 0).
    ///
    /// Returns None in other cases.
    #[must_use]
    #[inline]
    pub fn as_constant(&self) -> Option<F> {
        (self.num_evals() == 1).then_some(self.0[0])
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
}

impl<A: Clone + Copy + Default + Send + Sync> EvaluationsList<A> {
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
    pub fn compress<F: Clone + Copy + Default + Send + Sync>(&mut self, r: F)
    where
        A: Algebra<F>,
    {
        assert_ne!(self.num_variables(), 0);
        let num_evals = self.num_evals();
        let mid = num_evals / 2;

        // Evaluations at `a_i` and `a_{i + n/2}` slots are folded with `r` into `a_i` slot
        let (p0, p1) = self.0.split_at_mut(mid);
        if num_evals >= PARALLEL_THRESHOLD {
            p0.par_iter_mut()
                .zip(p1.par_iter())
                .for_each(|(a0, &a1)| *a0 += (a1 - *a0) * r);
        } else {
            p0.iter_mut()
                .zip(p1.iter())
                .for_each(|(a0, &a1)| *a0 += (a1 - *a0) * r);
        }
        // Free higher part of the evaluations
        self.0.truncate(mid);
    }

    /// Folds a list of evaluations from a base field `F` into packed form of extension field `EF`.
    ///
    /// ## Arguments
    /// * `r`: A value `r` from the extension field `EF`, used as the random challenge for folding.
    ///
    /// ## Returns
    /// A new `EvaluationsList<EF::ExtensionPacking>` containing the compressed evaluations in the extension field.
    ///
    /// The compression is achieved by applying the following formula to pairs of evaluations:
    /// ```text
    ///     p'(X_2, ..., X_n) = (p(1, X_2, ..., X_n) - p(0, X_2, ..., X_n)) \cdot r + p(0, X_2, ..., X_n)
    /// ```
    pub fn compress_into_packed<EF>(&self, zi: EF) -> EvaluationsList<EF::ExtensionPacking>
    where
        A: Field,
        EF: ExtensionField<A>,
    {
        let zi = EF::ExtensionPacking::from(zi);
        let poly = A::Packing::pack_slice(self.as_slice());
        let mid = poly.len() / 2;
        let (p0, p1) = poly.split_at(mid);

        let mut out = EF::ExtensionPacking::zero_vec(mid);
        if self.num_evals() >= PARALLEL_THRESHOLD {
            out.par_iter_mut()
                .zip(p0.par_iter().zip(p1.par_iter()))
                .for_each(|(out, (&a0, &a1))| *out = zi * (a1 - a0) + a0);
        } else {
            out.iter_mut()
                .zip(p0.iter().zip(p1.iter()))
                .for_each(|(out, (&a0, &a1))| *out = zi * (a1 - a0) + a0);
        }
        EvaluationsList(out)
    }
}

impl<Packed: Copy + Send + Sync> EvaluationsList<Packed> {
    /// Given a point `P` (as a slice), compute the evaluation vector of the equality
    /// function `eq(P, X)` for all points `X` in the boolean hypercube, scaled by a value.
    ///
    /// ## Arguments
    /// * `point`: A multilinear point.
    /// * `value`: A scalar value to multiply all evaluations by.
    ///
    /// ## Returns
    /// An packed `EvaluationsList` containing `value * eq(point, X)` for all `X` in `{0,1}^n`.
    #[inline]
    pub(crate) fn new_packed_from_point<F, EF>(point: &[EF], scale: EF) -> Self
    where
        F: Field,
        EF: ExtensionField<F, ExtensionPacking = Packed>,
        Packed: PackedFieldExtension<F, EF>,
    {
        fn eq_serial<F: Field, A: Algebra<F> + Copy>(out: &mut [A], point: &[F], scale: A) {
            assert_eq!(out.len(), 1 << point.len());
            out[0] = scale;
            for (i, &var) in point.iter().rev().enumerate() {
                let (lo, hi) = out.split_at_mut(1 << i);
                lo.iter_mut().zip(hi.iter_mut()).for_each(|(lo, hi)| {
                    *hi = *lo * var;
                    *lo -= *hi;
                });
            }
        }

        let n = point.len();
        assert_ne!(scale, EF::ZERO);
        let n_pack = log2_strict_usize(F::Packing::WIDTH);
        assert!(n >= n_pack);

        let (point_rest, point_init) = point.split_at(n - n_pack);

        // COMPUTE SUFFIX (Inside the SIMD lanes)
        //
        // We compute the equality polynomial for the last `n_pack` variables.
        // This forms a single `Packed` element which acts as the "seed" for the next stage.e
        let mut init: Vec<EF> = EF::zero_vec(1 << n_pack);
        eq_serial(&mut init, point_init, scale);

        // COMPUTE PREFIX (Vector Expansion)
        //
        // We expand the seed across the remaining variables using Packed arithmetic.
        let mut packed = unsafe { uninitialized_vec::<Packed>(1 << (n - n_pack)) };
        eq_serial(
            &mut packed,
            point_rest,
            // Initialize the first element with the seed computed above
            Packed::from_ext_slice(&init),
        );

        Self(packed)
    }

    /// Evaluates the multilinear polynomial at `point ∈ EF^n`.
    /// Polynomial evaluations are in packed form.
    ///
    /// Computes
    /// ```text
    ///     f(point) = \sum_{x ∈ {0,1}^n} eq(x, point) * f(x),
    /// ```
    /// where
    /// ```text
    ///     eq(x, point) = \prod_{i=1}^{n} (1 - p_i + 2 p_i x_i).
    /// ```
    pub fn eval_hypercube_packed<F, EF>(&self, point: &MultilinearPoint<EF>) -> EF
    where
        F: Field,
        EF: ExtensionField<F, ExtensionPacking = Packed>,
        Packed: PackedFieldExtension<F, EF>,
    {
        let n = point.num_variables();
        let n_pack = log2_strict_usize(F::Packing::WIDTH);
        assert_eq!(self.num_variables() + n_pack, n);
        assert!(n >= 2 * n_pack);

        let (right, left) = point.split_at(n / 2);
        let left = Self::new_packed_from_point(left.as_slice(), EF::ONE);
        let right = EvaluationsList::<EF>::new_from_point(right.as_slice(), EF::ONE);

        let sum = if self.num_evals() > PARALLEL_THRESHOLD {
            self.0
                .par_chunks(left.num_evals())
                .zip_eq(right.0.par_iter())
                .map(|(part, &c)| {
                    part.iter()
                        .zip_eq(left.iter())
                        .map(|(&a, &b)| b * a)
                        .sum::<Packed>()
                        * c
                })
                .sum()
        } else {
            self.0
                .chunks(left.num_evals())
                .zip_eq(right.0.iter())
                .map(|(part, &c)| {
                    part.iter()
                        .zip_eq(left.iter())
                        .map(|(&a, &b)| b * a)
                        .sum::<Packed>()
                        * c
                })
                .sum()
        };
        EF::ExtensionPacking::to_ext_iter([sum]).sum()
    }
}

impl<F> EvaluationsList<F>
where
    F: Field,
{
    /// Given a point `P` (as a slice), compute the evaluation vector of the equality
    /// function `eq(P, X)` for all points `X` in the boolean hypercube, scaled by a value.
    ///
    /// ## Arguments
    /// * `point`: A multilinear point.
    /// * `value`: A scalar value to multiply all evaluations by.
    ///
    /// ## Returns
    /// An `EvaluationsList` containing `value * eq(point, X)` for all `X` in `{0,1}^n`.
    #[inline]
    pub fn new_from_point(point: &[F], scale: F) -> Self {
        let n = point.len();
        if n == 0 {
            return Self(vec![scale]);
        }
        let mut evals = F::zero_vec(1 << n);
        eval_eq_batch::<_, _, false>(RowMajorMatrixView::new_col(point), &mut evals, &[scale]);
        Self(evals)
    }

    /// Evaluates the multilinear polynomial at `point ∈ F^n`.
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
    pub fn evaluate_hypercube_base<EF: ExtensionField<F>>(
        &self,
        point: &MultilinearPoint<EF>,
    ) -> EF {
        if point.num_variables() < MLE_RECURSION_THRESHOLD {
            eval_multilinear_recursive(&self.0, point.as_slice())
        } else {
            eval_multilinear_base::<F, EF>(&self.0, point.as_slice())
        }
    }

    /// Evaluates the multilinear polynomial at `point ∈ F^n`.
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
    pub fn evaluate_hypercube_ext<BaseField: Field>(&self, point: &MultilinearPoint<F>) -> F
    where
        F: ExtensionField<BaseField>,
    {
        if point.num_variables() < MLE_RECURSION_THRESHOLD {
            eval_multilinear_recursive(&self.0, point.as_slice())
        } else {
            eval_multilinear_ext::<BaseField, F>(&self.0, point.as_slice())
        }
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

    /// Folds the polynomial over the first `k` variables in a single batched step.
    ///
    /// - Let $p : \{0,1\}^n \to F$ be the multilinear polynomial represented by this
    ///   evaluation list.
    /// - Let `challenges` have length $k$ with elements in an extension field `EF`.
    ///
    /// This method returns the evaluations of the folded polynomial
    ///
    /// \begin{equation}
    ///   q(x') = p(r_0, \dots, r_{k-1}, x')
    /// \end{equation}
    ///
    /// for all $x' \in \{0,1\}^{n-k}$, where $(r_0, \dots, r_{k-1})$ come from `challenges`.
    ///
    /// # Memory layout and indexing
    ///
    /// The evaluations in `self.0` are stored in lexicographic order over $(x_0, \dots, x_{n-1})$.
    ///
    /// - The first `k` variables $(x_0, \dots, x_{k-1})$ correspond to the
    ///   most significant bits of the index.
    /// - The remaining `n - k` variables correspond to the least significant bits.
    ///
    /// We can view the table as a $2^k \times 2^{n-k}$ matrix:
    ///
    /// \begin{equation}
    ///   p(b, x'), \quad b \in \{0,1\}^k,\ x' \in \{0,1\}^{n-k},
    /// \end{equation}
    ///
    /// where each column (fixed $x'$) is contiguous in memory.
    ///
    /// For a fixed suffix $x'$, the folded value is
    ///
    /// \begin{equation}
    ///   q(x') = \sum_{b \in \{0,1\}^k} eq(r, b) \cdot p(b, x'),
    /// \end{equation}
    ///
    /// where $eq(r, b)$ is the multilinear equality polynomial evaluated at
    /// `challenges` and the Boolean vector $b$.
    ///
    /// # Arguments
    ///
    /// - `challenges`: Slice of length $k$ containing the extension field
    ///   challenges $(r_0, \dots, r_{k-1})$ to substitute for the first `k`
    ///   variables. May be empty.
    ///
    /// # Panics
    ///
    /// Panics if `challenges.len() > self.num_variables()`.
    ///
    /// # Returns
    ///
    /// An `EvaluationsList<EF>` with $2^{n-k}$ entries representing the folded
    /// polynomial $q$ over the remaining `n - k` variables.
    pub fn fold_batch<EF>(&self, challenges: &[EF]) -> EvaluationsList<EF>
    where
        EF: ExtensionField<F> + Send + Sync,
    {
        // Number of folded variables: k = challenges.len().
        let num_challenges = challenges.len();

        // Simple case: k = 0.
        //
        // No variables are folded. We just reinterpret the polynomial
        // by lifting each base-field value into the extension field.
        if num_challenges == 0 {
            // - If F and EF differ, this converts each F into EF.
            // - If they are the same type, this is a no-op conversion.
            return EvaluationsList(self.0.iter().map(|&x| EF::from(x)).collect());
        }

        // Total number of variables in the original polynomial: n.
        let n = self.num_variables();

        // Sanity check: cannot fold more variables than we have.
        assert!(
            num_challenges <= n,
            "fold_batch: challenges.len() = {num_challenges} exceeds num_variables() = {n}"
        );

        // Number of remaining variables after folding: n - k.
        let remaining_vars = n - num_challenges;

        // Number of evaluations in the folded table: 2^(n-k).
        //
        // Each such evaluation corresponds to one suffix x' ∈ {0,1}^{n-k}.
        let num_remaining_evals = 1 << remaining_vars;

        // Precompute equality weights:
        //
        // eq_evals[j] = eq(r, b_j) for the j-th Boolean vector b_j ∈ {0,1}^k,
        // in lexicographic order over b_j.
        //
        // This gives us the coefficients for the dot product in the folding formula:
        //
        //   q(x') = Σ_b eq(r, b) * p(b, x').
        let eq_evals = EvaluationsList::new_from_point(challenges, EF::ONE);

        // Prepare output buffer initialized with zeros so we can unconditionally accumulate.
        //
        // folded_evals_flat[i] will store q(x') for the suffix index i.
        let mut folded_evals_flat = EF::zero_vec(num_remaining_evals);

        // Compute the folded evaluations in parallel.
        //
        // We split the output buffer into disjoint chunks. Each chunk corresponds
        // to a contiguous range of suffix indices x'. Each thread owns its chunk
        // and updates it independently, so no synchronization is needed.
        //
        // This layout ensures that both reads from the input polynomial and writes
        // to the output are over contiguous slices, which is friendly to caches.
        folded_evals_flat
            .par_chunks_mut(PARALLEL_THRESHOLD)
            .enumerate()
            .for_each(|(chunk_idx, result_chunk)| {
                // Compute the global starting index for this chunk of suffixes.
                //
                // All suffix indices covered by this chunk are:
                //   i = chunk_start_offset .. chunk_start_offset + result_chunk.len().
                let chunk_start_offset = chunk_idx * PARALLEL_THRESHOLD;

                // For this fixed range of suffix indices, sum over all prefixes b.
                //
                // - j runs over all 2^k Boolean prefixes b_j,
                // - eq_val is the precomputed weight eq(r, b_j).
                for (j, &eq_val) in eq_evals.0.iter().enumerate() {
                    // Calculate where the block corresponding to prefix b_j starts in the input.
                    //
                    // The evaluation p(b_j, x') lives at index:
                    //
                    //   original_eval_index = j * 2^(n-k) + i,
                    //
                    // because prefixes are in the most significant bits and
                    // suffixes are in the least significant bits.
                    //
                    // For the current chunk, the block we need starts at:
                    let input_block_start = j * num_remaining_evals + chunk_start_offset;

                    // Get the contiguous slice of input evaluations for this prefix,
                    // restricted to the suffix range covered by `result_chunk`.
                    let input_chunk =
                        &self.0[input_block_start..input_block_start + result_chunk.len()];

                    // Accumulate eq(r, b_j) * p(b_j, x') into the result.
                    //
                    // This hot loop iterates contiguously over both input and result chunks.
                    for (res, &p_b_x) in result_chunk.iter_mut().zip(input_chunk) {
                        // Accumulate eq(r, b_j) * p(b_j, x') in the extension field.
                        *res += eq_val * p_b_x;
                    }
                }
            });

        // Wrap the flat vector into a new evaluation list in EF.
        //
        // It holds the values q(x') for all x' ∈ {0,1}^{n-k}, in lexicographic order.
        EvaluationsList::new(folded_evals_flat)
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
        final_poly.evaluate_hypercube_ext::<F>(&y_point)
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
            .map(|x_out| {
                // THREAD-LOCAL INITIALIZATION
                let mut local_accs = SvoAccumulators::<EF, N>::new();
                let mut temp_buffer = EF::zero_vec(1 << N);

                // STEP 1: Inner product (small multiplications)
                //
                // For each x_in, accumulate:
                //   temp[β] += E_in[x_in] · P(β, x_in, x_out)
                for (x_in, &e_in_val) in e_in.0.iter().enumerate().take(num_x_in) {
                    // Base index encodes: (β=0, x_in, x_out)
                    let base_index = (x_in << x_out_num_vars) | x_out;

                    // Iterate over all 2^N possible β prefixes
                    //
                    // Note: We need the index i for computing beta, not just for indexing
                    for (i, buf) in temp_buffer.iter_mut().enumerate().take(1 << N) {
                        let beta = i << x_num_vars;
                        let index = beta | base_index;

                        // SAFETY: index < 2^l by construction
                        let p_val = unsafe { *poly_evals.get_unchecked(index) };
                        *buf += e_in_val * p_val;
                    }
                }

                // STEP 2: Distribution (large multiplications)
                //
                // Multiply temp[β] by E_out and add to round accumulators
                for (r, e_out_slice) in e_out_slices.iter().enumerate().take(N) {
                    let block_size = 1 << (N - 1 - r);

                    // SAFETY: r < N by loop bound
                    let round_accs = unsafe { local_accs.at_round_unchecked_mut(r) };

                    for (acc_idx, acc) in round_accs.iter_mut().enumerate() {
                        let start_idx = acc_idx * block_size;

                        // Dot product: sum over block variations
                        for k in 0..block_size {
                            let e_idx = (k << x_out_num_vars) | x_out;
                            *acc += e_out_slice[e_idx] * temp_buffer[start_idx + k];
                        }
                    }
                }

                local_accs
            })
            // REDUCTION: Merge all thread-local accumulators
            .par_fold_reduce(SvoAccumulators::new, |a, b| a + b, |a, b| a + b)
    }

    /// Computes c_0 accumulators for batched SVO sumcheck.
    ///
    /// # Context
    ///
    /// In sumcheck round r, we compute the univariate polynomial:
    ///
    /// ```text
    /// s_r(X_r) = Σ_suffix W(prefix, X_r, suffix) · p(prefix, X_r, suffix)
    /// ```
    ///
    /// This means:
    ///
    /// ```text
    /// s_r(X) = c_0 + c_1·X + c_2·X^2
    /// ```
    ///
    /// This function pre-computes the partial sums needed for the c_0 coefficient
    /// across N rounds in a single pass over the evaluation table.
    ///
    /// # What Are Accumulators?
    ///
    /// For round r, the constant coefficient c_0 is obtained by evaluating at X_r = 0:
    ///
    /// ```text
    /// c_0 = s_r(0) = Σ_suffix W(prefix, 0, suffix) · p(prefix, 0, suffix)
    /// ```
    ///
    /// We partition this sum by the values of variables x_0, x_1, ..., x_{r-1} (the "prefix").
    ///
    /// The accumulator `A[prefix]` stores the partial sum for that prefix:
    ///
    /// ```text
    /// A[prefix] = Σ_suffix W(prefix, 0, suffix)·p(prefix, 0, suffix)
    /// ```
    ///
    /// # Variable Layout
    ///
    /// For a polynomial with l variables indexed as x_0, x_1, ..., x_{l-1}:
    ///
    /// ```text
    /// Round 0: [x_0 | x_1 x_2 ... x_{l-1}]
    ///            ↑     └─── suffix ───┘
    ///          prefix
    ///          (1 bit)
    ///
    /// Round 1: [x_0 x_1 | x_2 ... x_{l-1}]
    ///           └──┬──┘   └── suffix ──┘
    ///           prefix
    ///          (2 bits)
    ///
    /// Round r: [x_0 ... x_r | x_{r+1} ... x_{l-1}]
    ///          └── prefix ──┘   └─── suffix ───┘
    ///            (r+1 bits)
    /// ```
    ///
    /// # Why Pre-compute?
    ///
    /// Without pre-computation, each round would scan all 2^l evaluations.
    /// Pre-computing all accumulators in one pass reduces total work.
    ///
    /// # Performance Benefit
    ///
    /// The product W(x)·p(x) is an M_BE multiplication (base × extension).
    /// This is cheaper than M_EE (extension × extension) which occurs after folding.
    #[instrument(skip_all, level = "debug")]
    pub fn compute_svo_accumulators_with_weight<EF, const N: usize>(
        &self,
        weights: &EvaluationsList<EF>,
    ) -> SvoAccumulators<EF, N>
    where
        EF: ExtensionField<F> + Send + Sync,
    {
        let l = self.num_variables();
        assert_eq!(
            weights.num_variables(),
            l,
            "weight polynomial must have same number of variables"
        );
        assert!(N <= l, "cannot compute more SVO rounds than variables");

        let poly_evals = self.as_slice();
        let weight_evals = weights.as_slice();

        // SINGLE PASS: Iterate over all 2^l evaluation points exactly once.
        //
        // Each point (index) contributes to accumulators for ALL N rounds.
        // This avoids scanning the table N times (once per round).
        (0..(1 << l))
            .into_par_iter()
            .map(|index| {
                let mut accs = SvoAccumulators::<EF, N>::new();

                // STEP 1: Compute W(x) · p(x)
                //
                // This is M_BE (base × extension) multiplication.
                // Key benefit: cheaper than M_EE after p gets folded.
                let p_val = poly_evals[index];
                let w_val = weight_evals[index];
                let contribution = w_val * p_val;

                // STEP 2: Distribute to each round's accumulator
                //
                // For round r, extract the top (r+1) bits as "prefix".
                //
                // Example: l=4, index=0b1011 (binary representation)
                //
                //   index = [1] [0] [1] [1]
                //            ↑   ↑   ↑   ↑
                //           x_0 x_1 x_2 x_3
                //
                //   Round 0: prefix = 1     (1 bit)  → A_0[1]
                //   Round 1: prefix = 10    (2 bits) → A_1[2]
                //   Round 2: prefix = 101   (3 bits) → A_2[5]
                for r in 0..N {
                    let prefix_bits = r + 1;
                    let shift = l - prefix_bits;
                    let prefix = index >> shift;
                    accs.accumulate(r, prefix, contribution);
                }

                accs
            })
            .par_fold_reduce(SvoAccumulators::new, |a, b| a + b, |a, b| a + b)
    }

    /// Computes c_2 accumulators for batched SVO sumcheck.
    ///
    /// # Context
    ///
    /// In sumcheck round r, we compute the univariate polynomial:
    ///
    /// ```text
    /// s_r(X_r) = Σ_suffix W(prefix, X_r, suffix) · p(prefix, X_r, suffix)
    /// ```
    ///
    /// This means:
    ///
    /// ```text
    /// s_r(X) = c_0 + c_1·X + c_2·X^2
    /// ```
    ///
    /// This function pre-computes the partial sums needed for the c_2 (quadratic) coefficient
    /// across N rounds in a single pass over the evaluation table.
    ///
    /// # What Are c_2 Accumulators?
    ///
    /// For round r, the quadratic coefficient c_2 is computed from "deltas":
    ///
    /// ```text
    /// c_2 = Σ_suffix ΔW(prefix, suffix) · Δp(prefix, suffix)
    /// ```
    ///
    /// where the deltas measure the difference when X_r goes from 0 to 1:
    ///
    /// ```text
    /// ΔW = W(prefix, 1, suffix) - W(prefix, 0, suffix)
    /// Δp = p(prefix, 1, suffix) - p(prefix, 0, suffix)
    /// ```
    ///
    /// We partition this sum by the values of x_0, x_1, ..., x_{r-1} (the "prefix").
    ///
    /// The accumulator `C2[prefix]` stores the partial sum for that prefix:
    ///
    /// ```text
    /// C2[prefix] = Σ_suffix ΔW(prefix, suffix) · Δp(prefix, suffix)
    /// ```
    ///
    /// # Variable Layout
    ///
    /// For round r, variables are partitioned as:
    ///
    /// ```text
    /// [x_0 ... x_{r-1} | x_r | x_{r+1} ... x_{l-1}]
    ///  └─── prefix ────┘  ↑    └───── suffix ─────┘
    ///      (r bits)    current     (l-r-1 bits)
    ///                  variable
    /// ```
    ///
    /// # How Deltas Work
    ///
    /// For each (prefix, suffix) pair, there are two evaluation points:
    ///
    /// ```text
    /// "lo" index:  (prefix, 0, suffix)  →  x_r = 0
    /// "hi" index:  (prefix, 1, suffix)  →  x_r = 1
    ///
    /// Δ = value_at_hi - value_at_lo
    /// ```
    ///
    /// # Why pre-compute?
    ///
    /// Without pre-computation, each round would scan all 2^l evaluations.
    /// Pre-computing all accumulators in one pass reduces total work.
    ///
    /// # Performance Benefit
    ///
    /// The product ΔW · Δp uses M_BE multiplication (base × extension).
    /// This is cheaper than M_EE (extension × extension) which occurs after folding.
    #[instrument(skip_all, level = "debug")]
    pub fn compute_svo_c2_accumulators_with_weight<EF, const N: usize>(
        &self,
        weights: &EvaluationsList<EF>,
    ) -> SvoAccumulators<EF, N>
    where
        EF: ExtensionField<F> + Send + Sync,
    {
        let l = self.num_variables();
        assert_eq!(
            weights.num_variables(),
            l,
            "weight polynomial must have same number of variables"
        );
        assert!(N <= l, "cannot compute more SVO rounds than variables");
        assert!(l >= 1, "need at least 1 variable for c2 computation");

        let poly_evals = self.as_slice();
        let weight_evals = weights.as_slice();

        // SINGLE PASS: Iterate over all 2^l evaluation points exactly once.
        //
        // For each round r, we only process "lo" indices (where x_r = 0).
        // This avoids double-counting since each (lo, hi) pair appears once.
        (0..(1 << l))
            .into_par_iter()
            .map(|index| {
                let mut accs = SvoAccumulators::<EF, N>::new();

                for r in 0..N {
                    // STEP 1: Locate the current variable x_r
                    //
                    // Round r processes variable x_r.
                    // In the index bits, x_r sits at position (l-r-1).
                    //
                    // Example with l=4:
                    //
                    //   index bits: [x_0][x_1][x_2][x_3]
                    //   positions:    3    2    1    0
                    //
                    //   Round 0: x_0 at position 3 (MSB)
                    //   Round 1: x_1 at position 2
                    //   Round 2: x_2 at position 1
                    let current_var_position = l - r - 1;

                    // STEP 2: Only process "lo" indices (x_r = 0)
                    //
                    // Check if bit at current_var_position is 0.
                    // If so, this is the "lo" half of a (lo, hi) pair.
                    if (index >> current_var_position) & 1 == 0 {
                        // STEP 3: Compute the "hi" index
                        //
                        // Flip bit at current_var_position from 0 to 1.
                        //
                        // Example: l=4, r=1, index=0b0001
                        //   current_var_position = 2
                        //   hi_index = 0b0001 | 0b0100 = 0b0101
                        let hi_index = index | (1 << current_var_position);

                        // STEP 4: Compute deltas
                        //
                        // Δp = p(hi) - p(lo)
                        // ΔW = W(hi) - W(lo)
                        //
                        // These measure the change when x_r: 0 → 1.
                        let p_hi = poly_evals[hi_index];
                        let p_lo = poly_evals[index];
                        let delta_p = p_hi - p_lo;
                        let delta_w = weight_evals[hi_index] - weight_evals[index];

                        // STEP 5: Compute c_2 contribution
                        //
                        // c_2 contribution = ΔW · Δp
                        // This is M_BE multiplication (base × extension).
                        let contribution = delta_w * delta_p;

                        // STEP 6: Extract prefix and accumulate
                        //
                        // Prefix = the r bits ABOVE the current variable.
                        //
                        // Example: l=4, r=1, index=0b0010
                        //
                        //   index = [0]  [0]  [1]  [0]
                        //            ↑    ↑
                        //         prefix x_1
                        //
                        //   prefix = 0b0010 >> 3 = 0b0  → C2_1[0]
                        let prefix = index >> (current_var_position + 1);
                        accs.accumulate(r, prefix, contribution);
                    }
                }

                accs
            })
            .par_fold_reduce(SvoAccumulators::new, |a, b| a + b, |a, b| a + b)
    }
}

impl<A: Copy + Send + Sync + PrimeCharacteristicRing> EvaluationsList<A> {
    /// Computes the constant and quadratic coefficients of the sumcheck polynomial.
    ///
    /// Given evaluations `self[i]` and weights `weights[i]`, this computes the coefficients
    /// of the univariate polynomial:
    ///
    /// ```text
    /// h(X) = \sum_{b \in \{0,1\}^{n-1}} self(X, b) * weights(X, b)
    /// ```
    ///
    /// which is a quadratic polynomial in `X`.
    ///
    /// # Coefficient Formulas
    ///
    /// The polynomial `h(X) = c_0 + c_1 * X + c_2 * X^2` has coefficients:
    ///
    /// ```text
    /// c_0 = h(0) = \sum_b self(0, b) * weights(0, b)
    ///
    /// c_2 = \sum_b (self(1,b) - self(0,b)) * (weights(1,b) - weights(0,b))
    /// ```
    ///
    /// The linear coefficient `c_1` is not computed here; it's derived by the verifier
    /// from the sum constraint `h(0) + h(1) = claimed_sum`.
    ///
    /// # Memory Layout
    ///
    /// The arrays are organized such that:
    /// - First half (`lo`): evaluations where `X = 0`
    /// - Second half (`hi`): evaluations where `X = 1`
    ///
    /// # Arguments
    ///
    /// * `weights` - Weight polynomial evaluations (same length as `self`).
    ///
    /// # Returns
    ///
    /// A tuple `(c_0, c_2)` of the constant and quadratic coefficients.
    ///
    /// # Panics
    ///
    /// Panics if `self.len() != weights.len()` or `self.len() < 2`.
    #[instrument(skip_all, level = "debug")]
    pub fn sumcheck_coefficients<B>(&self, weights: &EvaluationsList<B>) -> (B, B)
    where
        B: Copy + Send + Sync + Algebra<A>,
    {
        let evals = self.as_slice();
        let weights = weights.as_slice();

        // Validate inputs: need at least 2 elements (1 variable).
        assert!(log2_strict_usize(evals.len()) >= 1);
        assert_eq!(evals.len(), weights.len());

        // Split arrays into lo (X=0) and hi (X=1) halves.
        let mid = evals.len() / 2;
        let (evals_lo, evals_hi) = evals.split_at(mid);
        let (weights_lo, weights_hi) = weights.split_at(mid);

        // Parallel computation of c_0 and c_2.
        evals_lo
            .par_iter()
            .zip(evals_hi.par_iter())
            .zip(weights_lo.par_iter().zip(weights_hi.par_iter()))
            .map(|((&e_lo, &e_hi), (&w_lo, &w_hi))| {
                // c_0 term: product at X=0.
                let c0_term = w_lo * e_lo;
                // c_2 term: cross-product of differences.
                let c2_term = (w_hi - w_lo) * (e_hi - e_lo);
                (c0_term, c2_term)
            })
            .par_fold_reduce(
                || (B::ZERO, B::ZERO),
                |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
                |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
            )
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

/// Evaluates a multilinear polynomial `evals` at `point` using a recursive strategy.
/// For small numbers of variables (<=4) it switches to the unrolled strategy.
fn eval_multilinear_recursive<F, EF>(evals: &[F], point: &[EF]) -> EF
where
    F: Field,
    EF: ExtensionField<F>,
{
    // Ensure that the number of evaluations matches the number of variables in the point.
    //
    // This is a critical invariant: `evals.len()` must be exactly `2^point.len()`.
    debug_assert_eq!(evals.len(), 1 << point.len());

    // Select the optimal evaluation strategy based on the number of variables.
    match point {
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
        [x, sub_point @ ..] => {
            // Split the evaluations into two halves, corresponding to the first variable being 0 or 1.
            let (f0, f1) = evals.split_at(evals.len() / 2);

            // Recursively evaluate on the two smaller hypercubes.
            let (f0_eval, f1_eval) = {
                // Only spawn parallel tasks if the subproblem is large enough to overcome
                // the overhead of threading.
                let work_size: usize = (1 << 15) / core::mem::size_of::<F>();
                if evals.len() > work_size {
                    join(
                        || eval_multilinear_recursive(f0, sub_point),
                        || eval_multilinear_recursive(f1, sub_point),
                    )
                } else {
                    // For smaller subproblems, execute sequentially.
                    (
                        eval_multilinear_recursive(f0, sub_point),
                        eval_multilinear_recursive(f1, sub_point),
                    )
                }
            };
            // Perform the final linear interpolation for the first variable `x`.
            f0_eval + (f1_eval - f0_eval) * *x
        }
    }
}

/// Evaluates a multilinear polynomial `evals` at `point` where `evals` are in the base field `F` and `point` is in the extension field `EF`.
fn eval_multilinear_base<F, EF>(evals: &[F], point: &[EF]) -> EF
where
    F: Field,
    EF: ExtensionField<F>,
{
    const PARALLEL_THRESHOLD: usize = 1 << 14;

    let num_vars = point.len();
    if num_vars < 2 * log2_strict_usize(F::Packing::WIDTH) {
        return eval_multilinear_recursive(evals, point);
    }

    let mid = num_vars / 2;

    let (right, left) = point.split_at(mid);
    let left = EvaluationsList::new_packed_from_point(left, EF::ONE);
    let right = EvaluationsList::new_from_point(right, EF::ONE);

    let evals = F::Packing::pack_slice(evals);
    let sum = if evals.len() > PARALLEL_THRESHOLD {
        evals
            .par_chunks(left.num_evals())
            .zip_eq(right.0.par_iter())
            .map(|(part, &c)| {
                part.iter()
                    .zip_eq(left.iter())
                    .map(|(&a, &b)| b * a)
                    .sum::<EF::ExtensionPacking>()
                    * c
            })
            .sum::<EF::ExtensionPacking>()
    } else {
        evals
            .chunks(left.num_evals())
            .zip_eq(right.0.iter())
            .map(|(part, &c)| {
                part.iter()
                    .zip_eq(left.iter())
                    .map(|(&a, &b)| b * a)
                    .sum::<EF::ExtensionPacking>()
                    * c
            })
            .sum::<EF::ExtensionPacking>()
    };
    EF::ExtensionPacking::to_ext_iter([sum]).sum()
}

/// Evaluates a multilinear polynomial `evals` at `point` where `evals` and `point` are in the extension field `EF`.
fn eval_multilinear_ext<F, EF>(evals: &[EF], point: &[EF]) -> EF
where
    F: Field,
    EF: ExtensionField<F>,
{
    const PARALLEL_THRESHOLD: usize = 1 << 14;

    let num_vars = point.len();
    if num_vars < 2 * log2_strict_usize(F::Packing::WIDTH) {
        return eval_multilinear_recursive(evals, point);
    }

    let mid = num_vars / 2;
    let (right, left) = point.split_at(mid);
    let left = EvaluationsList::new_packed_from_point(left, EF::ONE);
    let right = EvaluationsList::new_from_point(right, EF::ONE);

    let sum = if evals.len() > PARALLEL_THRESHOLD {
        evals
            .chunks(F::Packing::WIDTH * left.num_evals())
            .zip_eq(right.0.iter())
            .map(|(part, &c)| {
                part.chunks(F::Packing::WIDTH)
                    .zip_eq(left.iter())
                    .map(|(chunk, &b)| EF::ExtensionPacking::from_ext_slice(chunk) * b)
                    .sum::<EF::ExtensionPacking>()
                    * c
            })
            .sum::<EF::ExtensionPacking>()
    } else {
        evals
            .par_chunks(F::Packing::WIDTH * left.num_evals())
            .zip_eq(right.0.par_iter())
            .map(|(part, &c)| {
                part.chunks(F::Packing::WIDTH)
                    .zip_eq(left.iter())
                    .map(|(chunk, &b)| EF::ExtensionPacking::from_ext_slice(chunk) * b)
                    .sum::<EF::ExtensionPacking>()
                    * c
            })
            .sum::<EF::ExtensionPacking>()
    };
    EF::ExtensionPacking::to_ext_iter([sum]).sum()
}

#[cfg(test)]
mod tests {

    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::{
        PrimeCharacteristicRing, PrimeField64, TwoAdicField, dot_product,
        extension::BinomialExtensionField,
    };
    use proptest::prelude::*;
    use rand::{Rng, SeedableRng, rngs::SmallRng};

    use super::*;

    type F = BabyBear;
    type EF4 = BinomialExtensionField<F, 4>;

    /// Naive method to evaluate a multilinear polynomial for testing.
    fn eval_multilinear<F: Field, EF: ExtensionField<F>>(evals: &[F], point: &[EF]) -> EF {
        let eq = EvaluationsList::new_from_point(point, EF::ONE);
        dot_product(eq.iter().copied(), evals.iter().copied())
    }

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
            evals.evaluate_hypercube_base(&MultilinearPoint::new(vec![F::ZERO, F::ZERO])),
            e1
        );
        assert_eq!(
            evals.evaluate_hypercube_base(&MultilinearPoint::new(vec![F::ZERO, F::ONE])),
            e2
        );
        assert_eq!(
            evals.evaluate_hypercube_base(&MultilinearPoint::new(vec![F::ONE, F::ZERO])),
            e3
        );
        assert_eq!(
            evals.evaluate_hypercube_base(&MultilinearPoint::new(vec![F::ONE, F::ONE])),
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

        let result = evals.evaluate_hypercube_base(&point);

        // Expected result using `eval_multilinear`
        let expected = eval_multilinear(evals.as_slice(), point.as_slice());

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

        assert_eq!(eval_multilinear(&evals, &[x]), expected);
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

        assert_eq!(eval_multilinear(&evals, &[x, y]), expected);
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

        assert_eq!(eval_multilinear(&evals, &[x, y, z]), expected);
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
        assert_eq!(eval_multilinear(&evals, &[x, y, z, w]), expected);
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
            let eval_f = poly_f.evaluate_hypercube_base(&point_ef);
            let eval_ef = poly_ef.evaluate_hypercube_ext::<F>(&point_ef);

            prop_assert_eq!(eval_f, eval_ef);
        }
    }

    #[test]
    fn test_multilinear_eval_two_vars() {
        // Define a simple 2-variable multilinear polynomial:
        //
        // Variables: x_1, x_2
        // Evaluations ordered in lexicographic order of input points: (x_1, x_2)
        //
        // - evals[0] → f(0, 0)
        // - evals[1] → f(0, 1)
        // - evals[2] → f(1, 0)
        // - evals[3] → f(1, 1)
        //
        // Thus, the polynomial is represented by its values
        // on the Boolean hypercube {0,1}².
        //
        // where:
        let e0 = F::from_u64(5); // f(0, 0)
        let e1 = F::from_u64(6); // increment when x_2 = 1
        let e2 = F::from_u64(7); // increment when x_1 = 1
        let e3 = F::from_u64(8); // increment when x_1 = x_2 = 1
        //
        // So concretely:
        //
        //   f(0, 0) = 5
        //   f(0, 1) = 5 + 6 = 11
        //   f(1, 0) = 5 + 7 = 12
        //   f(1, 1) = 5 + 6 + 7 + 8 = 26
        let evals = EvaluationsList::new(vec![e0, e0 + e1, e0 + e2, e0 + e1 + e2 + e3]);

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
        let expected = e0 + e1 * x2 + e2 * x1 + e3 * x1 * x2;

        // Now evaluate using the function under test
        let result = evals.evaluate_hypercube_base(&coords);

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
        let e0 = F::from_u64(1);
        let e1 = F::from_u64(2);
        let e2 = F::from_u64(3);
        let e3 = F::from_u64(4);
        let e4 = F::from_u64(5);
        let e5 = F::from_u64(6);
        let e6 = F::from_u64(7);
        let e7 = F::from_u64(8);

        let evals = EvaluationsList::new(vec![
            e0,
            e0 + e1,
            e0 + e2,
            e0 + e1 + e2 + e3,
            e0 + e4,
            e0 + e1 + e4 + e5,
            e0 + e2 + e4 + e6,
            e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7,
        ]);

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
        let expected = e0
            + e1 * x2
            + e2 * x1
            + e3 * x1 * x2
            + e4 * x0
            + e5 * x0 * x2
            + e6 * x0 * x1
            + e7 * x0 * x1 * x2;

        let result = evals.evaluate_hypercube_base(&point);
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
        let e0 = F::from_u64(1);
        let e1 = F::from_u64(2);
        let e2 = F::from_u64(3);
        let e3 = F::from_u64(4);
        let e4 = F::from_u64(5);
        let e5 = F::from_u64(6);
        let e6 = F::from_u64(7);
        let e7 = F::from_u64(8);

        let evals = EvaluationsList::new(vec![
            e0,
            e0 + e1,
            e0 + e2,
            e0 + e1 + e2 + e3,
            e0 + e4,
            e0 + e1 + e4 + e5,
            e0 + e2 + e4 + e6,
            e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7,
        ]);

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
        let expected = EF4::from(e0)
            + EF4::from(e1) * x2
            + EF4::from(e2) * x1
            + EF4::from(e3) * x1 * x2
            + EF4::from(e4) * x0
            + EF4::from(e5) * x0 * x2
            + EF4::from(e6) * x0 * x1
            + EF4::from(e7) * x0 * x1 * x2;

        // Evaluate via `evaluate_hypercube` method
        let result = evals.evaluate_hypercube_base(&point);

        // Verify that result matches manual computation
        assert_eq!(result, expected);
    }

    #[test]
    fn test_folding_and_evaluation() {
        // Set number of Boolean input variables n = 10.
        let num_variables = 10;

        // Build a multilinear polynomial
        let evals = (0..(1 << num_variables)).map(F::from_u64).collect();

        // Wrap into EvaluationsList
        let evals_list = EvaluationsList::new(evals);

        // Define a fixed evaluation point in F^n: [0, 35, 70, ..., 35*(n-1)]
        let randomness: Vec<_> = (0..num_variables)
            .map(|i| F::from_u64(35 * i as u64))
            .collect();

        // Try folding at every possible prefix of the randomness vector: k = 1 to n-1
        for k in 1..num_variables {
            // Use the first k values as the fold coordinates (we will substitute those)
            let fold_part = randomness[0..k].to_vec();

            // The remaining coordinates are used as the evaluation input into the folded poly
            let eval_part = MultilinearPoint::new(randomness[k..randomness.len()].to_vec());

            // Convert to a MultilinearPoint (in EF) for folding
            let fold_random = MultilinearPoint::new(fold_part.clone());

            // Reconstruct the full point (x_0, ..., xₙ₋₁) = [eval_part || fold_part]
            // Used to evaluate the original uncompressed polynomial
            let eval_point1 =
                MultilinearPoint::new([fold_part.clone(), eval_part.0.clone()].concat());

            // Fold the evaluation list over the last `k` variables
            let folded_evals = evals_list.fold(&fold_random);

            // Verify that the number of variables has been folded correctly
            assert_eq!(folded_evals.num_variables(), num_variables - k);

            // Fold the coefficients list over the last `k` variables
            let folded_coeffs = evals_list.fold(&fold_random);

            // Verify that the number of variables has been folded correctly
            assert_eq!(folded_coeffs.num_variables(), num_variables - k);

            // Verify correctness:
            // folded(e) == original([e, r]) for all k
            assert_eq!(
                folded_evals.evaluate_hypercube_base(&eval_part),
                evals_list.evaluate_hypercube_base(&eval_point1)
            );
        }
    }

    #[test]
    fn test_fold_with_extension_one_var() {
        // Define a 2-variable polynomial:
        // f(x_0, x_1) = 1 + 2·x_1 + 3·x_0 + 4·x_0·x_1
        let evals = vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ];
        let poly = EvaluationsList::new(evals);

        // Convert coefficients into an EvaluationsList (for testing the fold on evals)
        let evals_list: EvaluationsList<F> = poly;

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
            let expected = evals_list.evaluate_hypercube_base(&full_point);

            // Evaluate folded poly at x_0
            let actual = folded.evaluate_hypercube_base(&folded_point);

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
        let actual_result = evals_list.evaluate_hypercube_base(&point);

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
        let poly = EvaluationsList::new((1..=16).map(EF4::from_u64).collect());

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
        //  TEST SETUP: N=3, n=6
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

        let poly = EvaluationsList::new((1..=64).map(EF4::from_u64).collect());

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
        // Round 0: block_size=4, so E_out[0] has 4 elements
        //   Represents eq(β_1, β_0; s_1, s_0) evaluated at all 4 Boolean combinations
        //   E_out[0][00] = (1-s_1)(1-s_0)
        //   E_out[0][01] = (1-s_1)·s_0
        //   E_out[0][10] = s_1·(1-s_0)
        //   E_out[0][11] = s_1·s_0
        //
        // Round 1: block_size=2, so E_out[1] has 2 elements
        //   Represents eq(β_0; s_0) evaluated at 2 Boolean values
        //   E_out[1][0] = (1-s_0)
        //   E_out[1][1] = s_0
        //
        // Round 2: block_size=1, so E_out[2] has 1 element
        //   Just a constant value (no variables left to evaluate)
        //   E_out[2][0] = 1
        let s0 = EF4::from_u64(7);
        let s1 = EF4::from_u64(11);

        let e_out = [
            // Round 0: 4 elements for block_size=4
            EvaluationsList::new(vec![
                (EF4::ONE - s1) * (EF4::ONE - s0), // 00
                (EF4::ONE - s1) * s0,              // 01
                s1 * (EF4::ONE - s0),              // 10
                s1 * s0,                           // 11
            ]),
            // Round 1: 2 elements for block_size=2
            EvaluationsList::new(vec![
                EF4::ONE - s0, // 0
                s0,            // 1
            ]),
            // Round 2: 1 element for block_size=1
            EvaluationsList::new(vec![EF4::ONE]),
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
        let poly_f = EvaluationsList::new((1..=16).map(F::from_u64).collect());

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

    #[test]
    fn test_fold_batch_no_challenges() {
        // Setup: 3-variable polynomial p(x_2, x_1, x_0) with simple values
        let poly = EvaluationsList::new(vec![
            F::from_u64(1), // 000
            F::from_u64(2), // 001
            F::from_u64(3), // 010
            F::from_u64(4), // 011
            F::from_u64(5), // 100
            F::from_u64(6), // 101
            F::from_u64(7), // 110
            F::from_u64(8), // 111
        ]);

        // Fold with empty challenges (k=0)
        let challenges: &[EF4] = &[];
        let result = poly.fold_batch(challenges);

        // VERIFY: Result should have same length and values (just lifted to EF4)
        assert_eq!(result.0.len(), 8, "Result should have 8 evaluations");

        // Each value should be the same, just in extension field
        let expected_poly = EvaluationsList::new(vec![
            EF4::from_u64(1), // 000
            EF4::from_u64(2), // 001
            EF4::from_u64(3), // 010
            EF4::from_u64(4), // 011
            EF4::from_u64(5), // 100
            EF4::from_u64(6), // 101
            EF4::from_u64(7), // 110
            EF4::from_u64(8), // 111
        ]);
        assert_eq!(
            result, expected_poly,
            "Result should be the original polynomial"
        );
    }

    #[test]
    fn test_fold_batch_single_variable() {
        // Fold a 3-variable polynomial over the first variable.
        //
        // Polynomial: p(x_2, x_1, x_0) where p(i) = i+1 for i=0..7
        //
        // Variable ordering: x_2 (MSB) x_1 x_0 (LSB)
        //
        //   Index | x_2 x_1 x_0 | p(x_2, x_1, x_0)
        //   ------|-------------|------------------
        //     0   |  0   0   0  |        1
        //     1   |  0   0   1  |        2
        //     2   |  0   1   0  |        3
        //     3   |  0   1   1  |        4
        //     4   |  1   0   0  |        5
        //     5   |  1   0   1  |        6
        //     6   |  1   1   0  |        7
        //     7   |  1   1   1  |        8

        let poly = EvaluationsList::new((1..=8).map(F::from_u64).collect());

        // Fold over x_2 with challenge r_2
        let r2 = EF4::from_u64(3);
        let challenges = vec![r2];

        let result = poly.fold_batch(&challenges);

        // VERIFY: Result has 2^(3-1) = 4 evaluations
        assert_eq!(
            result.0.len(),
            4,
            "Folded polynomial should have 4 evaluations"
        );

        //  MANUAL COMPUTATION
        //
        // After folding over x_2, we get q(x_1, x_0) where:
        //
        //   q(x_1, x_0) = eq(r_2, 0)·p(0, x_1, x_0) + eq(r_2, 1)·p(1, x_1, x_0)

        // Compute equality polynomial values
        let eq_r2_0 = EF4::ONE - r2; // eq(r_2, 0) = 1 - r_2
        let eq_r2_1 = r2; // eq(r_2, 1) = r_2

        // Polynomial values
        let p_000 = F::from_u64(1);
        let p_001 = F::from_u64(2);
        let p_010 = F::from_u64(3);
        let p_011 = F::from_u64(4);
        let p_100 = F::from_u64(5);
        let p_101 = F::from_u64(6);
        let p_110 = F::from_u64(7);
        let p_111 = F::from_u64(8);

        // Compute folded values
        let q_00 = eq_r2_0 * p_000 + eq_r2_1 * p_100;
        let q_01 = eq_r2_0 * p_001 + eq_r2_1 * p_101;
        let q_10 = eq_r2_0 * p_010 + eq_r2_1 * p_110;
        let q_11 = eq_r2_0 * p_011 + eq_r2_1 * p_111;

        assert_eq!(result.0[0], q_00, "q(0,0) mismatch");
        assert_eq!(result.0[1], q_01, "q(0,1) mismatch");
        assert_eq!(result.0[2], q_10, "q(1,0) mismatch");
        assert_eq!(result.0[3], q_11, "q(1,1) mismatch");
    }

    #[test]
    fn test_fold_batch_two_variables() {
        // Fold a 3-variable polynomial over the first two variables.
        //
        // Polynomial: p(x_2, x_1, x_0) = 2^{x_2·4 + x_1·2 + x_0}
        //
        // Using powers of 2 for easier manual verification:
        //   p(0,0,0) = 2^0 = 1
        //   p(0,0,1) = 2^1 = 2
        //   p(0,1,0) = 2^2 = 4
        //   p(0,1,1) = 2^3 = 8
        //   p(1,0,0) = 2^4 = 16
        //   p(1,0,1) = 2^5 = 32
        //   p(1,1,0) = 2^6 = 64
        //   p(1,1,1) = 2^7 = 128

        let poly = EvaluationsList::new(vec![
            F::from_u64(1),   // 000
            F::from_u64(2),   // 001
            F::from_u64(4),   // 010
            F::from_u64(8),   // 011
            F::from_u64(16),  // 100
            F::from_u64(32),  // 101
            F::from_u64(64),  // 110
            F::from_u64(128), // 111
        ]);

        // Fold over (x_2, x_1) with challenges (r_2=2, r_1=3)
        let r2 = EF4::from_u64(2);
        let r1 = EF4::from_u64(3);
        let challenges = vec![r2, r1];

        let result = poly.fold_batch(&challenges);

        // VERIFY: Result has 2^(3-2) = 2 evaluations
        assert_eq!(
            result.0.len(),
            2,
            "Folded polynomial should have 2 evaluations"
        );

        //  MANUAL COMPUTATION
        //
        // After folding over (x_2, x_1), we get q(x_0) where:
        //
        //   q(x_0) = Σ_{x_2,x_1 ∈ {0,1}} eq(r_2, r_1; x_2, x_1)·p(x_2, x_1, x_0)

        // Compute equality polynomial values
        let eq_r2_0 = EF4::ONE - r2; // (1 - r_2)
        let eq_r2_1 = r2; // r_2
        let eq_r1_0 = EF4::ONE - r1; // (1 - r_1)
        let eq_r1_1 = r1; // r_1

        let eq_00 = eq_r2_0 * eq_r1_0; // eq(r_2, r_1; 0, 0)
        let eq_01 = eq_r2_0 * eq_r1_1; // eq(r_2, r_1; 0, 1)
        let eq_10 = eq_r2_1 * eq_r1_0; // eq(r_2, r_1; 1, 0)
        let eq_11 = eq_r2_1 * eq_r1_1; // eq(r_2, r_1; 1, 1)

        // Polynomial values (lifted to extension field)
        let p_000 = EF4::from_u64(1);
        let p_001 = EF4::from_u64(2);
        let p_010 = EF4::from_u64(4);
        let p_011 = EF4::from_u64(8);
        let p_100 = EF4::from_u64(16);
        let p_101 = EF4::from_u64(32);
        let p_110 = EF4::from_u64(64);
        let p_111 = EF4::from_u64(128);

        // Compute folded values
        let q_0 = eq_00 * p_000 + eq_01 * p_010 + eq_10 * p_100 + eq_11 * p_110;
        let q_1 = eq_00 * p_001 + eq_01 * p_011 + eq_10 * p_101 + eq_11 * p_111;

        assert_eq!(result.0[0], q_0, "q(0) mismatch");
        assert_eq!(result.0[1], q_1, "q(1) mismatch");
    }

    #[test]
    fn test_fold_batch_all_variables() {
        // Fold all variables, resulting in a single evaluation.
        //
        // Polynomial: p(x_1, x_0) = index + 1
        //   p(0,0) = 1, p(0,1) = 2, p(1,0) = 3, p(1,1) = 4

        let poly = EvaluationsList::new(vec![
            F::from_u64(1), // 00
            F::from_u64(2), // 01
            F::from_u64(3), // 10
            F::from_u64(4), // 11
        ]);

        // Fold all variables with challenges (r_1=5, r_0=7)
        let r1 = EF4::from_u64(5);
        let r0 = EF4::from_u64(7);
        let challenges = vec![r1, r0];

        let result = poly.fold_batch(&challenges);

        // VERIFY: Result should have exactly 1 evaluation
        assert_eq!(
            result.0.len(),
            1,
            "Folding all variables should produce a single value"
        );

        //  MANUAL COMPUTATION
        //
        // q() = Σ_{x_1,x_0} eq(r_1, r_0; x_1, x_0)·p(x_1, x_0)

        // Compute equality polynomial values
        let eq_r1_0 = EF4::ONE - r1; // (1 - r_1)
        let eq_r1_1 = r1; // r_1
        let eq_r0_0 = EF4::ONE - r0; // (1 - r_0)
        let eq_r0_1 = r0; // r_0

        let eq_00 = eq_r1_0 * eq_r0_0; // eq(r_1, r_0; 0, 0)
        let eq_01 = eq_r1_0 * eq_r0_1; // eq(r_1, r_0; 0, 1)
        let eq_10 = eq_r1_1 * eq_r0_0; // eq(r_1, r_0; 1, 0)
        let eq_11 = eq_r1_1 * eq_r0_1; // eq(r_1, r_0; 1, 1)

        // Polynomial values (lifted to extension field)
        let p_00 = EF4::from_u64(1);
        let p_01 = EF4::from_u64(2);
        let p_10 = EF4::from_u64(3);
        let p_11 = EF4::from_u64(4);

        // Compute fully folded value
        let q = eq_00 * p_00 + eq_01 * p_01 + eq_10 * p_10 + eq_11 * p_11;

        assert_eq!(result.0[0], q, "Folded value mismatch");
    }

    #[test]
    #[should_panic(expected = "fold_batch: challenges.len() = 3 exceeds num_variables() = 2")]
    fn test_fold_batch_too_many_challenges() {
        // 2-variable polynomial
        let poly = EvaluationsList::new(vec![
            F::from_u64(1), // 00
            F::from_u64(2), // 01
            F::from_u64(3), // 10
            F::from_u64(4), // 11
        ]);

        // Try to fold 3 variables (more than the 2 that exist)
        let challenges = vec![EF4::from_u64(2), EF4::from_u64(3), EF4::from_u64(5)];

        // This should panic
        let _ = poly.fold_batch(&challenges);
    }

    fn test_base_eval_consistency() {
        let mut rng = SmallRng::seed_from_u64(1);

        for k in 0..=18 {
            let poly: Vec<F> = (0..1 << k).map(|_| rng.random()).collect();
            let point = MultilinearPoint::<EF4>::rand(&mut rng, k);
            let point = point.as_slice();
            let e0 = eval_multilinear_recursive(&poly, point);
            let e1 = eval_multilinear(&poly, point);
            assert_eq!(e0, e1);
            let e1 = eval_multilinear_base(&poly, point);
            assert_eq!(e0, e1);
        }
    }

    #[test]
    fn test_ext_eval_consistency() {
        let mut rng = SmallRng::seed_from_u64(1);

        for k in 0..=18 {
            let poly: Vec<EF4> = (0..1 << k).map(|_| rng.random()).collect();
            let point = MultilinearPoint::<EF4>::rand(&mut rng, k);
            let point = point.as_slice();
            let e0 = eval_multilinear_recursive(&poly, point);
            let e1 = eval_multilinear(&poly, point);
            assert_eq!(e0, e1);
            let e1 = eval_multilinear_base(&poly, point);
            assert_eq!(e0, e1);
            let e1 = eval_multilinear_ext::<F, _>(&poly, point);
            assert_eq!(e0, e1);
        }
    }

    #[test]
    fn test_ext_eval_packed_consistency() {
        let mut rng = SmallRng::seed_from_u64(1);
        for k in 4..=18 {
            let poly: Vec<EF4> = (0..1 << k).map(|_| rng.random()).collect();
            let point = MultilinearPoint::<EF4>::rand(&mut rng, k);
            let e0 = eval_multilinear_recursive(&poly, point.as_slice());
            let packed = poly
                .par_chunks(<F as Field>::Packing::WIDTH)
                .map(<EF4 as ExtensionField<F>>::ExtensionPacking::from_ext_slice)
                .collect();
            let e1 = EvaluationsList::new(packed).eval_hypercube_packed::<F, _>(&point);
            assert_eq!(e0, e1);
        }
    }

    #[test]
    fn test_sumcheck_coefficients_one_variable() {
        // For a 1-variable polynomial (2 evaluations):
        //   evals   = [e0, e1] where f(0) = e0, f(1) = e1
        //   weights = [w0, w1] where g(0) = w0, g(1) = w1
        //
        // The sumcheck polynomial h(X) = f(X) * g(X) where:
        //   f(X) = e0 + (e1 - e0)*X
        //   g(X) = w0 + (w1 - w0)*X
        //
        // h(X) = [e0 + (e1-e0)*X] * [w0 + (w1-w0)*X]
        //      = e0*w0 + [e0*(w1-w0) + (e1-e0)*w0]*X + (e1-e0)*(w1-w0)*X^2
        //      = c0 + c1*X + c2*X^2
        //
        // where:
        //   c0 = e0 * w0
        //   c2 = (e1 - e0) * (w1 - w0)
        let e0 = EF4::from_u64(3);
        let e1 = EF4::from_u64(7);
        let w0 = EF4::from_u64(2);
        let w1 = EF4::from_u64(5);

        let evals = EvaluationsList::new(vec![e0, e1]);
        let weights = EvaluationsList::new(vec![w0, w1]);

        let (c0, c2) = evals.sumcheck_coefficients(&weights);

        // c0 = e0 * w0
        let expected_c0 = e0 * w0;
        assert_eq!(c0, expected_c0);

        // c2 = (e1 - e0) * (w1 - w0)
        let expected_c2 = (e1 - e0) * (w1 - w0);
        assert_eq!(c2, expected_c2);
    }

    #[test]
    fn test_sumcheck_coefficients_two_variables() {
        // For a 2-variable polynomial (4 evaluations):
        //   evals   = [e0, e1, e2, e3] representing f(x0, x1)
        //   weights = [w0, w1, w2, w3] representing g(x0, x1)
        //
        // Memory layout (lo = X=0, hi = X=1):
        //   f(0,0) = e0, f(0,1) = e1 (lo half)
        //   f(1,0) = e2, f(1,1) = e3 (hi half)
        //
        // The sumcheck polynomial for X (first variable):
        //   h(X) = Σ_{x1∈{0,1}} f(X, x1) * g(X, x1)
        //
        // At X=0: h(0) = f(0,0)*g(0,0) + f(0,1)*g(0,1) = e0*w0 + e1*w1
        // At X=1: h(1) = f(1,0)*g(1,0) + f(1,1)*g(1,1) = e2*w2 + e3*w3
        //
        // Coefficients:
        //   c0 = h(0) = e0*w0 + e1*w1
        //   c2 = Σ_{x1} (f(1,x1) - f(0,x1)) * (g(1,x1) - g(0,x1))
        //      = (e2-e0)*(w2-w0) + (e3-e1)*(w3-w1)
        let e0 = EF4::from_u64(1);
        let e1 = EF4::from_u64(2);
        let e2 = EF4::from_u64(5);
        let e3 = EF4::from_u64(8);
        let w0 = EF4::from_u64(3);
        let w1 = EF4::from_u64(4);
        let w2 = EF4::from_u64(6);
        let w3 = EF4::from_u64(7);

        let evals = EvaluationsList::new(vec![e0, e1, e2, e3]);
        let weights = EvaluationsList::new(vec![w0, w1, w2, w3]);

        let (c0, c2) = evals.sumcheck_coefficients(&weights);

        // c0 = e0*w0 + e1*w1
        let expected_c0 = e0 * w0 + e1 * w1;
        assert_eq!(c0, expected_c0);

        // c2 = (e2-e0)*(w2-w0) + (e3-e1)*(w3-w1)
        let expected_c2 = (e2 - e0) * (w2 - w0) + (e3 - e1) * (w3 - w1);
        assert_eq!(c2, expected_c2);
    }

    #[test]
    fn test_sumcheck_coefficients_three_variables() {
        // For a 3-variable polynomial (8 evaluations):
        //   evals   = [e0, e1, e2, e3, e4, e5, e6, e7] representing f(x0, x1, x2)
        //   weights = [w0, w1, w2, w3, w4, w5, w6, w7] representing g(x0, x1, x2)
        //
        // Memory layout (lo = X=0, hi = X=1 for first variable):
        //   lo half: e0, e1, e2, e3 = f(0, x1, x2)
        //   hi half: e4, e5, e6, e7 = f(1, x1, x2)
        //
        // c0 = h(0) = Σ_{x1,x2} f(0,x1,x2) * g(0,x1,x2)
        //    = e0*w0 + e1*w1 + e2*w2 + e3*w3
        //
        // c2 = Σ_{x1,x2} (f(1,x1,x2) - f(0,x1,x2)) * (g(1,x1,x2) - g(0,x1,x2))
        //    = (e4-e0)*(w4-w0) + (e5-e1)*(w5-w1) + (e6-e2)*(w6-w2) + (e7-e3)*(w7-w3)
        let e0 = EF4::from_u64(1);
        let e1 = EF4::from_u64(2);
        let e2 = EF4::from_u64(3);
        let e3 = EF4::from_u64(4);
        let e4 = EF4::from_u64(5);
        let e5 = EF4::from_u64(6);
        let e6 = EF4::from_u64(7);
        let e7 = EF4::from_u64(8);
        let w0 = EF4::from_u64(10);
        let w1 = EF4::from_u64(20);
        let w2 = EF4::from_u64(30);
        let w3 = EF4::from_u64(40);
        let w4 = EF4::from_u64(50);
        let w5 = EF4::from_u64(60);
        let w6 = EF4::from_u64(70);
        let w7 = EF4::from_u64(80);

        let evals = EvaluationsList::new(vec![e0, e1, e2, e3, e4, e5, e6, e7]);
        let weights = EvaluationsList::new(vec![w0, w1, w2, w3, w4, w5, w6, w7]);

        let (c0, c2) = evals.sumcheck_coefficients(&weights);

        // c0 = e0*w0 + e1*w1 + e2*w2 + e3*w3
        let expected_c0 = e0 * w0 + e1 * w1 + e2 * w2 + e3 * w3;
        assert_eq!(c0, expected_c0);

        // c2 = (e4-e0)*(w4-w0) + (e5-e1)*(w5-w1) + (e6-e2)*(w6-w2) + (e7-e3)*(w7-w3)
        let expected_c2 = (e4 - e0) * (w4 - w0)
            + (e5 - e1) * (w5 - w1)
            + (e6 - e2) * (w6 - w2)
            + (e7 - e3) * (w7 - w3);
        assert_eq!(c2, expected_c2);
    }

    #[test]
    fn test_sumcheck_coefficients_sum_constraint() {
        // Verify the sumcheck constraint: h(0) + h(1) = claimed_sum
        //
        // For the polynomial h(X) = c0 + c1*X + c2*X^2:
        //   h(0) = c0
        //   h(1) = c0 + c1 + c2
        //   h(0) + h(1) = 2*c0 + c1 + c2
        //
        // The claimed sum is: Σ_{x∈{0,1}^n} f(x) * g(x)
        //   = Σ_{x1∈{0,1}^{n-1}} [f(0,x1)*g(0,x1) + f(1,x1)*g(1,x1)]
        //   = h(0) + h(1)
        let e0 = EF4::from_u64(3);
        let e1 = EF4::from_u64(7);
        let w0 = EF4::from_u64(2);
        let w1 = EF4::from_u64(5);

        let evals = EvaluationsList::new(vec![e0, e1]);
        let weights = EvaluationsList::new(vec![w0, w1]);

        let (c0, c2) = evals.sumcheck_coefficients(&weights);

        // h(0) = c0 = e0 * w0
        let h_0 = c0;

        // h(1) = e1 * w1 (direct computation)
        let h_1 = e1 * w1;

        // claimed_sum = Σ_{x∈{0,1}} f(x) * g(x) = e0*w0 + e1*w1
        let claimed_sum = e0 * w0 + e1 * w1;

        // Verify: h(0) + h(1) = claimed_sum
        assert_eq!(h_0 + h_1, claimed_sum);

        // Also verify c1 derivation: c1 = claimed_sum - 2*c0 - c2
        //   h(0) + h(1) = 2*c0 + c1 + c2
        //   c1 = h(0) + h(1) - 2*c0 - c2
        let c1 = claimed_sum - c0.double() - c2;

        // Verify h(1) = c0 + c1 + c2
        let h_1_from_coeffs = c0 + c1 + c2;
        assert_eq!(h_1_from_coeffs, h_1);
    }

    #[test]
    fn test_sumcheck_coefficients_evaluate_at_challenge() {
        // Verify that h(r) can be computed correctly from (c0, c2) and claimed_sum.
        //
        // Given:
        //   c0 = h(0)
        //   c2 = quadratic coefficient
        //   claimed_sum = h(0) + h(1)
        //
        // We can derive:
        //   h(1) = claimed_sum - c0
        //   c1 = h(1) - c0 - c2 = claimed_sum - 2*c0 - c2
        //
        // Then: h(r) = c0 + c1*r + c2*r^2
        let e0 = EF4::from_u64(1);
        let e1 = EF4::from_u64(2);
        let e2 = EF4::from_u64(5);
        let e3 = EF4::from_u64(8);
        let w0 = EF4::from_u64(3);
        let w1 = EF4::from_u64(4);
        let w2 = EF4::from_u64(6);
        let w3 = EF4::from_u64(7);

        let evals = EvaluationsList::new(vec![e0, e1, e2, e3]);
        let weights = EvaluationsList::new(vec![w0, w1, w2, w3]);

        let (c0, c2) = evals.sumcheck_coefficients(&weights);

        // h(1) = e2*w2 + e3*w3
        let h_1 = e2 * w2 + e3 * w3;

        // Derive c1 = h(1) - c0 - c2
        let c1 = h_1 - c0 - c2;

        // Evaluate at challenge r
        let r = EF4::from_u64(7);
        let h_r = c0 + c1 * r + c2 * r.square();

        // Alternative: compute h(r) by folding and computing inner product
        // After folding evals with challenge r:
        //   f'(x1) = f(0,x1) + r * (f(1,x1) - f(0,x1))
        //   g'(x1) = g(0,x1) + r * (g(1,x1) - g(0,x1))
        // h(r) = Σ_{x1} f'(x1) * g'(x1)
        let folded_e0 = e0 + r * (e2 - e0);
        let folded_e1 = e1 + r * (e3 - e1);
        let folded_w0 = w0 + r * (w2 - w0);
        let folded_w1 = w1 + r * (w3 - w1);

        let h_r_from_folding = folded_e0 * folded_w0 + folded_e1 * folded_w1;

        assert_eq!(h_r, h_r_from_folding);
    }

    #[test]
    fn test_sumcheck_coefficients_mixed_field_types() {
        // Test with base field evaluations and extension field weights.
        //
        // The method signature is:
        //   fn sumcheck_coefficients<B>(&self, weights: &EvaluationsList<B>) -> (B, B)
        //   where B: Algebra<A>
        //
        // Since EF4: Algebra<F>, we use:
        //   evals   = [e0, e1] in F (base field)
        //   weights = [w0, w1] in EF4 (extension field)
        //
        // c0 = w0 * e0
        // c2 = (w1 - w0) * (e1 - e0)
        let e0 = F::from_u64(3);
        let e1 = F::from_u64(7);
        let w0 = EF4::from_u64(2);
        let w1 = EF4::from_u64(5);

        let evals = EvaluationsList::new(vec![e0, e1]);
        let weights = EvaluationsList::new(vec![w0, w1]);

        let (c0, c2): (EF4, EF4) = evals.sumcheck_coefficients(&weights);

        // c0 = w0 * e0 (extension field * base field)
        let expected_c0 = w0 * e0;
        assert_eq!(c0, expected_c0);

        // c2 = (w1 - w0) * (e1 - e0)
        let expected_c2 = (w1 - w0) * (e1 - e0);
        assert_eq!(c2, expected_c2);
    }

    #[test]
    fn test_sumcheck_coefficients_all_zeros() {
        // Edge case: all evaluations are zero.
        //
        // c0 = 0*0 + 0*0 = 0
        // c2 = (0-0)*(0-0) + (0-0)*(0-0) = 0
        let evals = EvaluationsList::new(vec![EF4::ZERO; 4]);
        let weights = EvaluationsList::new(vec![EF4::ZERO; 4]);

        let (c0, c2) = evals.sumcheck_coefficients(&weights);

        assert_eq!(c0, EF4::ZERO);
        assert_eq!(c2, EF4::ZERO);
    }

    #[test]
    fn test_sumcheck_coefficients_constant_polynomial() {
        // Edge case: constant polynomial (all evaluations equal).
        //
        // If f(x) = c for all x, and g(x) = d for all x:
        //   c0 = Σ_{b} c * d = 2^{n-1} * c * d (for n-variable polynomial)
        //   c2 = Σ_{b} (c - c) * (d - d) = 0
        let c = EF4::from_u64(5);
        let d = EF4::from_u64(3);

        // 2-variable polynomial: 4 evaluations, lo/hi halves of size 2
        let evals = EvaluationsList::new(vec![c, c, c, c]);
        let weights = EvaluationsList::new(vec![d, d, d, d]);

        let (c0, c2) = evals.sumcheck_coefficients(&weights);

        // c0 = 2 * c * d (sum over 2 elements in lo half)
        let expected_c0 = c * d + c * d;
        assert_eq!(c0, expected_c0);

        // c2 = 0 (all differences are zero)
        assert_eq!(c2, EF4::ZERO);
    }

    #[test]
    fn test_sumcheck_coefficients_linear_in_first_variable() {
        // If f(X, b) = X for all b (linear in first variable):
        //   f(0, b) = 0, f(1, b) = 1
        //
        // And g(X, b) = 1 for all b:
        //   g(0, b) = 1, g(1, b) = 1
        //
        // Then h(X) = Σ_b f(X, b) * g(X, b) = Σ_b X * 1 = 2^{n-1} * X
        //
        // For n=2 (4 evals): h(X) = 2 * X
        //   c0 = h(0) = 0
        //   c2 = 0 (linear polynomial has no quadratic term)
        //   c1 = 2
        let zero = EF4::ZERO;
        let one = EF4::ONE;

        // f: [f(0,0), f(0,1), f(1,0), f(1,1)] = [0, 0, 1, 1]
        let evals = EvaluationsList::new(vec![zero, zero, one, one]);
        // g: [g(0,0), g(0,1), g(1,0), g(1,1)] = [1, 1, 1, 1]
        let weights = EvaluationsList::new(vec![one, one, one, one]);

        let (c0, c2) = evals.sumcheck_coefficients(&weights);

        // c0 = f(0,0)*g(0,0) + f(0,1)*g(0,1) = 0*1 + 0*1 = 0
        let expected_c0 = zero * one + zero * one;
        assert_eq!(c0, expected_c0);

        // c2 = (f(1,0)-f(0,0))*(g(1,0)-g(0,0)) + (f(1,1)-f(0,1))*(g(1,1)-g(0,1))
        //    = (1-0)*(1-1) + (1-0)*(1-1) = 0 + 0 = 0
        let expected_c2 = (one - zero) * (one - one) + (one - zero) * (one - one);
        assert_eq!(c2, expected_c2);
    }

    #[test]
    #[should_panic]
    fn test_sumcheck_coefficients_mismatched_lengths() {
        // Panic if evals and weights have different lengths.
        let evals = EvaluationsList::new(vec![EF4::ONE; 4]);
        let weights = EvaluationsList::new(vec![EF4::ONE; 8]);

        let _ = evals.sumcheck_coefficients(&weights);
    }

    #[test]
    #[should_panic]
    fn test_sumcheck_coefficients_single_element() {
        // Panic if fewer than 2 elements (need at least 1 variable).
        let evals = EvaluationsList::new(vec![EF4::ONE]);
        let weights = EvaluationsList::new(vec![EF4::ONE]);

        let _ = evals.sumcheck_coefficients(&weights);
    }
}
