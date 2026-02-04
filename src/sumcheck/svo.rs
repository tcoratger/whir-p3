//! Split-Value Optimization (SVO) for the sumcheck protocol.
//!
//! This module implements Algorithm 5 from "Speeding Up Sum-Check Proving"
//! (https://eprint.iacr.org/2025/1117), which optimizes sumcheck proving when
//! the polynomial includes an equality polynomial factor.
//!
//! # Mathematical Background
//!
//! Let us consider the sumcheck protocol applied to polynomials of the form:
//!
//! ```text
//! g(X) = eq(w, X) * p(X)
//! ```
//!
//! where `eq` is the multilinear extension of the equality function. The SVO exploits
//! the special structure of the equality polynomial to reduce prover costs.
//!
//! ## Key Insight: Splitting the Equality Polynomial
//!
//! The equality polynomial can be decomposed as:
//!
//! ```text
//! eq(w, (r_{<i}, X, x')) = eq(w_{<i}, r_{<i}) * eq(w_i, X) * eq(w_{>i}, x')
//! ```
//!
//! This allows us to:
//! 1. Pre-compute smaller tables for the left and right components
//! 2. Avoid materializing the full 2^l-sized equality table
//! 3. Use Lagrange interpolation to reconstruct round polynomials from accumulators

use alloc::{vec, vec::Vec};

use itertools::Itertools;
use p3_field::{ExtensionField, Field, PackedFieldExtension, PackedValue, dot_product};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;

use crate::{
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    utils::log3_strict_usize,
};

/// Generates grid points for SVO accumulator evaluation.
///
/// For a given depth `l`, generates two arrays of multilinear points:
/// - `pts_0`: Points in `{0,1,2}^{l-1} x {0}` (used to compute `h(0)`)
/// - `pts_2`: Points in `{0,1,2}^{l-1} x {2}` (used to compute `h(2)`)
///
/// # Why These Specific Points?
///
/// In the sumcheck protocol, each round's polynomial `h(X)` is quadratic (degree 2).
/// To uniquely determine `h(X)`, we need evaluations at 3 points. We choose:
/// - `h(0)`: Computed from `pts_0` accumulators
/// - `h(1)`: Derived from the sumcheck constraint `h(0) + h(1) = claimed_sum`
/// - `h(2)`: Computed from `pts_2` accumulators
///
/// This avoids explicitly computing `h(1)`, saving work.
///
/// # Grid Structure
///
/// For `l` rounds, we need to evaluate at points where the last coordinate is fixed
/// (either 0 or 2), while the first `l-1` coordinates range over `{0, 1, 2}`.
/// This gives `3^{l-1}` points in each array.
///
/// # Arguments
///
/// * `l` - The number of SVO rounds (must be positive).
///
/// # Returns
///
/// An array `[pts_0, pts_2]` where:
/// - `pts_0` contains `3^{l-1}` points ending in 0
/// - `pts_2` contains `3^{l-1}` points ending in 2
///
/// # Panics
///
/// Panics if `l == 0`.
///
/// # Complexity
///
/// Time: O(3^l), Space: O(3^l)
pub(super) fn points_012<F: Field>(l: usize) -> [Vec<Vec<F>>; 2] {
    /// Expands a set of points by appending each value from `values` to each point.
    ///
    /// If `pts` has `n` points and `values` has `m` elements, the result has `m * n` points.
    fn expand<F: Field>(pts: &[Vec<F>], values: &[usize]) -> Vec<Vec<F>> {
        values
            .iter()
            .flat_map(|&v| {
                // For each value, clone all existing points and append the value.
                pts.iter().cloned().map(move |mut p| {
                    p.push(F::from_u32(v as u32));
                    p
                })
            })
            .collect()
    }

    // We need at least one round.
    assert!(l > 0, "points_012: l must be positive");

    // Start with the empty point (representing 0 dimensions).
    let mut pts = vec![vec![]];

    // Build up points in {0,1,2}^{l-1} by iteratively expanding.
    // After this loop, pts contains 3^{l-1} points.
    for _ in 0..l - 1 {
        pts = expand(&pts, &[0, 1, 2]);
    }

    // Create final points by appending 0 or 2 as the last coordinate.
    [expand(&pts, &[0]), expand(&pts, &[2])]
}

/// Computes SVO accumulators for a set of grid points.
///
/// For each grid point `u` in `us`, computes:
///
/// ```text
/// A(u) = f(u) * eq(u, point)
/// ```
///
/// where `f(u)` is derived from `partial_evals` and `eq` is the equality polynomial.
/// These accumulator values are later used with Lagrange interpolation
/// (via [`lagrange_weights_012_multi`]) to reconstruct the round polynomial.
///
/// # Mathematical Details
///
/// The computation splits the point into two parts:
/// - `z0`: The first `k - offset` coordinates (used for the "inner" equality)
/// - `z1`: The remaining `offset` coordinates (used for the "outer" equality)
///
/// The partial evaluations are first reduced over `z1` using the equality polynomial,
/// then combined with `z0` evaluations for each grid point `u`.
///
/// # Arguments
///
/// * `us` - Grid points in `{0,1,2}^l` (typically from [`points_012`]).
/// * `partial_evals` - Pre-computed partial evaluations from [`SplitEq::partial_evals`].
/// * `point` - The challenge point for the equality polynomial.
///
/// # Returns
///
/// A vector of accumulator values, one for each point in `us`.
///
/// # Complexity
///
/// Time: O(|us| * 2^{offset} + |partial_evals|)
pub(super) fn calculate_accumulators<F: Field, EF: ExtensionField<F>>(
    us: &[Vec<F>],
    partial_evals: &[EF],
    point: &[EF],
) -> Vec<EF> {
    // Determine the dimensions involved.
    // l0: log2 of partial_evals length (total variables in the partial evaluation domain)
    // offset: number of variables handled by the "outer" equality polynomial
    let l0 = log2_strict_usize(partial_evals.len());
    let offset = l0 - log3_strict_usize(us.len()) - 1;

    // Split the challenge point into inner (z0) and outer (z1) components.
    // z0 corresponds to the variables covered by the grid points.
    // z1 corresponds to the remaining variables handled separately.
    let (z0, z1) = point.split_at(point.len() - offset);

    // Build equality polynomial evaluation tables for both components.
    //
    // eq0: evaluations of eq(z0, x) for x in {0,1}^{|z0|}
    // eq1: evaluations of eq(z1, x) for x in {0,1}^{|z1|}
    let eq0 = EvaluationsList::new_from_point(z0, EF::ONE);
    let eq1 = EvaluationsList::new_from_point(z1, EF::ONE);

    // Reduce partial evaluations over the outer variables using eq1.
    //
    // This computes: sum_{x1} eq(z1, x1) * partial_evals[chunk_for_x1]
    let reduced_evals: Vec<EF> = partial_evals
        .chunks(eq1.num_evals())
        .map(|chunk| dot_product::<EF, _, _>(eq1.iter().copied(), chunk.iter().copied()))
        .collect();

    // For each grid point u, compute the accumulator value.
    //
    // This uses parallel iteration for better performance when |us| is large.
    // The computation for each u is independent, making it embarrassingly parallel.
    us.par_iter()
        .map(|u| {
            // Build the Lagrange coefficient vector for this grid point.
            // coeffs[x] = prod_{i} L_{u_i}(x_i) where L is the Lagrange basis.
            let coeffs = EvaluationsList::new_from_point(u.as_slice(), F::ONE);

            // Compute: (sum_x eq(z0, x) * coeffs(x)) * (sum_x reduced_evals[x] * coeffs(x))
            // This gives f(u) * eq(u, point) via the Lagrange interpolation formula.
            dot_product::<EF, _, _>(eq0.iter().copied(), coeffs.iter().copied())
                * dot_product::<EF, _, _>(reduced_evals.iter().copied(), coeffs.iter().copied())
        })
        .collect()
}

/// Split equality polynomial representation for optimized sumcheck proving.
///
/// Implements Algorithm 5 from "Speeding Up Sum-Check Proving"
/// (https://eprint.iacr.org/2025/1117), Section 5.2.
///
/// # Mathematical Background
///
/// The equality polynomial `eq(w, X)` for a point `w in F^k` is defined as:
///
/// ```text
/// eq(w, X) = prod_{i=1}^{k} (w_i * X_i + (1 - w_i) * (1 - X_i))
/// ```
///
/// This struct exploits the product structure by splitting `w` into three parts:
/// - `z_svo`: SVO part of `w` in point form (indices 0 to l0)
/// - `eq0`: The first half of the rest of `w` in polynomial form (indices l0 to k-l0/2)
/// - `eq1`: The second half of the rest of `w` in packed polynomial form (indices k-l0/2 to k)
///
/// This allows computing `eq(w, x)` for many `x` values efficiently by:
/// 1. Pre-computing small tables for each half
/// 2. Combining them as needed during the sumcheck protocol
///
/// # Memory Optimization
///
/// Instead of storing a 2^k-sized table of `eq(w, x)` values, we store:
/// - `z0`: 2^{(k-l0)/2} extension field elements
/// - `z1`: 2^{(k-l0)/2} packed extension field elements
///
/// This reduces memory from O(2^{k-l0}) to O(2^{(k-l0)/2}).
///
/// # Packed Representation
///
/// The `eq1` half uses packed field elements for SIMD acceleration.
/// The packing width is determined by `F::Packing::WIDTH`.
#[derive(Debug, Clone)]
pub(crate) struct SplitEqInner<F: Field, EF: ExtensionField<F>> {
    /// First part of the point where we apply the SVO optimization.
    z_svo: MultilinearPoint<EF>,

    /// Evaluations of eq for the lower-index half of rest of the point.
    ///
    /// Contains 2^{(k-l0)/2} extension field elements.
    eq0: EvaluationsList<EF>,

    /// Evaluations of eq for the higher-index half of rest of the point.
    ///
    /// Stored in packed form for SIMD operations.
    /// Contains 2^{(k-l0)/2 - log2(packing_width)} packed elements.
    eq1: EvaluationsList<EF::ExtensionPacking>,
}

/// Split equality polynomial with precomputed accumulators for optimized sumcheck proving.
///
/// This struct wraps [`SplitEqInner`] and extends it with precomputed values needed
/// during the sumcheck protocol. It combines the split representation with:
/// - Precomputed accumulators for all SVO rounds
/// - The evaluation `sum_x eq(w, x) * poly(x)` of the weighted polynomial
/// - The original challenge point for reference
///
/// # Construction
///
/// Created via [`SplitEq::new`], this struct:
/// 1. Builds the inner split representation via [`SplitEqInner::new`]
/// 2. Computes partial evaluations of the polynomial weighted by the split eq
/// 3. Precomputes accumulators for rounds 1 through `l` (the SVO depth)
///
/// # Memory
///
/// By precomputing all accumulators upfront, we trade memory for speed:
/// - Memory: O(3^l) total accumulator values across all rounds
/// - Benefit: Each SVO round can directly access its accumulators without recomputation
///
/// # Fields
///
/// - `inner`: The underlying split equality representation
/// - `accumulators`: Precomputed accumulator values for each SVO round.
///   `accumulators[i]` contains `[acc_0, acc_2]` for round `i+1`, where:
///   - `acc_0`: Accumulators for grid points ending in 0 (for computing `h(0)`)
///   - `acc_2`: Accumulators for grid points ending in 2 (for computing `h(2)`)
/// - `eval`: The sum `sum_x eq(w, x) * poly(x)` over the boolean hypercube
/// - `point`: The original challenge point `w`
#[derive(Debug, Clone)]
pub(crate) struct SplitEq<F: Field, EF: ExtensionField<F>> {
    /// The underlying split equality polynomial representation.
    inner: SplitEqInner<F, EF>,

    /// Precomputed accumulators for all SVO rounds.
    ///
    /// `accumulators[i]` contains the accumulator values for round `i+1`:
    /// - `accumulators[i][0]`: Values for grid points in `{0,1,2}^i x {0}`
    /// - `accumulators[i][1]`: Values for grid points in `{0,1,2}^i x {2}`
    accumulators: Vec<[Vec<EF>; 2]>,

    /// The evaluation `sum_x eq(w, x) * poly(x)` of the polynomial weighted by eq.
    ///
    /// This is the initial claimed sum for the sumcheck protocol.
    pub(crate) eval: EF,

    /// The original challenge point `w` for the equality polynomial.
    pub(crate) point: MultilinearPoint<EF>,
}

impl<F: Field, EF: ExtensionField<F>> SplitEq<F, EF> {
    /// Creates a new split equality polynomial with precomputed accumulators.
    ///
    /// This constructor:
    /// 1. Builds the inner split representation from the challenge point
    /// 2. Computes partial evaluations of `poly` weighted by the equality polynomial
    /// 3. Evaluates `poly` at `point`, storing the result in `eval`
    /// 4. Precomputes accumulators for all `l` SVO rounds
    ///
    /// # Arguments
    ///
    /// * `point` - The challenge point `w in EF^k` for the equality polynomial.
    /// * `l` - The number of SVO rounds (depth of the SVO optimization).
    /// * `poly` - The polynomial in evaluation form to be weighted by eq.
    #[tracing::instrument(skip_all)]
    pub(crate) fn new(point: &MultilinearPoint<EF>, l: usize, poly: &EvaluationsList<F>) -> Self {
        assert_eq!(point.num_variables(), poly.num_variables());
        let inner = SplitEqInner::new(point, l);
        debug_assert_eq!(inner.k(), poly.num_variables());
        let (partial_evals, eval) = inner.partial_evals(poly);
        let accumulators = inner.accumulators(&partial_evals);
        Self {
            inner,
            accumulators,
            eval,
            point: point.clone(),
        }
    }

    /// Returns the total number of variables `k` in the original point.
    pub(crate) const fn num_variables(&self) -> usize {
        self.inner.k()
    }

    /// Returns the number of variables handled by the split eq (excluding SVO variables).
    ///
    /// This is `k - l` where `l` is the SVO depth.
    const fn k_split(&self) -> usize {
        self.inner.k_split()
    }

    /// Returns the precomputed accumulators for all SVO rounds.
    ///
    /// `accumulators()[i]` contains `[acc_0, acc_2]` for round `i+1`.
    pub(crate) const fn accumulators(&self) -> &Vec<[Vec<EF>; 2]> {
        &self.accumulators
    }

    #[tracing::instrument(skip_all, fields(k = log2_strict_usize(out.len()), selfs = selfs.len()))]
    pub(crate) fn combine_into_packed(
        out: &mut [EF::ExtensionPacking],
        selfs: &[Self],
        alpha: EF,
        rs: &[EF],
    ) {
        let inners = selfs.iter().map(|s| &s.inner).collect::<Vec<_>>();
        SplitEqInner::combine_into_packed(out, &inners, alpha, rs);
    }
}

impl<F: Field, EF: ExtensionField<F>> SplitEqInner<F, EF> {
    #[tracing::instrument(skip_all)]
    pub(crate) fn new(z: &MultilinearPoint<EF>, l: usize) -> Self {
        let k = z.num_variables();
        assert!(k > l);
        assert!(k >= 2 * log2_strict_usize(F::Packing::WIDTH));
        let (z_svo, z_rest) = z.split_at(l);
        let (z0, z1) = z_rest.split_at((k - l) / 2);
        let eq0 = EvaluationsList::new_from_point(z0.as_slice(), EF::ONE);
        let eq1 = EvaluationsList::new_packed_from_point(z1.as_slice(), EF::ONE);
        Self { z_svo, eq0, eq1 }
    }

    pub(crate) const fn k(&self) -> usize {
        self.k_split() + self.k_svo()
    }

    pub(crate) const fn k_svo(&self) -> usize {
        self.z_svo.num_variables()
    }

    const fn k_split(&self) -> usize {
        self.eq1.num_variables() + self.eq0.num_variables() + log2_strict_usize(F::Packing::WIDTH)
    }

    /// Computes accumulators for all SVO rounds from partial evaluations.
    ///
    /// For each round `i` (1 to `l`), computes accumulator values at grid points
    /// in `{0,1,2}^{i-1} x {0}` and `{0,1,2}^{i-1} x {2}`. These are used to
    /// reconstruct the round polynomial `h(X)` via Lagrange interpolation:
    /// - `acc_0` values contribute to `h(0)`
    /// - `acc_2` values contribute to `h(2)`
    /// - `h(1)` is derived from the sumcheck relation `h(0) + h(1) = claimed_sum`
    #[tracing::instrument(skip_all)]
    pub(crate) fn accumulators(&self, partial_evals: &[EF]) -> Vec<[Vec<EF>; 2]> {
        (1..=self.k_svo())
            .map(|i| {
                let us = points_012::<F>(i);
                let acc0 = calculate_accumulators(&us[0], partial_evals, self.z_svo.as_slice());
                let acc2 = calculate_accumulators(&us[1], partial_evals, self.z_svo.as_slice());
                [acc0, acc2]
            })
            .collect::<Vec<_>>()
    }

    /// Computes partial evaluations of a polynomial weighted by the split eq.
    ///
    /// For a polynomial `poly` over `F^n` where `n >= k`, computes partial sums
    /// that incorporate the equality polynomial weighting. The result is used
    /// by [`SplitEqInner::accumulators`] to build the accumulator values.
    ///
    /// # Mathematical Formula
    ///
    /// For each chunk of `poly` of size `2^{k-l}` (one per SVO hypercube point), computes:
    ///
    /// ```text
    /// sum_{x_0, x_1} eq0[x_0] * eq1[x_1] * poly[x_0, x_1]
    /// ```
    ///
    /// where `x_0` ranges over the `eq0` indices and `x_1` over the `eq1` indices.
    ///
    /// Also computes the full evaluation by combining partial evals with `eq_svo`.
    ///
    /// # Returns
    ///
    /// A tuple `(partial_evals, eval)` where:
    /// - `partial_evals`: One value per chunk of size `2^{k-l}` in `poly`
    /// - `eval`: The full weighted evaluation `sum_x eq(z, x) * poly(x)`
    #[tracing::instrument(skip_all)]
    pub(crate) fn partial_evals(&self, poly: &EvaluationsList<F>) -> (Vec<EF>, EF) {
        let chunk_size = 1
            << (self.eq1.num_variables()
                + self.eq0.num_variables()
                + log2_strict_usize(F::Packing::WIDTH));
        let partial_evals = poly
            .0
            .chunks(chunk_size)
            .map(|poly| {
                // Pack the polynomial chunk for SIMD operations.
                let poly = F::Packing::pack_slice(poly);

                // Compute the double sum: sum_{x_L} sum_{x_R} left[x_L] * right[x_R] * poly[x_L, x_R]
                // The outer sum is over right indices (parallelized).
                // The inner sum is over left indices (vectorized via packing).
                let sum = poly
                    .par_chunks(self.eq1.0.len())
                    .zip_eq(self.eq0.0.par_iter())
                    .map(|(poly, &eq0)| {
                        // Inner sum: sum_{x_1} eq1[x_1] * poly[x_0, x_1]
                        let inner_sum = poly
                            .iter()
                            .zip_eq(self.eq1.0.iter())
                            .map(|(&f, &eq1)| eq1 * f)
                            .sum::<EF::ExtensionPacking>();
                        // Multiply by the eq0 weight.
                        inner_sum * eq0
                    })
                    .sum::<EF::ExtensionPacking>();

                // Unpack the result to get a single extension field element.
                EF::ExtensionPacking::to_ext_iter([sum]).sum()
            })
            .collect::<Vec<_>>();

        let eq_svo = EvaluationsList::new_from_point(self.z_svo.as_slice(), EF::ONE);
        let eval = dot_product::<EF, _, _>(eq_svo.iter().copied(), partial_evals.iter().copied());
        (partial_evals, eval)
    }

    fn eval_z_svo(&self, alpha: EF, rs: &[EF]) -> EF {
        EvaluationsList::new_from_point(self.z_svo.as_slice(), alpha)
            .evaluate_hypercube_ext(&MultilinearPoint::new(rs.to_vec()))
    }

    fn combine_into_packed(
        out: &mut [EF::ExtensionPacking],
        selfs: &[&Self],
        alpha: EF,
        rs: &[EF],
    ) {
        if selfs.is_empty() {
            return;
        }
        let k = selfs
            .iter()
            .map(|eq_split| eq_split.k_split())
            .all_equal_value()
            .unwrap();
        assert_eq!(out.len(), 1 << (k - log2_strict_usize(F::Packing::WIDTH)));
        for (eq_split, alpha) in selfs.iter().zip(alpha.powers()) {
            let scale = eq_split.eval_z_svo(alpha, rs);
            out.par_chunks_mut(eq_split.eq1.num_evals())
                .zip(eq_split.eq0.0.par_iter())
                .for_each(|(chunk, &right)| {
                    // For each position in the chunk, add: left[pos] * right * scale
                    chunk
                        .iter_mut()
                        .zip(eq_split.eq1.0.iter())
                        .for_each(|(out, &left)| *out += left * right * scale);
                });
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_field::{PrimeCharacteristicRing, dot_product, extension::BinomialExtensionField};
    use p3_koala_bear::KoalaBear;
    use proptest::prelude::*;
    use rand::{Rng, SeedableRng, rngs::SmallRng};

    use super::*;

    type F = KoalaBear;
    type EF = BinomialExtensionField<F, 4>;

    #[test]
    fn test_points_012_l1() {
        // For l=1: pts should have 3^0 = 1 point each.
        // pts_0 = [(0,)]
        // pts_2 = [(2,)]
        let [pts_0, pts_2] = points_012::<F>(1);

        assert_eq!(pts_0.len(), 1);
        assert_eq!(pts_2.len(), 1);

        assert_eq!(pts_0[0], vec![F::ZERO]);
        assert_eq!(pts_2[0], vec![F::TWO]);
    }

    #[test]
    fn test_points_012_l2() {
        // For l=2: pts should have 3^1 = 3 points each.
        // First l-1=1 coordinates in {0,1,2}, last coordinate fixed.
        let [pts_0, pts_2] = points_012::<F>(2);

        assert_eq!(pts_0.len(), 3);
        assert_eq!(pts_2.len(), 3);

        // All pts_0 should end with 0.
        for pt in &pts_0 {
            assert_eq!(pt.len(), 2);
            assert_eq!(*pt.last().unwrap(), F::ZERO);
        }

        // All pts_2 should end with 2.
        for pt in &pts_2 {
            assert_eq!(pt.len(), 2);
            assert_eq!(*pt.last().unwrap(), F::TWO);
        }
    }

    #[test]
    fn test_points_012_sizes() {
        // Verify output sizes: each array should have 3^{l-1} points.
        for l in 1..=6 {
            let [pts_0, pts_2] = points_012::<F>(l);
            let expected_size = 3usize.pow((l - 1) as u32);

            assert_eq!(pts_0.len(), expected_size, "pts_0 size mismatch for l={l}");
            assert_eq!(pts_2.len(), expected_size, "pts_2 size mismatch for l={l}");

            // Each point should have l coordinates.
            for pt in pts_0.iter().chain(pts_2.iter()) {
                assert_eq!(pt.len(), l, "point dimension mismatch for l={l}");
            }
        }
    }

    #[test]
    fn test_points_012_values_in_range() {
        // All coordinates should be in {0, 1, 2}.
        let [pts_0, pts_2] = points_012::<F>(4);

        let valid_values = [F::ZERO, F::ONE, F::TWO];

        for pt in pts_0.iter().chain(pts_2.iter()) {
            for (i, &coord) in pt.iter().enumerate() {
                // Last coordinate is fixed (0 for pts_0, 2 for pts_2).
                // Other coordinates should be in {0, 1, 2}.
                if i < pt.len() - 1 {
                    assert!(
                        valid_values.contains(&coord),
                        "invalid coordinate value at position {i}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_points_012_unique() {
        // All points within each array should be unique.
        let [pts_0, pts_2] = points_012::<F>(4);

        let pts_0_set: alloc::collections::BTreeSet<_> = pts_0.iter().cloned().collect();
        let pts_2_set: alloc::collections::BTreeSet<_> = pts_2.iter().cloned().collect();

        assert_eq!(pts_0_set.len(), pts_0.len(), "pts_0 contains duplicates");
        assert_eq!(pts_2_set.len(), pts_2.len(), "pts_2 contains duplicates");
    }

    #[test]
    #[should_panic(expected = "l must be positive")]
    fn test_points_012_panics_on_zero() {
        let _: [Vec<Vec<F>>; 2] = points_012(0);
    }

    #[test]
    fn test_accumulators_correctness() {
        // Main correctness test: verify accumulators match naive computation.
        let k = 10;
        let mut rng = SmallRng::seed_from_u64(1);

        let f = EvaluationsList::new((0..1 << k).map(|_| rng.random()).collect());
        let z = MultilinearPoint::<EF>::rand(&mut rng, f.num_variables());
        let eq = EvaluationsList::new_from_point(z.as_slice(), EF::ONE);

        for l in 1..k / 2 {
            let split_eq = SplitEq::<F, EF>::new(&z, l, &f);

            let accumulators = split_eq.accumulators;
            assert_eq!(accumulators.len(), l);

            for (i, accumulator) in accumulators.iter().enumerate() {
                let us = points_012::<F>(i + 1);
                us[0]
                    .iter()
                    .zip(accumulator[0].iter())
                    .for_each(|(u, &acc)| {
                        let u = u.iter().copied().map(EF::from).collect::<Vec<_>>();
                        let f = f.compress_multi(&u);
                        let eq = eq.compress_multi(&u);
                        let e1: EF = dot_product(eq.iter().copied(), f.iter().copied());
                        assert_eq!(acc, e1);
                    });

                us[1]
                    .iter()
                    .zip(accumulator[1].iter())
                    .for_each(|(u, &acc)| {
                        let u = u.iter().copied().map(EF::from).collect::<Vec<_>>();
                        let f = f.compress_multi(&u);
                        let eq = eq.compress_multi(&u);
                        let e1: EF = dot_product(eq.iter().copied(), f.iter().copied());
                        assert_eq!(acc, e1);
                    });
            }
        }
    }

    #[test]
    fn test_split_eq_k_calculation() {
        // Verify that k() returns the original number of variables.
        let mut rng = SmallRng::seed_from_u64(42);

        for k in [8, 10, 12, 14] {
            let point = MultilinearPoint::<EF>::rand(&mut rng, k);
            let poly = EvaluationsList::new((0..1 << k).map(|_| rng.random()).collect());
            let split_eq = SplitEq::<F, EF>::new(&point, 0, &poly);
            assert_eq!(split_eq.inner.k(), k, "k() mismatch for k={k}");
        }
    }

    #[test]
    fn test_split_eq_partial_evals_size() {
        // Verify partial_evals returns the correct number of elements.
        let mut rng = SmallRng::seed_from_u64(42);
        let k = 10;
        let n = 14; // Total polynomial variables.

        let point = MultilinearPoint::<EF>::rand(&mut rng, k);
        let poly = EvaluationsList::new((0..1 << k).map(|_| rng.random()).collect());
        let split_eq = SplitEq::<F, EF>::new(&point, 0, &poly);

        let poly = EvaluationsList::new((0..1 << n).map(|_| rng.random()).collect());
        let (partial, _) = split_eq.inner.partial_evals(&poly);

        // Should have one element per 2^k chunk.
        let expected_len = 1 << (n - k);
        assert_eq!(partial.len(), expected_len);
    }

    proptest! {
        /// Verify that points_012 generates the correct number of points.
        #[test]
        fn prop_points_012_sizes(l in 1usize..=8) {
            let [pts_0, pts_2] = points_012::<F>(l);
            let expected = 3usize.pow((l - 1) as u32);

            prop_assert_eq!(pts_0.len(), expected);
            prop_assert_eq!(pts_2.len(), expected);
        }

        /// Verify that all pts_0 points end with 0 and all pts_2 points end with 2.
        #[test]
        fn prop_points_012_last_coordinate(l in 1usize..=6) {
            let [pts_0, pts_2] = points_012::<F>(l);

            for pt in &pts_0 {
                prop_assert_eq!(*pt.last().unwrap(), F::ZERO);
            }
            for pt in &pts_2 {
                prop_assert_eq!(*pt.last().unwrap(), F::TWO);
            }
        }

        /// Verify that SplitEq::k() returns the original point dimension.
        #[test]
        fn prop_split_eq_k_consistency(k in 8usize..=14) {
            let mut rng = SmallRng::seed_from_u64(k as u64);
            let point = MultilinearPoint::<EF>::rand(&mut rng, k);
            let poly = EvaluationsList::zero(k);
            let split_eq = SplitEq::<F, EF>::new(&point, 0, &poly);

            prop_assert_eq!(split_eq.inner.k(), k);
        }
    }
}
