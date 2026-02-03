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
pub(super) fn points_012<F: Field>(l: usize) -> [Vec<MultilinearPoint<F>>; 2] {
    /// Expands a set of points by appending each value from `values` to each point.
    ///
    /// If `pts` has `n` points and `values` has `m` elements, the result has `m * n` points.
    fn expand<F: Field>(pts: &[MultilinearPoint<F>], values: &[usize]) -> Vec<MultilinearPoint<F>> {
        values
            .iter()
            .flat_map(|&v| {
                // For each value, clone all existing points and append the value.
                pts.iter().cloned().map(move |mut p| {
                    p.0.push(F::from_u32(v as u32));
                    p
                })
            })
            .collect()
    }

    // We need at least one round.
    assert!(l > 0, "points_012: l must be positive");

    // Start with the empty point (representing 0 dimensions).
    let mut pts = vec![MultilinearPoint::new(vec![])];

    // Build up points in {0,1,2}^{l-1} by iteratively expanding.
    // After this loop, pts contains 3^{l-1} points.
    for _ in 0..l - 1 {
        pts = expand(pts.as_slice(), &[0, 1, 2]);
    }

    // Create final points by appending 0 or 2 as the last coordinate.
    [expand(pts.as_slice(), &[0]), expand(pts.as_slice(), &[2])]
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
    us: &[MultilinearPoint<F>],
    partial_evals: &[EF],
    point: &MultilinearPoint<EF>,
) -> Vec<EF> {
    // Determine the dimensions involved.
    // l0: log2 of partial_evals length (total variables in the partial evaluation domain)
    // offset: number of variables handled by the "outer" equality polynomial
    let l0 = log2_strict_usize(partial_evals.len());
    let offset = l0 - log3_strict_usize(us.len()) - 1;

    // Split the challenge point into inner (z0) and outer (z1) components.
    // z0 corresponds to the variables covered by the grid points.
    // z1 corresponds to the remaining variables handled separately.
    let (z0, z1) = point.split_at(point.num_variables() - offset);

    // Build equality polynomial evaluation tables for both components.
    //
    // eq0: evaluations of eq(z0, x) for x in {0,1}^{|z0|}
    // eq1: evaluations of eq(z1, x) for x in {0,1}^{|z1|}
    let eq0 = EvaluationsList::new_from_point(z0.as_slice(), EF::ONE);
    let eq1 = EvaluationsList::new_from_point(z1.as_slice(), EF::ONE);

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
/// This struct exploits the product structure by splitting `w` into two halves:
/// - `left`: The second half of `w` (indices k/2 to k-1)
/// - `right`: The first half of `w` (indices 0 to k/2-1)
///
/// This allows computing `eq(w, x)` for many `x` values efficiently by:
/// 1. Pre-computing small tables for each half
/// 2. Combining them as needed during the sumcheck protocol
///
/// # Memory Optimization
///
/// Instead of storing a 2^k-sized table of `eq(w, x)` values, we store:
/// - `left`: 2^{k/2} packed extension field elements
/// - `right`: 2^{k/2} extension field elements
///
/// This reduces memory from O(2^k) to O(2^{k/2}).
///
/// # Packed Representation
///
/// The `left` half uses packed field elements for SIMD acceleration.
/// The packing width is determined by `F::Packing::WIDTH`.
#[derive(Debug, Clone)]
pub(super) struct SplitEq<F: Field, EF: ExtensionField<F>> {
    /// Evaluations of eq for the left (higher-index) half of the point.
    ///
    /// Stored in packed form for SIMD operations.
    /// Contains 2^{k/2 - log2(packing_width)} packed elements.
    left: EvaluationsList<EF::ExtensionPacking>,

    /// Evaluations of eq for the right (lower-index) half of the point.
    ///
    /// Contains 2^{k/2} extension field elements.
    right: EvaluationsList<EF>,
}

impl<F: Field, EF: ExtensionField<F>> SplitEq<F, EF> {
    /// Creates a new split equality polynomial from a challenge point.
    ///
    /// Splits the point at the midpoint and pre-computes evaluation tables
    /// for each half. The left half is stored in packed form for SIMD acceleration.
    ///
    /// # Arguments
    ///
    /// * `point` - The challenge point `w in EF^k` for the equality polynomial.
    /// * `alpha` - A scaling factor applied to the left half evaluations.
    ///
    /// # Panics
    ///
    /// Panics if `k < 2 * log2(packing_width)`, as we need enough variables
    /// to fill at least one packed element on each side.
    ///
    /// # Complexity
    ///
    /// Time: O(2^{k/2}), Space: O(2^{k/2})
    #[tracing::instrument(skip_all)]
    pub(super) fn new(point: &MultilinearPoint<EF>, alpha: EF) -> Self {
        let k = point.num_variables();

        // Ensure we have enough variables for the packed representation.
        // We need at least log2(packing_width) variables on each side.
        assert!(
            k >= 2 * log2_strict_usize(F::Packing::WIDTH),
            "SplitEq::new: point must have at least {} variables, got {}",
            2 * log2_strict_usize(F::Packing::WIDTH),
            k
        );

        // Split at the midpoint: right gets [0, k/2), left gets [k/2, k).
        let (right, left) = point.split_at(k / 2);

        // Build evaluation tables for each half.
        // The left half is packed and scaled by alpha.
        let left = EvaluationsList::new_packed_from_point(left.as_slice(), alpha);
        let right = EvaluationsList::new_from_point(right.as_slice(), EF::ONE);

        Self { left, right }
    }

    /// Returns the total number of variables `k` in the original point.
    ///
    /// This is computed from the sizes of the left and right evaluation tables,
    /// accounting for the packing width used in the left table.
    const fn k(&self) -> usize {
        // left has 2^{k/2 - log2(packing)} elements
        // right has 2^{k/2} elements
        // k = left_vars + right_vars + log2(packing)
        self.left.num_variables()
            + self.right.num_variables()
            + log2_strict_usize(F::Packing::WIDTH)
    }

    /// Computes partial evaluations of a polynomial weighted by the split eq.
    ///
    /// For a polynomial `poly` over `F^n` where `n >= k`, computes partial sums
    /// that incorporate the equality polynomial weighting. The result is used
    /// by [`calculate_accumulators`] to build the final accumulator values.
    ///
    /// # Mathematical Formula
    ///
    /// For each chunk of `poly` of size `2^k`, computes:
    ///
    /// ```text
    /// sum_{x_L, x_R} left[x_L] * right[x_R] * poly[x_L, x_R]
    /// ```
    ///
    /// where `x_L` ranges over the left half indices and `x_R` over the right half.
    ///
    /// # Parallelization
    ///
    /// The inner loops are parallelized using rayon for large polynomials.
    ///
    /// # Arguments
    ///
    /// * `poly` - A polynomial in evaluation form over the base field.
    ///
    /// # Returns
    ///
    /// A vector of partial evaluations, one per chunk of size `2^k` in `poly`.
    ///
    /// # Complexity
    ///
    /// Time: O(|poly|), Space: O(|poly| / 2^k)
    #[tracing::instrument(skip_all)]
    pub(super) fn partial_evals(&self, poly: &EvaluationsList<F>) -> Vec<EF> {
        // Each chunk of size 2^k is processed independently.
        let chunk_size = 1 << self.k();

        poly.0
            .chunks(chunk_size)
            .map(|poly| {
                // Pack the polynomial chunk for SIMD operations.
                let poly = F::Packing::pack_slice(poly);

                // Compute the double sum: sum_{x_L} sum_{x_R} left[x_L] * right[x_R] * poly[x_L, x_R]
                // The outer sum is over right indices (parallelized).
                // The inner sum is over left indices (vectorized via packing).
                let e_part = poly
                    .par_chunks(self.left.0.len())
                    .zip_eq(self.right.0.par_iter())
                    .map(|(poly_chunk, &right)| {
                        // Inner sum: sum_{x_L} left[x_L] * poly[x_L, x_R]
                        let inner_sum: EF::ExtensionPacking = poly_chunk
                            .iter()
                            .zip_eq(self.left.0.iter())
                            .map(|(&f, &left)| left * f)
                            .sum();
                        // Multiply by the right weight.
                        inner_sum * right
                    })
                    .sum::<EF::ExtensionPacking>();

                // Unpack the result to get a single extension field element.
                EF::ExtensionPacking::to_ext_iter([e_part]).sum()
            })
            .collect()
    }

    /// Combines multiple split eq polynomials into a single packed output.
    ///
    /// This is used after the SVO rounds to merge the split eq representations
    /// back into a single weight vector for subsequent sumcheck rounds.
    ///
    /// # Mathematical Operation
    ///
    /// For each output position, accumulates:
    ///
    /// ```text
    /// out[i] += sum_j scale[j] * eqs[j].left[i % left_len] * eqs[j].right[i / left_len]
    /// ```
    ///
    /// # Parallelization
    ///
    /// The output is updated in parallel chunks for efficiency.
    ///
    /// # Arguments
    ///
    /// * `out` - Output buffer to accumulate into (must be correctly sized).
    /// * `eqs` - Slice of split eq polynomials to combine.
    /// * `scale` - Scaling factors for each split eq polynomial.
    ///
    /// # Panics
    ///
    /// - Panics if `eqs` is non-empty but the polynomials have different `k` values.
    /// - Panics if `out.len() != 2^{k - log2(packing_width)}`.
    /// - Panics if `scale.len() != eqs.len()`.
    #[tracing::instrument(skip_all, fields(k = log2_strict_usize(out.len()), eqs = eqs.len()))]
    pub(super) fn into_packed(out: &mut [EF::ExtensionPacking], eqs: &[Self], scale: &[EF]) {
        // Nothing to do if there are no split eqs.
        if eqs.is_empty() {
            return;
        }

        // Verify all split eqs have the same k.
        let k = eqs.iter().map(Self::k).all_equal_value().unwrap();

        // Verify output buffer size.
        assert_eq!(
            out.len(),
            1 << (k - log2_strict_usize(F::Packing::WIDTH)),
            "into_packed: output buffer has wrong size"
        );

        // Verify scale vector length.
        assert_eq!(
            scale.len(),
            eqs.len(),
            "into_packed: scale vector length mismatch"
        );

        // Accumulate each split eq into the output.
        for (eq, &scale) in eqs.iter().zip(scale.iter()) {
            // Process output in chunks matching the left table size.
            out.par_chunks_mut(eq.left.0.len())
                .zip(eq.right.0.par_iter())
                .for_each(|(chunk, &right)| {
                    // For each position in the chunk, add: left[pos] * right * scale
                    chunk
                        .iter_mut()
                        .zip(eq.left.iter())
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

        assert_eq!(pts_0[0].0, vec![F::ZERO]);
        assert_eq!(pts_2[0].0, vec![F::TWO]);
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
            assert_eq!(pt.0.len(), 2);
            assert_eq!(*pt.0.last().unwrap(), F::ZERO);
        }

        // All pts_2 should end with 2.
        for pt in &pts_2 {
            assert_eq!(pt.0.len(), 2);
            assert_eq!(*pt.0.last().unwrap(), F::TWO);
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
                assert_eq!(pt.0.len(), l, "point dimension mismatch for l={l}");
            }
        }
    }

    #[test]
    fn test_points_012_values_in_range() {
        // All coordinates should be in {0, 1, 2}.
        let [pts_0, pts_2] = points_012::<F>(4);

        let valid_values = [F::ZERO, F::ONE, F::TWO];

        for pt in pts_0.iter().chain(pts_2.iter()) {
            for (i, &coord) in pt.0.iter().enumerate() {
                // Last coordinate is fixed (0 for pts_0, 2 for pts_2).
                // Other coordinates should be in {0, 1, 2}.
                if i < pt.0.len() - 1 {
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

        let pts_0_set: alloc::collections::BTreeSet<_> =
            pts_0.iter().map(|p| p.0.clone()).collect();
        let pts_2_set: alloc::collections::BTreeSet<_> =
            pts_2.iter().map(|p| p.0.clone()).collect();

        assert_eq!(pts_0_set.len(), pts_0.len(), "pts_0 contains duplicates");
        assert_eq!(pts_2_set.len(), pts_2.len(), "pts_2 contains duplicates");
    }

    #[test]
    #[should_panic(expected = "l must be positive")]
    fn test_points_012_panics_on_zero() {
        let _: [Vec<MultilinearPoint<F>>; 2] = points_012(0);
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
            let (z_svo, z_split) = z.split_at(l);
            let split_eq = SplitEq::<F, EF>::new(&z_split, EF::ONE);
            let partial_evals = split_eq.partial_evals(&f);
            let us = points_012::<F>(l);

            // Test pts_0 (ending in 0).
            let u_evals0 = calculate_accumulators(&us[0], &partial_evals, &z_svo);
            us[0].iter().zip(u_evals0.iter()).for_each(|(u, &e0)| {
                let u = u.iter().copied().map(EF::from).collect::<Vec<_>>();
                let f = f.compress_multi(&u);
                let eq = eq.compress_multi(&u);
                let e1: EF = dot_product(eq.iter().copied(), f.iter().copied());
                assert_eq!(e0, e1);
            });

            // Test pts_2 (ending in 2).
            let u_evals2 = calculate_accumulators(&us[1], &partial_evals, &z_svo);
            us[1].iter().zip(u_evals2.iter()).for_each(|(u, &e0)| {
                let u = u.iter().copied().map(EF::from).collect::<Vec<_>>();
                let f = f.compress_multi(&u);
                let eq = eq.compress_multi(&u);
                let e1: EF = dot_product(eq.iter().copied(), f.iter().copied());
                assert_eq!(e0, e1);
            });
        }
    }

    #[test]
    fn test_split_eq_k_calculation() {
        // Verify that k() returns the original number of variables.
        let mut rng = SmallRng::seed_from_u64(42);

        for k in [8, 10, 12, 14] {
            let point = MultilinearPoint::<EF>::rand(&mut rng, k);
            let split_eq = SplitEq::<F, EF>::new(&point, EF::ONE);
            assert_eq!(split_eq.k(), k, "k() mismatch for k={k}");
        }
    }

    #[test]
    fn test_split_eq_partial_evals_size() {
        // Verify partial_evals returns the correct number of elements.
        let mut rng = SmallRng::seed_from_u64(42);
        let k = 10;
        let n = 14; // Total polynomial variables.

        let point = MultilinearPoint::<EF>::rand(&mut rng, k);
        let split_eq = SplitEq::<F, EF>::new(&point, EF::ONE);

        let poly = EvaluationsList::new((0..1 << n).map(|_| rng.random()).collect());
        let partial = split_eq.partial_evals(&poly);

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
                prop_assert_eq!(*pt.0.last().unwrap(), F::ZERO);
            }
            for pt in &pts_2 {
                prop_assert_eq!(*pt.0.last().unwrap(), F::TWO);
            }
        }

        /// Verify that SplitEq::k() returns the original point dimension.
        #[test]
        fn prop_split_eq_k_consistency(k in 8usize..=14) {
            let mut rng = SmallRng::seed_from_u64(k as u64);
            let point = MultilinearPoint::<EF>::rand(&mut rng, k);
            let split_eq = SplitEq::<F, EF>::new(&point, EF::ONE);

            prop_assert_eq!(split_eq.k(), k);
        }
    }
}
