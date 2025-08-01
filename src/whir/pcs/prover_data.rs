use core::{cmp::Reverse, ops::Range};

use itertools::izip;
use p3_field::Field;
use p3_matrix::{
    Dimensions, Matrix,
    dense::{RowMajorMatrix, RowMajorMatrixView},
    horizontally_truncated::HorizontallyTruncated,
};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_ceil_usize;

use crate::{
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    utils::eval_eq,
    whir::{pcs::query::MlQuery, statement::weights::Weights},
};

/// Metadata for a concatenation of multiple row-major matrices into a single
/// logical multilinear polynomial domain.
#[derive(Debug)]
pub struct ConcatMatsMeta {
    /// Total number of variables (`log_b`) required to index the entire
    /// concatenated evaluation vector.
    ///
    /// This equals `ceil(log2(total_padded_size))`, where each matrix is padded
    /// to a power-of-two width per row.
    pub(crate) log_b: usize,

    /// Original dimensions (height and unpadded width) of each matrix,
    /// in the same order they were passed in.
    ///
    /// This is used to reconstruct the layout, apply constraints, or access
    /// individual submatrices.
    pub(crate) dimensions: Vec<Dimensions>,

    /// Contiguous memory ranges inside the full evaluation vector that
    /// correspond to each input matrix.
    ///
    /// Each range is aligned and padded such that rows are stored with
    /// `width.next_power_of_two()` stride, ensuring compatibility with DFTs.
    /// These ranges are disjoint and span a domain of size `1 << log_b`.
    pub(crate) ranges: Vec<Range<usize>>,
}

impl ConcatMatsMeta {
    /// Construct metadata for a list of matrices to be concatenated into a single
    /// multilinear domain.
    ///
    /// This function computes memory layout information for embedding multiple
    /// row-major matrices into a shared evaluation vector, where each matrix is
    /// padded to ensure rows are power-of-two aligned.
    ///
    /// The function preserves original input order in the `dimensions` and `ranges`
    /// vectors, but internally sorts by padded size to minimize gaps.
    ///
    /// # Parameters
    /// - `dims`: A slice of `Dimensions`, one for each input matrix.
    ///
    /// # Returns
    /// - A new `ConcatMatsMeta` instance, with computed ranges and total size metadata.
    pub(crate) fn new(dims: &[Dimensions]) -> Self {
        // Create an indexed list of dimensions to track original input order.
        let mut indexed_dims: Vec<_> = dims.iter().enumerate().collect();

        // Sort the matrices by padded size (height × next_power_of_two(width)),
        // in descending order. This minimizes fragmentation when laying them out
        // into a contiguous domain.
        indexed_dims
            .sort_unstable_by_key(|(_, dim)| Reverse(dim.width.next_power_of_two() * dim.height));

        // Initialize a vector to store the memory range of each matrix in the
        // final evaluation vector. Each entry will be a `Range<usize>`.
        let mut ranges = vec![0..0; dims.len()];

        // Track the current offset into the flattened domain.
        let mut current_offset = 0;

        // Compute the range for each matrix, placing it back into its original index.
        for (original_index, dim) in indexed_dims {
            // Pad the width to the next power of two to align with DFT constraints.
            let padded_size = dim.width.next_power_of_two() * dim.height;

            // Assign the range for this matrix starting at current offset.
            let start = current_offset;
            let end = start + padded_size;
            ranges[original_index] = start..end;

            // Advance the offset for the next matrix.
            current_offset = end;
        }

        // Compute the total number of variables (`log_b`) required to address the
        // full evaluation vector: log2_ceil(total_size).
        let log_b = log2_ceil_usize(current_offset);

        // Construct and return the metadata with original dimensions and computed ranges.
        Self {
            log_b,
            dimensions: dims.to_vec(),
            ranges,
        }
    }

    /// Return the maximum number of bits required to index any row within the concatenated matrices.
    ///
    /// This computes the maximum `ceil(log2(width))` across all input matrices,
    /// which corresponds to the number of variables needed to index the **columns**
    /// of the widest matrix (after padding).
    ///
    /// # Returns
    /// - The largest `log2_ceil(width)` value among all matrices in the metadata.
    /// - Returns `0` if no matrices are present.
    pub(crate) fn max_log_width(&self) -> usize {
        self.dimensions
            .iter()
            .map(|dim| log2_ceil_usize(dim.width))
            .max()
            .unwrap_or_default()
    }

    /// Construct the constraint weights and expected sum for a given matrix index and query.
    ///
    /// This method supports two types of multilinear queries:
    /// - `MlQuery::Eq(z)`
    ///   Produces a point constraint weight used in multilinear evaluation.
    /// - `MlQuery::EqRotateRight(z, rot)`
    ///   Produces a full evaluation vector as a `Weights::Linear`, aligned with the
    ///   domain layout for the specified matrix.
    ///
    /// Internally, the method constructs an equality polynomial `eq_r` over the
    /// column selector `r`, computes the dot product with the row `ys`, and builds
    /// the appropriate `Weights` object to enforce the constraint.
    ///
    /// # Parameters
    /// - `idx`: Index of the target matrix in the concatenation.
    /// - `query`: The `MlQuery` describing the row selection logic.
    /// - `ys`: The values in the selected row of the matrix.
    /// - `r`: The challenge values used to evaluate the equality polynomial over the column domain.
    ///
    /// # Returns
    /// - A pair `(weights, sum)`:
    ///   - `weights`: The constructed `Weights<F>` used to enforce the constraint.
    ///   - `sum`: The expected value of `dot(ys, eq_r)` to match during verification.
    pub(crate) fn constraint<F: Field>(
        &self,
        idx: usize,
        query: &MlQuery<F>,
        ys: &[F],
        r: &[F],
    ) -> (Weights<F>, F) {
        // Determine how many bits are needed to index columns in this matrix.
        let log_width = log2_ceil_usize(self.dimensions[idx].width);

        // Truncate r to the column selection dimension.
        let r = &r[..log_width];

        // Build the equality polynomial eq_r over the input challenge.
        let mut eq_r = F::zero_vec(1 << r.len());
        eval_eq::<_, _, false>(r, &mut eq_r, F::ONE);

        // Compute the dot product of the selected row and eq_r.
        let sum = ys.iter().zip(&eq_r).map(|(y, e)| *y * *e).sum();

        // Construct the appropriate constraint weights based on the query type.
        let weights = match query {
            MlQuery::Eq(z) => {
                // Construct the full evaluation point for this query:
                //   - r: column selector
                //   - z: row selector
                //   - additional bits: identify the matrix index within the domain
                //
                // We assemble the point in order of significance: offset -> row -> column.
                let point = {
                    // 1. Offset bits (most significant), generated in big-endian order.
                    let num_non_offset_vars = z.len() + r.len();
                    let num_offset_vars = self.log_b - num_non_offset_vars;

                    let offset_vars = (0..num_offset_vars).map(|i| {
                        // To get the k-th MSB of the offset, we need to shift by (log_b - 1 - k)
                        let bit_pos = self.log_b - 1 - i;
                        F::from_bool((self.ranges[idx].start >> bit_pos) & 1 == 1)
                    });

                    // 2. Chain with row bits `z` and column bits `r`.
                    offset_vars
                        .chain(z.iter().copied())
                        .chain(r.iter().copied())
                        .collect()
                };

                Weights::evaluation(MultilinearPoint(point))
            }

            MlQuery::EqRotateRight(_, _) => {
                // Evaluate the rotated query as a multilinear polynomial.
                let query_evals = query.to_mle(F::ONE);

                // If parallel feature is not enables, this will falls back
                // to into_iter().
                let range_iter = self.ranges[idx].clone().into_par_iter();

                // Compute the constraint weights for each cell in the matrix.
                //
                // The evaluation domain is stored flat, but conceptually each index `i` corresponds
                // to a matrix cell in row-major order.
                //
                // We identify the logical `(row, col)` pair inside the matrix by offsetting
                // and decomposing the index:
                //   - `local_i = i - start` gives the offset inside the matrix
                //   - `row = local_i / width_padded`, `col = local_i % width_padded`
                // Then we compute the value: `query_evals[row] * eq_r[col]`
                // This represents the constraint weight for that cell.
                let weight_values: Vec<_> = range_iter
                    .map(|i| {
                        let local_i = i - self.ranges[idx].start;
                        let row = local_i / eq_r.len();
                        let col = local_i % eq_r.len();
                        query_evals[row] * eq_r[col]
                    })
                    .collect();

                // Allocate the full domain with zeros, and place the computed weights into the appropriate range.
                //
                // This ensures the result is shaped for a full multilinear polynomial domain (size = 2^log_b),
                // while only the subrange corresponding to the matrix has non-zero entries.
                let mut final_weights = F::zero_vec(1 << self.log_b);
                final_weights[self.ranges[idx].clone()].copy_from_slice(&weight_values);

                // Return the computed weights as a full multilinear evaluation list.
                Weights::linear(EvaluationsList::new(final_weights))
            }
        };

        (weights, sum)
    }
}

/// A structure that flattens and concatenates multiple row-major matrices
/// into a single evaluation vector compatible with multilinear operations.
///
/// Each input matrix is padded along the row width to the next power of two,
/// and its rows are laid out contiguously. The metadata tracks how to recover
/// each original matrix view from the global layout.
///
/// This is useful in protocols where multiple matrix domains must be unified
/// into a single multilinear domain for evaluation or constraint application.
#[derive(Debug)]
pub struct ConcatMats<Val> {
    /// A single flattened evaluation vector of all input matrices.
    ///
    /// Each matrix is stored row-by-row with rows padded to the next power-of-two width,
    /// and the concatenated layout spans a total domain of size `2^log_b`.
    pub(crate) values: Vec<Val>,

    /// Metadata describing the logical layout of the concatenated matrices.
    ///
    /// Includes dimension, range, and variable information for each original matrix
    /// within the unified domain.
    pub(crate) meta: ConcatMatsMeta,
}

impl<Val: Field> ConcatMats<Val> {
    /// Construct a new `ConcatMats` from a list of row-major matrices.
    ///
    /// Internally, this flattens and pads each matrix row to align with
    /// power-of-two DFT-compatible widths, and lays them out contiguously in memory.
    ///
    /// # Parameters
    /// - `mats`: A list of row-major matrices to concatenate.
    ///
    /// # Returns
    /// - A `ConcatMats` instance containing the unified evaluation vector and metadata.
    pub(crate) fn new(mats: Vec<RowMajorMatrix<Val>>) -> Self {
        let meta = ConcatMatsMeta::new(&mats.iter().map(Matrix::dimensions).collect::<Vec<_>>());
        let mut values = Val::zero_vec(1 << meta.log_b);

        izip!(&meta.ranges, mats).for_each(|(range, mat)| {
            // Parallel row-wise copying into padded storage
            values[range.clone()]
                .par_chunks_mut(mat.width().next_power_of_two())
                .zip(mat.par_row_slices())
                .for_each(|(dst, src)| dst[..src.len()].copy_from_slice(src));
        });

        Self { values, meta }
    }

    /// Return a view of the original matrix at the given index.
    ///
    /// This reconstructs the logical matrix layout (with original width)
    /// from the padded and concatenated evaluation vector using metadata.
    ///
    /// # Parameters
    /// - `idx`: The index of the original matrix in the input list.
    ///
    /// # Returns
    /// - A `HorizontallyTruncated<RowMajorMatrixView>` that presents a view
    ///   into the padded storage but with the original width restored.
    pub(crate) fn mat(
        &self,
        idx: usize,
    ) -> HorizontallyTruncated<Val, RowMajorMatrixView<'_, Val>> {
        HorizontallyTruncated::new(
            RowMajorMatrixView::new(
                &self.values[self.meta.ranges[idx].clone()],
                self.meta.dimensions[idx].width.next_power_of_two(),
            ),
            self.meta.dimensions[idx].width,
        )
        .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use itertools::chain;
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_util::log2_strict_usize;

    use super::*;
    use crate::whir::pcs::query::MlQuery;

    type F = BabyBear;

    #[test]
    fn test_new_meta_single_matrix() {
        // Test creation with a single matrix of 3x2 (height = 3, width = 2)
        let dims = vec![Dimensions {
            height: 3,
            width: 2,
        }];
        let meta = ConcatMatsMeta::new(&dims);

        // height * width = 3 * 2 = 6
        assert_eq!(meta.ranges, vec![0..6]);
        // ceil(log2(6)) = 3
        assert_eq!(meta.log_b, 3);
        // Original dimensions should be preserved
        assert_eq!(meta.dimensions, dims);
    }

    #[test]
    fn test_new_meta_multiple_matrices_ordering_and_ranges() {
        // Matrix 0: 3x2 = 6
        // Matrix 1: 1x4 = 4
        let dims = vec![
            Dimensions {
                height: 3,
                width: 2,
            },
            Dimensions {
                height: 1,
                width: 4,
            },
        ];
        let meta = ConcatMatsMeta::new(&dims);

        // Check log_b = ceil(log2(12)) = 4
        assert_eq!(meta.log_b, 4);

        // Check that ranges are padded and disjoint
        assert_eq!(meta.ranges, vec![0..6, 6..10]);

        // Dimensions preserved in order
        assert_eq!(meta.dimensions, dims);
    }

    #[test]
    fn test_new_meta_three_matrices_varying_sizes() {
        let dims = vec![
            Dimensions {
                height: 2,
                width: 3,
            }, // size = 2 * 4 = 8 (padded)
            Dimensions {
                height: 4,
                width: 2,
            }, // size = 4 * 2 = 8
            Dimensions {
                height: 1,
                width: 5,
            }, // size = 1 * 8 = 8
        ];
        let meta = ConcatMatsMeta::new(&dims);

        // All three matrices are padded to 8 elements
        // So total size = 24, and log_b = ceil(log2(24)) = 5
        assert_eq!(meta.log_b, 5);

        // Ranges must be disjoint and cover [0..24]
        let all_indices: Vec<usize> = meta.ranges.iter().flat_map(Clone::clone).collect();
        assert_eq!(all_indices.len(), 24);
        let mut sorted = all_indices;
        sorted.sort_unstable();
        assert_eq!(sorted, (0..24).collect::<Vec<_>>());

        // Dimensions must preserve input order
        assert_eq!(meta.dimensions, dims);

        // Each range should be 8 elements long
        assert!(meta.ranges.iter().all(|r| r.len() == 8));
    }

    #[test]
    fn test_new_meta_power_of_two_aligned_sizes() {
        let dims = vec![
            Dimensions {
                height: 4,
                width: 4,
            }, // size = 4 * 4 = 16
            Dimensions {
                height: 2,
                width: 8,
            }, // size = 2 * 8 = 16
            Dimensions {
                height: 1,
                width: 16,
            }, // size = 1 * 16 = 16
        ];
        let meta = ConcatMatsMeta::new(&dims);

        // All matrices are already aligned to power-of-two widths
        // Total = 16 * 3 = 48 -> log_b = ceil(log2(48)) = 6
        assert_eq!(meta.log_b, 6);

        assert_eq!(meta.ranges.len(), 3);
        assert!(meta.ranges.iter().all(|r| r.len() == 16));
        assert_eq!(meta.ranges[0], 0..16);
        assert_eq!(meta.ranges[1], 16..32);
        assert_eq!(meta.ranges[2], 32..48);
    }

    #[test]
    fn test_new_meta_identical_matrix_sizes_preserves_order() {
        let dims = vec![
            Dimensions {
                height: 2,
                width: 2,
            }, // size = 2 * 2 = 4 → padded to 2*2 = 4
            Dimensions {
                height: 2,
                width: 2,
            },
            Dimensions {
                height: 2,
                width: 2,
            },
        ];
        let meta = ConcatMatsMeta::new(&dims);

        // Total = 3 * 4 = 12 → log2(12) = 4
        assert_eq!(meta.log_b, 4);

        // All ranges padded to 4 elements
        assert!(meta.ranges.iter().all(|r| r.len() == 4));

        // Check that ranges are disjoint and contiguous
        assert_eq!(meta.ranges[0], 0..4);
        assert_eq!(meta.ranges[1], 4..8);
        assert_eq!(meta.ranges[2], 8..12);

        // Input order preserved
        assert_eq!(meta.dimensions, dims);
    }

    #[test]
    fn test_new_meta_stress_large_matrix_list() {
        let dims: Vec<Dimensions> = (0..20)
            .map(|i| Dimensions {
                height: (i % 5) + 1,
                width: (i % 7) + 1,
            })
            .collect();
        let meta = ConcatMatsMeta::new(&dims);

        // Total allocated space should match the sum of padded sizes
        let expected_total: usize = dims
            .iter()
            .map(|dim| dim.height * dim.width.next_power_of_two())
            .sum();

        let actual_total: usize = meta
            .ranges
            .iter()
            .map(std::iter::ExactSizeIterator::len)
            .sum();
        assert_eq!(actual_total, expected_total);

        // Ranges should be disjoint and sorted in memory order
        let all_indices: Vec<usize> = meta.ranges.iter().flat_map(Clone::clone).collect();
        let mut sorted = all_indices;
        sorted.sort_unstable();
        assert_eq!(sorted, (0..expected_total).collect::<Vec<_>>());

        // log_b should be log2_ceil(expected_total)
        let expected_log_b = log2_ceil_usize(expected_total);
        assert_eq!(meta.log_b, expected_log_b);
    }

    #[test]
    fn test_max_log_width() {
        // Define two matrices with different widths
        let dims = vec![
            Dimensions {
                height: 2,
                width: 3,
            }, // log2_ceil(3) = 2
            Dimensions {
                height: 2,
                width: 5,
            }, // log2_ceil(5) = 3
        ];
        let meta = ConcatMatsMeta::new(&dims);

        // The maximum log2_ceil(width) should be 3
        assert_eq!(meta.max_log_width(), 3);
    }

    #[test]
    fn test_constraint_eq_point() {
        let dims = vec![Dimensions {
            height: 1,
            width: 2,
        }];
        let meta = ConcatMatsMeta::new(&dims);

        // Query selects row: since height = 1, z = [].
        let query = MlQuery::Eq(MultilinearPoint::default());

        // Row of values: y = [7, 9]
        let y0 = F::from_u8(7);
        let y1 = F::from_u8(9);
        let ys = vec![y0, y1];

        // r selects column. We only need the first value since width = 2.
        let r0 = F::from_u8(2);
        let r = vec![r0];

        // Manually compute eq_r
        // r = [2], so we're evaluating eq_{[2]}(x) at x in {0, 1}
        //
        // For x = 0: eq(x) = 1 - r[0]
        // For x = 1: eq(x) = r[0]
        let eq_r0 = F::ONE - r[0];
        let eq_r1 = r[0];

        // Manually compute dot product y ⋅ eq_r
        let expected_sum = y0 * eq_r0 + y1 * eq_r1;

        let (weights, sum) = meta.constraint(0, &query, &ys, &r);
        assert_eq!(sum, expected_sum);

        // Manually compute weight point
        // point = rev(chain![r_sliced, z, range_bits])
        //
        // - r_sliced = [2]
        // - z = []
        // - range_bits = []
        //
        // So point = [2]
        match weights {
            Weights::Evaluation { point } => {
                assert_eq!(point.0, vec![r0]);
            }
            Weights::Linear { .. } => panic!("Expected Weights::Evaluation"),
        }
    }

    #[test]
    fn test_constraint_eq_rotate_right() {
        // Matrix: width = 2, height = 2. This makes log_width = 1.
        let mat = RowMajorMatrix::new(
            vec![F::from_u8(5), F::from_u8(1), F::from_u8(6), F::from_u8(2)],
            2, // width
        );
        let concat = ConcatMats::new(vec![mat]);
        let meta = &concat.meta;

        // Rotated query.
        let query = MlQuery::EqRotateRight(MultilinearPoint(vec![F::from_u8(7)]), 1);

        let y0 = F::from_u8(9);
        let y1 = F::from_u8(10);
        let ys = vec![y0, y1];

        let r0 = F::from_u8(3);
        let r = vec![r0];

        let (weights, sum) = meta.constraint(0, &query, &ys, &r);

        // eq_r for r=[3] is [1-3, 3] -> [-2, 3].
        // dot(ys, eq_r) = 9 * (-2) + 10 * 3 = -18 + 30 = 12.
        assert_eq!(sum, F::from_u8(12));

        match weights {
            Weights::Linear { weight } => {
                let full: &[F] = weight.as_ref();
                assert_eq!(full.len(), 1 << meta.log_b);

                // query_mle for z=[7] produces [7, 2013265915].
                // eq_r for r=[3] is [-2, 3].
                let expected_weights = vec![
                    // Row 1 weights appear first in memory
                    F::from_u8(7) * F::from_i32(-2), // col 0: 7 * -2  = -14
                    F::from_u8(7) * F::from_u8(3),   // col 1: 7 * 3   = 21
                    // Row 0 weights appear second in memory
                    F::from_i32(2_013_265_915) * F::from_i32(-2), // col 0: -6 * -2 = 12
                    F::from_i32(2_013_265_915) * F::from_u8(3),   // col 1: -6 * 3  = -18
                ];

                assert_eq!(
                    full, expected_weights,
                    "The calculated weights do not match the expected order and values."
                );
            }
            Weights::Evaluation { .. } => panic!("Expected Weights::Linear"),
        }
    }

    #[test]
    fn test_constraint_eq_multiple_matrices_large_width() {
        // Matrix 0: 1x4, Matrix 1: 2x3
        let mats = vec![
            RowMajorMatrix::new(
                vec![F::from_u8(1), F::from_u8(2), F::from_u8(3), F::from_u8(4)],
                4,
            ),
            RowMajorMatrix::new(
                vec![
                    F::from_u8(5),
                    F::from_u8(6),
                    F::from_u8(7), // row 0
                    F::from_u8(8),
                    F::from_u8(9),
                    F::from_u8(10), // row 1
                ],
                3,
            ),
        ];
        let concat = ConcatMats::new(mats);
        let meta = &concat.meta;

        // Index 1 (second matrix): shape 2x3
        // log2_ceil(3) = 2
        let query = MlQuery::Eq(MultilinearPoint(vec![F::from_u8(0)])); // z selects row 0
        let ys = vec![F::from_u8(5), F::from_u8(6), F::from_u8(7)];
        let r = vec![F::from_u8(1), F::from_u8(2), F::from_u8(9)]; // extra bits; truncated

        let (weights, sum) = meta.constraint(1, &query, &ys, &r);

        // eq_r = [(1 - r0)*(1 - r1), (1 - r0)*r1, r0*(1 - r1), r0*r1]
        let r0 = r[0];
        let r1 = r[1];
        let eq_r = [
            (F::ONE - r0) * (F::ONE - r1),
            (F::ONE - r0) * r1,
            r0 * (F::ONE - r1),
            r0 * r1,
        ];

        // Since matrix has width 3, we truncate eq_r to 3
        let expected_sum = ys[0] * eq_r[0] + ys[1] * eq_r[1] + ys[2] * eq_r[2];
        assert_eq!(sum, expected_sum);

        match weights {
            Weights::Evaluation { point } => {
                // Should contain rev([r0, r1] + z + range bits)
                let bits = (log2_strict_usize(meta.ranges[1].len())..meta.log_b)
                    .map(|i| F::from_bool((meta.ranges[1].start >> i) & 1 == 1));
                let expected =
                    chain![bits, vec![F::from_u8(0)], r[..2].iter().copied()].collect::<Vec<_>>();
                assert_eq!(point.0, expected);
            }
            Weights::Linear { .. } => panic!("Expected Weights::Evaluation"),
        }
    }

    #[test]
    fn test_constraint_eq_rotate_right_high_dim() {
        // -----------------------------------------------
        // Matrix M: a 4×4 matrix, flattened row-wise
        //
        // M = [
        //   [ 1,  2,  3,  4 ],   // row 0
        //   [ 5,  6,  7,  8 ],   // row 1
        //   [ 9, 10, 11, 12 ],   // row 2
        //   [13, 14, 15, 16 ],   // row 3
        // ]
        //
        // This corresponds to a function f(x0, x1, x2, x3) defined over {0,1}^4.
        // Each row is indexed by x2, x3 and each column by x0, x1.
        //
        // f(0,0,0,0) = 1
        // f(0,0,0,1) = 2
        // f(0,0,1,0) = 3
        // f(0,0,1,1) = 4
        // f(0,1,0,0) = 5
        // f(0,1,0,1) = 6
        // f(0,1,1,0) = 7
        // f(0,1,1,1) = 8
        // f(1,0,0,0) = 9
        // f(1,0,0,1) = 10
        // ...
        // -----------------------------------------------
        let mat = RowMajorMatrix::new(
            vec![
                1, 2, 3, 4, // row 0: (x2,x3) = (0,0)
                5, 6, 7, 8, // row 1: (x2,x3) = (0,1)
                9, 10, 11, 12, // row 2: (x2,x3) = (1,0)
                13, 14, 15, 16, // row 3: (x2,x3) = (1,1)
            ]
            .into_iter()
            .map(F::from_u8)
            .collect(),
            4, // width (number of columns)
        );

        // Pack the matrix into a unified multilinear domain
        let concat = ConcatMats::new(vec![mat]);
        let meta = &concat.meta;

        // -----------------------------------------------
        // Query: EqRotateRight(z = [1, 0], rotation = 1)
        //
        // This rotates z = [1,0] right by 1 bit → becomes [0,1]
        //
        // The rotated z is treated as a Boolean point representing a row index.
        // So this selects the row (0,1), i.e., row index 1.
        // -----------------------------------------------
        let query = MlQuery::EqRotateRight(MultilinearPoint(vec![F::from_u8(1), F::from_u8(0)]), 1);

        // ys: values associated with each row in the matrix.
        //
        // These are arbitrary and not derived from the matrix.
        // For example, suppose the prover computed a polynomial g(x2,x3)
        // and evaluated it on all 4 rows (x2,x3) ∈ {0,1}^2:
        //
        // g(0,0) = 21
        // g(0,1) = 22
        // g(1,0) = 23
        // g(1,1) = 24
        //
        // So:
        // ys = [g(0,0), g(0,1), g(1,0), g(1,1)]
        // -----------------------------------------------
        let ys = vec![
            F::from_u8(21),
            F::from_u8(22),
            F::from_u8(23),
            F::from_u8(24),
        ];

        // -----------------------------------------------
        // r: the column selector challenge.
        // It selects a column via the equality polynomial eq_r
        //
        // Since the width is 4, we need 2 bits to index a column:
        // → x0, x1 index the column domain.
        // -----------------------------------------------
        let r = vec![F::from_u8(2), F::from_u8(1)];

        // Compute the constraint using meta (on matrix 0)
        let (weights, sum) = meta.constraint(0, &query, &ys, &r);

        // ---------------------------------------------------------
        // Naively compute eq_r(x) for all x ∈ {0,1}^2, where:
        // eq_r(x) = ∏_i ((1 - r_i)(1 - x_i) + r_i x_i)
        // ---------------------------------------------------------
        let eq_r = {
            let r0 = r[0];
            let r1 = r[1];
            let mut values = Vec::with_capacity(4);
            for &x0 in &[F::ZERO, F::ONE] {
                for &x1 in &[F::ZERO, F::ONE] {
                    let term0 = (F::ONE - r0) * (F::ONE - x0) + r0 * x0;
                    let term1 = (F::ONE - r1) * (F::ONE - x1) + r1 * x1;
                    values.push(term0 * term1);
                }
            }
            values
        };

        // Compute expected sum:
        //
        // ∑_row ys[row] * eq_r[column selected]
        // -----------------------------------------------
        let expected_sum = ys[0] * eq_r[0] + ys[1] * eq_r[1] + ys[2] * eq_r[2] + ys[3] * eq_r[3];
        assert_eq!(sum, expected_sum);

        match weights {
            Weights::Linear { weight } => {
                let full = weight;
                assert_eq!(full.len(), 1 << meta.log_b);

                // Pull out the weight range corresponding to matrix 0
                let expected_weight_range = &full[meta.ranges[0].clone()];

                // mle = query.to_mle(F::ONE)
                // This gives evaluations of query polynomial on each row (over z-variables)
                //
                // Since z = [1,0], rot = 1 → rotated z = [0,1]
                let mle = query.to_mle(F::ONE);

                // The final weight matrix W is constructed as:
                //
                // W[i][j] = mle[i] * eq_r[j]
                //
                // Each row corresponds to a row in the matrix
                // Each column corresponds to a column selected by r
                //
                // So expected_weight_range should equal:
                //
                // Row 0: mle[0] * eq_r[0], ..., mle[0] * eq_r[3]
                // Row 1: mle[1] * eq_r[0], ..., mle[1] * eq_r[3]
                // Row 2: mle[2] * eq_r[0], ..., mle[2] * eq_r[3]
                // Row 3: mle[3] * eq_r[0], ..., mle[3] * eq_r[3]
                //
                // And this is laid out row-major
                for (row, chunk) in expected_weight_range.chunks(ys.len()).enumerate() {
                    for col in 0..ys.len() {
                        let expected = eq_r[col] * mle[row];
                        assert_eq!(chunk[col], expected);
                    }
                }
            }
            Weights::Evaluation { .. } => panic!("Expected Weights::Linear"),
        }
    }

    #[test]
    fn test_constraint_eq_high_dim_no_rotate_simple() {
        // -----------------------------------------------------
        // Construct a 4×4 matrix, flattened row-wise.
        //
        // Matrix layout (row-major):
        // [
        //   [  1,  2,  3,  4 ],    // row 0
        //   [  5,  6,  7,  8 ],    // row 1
        //   [  9, 10, 11, 12 ],    // row 2
        //   [ 13, 14, 15, 16 ],    // row 3
        // ]
        //
        // We treat this matrix as evaluations of a 4-variable multilinear
        // polynomial f(x_0,x_1,x_2,x_3) over {0,1}^4, where:
        //   - (x_0,x_1) index the rows
        //   - (x_2,x_3) index the columns
        // -----------------------------------------------------
        let mat_data = (1u8..=16).map(F::from_u8).collect::<Vec<_>>();
        let mat = RowMajorMatrix::new(mat_data.clone(), 4); // width = 4

        let concat = ConcatMats::new(vec![mat]);
        let meta = &concat.meta;

        // -----------------------------------------------------
        // Define an equality constraint query over rows.
        //
        // z = (0, 1) selects the row with bits (x_2, x_3) = (0, 1)
        // This corresponds to row index = 1 (second row):
        //     [5, 6, 7, 8]
        //
        // Note: meta.log_b = 4 because total input dim is 4 bits.
        // -----------------------------------------------------
        let query = MlQuery::Eq(MultilinearPoint(vec![F::ZERO, F::ONE]));

        // -----------------------------------------------------
        // Select a random challenge r = (1, 2) ∉ {0,1}^2
        //
        // This will be used to build eq_r(x_0, x_1), a multilinear
        // equality polynomial that "selects" a virtual column.
        // -----------------------------------------------------
        let r = vec![F::ONE, F::ZERO];

        // -----------------------------------------------------
        // Extract the corresponding row values
        // ys = g(0,1,⋅,⋅) = [5, 6, 7, 8]
        //
        // These values will be dot-multiplied against eq_r(x)
        // to produce the expected sum.
        // -----------------------------------------------------
        let ys = mat_data[4..8].to_vec(); // row 1

        // -----------------------------------------------------
        // Call the constraint system to compute:
        // - weights: either Evaluation or Linear variant
        // - sum: the weighted sum over the selected row
        // -----------------------------------------------------
        let (weights, sum) = meta.constraint(0, &query, &ys, &r);

        // -----------------------------------------------------
        // The index corresponds to:
        //   - row : (0, 1) -> row index = 1
        //   - col : (1, 0) -> column index = 2
        //
        // f(0,1,1,0) = 7 (from the matrix)
        // -----------------------------------------------------
        assert_eq!(
            sum,
            F::from_u64(7),
            "Sum from constraint does not match expected"
        );

        // -----------------------------------------------------
        // Step 8: Evaluate the full matrix at the returned point
        //
        // When the system returns Weights::Evaluation, it means
        // the sum can be viewed as f(point), where f is the full
        // multilinear extension of the matrix.
        //
        // We check that f(0,1,1,0) = 7
        // -----------------------------------------------------
        match weights {
            Weights::Evaluation { point } => {
                let eval_list = EvaluationsList::new(concat.values.clone());
                let evaluation_at_point = eval_list.evaluate(&point);
                assert_eq!(
                    evaluation_at_point,
                    F::from_u64(7),
                    "Multilinear evaluation at point doesn't match expected"
                );
            }
            Weights::Linear { .. } => panic!("Expected Weights::Evaluation from Eq query"),
        }
    }

    #[test]
    fn test_constraint_eq_high_dim_no_rotate() {
        // -----------------------------------------------------
        // Construct a 4×4 matrix, flattened row-wise.
        //
        // Matrix layout (row-major):
        // [
        //   [  1,  2,  3,  4 ],    // row 0
        //   [  5,  6,  7,  8 ],    // row 1
        //   [  9, 10, 11, 12 ],    // row 2
        //   [ 13, 14, 15, 16 ],    // row 3
        // ]
        //
        // We treat this matrix as evaluations of a 4-variable multilinear
        // polynomial f(x_0,x_1,x_2,x_3) over {0,1}^4, where:
        //   - (x_0,x_1) index the rows
        //   - (x_2,x_3) index the columns
        // -----------------------------------------------------
        let mat_data = (1u8..=16).map(F::from_u8).collect::<Vec<_>>();
        let mat = RowMajorMatrix::new(mat_data.clone(), 4); // width = 4

        // To complexify things, we'll also construct a second matrix
        let mat_data1 = (1u8..=16).map(F::from_u8).collect::<Vec<_>>();
        let mat1 = RowMajorMatrix::new(mat_data1, 4); // width = 4

        let concat = ConcatMats::new(vec![mat, mat1]);
        let meta = &concat.meta;

        // -----------------------------------------------------
        // Define an equality constraint query over rows.
        //
        // z = (0, 1) selects the row with bits (x_2, x_3) = (0, 1)
        // This corresponds to row index = 1 (second row):
        //     [5, 6, 7, 8]
        //
        // Note: meta.log_b = 4 because total input dim is 4 bits.
        // -----------------------------------------------------
        let query = MlQuery::Eq(MultilinearPoint(vec![F::ZERO, F::ONE]));

        // -----------------------------------------------------
        // Select a random challenge r = (1, 2) ∉ {0,1}^2
        //
        // This will be used to build eq_r(x_0, x_1), a multilinear
        // equality polynomial that "selects" a virtual column.
        // -----------------------------------------------------
        let r = vec![F::from_u8(1), F::from_u8(2)];

        // -----------------------------------------------------
        // Extract the corresponding row values
        // ys = g(0,1,⋅,⋅) = [5, 6, 7, 8]
        //
        // These values will be dot-multiplied against eq_r(x)
        // to produce the expected sum.
        // -----------------------------------------------------
        let ys = mat_data[4..8].to_vec(); // row 1

        // -----------------------------------------------------
        // Call the constraint system to compute:
        // - weights: either Evaluation or Linear variant
        // - sum: the weighted sum over the selected row
        // -----------------------------------------------------
        let (weights, sum) = meta.constraint(0, &query, &ys, &r);

        // -----------------------------------------------------
        // Manually evaluate eq_r(x) on all x ∈ {0,1}^2
        //
        // eq_r(x) = ∏_i ((1 - r_i)(1 - x_i) + r_i x_i)
        //
        // This gives 4 values, one per Boolean point (x_0, x_1):
        //   - (0, 0)
        //   - (0, 1)
        //   - (1, 0)
        //   - (1, 1)
        // -----------------------------------------------------
        let eq_r = {
            let r0 = r[0];
            let r1 = r[1];
            let mut values = Vec::with_capacity(4);
            for &x0 in &[F::ZERO, F::ONE] {
                for &x1 in &[F::ZERO, F::ONE] {
                    let term0 = (F::ONE - r0) * (F::ONE - x0) + r0 * x0;
                    let term1 = (F::ONE - r1) * (F::ONE - x1) + r1 * x1;
                    values.push(term0 * term1);
                }
            }
            values
        };

        // -----------------------------------------------------
        // Step 7: Manually compute expected sum:
        //
        // expected_sum = ∑_{j} ys[j] * eq_r[j]
        //
        // where j ∈ {0, 1, 2, 3} are the column indices.
        // -----------------------------------------------------
        let expected_sum = ys
            .iter()
            .zip(&eq_r)
            .map(|(y, w)| *y * *w)
            .fold(F::ZERO, |acc, val| acc + val);

        assert_eq!(
            sum, expected_sum,
            "Sum from constraint does not match expected"
        );

        // -----------------------------------------------------
        // Step 8: Evaluate the full matrix at the returned point
        //
        // When the system returns Weights::Evaluation, it means
        // the sum can be viewed as f(point), where f is the full
        // multilinear extension of the matrix.
        //
        // We check that f(point) = expected_sum
        // -----------------------------------------------------
        match weights {
            Weights::Evaluation { point } => {
                let eval_list = EvaluationsList::new(concat.values.clone());
                let evaluation_at_point = eval_list.evaluate(&point);
                assert_eq!(
                    evaluation_at_point, sum,
                    "Multilinear evaluation at point doesn't match expected"
                );
            }
            Weights::Linear { .. } => panic!("Expected Weights::Evaluation from Eq query"),
        }
    }

    #[test]
    fn test_concat_mats_new_roundtrip_matrix_extraction() {
        let mat0 = RowMajorMatrix::new(vec![F::from_u8(1), F::from_u8(2), F::from_u8(3)], 3);
        let mat1 = RowMajorMatrix::new(vec![F::from_u8(4), F::from_u8(5)], 2);

        let concat = ConcatMats::new(vec![mat0.clone(), mat1.clone()]);

        // Compare mat0 rows
        let m0_view = concat.mat(0);
        for i in 0..mat0.height() {
            let expected = mat0.row_slice(i).unwrap();
            let actual = m0_view.row_slice(i).unwrap();
            assert_eq!(&*actual, &*expected);
        }

        // Compare mat1 rows
        let m1_view = concat.mat(1);
        for i in 0..mat1.height() {
            let expected = mat1.row_slice(i).unwrap();
            let actual = m1_view.row_slice(i).unwrap();
            assert_eq!(&*actual, &*expected);
        }
    }

    #[test]
    fn test_concat_mats_single_matrix_power_of_two_width() {
        // Create a 2x4 matrix (width is already a power of two)
        // Row 0: [10, 11, 12, 13]
        // Row 1: [14, 15, 16, 17]
        let mat = RowMajorMatrix::new(
            vec![
                F::from_u8(10),
                F::from_u8(11),
                F::from_u8(12),
                F::from_u8(13), // [10, 11, 12, 13]
                F::from_u8(14),
                F::from_u8(15),
                F::from_u8(16),
                F::from_u8(17), // [14, 15, 16, 17]
            ],
            4,
        );

        // Wrap in ConcatMats
        let concat = ConcatMats::new(vec![mat.clone()]);

        // Extract view of the single matrix
        let view = concat.mat(0);

        // Each row should match exactly
        for i in 0..mat.height() {
            let expected = mat.row_slice(i).unwrap();
            let actual = view.row_slice(i).unwrap();
            assert_eq!(&*actual, &*expected);
        }
    }

    #[test]
    fn test_concat_mats_multiple_matrices_different_heights() {
        // Matrix 0: 1x3 → padded width = 4
        let mat0 = RowMajorMatrix::new(vec![F::from_u8(1), F::from_u8(2), F::from_u8(3)], 3);

        // Matrix 1: 3x2 → padded width = 2
        let mat1 = RowMajorMatrix::new(
            vec![
                F::from_u8(4),
                F::from_u8(5),
                F::from_u8(6),
                F::from_u8(7),
                F::from_u8(8),
                F::from_u8(9),
            ],
            2,
        );

        let concat = ConcatMats::new(vec![mat0.clone(), mat1.clone()]);

        // Check matrix 0
        let view0 = concat.mat(0);
        for i in 0..mat0.height() {
            let expected = mat0.row_slice(i).unwrap();
            let actual = view0.row_slice(i).unwrap();
            assert_eq!(&*actual, &*expected);
        }

        // Check matrix 1
        let view1 = concat.mat(1);
        for i in 0..mat1.height() {
            let expected = mat1.row_slice(i).unwrap();
            let actual = view1.row_slice(i).unwrap();
            assert_eq!(&*actual, &*expected);
        }
    }

    #[test]
    fn test_concat_mats_matrix_with_padding_width_5() {
        // Matrix with width = 5 → padded to 8 per row
        let mat = RowMajorMatrix::new(
            vec![
                F::from_u8(1),
                F::from_u8(2),
                F::from_u8(3),
                F::from_u8(4),
                F::from_u8(5),
                F::from_u8(6),
                F::from_u8(7),
                F::from_u8(8),
                F::from_u8(9),
                F::from_u8(10),
            ],
            5,
        ); // height = 2

        let concat = ConcatMats::new(vec![mat.clone()]);
        let view = concat.mat(0);

        // Even though the row is padded internally, `.mat()` view should truncate it back to width = 5
        assert_eq!(view.width(), 5);
        assert_eq!(view.height(), 2);

        // Check row-wise equality
        for i in 0..mat.height() {
            let expected = mat.row_slice(i).unwrap();
            let actual = view.row_slice(i).unwrap();
            assert_eq!(&*actual, &*expected);
        }
    }
}
