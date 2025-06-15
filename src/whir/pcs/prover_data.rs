use core::{cmp::Reverse, ops::Range};

use itertools::izip;
use p3_field::Field;
use p3_matrix::{
    Dimensions, Matrix,
    dense::{RowMajorMatrix, RowMajorMatrixView},
    horizontally_truncated::HorizontallyTruncated,
};
use p3_util::{log2_ceil_usize, log2_strict_usize};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

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
    log_b: usize,

    /// Original dimensions (height and unpadded width) of each matrix,
    /// in the same order they were passed in.
    ///
    /// This is used to reconstruct the layout, apply constraints, or access
    /// individual submatrices.
    dimensions: Vec<Dimensions>,

    /// Contiguous memory ranges inside the full evaluation vector that
    /// correspond to each input matrix.
    ///
    /// Each range is aligned and padded such that rows are stored with
    /// `width.next_power_of_two()` stride, ensuring compatibility with DFTs.
    /// These ranges are disjoint and span a domain of size `1 << log_b`.
    ranges: Vec<Range<usize>>,
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
    fn new(dims: &[Dimensions]) -> Self {
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
    fn max_log_width(&self) -> usize {
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
    ///   - `weights`: The constructed `Weights<Challenge>` used to enforce the constraint.
    ///   - `sum`: The expected value of `dot(ys, eq_r)` to match during verification.
    fn constraint<Challenge: Field>(
        &self,
        idx: usize,
        query: &MlQuery<Challenge>,
        ys: &[Challenge],
        r: &[Challenge],
    ) -> (Weights<Challenge>, Challenge) {
        // Determine how many bits are needed to index columns in this matrix.
        let log_width = log2_ceil_usize(self.dimensions[idx].width);

        // Truncate r to the column selection dimension.
        let r = &r[..log_width];

        // Build the equality polynomial eq_r over the input challenge.
        let mut eq_r = vec![Challenge::ZERO; 1 << r.len()];
        eval_eq::<_, _, false>(r, &mut eq_r, Challenge::ONE);

        // Compute the dot product of the selected row and eq_r.
        let sum = ys.iter().zip(&eq_r).map(|(y, e)| *y * *e).sum();

        // Construct the appropriate constraint weights based on the query type.
        let weights = match query {
            MlQuery::Eq(z) => {
                // Construct the full evaluation point for this query:
                //   - r: column selector (low bits)
                //   - z: row selector
                //   - additional bits: identify the matrix index within the domain
                let point = r
                    .iter()
                    .chain(z)
                    .copied()
                    .chain(
                        (log2_strict_usize(self.ranges[idx].len())..self.log_b)
                            .map(|i| Challenge::from_bool((self.ranges[idx].start >> i) & 1 == 1)),
                    )
                    .rev() // Reverse to match multilinear encoding order
                    .collect();
                Weights::evaluation(MultilinearPoint(point))
            }

            MlQuery::EqRotateRight(_, _) => {
                // Evaluate the rotated query as a multilinear polynomial.
                let query_evals = query.to_mle(Challenge::ONE);

                // Choose the appropriate iterator depending on the build configuration.
                #[cfg(feature = "parallel")]
                let range_iter = self.ranges[idx].clone().into_par_iter();

                #[cfg(not(feature = "parallel"))]
                let range_iter = self.ranges[idx].clone();

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
                let mut final_weights = Challenge::zero_vec(1 << self.log_b);
                final_weights[self.ranges[idx].clone()].copy_from_slice(&weight_values);

                // Return the computed weights as a full multilinear evaluation list.
                Weights::linear(EvaluationsList::new(final_weights))
            }
        };

        (weights, sum)
    }
}

#[derive(Debug)]
pub struct ConcatMats<Val> {
    values: Vec<Val>,
    meta: ConcatMatsMeta,
}

impl<Val: Field> ConcatMats<Val> {
    fn new(mats: Vec<RowMajorMatrix<Val>>) -> Self {
        let meta = ConcatMatsMeta::new(&mats.iter().map(Matrix::dimensions).collect::<Vec<_>>());
        let mut values = Val::zero_vec(1 << meta.log_b);

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            izip!(&meta.ranges, mats).for_each(|(range, mat)| {
                // Parallel row-wise copying into padded storage
                values[range.clone()]
                    .par_chunks_mut(mat.width().next_power_of_two())
                    .zip(mat.par_row_slices())
                    .for_each(|(dst, src)| dst[..src.len()].copy_from_slice(src));
            });
        }

        #[cfg(not(feature = "parallel"))]
        {
            izip!(&meta.ranges, mats).for_each(|(range, mat)| {
                // Sequential row-wise copying into padded storage
                values[range.clone()]
                    .chunks_mut(mat.width().next_power_of_two())
                    .zip(mat.row_slices())
                    .for_each(|(dst, src)| dst[..src.len()].copy_from_slice(src));
            });
        }

        Self { values, meta }
    }

    fn mat(&self, idx: usize) -> HorizontallyTruncated<Val, RowMajorMatrixView<'_, Val>> {
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

    use itertools::{chain, cloned, rev};
    use p3_baby_bear::BabyBear;
    use p3_field::{PrimeCharacteristicRing, dot_product};
    use p3_matrix::dense::RowMajorMatrix;

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
        let query = MlQuery::Eq(vec![]);

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
        let query = MlQuery::EqRotateRight(vec![F::from_u8(7)], 1);

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
                let full: &[F] = weight.evals();
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
        let query = MlQuery::Eq(vec![F::from_u8(0)]); // z selects row 0
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
                let expected = rev(chain![r[..2].iter().copied(), vec![F::from_u8(0)], bits])
                    .collect::<Vec<_>>();
                assert_eq!(point.0, expected);
            }
            Weights::Linear { .. } => panic!("Expected Weights::Evaluation"),
        }
    }

    #[test]
    fn test_constraint_eq_rotate_right_high_dim() {
        // Matrix: height = 4, width = 4
        let mat = RowMajorMatrix::new(
            vec![
                1, 2, 3, 4, // row 0
                5, 6, 7, 8, // row 1
                9, 10, 11, 12, // row 2
                13, 14, 15, 16, // row 3
            ]
            .into_iter()
            .map(F::from_u8)
            .collect(),
            4,
        );
        let concat = ConcatMats::new(vec![mat]);
        let meta = &concat.meta;

        // Query: rotate z = [1, 0], rotation = 1
        let query = MlQuery::EqRotateRight(vec![F::from_u8(1), F::from_u8(0)], 1);
        let ys = vec![
            F::from_u8(21),
            F::from_u8(22),
            F::from_u8(23),
            F::from_u8(24),
        ];
        let r = vec![F::from_u8(2), F::from_u8(1)];

        let (weights, sum) = meta.constraint(0, &query, &ys, &r);

        // eq_r = eq_{[2,1]} = list of 4 values, based on r
        let eq_r = {
            let r0 = r[0];
            let r1 = r[1];
            vec![
                (F::ONE - r0) * (F::ONE - r1),
                (F::ONE - r0) * r1,
                r0 * (F::ONE - r1),
                r0 * r1,
            ]
        };
        let expected_sum = dot_product(cloned(&ys), cloned(&eq_r[..ys.len()]));
        assert_eq!(sum, expected_sum);

        match weights {
            Weights::Linear { weight } => {
                let full = weight.evals();
                assert_eq!(full.len(), 1 << meta.log_b);

                let expected_weight_range = &full[meta.ranges[0].clone()];
                let mle = query.to_mle(F::ONE);

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
}
