use p3_field::{Field, PackedValue};
use p3_matrix::{Matrix, dense::RowMajorMatrixViewMut, util::reverse_matrix_index_bits};
use p3_util::log2_strict_usize;
#[cfg(feature = "parallel")]
use {super::utils::workload_size, rayon::prelude::*};

/// Perform the inverse wavelet transform on each row of a matrix.
///
/// This is a wrapper for `inverse_wavelet_transform_batch`, applying the inverse transform
/// over the full height of the matrix. Assumes the matrix has a power-of-two height.
///
/// This function is typically used to recover coefficients of a multilinear polynomial
/// from its evaluation form (e.g. after forward wavelet transform).
///
/// # Panics
/// Panics in debug mode if the matrix height is not a power of two.
pub fn inverse_wavelet_transform<F: Field>(mat: &mut RowMajorMatrixViewMut<'_, F>) {
    let height = mat.height();
    debug_assert!(height.is_power_of_two());
    inverse_wavelet_transform_batch(mat, height);
}

/// Apply the inverse wavelet transform in-place to a matrix, grouped into batches of `size` rows.
///
/// The matrix is assumed to contain evaluations of a multilinear polynomial.
/// This function inverts the recursive structure of the forward wavelet transform,
/// recovering the original coefficient representation for each batch.
///
/// Optimized paths exist for sizes 2, 4, and 8, and larger powers of two are handled recursively.
/// Parallelism is applied when beneficial.
///
/// # Parameters
/// - `mat`: Mutable view of the matrix to transform in-place.
/// - `size`: Number of rows per transform block (must be a power of two).
///
/// # Panics
/// Panics in debug mode if:
/// - `size` is not a power of two,
/// - `mat.height()` is not divisible by `size`.
pub fn inverse_wavelet_transform_batch<F: Field>(
    mat: &mut RowMajorMatrixViewMut<'_, F>,
    size: usize,
) {
    debug_assert!(mat.height() % size == 0 && size.is_power_of_two());

    #[cfg(feature = "parallel")]
    if mat.height() > workload_size::<F>() && mat.height() != size {
        // Split into chunks of `chunk_rows` for parallelism if possible.
        let num_chunks_approx = mat.height() / std::cmp::max(size, workload_size::<F>());
        let chunk_rows = ((mat.height() / size) / std::cmp::max(1, num_chunks_approx)) * size;
        let chunk_rows = std::cmp::max(size, chunk_rows);

        if chunk_rows > 0 && mat.height() % chunk_rows == 0 && chunk_rows != mat.height() {
            mat.par_row_chunks_mut(chunk_rows)
                .for_each(|mut chunk| inverse_wavelet_transform_batch(&mut chunk, size));
            return;
        }
    }

    match size {
        0 | 1 => {} // No operation for size 0 or 1
        2 => {
            for v in mat.row_chunks_exact_mut(2) {
                v.values[1] -= v.values[0];
            }
        }
        4 => {
            for v in mat.row_chunks_exact_mut(4) {
                v.values[3] -= v.values[1];
                v.values[2] -= v.values[0];
                v.values[3] -= v.values[2];
                v.values[1] -= v.values[0];
            }
        }
        8 => {
            for v in mat.row_chunks_exact_mut(8) {
                // Undo top-level cross-accumulations
                v.values[7] -= v.values[3];
                v.values[6] -= v.values[2];
                v.values[5] -= v.values[1];
                v.values[4] -= v.values[0];

                // Undo second block (v[4] to v[7])
                v.values[7] -= v.values[5];
                v.values[6] -= v.values[4];
                v.values[7] -= v.values[6];
                v.values[5] -= v.values[4];

                // Undo first block (v[0] to v[3])
                v.values[3] -= v.values[1];
                v.values[2] -= v.values[0];
                v.values[3] -= v.values[2];
                v.values[1] -= v.values[0];
            }
        }
        n => {
            // Recursive case: undo forward structure step-by-step.

            // Choose factors: n = n1 * n2, where n1 = 2^{floor(k/2)}, n2 = 2^{ceil(k/2)}
            let n1 = 1 << (n.trailing_zeros() / 2);
            let n2 = size / n1;

            // Undo the second transpose (forward had new(..., n2).transpose())
            // Current layout: n2 rows × n1 columns → transpose using width = n1
            mat.par_row_chunks_exact_mut(n1 * n2).for_each(|chunk| {
                let m = RowMajorMatrixViewMut::new(chunk.values, n1).transpose();
                chunk.values.copy_from_slice(&m.values);
            });

            // Apply inverse wavelet transform to each n2-sized block
            inverse_wavelet_transform_batch(mat, n2);

            // Undo the first transpose (forward had new(..., n1).transpose())
            // Current layout: n1 rows × n2 columns → transpose using width = n2
            mat.par_row_chunks_exact_mut(n1 * n2).for_each(|chunk| {
                let m = RowMajorMatrixViewMut::new(chunk.values, n2).transpose();
                chunk.values.copy_from_slice(&m.values);
            });

            // Apply inverse wavelet transform to each n1-sized block
            inverse_wavelet_transform_batch(mat, n1);
        }
    }
}

/// In-place Fast Wavelet Transform on a matrix view.
///
/// Applies the kernel:
///   [1 0]
///   [1 1]
///
/// Assumes the number of rows is a power of two.
pub fn wavelet_transform<F: Field>(mat: &mut RowMajorMatrixViewMut<'_, F>) {
    let height = mat.height();
    let log_height = log2_strict_usize(height);
    let half_log = log_height / 2;
    let rough_sqrt_height = 1 << half_log;

    // When applying the wavelet transform in parallel, we want to try and split up the work
    // into medium sized chunks which fit nicely on the different cores.
    // As a rough rule of thumb, we will use chuncks of size `workload_size::<F>()` unless the
    // transform is much larger than this.
    let num_par_rows = (workload_size::<F>() / mat.width())
        .next_power_of_two()
        .max(rough_sqrt_height)
        .min(height);

    // If `height < workload_size::<F>() / mat.width()` there isn't a huge advantage to
    // parallelizing the work so we just use a single core.

    let log_num_par_rows = log2_strict_usize(num_par_rows);
    let half_log_num_par_rows = log_num_par_rows / 2;

    assert!(log_num_par_rows >= half_log);

    // When possible we want to maximise the amount of work done on a given core avoiding
    // passing information between cores as much as possible.
    mat.par_row_chunks_exact_mut(num_par_rows)
        .for_each(|mut chunk| {
            // As each core has access to `num_par_rows`, it can do
            // `log_num_par_rows` passes of the wavelet kernel.

            // The `i`'th pass takes `2^{i + 1}` rows, splits them in half to get `lo` and `hi`,
            // and sets `hi += lo`. If we bit reverse the rows, then the `i`'th pass will take
            // `2^{log_num_par_rows - i}` rows. We want to take as many rows as possible as this
            // will let us use SIMD instructions. Hence we want to start with things bit-reversed
            // but switch to the original order halfway through.

            // Initial index reversal. (This could be avoided if mat.width() is assumed to be large)
            reverse_matrix_index_bits(&mut chunk);
            // Perform passes `0` through `half_log_num_par_rows - 1`.
            for i in 0..half_log_num_par_rows {
                let block_size = 1 << (log_num_par_rows - i);

                // Apply the wavelet kernel on blocks of size `block_size`.
                wavelet_kernel(&mut chunk, block_size);
            }

            // Revert rows to initial ordering.
            reverse_matrix_index_bits(&mut chunk);
            // Perform passes `half_log_num_par_rows` through `log_num_par_rows - 1`.
            for i in half_log_num_par_rows..log_num_par_rows {
                let block_size = 1 << (i + 1);

                // Apply the wavelet kernel on blocks of size `block_size`.
                wavelet_kernel(&mut chunk, block_size);
            }
        });

    for i in log_num_par_rows..log_height {
        let block_size = 1 << (i + 1);
        // Apply the wavelet kernel on blocks of size `block_size`.
        par_wavelet_kernel(mat, block_size);
    }

    // // We have now done the first `log_num_par_rows` passes.
    // // For the final layers we reverse the indices of our matrix and do a similar idea to
    // // the first `log_num_par_rows` passes. The only complication is that we may not need
    // // to do the full number of passes as `final_layers_start` might be less than `log_num_par_rows`.
    // // We should also skip this in the case that `num_par_rows = log_height`.
    // if log_num_par_rows < log_height {
    //     let num_final_layers = log_height - log_num_par_rows;
    //     reverse_matrix_index_bits(mat);
    //     mat.par_row_chunks_exact_mut(num_par_rows)
    //         .for_each(|mut chunk| {
    //             // Perform passes `half_log_num_par_rows` through `log_num_par_rows - 1`.
    //             for i in (half_log_num_par_rows..num_final_layers).rev() {
    //                 let block_size = 1 << (i + 1);

    //                 // Apply the wavelet kernel on blocks of size `block_size`.
    //                 wavelet_kernel(&mut chunk, block_size);
    //             }

    //             reverse_matrix_index_bits(&mut chunk);
    //             // Perform passes `0` through `half_log_num_par_rows - 1`.
    //             for i in (0..half_log_num_par_rows.min(num_final_layers)).rev() {
    //                 let block_size = 1 << (log_num_par_rows - i);

    //                 // Apply the wavelet kernel on blocks of size `block_size`.
    //                 wavelet_kernel(&mut chunk, block_size);
    //             }
    //             reverse_matrix_index_bits(&mut chunk);
    //         });

    //     reverse_matrix_index_bits(mat);
    // }
}

/// Apply the wavelet kernel on blocks of a given size on a single core.
fn wavelet_kernel<F: Field>(mat: &mut RowMajorMatrixViewMut<'_, F>, block_size: usize) {
    let half_block_size = mat.width() * block_size / 2;
    mat.row_chunks_exact_mut(block_size).for_each(|block| {
        let (lo, hi) = block.values.split_at_mut(half_block_size);
        let (pack_lo, sfx_lo) = F::Packing::pack_slice_with_suffix_mut(lo);
        let (pack_hi, sfx_hi) = F::Packing::pack_slice_with_suffix_mut(hi);
        pack_hi.iter_mut().zip(pack_lo).for_each(|(hi, lo)| {
            *hi += *lo;
        });
        sfx_hi.iter_mut().zip(sfx_lo).for_each(|(hi, lo)| {
            *hi += *lo;
        });
    });
}

/// Apply the wavelet kernel on blocks of a given size making use of parallelization.
fn par_wavelet_kernel<F: Field>(mat: &mut RowMajorMatrixViewMut<'_, F>, block_size: usize) {
    let half_block_size = mat.width() * block_size / 2;
    mat.par_row_chunks_exact_mut(block_size).for_each(|block| {
        let (lo, hi) = block.values.split_at_mut(half_block_size);
        let (pack_lo, sfx_lo) = F::Packing::pack_slice_with_suffix_mut(lo);
        let (pack_hi, sfx_hi) = F::Packing::pack_slice_with_suffix_mut(hi);
        pack_hi.iter_mut().zip(pack_lo).for_each(|(hi, lo)| {
            *hi += *lo;
        });
        sfx_hi.iter_mut().zip(sfx_lo).for_each(|(hi, lo)| {
            *hi += *lo;
        });
    });
}

#[cfg(test)]
mod tests {
    use core::panic;

    use p3_baby_bear::BabyBear;
    use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, extension::BinomialExtensionField};
    use p3_matrix::dense::RowMajorMatrix;
    use proptest::prelude::*;

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    #[test]
    fn test_wavelet_transform_single_element() {
        let values = vec![F::from_u64(5)];
        let mut mat = RowMajorMatrix::new_col(values);
        wavelet_transform(&mut mat.as_view_mut());
        assert_eq!(mat.values, vec![F::from_u64(5)]);
    }

    #[test]
    fn test_wavelet_transform_size_2() {
        let v1 = F::from_u64(3);
        let v2 = F::from_u64(7);
        let values = vec![v1, v2];
        let mut mat = RowMajorMatrix::new_col(values);
        wavelet_transform(&mut mat.as_view_mut());
        assert_eq!(mat.values, vec![v1, v1 + v2]);
    }

    #[test]
    fn test_wavelet_transform_size_4() {
        let v1 = F::from_u64(1);
        let v2 = F::from_u64(2);
        let v3 = F::from_u64(3);
        let v4 = F::from_u64(4);
        let values = vec![v1, v2, v3, v4];
        let mut mat = RowMajorMatrix::new_col(values);

        wavelet_transform(&mut mat.as_view_mut());

        assert_eq!(mat.values, vec![v1, v1 + v2, v3 + v1, v1 + v2 + v3 + v4]);
    }

    #[test]
    fn test_wavelet_transform_size_8() {
        let values = (1..=8).map(F::from_u64).collect::<Vec<_>>();
        let v1 = values[0];
        let v2 = values[1];
        let v3 = values[2];
        let v4 = values[3];
        let v5 = values[4];
        let v6 = values[5];
        let v7 = values[6];
        let v8 = values[7];

        let mut mat = RowMajorMatrix::new_col(values);

        wavelet_transform(&mut mat.as_view_mut());

        assert_eq!(
            mat.values,
            vec![
                v1,
                v1 + v2,
                v3 + v1,
                v1 + v2 + v3 + v4,
                v5 + v1,
                v1 + v2 + v5 + v6,
                v3 + v1 + v5 + v7,
                v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8
            ]
        );
    }

    #[test]
    fn test_wavelet_transform_size_16() {
        let values = (1..=16).map(F::from_u64).collect::<Vec<_>>();
        let v1 = values[0];
        let v2 = values[1];
        let v3 = values[2];
        let v4 = values[3];
        let v5 = values[4];
        let v6 = values[5];
        let v7 = values[6];
        let v8 = values[7];
        let v9 = values[8];
        let v10 = values[9];
        let v11 = values[10];
        let v12 = values[11];
        let v13 = values[12];
        let v14 = values[13];
        let v15 = values[14];
        let v16 = values[15];

        let mut mat = RowMajorMatrix::new_col(values);

        wavelet_transform(&mut mat.as_view_mut());

        assert_eq!(
            mat.values,
            vec![
                v1,
                v1 + v2,
                v3 + v1,
                v1 + v2 + v3 + v4,
                v5 + v1,
                v1 + v2 + v5 + v6,
                v3 + v1 + v5 + v7,
                v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8,
                v9 + v1,
                v1 + v2 + v9 + v10,
                v3 + v1 + v9 + v11,
                v1 + v2 + v3 + v4 + v9 + v10 + v11 + v12,
                v5 + v1 + v9 + v13,
                v1 + v2 + v5 + v6 + v9 + v10 + v13 + v14,
                v3 + v1 + v5 + v7 + v9 + v11 + v13 + v15,
                v1 + v2
                    + v3
                    + v4
                    + v5
                    + v6
                    + v7
                    + v8
                    + v9
                    + v10
                    + v11
                    + v12
                    + v13
                    + v14
                    + v15
                    + v16
            ]
        );
    }

    #[test]
    fn test_wavelet_transform_large() {
        let size = 2_i32.pow(10) as u64;
        let values = (1..=size).map(F::from_u64).collect::<Vec<_>>();
        let v1 = values[0];

        let mut mat = RowMajorMatrix::new_col(values);

        wavelet_transform(&mut mat.as_view_mut());

        // Verify the first element remains unchanged
        assert_eq!(mat.values[0], v1);

        // Verify last element has accumulated all previous values
        let expected_last = (1..=size).sum::<u64>();
        assert_eq!(mat.values[size as usize - 1], F::from_u64(expected_last));
    }

    #[test]
    fn test_wavelet_transform_extension() {
        let size = 2_i32.pow(16) as u32;
        let values = (1..=size)
            .map(|i| {
                EF::from_basis_coefficients_iter((0..4).map(|j| F::from_u32(4 * i + j))).unwrap()
            })
            .collect::<Vec<_>>();

        let mut mat_ef = RowMajorMatrix::new_col(values);
        let mut mat_base = mat_ef.clone().flatten_to_base::<F>();

        wavelet_transform(&mut mat_ef.as_view_mut());
        wavelet_transform(&mut mat_base.as_view_mut());

        let out_ef = RowMajorMatrix::new_col(EF::reconstitute_from_base(mat_base.values));

        assert_eq!(mat_ef, out_ef);
    }

    #[test]
    fn test_wavelet_roundtrip() {
        let values = (1..=32).map(F::from_u64).collect::<Vec<_>>();
        let original = values.clone();

        let mut mat = RowMajorMatrix::new_col(values);
        wavelet_transform(&mut mat.as_view_mut());
        inverse_wavelet_transform(&mut mat.as_view_mut());

        assert_eq!(mat.values, original);
    }

    #[test]
    fn test_wavelet_roundtrip_vlarge() {
        let values = (1..=(1 << 20)).map(F::from_u64).collect::<Vec<_>>();
        let original = values.clone();

        let mut mat = RowMajorMatrix::new_col(values);
        wavelet_transform(&mut mat.as_view_mut());
        inverse_wavelet_transform(&mut mat.as_view_mut());

        assert_eq!(mat.values, original, "Roundtrip failed");
    }

    proptest! {
        #[test]
        fn prop_wavelet_roundtrip(values in prop::collection::vec(0u64..u64::from(u32::MAX), 1..=100).prop_map(|v| {
            let size = 1 << v.len().ilog2(); // Truncate to nearest power of 2
            v.into_iter().take(size).collect::<Vec<_>>()
        })) {
            // Convert u64 input values into field elements
            let values: Vec<F> = values.into_iter().map(F::from_u64).collect();

            // Clone the original values for later comparison
            let original = values.clone();

            // Wrap values in a column-major matrix
            let mut mat = RowMajorMatrix::new_col(values);

            // Apply the forward wavelet transform
            wavelet_transform(&mut mat.as_view_mut());

            // Apply the inverse wavelet transform
            inverse_wavelet_transform(&mut mat.as_view_mut());

            // Ensure roundtrip: output matches original input
            prop_assert_eq!(mat.values, original);
        }
    }
}
