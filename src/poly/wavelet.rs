use p3_field::{Field, PackedValue};
use p3_matrix::{Matrix, dense::RowMajorMatrixViewMut, util::reverse_matrix_index_bits};
use p3_util::log2_strict_usize;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::whir::utils::workload_size;

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
    // As a rough rule of thumb, we will use chunks of size `workload_size::<F>()` unless the
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
    // passing information between cores as much as possible. The strategy is to split the
    // matrix into `num_par_row` chunks, send each chunk to a core and do the first
    // `log_num_par_rows` rounds on that core.
    // We then do the remaining `log_height - log_num_par_rows` using `par_row_chunks_exact_mut`
    // in the standard way. This avoids needing any large scale data fiddling (e.g. doing
    // a reverse_matrix_index_bits on the entire matrix) which seems to
    // cost more than the time it saves.

    // Split the matrix into `num_par_rows` chunks.
    mat.par_row_chunks_exact_mut(num_par_rows)
        .for_each(|mut chunk| {
            // As each core has access to `num_par_rows`, it can do
            // `log_num_par_rows` passes of the wavelet kernel.

            // The `i`'th pass takes `2^{i + 1}` rows, splits them in half to get `lo` and `hi`,
            // and sets `hi += lo`. If we bit reverse the rows, then the `i`'th pass will take
            // `2^{log_num_par_rows - i}` rows. We want to take as many rows as possible as this
            // will let us use SIMD instructions. Hence we want to start with things bit-reversed
            // but switch to the original order halfway through.

            // Initial index reversal. Note that this is very cheap as chunk is small and all contained
            // in the L1 cache.
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

    // Do the final `log_height - log_num_par_rows` passes.
    for i in log_num_par_rows..log_height {
        let block_size = 1 << (i + 1);
        // Apply the wavelet kernel on blocks of size `block_size`.
        par_wavelet_kernel(mat, block_size);
    }
}

/// Apply the wavelet kernel on blocks of a given size on a single core.
///
/// Intended for use in cases where mat fits on a single core and block size is large
/// enough to benefit from SIMD instructions.
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
///
/// Intended for use in cases where mat does not fit on a single core and block_size is large.
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

/// Perform the inverse wavelet transform on each row of a matrix.
///
/// This function reverses the operation of `wavelet_transform`, recovering the original
/// coefficients of a multilinear polynomial from its wavelet-transformed evaluation form.
///
/// The implementation mirrors `wavelet_transform`, with the same core ideas:
/// - Decompose work into per-core chunks for efficient parallelization.
/// - Use bit-reversal to maximize SIMD-friendly access patterns in early rounds.
/// - Apply high-order passes in-place using SIMD, undoing the transform step `hi -= lo`.
///
/// For a matrix of height `2^k`, this function performs `k` rounds in reverse order:
/// - The high-order rounds are done in place without reversing.
/// - Then each chunk is reversed and lower-order rounds are applied.
/// - The rows are then restored to original order.
/// - Finally, any remaining global rounds are applied in parallel.
///
/// # Panics
/// Panics in debug mode if the matrix height is not a power of two.
pub fn inverse_wavelet_transform<F: Field>(mat: &mut RowMajorMatrixViewMut<'_, F>) {
    let height = mat.height();
    debug_assert!(height.is_power_of_two());
    let log_height = log2_strict_usize(height);
    let half_log = log_height / 2;
    let rough_sqrt_height = 1 << half_log;

    // Choose the number of rows to process in parallel per chunk
    let num_par_rows = (workload_size::<F>() / mat.width())
        .next_power_of_two()
        .max(rough_sqrt_height)
        .min(height);

    let log_num_par_rows = log2_strict_usize(num_par_rows);
    let half_log_num_par_rows = log_num_par_rows / 2;

    assert!(log_num_par_rows >= half_log);

    // Process each chunk in parallel
    mat.par_row_chunks_exact_mut(num_par_rows)
        .for_each(|mut chunk| {
            // First, perform the high-order rounds in normal layout (descending order)
            for i in (half_log_num_par_rows..log_num_par_rows).rev() {
                let block_size = 1 << (i + 1);
                inverse_wavelet_kernel(&mut chunk, block_size);
            }

            // Reverse rows to simulate the recursive layout of the original transform
            reverse_matrix_index_bits(&mut chunk);

            // Then perform the low-order rounds with reversed layout (descending order)
            for i in (0..half_log_num_par_rows).rev() {
                let block_size = 1 << (log_num_par_rows - i);
                inverse_wavelet_kernel(&mut chunk, block_size);
            }

            // Restore rows to original order
            reverse_matrix_index_bits(&mut chunk);
        });

    // Finish remaining global rounds (descending order)
    for i in (0..log_height - log_num_par_rows).rev() {
        let block_size = 1 << (log_num_par_rows + i + 1);
        par_inverse_wavelet_kernel(mat, block_size);
    }
}

/// Apply the inverse wavelet kernel on blocks of a given size on a single core.
///
/// This reverses the wavelet transform step:
/// ```text
/// hi += lo  →  hi -= lo
/// ```
/// where `lo` and `hi` are the two halves of each block.
fn inverse_wavelet_kernel<F: Field>(mat: &mut RowMajorMatrixViewMut<'_, F>, block_size: usize) {
    let half_block_size = mat.width() * block_size / 2;
    mat.row_chunks_exact_mut(block_size).for_each(|block| {
        let (lo, hi) = block.values.split_at_mut(half_block_size);
        let (pack_lo, sfx_lo) = F::Packing::pack_slice_with_suffix_mut(lo);
        let (pack_hi, sfx_hi) = F::Packing::pack_slice_with_suffix_mut(hi);
        pack_hi.iter_mut().zip(pack_lo).for_each(|(hi, lo)| {
            *hi -= *lo;
        });
        sfx_hi.iter_mut().zip(sfx_lo).for_each(|(hi, lo)| {
            *hi -= *lo;
        });
    });
}

/// Apply the inverse wavelet kernel in parallel across matrix blocks.
///
/// This is used for the remaining global rounds after per-core chunked passes.
fn par_inverse_wavelet_kernel<F: Field>(mat: &mut RowMajorMatrixViewMut<'_, F>, block_size: usize) {
    let half_block_size = mat.width() * block_size / 2;
    mat.par_row_chunks_exact_mut(block_size).for_each(|block| {
        let (lo, hi) = block.values.split_at_mut(half_block_size);
        let (pack_lo, sfx_lo) = F::Packing::pack_slice_with_suffix_mut(lo);
        let (pack_hi, sfx_hi) = F::Packing::pack_slice_with_suffix_mut(hi);
        pack_hi.iter_mut().zip(pack_lo).for_each(|(hi, lo)| {
            *hi -= *lo;
        });
        sfx_hi.iter_mut().zip(sfx_lo).for_each(|(hi, lo)| {
            *hi -= *lo;
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
