use p3_field::Field;
use p3_matrix::{Matrix, dense::RowMajorMatrixViewMut};
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
    debug_assert!(height.is_power_of_two());
    wavelet_transform_batch(mat, height);
}

pub fn wavelet_transform_batch<F: Field>(mat: &mut RowMajorMatrixViewMut<'_, F>, size: usize) {
    debug_assert!(mat.height() % size == 0 && size.is_power_of_two());

    #[cfg(feature = "parallel")]
    if mat.height() > workload_size::<F>() && mat.height() != size {
        let chunk_rows = size * std::cmp::max(1, workload_size::<F>() / size);
        mat.par_row_chunks_mut(chunk_rows)
            .for_each(|mut chunk| wavelet_transform_batch(&mut chunk, size));
        return;
    }

    match size {
        0 | 1 => {}
        2 => {
            for v in mat.row_chunks_exact_mut(2) {
                v.values[1] += v.values[0];
            }
        }
        4 => {
            for v in mat.row_chunks_exact_mut(4) {
                v.values[1] += v.values[0];
                v.values[3] += v.values[2];
                v.values[2] += v.values[0];
                v.values[3] += v.values[1];
            }
        }
        8 => {
            for v in mat.row_chunks_exact_mut(8) {
                v.values[1] += v.values[0];
                v.values[3] += v.values[2];
                v.values[2] += v.values[0];
                v.values[3] += v.values[1];
                v.values[5] += v.values[4];
                v.values[7] += v.values[6];
                v.values[6] += v.values[4];
                v.values[7] += v.values[5];
                v.values[4] += v.values[0];
                v.values[5] += v.values[1];
                v.values[6] += v.values[2];
                v.values[7] += v.values[3];
            }
        }
        16 => {
            for mut v in mat.row_chunks_exact_mut(16) {
                for v in v.row_chunks_exact_mut(4) {
                    v.values[1] += v.values[0];
                    v.values[3] += v.values[2];
                    v.values[2] += v.values[0];
                    v.values[3] += v.values[1];
                }
                let (a, mut v) = v.split_rows_mut(4);
                let (b, mut v) = v.split_rows_mut(4);
                let (c, d) = v.split_rows_mut(4);
                for i in 0..4 {
                    b.values[i] += a.values[i];
                    d.values[i] += c.values[i];
                    c.values[i] += a.values[i];
                    d.values[i] += b.values[i];
                }
            }
        }
        n => {
            let n1 = 1 << (n.trailing_zeros() / 2);
            let n2 = size / n1;

            wavelet_transform_batch(mat, n1);
            mat.par_row_chunks_exact_mut(n1 * n2).for_each(|matrix| {
                let m = RowMajorMatrixViewMut::new(matrix.values, n1).transpose();
                matrix.values.copy_from_slice(&m.values);
            });

            wavelet_transform_batch(mat, n2);
            mat.par_row_chunks_exact_mut(n1 * n2).for_each(|matrix| {
                let m = RowMajorMatrixViewMut::new(matrix.values, n2).transpose();
                matrix.values.copy_from_slice(&m.values);
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::dense::RowMajorMatrix;
    use proptest::prelude::*;

    use super::*;

    type F = BabyBear;

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
    fn test_wavelet_transform_batch_parallel_chunks() {
        // Define the size for the wavelet transform batch
        let batch_size = 2_i32.pow(20) as usize;
        // Ensure values.len() > size to enter parallel execution
        let total_size = batch_size * 4;
        let values = (1..=total_size as u64).map(F::from_u64).collect::<Vec<_>>();

        // Keep a copy to compare later
        let original_values = values.clone();

        let mut mat = RowMajorMatrix::new_col(values);

        // Run batch transform on 256-sized chunks
        wavelet_transform_batch(&mut mat.as_view_mut(), batch_size);

        // Verify that the first chunk has been transformed correctly
        let mut mat1 = RowMajorMatrix::new_col(original_values[..batch_size].to_vec());

        wavelet_transform_batch(&mut mat1.as_view_mut(), batch_size);
        assert_eq!(mat.values[..batch_size], mat1.values);

        // Ensure that the transformation occurred separately for each chunk
        for i in 1..4 {
            let start = i * batch_size;
            let end = start + batch_size;

            let mut mat_loop = RowMajorMatrix::new_col(original_values[start..end].to_vec());
            wavelet_transform_batch(&mut mat_loop.as_view_mut(), batch_size);

            assert_eq!(
                mat.values[start..end],
                mat_loop.values,
                "Mismatch in chunk {i}"
            );
        }

        // Ensure the first element remains unchanged
        assert_eq!(mat.values[0], F::from_u64(1));

        // Ensure the last element has accumulated all values from its own chunk
        let expected_last_chunk_sum =
            (total_size as u64 - batch_size as u64 + 1..=total_size as u64).sum::<u64>();
        assert_eq!(
            mat.values[total_size - 1],
            F::from_u64(expected_last_chunk_sum),
            "Final element mismatch"
        );
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
