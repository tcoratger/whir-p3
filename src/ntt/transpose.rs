use std::mem::swap;

use super::matrix::MatrixMut;
use crate::ntt::utils::workload_size;

/// Transpose a matrix in-place.
/// Will batch transpose multiple matrices if the length of the slice is a multiple of rows * cols.
/// This algorithm assumes that both rows and cols are powers of two.
pub fn transpose<F: Sized + Copy + Send>(matrix: &mut [F], rows: usize, cols: usize) {
    debug_assert_eq!(matrix.len() % (rows * cols), 0);
    debug_assert!(rows.is_power_of_two());
    debug_assert!(cols.is_power_of_two());

    if rows == cols {
        for matrix in matrix.chunks_exact_mut(rows * cols) {
            let matrix = MatrixMut::from_mut_slice(matrix, rows, cols);
            transpose_square(matrix);
        }
    } else {
        // TODO: Special case for rows = 2 * cols and cols = 2 * rows.
        // TODO: Special case for very wide matrices (e.g. n x 16).
        let mut scratch = vec![matrix[0]; rows * cols];
        for matrix in matrix.chunks_exact_mut(rows * cols) {
            scratch.copy_from_slice(matrix);
            let src = MatrixMut::from_mut_slice(scratch.as_mut_slice(), rows, cols);
            let dst = MatrixMut::from_mut_slice(matrix, cols, rows);
            transpose_copy(src, dst);
        }
    }
}

/// Transpose and swap two square size matrices (parallel version).
///
/// The size must be a power of two.
fn transpose_square_swap<F: Sized + Send>(mut a: MatrixMut<'_, F>, mut b: MatrixMut<'_, F>) {
    debug_assert!(a.is_square());
    debug_assert_eq!(a.rows(), b.cols());
    debug_assert_eq!(a.cols(), b.rows());
    debug_assert!(a.rows().is_power_of_two());
    debug_assert!(workload_size::<F>() >= 2);

    let size = a.rows();

    // Direct swaps for small matrices (≤8x8)
    // - Avoids recursion overhead
    // - Uses basic element-wise swaps
    if size <= 8 {
        for i in 0..size {
            for j in 0..size {
                swap(&mut a[(i, j)], &mut b[(j, i)]);
            }
        }
        return;
    }

    // If the matrix is large, use recursive subdivision:
    // - Improves cache efficiency by working on smaller blocks
    // - Enables parallel execution
    if 2 * size * size > workload_size::<F>() {
        let n = size / 2;
        let (aa, ab, ac, ad) = a.split_quadrants(n, n);
        let (ba, bb, bc, bd) = b.split_quadrants(n, n);

        #[cfg(feature = "parallel")]
        rayon::join(
            || {
                rayon::join(
                    || transpose_square_swap(aa, ba),
                    || transpose_square_swap(ab, bc),
                )
            },
            || {
                rayon::join(
                    || transpose_square_swap(ac, bb),
                    || transpose_square_swap(ad, bd),
                )
            },
        );

        #[cfg(not(feature = "parallel"))]
        {
            transpose_square_swap(aa, ba);
            transpose_square_swap(ab, bc);
            transpose_square_swap(ac, bb);
            transpose_square_swap(ad, bd);
        }
    } else {
        // Optimized 2×2 loop unrolling for larger blocks
        // - Reduces loop overhead
        // - Increases memory access efficiency
        for i in (0..size).step_by(2) {
            for j in (0..size).step_by(2) {
                swap(&mut a[(i, j)], &mut b[(j, i)]);
                swap(&mut a[(i + 1, j)], &mut b[(j, i + 1)]);
                swap(&mut a[(i, j + 1)], &mut b[(j + 1, i)]);
                swap(&mut a[(i + 1, j + 1)], &mut b[(j + 1, i + 1)]);
            }
        }
    }
}

/// Transpose a square matrix in-place. Asserts that the size of the matrix is a power of two.
/// This is the parallel version.
fn transpose_square<F: Sized + Send>(mut m: MatrixMut<'_, F>) {
    debug_assert!(m.is_square());
    debug_assert!(m.rows().is_power_of_two());
    let size = m.rows();
    if size * size > workload_size::<F>() {
        // Recurse into quadrants.
        // This results in a cache-oblivious algorithm.
        let n = size / 2;
        let (a, b, c, d) = m.split_quadrants(n, n);

        #[cfg(feature = "parallel")]
        rayon::join(
            || transpose_square_swap(b, c),
            || rayon::join(|| transpose_square(a), || transpose_square(d)),
        );

        #[cfg(not(feature = "parallel"))]
        {
            transpose_square(a);
            transpose_square(d);
            transpose_square_swap(b, c);
        }
    } else {
        for i in 0..size {
            for j in (i + 1)..size {
                // unsafe needed due to lack of bounds-check by swap. We are guaranteed that (i,j)
                // and (j,i) are within the bounds.
                unsafe {
                    m.swap((i, j), (j, i));
                }
            }
        }
    }
}

/// Efficient parallel matrix transposition.
///
/// Uses cache-friendly recursive decomposition and direct pointer manipulation for maximum
/// performance.
fn transpose_copy<F: Copy + Send>(src: MatrixMut<'_, F>, mut dst: MatrixMut<'_, F>) {
    assert_eq!(src.rows(), dst.cols());
    assert_eq!(src.cols(), dst.rows());

    let (rows, cols) = (src.rows(), src.cols());

    // Direct element-wise transposition for small matrices (avoids recursion overhead)
    if rows * cols <= 64 {
        unsafe {
            for i in 0..rows {
                for j in 0..cols {
                    *dst.ptr_at_mut(j, i) = *src.ptr_at(i, j);
                }
            }
        }
        return;
    }

    // Determine optimal split axis
    let (split_size, split_vertical) = if rows > cols {
        (rows / 2, true)
    } else {
        (cols / 2, false)
    };

    // Split source and destination matrices accordingly
    let ((src_a, src_b), (dst_a, dst_b)) = if split_vertical {
        (
            src.split_vertical(split_size),
            dst.split_horizontal(split_size),
        )
    } else {
        (
            src.split_horizontal(split_size),
            dst.split_vertical(split_size),
        )
    };

    // Execute recursive transposition
    #[cfg(feature = "parallel")]
    rayon::join(
        || transpose_copy(src_a, dst_a),
        || transpose_copy(src_b, dst_b),
    );

    #[cfg(not(feature = "parallel"))]
    for (s, mut d) in [(src_a, dst_a), (src_b, dst_b)] {
        for i in 0..s.rows() {
            for j in 0..s.cols() {
                d[(j, i)] = s[(i, j)];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    type Pair = (usize, usize);
    type Triple = (usize, usize, usize);

    /// Creates a rectangular `rows x cols` matrix stored as a flat vector.
    /// Each element `(i, j)` represents its row and column position.
    fn create_matrix(rows: usize, cols: usize) -> Vec<Pair> {
        (0..rows)
            .flat_map(|i| (0..cols).map(move |j| (i, j)))
            .collect()
    }

    /// Creates multiple matrices where each element is `(index, row, col)`.
    ///
    /// - `rows`: The number of rows.
    /// - `cols`: The number of columns (`None` means a square matrix where `cols = rows`).
    /// - `instances`: The number of matrices to generate.
    ///
    /// Each matrix has an identifier `index` and its elements store `(index, row, col)`.
    fn create_matrices(rows: usize, cols: usize, instances: usize) -> Vec<Vec<Triple>> {
        (0..instances)
            .map(|index| {
                (0..rows)
                    .flat_map(|row| (0..cols).map(move |col| (index, row, col)))
                    .collect()
            })
            .collect()
    }

    /// Asserts that `matrix` is correctly transposed.
    fn assert_transposed(matrix: &mut [Pair], rows: usize, cols: usize) {
        let view = MatrixMut::from_mut_slice(matrix, cols, rows);
        for i in 0..cols {
            for j in 0..rows {
                assert_eq!(view[(i, j)], (j, i), "Mismatch at ({i}, {j})");
            }
        }
    }

    #[test]
    fn test_transpose_square_swap_small() {
        for size in [2, 4] {
            let mut matrices = create_matrices(size, size, 2);
            let (matrix_a, matrix_b) = matrices.split_at_mut(1);

            let view_a = MatrixMut::from_mut_slice(&mut matrix_a[0], size, size);
            let view_b = MatrixMut::from_mut_slice(&mut matrix_b[0], size, size);

            // Perform swap transpose
            transpose_square_swap(view_a, view_b);

            // Verify swap was applied correctly
            let view_a = MatrixMut::from_mut_slice(&mut matrix_a[0], size, size);
            let view_b = MatrixMut::from_mut_slice(&mut matrix_b[0], size, size);
            for i in 0..size {
                for j in 0..size {
                    assert_eq!(view_a[(i, j)], (1, j, i), "Matrix A incorrect");
                    assert_eq!(view_b[(i, j)], (0, j, i), "Matrix B incorrect");
                }
            }
        }
    }

    #[test]
    fn test_transpose_square_swap_medium() {
        let size = 8;
        let mut matrices = create_matrices(size, size, 2);
        let (matrix_a, matrix_b) = matrices.split_at_mut(1);

        let view_a = MatrixMut::from_mut_slice(&mut matrix_a[0], size, size);
        let view_b = MatrixMut::from_mut_slice(&mut matrix_b[0], size, size);

        transpose_square_swap(view_a, view_b);

        let view_a = MatrixMut::from_mut_slice(&mut matrix_a[0], size, size);
        let view_b = MatrixMut::from_mut_slice(&mut matrix_b[0], size, size);
        for i in 0..size {
            for j in 0..size {
                assert_eq!(view_a[(i, j)], (1, j, i), "Matrix A incorrect");
                assert_eq!(view_b[(i, j)], (0, j, i), "Matrix B incorrect");
            }
        }
    }

    #[test]
    fn test_transpose_square_swap_large() {
        let size = 1024;
        assert!(size * size > 2 * workload_size::<Triple>());

        let mut matrices = create_matrices(size, size, 2);
        let (matrix_a, matrix_b) = matrices.split_at_mut(1);

        let view_a = MatrixMut::from_mut_slice(&mut matrix_a[0], size, size);
        let view_b = MatrixMut::from_mut_slice(&mut matrix_b[0], size, size);

        transpose_square_swap(view_a, view_b);

        let view_a = MatrixMut::from_mut_slice(&mut matrix_a[0], size, size);
        let view_b = MatrixMut::from_mut_slice(&mut matrix_b[0], size, size);
        for i in 0..size {
            for j in 0..size {
                assert_eq!(view_a[(i, j)], (1, j, i), "Matrix A incorrect");
                assert_eq!(view_b[(i, j)], (0, j, i), "Matrix B incorrect");
            }
        }
    }

    #[test]
    fn test_transpose_square_small() {
        for size in [2, 4] {
            let mut matrix = create_matrix(size, size);
            let view = MatrixMut::from_mut_slice(&mut matrix, size, size);

            transpose_square(view);

            assert_transposed(&mut matrix, size, size);
        }
    }

    #[test]
    fn test_transpose_square_medium() {
        let size = 8;
        let mut matrix = create_matrix(size, size);
        let view = MatrixMut::from_mut_slice(&mut matrix, size, size);

        transpose_square(view);

        assert_transposed(&mut matrix, size, size);
    }

    #[test]
    fn test_transpose_square_large() {
        let size = 1024;
        assert!(size * size > 2 * workload_size::<Triple>());

        let mut matrix = create_matrix(size, size);
        let view = MatrixMut::from_mut_slice(&mut matrix, size, size);

        transpose_square(view);

        assert_transposed(&mut matrix, size, size);
    }

    #[test]
    fn test_transpose_copy_small() {
        let rows = 2;
        let cols = 4;

        let mut src = create_matrix(rows, cols);
        let mut dst = vec![(0, 0); cols * rows];

        let src_view = MatrixMut::from_mut_slice(&mut src, rows, cols);
        let dst_view = MatrixMut::from_mut_slice(&mut dst, cols, rows);

        transpose_copy(src_view, dst_view);

        assert_transposed(&mut dst, rows, cols);
    }

    #[test]
    fn test_transpose_copy_medium() {
        let rows = 8;
        let cols = 16;

        let mut src = create_matrix(rows, cols);
        let mut dst = vec![(0, 0); cols * rows];

        let src_view = MatrixMut::from_mut_slice(&mut src, rows, cols);
        let dst_view = MatrixMut::from_mut_slice(&mut dst, cols, rows);

        transpose_copy(src_view, dst_view);

        assert_transposed(&mut dst, rows, cols);
    }

    #[test]
    fn test_transpose_copy_large() {
        let rows = 64;
        let cols = 128;

        let mut src = create_matrix(rows, cols);
        let mut dst = vec![(0, 0); cols * rows];

        let src_view = MatrixMut::from_mut_slice(&mut src, rows, cols);
        let dst_view = MatrixMut::from_mut_slice(&mut dst, cols, rows);

        transpose_copy(src_view, dst_view);

        assert_transposed(&mut dst, rows, cols);
    }

    #[test]
    fn test_transpose_copy_tall_matrix() {
        let rows = 32;
        let cols = 4;

        let mut src = create_matrix(rows, cols);
        let mut dst = vec![(0, 0); cols * rows];

        let src_view = MatrixMut::from_mut_slice(&mut src, rows, cols);
        let dst_view = MatrixMut::from_mut_slice(&mut dst, cols, rows);

        transpose_copy(src_view, dst_view);

        assert_transposed(&mut dst, rows, cols);
    }

    #[test]
    fn test_transpose_copy_wide_matrix() {
        let rows = 4;
        let cols = 32;

        let mut src = create_matrix(rows, cols);
        let mut dst = vec![(0, 0); cols * rows];

        let src_view = MatrixMut::from_mut_slice(&mut src, rows, cols);
        let dst_view = MatrixMut::from_mut_slice(&mut dst, cols, rows);

        transpose_copy(src_view, dst_view);

        assert_transposed(&mut dst, rows, cols);
    }

    #[test]
    fn test_transpose_copy_edge_case_double_rows() {
        let rows = 16;
        let cols = 8;

        let mut src = create_matrix(rows, cols);
        let mut dst = vec![(0, 0); cols * rows];

        let src_view = MatrixMut::from_mut_slice(&mut src, rows, cols);
        let dst_view = MatrixMut::from_mut_slice(&mut dst, cols, rows);

        transpose_copy(src_view, dst_view);

        assert_transposed(&mut dst, rows, cols);
    }

    #[test]
    fn test_transpose_copy_edge_case_double_cols() {
        let rows = 8;
        let cols = 16;

        let mut src = create_matrix(rows, cols);
        let mut dst = vec![(0, 0); cols * rows];

        let src_view = MatrixMut::from_mut_slice(&mut src, rows, cols);
        let dst_view = MatrixMut::from_mut_slice(&mut dst, cols, rows);

        transpose_copy(src_view, dst_view);

        assert_transposed(&mut dst, rows, cols);
    }

    #[test]
    fn test_transpose_small() {
        let size = 4;
        let mut matrix = create_matrix(size, size);
        transpose(&mut matrix, size, size);
        assert_transposed(&mut matrix, size, size);
    }

    #[test]
    fn test_transpose_large() {
        let size = 1024;
        assert!(size * size > 2 * workload_size::<Pair>());

        let mut matrix = create_matrix(size, size);
        transpose(&mut matrix, size, size);
        assert_transposed(&mut matrix, size, size);
    }

    #[test]
    fn test_transpose_rectangular_tall() {
        let rows = 32;
        let cols = 4;
        let mut matrix = create_matrix(rows, cols);

        transpose(&mut matrix, rows, cols);
        assert_transposed(&mut matrix, rows, cols);
    }

    #[test]
    fn test_transpose_rectangular_wide() {
        let rows = 4;
        let cols = 32;
        let mut matrix = create_matrix(rows, cols);

        transpose(&mut matrix, rows, cols);
        assert_transposed(&mut matrix, rows, cols);
    }

    #[test]
    fn test_transpose_multiple_matrices() {
        let num_matrices = 10;
        let rows = 8;
        let cols = 16;

        // Create matrices and flatten them into a single vector
        let mut matrices: Vec<Triple> = create_matrices(rows, cols, num_matrices)
            .into_iter()
            .flatten()
            .collect();

        transpose(&mut matrices, rows, cols);

        for index in 0..num_matrices {
            let view = MatrixMut::from_mut_slice(
                &mut matrices[index * rows * cols..(index + 1) * rows * cols],
                cols,
                rows,
            );
            for i in 0..cols {
                for j in 0..rows {
                    assert_eq!(
                        view[(i, j)],
                        (index, j, i),
                        "Mismatch at ({i}, {j}) in matrix {index}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_transpose_edge_case_double_rows() {
        let rows = 16;
        let cols = 8;
        let mut matrix = create_matrix(rows, cols);

        transpose(&mut matrix, rows, cols);

        assert_transposed(&mut matrix, rows, cols);
    }

    #[test]
    fn test_transpose_edge_case_double_cols() {
        let rows = 8;
        let cols = 16;
        let mut matrix = create_matrix(rows, cols);

        transpose(&mut matrix, rows, cols);
        assert_transposed(&mut matrix, rows, cols);
    }

    #[test]
    fn test_transpose_square_multiple_matrices() {
        let num_matrices = 5;
        let size = 64;

        // Create matrices and flatten them into a single vector
        let mut matrices: Vec<Triple> = create_matrices(size, size, num_matrices)
            .into_iter()
            .flatten()
            .collect();

        transpose(&mut matrices, size, size);

        for index in 0..num_matrices {
            let view = MatrixMut::from_mut_slice(
                &mut matrices[index * size * size..(index + 1) * size * size],
                size,
                size,
            );
            for i in 0..size {
                for j in 0..size {
                    assert_eq!(
                        view[(i, j)],
                        (index, j, i),
                        "Mismatch at ({i}, {j}) in matrix {index}"
                    );
                }
            }
        }
    }

    /// Generates random square matrices with sizes that are powers of two.
    #[allow(clippy::cast_sign_loss)]
    fn arb_square_matrix() -> impl Strategy<Value = (Vec<usize>, usize)> {
        (2usize..=64)
            .prop_filter("Must be power of two", |&size| size.is_power_of_two())
            .prop_map(|size| size * size)
            .prop_flat_map(|matrix_size| {
                prop::collection::vec(0usize..1000, matrix_size)
                    .prop_map(move |matrix| (matrix, (matrix_size as f64).sqrt() as usize))
            })
    }

    /// Generates random rectangular matrices where rows and columns are powers of two.
    fn arb_rect_matrix() -> impl Strategy<Value = (Vec<usize>, usize, usize)> {
        (2usize..=64, 2usize..=64)
            .prop_filter("Rows and columns must be power of two", |&(r, c)| {
                r.is_power_of_two() && c.is_power_of_two()
            })
            .prop_flat_map(|(rows, cols)| {
                prop::collection::vec(0usize..1000, rows * cols)
                    .prop_map(move |matrix| (matrix, rows, cols))
            })
    }

    proptest! {
        #[test]
        fn proptest_transpose_square((mut matrix, size) in arb_square_matrix()) {
            let original = matrix.clone();
            transpose(&mut matrix, size, size);
            transpose(&mut matrix, size, size);
            prop_assert_eq!(matrix, original);
        }

        #[test]
        fn proptest_transpose_rect((mut matrix, rows, cols) in arb_rect_matrix()) {
            let original = matrix.clone();
            transpose(&mut matrix, rows, cols);

            let view = MatrixMut::from_mut_slice(&mut matrix, cols, rows);

            // Verify that each (i, j) moved to (j, i)
            for i in 0..cols {
                for j in 0..rows {
                    prop_assert_eq!(view[(i, j)], original[j * cols + i]);
                }
            }
        }
    }
}
