use super::matrix::MatrixMut;
use crate::{ntt::utils::workload_size, utils::is_power_of_two};
use rayon::join;
use std::mem::swap;

/// Transpose and swap two square size matrices (parallel version).
///
/// The size must be a power of two.
pub fn transpose_square_swap_parallel<F: Sized + Send>(
    mut a: MatrixMut<'_, F>,
    mut b: MatrixMut<'_, F>,
) {
    debug_assert!(a.is_square());
    debug_assert_eq!(a.rows(), b.cols());
    debug_assert_eq!(a.cols(), b.rows());
    debug_assert!(is_power_of_two(a.rows()));
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

        join(
            || {
                join(
                    || transpose_square_swap_parallel(aa, ba),
                    || transpose_square_swap_parallel(ab, bc),
                )
            },
            || {
                join(
                    || transpose_square_swap_parallel(ac, bb),
                    || transpose_square_swap_parallel(ad, bd),
                )
            },
        );
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
fn transpose_square_parallel<F: Sized + Send>(mut m: MatrixMut<'_, F>) {
    debug_assert!(m.is_square());
    debug_assert!(m.rows().is_power_of_two());
    let size = m.rows();
    if size * size > workload_size::<F>() {
        // Recurse into quadrants.
        // This results in a cache-oblivious algorithm.
        let n = size / 2;
        let (a, b, c, d) = m.split_quadrants(n, n);

        join(
            || transpose_square_swap_parallel(b, c),
            || join(|| transpose_square_parallel(a), || transpose_square_parallel(d)),
        );
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

/// Sets `dst` to the transpose of `src`. This will panic if the sizes of `src` and `dst` are not
/// compatible.
fn transpose_copy_parallel<F: Sized + Copy + Send>(
    src: MatrixMut<'_, F>,
    mut dst: MatrixMut<'_, F>,
) {
    assert_eq!(src.rows(), dst.cols());
    assert_eq!(src.cols(), dst.rows());
    if src.rows() * src.cols() > workload_size::<F>() {
        // Split along longest axis and recurse.
        // This results in a cache-oblivious algorithm.
        let ((a, b), (x, y)) = if src.rows() > src.cols() {
            let n = src.rows() / 2;
            (src.split_vertical(n), dst.split_horizontal(n))
        } else {
            let n = src.cols() / 2;
            (src.split_horizontal(n), dst.split_vertical(n))
        };
        join(|| transpose_copy_parallel(a, x), || transpose_copy_parallel(b, y));
    } else {
        for i in 0..src.rows() {
            for j in 0..src.cols() {
                dst[(j, i)] = src[(i, j)];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Pair = (usize, usize);
    type Triple = (usize, usize, usize);

    /// Creates a rectangular `rows x cols` matrix stored as a flat vector.
    /// Each element `(i, j)` represents its row and column position.
    fn create_debug_matrix(rows: usize, cols: usize) -> Vec<Pair> {
        (0..rows).flat_map(|i| (0..cols).map(move |j| (i, j))).collect()
    }

    /// Creates `N x N` matrices where each element is `(index, row, col)`.
    ///
    /// - `rows`: The number of rows (and columns, since it's a square matrix).
    /// - `instances`: The number of matrices to generate.
    ///
    /// Each matrix has an identifier `index` and its elements store `(index, row, col)`.
    fn create_example_matrices(rows: usize, instances: usize) -> Vec<Vec<Triple>> {
        let mut matrices = Vec::new();

        for index in 0..instances {
            let mut matrix = Vec::new();
            for row in 0..rows {
                for col in 0..rows {
                    matrix.push((index, row, col));
                }
            }
            matrices.push(matrix);
        }

        matrices
    }

    #[test]
    fn test_transpose_square_swap_small() {
        for size in [2, 4] {
            let mut matrices = create_example_matrices(size, 2);
            let (matrix_a, matrix_b) = matrices.split_at_mut(1);

            let view_a = MatrixMut::from_mut_slice(&mut matrix_a[0], size, size);
            let view_b = MatrixMut::from_mut_slice(&mut matrix_b[0], size, size);

            // Perform swap transpose
            transpose_square_swap_parallel(view_a, view_b);

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
        let mut matrices = create_example_matrices(size, 2);
        let (matrix_a, matrix_b) = matrices.split_at_mut(1);

        let view_a = MatrixMut::from_mut_slice(&mut matrix_a[0], size, size);
        let view_b = MatrixMut::from_mut_slice(&mut matrix_b[0], size, size);

        transpose_square_swap_parallel(view_a, view_b);

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

        let mut matrices = create_example_matrices(size, 2);
        let (matrix_a, matrix_b) = matrices.split_at_mut(1);

        let view_a = MatrixMut::from_mut_slice(&mut matrix_a[0], size, size);
        let view_b = MatrixMut::from_mut_slice(&mut matrix_b[0], size, size);

        transpose_square_swap_parallel(view_a, view_b);

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
    fn test_transpose_square_parallel_small() {
        for size in [2, 4] {
            let mut matrix = create_debug_matrix(size, size);
            let view = MatrixMut::from_mut_slice(&mut matrix, size, size);

            transpose_square_parallel(view);

            let view = MatrixMut::from_mut_slice(&mut matrix, size, size);
            for i in 0..size {
                for j in 0..size {
                    assert_eq!(view[(i, j)], (j, i), "Matrix incorrect");
                }
            }
        }
    }

    #[test]
    fn test_transpose_square_parallel_medium() {
        let size = 8;
        let mut matrix = create_debug_matrix(size, size);
        let view = MatrixMut::from_mut_slice(&mut matrix, size, size);

        transpose_square_parallel(view);

        let view = MatrixMut::from_mut_slice(&mut matrix, size, size);
        for i in 0..size {
            for j in 0..size {
                assert_eq!(view[(i, j)], (j, i), "Matrix incorrect");
            }
        }
    }

    #[test]
    fn test_transpose_square_parallel_large() {
        let size = 1024;
        assert!(size * size > 2 * workload_size::<Triple>());

        let mut matrix = create_debug_matrix(size, size);
        let view = MatrixMut::from_mut_slice(&mut matrix, size, size);

        transpose_square_parallel(view);

        let view = MatrixMut::from_mut_slice(&mut matrix, size, size);
        for i in 0..size {
            for j in 0..size {
                assert_eq!(view[(i, j)], (j, i), "Matrix incorrect");
            }
        }
    }

    #[test]
    fn test_transpose_copy_parallel_small() {
        let rows = 2;
        let cols = 4;

        let mut src = create_debug_matrix(rows, cols);
        let mut dst = vec![(0, 0); cols * rows];

        let src_view = MatrixMut::from_mut_slice(&mut src, rows, cols);
        let dst_view = MatrixMut::from_mut_slice(&mut dst, cols, rows);

        transpose_copy_parallel(src_view, dst_view);

        // Check the transposed output
        let dst_view = MatrixMut::from_mut_slice(&mut dst, cols, rows);
        for i in 0..rows {
            for j in 0..cols {
                assert_eq!(dst_view[(j, i)], (i, j), "Mismatch at ({}, {})", j, i);
            }
        }
    }

    #[test]
    fn test_transpose_copy_parallel_medium() {
        let rows = 8;
        let cols = 16;

        let mut src = create_debug_matrix(rows, cols);
        let mut dst = vec![(0, 0); cols * rows];

        let src_view = MatrixMut::from_mut_slice(&mut src, rows, cols);
        let dst_view = MatrixMut::from_mut_slice(&mut dst, cols, rows);

        transpose_copy_parallel(src_view, dst_view);

        let dst_view = MatrixMut::from_mut_slice(&mut dst, cols, rows);
        for i in 0..rows {
            for j in 0..cols {
                assert_eq!(dst_view[(j, i)], (i, j), "Mismatch at ({}, {})", j, i);
            }
        }
    }

    #[test]
    fn test_transpose_copy_parallel_large() {
        let rows = 64;
        let cols = 128;

        let mut src = create_debug_matrix(rows, cols);
        let mut dst = vec![(0, 0); cols * rows];

        let src_view = MatrixMut::from_mut_slice(&mut src, rows, cols);
        let dst_view = MatrixMut::from_mut_slice(&mut dst, cols, rows);

        transpose_copy_parallel(src_view, dst_view);

        let dst_view = MatrixMut::from_mut_slice(&mut dst, cols, rows);
        for i in 0..rows {
            for j in 0..cols {
                assert_eq!(dst_view[(j, i)], (i, j), "Mismatch at ({}, {})", j, i);
            }
        }
    }

    #[test]
    fn test_transpose_copy_parallel_tall_matrix() {
        let rows = 32;
        let cols = 4;

        let mut src = create_debug_matrix(rows, cols);
        let mut dst = vec![(0, 0); cols * rows];

        let src_view = MatrixMut::from_mut_slice(&mut src, rows, cols);
        let dst_view = MatrixMut::from_mut_slice(&mut dst, cols, rows);

        transpose_copy_parallel(src_view, dst_view);

        let dst_view = MatrixMut::from_mut_slice(&mut dst, cols, rows);
        for i in 0..rows {
            for j in 0..cols {
                assert_eq!(dst_view[(j, i)], (i, j), "Mismatch at ({}, {})", j, i);
            }
        }
    }

    #[test]
    fn test_transpose_copy_parallel_wide_matrix() {
        let rows = 4;
        let cols = 32;

        let mut src = create_debug_matrix(rows, cols);
        let mut dst = vec![(0, 0); cols * rows];

        let src_view = MatrixMut::from_mut_slice(&mut src, rows, cols);
        let dst_view = MatrixMut::from_mut_slice(&mut dst, cols, rows);

        transpose_copy_parallel(src_view, dst_view);

        let dst_view = MatrixMut::from_mut_slice(&mut dst, cols, rows);
        for i in 0..rows {
            for j in 0..cols {
                assert_eq!(dst_view[(j, i)], (i, j), "Mismatch at ({}, {})", j, i);
            }
        }
    }

    #[test]
    fn test_transpose_copy_parallel_edge_case_double_rows() {
        let rows = 16;
        let cols = 8;

        let mut src = create_debug_matrix(rows, cols);
        let mut dst = vec![(0, 0); cols * rows];

        let src_view = MatrixMut::from_mut_slice(&mut src, rows, cols);
        let dst_view = MatrixMut::from_mut_slice(&mut dst, cols, rows);

        transpose_copy_parallel(src_view, dst_view);

        let dst_view = MatrixMut::from_mut_slice(&mut dst, cols, rows);
        for i in 0..rows {
            for j in 0..cols {
                assert_eq!(dst_view[(j, i)], (i, j), "Mismatch at ({}, {})", j, i);
            }
        }
    }

    #[test]
    fn test_transpose_copy_parallel_edge_case_double_cols() {
        let rows = 8;
        let cols = 16;

        let mut src = create_debug_matrix(rows, cols);
        let mut dst = vec![(0, 0); cols * rows];

        let src_view = MatrixMut::from_mut_slice(&mut src, rows, cols);
        let dst_view = MatrixMut::from_mut_slice(&mut dst, cols, rows);

        transpose_copy_parallel(src_view, dst_view);

        let dst_view = MatrixMut::from_mut_slice(&mut dst, cols, rows);
        for i in 0..rows {
            for j in 0..cols {
                assert_eq!(dst_view[(j, i)], (i, j), "Mismatch at ({}, {})", j, i);
            }
        }
    }
}
