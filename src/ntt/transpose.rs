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

#[cfg(test)]
mod tests {
    use super::*;

    type Pair = (usize, usize);
    type Triple = (usize, usize, usize);

    /// Creates an `NxN` matrix with elements `(row, col)`, useful for testing purposes.
    fn create_debug_matrix(rows: usize) -> Vec<Pair> {
        (0..rows).flat_map(|i| (0..rows).map(move |j| (i, j))).collect()
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
            let mut matrix = create_debug_matrix(size);
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
        let mut matrix = create_debug_matrix(size);
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

        let mut matrix = create_debug_matrix(size);
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
