use super::matrix::MatrixMut;
use crate::{ntt::utils::workload_size, utils::is_power_of_two};
use rayon::join;
use std::mem::swap;

fn transpose_square_swap_parallel<F: Sized + Send>(
    mut a: MatrixMut<'_, F>,
    mut b: MatrixMut<'_, F>,
) {
    debug_assert!(a.is_square());
    debug_assert_eq!(a.rows(), b.cols());
    debug_assert_eq!(a.cols(), b.rows());
    debug_assert!(is_power_of_two(a.rows()));
    debug_assert!(workload_size::<F>() >= 2); // otherwise, we would recurse even if size == 1.
    let size = a.rows();
    if 2 * size * size > workload_size::<F>() {
        // Recurse into quadrants.
        // This results in a cache-oblivious algorithm.
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
        for i in 0..size {
            for j in 0..size {
                swap(&mut a[(i, j)], &mut b[(j, i)]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Triple = (usize, usize, usize);

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
}
