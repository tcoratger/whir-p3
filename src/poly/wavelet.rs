//! This module implements the wavelet transform which converts between the hypercube evaluation
//! and coefficient domains of multilinear polynomials.
//!
//! Lets focus on the case where we have `3` variables. Then a multilinear polynomial is a polynomial of the form
//! ```text
//!     p(x_1, x_2, x_3) = p_0 + p_1 * x_1 + p_2 * x_2 + p_3 * x_1 * x_2 + p_4 * x_3 +
//!                             p_5 * x_1 * x_3 + p_6 * x_2 * x_3 + p_7 * x_1 * x_2 * x_3
//!````
//! and it's evaluations on the hypercube are given by
//! ```text
//!     p(0, 0, 0) = p_0
//!     p(1, 0, 0) = p_0 + p_1
//!     p(0, 1, 0) = p_0 + p_2
//!     p(1, 1, 0) = p_0 + p_1 + p_2 + p_3
//!     p(0, 0, 1) = p_0 + p_4
//!     p(1, 0, 1) = p_0 + p_1 + p_4 + p_5
//!     p(0, 1, 1) = p_0 + p_2 + p_4 + p_6
//!     p(1, 1, 1) = p_0 + p_1 + p_2 + p_3 + p_4 + p_5 + p_6 + p_7
//! ```
//!
//! The idea of the wavelet transform is to compute this via an FFT-like divide and conquer algorithm. Starting
//! with the evaluation vector `p = [p0, p_1, p_2, p_3, p_4, p_5, p_6, p_7]` we apply the kernel
//! ```text
//!     [1 0]
//!     [1 1]
//! ```
//! On chunks of sizes `2, 4` and `8` respectively which looks like
//! ```text
//!    [p_0, p_1, p_2, p_3, p_4, p_5, p_6, p_7] -> [p_0, p_0 + p_1, p_2, p_2 + p_3, p_4, p_4 + p_5, p_6, p_6 + p_7]
//!                                             -> [p_0, p_0 + p_1, p_0 + p_2, p_0 + p_1 + p_2 + p_3,
//!                                                     p_4, p_4 + p_5, p_4 + p_6, p_4 + p_5 + p_6 + p_7]
//!                                             -> [p_0, p_0 + p_1, p_0 + p_2, p_0 + p_1 + p_2 + p_3,
//!                                                     p_0 + p_4, p_0 + p_1 + p_4 + p_5, p_0 + p_2 + p_4 + p_6,
//!                                                     p_0 + p_1 + p_2 + p_3 + p_4 + p_5 + p_6 + p_7]
//! ```
//! As can be easily seen, in each pass we do `N/2` additions (where `N = 2^{num_variables} = |vec|`)
//! leading to an algebraic complexity of `N log(N)/2`.
//!
//! The inverse transform is essentially identical except it uses the inverse kernel
//! ```text
//!     [1  0]
//!     [-1 1]
//! ```
//! and applies the rounds in the reverse order.

use std::marker::PhantomData;

use p3_field::{BasedVectorSpace, Field, PackedValue};
use p3_matrix::{
    Matrix,
    dense::{RowMajorMatrix, RowMajorMatrixViewMut},
    util::reverse_matrix_index_bits,
};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::whir::utils::workload_size;

/// A kernel which converts between the hypercube evaluation and coefficient domains for
/// a multilinear polynomial.
///
/// This works for both a single polynomial as well as
/// a batch of polynomials given as a `RowMajorMatrix`. Moreover, as this transformation
/// is linear, it supports polynomials with coefficients/evaluations in any
/// vector space `V` over the field `F`. In general, given a polynomial in an extension
/// field `EF`, using `Radix2WaveletKernel<F>` should be preferred over `Radix2WaveletKernel<EF>`
/// as the former will be faster.
///
/// This converts between these domains using an FFT-like transform with to `N log(N)/2` algebraic complexity.
#[derive(Default, Clone, Debug)]
pub struct Radix2WaveletKernel<F: Field> {
    _phantom: PhantomData<F>,
}

impl<F: Field> Radix2WaveletKernel<F> {
    /// Convert from a vector of multilinear coefficients to a vector of hypercube evaluations.
    ///
    /// # Panics
    /// Panics if the number of rows is not a power of two.
    #[must_use]
    pub fn wavelet_transform(&self, vec: Vec<F>) -> Vec<F> {
        let mat = RowMajorMatrix::new_col(vec);
        self.wavelet_transform_batch(mat).values
    }

    /// Convert every column of the matrix from multilinear coefficients to hypercube evaluations.
    ///
    /// This is the inverse of `inverse_wavelet_transform_batch` and it's implementation is similar.
    /// - Decompose as much work as possible into per-core chunks for efficient parallelization.
    /// - Use bit-reversal to maximize SIMD-friendly access patterns on each core.
    /// - Apply remaining global rounds using further parallelization.
    /// # Panics
    /// Panics if the number of rows is not a power of two.
    #[instrument(skip_all, level = "info", fields(
        num_rows = mat.height(),
        num_cols = mat.width(),
    ))]
    #[must_use]
    pub fn wavelet_transform_batch(&self, mut mat: RowMajorMatrix<F>) -> RowMajorMatrix<F> {
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
            par_wavelet_kernel(&mut mat, block_size);
        }

        mat
    }

    /// Convert from a vector of multilinear coefficients to a vector of hypercube evaluations.
    ///
    /// This flattens to the base field and applies the base field wavelet transform before
    /// reconstituting back to the original vector space elements. This is valid as the
    /// wavelet transform is linear.
    ///
    /// # Panics
    /// Panics if the number of rows is not a power of two.
    #[must_use]
    pub fn wavelet_transform_algebra<V: BasedVectorSpace<F> + Clone + Send + Sync>(
        &self,
        vec: Vec<V>,
    ) -> Vec<V> {
        let mat = RowMajorMatrix::new_col(vec);
        self.wavelet_transform_algebra_batch(mat).values
    }

    /// Convert every column of the matrix from multilinear coefficients to hypercube evaluations.
    ///
    /// This flattens to the base field and applies the base field wavelet transform before
    /// reconstituting back to the original vector space elements. This is valid as the
    /// wavelet transform is linear.
    ///
    /// # Panics
    /// Panics if the number of rows is not a power of two.
    #[must_use]
    pub fn wavelet_transform_algebra_batch<V: BasedVectorSpace<F> + Clone + Send + Sync>(
        &self,
        mat: RowMajorMatrix<V>,
    ) -> RowMajorMatrix<V> {
        let init_width = mat.width();
        let base_mat =
            RowMajorMatrix::new(V::flatten_to_base(mat.values), init_width * V::DIMENSION);
        let base_dft_output = self.wavelet_transform_batch(base_mat);
        RowMajorMatrix::new(
            V::reconstitute_from_base(base_dft_output.values),
            init_width,
        )
    }

    /// Convert from a vector of hypercube evaluations to a vector of multilinear coefficients.
    ///
    /// # Panics
    /// Panics if the number of rows is not a power of two.
    #[must_use]
    fn inverse_wavelet_transform(&self, vec: Vec<F>) -> Vec<F> {
        let mat = RowMajorMatrix::new_col(vec);
        self.inverse_wavelet_transform_batch(mat).values
    }

    /// Convert every column of the matrix from hypercube evaluations to multilinear coefficients.
    ///
    /// This is the inverse of `wavelet_transform_batch` and it's implementation is similar.
    /// - Decompose as much work as possible into per-core chunks for efficient parallelization.
    /// - Use bit-reversal to maximize SIMD-friendly access patterns on each core.
    /// - Apply remaining global rounds using further parallelization.
    ///
    /// # Panics
    /// Panics if the number of rows is not a power of two.
    #[instrument(skip_all, level = "debug", fields(
        num_rows = mat.height(),
        num_cols = mat.width(),
    ))]
    #[must_use]
    pub fn inverse_wavelet_transform_batch(&self, mut mat: RowMajorMatrix<F>) -> RowMajorMatrix<F> {
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
            par_inverse_wavelet_kernel(&mut mat, block_size);
        }

        mat
    }

    /// Convert from a vector of hypercube evaluations to a vector of multilinear coefficients.
    ///
    /// This flattens to the base field and applies the base field wavelet transform before
    /// reconstituting back to the original vector space elements. This is valid as the
    /// wavelet transform is linear.
    ///
    /// # Panics
    /// Panics if the number of rows is not a power of two.
    #[must_use]
    pub fn inverse_wavelet_transform_algebra<V: BasedVectorSpace<F> + Clone + Send + Sync>(
        &self,
        vec: Vec<V>,
    ) -> Vec<V> {
        let mat = RowMajorMatrix::new_col(vec);
        self.inverse_wavelet_transform_algebra_batch(mat).values
    }

    /// Convert every column of the matrix from hypercube evaluations to multilinear coefficients.
    ///
    /// This flattens to the base field and applies the base field wavelet transform before
    /// reconstituting back to the original vector space elements. This is valid as the
    /// wavelet transform is linear.
    ///
    /// # Panics
    /// Panics if the number of rows is not a power of two.
    #[must_use]
    pub fn inverse_wavelet_transform_algebra_batch<V: BasedVectorSpace<F> + Clone + Send + Sync>(
        &self,
        mat: RowMajorMatrix<V>,
    ) -> RowMajorMatrix<V> {
        let init_width = mat.width();
        let base_mat =
            RowMajorMatrix::new(V::flatten_to_base(mat.values), init_width * V::DIMENSION);
        let base_dft_output = self.inverse_wavelet_transform_batch(base_mat);
        RowMajorMatrix::new(
            V::reconstitute_from_base(base_dft_output.values),
            init_width,
        )
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
fn par_wavelet_kernel<F: Field>(mat: &mut RowMajorMatrix<F>, block_size: usize) {
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
fn par_inverse_wavelet_kernel<F: Field>(mat: &mut RowMajorMatrix<F>, block_size: usize) {
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
    use p3_baby_bear::BabyBear;
    use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, extension::BinomialExtensionField};
    use p3_matrix::dense::RowMajorMatrix;
    use proptest::prelude::*;

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    #[test]
    fn test_wavelet_transform_single_element() {
        let values = vec![F::from_u8(5)];
        let kernel = Radix2WaveletKernel::default();

        let evals = kernel.wavelet_transform(values);
        assert_eq!(evals, vec![F::from_u8(5)]);
    }

    #[test]
    fn test_wavelet_transform_size_2() {
        let v1 = F::from_u8(3);
        let v2 = F::from_u8(7);
        let values = vec![v1, v2];
        let kernel = Radix2WaveletKernel::default();

        let evals = kernel.wavelet_transform(values);
        assert_eq!(evals, vec![v1, v1 + v2]);
    }

    #[test]
    fn test_wavelet_transform_size_4() {
        let v1 = F::ONE;
        let v2 = F::TWO;
        let v3 = F::ONE + F::TWO;
        let v4 = F::TWO.double();
        let values = vec![v1, v2, v3, v4];

        let kernel = Radix2WaveletKernel::default();
        let evals = kernel.wavelet_transform(values);

        assert_eq!(evals, vec![v1, v1 + v2, v3 + v1, v1 + v2 + v3 + v4]);
    }

    #[test]
    fn test_wavelet_transform_size_4x2() {
        let values = (1..=8).map(F::from_u64).collect::<Vec<_>>();
        let v1 = values[0];
        let v2 = values[1];
        let v3 = values[2];
        let v4 = values[3];
        let v5 = values[4];
        let v6 = values[5];
        let v7 = values[6];
        let v8 = values[7];
        let mat = RowMajorMatrix::new(values, 2);

        let kernel = Radix2WaveletKernel::default();
        let evals = kernel.wavelet_transform_batch(mat);

        assert_eq!(
            evals.values,
            vec![
                v1,
                v2,
                v1 + v3,
                v2 + v4,
                v1 + v5,
                v2 + v6,
                v1 + v3 + v5 + v7,
                v2 + v4 + v6 + v8
            ]
        );
    }

    #[test]
    fn test_wavelet_transform_extension() {
        let size = 2_i32.pow(16) as u32;
        let values = (1..=size)
            .map(|i| {
                EF::from_basis_coefficients_iter((0..4).map(|j| F::from_u32(4 * i + j))).unwrap()
            })
            .collect::<Vec<_>>();

        let base_kernel = Radix2WaveletKernel::<F>::default();
        let ef_kernel = Radix2WaveletKernel::<EF>::default();

        let evals1 = base_kernel.wavelet_transform_algebra(values.clone());
        let evals2 = ef_kernel.wavelet_transform(values);

        assert_eq!(evals1, evals2);
    }

    #[test]
    fn test_wavelet_roundtrip() {
        let values = (1..=32).map(F::from_u64).collect::<Vec<_>>();
        let original = values.clone();

        let kernel = Radix2WaveletKernel::default();

        let evaluations = kernel.wavelet_transform(values);
        let coefficients = kernel.inverse_wavelet_transform(evaluations);

        assert_eq!(coefficients, original);
    }

    #[test]
    fn test_wavelet_batch_roundtrip() {
        let values = (1..=(32 * 11)).map(F::from_u64).collect::<Vec<_>>();
        let original = values.clone();

        // We use `width = 11` as a nice small prime which should detect any issues with packing/suffixes.
        let mat = RowMajorMatrix::new(values, 11);
        let kernel = Radix2WaveletKernel::default();

        let evaluations = kernel.wavelet_transform_batch(mat);
        let coefficients = kernel.inverse_wavelet_transform_batch(evaluations);

        assert_eq!(coefficients.values, original);
    }

    #[test]
    fn test_wavelet_extension_roundtrip() {
        let size = 2_i32.pow(16) as u32;
        let values = (1..=size)
            .map(|i| {
                EF::from_basis_coefficients_iter((0..4).map(|j| F::from_u32(4 * i + j))).unwrap()
            })
            .collect::<Vec<_>>();

        let original = values.clone();

        let kernel = Radix2WaveletKernel::<F>::default();

        let evaluations = kernel.wavelet_transform_algebra(values);
        let coefficients = kernel.inverse_wavelet_transform_algebra(evaluations);

        assert_eq!(coefficients, original);
    }

    #[test]
    fn test_wavelet_extension_batch_roundtrip() {
        let size = 2_i32.pow(16) as u32;
        let values = (1..=(size * 11))
            .map(|i| {
                EF::from_basis_coefficients_iter((0..4).map(|j| F::from_u32(4 * i + j))).unwrap()
            })
            .collect::<Vec<_>>();

        let original = values.clone();

        // We use `width = 11` as a nice small prime which should detect any issues with packing/suffixes.
        let mat = RowMajorMatrix::new(values, 11);
        let kernel = Radix2WaveletKernel::<F>::default();

        let evaluations = kernel.wavelet_transform_algebra_batch(mat);
        let coefficients = kernel.inverse_wavelet_transform_algebra_batch(evaluations);

        assert_eq!(coefficients.values, original);
    }

    #[test]
    fn test_wavelet_roundtrip_large() {
        let values = (1..=(1 << 20)).map(F::from_u64).collect::<Vec<_>>();
        let original = values.clone();

        let kernel = Radix2WaveletKernel::default();

        let evaluations = kernel.wavelet_transform(values);
        let coefficients = kernel.inverse_wavelet_transform(evaluations);

        assert_eq!(coefficients, original, "Roundtrip failed");
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

            // Initialize the wavelet kernel
            let kernel = Radix2WaveletKernel::default();

            // Apply the forward wavelet transform
            let evals = kernel.wavelet_transform(values);

            // Apply the inverse wavelet transform
            let coeffs = kernel.inverse_wavelet_transform(evals);

            // Ensure roundtrip: output matches original input
            prop_assert_eq!(coeffs, original);
        }
    }
}
