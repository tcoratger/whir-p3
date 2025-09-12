/* DFT (Discrete Fourier Transform) on "evaluations".

Often, the polynomial used in the PIOP is represented by its evaluations on the boolean hypercube.
It turns out we also need this representation in the Sumcheck of WHIR.

When the prover must "Reed Solomon" encode a multilinear polynomial `P(x_1, ..., x_n)`,
i.e compute `P(α, α², α⁴, ..., α^(2^(n-1)))` for every `α` such that `α^(2^(n + log_inv_rate)) = 1`,
the more straightforward approach is to convert the polynomial represented by its evals to
the coefficients representation (canonical basis), and then to apply a well known DFT algorithm.

However this approach is not the most efficient because the conversion evals -> coeffs is `n * log(n)`.

To avoid dealing with the coeffs, we can directly perform the DFT on the evals, using the fact that:
```text
    P(α, α², α⁴, ..., α^(2^(n-1))) = (1-α) * P(0, α², α⁴, ..., α^(2^(n-1))) + α * P(1, α², α⁴, ..., α^(2^(n-1)))
                = P(0, α², α⁴, ..., α^(2^(n-1))) + α * (P(1, α², α⁴, ..., α^(2^(n-1))) - P(0, α², α⁴, ..., α^(2^(n-1))))
```

As a result, the algorithm we use is not the standard one and the twiddles look quite different.

Credits: https://github.com/Plonky3/Plonky3 (radix_2_small_batch.rs)
(the main difference is in `TwiddleFreeButterfly` and `DitButterfly`)
*/

use std::cell::RefCell;

use itertools::Itertools;
use p3_dft::Butterfly;
use p3_field::{BasedVectorSpace, Field, PackedField, TwoAdicField};
use p3_matrix::{
    Matrix,
    dense::{RowMajorMatrix, RowMajorMatrixViewMut},
    util::reverse_matrix_index_bits,
};
use p3_maybe_rayon::prelude::*;
use p3_util::{log2_strict_usize, reverse_slice_index_bits};

/// The number of layers to compute in each parallelization.
const LAYERS_PER_GROUP: usize = 3;

/// A specialized Discrete Fourier Transform (DFT) implementation for a univariate polynomial
/// stored as a set of multivariate evaluations.
///
/// This struct provides efficient DFT computation conversion to coefficient representation.
#[derive(Default, Clone, Debug)]
pub struct EvalsDft<F> {
    twiddles: RefCell<Vec<Vec<F>>>,
}

impl<F: TwoAdicField> EvalsDft<F> {
    /// Create a new `EvalsDft` instance with precomputed twiddles for the given size.
    ///
    /// The input `n` should be a power of two, representing the maximal FFT size you expect to handle.
    #[must_use]
    pub fn new(n: usize) -> Self {
        let res = Self {
            twiddles: RefCell::default(),
        };
        res.update_twiddles(n);
        res
    }

    fn roots_of_unity_table(n: usize) -> Vec<Vec<F>> {
        let lg_n = log2_strict_usize(n);
        let generator = F::two_adic_generator(lg_n);
        let half_n = 1 << (lg_n - 1);
        // nth_roots = [1, g, g^2, g^3, ..., g^{n/2 - 1}]
        let nth_roots = generator.powers().collect_n(half_n);

        (0..lg_n)
            .map(|i| nth_roots.iter().step_by(1 << i).copied().collect())
            .collect()
    }

    /// Compute twiddle and inv_twiddle factors, or take memoized ones if already available.
    pub fn update_twiddles(&self, fft_len: usize) {
        // TODO: This recomputes the entire table from scratch if we
        // need it to be larger, which is wasteful.

        // roots_of_unity_table(fft_len) returns a vector of twiddles of length log_2(fft_len).
        let curr_max_fft_len = 1 << self.twiddles.borrow().len();
        if fft_len > curr_max_fft_len {
            let mut new_twiddles = Self::roots_of_unity_table(fft_len);
            for ts in &mut new_twiddles {
                reverse_slice_index_bits(ts);
            }

            self.twiddles.replace(new_twiddles);
        }
    }

    pub fn dft_algebra_batch_by_evals<V: BasedVectorSpace<F> + Clone + Send + Sync>(
        &self,
        mat: RowMajorMatrix<V>,
    ) -> RowMajorMatrix<V> {
        let init_width = mat.width();
        let base_mat =
            RowMajorMatrix::new(V::flatten_to_base(mat.values), init_width * V::DIMENSION);
        let base_dft_output = self.dft_batch_by_evals(base_mat).to_row_major_matrix();
        RowMajorMatrix::new(
            V::reconstitute_from_base(base_dft_output.values),
            init_width,
        )
    }

    pub fn dft_batch_by_evals(&self, mut mat: RowMajorMatrix<F>) -> RowMajorMatrix<F> {
        let h = mat.height();
        let w = mat.width();
        let log_h = log2_strict_usize(h);

        self.update_twiddles(h);
        let root_table = self.twiddles.borrow();
        let len = root_table.len();
        let root_table = &root_table[len - log_h..];

        // The strategy will be to do a standard round-by-round parallelization
        // until the chunk size is smaller than `num_par_rows * mat.width()` after which we
        // send `num_par_rows` chunks to each thread and do the remainder of the
        // fft without transferring any more data between threads.
        let num_par_rows = estimate_num_rows_in_l1::<F>(h, w);
        let log_num_par_rows = log2_strict_usize(num_par_rows);
        let chunk_size = num_par_rows * w;

        // For the layers involving blocks larger than `num_par_rows`, we will
        // parallelize across the blocks.

        // We do `LAYERS_PER_GROUP` layers of the DFT at once, to minimize how much data we need to transfer
        // between threads.
        for (twiddles_0, twiddles_1, twiddles_2) in
            root_table[log_num_par_rows..].iter().rev().tuples()
        {
            dit_layer_par_triple(&mut mat.as_view_mut(), twiddles_0, twiddles_1, twiddles_2);
        }

        // If the total number of layers is not a multiple of `LAYERS_PER_GROUP`,
        // we need to handle the remaining layers separately.
        if (log_h - log_num_par_rows) % LAYERS_PER_GROUP == 1 {
            dit_layer_par(&mut mat.as_view_mut(), &root_table[log_num_par_rows]);
        } else if (log_h - log_num_par_rows) % LAYERS_PER_GROUP == 2 {
            dit_layer_par_double(
                &mut mat.as_view_mut(),
                &root_table[log_num_par_rows + 1],
                &root_table[log_num_par_rows],
            );
        }

        // Once the blocks are small enough, we can split the matrix
        // into chunks of size `chunk_size` and process them in parallel.
        // This avoids passing data between threads, which can be expensive.
        par_remaining_layers(&mut mat.values, chunk_size, &root_table[..log_num_par_rows]);

        // Finally we bit-reverse the matrix to ensure the output is in the correct order.
        reverse_matrix_index_bits(&mut mat);
        mat
    }
}

/// Applies one layer of the Radix-2 DIT FFT butterfly network making use of parallelization.
///
/// Splits the matrix into blocks of rows and performs in-place butterfly operations
/// on each block. Uses a `TwiddleFreeButterfly` for the first pair and `DitButterfly`
/// with precomputed twiddles for the rest.
///
/// Each block is processed in parallel, if the blocks are large enough they themselves
/// are split into parallel sub-blocks.
///
/// # Arguments
/// - `mat`: Mutable matrix whose height is a power of two.
/// - `twiddles`: Precomputed twiddle factors for this layer.
#[inline]
fn dit_layer_par<F: Field>(mat: &mut RowMajorMatrixViewMut<'_, F>, twiddles: &[F]) {
    debug_assert!(
        mat.height() % twiddles.len() == 0,
        "Matrix height must be divisible by the number of twiddles"
    );
    let size = mat.values.len();
    let num_blocks = twiddles.len();

    let outer_block_size = size / num_blocks;
    let half_outer_block_size = outer_block_size / 2;

    mat.values
        .par_chunks_exact_mut(outer_block_size)
        .enumerate()
        .for_each(|(ind, block)| {
            // Split each block vertically into top (hi) and bottom (lo) halves
            let (hi_chunk, lo_chunk) = block.split_at_mut(half_outer_block_size);

            // If num_blocks is small, we probably are not using all available threads.
            let num_threads = current_num_threads();
            let inner_block_size = size / (2 * num_blocks).max(num_threads);

            hi_chunk
                .par_chunks_mut(inner_block_size)
                .zip(lo_chunk.par_chunks_mut(inner_block_size))
                .for_each(|(hi_chunk, lo_chunk)| {
                    if ind == 0 {
                        // The first pair doesn't require a twiddle factor
                        TwiddleFreeEvalsButterfly.apply_to_rows(hi_chunk, lo_chunk);
                    } else {
                        // Apply DIT butterfly using the twiddle factor at index `ind - 1`
                        DitEvalsButterfly(twiddles[ind]).apply_to_rows(hi_chunk, lo_chunk);
                    }
                });
        });
}

/// Splits the matrix into chunks of size `chunk_size` and performs
/// the remaining layers of the FFT in parallel on each chunk.
///
/// This avoids passing data between threads, which can be expensive.
#[inline]
fn par_remaining_layers<F: Field>(mat: &mut [F], chunk_size: usize, root_table: &[Vec<F>]) {
    mat.par_chunks_exact_mut(chunk_size)
        .enumerate()
        .for_each(|(index, chunk)| {
            for (layer, twiddles) in root_table.iter().rev().enumerate() {
                let num_twiddles_per_block = 1 << layer;
                let start = index * num_twiddles_per_block;
                let twiddle_range = start..(start + num_twiddles_per_block);
                dit_layer(chunk, &twiddles[twiddle_range]);
            }
        });
}

/// Applies one layer of the Radix-2 DIT FFT butterfly network on a single core.
///
/// Splits the matrix into blocks of rows and performs in-place butterfly operations
/// on each block.
///
/// # Arguments
/// - `vec`: Mutable vector whose height is a power of two.
/// - `twiddles`: Precomputed twiddle factors for this layer.
#[inline]
fn dit_layer<F: Field>(vec: &mut [F], twiddles: &[F]) {
    debug_assert_eq!(
        vec.len() % twiddles.len(),
        0,
        "Vector length must be divisible by the number of twiddles"
    );
    let size = vec.len();
    let num_blocks = twiddles.len();

    let block_size = size / num_blocks;
    let half_block_size = block_size / 2;

    vec.chunks_exact_mut(block_size)
        .zip(twiddles)
        .for_each(|(block, &twiddle)| {
            // Split each block vertically into top (hi) and bottom (lo) halves
            let (hi_chunk, lo_chunk) = block.split_at_mut(half_block_size);

            // Apply DIT butterfly
            DitEvalsButterfly(twiddle).apply_to_rows(hi_chunk, lo_chunk);
        });
}

/// Applies two layers of the Radix-2 DIT FFT butterfly network making use of parallelization.
///
/// Splits the matrix into blocks of rows and performs in-place butterfly operations
/// on each block. Advantage of doing two layers at once is it reduces the amount of
/// data transferred between threads.
///
/// # Arguments
/// - `mat`: Mutable matrix whose height is a power of two.
/// - `twiddles_0`: Precomputed twiddle factors for the first layer.
/// - `twiddles_1`: Precomputed twiddle factors for the second layer.
#[inline]
fn dit_layer_par_double<F: Field>(
    mat: &mut RowMajorMatrixViewMut<'_, F>,
    twiddles_0: &[F],
    twiddles_1: &[F],
) {
    debug_assert!(
        mat.height() % twiddles_0.len() == 0,
        "Matrix height must be divisible by the number of twiddles"
    );
    let size = mat.values.len();
    let num_blocks = twiddles_0.len();

    let outer_block_size = size / num_blocks;
    let quarter_outer_block_size = outer_block_size / 4;

    // Estimate the optimal size of the inner chunks so that all data fits in L1 cache.
    // Note that 4 inner chunks are processed in each parallel thread so we divide by 4.
    let inner_chunk_size =
        (workload_size::<F>().next_power_of_two() / 4).min(quarter_outer_block_size);

    mat.values
        .par_chunks_exact_mut(outer_block_size)
        .enumerate()
        .for_each(|(ind, block)| {
            // Split each block into four quarters. Each quarter will be further split into
            // sub-chunks processed in parallel.
            let chunk_par_iters_0 = block
                .chunks_exact_mut(quarter_outer_block_size)
                .map(|chunk| chunk.par_chunks_mut(inner_chunk_size))
                .collect::<Vec<_>>();
            let chunk_par_iters_1 = zip_par_iter_vec(chunk_par_iters_0);
            chunk_par_iters_1.into_iter().tuples().for_each(|(hi, lo)| {
                hi.zip(lo)
                    .for_each(|((hi_hi_chunk, hi_lo_chunk), (lo_hi_chunk, lo_lo_chunk))| {
                        // Do 2 layers of the DIT FFT butterfly network at once.
                        if ind == 0 {
                            // Layer 0:
                            TwiddleFreeEvalsButterfly.apply_to_rows(hi_hi_chunk, lo_hi_chunk);
                            TwiddleFreeEvalsButterfly.apply_to_rows(hi_lo_chunk, lo_lo_chunk);

                            // Layer 1:
                            TwiddleFreeEvalsButterfly.apply_to_rows(hi_hi_chunk, hi_lo_chunk);
                            DitEvalsButterfly(twiddles_1[1])
                                .apply_to_rows(lo_hi_chunk, lo_lo_chunk);
                        } else {
                            // Layer 0:
                            DitEvalsButterfly(twiddles_0[ind])
                                .apply_to_rows(hi_hi_chunk, lo_hi_chunk);
                            DitEvalsButterfly(twiddles_0[ind])
                                .apply_to_rows(hi_lo_chunk, lo_lo_chunk);

                            // Layer 1:
                            DitEvalsButterfly(twiddles_1[2 * ind])
                                .apply_to_rows(hi_hi_chunk, hi_lo_chunk);
                            DitEvalsButterfly(twiddles_1[2 * ind + 1])
                                .apply_to_rows(lo_hi_chunk, lo_lo_chunk);
                        }
                    });
            });
        });
}

/// Applies three layers of the Radix-2 DIT FFT butterfly network making use of parallelization.
///
/// Splits the matrix into blocks of rows and performs in-place butterfly operations
/// on each block. Advantage of doing three layers at once is it reduces the amount of
/// data transferred between threads.
///
/// # Arguments
/// - `mat`: Mutable matrix whose height is a power of two.
/// - `twiddles_0`: Precomputed twiddle factors for the first layer.
/// - `twiddles_1`: Precomputed twiddle factors for the second layer.
/// - `twiddles_2`: Precomputed twiddle factors for the third layer.
#[inline]
fn dit_layer_par_triple<F: Field>(
    mat: &mut RowMajorMatrixViewMut<'_, F>,
    twiddles_0: &[F],
    twiddles_1: &[F],
    twiddles_2: &[F],
) {
    debug_assert!(
        mat.height() % twiddles_0.len() == 0,
        "Matrix height must be divisible by the number of twiddles"
    );
    let size = mat.values.len();
    let num_blocks = twiddles_0.len();

    let outer_block_size = size / num_blocks;
    let eighth_outer_block_size = outer_block_size / 8;

    // Estimate the optimal size of the inner chunks so that all data fits in L1 cache.
    // Note that 8 inner chunks are processed in each parallel thread so we divide by 8.
    let inner_chunk_size =
        (workload_size::<F>().next_power_of_two() / 8).min(eighth_outer_block_size);

    mat.values
        .par_chunks_exact_mut(outer_block_size)
        .enumerate()
        .for_each(|(ind, block)| {
            // Split each block into eight equal parts. Each part will be further split into
            // sub-chunks processed in parallel.
            let chunk_par_iters_0 = block
                .chunks_exact_mut(eighth_outer_block_size)
                .map(|chunk| chunk.par_chunks_mut(inner_chunk_size))
                .collect::<Vec<_>>();
            let chunk_par_iters_1 = zip_par_iter_vec(chunk_par_iters_0);
            let chunk_par_iters_2 = zip_par_iter_vec(chunk_par_iters_1);
            chunk_par_iters_2.into_iter().tuples().for_each(|(hi, lo)| {
                hi.zip(lo).for_each(
                    |(
                        ((hi_hi_hi_chunk, hi_hi_lo_chunk), (hi_lo_hi_chunk, hi_lo_lo_chunk)),
                        ((lo_hi_hi_chunk, lo_hi_lo_chunk), (lo_lo_hi_chunk, lo_lo_lo_chunk)),
                    )| {
                        // Do 3 layers of the DIT FFT butterfly network at once.
                        if ind == 0 {
                            // Layer 0:
                            TwiddleFreeEvalsButterfly.apply_to_rows(hi_hi_hi_chunk, lo_hi_hi_chunk);
                            TwiddleFreeEvalsButterfly.apply_to_rows(hi_hi_lo_chunk, lo_hi_lo_chunk);
                            TwiddleFreeEvalsButterfly.apply_to_rows(hi_lo_hi_chunk, lo_lo_hi_chunk);
                            TwiddleFreeEvalsButterfly.apply_to_rows(hi_lo_lo_chunk, lo_lo_lo_chunk);

                            // Layer 1:
                            TwiddleFreeEvalsButterfly.apply_to_rows(hi_hi_hi_chunk, hi_lo_hi_chunk);
                            TwiddleFreeEvalsButterfly.apply_to_rows(hi_hi_lo_chunk, hi_lo_lo_chunk);
                            DitEvalsButterfly(twiddles_1[1])
                                .apply_to_rows(lo_hi_hi_chunk, lo_lo_hi_chunk);
                            DitEvalsButterfly(twiddles_1[1])
                                .apply_to_rows(lo_hi_lo_chunk, lo_lo_lo_chunk);

                            // Layer 2:
                            TwiddleFreeEvalsButterfly.apply_to_rows(hi_hi_hi_chunk, hi_hi_lo_chunk);
                            DitEvalsButterfly(twiddles_2[1])
                                .apply_to_rows(hi_lo_hi_chunk, hi_lo_lo_chunk);
                            DitEvalsButterfly(twiddles_2[2])
                                .apply_to_rows(lo_hi_hi_chunk, lo_hi_lo_chunk);
                            DitEvalsButterfly(twiddles_2[3])
                                .apply_to_rows(lo_lo_hi_chunk, lo_lo_lo_chunk);
                        } else {
                            // Layer 0:
                            DitEvalsButterfly(twiddles_0[ind])
                                .apply_to_rows(hi_hi_hi_chunk, lo_hi_hi_chunk);
                            DitEvalsButterfly(twiddles_0[ind])
                                .apply_to_rows(hi_hi_lo_chunk, lo_hi_lo_chunk);
                            DitEvalsButterfly(twiddles_0[ind])
                                .apply_to_rows(hi_lo_hi_chunk, lo_lo_hi_chunk);
                            DitEvalsButterfly(twiddles_0[ind])
                                .apply_to_rows(hi_lo_lo_chunk, lo_lo_lo_chunk);

                            // Layer 1:
                            DitEvalsButterfly(twiddles_1[2 * ind])
                                .apply_to_rows(hi_hi_hi_chunk, hi_lo_hi_chunk);
                            DitEvalsButterfly(twiddles_1[2 * ind])
                                .apply_to_rows(hi_hi_lo_chunk, hi_lo_lo_chunk);
                            DitEvalsButterfly(twiddles_1[2 * ind + 1])
                                .apply_to_rows(lo_hi_hi_chunk, lo_lo_hi_chunk);
                            DitEvalsButterfly(twiddles_1[2 * ind + 1])
                                .apply_to_rows(lo_hi_lo_chunk, lo_lo_lo_chunk);

                            // Layer 2:
                            DitEvalsButterfly(twiddles_2[4 * ind])
                                .apply_to_rows(hi_hi_hi_chunk, hi_hi_lo_chunk);
                            DitEvalsButterfly(twiddles_2[4 * ind + 1])
                                .apply_to_rows(hi_lo_hi_chunk, hi_lo_lo_chunk);
                            DitEvalsButterfly(twiddles_2[4 * ind + 2])
                                .apply_to_rows(lo_hi_hi_chunk, lo_hi_lo_chunk);
                            DitEvalsButterfly(twiddles_2[4 * ind + 3])
                                .apply_to_rows(lo_lo_hi_chunk, lo_lo_lo_chunk);
                        }
                    },
                );
            });
        });
}

/// Estimates the optimal workload size for `T` to fit in L1 cache.
///
/// Approximates the size of the L1 cache by 32 KB. Used to determine the number of
/// chunks to process in parallel.
#[must_use]
const fn workload_size<T: Sized>() -> usize {
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    const CACHE_SIZE: usize = 1 << 17; // 128KB for Apple Silicon

    #[cfg(all(target_arch = "aarch64", any(target_os = "ios", target_os = "android")))]
    const CACHE_SIZE: usize = 1 << 16; // 64KB for mobile ARM

    #[cfg(target_arch = "x86_64")]
    const CACHE_SIZE: usize = 1 << 15; // 32KB for x86-64

    #[cfg(not(any(
        all(target_arch = "aarch64", target_os = "macos"),
        all(target_arch = "aarch64", any(target_os = "ios", target_os = "android")),
        target_arch = "x86_64"
    )))]
    const CACHE_SIZE: usize = 1 << 15; // 32KB default

    CACHE_SIZE / size_of::<T>()
}

/// Estimates the optimal number of rows of a `RowMajorMatrix<T>` to take in each parallel chunk.
///
/// Designed to ensure that `<T> * estimate_num_rows_par() * width` is roughly the size of the L1 cache.
///
/// Assumes that height is a power of two and always outputs a power of two.
#[must_use]
fn estimate_num_rows_in_l1<T: Sized>(height: usize, width: usize) -> usize {
    (workload_size::<T>() / width)
        .next_power_of_two()
        .min(height) // Ensure we don't exceed the height of the matrix.
}

/// Given a vector of parallel iterators, zip all pairs together.
///
/// This lets us simulate the izip!() macro but for our possibly parallel iterators.
///
/// This function assumes that the input vector has an even number of elements. If
/// it is given an odd number of elements, the last element will be ignored.
#[inline]
fn zip_par_iter_vec<I: IndexedParallelIterator>(
    in_vec: Vec<I>,
) -> Vec<impl IndexedParallelIterator<Item = (I::Item, I::Item)>> {
    in_vec
        .into_iter()
        .tuples()
        .map(|(hi, lo)| hi.zip(lo))
        .collect::<Vec<_>>()
}

/// Butterfly with no twiddle factor (`twiddle = 1`).
///
/// This is used when no root-of-unity scaling is needed.
/// It works for either DIT or DIF, and is often used at
/// the final or base level of a transform tree.
///
/// This butterfly computes:
/// ```text
///   - output_1 = x2
///   - output_2 = 2.x1 - x2
/// ```
#[derive(Copy, Clone, Debug)]
pub struct TwiddleFreeEvalsButterfly;

impl<F: Field> Butterfly<F> for TwiddleFreeEvalsButterfly {
    #[inline]
    fn apply<PF: PackedField<Scalar = F>>(&self, x_1: PF, x_2: PF) -> (PF, PF) {
        (x_2, x_1.double() - x_2)
    }
}

/// DIT (Decimation-In-Time) butterfly operation.
///
/// Used in the *input-ordering* variant of NTT/FFT.
/// This butterfly computes:
/// ```text
///   output_1 = (1 - twiddle) * x1 + twiddle * x2 = x1 + twiddle * (x2 - x1)
///   output_2 = (1 + twiddle) * x1 - twiddle * x2 = x1 - twiddle * (x2 - x1)
/// ```
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct DitEvalsButterfly<F>(pub F);

impl<F: Field> Butterfly<F> for DitEvalsButterfly<F> {
    #[inline]
    fn apply<PF: PackedField<Scalar = F>>(&self, x_1: PF, x_2: PF) -> (PF, PF) {
        let x_2_twiddle = (x_2 - x_1) * self.0;
        (x_1 + x_2_twiddle, x_1 - x_2_twiddle)
    }
}

#[cfg(test)]
mod test {
    use p3_field::PrimeCharacteristicRing;
    use p3_koala_bear::KoalaBear;
    use p3_matrix::dense::DenseMatrix;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use super::*;
    use crate::poly::evals::EvaluationsList;

    #[test]
    fn test_evals_dft() {
        type F = KoalaBear;
        let n_vars = 13;
        let width = 4;
        let mut rng = StdRng::seed_from_u64(0);

        let pols = (0..width)
            .map(|_| EvaluationsList::<F>::new((0..1 << n_vars).map(|_| rng.random()).collect()))
            .collect::<Vec<_>>();

        let dft = EvalsDft::<F>::default();
        let mut matrix = vec![F::ZERO; (1 << n_vars) * width];
        for (i, pol) in pols.iter().enumerate() {
            for (j, &eval) in pol.iter().enumerate() {
                matrix[i + j * width] = eval;
            }
        }
        let dft_res = dft
            .dft_batch_by_evals(DenseMatrix::new(matrix, width))
            .values;

        let root = F::two_adic_generator(n_vars);
        for (i, pol) in pols.into_iter().enumerate() {
            let pol_coeffs = pol.to_coefficients();
            for j in 0..(1 << n_vars) {
                assert_eq!(
                    dft_res[i + j * width],
                    pol_coeffs.evaluate_at_univariate(&[root.exp_u64(j as u64)])[0]
                );
            }
        }
    }
}
