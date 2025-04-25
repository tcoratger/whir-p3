#![allow(unsafe_code)]

use std::{mem, mem::MaybeUninit};

use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_util::log2_strict_usize;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub mod utils;
pub mod wavelet;

/// Performs Reed-Solomon encoding by evaluating a polynomial on multiple cosets.
///
/// This function takes polynomial coefficients (`coeffs`) and evaluates the polynomial
/// $P(x) = \sum_{j=0}^{n-1} \text{coeffs}[j] \cdot x^j$ over an expanded domain.
/// The domain consists of `expansion` cosets of a subgroup, effectively evaluating
/// on points related to $\omega_{n \cdot e}^i \cdot \omega_n^k$, where $n$ is the number
/// of coefficients and $e$ is the `expansion` factor.
///
/// It uses the provided `TwoAdicSubgroupDft` implementation (`dft`) to perform
/// the necessary Discrete Fourier Transforms efficiently.
///
/// # Optimizations:
/// - Generates the evaluation data directly in the layout required by the DFT step,
///   avoiding an intermediate matrix transpose.
/// - Uses `MaybeUninit` to allocate the result vector, potentially speeding up
///   allocation by skipping zero-initialization.
///
/// # Safety
/// - This function uses `unsafe` blocks to work with `MaybeUninit` and perform
///   the final type conversion. Safety is guaranteed because the logic ensures
///   all elements of the `result_uninit` vector are written to exactly once
///   before the final conversion to `Vec<EF>`.
///
/// # Arguments
/// * `dft`: An implementation of `TwoAdicSubgroupDft` for the base field `F`.
/// * `coeffs`: A slice containing the polynomial coefficients in the extension field `EF`.
/// * `expansion`: The encoding expansion factor (rate = 1 / `expansion`).
///
/// # Returns
/// A `Vec<EF>` containing the polynomial evaluations on the expanded domain.
/// The total number of evaluations is `coeffs.len() * expansion`.
#[inline]
pub fn expand_from_coeff<F, EF, D>(dft: &D, coeffs: &[EF], expansion: usize) -> Vec<EF>
where
    F: Field + TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField + Send + Sync,
    D: TwoAdicSubgroupDft<F>,
{
    // Get the number of coefficients (polynomial degree + 1)
    let n = coeffs.len();
    // Handle the edge case of empty input coefficients
    if n == 0 {
        return Vec::new();
    }
    // Calculate the total size of the expanded evaluation domain
    let expanded_size = n * expansion;
    // Calculate the logarithm (base 2) of the expanded size for root calculation
    let log_expanded_size = log2_strict_usize(expanded_size);

    // Get the principal root of unity for the full domain, omega_{n*e}
    let root = F::two_adic_generator(log_expanded_size);

    // Allocate memory for the result vector without initializing elements
    let mut result_uninit: Vec<MaybeUninit<EF>> = Vec::with_capacity(expanded_size);

    // Define a closure to calculate and write values for one coefficient's contribution.
    // - `j` is the coefficient index,
    // - `col_slice` is the target slice in `result_uninit`
    let process_chunk = |(j, (col_slice, coeff)): (usize, (&mut [MaybeUninit<EF>], &EF))| {
        // Calculate the j-th power of the principal root: omega_{n*e}^j
        let root_j = root.exp_u64(j as u64);
        // Initialize the running power for the inner loop: (omega_{n*e}^j)^i starting with i=0
        let mut current_root_j_pow_i = F::ONE;
        // Iterate through the `expansion` factor
        for c in col_slice.iter_mut().take(expansion) {
            // Calculate the value: coeff[j] * (root^j)^i
            let value = *coeff * current_root_j_pow_i;
            // Write the calculated value into the uninitialized memory slot
            // SAFETY: `write` correctly initializes the `MaybeUninit` cell
            c.write(value);
            // Multiply by root_j to get the next power for the next iteration (i+1)
            current_root_j_pow_i *= root_j;
        }
    };

    // Obtain a mutable slice covering the entire capacity of the uninitialized vector.
    // SAFETY: We get a slice to the vector's capacity. This slice contains `MaybeUninit`
    // elements, which do not require initialization themselves. We MUST ensure that
    // every element of this slice is written to by `process_chunk` before we convert
    // the vector type back to `Vec<EF>`. The loops below guarantee this.
    let slice_uninit = unsafe {
        // Set the vector's length to its capacity temporarily.
        // This is necessary to obtain `&mut [MaybeUninit<EF>]` covering the full capacity.
        result_uninit.set_len(expanded_size);
        // Get the mutable slice.
        result_uninit.as_mut_slice()
    };

    // Fill the uninitialized vector using parallel or sequential iteration.
    // The data is generated directly in the n x expansion (row-major) layout.
    #[cfg(feature = "parallel")]
    {
        // Process chunks in parallel. Each chunk corresponds to a coefficient/row.
        slice_uninit
            .par_chunks_mut(expansion)
            .zip(coeffs.par_iter())
            .enumerate()
            .for_each(process_chunk);
    }
    #[cfg(not(feature = "parallel"))]
    {
        // Process chunks sequentially.
        slice_uninit
            .chunks_mut(expansion)
            .zip(coeffs.iter())
            .enumerate()
            .for_each(process_chunk);
    }

    // Convert the now fully initialized `Vec<MaybeUninit<EF>>` to `Vec<EF>`.
    // SAFETY: The parallel/sequential `for_each` loops above guarantee that every
    // element in `result_uninit` (indices 0..expanded_size-1) has been initialized
    // via `MaybeUninit::write`. We can now safely change the vector's type.
    let result = unsafe {
        // Get the raw pointer, length, and capacity of the source vector.
        let ptr = result_uninit.as_mut_ptr().cast::<EF>(); // Cast pointer type
        let len = result_uninit.len();
        let cap = result_uninit.capacity();
        // Prevent the `Vec<MaybeUninit<EF>>` from being dropped (which would deallocate).
        mem::forget(result_uninit);
        // Reconstruct the vector as `Vec<EF>` using the same memory allocation.
        Vec::from_raw_parts(ptr, len, cap)
    };

    // Create a matrix view wrapping the initialized `result` vector.
    // The matrix has `n` rows and `expansion` columns, stored in row-major order.
    let matrix = RowMajorMatrix::new(result, expansion);

    // Perform the batched DFT operation on the columns of the matrix.
    //
    // The `dft_algebra_batch` method handles the base/extension field logic.
    // `to_row_major_matrix().values` extracts the final flat vector of evaluations.
    dft.dft_algebra_batch(matrix).to_row_major_matrix().values
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_dft::NaiveDft;
    use p3_field::PrimeCharacteristicRing;

    use super::*;

    type F = BabyBear;

    #[test]
    fn test_expand_from_coeff_size_2() {
        let c0 = F::from_u64(1);
        let c1 = F::from_u64(2);
        let coeffs = vec![c0, c1];
        let expansion = 2;

        let omega = F::two_adic_generator(expansion); // ω = primitive 2nd root of unity

        // Let the polynomial be:
        //
        // ```text
        //     f(x) = c_0 + c_1 · x = 1 + 2x
        // ```
        //
        // === Expansion Step ===
        //
        // We expand the coefficient vector [c_0, c_1] by evaluating over coset shifts.
        //
        // Let ω be the 2nd root of unity. We generate new terms using powers of ω:
        //
        // ```text
        //   f_0 = c_0
        //   f_1 = c_1
        //
        //   f_2 = c_0 · ω^0 = c_0
        //   f_3 = c_1 · ω^1 = c_1 · ω
        // ```
        //
        // So the full expanded vector is:
        //
        // ```text
        //   [ f_0, f_1, f_2, f_3 ]
        // = [ c_0, c_1, c_0, c_1 · ω ]
        // ```

        let f0 = c0;
        let f1 = c1;
        let f2 = c0 * omega.exp_u64(0);
        let f3 = c1 * omega.exp_u64(1);

        // === NTT Step ===
        //
        // Now we perform a size-2 NTT on each of the two rows:
        //
        // Let the NTT matrix be:
        //
        // ```text
        //   NTT_2 =
        //   [ 1  1 ]
        //   [ 1 -1 ]
        // ```
        //
        // First chunk: [f_0, f_1]
        //
        // ```text
        //   F_0 = f_0 + f_1 = c_0 + c_1
        //   F_1 = f_0 - f_1 = c_0 - c_1
        // ```
        //
        // Second chunk: [f_2, f_3]
        //
        // ```text
        //   F_2 = f_2 + f_3 = c_0 + c_1 · ω
        //   F_3 = f_2 - f_3 = c_0 - c_1 · ω
        // ```
        //
        // After transposing the matrix (interleaving chunks), the expected output is:
        //
        // ```text
        //   [ F_0, F_2, F_1, F_3 ]
        // ```

        let expected_f0 = f0 + f1;
        let expected_f1 = f0 - f1;
        let expected_f2 = f2 + f3;
        let expected_f3 = f2 - f3;

        let expected_values_transposed = vec![expected_f0, expected_f2, expected_f1, expected_f3];

        let computed_values = expand_from_coeff(&NaiveDft, &coeffs, expansion);
        assert_eq!(computed_values, expected_values_transposed);
    }

    #[test]
    fn test_expand_from_coeff_size_4() {
        let c0 = F::from_u64(1);
        let c1 = F::from_u64(2);
        let c2 = F::from_u64(3);
        let c3 = F::from_u64(4);
        let coeffs = vec![c0, c1, c2, c3];
        let expansion = 4;

        let omega = F::two_adic_generator(4); // 4th root of unity for expansion 4

        // Manual expansion of the coefficient vector
        //
        // The expansion factor is 4, so we extend the original coefficients into 16 values:
        //
        //   f0  = c0
        //   f1  = c1
        //   f2  = c2
        //   f3  = c3
        //   f4  = c0 * ω⁰  = c0
        //   f5  = c1 * ω¹  = c1 * ω
        //   f6  = c2 * ω²  = c2 * ω²
        //   f7  = c3 * ω³  = c3 * ω³
        //   f8  = c0 * ω⁰  = c0
        //   f9  = c1 * ω²  = c1 * ω²
        //   f10 = c2 * ω⁴  = c2 * ω⁴
        //   f11 = c3 * ω⁶  = c3 * ω⁶
        //   f12 = c0 * ω⁰  = c0
        //   f13 = c1 * ω³  = c1 * ω³
        //   f14 = c2 * ω⁶  = c2 * ω⁶
        //   f15 = c3 * ω⁹  = c3 * ω⁹
        //
        // With c0 = 1, c1 = 2, c2 = 3, c3 = 4, and ω as the generator:

        let f0 = c0;
        let f1 = c1;
        let f2 = c2;
        let f3 = c3;

        let f4 = c0 * omega.exp_u64(1).exp_u64(0);
        let f5 = c1 * omega.exp_u64(1).exp_u64(1);
        let f6 = c2 * omega.exp_u64(1).exp_u64(2);
        let f7 = c3 * omega.exp_u64(1).exp_u64(3);

        let f8 = c0 * omega.exp_u64(2).exp_u64(0);
        let f9 = c1 * omega.exp_u64(2).exp_u64(1);
        let f10 = c2 * omega.exp_u64(2).exp_u64(2);
        let f11 = c3 * omega.exp_u64(2).exp_u64(3);

        let f12 = c0 * omega.exp_u64(3).exp_u64(0);
        let f13 = c1 * omega.exp_u64(3).exp_u64(1);
        let f14 = c2 * omega.exp_u64(3).exp_u64(2);
        let f15 = c3 * omega.exp_u64(3).exp_u64(3);

        // Compute the expected NTT manually using omega powers
        //
        // We process the values in **four chunks of four elements**, following the radix-2
        // butterfly structure.

        let omega = F::two_adic_generator(2);
        let omega1 = omega; // ω
        let omega2 = omega * omega; // ω²
        let omega3 = omega * omega2; // ω³
        let omega4 = omega * omega3; // ω⁴

        // Chunk 1 (f0 to f3)
        let expected_f0 = f0 + f1 + f2 + f3;
        let expected_f1 = f0 + f1 * omega1 + f2 * omega2 + f3 * omega3;
        let expected_f2 = f0 + f1 * omega2 + f2 * omega4 + f3 * omega2;
        let expected_f3 = f0 + f1 * omega3 + f2 * omega2 + f3 * omega1;

        // Chunk 2 (f4 to f7)
        let expected_f4 = f4 + f5 + f6 + f7;
        let expected_f5 = f4 + f5 * omega1 + f6 * omega2 + f7 * omega3;
        let expected_f6 = f4 + f5 * omega2 + f6 * omega4 + f7 * omega2;
        let expected_f7 = f4 + f5 * omega3 + f6 * omega2 + f7 * omega1;

        // Chunk 3 (f8 to f11)
        let expected_f8 = f8 + f9 + f10 + f11;
        let expected_f9 = f8 + f9 * omega1 + f10 * omega2 + f11 * omega3;
        let expected_f10 = f8 + f9 * omega2 + f10 * omega4 + f11 * omega2;
        let expected_f11 = f8 + f9 * omega3 + f10 * omega2 + f11 * omega1;

        // Chunk 4 (f12 to f15)
        let expected_f12 = f12 + f13 + f14 + f15;
        let expected_f13 = f12 + f13 * omega1 + f14 * omega2 + f15 * omega3;
        let expected_f14 = f12 + f13 * omega2 + f14 * omega4 + f15 * omega2;
        let expected_f15 = f12 + f13 * omega3 + f14 * omega2 + f15 * omega1;

        // Step 3: Ensure correct NTT ordering
        let expected_values_transposed = vec![
            expected_f0,
            expected_f4,
            expected_f8,
            expected_f12,
            expected_f1,
            expected_f5,
            expected_f9,
            expected_f13,
            expected_f2,
            expected_f6,
            expected_f10,
            expected_f14,
            expected_f3,
            expected_f7,
            expected_f11,
            expected_f15,
        ];

        let computed_values = expand_from_coeff(&NaiveDft, &coeffs, expansion);
        assert_eq!(computed_values, expected_values_transposed);
    }
}
