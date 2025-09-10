//! # Mixed-Radix DFT on Coefficients
//!
//! This module provides an implementation of the Discrete Fourier Transform (DFT)
//! for composite domain sizes. It is designed to work with polynomial coefficients and is
//! useful for finite fields with limited two-adicity, where traditional power-of-two
//! FFTs are insufficient for large domains.
//!
//! ## Algorithm Overview
//!
//! This implementation is a direct realization of the **mixed-radix algorithm**, as
//! detailed in the [technical specification](https://hackmd.io/@tcoratger/S1JQUXarel).
//! The core idea is to decompose a large DFT of size $N$ into smaller, more manageable transforms.
//!
//! The process follows three main stages:
//!
//! 1.  **Decomposition & Inner DFTs**: The total domain size $N$ is factored into its largest
//!     power-of-two component, $N_1 = 2^k$, and the remaining odd component, $N_2$. The
//!     algorithm then performs $N_2$ individual DFTs of size $N_1$. These are executed efficiently
//!     using a highly optimized, injected radix-2 FFT implementation.
//!
//! 2.  **Twiddle Factor Multiplication**: After the inner DFTs, the intermediate results are
//!     multiplied by "twiddle factors." These are complex roots of unity, $\omega_N^{j_2 k_1}$,
//!     that correctly stitch the smaller transforms back together, ensuring proper phase alignment.
//!
//! 3.  **Outer DFTs**: Finally, the algorithm performs $N_1$ DFTs of the odd size $N_2$.
//!     To handle these arbitrary-sized transforms efficiently, this implementation uses
//!     **Bluestein's algorithm**. This technique converts a DFT of any size into a
//!     convolution, which can then be solved rapidly using a larger power-of-two FFT.

use std::marker::PhantomData;

use p3_dft::TwoAdicSubgroupDft;
use p3_field::{PrimeField64, TwoAdicField, par_scale_slice_in_place};
use p3_matrix::{
    Matrix,
    dense::{RowMajorMatrix, RowMajorMatrixViewMut},
};

/// The main struct for performing a mixed-radix Discrete Fourier Transform.
///
/// It is generic over:
/// - The field `F`,
/// - A radix-2 DFT implementation `R2`.
///
/// This allows to delegate all power-of-two transforms to a specialized backend.
#[derive(Clone, Debug)]
pub struct MixedRadixDft<F, R2>
where
    F: TwoAdicField,
    R2: TwoAdicSubgroupDft<F>,
{
    /// An instance of a radix-2 DFT implementation, used to handle the power-of-two components of the transform.
    radix2_dft: R2,
    /// A zero-sized marker to handle the generic parameter `F`.
    _phantom: PhantomData<F>,
}

impl<F, R2> Default for MixedRadixDft<F, R2>
where
    F: TwoAdicField,
    R2: TwoAdicSubgroupDft<F>,
{
    fn default() -> Self {
        Self {
            radix2_dft: R2::default(),
            _phantom: PhantomData,
        }
    }
}

impl<F, R2> MixedRadixDft<F, R2>
where
    F: TwoAdicField + PrimeField64,
    R2: TwoAdicSubgroupDft<F>,
{
    /// Creates a new `MixedRadixDft` with a specific radix-2 DFT implementation.
    #[must_use]
    pub const fn new(radix2_dft: R2) -> Self {
        Self {
            radix2_dft,
            _phantom: PhantomData,
        }
    }

    /// Performs a mixed-radix DFT on a matrix of polynomial coefficients.
    ///
    /// This is the main entry point of the algorithm. It takes a matrix where each column
    /// represents the coefficients of a polynomial and evaluates these polynomials over a
    /// domain of size `N`, where `N` is the matrix height.
    ///
    /// # Arguments
    /// * `coeffs`: A `RowMajorMatrix` containing the polynomial coefficients. The height of the
    ///   matrix determines the size of the DFT.
    ///
    /// # Returns
    /// A `RowMajorMatrix` containing the evaluations of the polynomials.
    pub fn dft_batch(&self, coeffs: RowMajorMatrix<F>) -> RowMajorMatrix<F> {
        // Get the total size of the DFT from the matrix height.
        let n = coeffs.height();
        // If the matrix is empty, there's nothing to do.
        if n == 0 {
            return RowMajorMatrix::new(vec![], coeffs.width());
        }

        // Decompose N into its power-of-two part (n1) and its odd part (n2).
        //
        // We compute n1 and n2 such that n = n1 * n2.
        let two_adicity = n.trailing_zeros() as usize;
        // The power-of-two part of the DFT.
        let n1 = 1 << two_adicity;
        // The odd part of the DFT.
        let n2 = n / n1;

        // If N is already a pure power of two (n2=1), the mixed-radix algorithm is unnecessary.
        //
        // We delegate directly to the highly optimized radix-2 implementation for best performance.
        if n2 == 1 {
            return self.radix2_dft.dft_batch(coeffs).to_row_major_matrix();
        }

        // The number of polynomials being transformed simultaneously (i.e., the number of columns).
        let width = coeffs.width();

        // Create the temporary matrix that will hold the transposed and intermediate results.
        //
        // This corresponds to the matrix A in the technical note.
        let mut matrix_a = RowMajorMatrix::new(F::zero_vec(n * width), width);

        // ## Inner DFTs (Radix-2)
        //
        // ### Mathematical Description
        // This block corresponds to the first stage of the FFT algorithm.
        // For a DFT of size N = n1 * n2, we perform `n2` separate DFTs, each of size `n1`.
        //
        // The formula for this stage is:
        // $$ A_{k_1, j_2} = \sum_{j_1=0}^{n1-1} a_{j_1, j_2} \cdot \omega_{n1}^{j_1 k_1} $$
        //
        // The input coefficients `a_j` are conceptually rearranged into a matrix of
        // size `n1 x n2`. The term `a_{j_1, j_2}` corresponds to the input coefficient at the
        // linear index `j = j1*n2 + j2`.
        //
        // ### Example with N=6 (n1=2, n2=3)
        // This loop effectively reads columns from a conceptual rearrangement of the input.
        //
        // Input `coeffs` vector:
        // `[a0, a1, a2, a3, a4, a5]`
        //
        // Conceptually, the algorithm reads this as a 2x3 matrix (n1 x n2):
        // `[ a0, a1, a2 ]`  <-- j1=0
        // `[ a3, a4, a5 ]`  <-- j1=1
        //
        // The outer loop iterates through j2=0, 1, 2 (the columns):
        //
        // - When `j2=0`, we gather the 0th column:
        //   `inner_dft_input` becomes `[a0, a3]`
        //
        // - When `j2=1`, we gather the 1st column:
        //   `inner_dft_input` becomes `[a1, a4]`
        //
        // - When `j2=2`, we gather the 2nd column:
        //   `inner_dft_input` becomes `[a2, a5]`
        //
        // Each of these `inner_dft_input` vectors is then passed to the size-2 radix-2 DFT.
        for j2 in 0..n2 {
            // Create a temporary contiguous matrix to hold the input for one batch of inner DFTs.
            let mut inner_dft_input = RowMajorMatrix::new(F::zero_vec(n1 * width), width);

            // This inner loop implements the "gather" step illustrated in the example above.
            // It collects the `j2`-th "column" of the conceptual n1 x n2 matrix into the
            // contiguous buffer (`inner_dft_input`) so the radix-2 FFT can be applied.
            //
            // Let us consider our example again, where `n1=2` and `n2=3`.
            //
            // The conceptual matrix is:
            // `[ a0, a1, a2 ]`  <-- j1=0
            // `[ a3, a4, a5 ]`  <-- j1=1
            //
            // For instance, when `j2=1`, we want to gather the index 1 column ([a1, a4 ]):
            // 1. For `j1=0`, `src_row_idx` is `0*3 + 1 = 1`. `coeffs[1]` (which is `a1`) is copied.
            // 2. For `j1=1`, `src_row_idx` is `1*3 + 1 = 4`. `coeffs[4]` (which is `a4`) is copied.
            //
            // After this loop, `inner_dft_input` correctly holds:
            // - For j2 = 0, the 2x1 vector `[a0, a3]`
            // - For j2 = 1, the 2x1 vector `[a1, a4]`
            // - For j2 = 2, the 2x1 vector `[a2, a5]`
            for j1 in 0..n1 {
                let src_row_idx = j1 * n2 + j2;
                let row_slice = coeffs.row_slice(src_row_idx).unwrap();
                inner_dft_input.row_mut(j1).copy_from_slice(&row_slice);
            }

            // Perform the fast radix-2 DFT on the gathered data.
            let inner_dft_output = self.radix2_dft.dft_batch(inner_dft_input);
            // Convert the output back to a standard row-major matrix layout.
            let inner_dft_rm = inner_dft_output.to_row_major_matrix();

            // This loop scatters the results from the inner DFT back into the `temp_values` buffer.
            //
            // ### Example with N=6 (n1=2, n2=3)
            //
            // The output `inner_dft_rm`:
            // - For j2 = 0, the 2x1 vector: `[ A_{0,0}, A_{1,0} ]`
            // - For j2 = 1, the 2x1 vector: `[ A_{0,1}, A_{1,1} ]`
            // - For j2 = 2, the 2x1 vector: `[ A_{0,2}, A_{1,2} ]`
            //
            // This loop writes these two values (for a given j2 = 1) into `temp_values`:
            //
            // - When `k1=0`, `dest_row_idx` is `0*3 + 1 = 1`. `A_{0,1}` is written to line 1 of `A` matrix.
            // - When `k1=1`, `dest_row_idx` is `1*3 + 1 = 4`. `A_{1,1}` is written to line 4 of `A` matrix.
            //
            // After all `j2` iterations (0, 1, and 2) are complete, we have virtually a 2x3 `A` matrix:
            // `[ A_{0,0}, A_{0,1}, A_{0,2} ]`
            // `[ A_{1,0}, A_{1,1}, A_{1,2} ]`
            //
            // But it is reprenseted as a 1x6 `matrix_a` matrix, which is the input to Step 2:
            // `[ A_{0,0} ]`  <-- k1=0, j2=0
            // `[ A_{0,1} ]`  <-- k1=0, j2=1
            // `[ A_{0,2} ]`  <-- k1=0, j2=2
            // `[ A_{1,0} ]`  <-- k1=1, j2=0
            // `[ A_{1,1} ]`  <-- k1=1, j2=1
            // `[ A_{1,2} ]`  <-- k1=1, j2=2
            //
            // This is done like this to accomodate multiple polynomials in the same DFT.
            for k1 in 0..n1 {
                let dest_row_idx = k1 * n2 + j2;
                let res_row = inner_dft_rm.row_slice(k1).unwrap();
                matrix_a.row_mut(dest_row_idx).copy_from_slice(&res_row);
            }
        }

        // Step 2: Twiddle Factor Multiplication
        //
        // ### Mathematical Description
        // This stage corresponds to applying the "twiddle factors" from the Cooley-Tukey formula,
        // which correctly phase-aligns the results of the inner DFTs. The formula is:
        // $$ A'_{k_1, j_2} = A_{k_1, j_2} \cdot \omega_N^{j_2 k_1} $$
        //
        // After Step 1, `matrix_a` holds the `A_{k1, j2}` values.
        // This block computes all the `ω_N^(j2*k1)` factors and applies them in an element-wise multiplication.
        //
        // ### Example with N=6 (n1=2, n2=3)
        // `matrix_a` before this step holds the results from the inner DFTs:
        // `[ A_{0,0} ]`  <-- k1=0, j2=0
        // `[ A_{0,1} ]`  <-- k1=0, j2=1
        // `[ A_{0,2} ]`  <-- k1=0, j2=2
        // `[ A_{1,0} ]`  <-- k1=1, j2=0
        // `[ A_{1,1} ]`  <-- k1=1, j2=1
        // `[ A_{1,2} ]`  <-- k1=1, j2=2
        //
        // This code calculates the corresponding matrix of twiddle factors, `ω_6^(k1*j2)`:
        // `[ ω_6^{0 * 0}, ω_6^{0 * 1}, ω_6^{0 * 2} ]`  <- k1=0
        // `[ ω_6^{1 * 0}, ω_6^{1 * 1}, ω_6^{1 * 2} ]`  <- k1=1
        //
        // The `flat_map` operation generates these twiddles in a flat `Vec` in the correct
        // row-major order:
        // `[ω_6^{0 * 0}, ω_6^{0 * 1}, ω_6^{0 * 2}, ω_6^{1 * 0}, ω_6^{1 * 1}, ω_6^{1 * 2}]`.
        //
        // The subsequent `zip` then multiplies `matrix_a` by these factors element-wise.

        // Calculate the exponent needed to shrink the full multiplicative group to size `n`.
        //  - `F::ORDER_U64 - 1` is the order of the field's multiplicative group (often called `p-1`).
        //  - Dividing by `n` gives the correct exponent.
        let exponent = (F::ORDER_U64 - 1) / (n as u64);

        // Raise the field's main generator to that power.
        //  - `F::GENERATOR` is an element that generates the entire multiplicative group of `p-1` elements.
        //  - Raising it to the power `(p-1)/n` yields an element of order exactly `n`.
        let omega = F::GENERATOR.exp_u64(exponent);

        // Twiddle Generation Logic
        //
        // This `flat_map` operation generates the twiddles `ω_N^(k1*j2)` in a flat vector.
        //
        // `[ ω_6^{0 * 0}, ω_6^{0 * 1}, ω_6^{0 * 2} ]`  <- k1=0
        // `[ ω_6^{1 * 0}, ω_6^{1 * 1}, ω_6^{1 * 2} ]`  <- k1=1
        let twiddles_vec: Vec<_> = (0..n1 as u64)
            .flat_map(|k1| {
                // Calculate ω_N^(k1)
                let k1_omega = omega.exp_u64(k1);
                // Generate the `n2` twiddles for this row.
                // This corresponds to ω_N^(k1*j2) up to the `n2` power.
                k1_omega.powers().take(n2)
            })
            .collect();

        // Then, wrap these values in a matrix structure for easier, more intuitive access.
        // The twiddle matrix has n1 rows and n2 columns.
        let twiddles_matrix = RowMajorMatrix::new(twiddles_vec, n2);

        // Apply the twiddle factors to the intermediate results.
        //
        // ### Example with N=6 (n1=2, n2=3)
        //
        // This code performs an element-wise multiplication between the intermediate matrix `matrix_a`
        // and the `twiddles` vector we just generated.
        //
        // `matrix_a` (conceptually):
        // `[ A_{0,0}, A_{0,1}, A_{0,2} ]`
        // `[ A_{1,0}, A_{1,1}, A_{1,2} ]`
        //
        // In reality `matrix_a` is:
        // `[ A_{0,0} ]`  <-- k1=0, j2=0
        // `[ A_{0,1} ]`  <-- k1=0, j2=1
        // `[ A_{0,2} ]`  <-- k1=0, j2=2
        // `[ A_{1,0} ]`  <-- k1=1, j2=0
        // `[ A_{1,1} ]`  <-- k1=1, j2=1
        // `[ A_{1,2} ]`  <-- k1=1, j2=2
        //
        // `twiddles` matrix:
        // `[ ω_6^0, ω_6^0, ω_6^0 ]`
        // `[ ω_6^0, ω_6^1, ω_6^2 ]`
        //
        // The element-wise product results in the final matrix `A'` needed for the next stage:
        // `[ A_{0,0}*ω_6^0 ]`  <-- k1=0, j2=0
        // `[ A_{0,1}*ω_6^0 ]`  <-- k1=0, j2=1
        // `[ A_{0,2}*ω_6^0 ]`  <-- k1=0, j2=2
        // `[ A_{1,0}*ω_6^0 ]`  <-- k1=1, j2=0
        // `[ A_{1,1}*ω_6^1 ]`  <-- k1=1, j2=1
        // `[ A_{1,2}*ω_6^2 ]`  <-- k1=1, j2=2
        matrix_a
            .row_chunks_exact_mut(n2)
            .zip(twiddles_matrix.rows())
            .for_each(|(mut data_chunk, twiddles_row)| {
                data_chunk
                    .rows_mut()
                    .zip(twiddles_row)
                    .for_each(|(data_row_slice, twiddle)| {
                        for val in data_row_slice {
                            *val *= twiddle;
                        }
                    });
            });

        // Let's rename the matrix for clarity in the next step.
        let mut twiddled_matrix = matrix_a;

        // ## Outer DFTs (Bluestein's Algorithm)
        //
        // ### Mathematical Description
        // This stage performs `n1` separate DFTs, each of size `n2`, on the twiddled data.
        // The formula is: $$ Y_{k_1, k_2} = \sum_{j_2=0}^{n2-1} A'_{k_1, j_2} \cdot \omega_{n2}^{j_2 k_2} $$
        // Thanks to the transpose in Step 1, the data for each DFT is now contiguous. This
        // loop iterates through the `n1` conceptual rows of the intermediate matrix to
        // perform these transforms.
        //
        // ### Example with N=6 (n1=2, n2=3)
        // The `twiddled_matrix` holds the values `A'`:
        // `[ A_{0,0}*ω_6^0 ]`  <-- k1=0, j2=0 <- Chunk 0
        // `[ A_{0,1}*ω_6^0 ]`  <-- k1=0, j2=1 <- Chunk 0
        // `[ A_{0,2}*ω_6^0 ]`  <-- k1=0, j2=2 <- Chunk 0
        // `[ A_{1,0}*ω_6^0 ]`  <-- k1=1, j2=0 <- Chunk 1
        // `[ A_{1,1}*ω_6^1 ]`  <-- k1=1, j2=1 <- Chunk 1
        // `[ A_{1,2}*ω_6^2 ]`  <-- k1=1, j2=2 <- Chunk 1
        //
        // The loop processes each chunk:
        // - Chunk 0 is passed to `bluestein_fft`, which transforms it in-place to the result:
        //   `[ Y_{0,0} ]`
        //   `[ Y_{0,1} ]`
        //   `[ Y_{0,2} ]`
        // - Chunk 1 is similarly transformed to:
        //   `[ Y_{1,0} ]`
        //   `[ Y_{1,1} ]`
        //   `[ Y_{1,2} ]`
        twiddled_matrix
            .row_chunks_exact_mut(n2)
            .for_each(|mut sub_matrix| {
                // `sub_matrix` is a `n2 x width` view, representing one of the `n1` problems.
                //
                // We execute the odd-sized DFT on this contiguous data.
                self.bluestein_fft(&mut sub_matrix);
            });

        // ## Final Transpose
        //
        // ### Mathematical Description
        // The Cooley-Tukey algorithm naturally produces its output in a permuted order.
        // To restore the natural order `Y_0, Y_1, ..., Y_{N-1}`, we need to perform a final
        // transpose of the conceptual `n1 x n2` matrix. The correct index mapping is:
        // $$ Y_k = Y_{k_1, k_2} \quad \text{where} \quad k = k_2 \cdot n1 + k1 $$
        //
        // ### Visual Example with N=6 (n1=2, n2=3)
        // After the outer DFTs, the data is ordered by `k1`, then `j2` (which becomes `k2`):
        // `[ Y_{0,0}, Y_{0,1}, Y_{0,2}, Y_{1,0}, Y_{1,1}, Y_{1,2} ]`
        //
        // The correct order is `Y_0, Y_1, Y_2, Y_3, Y_4, Y_5`. Applying the mapping `k = k2*n1 + k1`:
        // Y_0 = Y_{0,0} (k1=0, k2=0) -> index 0
        // Y_1 = Y_{1,0} (k1=1, k2=0) -> index 3
        // Y_2 = Y_{0,1} (k1=0, k2=1) -> index 1
        // Y_3 = Y_{1,1} (k1=1, k2=1) -> index 4
        // Y_4 = Y_{0,2} (k1=0, k2=2) -> index 2
        // Y_5 = Y_{1,2} (k1=1, k2=2) -> index 5
        //
        // This loop performs that permutation, writing the data into a new matrix in the correct order.
        let mut evaluations = RowMajorMatrix::new(F::zero_vec(n * width), width);
        for k1 in 0..n1 {
            for k2 in 0..n2 {
                let src_row_idx = k1 * n2 + k2;
                let dest_row_idx = k2 * n1 + k1;
                let row_slice = twiddled_matrix.row_slice(src_row_idx).unwrap();
                evaluations
                    .row_mut(dest_row_idx)
                    .copy_from_slice(&row_slice);
            }
        }

        evaluations
    }

    /// Performs a DFT of an arbitrary size `n` using Bluestein's algorithm.
    ///
    /// This method handles the odd-sized `n2` part of the transform. It
    /// converts a DFT of any size into a convolution problem, which can then be solved
    /// efficiently using a larger, power-of-two FFT.
    ///
    /// ### Mathematical Description
    /// The core of the algorithm is the identity $jk = \frac{j^2 + k^2 - (j-k)^2}{2}$, which
    /// allows the DFT sum to be rewritten as:
    /// \begin{equation}
    ///     Y_k = \omega_n^{k^2/2} \sum_{j=0}^{n-1} (a_j \omega_n^{j^2/2}) \cdot \omega_n^{-(k-j)^2/2}
    /// \end{equation}
    ///
    /// This has the form of a linear convolution, which can be solved with FFTs:
    /// 1. Define a "chirp" sequence $g_j = \omega_n^{j^2/2}$.
    /// 2. Create two new sequences: $f'_j = a_j \cdot g_j$ and the kernel $h_j = g_j^{-1}$.
    /// 3. Compute the convolution $c_k = (f' * h)_k = \text{IFFT}(\text{FFT}(f') \cdot \text{FFT}(h))$.
    /// 4. The final result is obtained by a final multiplication: $Y_k = c_k \cdot g_k$.
    ///
    /// # Arguments
    /// * `mat`: A mutable matrix view representing the coefficients for the odd-sized DFT.
    fn bluestein_fft(&self, mat: &mut RowMajorMatrixViewMut<'_, F>) {
        // The size of this DFT, corresponding to `n2` in the outer function.
        let n = mat.height();
        // If the matrix is empty, there's nothing to do.
        if n == 0 {
            return;
        }

        // To compute a linear convolution of two size-`n` sequences using FFTs, we need a
        // power-of-two domain of at least size `2n - 1`. `m` is the smallest such power of two.
        let m = (2 * n - 1).next_power_of_two();
        let width = mat.width();

        // ## Chirp Generation
        //
        // ### Mathematical Description
        // Bluestein's algorithm rewrites a DFT using the identity `jk = (j^2 + k^2 - (j-k)^2)/2`.
        // This introduces terms of the form `ω_n^(j^2/2)`, which are used to define a "chirp" sequence.
        //
        // The term `ω_n^(j^2/2)` can be rewritten as `(ω_n^(1/2))^(j^2)`. The base of this exponent,
        // `ω_n^(1/2)`, is an element that squares to `ω_n`. This is a **primitive 2n-th root of unity**,
        // which we will call `zeta`.
        //
        // The chirp vector `g` is then defined as `g[j] = zeta^(j^2)`.

        // Compute `zeta` and then use it to generate the chirp vector `g`.
        let n_u64 = n as u64;

        // Calculate the exponent `(p-1)/(2n)`. This is the power we need to raise the
        // field's main generator to in order to obtain a primitive `2n`-th root of unity.
        let exponent = (F::ORDER_U64 - 1) / (2 * n_u64);

        // Compute `zeta` by raising the field's generator `g` to that power.
        // The result is `zeta = g^((p-1)/(2n))`, an element of order `2n`.
        let zeta = F::GENERATOR.exp_u64(exponent);

        // Precompute the chirp sequence `g[j] = zeta^(j^2)` for j from 0 to n-1.
        //
        // Example with n=3:
        //   - The field's main generator `zeta` is a primitive 6th root of unity.
        //   - The loop calculates the following values for the `g` vector:
        //     - `g[0] = zeta^(0*0)`
        //     - `g[1] = zeta^(1*1)`
        //     - `g[2] = zeta^(2*2)`
        let g: Vec<_> = (0..n_u64).map(|j| zeta.exp_u64(j * j)).collect();

        // ## FFT-based Convolution
        //
        // Prepare for Convolution - Scale and Pad Input
        //
        // We create the sequence `f'_j = a_j * g_j` and pad it with zeros to the
        // convolution size `m`.
        //
        // #### Example (n=3, m=8, width=2)
        //
        // Input `mat` (3x2):         Chirp `g` (3x1):
        // `[ Y_{0,0}  Y1_{0,0} ]`           `[ g_0 ]`
        // `[ Y_{0,1}  Y1_{0,1} ]`           `[ g_1 ]`
        // `[ Y_{0,2}  Y1_{0,2} ]`           `[ g_2 ]`
        //
        // The loop computes `f_prime_padded[r] = mat[r] * g[r]` for the first `n` rows.
        // The remaining `m-n` rows are left as zeros, achieving the padding.
        //
        // Output `f_prime_padded` (8x2):
        // `[ Y_{0,0}*g_0  Y1_{0,0}*g_0 ]`
        // `[ Y_{0,1}*g_1  Y1_{0,1}*g_1 ]`
        // `[ Y_{0,2}*g_2  Y1_{0,2}*g_2 ]`
        // `[          0         0      ]`  <--
        // `[          0         0      ]`      |
        // `[          0         0      ]`      -- Zero-padding to size m=8
        // `[          0         0      ]`      |
        // `[          0         0      ]`  <--

        // Create the padded matrix, initialized with zeros.
        let mut f_prime_padded = RowMajorMatrix::new(F::zero_vec(m * width), width);
        {
            // Get a mutable view of the top `n` rows, which is our working area.
            let (mut f_prime_top, _) = f_prime_padded.split_rows_mut(n);

            // Copy the original `mat` data into this working area.
            f_prime_top.copy_from(&mat.as_view());

            // Scale each row of the working area by the corresponding chirp factor.
            f_prime_top
                .rows_mut()
                .zip(g.iter())
                .for_each(|(row_slice, &g_r)| {
                    par_scale_slice_in_place(row_slice, g_r);
                });
        }

        // ### Prepare for Convolution - Create the Kernel
        //
        // #### Mathematical Description
        // This step prepares the kernel `h` for the convolution. The kernel is defined as the
        // inverse of the chirp sequence:
        // \begin{equation}
        //  h_j = g_j^{-1} = \omega_n^{-j^2/2}
        // \end{equation}
        //
        // To compute a linear convolution using FFTs (which naturally compute a cyclic
        // convolution), the kernel must be padded to size `m` in a specific symmetric way.
        // This padding ensures `h[k-j]` and `h[(k-j) mod m]` are equivalent for the
        // output terms we care about.
        //
        // #### Example (n=3, m=8)
        //
        // Inverse Chirp `g_inv` = `[ g_0^-1, g_1^-1, g_2^-1 ]`
        //
        // The first loop constructs the kernel within the **first column** of the `h_padded` matrix:
        // j=0: `h_padded[0][0] = g_0^-1`
        // j=1: `h_padded[1][0] = g_1^-1` and `h_padded[7][0] = g_1^-1`
        // j=2: `h_padded[2][0] = g_2^-1` and `h_padded[6][0] = g_2^-1`
        //
        // Resulting first column of `h_padded`:
        // `[ g_0^-1, g_1^-1, g_2^-1, 0, 0, 0, g_2^-1, g_1^-1 ]`
        let mut h_padded = RowMajorMatrix::new(F::zero_vec(m * width), width);

        // This loop builds the first column of the kernel matrix using the special padding.
        for (j, g_j) in g.iter().enumerate() {
            let g_inv = g_j.inverse();
            // Set the positive-indexed part of the kernel: h[j] = g_j^(-1).
            h_padded.row_mut(j)[0] = g_inv;
            // Set the negative-indexed part (wrapped around the buffer of size m).
            if j > 0 {
                h_padded.row_mut(m - j)[0] = g_inv;
            }
        }

        // Broadcast the generated kernel to all other columns using matrix row utilities.
        h_padded.rows_mut().for_each(|row| {
            // Read the value from the first column before mutably borrowing the rest of the row.
            let fill_value = row[0];
            // Fill all subsequent columns with the value from the first column.
            row[1..].fill(fill_value);
        });

        // ## Execute Convolution via Polynomial Multiplication
        //
        // #### Mathematical Description
        // This is the core of the algorithm, where we apply the Polynomial Convolution Theorem.
        // This theorem states that the convolution of two coefficient vectors is equivalent to
        // finding the coefficients of their product polynomial. We do this efficiently by:
        //
        // 1.  Transforming both polynomials from the **coefficient domain** to the **evaluation domain**
        //     using a fast DFT (`dft_batch`).
        // 2.  Performing a simple element-wise multiplication of their evaluations.
        // 3.  Transforming the result back to the **coefficient domain** using an inverse DFT (`idft_batch`).
        //
        // The result, `conv`, contains the coefficients of the polynomial product `f'(x) * h(x)`,
        // which is the linear convolution we need.
        //
        // `coeffs(f' * h) = IDFT(evals(f') ⋅ evals(h))`

        // Transform the scaled input and the kernel to the evaluation domain.
        let f_prime_evals = self.radix2_dft.dft_batch(f_prime_padded);
        let h_evals = self.radix2_dft.dft_batch(h_padded);

        // Perform pointwise multiplication in the evaluation domain.
        let mut conv_evals = f_prime_evals.to_row_major_matrix();
        let h_evals_rm = h_evals.to_row_major_matrix();
        for i in 0..conv_evals.values.len() {
            conv_evals.values[i] *= h_evals_rm.values[i];
        }

        // Transform the result back to the coefficient domain to get the convolution.
        let conv = self.radix2_dft.idft_batch(conv_evals);

        // ## Final Correction
        //
        // #### Mathematical Description
        // The convolution result `c_k` (the coefficients in `conv`) is almost the answer. We
        // must apply the final phase correction from the Bluestein formula by multiplying
        // by the original chirp sequence `g` one last time:
        // $$ Y_k = c_k \cdot g_k $$
        //
        // #### Example (n=3, width=2)
        //
        // The diagram below illustrates this final multiplication.
        // - The matrix on the left (`conv`) contains the convolution coefficients `c_k`.
        // - The vector in the middle (`g`) contains the chirp factors `g_k`.
        // - The matrix on the right (`mat`) is the final output.
        //
        // The operation is a row-wise scaling: each of the `n` rows of the `conv` matrix is
        // multiplied by the corresponding scalar element from the `g` vector.
        //
        // `conv` (top 3 rows):     Chirp `g`:          Output `mat` (final DFT):
        // `[ c_0,0  c_0,1 ]`       `[ g_0 ]`           `[ c_0,0*g_0  c_0,1*g_0 ]`
        // `[ c_1,0  c_1,1 ]`   x   `[ g_1 ]`  =======> `[ c_1,0*g_1  c_1,1*g_1 ]`
        // `[ c_2,0  c_2,1 ]`       `[ g_2 ]`           `[ c_2,0*g_2  c_2,1*g_2 ]`
        //
        // The code implements this by first copying the relevant data from `conv` to `mat`,
        // and then scaling each row of `mat` in place.
        //
        // Get a view of the top `n` rows of the convolution result, which are the only ones we need.
        let (conv_top, _) = conv.split_rows(n);

        // Copy the relevant data from `conv_top` into `mat`.
        mat.copy_from(&conv_top.as_view());

        // Now, scale each row of `mat` by the corresponding chirp factor from `g`.
        mat.rows_mut().zip(g.iter()).for_each(|(row_slice, &g_r)| {
            par_scale_slice_in_place(row_slice, g_r);
        });
    }
}

#[cfg(test)]
mod test {
    use p3_dft::NaiveDft;
    use p3_koala_bear::KoalaBear;
    use rand::{SeedableRng, rngs::StdRng};

    use super::*;

    type F = KoalaBear;

    #[test]
    fn test_mixed_radix_dft() {
        // A composite DFT size `N` that is not a pure power of two.
        //
        // Here, N = 96, which decomposes into:
        // - N1=32 (power-of-two)
        // - N2=3 (odd).
        const N: usize = 96;
        const WIDTH: usize = 4;

        // A matrix of random polynomial coefficients.
        let mut rng = StdRng::seed_from_u64(0);
        let coeffs = RowMajorMatrix::<F>::rand(&mut rng, N, WIDTH);

        // Mixed-radix DFT on the coefficients.
        let dft = MixedRadixDft::<F, NaiveDft>::default();
        let _result_mat = dft.dft_batch(coeffs);
    }
}
