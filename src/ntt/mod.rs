#![allow(unsafe_code)]

use cooley_tukey::ntt_batch;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use transpose::transpose;

pub mod cooley_tukey;
pub mod matrix;
pub mod transpose;
pub mod utils;
pub mod wavelet;

/// RS encode at a rate 1/`expansion`.
pub fn expand_from_coeff<F: Field + TwoAdicField>(coeffs: &[F], expansion: usize) -> Vec<F> {
    let engine = cooley_tukey::NttEngine::<F>::new_from_cache();
    let expanded_size = coeffs.len() * expansion;
    let mut result = Vec::with_capacity(expanded_size);
    // Note: We can also zero-extend the coefficients and do a larger NTT.
    // But this is more efficient.

    // Do coset NTT.
    let root = engine.root(expanded_size);
    result.extend_from_slice(coeffs);
    #[cfg(not(feature = "parallel"))]
    for i in 1..expansion {
        let root = root.exp_u64(i as u64);
        let mut offset = F::ONE;
        result.extend(coeffs.iter().map(|x| {
            let val = *x * offset;
            offset *= root;
            val
        }));
    }
    #[cfg(feature = "parallel")]
    result.par_extend((1..expansion).into_par_iter().flat_map(|i| {
        let root_i = root.exp_u64(i as u64);
        coeffs
            .par_iter()
            .enumerate()
            .map_with(F::ZERO, move |root_j, (j, coeff)| {
                if root_j.is_zero() {
                    *root_j = root_i.exp_u64(j as u64);
                } else {
                    *root_j *= root_i;
                }
                *coeff * *root_j
            })
    }));

    ntt_batch(&mut result, coeffs.len());
    transpose(&mut result, expansion, coeffs.len());
    result
}

/// RS encode at a rate 1/`expansion`
pub fn expand_from_coeff_plonky3<F, EF, D>(dft: &D, coeffs: &[EF], expansion: usize) -> Vec<EF>
where
    F: Field + TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    D: TwoAdicSubgroupDft<F>,
{
    let engine = cooley_tukey::NttEngine::<F>::new_from_cache();
    let n = coeffs.len();
    let expanded_size = n * expansion;
    let mut result = Vec::with_capacity(expanded_size);
    // Note: We can also zero-extend the coefficients and do a larger NTT.
    // But this is more efficient.

    // Do coset NTT.
    let root = engine.root(expanded_size);
    result.extend_from_slice(coeffs);
    #[cfg(not(feature = "parallel"))]
    for i in 1..expansion {
        let root = root.exp_u64(i as u64);
        let mut offset = F::ONE;
        result.extend(coeffs.iter().map(|x| {
            let val = *x * offset;
            offset *= root;
            val
        }));
    }
    #[cfg(feature = "parallel")]
    result.par_extend((1..expansion).into_par_iter().flat_map(|i| {
        let root_i = root.exp_u64(i as u64);
        coeffs
            .par_iter()
            .enumerate()
            .map_with(F::ZERO, move |root_j, (j, coeff)| {
                if root_j.is_zero() {
                    *root_j = root_i.exp_u64(j as u64);
                } else {
                    *root_j *= root_i;
                }
                *coeff * *root_j
            })
    }));

    dft.dft_algebra_batch(RowMajorMatrix::new(result, n).transpose())
        .to_row_major_matrix()
        .values
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_dft::NaiveDft;
    use p3_field::PrimeCharacteristicRing;

    use super::*;

    #[test]
    fn test_expand_from_coeff_size_2() {
        let c0 = BabyBear::from_u64(1);
        let c1 = BabyBear::from_u64(2);
        let coeffs = vec![c0, c1];
        let expansion = 2;

        let omega = BabyBear::two_adic_generator(expansion); // ω = primitive 2nd root of unity

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

        let computed_values = expand_from_coeff_plonky3(&NaiveDft, &coeffs, expansion);
        assert_eq!(computed_values, expected_values_transposed);
    }

    #[test]
    fn test_expand_from_coeff_size_4() {
        let c0 = BabyBear::from_u64(1);
        let c1 = BabyBear::from_u64(2);
        let c2 = BabyBear::from_u64(3);
        let c3 = BabyBear::from_u64(4);
        let coeffs = vec![c0, c1, c2, c3];
        let expansion = 4;

        let omega = BabyBear::two_adic_generator(4); // 4th root of unity for expansion 4

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

        let omega = BabyBear::two_adic_generator(2);
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

        let computed_values = expand_from_coeff_plonky3(&NaiveDft, &coeffs, expansion);
        assert_eq!(computed_values, expected_values_transposed);
    }
}
