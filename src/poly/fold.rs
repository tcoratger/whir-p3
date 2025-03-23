use p3_field::{Field, TwoAdicField};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{
    ntt::{cooley_tukey::intt_batch, transpose::transpose},
    parameters::FoldType,
};

/// Computes the folded value of a function evaluated on a coset.
///
/// This function applies a recursive folding transformation to a given set of function
/// evaluations on a coset, progressively reducing the number of evaluations while incorporating
/// randomness and coset transformations. The folding process is performed `folding_factor` times,
/// halving the number of evaluations at each step.
///
/// Mathematical Formulation:
/// Given an initial evaluation vector:
/// \begin{equation}
/// f(x) = [f_0, f_1, ..., f_{2^m - 1}]
/// \end{equation}
///
/// Each folding step computes:
/// \begin{equation}
/// g_i = \frac{f_i + f_{i + N/2} + r \cdot (f_i - f_{i + N/2}) \cdot (o^{-1} \cdot g^{-i})}{2}
/// \end{equation}
///
/// where:
/// - \( r \) is the folding randomness
/// - \( o^{-1} \) is the inverse coset offset
/// - \( g^{-i} \) is the inverse generator raised to index \( i \)
/// - The function is recursively applied until the vector reduces to size 1.
pub fn compute_fold<F: Field>(
    answers: &[F],
    folding_randomness: &[F],
    mut coset_offset_inv: F,
    mut coset_gen_inv: F,
    two_inv: F,
    folding_factor: usize,
) -> F {
    let mut answers = answers.to_vec();

    // Perform the folding process `folding_factor` times.
    for rec in 0..folding_factor {
        let r = folding_randomness[folding_randomness.len() - 1 - rec];
        let offset = answers.len() / 2;
        let mut coset_index_inv = F::ONE;

        // Compute the new folded values, iterating over the first half of `answers`.
        for i in 0..offset {
            let f0 = answers[i];
            let f1 = answers[i + offset];
            let point_inv = coset_offset_inv * coset_index_inv;

            let left = f0 + f1;
            let right = point_inv * (f0 - f1);

            // Apply the folding transformation with randomness
            answers[i] = two_inv * (left + r * right);
            coset_index_inv *= coset_gen_inv;
        }

        // Reduce answers to half its size without allocating a new vector
        answers.truncate(offset);

        // Update for next iteration
        coset_offset_inv *= coset_offset_inv;
        coset_gen_inv *= coset_gen_inv;
    }

    answers[0]
}

/// Applies a folding transformation to evaluation vectors in-place.
///
/// This is used to prepare a set of evaluations for a sumcheck-style polynomial folding,
/// supporting two modes:
///
/// - `FoldType::Naive`: applies only the reshaping step (transposition).
/// - `FoldType::ProverHelps`: performs reshaping, inverse NTTs, and applies coset + scaling
///   correction.
///
/// The evaluations are grouped into `2^folding_factor` blocks of size `N / 2^folding_factor`.
/// For each group, the function performs the following (if `ProverHelps`):
///
/// 1. Transpose: reshape layout to enable independent processing of each sub-coset.
/// 2. Inverse NTT: convert each sub-coset from evaluation to coefficient form (no 1/N scaling).
/// 3. Scale correction: Each output is multiplied by:
///
///    ```ignore
///    size_inv * (domain_gen_inv^i)^j
///    ```
///
///    where:
///      - `size_inv = 1 / 2^folding_factor`
///      - `i` is the subdomain index
///      - `j` is the index within the block
///
/// # Panics
/// Panics if the input size is not divisible by `2^folding_factor`.
pub fn transform_evaluations<F: Field + TwoAdicField>(
    evals: &mut [F],
    fold_type: FoldType,
    _domain_gen: F,
    domain_gen_inv: F,
    folding_factor: usize,
) {
    // Compute the number of sub-cosets = 2^folding_factor
    let folding_factor_exp = 1 << folding_factor;

    // Ensure input is divisible by folding factor
    assert!(evals.len() % folding_factor_exp == 0);

    // Number of rows (one per subdomain)
    let size_of_new_domain = evals.len() / folding_factor_exp;

    match fold_type {
        FoldType::Naive => {
            // Simply transpose into column-major form: shape = [folding_factor_exp ×
            // size_of_new_domain]
            transpose(evals, folding_factor_exp, size_of_new_domain);
        }
        FoldType::ProverHelps => {
            // Step 1: Reshape via transposition
            transpose(evals, folding_factor_exp, size_of_new_domain);

            // Step 2: Apply inverse NTTs
            intt_batch(evals, folding_factor_exp);

            // Step 3: Apply scaling to match the desired domain layout
            // Each value is scaled by: size_inv * offset^j
            let size_inv = F::from_u64(folding_factor_exp as u64).inverse();
            #[cfg(not(feature = "parallel"))]
            {
                let mut coset_offset_inv = F::ONE;
                for answers in evals.chunks_exact_mut(folding_factor_exp) {
                    let mut scale = size_inv;
                    for v in answers.iter_mut() {
                        *v *= scale;
                        scale *= coset_offset_inv;
                    }
                    coset_offset_inv *= domain_gen_inv;
                }
            }
            #[cfg(feature = "parallel")]
            evals.par_chunks_exact_mut(folding_factor_exp).enumerate().for_each_with(
                F::ZERO,
                |offset, (i, answers)| {
                    if *offset == F::ZERO {
                        *offset = domain_gen_inv.exp_u64(i as u64);
                    } else {
                        *offset *= domain_gen_inv;
                    }
                    let mut scale = size_inv;
                    for v in answers.iter_mut() {
                        *v *= scale;
                        scale *= *offset;
                    }
                },
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    use super::*;
    use crate::poly::{coeffs::CoefficientList, multilinear::MultilinearPoint};

    #[test]
    fn test_folding() {
        // Number of variables in the multilinear polynomial
        let num_variables = 5;

        // Number of coefficients in the polynomial (since it's multilinear, it's 2^num_variables)
        let num_coeffs = 1 << num_variables;

        // Domain size for evaluation (fixed to 256)
        let domain_size = 256;

        // Folding factor: 3 means we fold in 8 (2^3)
        let folding_factor = 3;
        let folding_factor_exp = 1 << folding_factor; // 2^folding_factor = 8

        // Construct a multilinear polynomial where coefficients are just consecutive integers
        let poly = CoefficientList::new((0..num_coeffs).map(BabyBear::from_u64).collect());

        // Root of unity used to generate the evaluation domain
        let root_of_unity = BabyBear::from_u64(0x6745_6167);

        // The coset index for evaluation
        let index = 15;

        // Folding randomness values: {0, 1, 2} (since folding_factor = 3)
        let folding_randomness: Vec<_> =
            (0..folding_factor).map(|i| BabyBear::from_u64(i as u64)).collect();

        // Compute coset offset: ω^index
        let coset_offset = root_of_unity.exp_u64(index);

        // Compute coset generator: ω^(N / 2^folding_factor) = ω^(256 / 8) = ω^32
        let coset_gen = root_of_unity.exp_u64(domain_size / folding_factor_exp);

        // Step 1: Evaluate the polynomial on the coset domain
        // We evaluate at points: {coset_offset * coset_gen^i} for i in {0, 1, ..., 7}
        let poly_eval: Vec<_> = (0..folding_factor_exp)
            .map(|i| {
                poly.evaluate(&MultilinearPoint::expand_from_univariate(
                    coset_offset * coset_gen.exp_u64(i),
                    num_variables,
                ))
            })
            .collect();

        // Step 2: Compute the folded value using the `compute_fold` function
        let fold_value = compute_fold(
            &poly_eval,                      // Evaluations on coset
            &folding_randomness,             // Folding randomness vector
            coset_offset.inverse(),          // Inverse of coset offset
            coset_gen.inverse(),             // Inverse of coset generator
            BabyBear::from_u64(2).inverse(), // Scaling factor (1/2)
            folding_factor,                  // Number of folding steps
        );

        // Step 3: Compute the expected folded value using the polynomial's `fold` function
        let truth_value = poly
            .fold(&MultilinearPoint(folding_randomness)) // Fold the polynomial
            .evaluate(&MultilinearPoint::expand_from_univariate(
                root_of_unity.exp_u64(folding_factor_exp * index),
                2,
            ));

        // Step 4: Ensure computed and expected values match
        assert_eq!(fold_value, truth_value);
    }

    #[test]
    fn test_folding_optimised() {
        // Number of variables in the multilinear polynomial
        let num_variables = 5;

        // Number of coefficients in the polynomial (since it's multilinear, it's 2^num_variables)
        let num_coeffs = 1 << num_variables;

        // Domain size for evaluation (fixed to 256)
        let domain_size = 256;

        // Folding factor: 3 means we fold in 8 (2^3)
        let folding_factor = 3;
        let folding_factor_exp = 1 << folding_factor; // 2^folding_factor = 8

        // Construct a multilinear polynomial where coefficients are just consecutive integers
        let poly = CoefficientList::new((0..num_coeffs).map(BabyBear::from_u64).collect());

        // Root of unity used to generate the evaluation domain
        let root_of_unity = BabyBear::from_u64(0x5ee9_9486);

        // Compute the inverse of the root of unity
        let root_of_unity_inv = root_of_unity.inverse();

        // Folding randomness values: {0, 1, 2} (since folding_factor = 3)
        let folding_randomness: Vec<_> =
            (0..folding_factor).map(|i| BabyBear::from_u64(i as u64)).collect();

        // Step 1: Evaluate the polynomial on the entire domain of size 256
        // We evaluate at points: {ω^w} for w in {0, 1, ..., 255}
        let mut domain_evaluations: Vec<_> = (0..domain_size)
            .map(|w| root_of_unity.exp_u64(w))
            .map(|point| {
                poly.evaluate(&MultilinearPoint::expand_from_univariate(point, num_variables))
            })
            .collect();

        // Step 2: Stack evaluations into groups of size 2^folding_factor = 8
        // This groups evaluations for efficient folding
        let mut unprocessed = domain_evaluations.clone();
        transpose(
            &mut unprocessed,
            folding_factor_exp as usize,
            domain_evaluations.len() / folding_factor_exp as usize,
        );

        // Step 3: Restructure evaluations using the `ProverHelps` folding strategy
        transform_evaluations(
            &mut domain_evaluations,
            crate::parameters::FoldType::ProverHelps,
            root_of_unity,
            root_of_unity_inv,
            folding_factor,
        );

        // Step 4: Compute expected folded values and compare against processed results
        let num = domain_size / folding_factor_exp; // Number of cosets (256 / 8 = 32)
        let coset_gen_inv = root_of_unity_inv.exp_u64(num); // Compute inverse coset generator

        for index in 0..num {
            // Compute the coset offset inverse: ω^{-index}
            let offset_inv = root_of_unity_inv.exp_u64(index);

            // Define the range of evaluations corresponding to the current coset
            let span =
                (index * folding_factor_exp) as usize..((index + 1) * folding_factor_exp) as usize;

            // Compute the folded result using `compute_fold`
            let answer_unprocessed = compute_fold(
                &unprocessed[span.clone()],      // Extract the evaluations for this coset
                &folding_randomness,             // Folding randomness vector
                offset_inv,                      // Coset offset inverse
                coset_gen_inv,                   // Coset generator inverse
                BabyBear::from_u64(2).inverse(), // Scaling factor (1/2)
                folding_factor,                  // Number of folding steps
            );

            // Compute the expected folded value using the processed evaluations
            let answer_processed = CoefficientList::new(domain_evaluations[span].to_vec())
                .evaluate(&MultilinearPoint(folding_randomness.clone()));

            // Ensure computed and expected values match
            assert_eq!(answer_processed, answer_unprocessed);
        }
    }

    #[test]
    fn test_compute_fold_single_layer() {
        // Folding a vector of size 2: f(x) = [1, 3]
        let f0 = BabyBear::from_u64(1);
        let f1 = BabyBear::from_u64(3);
        let answers = vec![f0, f1];

        let r = BabyBear::from_u64(2); // folding randomness
        let folding_randomness = vec![r];

        let coset_offset_inv = BabyBear::from_u64(5); // arbitrary inverse offset
        let coset_gen_inv = BabyBear::from_u64(7); // arbitrary generator inverse
        let two_inv = BabyBear::from_u64(2).inverse();

        // g = (f0 + f1 + r * (f0 - f1) * coset_offset_inv) / 2
        // Here coset_index_inv = 1
        // => left = f0 + f1
        // => right = r * (f0 - f1) * coset_offset_inv
        // => g = (left + right) / 2
        let expected = two_inv * (f0 + f1 + r * (f0 - f1) * coset_offset_inv);

        let result = compute_fold(
            &answers,
            &folding_randomness,
            coset_offset_inv,
            coset_gen_inv,
            two_inv,
            1,
        );
        assert_eq!(result, expected);
    }

    #[test]
    fn test_compute_fold_two_layers() {
        // Define the input evaluations: f(x) = [f00, f01, f10, f11]
        let f00 = BabyBear::from_u64(1);
        let f01 = BabyBear::from_u64(2);
        let f10 = BabyBear::from_u64(3);
        let f11 = BabyBear::from_u64(4);

        // Create the input vector for folding
        let answers = vec![f00, f01, f10, f11];

        // Folding randomness used in each layer (innermost first)
        let r0 = BabyBear::from_u64(5); // randomness for layer 1 (first fold)
        let r1 = BabyBear::from_u64(7); // randomness for layer 2 (second fold)
        let folding_randomness = vec![r1, r0]; // reversed because fold reads from the back

        // Precompute constants
        let two_inv = BabyBear::from_u64(2).inverse(); // 1/2 used in folding formula
        let coset_offset_inv = BabyBear::from_u64(9); // offset⁻¹
        let coset_gen_inv = BabyBear::from_u64(3); // generator⁻¹

        // --- First layer of folding ---

        // Fold the pair [f00, f10] using coset_index_inv = 1
        // left = f00 + f10
        let g0_left = f00 + f10;

        // right = (f00 - f10) * coset_offset_inv * coset_index_inv
        // where coset_index_inv = 1
        let g0_right = coset_offset_inv * (f00 - f10);

        // g0 = (left + r0 * right) / 2
        let g0 = two_inv * (g0_left + r0 * g0_right);

        // Fold the pair [f01, f11] using coset_index_inv = coset_gen_inv
        let coset_index_inv_1 = coset_gen_inv;

        // left = f01 + f11
        let g1_left = f01 + f11;

        // right = (f01 - f11) * coset_offset_inv * coset_index_inv_1
        let g1_right = coset_offset_inv * coset_index_inv_1 * (f01 - f11);

        // g1 = (left + r0 * right) / 2
        let g1 = two_inv * (g1_left + r0 * g1_right);

        // --- Second layer of folding ---

        // Update the coset offset for next layer: offset⁻¹ → offset⁻¹²
        let next_coset_offset_inv = coset_offset_inv * coset_offset_inv;

        // Fold the pair [g0, g1] using coset_index_inv = 1
        // left = g0 + g1
        let g_final_left = g0 + g1;

        // right = (g0 - g1) * next_coset_offset_inv
        let g_final_right = next_coset_offset_inv * (g0 - g1);

        // Final folded value
        let expected = two_inv * (g_final_left + r1 * g_final_right);

        // Compute using the actual implementation
        let result = compute_fold(
            &answers,
            &folding_randomness,
            coset_offset_inv,
            coset_gen_inv,
            two_inv,
            2,
        );

        // Assert that the result matches the manually computed expected value
        assert_eq!(result, expected);
    }

    #[test]
    fn test_compute_fold_with_zero_randomness() {
        // Inputs: f(x) = [f0, f1]
        let f0 = BabyBear::from_u64(6);
        let f1 = BabyBear::from_u64(2);
        let answers = vec![f0, f1];

        let r = BabyBear::ZERO;
        let folding_randomness = vec![r];

        let two_inv = BabyBear::from_u64(2).inverse();
        let coset_offset_inv = BabyBear::from_u64(10);
        let coset_gen_inv = BabyBear::from_u64(3);

        let left = f0 + f1;
        // with r = 0, this simplifies to (f0 + f1) / 2
        let expected = two_inv * left;

        let result = compute_fold(
            &answers,
            &folding_randomness,
            coset_offset_inv,
            coset_gen_inv,
            two_inv,
            1,
        );

        assert_eq!(result, expected);
    }

    #[test]
    fn test_compute_fold_all_zeros() {
        // All values are zero: f(x) = [0, 0, ..., 0]
        let answers = vec![BabyBear::ZERO; 8];
        let folding_randomness = vec![BabyBear::from_u64(3); 3];
        let two_inv = BabyBear::from_u64(2).inverse();
        let coset_offset_inv = BabyBear::from_u64(4);
        let coset_gen_inv = BabyBear::from_u64(7);

        // each fold step is (0 + 0 + r * (0 - 0) * _) / 2 = 0
        let expected = BabyBear::ZERO;

        let result = compute_fold(
            &answers,
            &folding_randomness,
            coset_offset_inv,
            coset_gen_inv,
            two_inv,
            3,
        );

        assert_eq!(result, expected);
    }

    #[test]
    fn test_transform_evaluations_naive_manual() {
        // Input: 2×2 matrix (folding_factor = 1 → folding_factor_exp = 2)
        // Stored row-major as: [a00, a01, a10, a11]
        // We expect it to transpose to: [a00, a10, a01, a11]

        let folding_factor = 1;
        let mut evals = vec![
            BabyBear::from_u64(1), // a00
            BabyBear::from_u64(2), // a01
            BabyBear::from_u64(3), // a10
            BabyBear::from_u64(4), // a11
        ];

        // Expected transpose: column-major layout
        let expected = vec![
            BabyBear::from_u64(1), // a00
            BabyBear::from_u64(3), // a10
            BabyBear::from_u64(2), // a01
            BabyBear::from_u64(4), // a11
        ];

        // Naive transpose — only reshuffling the data
        transform_evaluations(
            &mut evals,
            FoldType::Naive,
            BabyBear::ONE,
            BabyBear::ONE,
            folding_factor,
        );

        assert_eq!(evals, expected);
    }

    #[test]
    #[allow(clippy::cast_sign_loss)]
    fn test_transform_evaluations_prover_helps() {
        // Setup:
        // - 2×2 matrix: 2 cosets (rows), each with 2 points
        // - folding_factor = 1 → folding_factor_exp = 2
        let folding_factor = 1;
        let folding_factor_exp = 1 << folding_factor;

        // Domain generator and its inverse (arbitrary but consistent)
        let domain_gen = BabyBear::from_u64(4);
        let domain_gen_inv = domain_gen.inverse();

        // Row-major input:
        //   row 0: [a0, a1]
        //   row 1: [b0, b1]
        let a0 = BabyBear::from_u64(1);
        let a1 = BabyBear::from_u64(3);
        let b0 = BabyBear::from_u64(2);
        let b1 = BabyBear::from_u64(4);
        let mut evals = vec![a0, a1, b0, b1];

        // Step 1: Transpose (rows become columns)
        // After transpose: [a0, b0, a1, b1]
        let t0 = a0;
        let t1 = b0;
        let t2 = a1;
        let t3 = b1;

        // Step 2: Inverse NTT on each row (length 2) without scaling
        //
        // For [x0, x1], inverse NTT without scaling is:
        //     [x0 + x1, x0 - x1]
        //
        // Row 0: [a0, b0] → [a0 + b0, a0 - b0]
        let intt0 = t0 + t1;
        let intt1 = t0 - t1;

        // Row 1: [a1, b1] → [a1 + b1, a1 - b1]
        let intt2 = t2 + t3;
        let intt3 = t2 - t3;

        // Step 3: Apply scaling
        //
        // Each row is scaled by:
        //    v[j] *= size_inv * (coset_offset_inv)^j
        //
        // For row 0, offset = 1 (coset_offset_inv^j = 1)
        // For row 1, offset = domain_gen_inv
        let size_inv = BabyBear::from_u64(folding_factor_exp as u64).inverse();

        let expected = vec![
            intt0 * size_inv * BabyBear::ONE,
            intt1 * size_inv * BabyBear::ONE,
            intt2 * size_inv * BabyBear::ONE, // first entry of row 1, scale = 1
            intt3 * size_inv * domain_gen_inv, // second entry of row 1
        ];

        // Run transform
        transform_evaluations(
            &mut evals,
            FoldType::ProverHelps,
            domain_gen,
            domain_gen_inv,
            folding_factor,
        );

        // Validate output
        assert_eq!(evals, expected);
    }

    #[test]
    #[should_panic]
    fn test_transform_evaluations_invalid_length() {
        let mut evals = vec![BabyBear::from_u64(1); 6]; // Not a power of 2
        transform_evaluations(&mut evals, FoldType::Naive, BabyBear::ONE, BabyBear::ONE, 2);
    }
}
