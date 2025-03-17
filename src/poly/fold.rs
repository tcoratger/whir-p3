use crate::{
    ntt::{cooley_tukey::intt_batch, transpose::transpose},
    parameters::FoldType,
};
use p3_field::{Field, TwoAdicField};
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
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
            answers[i] =
                two_inv * (left + folding_randomness[folding_randomness.len() - 1 - rec] * right);
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

pub fn transform_evaluations<F: Field + TwoAdicField>(
    evals: &mut [F],
    fold_type: FoldType,
    _domain_gen: F,
    domain_gen_inv: F,
    folding_factor: usize,
) {
    let folding_factor_exp = 1 << folding_factor;
    assert!(evals.len() % folding_factor_exp == 0);
    let size_of_new_domain = evals.len() / folding_factor_exp;

    match fold_type {
        FoldType::Naive => {
            // Perform only the stacking step (transpose)
            transpose(evals, folding_factor_exp, size_of_new_domain);
        }
        FoldType::ProverHelps => {
            // Perform stacking (transpose)
            transpose(evals, folding_factor_exp, size_of_new_domain);

            // Batch inverse NTTs
            intt_batch(evals, folding_factor_exp);

            // Apply coset and size correction.
            // Stacked evaluation at i is f(B_l) where B_l = w^i * <w^n/k>
            let size_inv = F::from_u64(folding_factor_exp as u64).inverse();
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
    use super::*;
    use crate::poly::{coeffs::CoefficientList, multilinear::MultilinearPoint};
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

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
}
