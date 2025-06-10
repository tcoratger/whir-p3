use p3_field::{ExtensionField, Field};

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
pub fn compute_fold<EF, F>(
    answers: &[F],
    folding_randomness: &[EF],
    mut coset_offset_inv: EF,
    mut coset_gen_inv: EF,
    folding_factor: usize,
) -> EF
where
    EF: Field + ExtensionField<F>,
    F: Field,
{
    assert_eq!(
        answers.len(),
        1 << folding_factor,
        "Invalid number of answers"
    );

    if folding_factor == 0 {
        return EF::from(*answers.first().unwrap());
    }

    // We do the first folding step separately as in this step answers switches
    // from a base field vector to an extension field vector.
    let r = folding_randomness[folding_randomness.len() - 1];
    let half = answers.len() / 2;
    let (lo, hi) = answers.split_at(half);
    let mut answers: Vec<EF> = lo
        .iter()
        .zip(hi)
        .zip(coset_gen_inv.shifted_powers(r * coset_offset_inv))
        .map(|((a, b), point_inv)| {
            let left = *a + *b;
            let right = *a - *b;
            point_inv * right + left
        })
        .collect();

    coset_offset_inv = coset_offset_inv.square();
    coset_gen_inv = coset_gen_inv.square();

    for r in folding_randomness[..folding_randomness.len() - 1]
        .iter()
        .rev()
    {
        let half = answers.len() / 2;
        let (lo, hi) = answers.split_at_mut(half);
        lo.iter_mut()
            .zip(hi)
            .zip(coset_gen_inv.shifted_powers(*r * coset_offset_inv))
            .for_each(|((a, b), point_inv)| {
                let left = *a + *b;
                let right = *a - *b;
                *a = point_inv * right + left;
            });

        answers.truncate(half);
        coset_offset_inv = coset_offset_inv.square();
        coset_gen_inv = coset_gen_inv.square();
    }

    answers.first().unwrap().div_2exp_u64(folding_factor as u64)
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    use super::*;
    use crate::poly::{coeffs::CoefficientList, multilinear::MultilinearPoint};

    type F = BabyBear;

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
        let poly = CoefficientList::new((0..num_coeffs).map(F::from_u64).collect());

        // Root of unity used to generate the evaluation domain
        let root_of_unity = F::from_u64(0x6745_6167);

        // The coset index for evaluation
        let index = 15;

        // Folding randomness values: {0, 1, 2} (since folding_factor = 3)
        let folding_randomness: Vec<_> =
            (0..folding_factor).map(|i| F::from_u64(i as u64)).collect();

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
            &poly_eval,             // Evaluations on coset
            &folding_randomness,    // Folding randomness vector
            coset_offset.inverse(), // Inverse of coset offset
            coset_gen.inverse(),    // Inverse of coset generator
            folding_factor,         // Number of folding steps
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
    fn test_compute_fold_single_layer() {
        // Folding a vector of size 2: f(x) = [1, 3]
        let f0 = F::from_u64(1);
        let f1 = F::from_u64(3);
        let answers = vec![f0, f1];

        let r = F::from_u64(2); // folding randomness
        let folding_randomness = vec![r];

        let coset_offset_inv = F::from_u64(5); // arbitrary inverse offset
        let coset_gen_inv = F::from_u64(7); // arbitrary generator inverse

        // g = (f0 + f1 + r * (f0 - f1) * coset_offset_inv) / 2
        // Here coset_index_inv = 1
        // => left = f0 + f1
        // => right = r * (f0 - f1) * coset_offset_inv
        // => g = (left + right) / 2
        let expected = (f0 + f1 + r * (f0 - f1) * coset_offset_inv).halve();

        let result = compute_fold(
            &answers,
            &folding_randomness,
            coset_offset_inv,
            coset_gen_inv,
            1,
        );
        assert_eq!(result, expected);
    }

    #[test]
    fn test_compute_fold_two_layers() {
        // Define the input evaluations: f(x) = [f00, f01, f10, f11]
        let f00 = F::from_u64(1);
        let f01 = F::from_u64(2);
        let f10 = F::from_u64(3);
        let f11 = F::from_u64(4);

        // Create the input vector for folding
        let answers = vec![f00, f01, f10, f11];

        // Folding randomness used in each layer (innermost first)
        let r0 = F::from_u64(5); // randomness for layer 1 (first fold)
        let r1 = F::from_u64(7); // randomness for layer 2 (second fold)
        let folding_randomness = vec![r1, r0]; // reversed because fold reads from the back

        // Precompute constants
        let coset_offset_inv = F::from_u64(9); // offset⁻¹
        let coset_gen_inv = F::from_u64(3); // generator⁻¹

        // --- First layer of folding ---

        // Fold the pair [f00, f10] using coset_index_inv = 1
        // left = f00 + f10
        let g0_left = f00 + f10;

        // right = (f00 - f10) * coset_offset_inv * coset_index_inv
        // where coset_index_inv = 1
        let g0_right = coset_offset_inv * (f00 - f10);

        // g0 = (left + r0 * right) / 2
        let g0 = (g0_left + r0 * g0_right).halve();

        // Fold the pair [f01, f11] using coset_index_inv = coset_gen_inv
        let coset_index_inv_1 = coset_gen_inv;

        // left = f01 + f11
        let g1_left = f01 + f11;

        // right = (f01 - f11) * coset_offset_inv * coset_index_inv_1
        let g1_right = coset_offset_inv * coset_index_inv_1 * (f01 - f11);

        // g1 = (left + r0 * right) / 2
        let g1 = (g1_left + r0 * g1_right).halve();

        // --- Second layer of folding ---

        // Update the coset offset for next layer: offset⁻¹ → offset⁻¹²
        let next_coset_offset_inv = coset_offset_inv * coset_offset_inv;

        // Fold the pair [g0, g1] using coset_index_inv = 1
        // left = g0 + g1
        let g_final_left = g0 + g1;

        // right = (g0 - g1) * next_coset_offset_inv
        let g_final_right = next_coset_offset_inv * (g0 - g1);

        // Final folded value
        let expected = (g_final_left + r1 * g_final_right).halve();

        // Compute using the actual implementation
        let result = compute_fold(
            &answers,
            &folding_randomness,
            coset_offset_inv,
            coset_gen_inv,
            2,
        );

        // Assert that the result matches the manually computed expected value
        assert_eq!(result, expected);
    }

    #[test]
    fn test_compute_fold_with_zero_randomness() {
        // Inputs: f(x) = [f0, f1]
        let f0 = F::from_u64(6);
        let f1 = F::from_u64(2);
        let answers = vec![f0, f1];

        let r = F::ZERO;
        let folding_randomness = vec![r];

        let coset_offset_inv = F::from_u64(10);
        let coset_gen_inv = F::from_u64(3);

        let left = f0 + f1;
        // with r = 0, this simplifies to (f0 + f1) / 2
        let expected = left.halve();

        let result = compute_fold(
            &answers,
            &folding_randomness,
            coset_offset_inv,
            coset_gen_inv,
            1,
        );

        assert_eq!(result, expected);
    }

    #[test]
    fn test_compute_fold_all_zeros() {
        // All values are zero: f(x) = [0, 0, ..., 0]
        let answers = vec![F::ZERO; 8];
        let folding_randomness = vec![F::from_u64(3); 3];
        let coset_offset_inv = F::from_u64(4);
        let coset_gen_inv = F::from_u64(7);

        // each fold step is (0 + 0 + r * (0 - 0) * _) / 2 = 0
        let expected = F::ZERO;

        let result = compute_fold(
            &answers,
            &folding_randomness,
            coset_offset_inv,
            coset_gen_inv,
            3,
        );

        assert_eq!(result, expected);
    }

    #[test]
    #[should_panic]
    fn test_invalid_answers_length() {
        let answers = vec![F::from_u64(1); 7]; // 7 != 2^3  
        let folding_randomness = vec![F::from_u64(3); 3];
        compute_fold(&answers, &folding_randomness, F::ONE, F::ONE, 3);
    }

    #[test]
    fn test_compute_fold_zero_factor() {
        let answers = vec![F::from_u64(42)];
        let folding_randomness = vec![];
        let result = compute_fold(&answers, &folding_randomness, F::ONE, F::ONE, 0);
        assert_eq!(result, F::from_u64(42));
    }
}
