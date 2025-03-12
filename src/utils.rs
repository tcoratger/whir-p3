use p3_field::Field;

use crate::ntt::transpose::transpose;

/// Returns true if `n`:
/// - is a power of two
/// - is greater than zero
pub const fn is_power_of_two(n: usize) -> bool {
    n != 0 && n.is_power_of_two()
}

/// Stacks evaluations by grouping them into cosets and transposing in-place.
///
/// Given `evals[i] = f(ω^i)`, reorganizes values into `2^folding_factor` cosets.
/// The transformation follows:
///
/// ```
/// stacked[i, j] = f(ω^(i + j * (N / 2^folding_factor)))
/// ```
///
/// The input length must be a multiple of `2^folding_factor`.
pub fn stack_evaluations<F: Field>(mut evals: Vec<F>, folding_factor: usize) -> Vec<F> {
    let folding_factor_exp = 1 << folding_factor;
    assert!(evals.len() % folding_factor_exp == 0);
    let size_of_new_domain = evals.len() / folding_factor_exp;

    // Interpret evals as (folding_factor_exp x size_of_new_domain)-matrix and transpose in-place
    transpose(&mut evals, folding_factor_exp, size_of_new_domain);
    evals
}

#[cfg(test)]
#[allow(clippy::should_panic_without_expect)]
mod tests {
    use super::*;
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    #[test]
    fn test_is_power_of_two() {
        assert!(!is_power_of_two(0));
        assert!(is_power_of_two(1));
        assert!(is_power_of_two(2));
        assert!(!is_power_of_two(3));
        assert!(!is_power_of_two(usize::MAX));
    }

    #[test]
    fn test_evaluations_stack() {
        let num = 256;
        let folding_factor = 3;
        let fold_size = 1 << folding_factor;
        assert_eq!(num % fold_size, 0);
        let evals: Vec<_> = (0..num as u64).map(BabyBear::from_u64).collect();

        let stacked = stack_evaluations(evals, folding_factor);
        assert_eq!(stacked.len(), num);

        for (i, fold) in stacked.chunks_exact(fold_size).enumerate() {
            assert_eq!(fold.len(), fold_size);
            for (j, &f) in fold.iter().enumerate().take(fold_size) {
                assert_eq!(f, BabyBear::from_u64((i + j * num / fold_size) as u64));
            }
        }
    }

    #[test]
    fn test_stack_evaluations_basic() {
        // Basic test with 8 elements and folding factor of 2 (groups of 4)
        let evals: Vec<_> = (0..8).map(BabyBear::from_u64).collect();
        let folding_factor = 2;

        let stacked = stack_evaluations(evals.clone(), folding_factor);

        // Check that the length remains unchanged after transformation
        assert_eq!(stacked.len(), evals.len());

        // Original matrix before stacking (4 rows, 2 columns):
        // 0  1
        // 2  3
        // 4  5
        // 6  7
        //
        // After transposition:
        // 0  2  4  6
        // 1  3  5  7
        let expected: Vec<_> =
            vec![0, 2, 4, 6, 1, 3, 5, 7].into_iter().map(BabyBear::from_u64).collect();

        assert_eq!(stacked, expected);
    }

    #[test]
    fn test_stack_evaluations_power_of_two() {
        // Test with 16 elements and a folding factor of 3 (groups of 8)
        let evals: Vec<_> = (0..16).map(BabyBear::from_u64).collect();
        let folding_factor = 3;

        let stacked = stack_evaluations(evals.clone(), folding_factor);

        // Ensure the length remains unchanged
        assert_eq!(stacked.len(), evals.len());

        // Original matrix (8 rows, 2 columns):
        //  0   1
        //  2   3
        //  4   5
        //  6   7
        //  8   9
        // 10  11
        // 12  13
        // 14  15
        //
        // After stacking:
        //  0   2   4   6   8  10  12  14
        //  1   3   5   7   9  11  13  15
        let expected: Vec<_> = vec![0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15]
            .into_iter()
            .map(BabyBear::from_u64)
            .collect();

        assert_eq!(stacked, expected);
    }

    #[test]
    fn test_stack_evaluations_identity_case() {
        // When folding_factor is 0, the function should return the input unchanged
        let evals: Vec<_> = (0..4).map(BabyBear::from_u64).collect();
        let folding_factor = 0;

        let stacked = stack_evaluations(evals.clone(), folding_factor);

        // Expected result: No change
        assert_eq!(stacked, evals);
    }

    #[test]
    fn test_stack_evaluations_large_case() {
        // Test with 32 elements and a folding factor of 4 (groups of 16)
        let evals: Vec<_> = (0..32).map(BabyBear::from_u64).collect();
        let folding_factor = 4;

        let stacked = stack_evaluations(evals.clone(), folding_factor);

        // Ensure the length remains unchanged
        assert_eq!(stacked.len(), evals.len());

        // Original matrix before stacking (16 rows, 2 columns):
        //  0   1
        //  2   3
        //  4   5
        //  6   7
        //  8   9
        // 10  11
        // 12  13
        // 14  15
        // 16  17
        // 18  19
        // 20  21
        // 22  23
        // 24  25
        // 26  27
        // 28  29
        // 30  31
        //
        // After stacking:
        //  0   2   4   6   8  10  12  14  16  18  20  22  24  26  28  30
        //  1   3   5   7   9  11  13  15  17  19  21  23  25  27  29  31
        let expected: Vec<_> = vec![
            0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 1, 3, 5, 7, 9, 11, 13, 15,
            17, 19, 21, 23, 25, 27, 29, 31,
        ]
        .into_iter()
        .map(BabyBear::from_u64)
        .collect();

        assert_eq!(stacked, expected);
    }

    #[test]
    #[should_panic]
    fn test_stack_evaluations_invalid_size() {
        let evals: Vec<_> = (0..10).map(BabyBear::from_u64).collect();
        let folding_factor = 2; // folding size = 4, but 10 is not divisible by 4
        stack_evaluations(evals, folding_factor);
    }
}
