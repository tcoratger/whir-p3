use p3_field::Field;

/// Computes the equality polynomial evaluations efficiently.
///
/// Given an evaluation point vector `eval`, the function computes
/// the equality polynomial recursively using the formula:
///
/// ```text
/// eq(X) = ∏ (1 - X_i + 2X_i z_i)
/// ```
///
/// where `z_i` are the constraint points.
pub(crate) fn eval_eq<F: Field>(eval: &[F], out: &mut [F], scalar: F) {
    const PARALLEL_THRESHOLD: usize = 10;

    // Ensure that the output buffer size is correct:
    // It should be of size `2^n`, where `n` is the number of variables.
    debug_assert_eq!(out.len(), 1 << eval.len());

    // Base case: When there are no more variables to process, update the final value.
    if let Some((&x, tail)) = eval.split_first() {
        // Divide the output buffer into two halves: one for `X_i = 0` and one for `X_i = 1`
        let (low, high) = out.split_at_mut(out.len() / 2);

        // Compute weight updates for the two branches:
        // - `s0` corresponds to the case when `X_i = 0`
        // - `s1` corresponds to the case when `X_i = 1`
        //
        // Mathematically, this follows the recurrence:
        // ```text
        // eq_{X1, ..., Xn}(X) = (1 - X_1) * eq_{X2, ..., Xn}(X) + X_1 * eq_{X2, ..., Xn}(X)
        // ```
        let s1 = scalar * x; // Contribution when `X_i = 1`
        let s0 = scalar - s1; // Contribution when `X_i = 0`

        // Use parallel execution if the number of remaining variables is large.
        #[cfg(feature = "parallel")]
        {
            const PARALLEL_THRESHOLD: usize = 10;
            if tail.len() > PARALLEL_THRESHOLD {
                rayon::join(|| eval_eq(tail, low, s0), || eval_eq(tail, high, s1));
                return;
            }
        }

        // Default sequential execution
        eval_eq(tail, low, s0);
        eval_eq(tail, high, s1);
    } else {
        // Leaf case: Add the accumulated scalar to the final output slot.
        out[0] += scalar;
    }
}

/// Generates a sequence of powers of `base`, starting from `1`.
///
/// This function returns a vector containing the sequence:
/// `[1, base, base^2, base^3, ..., base^(len-1)]`
pub fn expand_randomness<F: Field>(base: F, len: usize) -> Vec<F> {
    let mut res = Vec::with_capacity(len);
    let mut acc = F::ONE;
    for _ in 0..len {
        res.push(acc);
        acc *= base;
    }
    res
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    use super::*;

    type F = BabyBear;

    #[test]
    fn test_eval_eq_functionality() {
        let mut output = vec![F::ZERO; 4]; // n=2 → 2^2 = 4 elements
        let eval = vec![F::from_u64(1), F::from_u64(0)]; // (X1, X2) = (1,0)
        let scalar = F::from_u64(2);

        eval_eq(&eval, &mut output, scalar);

        // Expected results for (X1, X2) = (1,0)
        let expected_output = vec![F::ZERO, F::ZERO, F::from_u64(2), F::ZERO];

        assert_eq!(output, expected_output);
    }

    #[test]
    fn test_expand_randomness_basic() {
        // Test with base = 2 and length = 5
        let base = F::from_u64(2);
        let len = 5;

        let expected = vec![
            F::ONE,
            F::from_u64(2),
            F::from_u64(4),
            F::from_u64(8),
            F::from_u64(16),
        ];

        assert_eq!(expand_randomness(base, len), expected);
    }

    #[test]
    fn test_expand_randomness_zero_length() {
        // If len = 0, should return an empty vector
        let base = F::from_u64(3);
        assert!(expand_randomness(base, 0).is_empty());
    }

    #[test]
    fn test_expand_randomness_one_length() {
        // If len = 1, should return [1]
        let base = F::from_u64(5);
        assert_eq!(expand_randomness(base, 1), vec![F::ONE]);
    }

    #[test]
    fn test_expand_randomness_large_base() {
        // Test with a large base value
        let base = F::from_u64(10);
        let len = 4;

        let expected = vec![F::ONE, F::from_u64(10), F::from_u64(100), F::from_u64(1000)];

        assert_eq!(expand_randomness(base, len), expected);
    }

    #[test]
    fn test_expand_randomness_identity_case() {
        // If base = 1, all values should be 1
        let base = F::ONE;
        let len = 6;

        let expected = vec![F::ONE; len];
        assert_eq!(expand_randomness(base, len), expected);
    }

    #[test]
    fn test_expand_randomness_zero_base() {
        // If base = 0, all values after the first should be 0
        let base = F::ZERO;
        let len = 5;

        let expected = vec![F::ONE, F::ZERO, F::ZERO, F::ZERO, F::ZERO];
        assert_eq!(expand_randomness(base, len), expected);
    }

    #[test]
    fn test_expand_randomness_negative_base() {
        // Test with base = -1, which should alternate between 1 and -1
        let base = -F::ONE;
        let len = 6;

        let expected = vec![F::ONE, -F::ONE, F::ONE, -F::ONE, F::ONE, -F::ONE];

        assert_eq!(expand_randomness(base, len), expected);
    }
}
