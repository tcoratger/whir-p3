use p3_field::Field;

/// Decomposes `value` into its base-`base` representation with `n_bits` digits.
/// The result follows big-endian order:
/// ```ignore
/// value = v[0] * base^(n_bits-1) + v[1] * base^(n_bits-2) + ... + v[n_bits-1] * 1
/// ```
/// where each `v[i]` is in `0..base`. Always returns exactly `n_bits` digits (padded with 0).
#[must_use]
pub fn base_decomposition(mut value: usize, base: u8, n_bits: usize) -> Vec<u8> {
    debug_assert!(base > 1, "Base must be at least 2");

    // Preallocate a vector with zeroes, ensuring it's exactly `n_bits` long
    let mut result = vec![0u8; n_bits];

    // Compute the decomposition in reverse order (avoids shifting later)
    for digit in result.iter_mut().rev() {
        *digit = (value % (base as usize)) as u8;
        value /= base as usize;
    }

    result
}

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
    fn test_base_decomposition_binary() {
        // Base-2 decomposition (big-endian, padded to n_bits)
        // 11 in binary (6-bit representation): 001011
        assert_eq!(base_decomposition(0b1011, 2, 6), vec![0, 0, 1, 0, 1, 1]);

        // 5 in binary (4-bit representation): 0101
        assert_eq!(base_decomposition(5, 2, 4), vec![0, 1, 0, 1]);

        // 10 in binary (4-bit representation): 1010
        assert_eq!(base_decomposition(10, 2, 4), vec![1, 0, 1, 0]);

        // 0 in binary (4-bit representation): 0000
        assert_eq!(base_decomposition(0, 2, 4), vec![0, 0, 0, 0]);
    }

    #[test]
    fn test_base_decomposition_ternary() {
        // Base-3 decomposition (big-endian, padded to n_bits)
        // 15 in base-3 (3-digit representation): 120
        // Computation:
        // 15 / 3 = 5 remainder 0
        // 5 / 3  = 1 remainder 2
        // 1 / 3  = 0 remainder 1
        // Result = [1, 2, 0]
        assert_eq!(base_decomposition(15, 3, 3), vec![1, 2, 0]);

        // 8 in base-3 (4-digit representation): 0022
        // Computation:
        // 8 / 3 = 2 remainder 2
        // 2 / 3 = 0 remainder 2
        // Result = [0, 0, 2, 2] (padded to 4 bits)
        assert_eq!(base_decomposition(8, 3, 4), vec![0, 0, 2, 2]);
    }

    #[test]
    fn test_base_decomposition_large_values() {
        // Base-5 decomposition (4-digit representation of 123)
        // Computation:
        // 123 / 5 = 24 remainder 3
        // 24 / 5  = 4 remainder 4
        // 4 / 5   = 0 remainder 4
        // Result = [0, 4, 4, 3] (padded to 4 bits)
        assert_eq!(base_decomposition(123, 5, 4), vec![0, 4, 4, 3]);

        // Base-7 decomposition (5-digit representation of 100)
        // Computation:
        // 100 / 7 = 14 remainder 2
        // 14 / 7  = 2 remainder 0
        // 2 / 7   = 0 remainder 2
        // Result = [0, 0, 2, 0, 2] (padded to 5 bits)
        assert_eq!(base_decomposition(100, 7, 5), vec![0, 0, 2, 0, 2]);
    }

    #[test]
    fn test_base_decomposition_padding() {
        // Ensure correct padding when value is smaller than max base^n_bits
        // Base-4 decomposition (5 in base-4 with 5-bit representation)
        // 5 in base-4: 11 → padded to [0, 0, 0, 1, 1]
        assert_eq!(base_decomposition(5, 4, 5), vec![0, 0, 0, 1, 1]);

        // 2 in base-3 (4-bit representation): 0002
        assert_eq!(base_decomposition(2, 3, 4), vec![0, 0, 0, 2]);

        // 0 in base-5 (4-bit representation): 0000
        assert_eq!(base_decomposition(0, 5, 4), vec![0, 0, 0, 0]);
    }

    #[test]
    fn test_base_decomposition_edge_cases() {
        // Edge case: Maximum value within n_bits range
        // Base-2 decomposition of 15 in 4-bit representation: 1111
        assert_eq!(base_decomposition(15, 2, 4), vec![1, 1, 1, 1]);

        // Edge case: Maximum value within n_bits range for base-3
        // Base-3 decomposition of 26 in 3-bit representation (26 mod 3^3)
        // 26 in base-3: 222 → padded to 3 bits
        assert_eq!(base_decomposition(26, 3, 3), vec![2, 2, 2]);

        // Edge case: Maximum value that can fit in n_bits
        // Base-4 decomposition of 63 in 3-bit representation (63 mod 4^3)
        // 63 in base-4: 333 → padded to 3 bits
        assert_eq!(base_decomposition(63, 4, 3), vec![3, 3, 3]);
    }

    #[test]
    fn test_base_decomposition_truncation_behavior() {
        // Ensure truncation when value exceeds `base^n_bits`
        // (Undocumented but current behavior: computes modulo base^n_bits)
        //
        // Base-3 decomposition of (15 + 81) with 3-bit representation
        // (15 + 81) = 96, which is equivalent to (15 mod 27) = 15
        // 15 in base-3: 120
        assert_eq!(base_decomposition(15 + 81, 3, 3), vec![1, 2, 0]);

        // Base-2 decomposition of (20 + 16) with 4-bit representation
        // (20 + 16) = 36, which is equivalent to (20 mod 16) = 4
        // 4 in base-2: 0100
        assert_eq!(base_decomposition(20 + 16, 2, 4), vec![0, 1, 0, 0]);
    }

    #[test]
    #[should_panic]
    fn test_base_decomposition_invalid_base() {
        // Base cannot be 0 or 1 (should panic)
        let _ = base_decomposition(10, 0, 5);
    }

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
