/// Returns true if `n`:
/// - is a power of two
/// - is greater than zero
pub const fn is_power_of_two(n: usize) -> bool {
    n != 0 && n.is_power_of_two()
}

/// performs big-endian binary decomposition of `value` and returns the result.
///
/// `n_bits` must be at must usize::BITS. If it is strictly smaller, the most significant bits of
/// `value` are ignored. The returned vector v ends with the least significant bit of `value` and
/// always has exactly `n_bits` many elements.
pub fn to_binary(value: usize, n_bits: usize) -> Vec<bool> {
    // Ensure that n is within the bounds of the input integer type
    assert!(n_bits <= usize::BITS as usize);
    let mut result = vec![false; n_bits];
    for i in 0..n_bits {
        result[n_bits - 1 - i] = (value & (1 << i)) != 0;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_power_of_two() {
        assert!(!is_power_of_two(0));
        assert!(is_power_of_two(1));
        assert!(is_power_of_two(2));
        assert!(!is_power_of_two(3));
        assert!(!is_power_of_two(usize::MAX));
    }

    #[test]
    fn test_to_binary() {
        assert_eq!(to_binary(0b10111, 5), vec![true, false, true, true, true]);
        assert_eq!(to_binary(0b11001, 2), vec![false, true]); // truncate
        let empty_vec: Vec<bool> = vec![]; // just for the explicit bool type.
        assert_eq!(to_binary(1, 0), empty_vec);
        assert_eq!(to_binary(0, 0), empty_vec);
    }
}
