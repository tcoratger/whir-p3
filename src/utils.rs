//! Utility functions for the WHIR library.
//!
//! This module provides general-purpose utility functions used throughout the codebase,
//! including memory allocation helpers and mathematical utilities.

use alloc::vec::Vec;

/// Returns a vector of uninitialized elements of type `A` with the specified length.
///
/// # Safety
///
/// Entries should be overwritten before use.
#[must_use]
pub unsafe fn uninitialized_vec<A>(len: usize) -> Vec<A> {
    #[allow(clippy::uninit_vec)]
    unsafe {
        let mut vec = Vec::with_capacity(len);
        vec.set_len(len);
        vec
    }
}

/// Computes the strict base-3 logarithm of `n`.
///
/// Returns `k` such that `3^k == n`. Panics if `n` is not a power of 3.
///
/// This is the base-3 analogue of plonky3's `log2_strict_usize`.
///
/// # Arguments
///
/// * `n` - A positive integer that must be a power of 3 (i.e., 1, 3, 9, 27, 81, ...).
///
/// # Returns
///
/// The exponent `k` where `3^k == n`.
///
/// # Panics
///
/// Panics if:
/// - `n` is zero
/// - `n` is not a power of 3
#[must_use]
pub fn log3_strict_usize(n: usize) -> usize {
    // Zero has no logarithm.
    debug_assert_ne!(n, 0, "log3_strict_usize: input must be non-zero");

    // Verify that n is a power of 3 before computing the logarithm.
    // This check is done upfront to provide a clear error message.
    debug_assert!(
        is_power_of_3(n),
        "log3_strict_usize: {n} is not a power of 3"
    );

    // Exponent counter: tracks how many times we've divided by 3.
    let mut res = 0usize;

    // Working value: we repeatedly divide by 3 until we reach 0.
    let mut t = n;

    // Main loop: divide by 3 and count iterations.
    loop {
        // Integer division by 3. When t < 3, this yields 0.
        t /= 3;

        // Exit when we've exhausted all factors of 3.
        if t == 0 {
            break;
        }

        // Increment the exponent for each successful division.
        res += 1;
    }

    res
}

/// Checks if `n` is a power of 3.
///
/// Returns `true` if `n == 3^k` for some non-negative integer `k`.
#[inline]
const fn is_power_of_3(n: usize) -> bool {
    // Zero is not a power of 3.
    if n == 0 {
        return false;
    }

    // Repeatedly divide by 3 until we can't anymore.
    let mut value = n;
    while value.is_multiple_of(3) {
        value /= 3;
    }

    // If we end up at 1, n was a power of 3.
    value == 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log3_strict_powers_of_3() {
        // Test all powers of 3 up to 3^12 = 531441.
        assert_eq!(log3_strict_usize(1), 0);
        assert_eq!(log3_strict_usize(3), 1);
        assert_eq!(log3_strict_usize(9), 2);
        assert_eq!(log3_strict_usize(27), 3);
        assert_eq!(log3_strict_usize(81), 4);
        assert_eq!(log3_strict_usize(243), 5);
        assert_eq!(log3_strict_usize(729), 6);
        assert_eq!(log3_strict_usize(2187), 7);
        assert_eq!(log3_strict_usize(6561), 8);
        assert_eq!(log3_strict_usize(19683), 9);
        assert_eq!(log3_strict_usize(59049), 10);
        assert_eq!(log3_strict_usize(177_147), 11);
        assert_eq!(log3_strict_usize(531_441), 12);
    }

    #[test]
    #[should_panic(expected = "input must be non-zero")]
    fn test_log3_strict_panics_on_zero() {
        let _ = log3_strict_usize(0);
    }

    #[test]
    #[should_panic(expected = "is not a power of 3")]
    fn test_log3_strict_panics_on_non_power_of_3() {
        // 2 is not a power of 3.
        let _ = log3_strict_usize(2);
    }

    #[test]
    #[should_panic(expected = "is not a power of 3")]
    fn test_log3_strict_panics_on_power_of_2() {
        // 8 = 2^3 is not a power of 3.
        let _ = log3_strict_usize(8);
    }

    #[test]
    #[should_panic(expected = "is not a power of 3")]
    fn test_log3_strict_panics_on_product_with_other_primes() {
        // 6 = 2 * 3 is not a power of 3.
        let _ = log3_strict_usize(6);
    }
}
