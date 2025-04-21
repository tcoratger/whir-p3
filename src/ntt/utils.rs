/// Computes the optimal workload size for `T` to fit in L1 cache (32 KB).
///
/// Ensures efficient memory access by dividing the cache size by `T`'s size.
/// The result represents how many elements of `T` can be processed per thread.
///
/// Helps minimize cache misses and improve performance in parallel workloads.
pub const fn workload_size<T: Sized>() -> usize {
    const L1_CACHE_SIZE: usize = 1 << 15; // 32 KB
    L1_CACHE_SIZE / size_of::<T>()
}

/// Least common multiple.
///
/// Note that lcm(0,0) will panic (rather than give the correct answer 0).
pub const fn lcm(a: usize, b: usize) -> usize {
    a * (b / gcd(a, b))
}

/// Greatest common divisor.
pub const fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Computes the largest factor of `x` that is ≤ sqrt(x).
    /// If `x` is 0, returns 0.
    fn get_largest_divisor_up_to_sqrt(x: usize) -> usize {
        if x == 0 {
            return 0;
        }

        let mut result = 1;

        // Compute integer square root of `x` using floating point arithmetic.
        #[allow(clippy::cast_sign_loss)]
        let isqrt_x = (x as f64).sqrt() as usize;

        // Iterate from 1 to `isqrt_x` to find the largest factor of `x`.
        for i in 1..=isqrt_x {
            if x % i == 0 {
                // Update `result` with the largest divisor found.
                result = i;
            }
        }

        result
    }

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(4, 6), 2);
        assert_eq!(gcd(0, 4), 4);
        assert_eq!(gcd(4, 0), 4);
        assert_eq!(gcd(1, 1), 1);
        assert_eq!(gcd(64, 16), 16);
        assert_eq!(gcd(81, 9), 9);
        assert_eq!(gcd(0, 0), 0);
    }

    #[test]
    fn test_lcm() {
        assert_eq!(lcm(5, 6), 30);
        assert_eq!(lcm(3, 7), 21);
        assert_eq!(lcm(0, 10), 0);
    }
}
