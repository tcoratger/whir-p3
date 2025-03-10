use crate::ntt::utils::lcm;
use p3_field::Field;
use rayon::prelude::*;
use std::sync::{RwLock, RwLockReadGuard};

/// Number-Theoretic Transform (NTT) engine for computing forward and inverse transforms.
///
/// This struct precomputes and stores twiddle factors (roots of unity) for efficient NTT
/// computations. It assumes that the underlying field has high two-adicity.
#[derive(Debug, Default)]
pub struct NttEngine<F: Field> {
    /// Order of the root of unity used for NTT computation.
    order: usize,
    /// Primitive root of unity of order `order`.
    omega_order: F,
    /// Precomputed value ½(ω₃ + ω₃²), used for 3-point NTT optimizations.
    half_omega_3_1_plus_2: F,
    /// Precomputed value ½(ω₃ - ω₃²), used for 3-point NTT optimizations.
    half_omega_3_1_min_2: F,
    /// A precomputed 4th root of unity, used for radix-4 NTT decompositions.
    omega_4_1: F,
    /// A precomputed 8th root of unity, used for radix-8 NTT optimizations.
    omega_8_1: F,
    /// The cube of the 8th root of unity (ω₈³).
    omega_8_3: F,
    /// A precomputed 16th root of unity, used in higher-radix transforms.
    omega_16_1: F,
    /// The cube of the 16th root of unity (ω₁₆³).
    omega_16_3: F,
    /// The 9th power of the 16th root of unity (ω₁₆⁹).
    omega_16_9: F,
    /// Lookup table for storing precomputed powers of the root of unity, extended on demand.
    roots: RwLock<Vec<F>>,
}

impl<F: Field> NttEngine<F> {
    /// Creates a new NTT engine with a given order and corresponding primitive root of unity.
    ///
    /// The engine precomputes small-order roots of unity for efficient NTT execution.
    pub fn new(order: usize, omega_order: F) -> Self {
        assert!(order.trailing_zeros() > 0, "Order must be a multiple of 2.");

        // Ensure `omega_order` is the correct root of unity
        let computed_root = omega_order.exp_u64(order as u64);
        assert_eq!(computed_root, F::ONE, "ω_order is not a valid root of unity.");

        assert_ne!(omega_order.exp_u64(order as u64 / 2), F::ONE);

        let mut res = Self {
            order,
            omega_order,
            half_omega_3_1_plus_2: F::ZERO,
            half_omega_3_1_min_2: F::ZERO,
            omega_4_1: F::ZERO,
            omega_8_1: F::ZERO,
            omega_8_3: F::ZERO,
            omega_16_1: F::ZERO,
            omega_16_3: F::ZERO,
            omega_16_9: F::ZERO,
            roots: RwLock::new(Vec::new()),
        };

        // Precompute constants for small-radix optimizations
        if order % 3 == 0 {
            let omega_3_1 = res.root(3);
            let omega_3_2 = omega_3_1 * omega_3_1;
            // Precompute helper values for 3-point NTT optimizations
            // Note: char F cannot be 2 and so division by 2 works, because primitive roots of unity
            // with even order exist.
            res.half_omega_3_1_min_2 = (omega_3_1 - omega_3_2) / F::from_u64(2u64);
            res.half_omega_3_1_plus_2 = (omega_3_1 + omega_3_2) / F::from_u64(2u64);
        }
        if order % 8 == 0 {
            res.omega_8_1 = res.root(8);
            res.omega_8_3 = res.omega_8_1.exp_const_u64::<3>();
        }
        if order % 16 == 0 {
            res.omega_16_1 = res.root(16);
            res.omega_16_3 = res.omega_16_1.exp_const_u64::<3>();
            res.omega_16_9 = res.omega_16_1.exp_const_u64::<9>();
        }
        res
    }

    /// Computes the `order`-th root of unity by exponentiating `omega_order`.
    pub fn root(&self, order: usize) -> F {
        assert!(self.order % order == 0, "Subgroup of requested order does not exist.");
        self.omega_order.exp_u64((self.order / order) as u64)
    }

    /// Returns a cached table of roots of unity of the given order.
    pub fn roots_table(&self, order: usize) -> RwLockReadGuard<'_, Vec<F>> {
        // Precompute more roots of unity if requested.
        let roots = self.roots.read().unwrap();
        if roots.is_empty() || roots.len() % order != 0 {
            // Obtain write lock to update the cache.
            drop(roots);
            let mut roots = self.roots.write().unwrap();
            // Race condition: check if another thread updated the cache.
            if roots.is_empty() || roots.len() % order != 0 {
                // Compute minimal size to support all sizes seen so far.
                // TODO: Do we really need all of these? Can we leverage omega_2 = -1?
                let size = if roots.is_empty() { order } else { lcm(roots.len(), order) };
                roots.clear();
                roots.reserve_exact(size);

                // Compute powers of roots of unity.
                let root = self.root(size);
                roots.par_extend((0..size).into_par_iter().map_with(F::ZERO, |root_i, i| {
                    if root_i.is_zero() {
                        *root_i = root.exp_u64(i as u64);
                    } else {
                        *root_i *= root;
                    }
                    *root_i
                }));
            }
            // Back to read lock.
            drop(roots);
            self.roots.read().unwrap()
        } else {
            roots
        }
    }
}

#[cfg(test)]
#[allow(clippy::significant_drop_tightening)]
mod tests {
    use super::*;
    use p3_baby_bear::BabyBear;
    use p3_field::{PrimeCharacteristicRing, PrimeField64, TwoAdicField};

    #[test]
    fn test_root_computation() {
        // Use a valid 8th root of unity from BabyBearParameters
        let omega = BabyBear::two_adic_generator(3);
        let engine = NttEngine::new(8, omega);

        // Ensure the root exponentiates correctly
        assert_eq!(engine.root(8).exp_u64(8), BabyBear::ONE);
        assert_eq!(engine.root(4).exp_u64(4), BabyBear::ONE);
        assert_eq!(engine.root(2).exp_u64(2), BabyBear::ONE);

        // Ensure it's not a lower-order root
        assert_ne!(engine.root(8).exp_u64(4), BabyBear::ONE);
        assert_ne!(engine.root(4).exp_u64(2), BabyBear::ONE);
    }

    #[test]
    fn test_order_of_unity_validity() {
        for k in 1..=BabyBear::TWO_ADICITY {
            let omega = BabyBear::two_adic_generator(k);
            let engine = NttEngine::new(1 << k, omega);
            assert_eq!(engine.root(1 << k).exp_u64(1 << k), BabyBear::ONE);
        }
    }

    #[test]
    fn test_maximum_supported_two_adicity() {
        let omega = BabyBear::two_adic_generator(BabyBear::TWO_ADICITY);
        let engine = NttEngine::new(1 << BabyBear::TWO_ADICITY, omega);
        assert_eq!(
            engine.root(1 << BabyBear::TWO_ADICITY).exp_u64(1 << BabyBear::TWO_ADICITY),
            BabyBear::ONE
        );
    }

    #[test]
    fn test_root_of_unity_multiplication() {
        let omega = BabyBear::two_adic_generator(4);
        let engine = NttEngine::new(16, omega);
        let root = engine.root(16);

        // Multiply root by itself repeatedly and verify expected outcomes
        assert_eq!(root.exp_u64(2), engine.root(8));
        assert_eq!(root.exp_u64(4), engine.root(4));
        assert_eq!(root.exp_u64(8), engine.root(2));
    }

    #[test]
    fn test_root_of_unity_inversion() {
        let omega = BabyBear::two_adic_generator(4);
        let engine = NttEngine::new(16, omega);
        let root = engine.root(16);

        // The inverse of ω is ω^{-1}, computed as ω^(p-2) in BabyBear
        let inverse_root = root.exp_u64(BabyBear::ORDER_U64 - 2);
        assert_eq!(root * inverse_root, BabyBear::ONE);
    }

    #[test]
    fn test_precomputed_small_roots() {
        // Use a valid primitive root of unity from the predefined generator table
        let omega = BabyBear::two_adic_generator(4); // 2^4 = 16th root of unity
        let engine = NttEngine::new(16, omega);

        // Check that precomputed values are correctly initialized
        assert_eq!(engine.omega_8_1.exp_u64(8), BabyBear::ONE);
        assert_eq!(engine.omega_8_3.exp_u64(8), BabyBear::ONE);
        assert_eq!(engine.omega_16_1.exp_u64(16), BabyBear::ONE);
        assert_eq!(engine.omega_16_3.exp_u64(16), BabyBear::ONE);
        assert_eq!(engine.omega_16_9.exp_u64(16), BabyBear::ONE);
    }

    #[test]
    #[should_panic(expected = "Subgroup of requested order does not exist.")]
    fn test_invalid_root_order() {
        let omega = BabyBear::two_adic_generator(3);
        let engine = NttEngine::new(8, omega);

        // Requesting a root that does not divide 8 should panic
        engine.root(5);
    }

    #[test]
    #[should_panic(expected = "ω_order is not a valid root of unity.")]
    fn test_invalid_omega_order() {
        // Choose an arbitrary value that is not a root of unity
        let invalid_omega = BabyBear::from_u64(3);
        let _ = NttEngine::new(8, invalid_omega);
    }

    #[test]
    fn test_consistency_across_multiple_instances() {
        let omega = BabyBear::two_adic_generator(3);
        let engine1 = NttEngine::new(8, omega);
        let engine2 = NttEngine::new(8, omega);

        // Ensure that multiple instances yield the same results
        assert_eq!(engine1.root(8), engine2.root(8));
        assert_eq!(engine1.root(4), engine2.root(4));
        assert_eq!(engine1.root(2), engine2.root(2));
    }

    #[test]
    fn test_roots_table_basic() {
        let omega = BabyBear::two_adic_generator(3);
        let engine = NttEngine::new(8, omega);
        let roots_4 = engine.roots_table(4);

        // Check hardcoded expected values (ω^i)
        assert_eq!(roots_4[0], BabyBear::ONE);
        assert_eq!(roots_4[1], engine.root(4));
        assert_eq!(roots_4[2], engine.root(4).exp_u64(2));
        assert_eq!(roots_4[3], engine.root(4).exp_u64(3));
    }

    #[test]
    fn test_roots_table_minimal_order() {
        let omega = BabyBear::two_adic_generator(1);
        let engine = NttEngine::new(2, omega);

        let roots_2 = engine.roots_table(2);

        // Must contain only ω^0 and ω^1
        assert_eq!(roots_2.len(), 2);
        assert_eq!(roots_2[0], BabyBear::ONE);
        assert_eq!(roots_2[1], engine.root(2));
    }

    #[test]
    fn test_roots_table_progression() {
        let omega = BabyBear::two_adic_generator(3);
        let engine = NttEngine::new(8, omega);

        let roots_4 = engine.roots_table(4);

        // Ensure the sequence follows expected powers of the root of unity
        for i in 0..4 {
            assert_eq!(roots_4[i], engine.root(4).exp_u64(i as u64));
        }
    }

    #[test]
    fn test_roots_table_cached_results() {
        let omega = BabyBear::two_adic_generator(2);
        let engine = NttEngine::new(4, omega);

        let first_access = engine.roots_table(4);
        let second_access = engine.roots_table(4);

        // The memory location should be the same, meaning it's cached
        assert!(std::ptr::eq(first_access.as_ptr(), second_access.as_ptr()));
    }

    #[test]
    fn test_roots_table_recompute_factor_order() {
        let omega = BabyBear::two_adic_generator(3);
        let engine = NttEngine::new(8, omega);

        let roots_4 = engine.roots_table(4);
        let roots_2 = engine.roots_table(2);

        // Ensure first two elements of roots_4 match the first two elements of roots_2
        assert_eq!(&roots_4[..2], &roots_2[..2]);
    }

    #[test]
    fn test_roots_table_standard_sizes() {
        let omega = BabyBear::two_adic_generator(2); // Using order 2 for faster computation
        let engine = NttEngine::new(4, omega); // Smaller NTT size

        let roots_2 = engine.roots_table(2);
        let roots_4 = engine.roots_table(4);

        // Check that the first element is always 1 (ω^0 = 1)
        assert_eq!(roots_2[0], BabyBear::ONE);
        assert_eq!(roots_4[0], BabyBear::ONE);

        // Check that ω^1 is correct for different orders
        assert_eq!(roots_2[1], engine.root(2));
        assert_eq!(roots_4[1], engine.root(4));
    }
}
