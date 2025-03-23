use std::{
    any::{Any, TypeId},
    collections::HashMap,
    sync::{Arc, LazyLock, Mutex, RwLock, RwLockReadGuard},
};

use p3_field::{Field, TwoAdicField};
#[cfg(feature = "parallel")]
use {super::utils::workload_size, rayon::prelude::*, std::cmp::max};

use crate::ntt::{
    transpose::transpose,
    utils::{lcm, sqrt_factor},
};

/// Global cache for NTT engines, indexed by field.
static ENGINE_CACHE: LazyLock<Mutex<HashMap<TypeId, Arc<dyn Any + Send + Sync>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Compute the many NTTs of size `size` using a cached engine.
pub fn ntt_batch<F: Field + TwoAdicField>(values: &mut [F], size: usize) {
    NttEngine::<F>::new_from_cache().ntt_batch(values, size);
}

/// Compute the inverse NTT of multiple slice of field elements, each of size `size`, without the
/// 1/n scaling factor and using a cached engine.
pub fn intt_batch<F: Field + TwoAdicField>(values: &mut [F], size: usize) {
    NttEngine::<F>::new_from_cache().intt_batch(values, size);
}

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
        if order % 4 == 0 {
            res.omega_4_1 = res.root(4);
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

    pub fn ntt(&self, values: &mut [F]) {
        self.ntt_batch(values, values.len());
    }

    pub fn ntt_batch(&self, values: &mut [F], size: usize) {
        assert!(values.len() % size == 0);
        let roots = self.roots_table(size);
        self.ntt_dispatch(values, &roots, size);
    }

    /// Inverse NTT. Does not aply 1/n scaling factor.
    pub fn intt(&self, values: &mut [F]) {
        values[1..].reverse();
        self.ntt(values);
    }

    /// Inverse batch NTT. Does not apply 1/n scaling factor.
    pub fn intt_batch(&self, values: &mut [F], size: usize) {
        assert!(values.len() % size == 0);

        #[cfg(not(feature = "parallel"))]
        values.chunks_exact_mut(size).for_each(|values| {
            values[1..].reverse();
        });

        #[cfg(feature = "parallel")]
        values.par_chunks_exact_mut(size).for_each(|values| {
            values[1..].reverse();
        });

        self.ntt_batch(values, size);
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
                #[cfg(not(feature = "parallel"))]
                {
                    let mut root_i = F::ONE;
                    for _ in 0..size {
                        roots.push(root_i);
                        root_i *= root;
                    }
                }
                #[cfg(feature = "parallel")]
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

    /// Compute NTTs in place by splititng into two factors.
    /// Recurses using the sqrt(N) Cooley-Tukey Six step NTT algorithm.
    fn ntt_recurse(&self, values: &mut [F], roots: &[F], size: usize) {
        debug_assert_eq!(values.len() % size, 0);
        let n1 = sqrt_factor(size);
        let n2 = size / n1;

        transpose(values, n1, n2);
        self.ntt_dispatch(values, roots, n1);
        transpose(values, n2, n1);
        // TODO: When (n1, n2) are coprime we can use the
        // Good-Thomas NTT algorithm and avoid the twiddle loop.
        apply_twiddles(values, roots, n1, n2);
        self.ntt_dispatch(values, roots, n2);
        transpose(values, n1, n2);
    }

    fn ntt_dispatch(&self, values: &mut [F], roots: &[F], size: usize) {
        debug_assert_eq!(values.len() % size, 0);
        debug_assert_eq!(roots.len() % size, 0);
        #[cfg(feature = "parallel")]
        if values.len() > workload_size::<F>() && values.len() != size {
            // Multiple NTTs, compute in parallel.
            // Work size is largest multiple of `size` smaller than `WORKLOAD_SIZE`.
            let workload_size = size * max(1, workload_size::<F>() / size);
            return values.par_chunks_mut(workload_size).for_each(|values| {
                self.ntt_dispatch(values, roots, size);
            });
        }
        match size {
            0 | 1 => {}
            2 => {
                for v in values.chunks_exact_mut(2) {
                    (v[0], v[1]) = (v[0] + v[1], v[0] - v[1]);
                }
            }
            3 => {
                for v in values.chunks_exact_mut(3) {
                    // Rader NTT to reduce 3 to 2.
                    let v0 = v[0];
                    (v[1], v[2]) = (v[1] + v[2], v[1] - v[2]);
                    v[0] += v[1];
                    v[1] *= self.half_omega_3_1_plus_2; // ½(ω₃ + ω₃²)
                    v[2] *= self.half_omega_3_1_min_2; // ½(ω₃ - ω₃²)
                    v[1] += v0;
                    (v[1], v[2]) = (v[1] + v[2], v[1] - v[2]);
                }
            }
            4 => {
                for v in values.chunks_exact_mut(4) {
                    (v[0], v[2]) = (v[0] + v[2], v[0] - v[2]);
                    (v[1], v[3]) = (v[1] + v[3], v[1] - v[3]);
                    v[3] *= self.omega_4_1;
                    (v[0], v[1]) = (v[0] + v[1], v[0] - v[1]);
                    (v[2], v[3]) = (v[2] + v[3], v[2] - v[3]);
                    (v[1], v[2]) = (v[2], v[1]);
                }
            }
            8 => {
                for v in values.chunks_exact_mut(8) {
                    // Cooley-Tukey with v as 2x4 matrix.
                    (v[0], v[4]) = (v[0] + v[4], v[0] - v[4]);
                    (v[1], v[5]) = (v[1] + v[5], v[1] - v[5]);
                    (v[2], v[6]) = (v[2] + v[6], v[2] - v[6]);
                    (v[3], v[7]) = (v[3] + v[7], v[3] - v[7]);
                    v[5] *= self.omega_8_1;
                    v[6] *= self.omega_4_1; // == omega_8_2
                    v[7] *= self.omega_8_3;
                    (v[0], v[2]) = (v[0] + v[2], v[0] - v[2]);
                    (v[1], v[3]) = (v[1] + v[3], v[1] - v[3]);
                    v[3] *= self.omega_4_1;
                    (v[0], v[1]) = (v[0] + v[1], v[0] - v[1]);
                    (v[2], v[3]) = (v[2] + v[3], v[2] - v[3]);
                    (v[4], v[6]) = (v[4] + v[6], v[4] - v[6]);
                    (v[5], v[7]) = (v[5] + v[7], v[5] - v[7]);
                    v[7] *= self.omega_4_1;
                    (v[4], v[5]) = (v[4] + v[5], v[4] - v[5]);
                    (v[6], v[7]) = (v[6] + v[7], v[6] - v[7]);
                    (v[1], v[4]) = (v[4], v[1]);
                    (v[3], v[6]) = (v[6], v[3]);
                }
            }
            16 => {
                for v in values.chunks_exact_mut(16) {
                    // Cooley-Tukey with v as 4x4 matrix.
                    for i in 0..4 {
                        let v = &mut v[i..];
                        (v[0], v[8]) = (v[0] + v[8], v[0] - v[8]);
                        (v[4], v[12]) = (v[4] + v[12], v[4] - v[12]);
                        v[12] *= self.omega_4_1;
                        (v[0], v[4]) = (v[0] + v[4], v[0] - v[4]);
                        (v[8], v[12]) = (v[8] + v[12], v[8] - v[12]);
                        (v[4], v[8]) = (v[8], v[4]);
                    }
                    v[5] *= self.omega_16_1;
                    v[6] *= self.omega_8_1;
                    v[7] *= self.omega_16_3;
                    v[9] *= self.omega_8_1;
                    v[10] *= self.omega_4_1;
                    v[11] *= self.omega_8_3;
                    v[13] *= self.omega_16_3;
                    v[14] *= self.omega_8_3;
                    v[15] *= self.omega_16_9;
                    for i in 0..4 {
                        let v = &mut v[i * 4..];
                        (v[0], v[2]) = (v[0] + v[2], v[0] - v[2]);
                        (v[1], v[3]) = (v[1] + v[3], v[1] - v[3]);
                        v[3] *= self.omega_4_1;
                        (v[0], v[1]) = (v[0] + v[1], v[0] - v[1]);
                        (v[2], v[3]) = (v[2] + v[3], v[2] - v[3]);
                        (v[1], v[2]) = (v[2], v[1]);
                    }
                    (v[1], v[4]) = (v[4], v[1]);
                    (v[2], v[8]) = (v[8], v[2]);
                    (v[3], v[12]) = (v[12], v[3]);
                    (v[6], v[9]) = (v[9], v[6]);
                    (v[7], v[13]) = (v[13], v[7]);
                    (v[11], v[14]) = (v[14], v[11]);
                }
            }
            size => self.ntt_recurse(values, roots, size),
        }
    }
}

impl<F: Field + TwoAdicField> NttEngine<F> {
    /// Get or create a cached engine for the field `F`.
    pub fn new_from_cache() -> Arc<Self> {
        let mut cache = ENGINE_CACHE.lock().unwrap();
        let type_id = TypeId::of::<F>();
        #[allow(clippy::option_if_let_else)]
        if let Some(engine) = cache.get(&type_id) {
            engine.clone().downcast::<Self>().unwrap()
        } else {
            let engine = Arc::new(Self::new_from_fftfield());
            cache.insert(type_id, engine.clone());
            engine
        }
    }

    /// Construct a new engine from the field's `FftField` trait.
    fn new_from_fftfield() -> Self {
        // TODO: Support SMALL_SUBGROUP
        if F::TWO_ADICITY <= 63 {
            Self::new(1 << F::TWO_ADICITY, F::two_adic_generator(F::TWO_ADICITY))
        } else {
            let mut generator = F::two_adic_generator(F::TWO_ADICITY);
            for _ in 0..(F::TWO_ADICITY - 63) {
                generator = generator.square();
            }
            Self::new(1 << 63, generator)
        }
    }
}

#[cfg(not(feature = "parallel"))]
fn apply_twiddles<F: Field>(values: &mut [F], roots: &[F], rows: usize, cols: usize) {
    debug_assert_eq!(values.len() % (rows * cols), 0);
    let step = roots.len() / (rows * cols);
    for values in values.chunks_exact_mut(rows * cols) {
        for (i, row) in values.chunks_exact_mut(cols).enumerate().skip(1) {
            let step = (i * step) % roots.len();
            let mut index = step;
            for value in row.iter_mut().skip(1) {
                index %= roots.len();
                *value *= roots[index];
                index += step;
            }
        }
    }
}

/// Applies precomputed twiddle factors to the given values matrix.
///
/// This function modifies `values` in-place by multiplying specific elements with
/// corresponding twiddle factors from `roots`. The twiddle factors are applied to all
/// rows except the first one, and only to columns beyond the first column.
#[cfg(feature = "parallel")]
fn apply_twiddles<F: Field>(values: &mut [F], roots: &[F], rows: usize, cols: usize) {
    let size = rows * cols;
    debug_assert_eq!(values.len() % size, 0);
    let step = roots.len() / size;

    // Optimize for large workloads by processing in parallel
    if values.len() > workload_size::<F>() {
        if values.len() == size {
            values.par_chunks_exact_mut(cols).enumerate().skip(1).for_each(|(i, row)| {
                let step = (i * step) % roots.len();
                let mut index = step;
                for value in row.iter_mut().skip(1) {
                    index %= roots.len();
                    unsafe {
                        *value *= *roots.get_unchecked(index);
                    }
                    index += step;
                }
            });
        } else {
            let workload_size = size * max(1, workload_size::<F>() / size);
            values.par_chunks_mut(workload_size).for_each(|values| {
                apply_twiddles(values, roots, rows, cols);
            });
        }

        return;
    }

    // Sequential processing for smaller workloads
    for values in values.chunks_exact_mut(size) {
        for (i, row) in values.chunks_exact_mut(cols).enumerate().skip(1) {
            let step = (i * step) % roots.len();
            let mut index = step;
            for value in row.iter_mut().skip(1) {
                index %= roots.len();
                unsafe {
                    *value *= *roots.get_unchecked(index);
                }
                index += step;
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::significant_drop_tightening)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::{PrimeCharacteristicRing, PrimeField64, TwoAdicField};

    use super::*;

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
    fn test_apply_twiddles_basic() {
        let omega = BabyBear::two_adic_generator(2);
        let engine = NttEngine::new(4, omega);

        let mut values = vec![
            BabyBear::from_u64(1),
            BabyBear::from_u64(2),
            BabyBear::from_u64(3),
            BabyBear::from_u64(4),
            BabyBear::from_u64(5),
            BabyBear::from_u64(6),
            BabyBear::from_u64(7),
            BabyBear::from_u64(8),
        ];

        // Mock roots
        let r1 = BabyBear::from_u64(33);
        let roots = vec![r1];

        // Ensure the root of unity is correct
        assert_eq!(engine.root(4).exp_u64(4), BabyBear::ONE);

        apply_twiddles(&mut values, &roots, 2, 4);

        // The first row should remain unchanged
        assert_eq!(values[0], BabyBear::from_u64(1));
        assert_eq!(values[1], BabyBear::from_u64(2));
        assert_eq!(values[2], BabyBear::from_u64(3));
        assert_eq!(values[3], BabyBear::from_u64(4));

        // The second row should be multiplied by the correct twiddle factors
        assert_eq!(values[4], BabyBear::from_u64(5)); // No change for first column
        assert_eq!(values[5], BabyBear::from_u64(6) * r1);
        assert_eq!(values[6], BabyBear::from_u64(7) * r1);
        assert_eq!(values[7], BabyBear::from_u64(8) * r1);
    }

    #[test]
    fn test_apply_twiddles_single_row() {
        let mut values = vec![BabyBear::from_u64(1), BabyBear::from_u64(2)];

        // Mock roots
        let r1 = BabyBear::from_u64(12);
        let roots = vec![r1];

        apply_twiddles(&mut values, &roots, 1, 2);

        // Everything should remain unchanged
        assert_eq!(values[0], BabyBear::from_u64(1));
        assert_eq!(values[1], BabyBear::from_u64(2));
    }

    #[test]
    fn test_apply_twiddles_varying_rows() {
        let mut values = vec![
            BabyBear::from_u64(1),
            BabyBear::from_u64(2),
            BabyBear::from_u64(3),
            BabyBear::from_u64(4),
            BabyBear::from_u64(5),
            BabyBear::from_u64(6),
            BabyBear::from_u64(7),
            BabyBear::from_u64(8),
            BabyBear::from_u64(9),
        ];

        // Mock roots
        let roots = (2..100).map(BabyBear::from_u64).collect::<Vec<_>>();

        apply_twiddles(&mut values, &roots, 3, 3);

        // First row remains unchanged
        assert_eq!(values[0], BabyBear::from_u64(1));
        assert_eq!(values[1], BabyBear::from_u64(2));
        assert_eq!(values[2], BabyBear::from_u64(3));

        // Second row multiplied by twiddle factors
        assert_eq!(values[3], BabyBear::from_u64(4));
        assert_eq!(values[4], BabyBear::from_u64(5) * roots[10]);
        assert_eq!(values[5], BabyBear::from_u64(6) * roots[20]);

        // Third row multiplied by twiddle factors
        assert_eq!(values[6], BabyBear::from_u64(7));
        assert_eq!(values[7], BabyBear::from_u64(8) * roots[20]);
        assert_eq!(values[8], BabyBear::from_u64(9) * roots[40]);
    }

    #[test]
    fn test_apply_twiddles_large_table() {
        let rows = 320;
        let cols = 320;
        let size = rows * cols;

        let mut values: Vec<BabyBear> = (0..size as u64).map(BabyBear::from_u64).collect();

        // Generate a large set of twiddle factors
        let roots: Vec<BabyBear> = (0..(size * 2) as u64).map(BabyBear::from_u64).collect();

        apply_twiddles(&mut values, &roots, rows, cols);

        // Verify the first row remains unchanged
        for (i, &col) in values.iter().enumerate().take(cols) {
            assert_eq!(col, BabyBear::from_u64(i as u64));
        }

        // Verify the first column remains unchanged
        for row in 1..rows {
            let index = row * cols;
            assert_eq!(
                values[index],
                BabyBear::from_u64(index as u64),
                "Mismatch in first column at row={row}"
            );
        }

        // Verify that other rows have been modified using the twiddle factors
        for row in 1..rows {
            let mut idx = row * 2;
            for col in 1..cols {
                let index = row * cols + col;
                let expected = BabyBear::from_u64(index as u64) * roots[idx];
                assert_eq!(values[index], expected, "Mismatch at row={row}, col={col}");
                idx += 2 * row;
            }
        }
    }

    #[test]
    fn test_new_from_fftfield_basic() {
        // Ensure that an engine is created correctly from FFT field properties
        let engine = NttEngine::<BabyBear>::new_from_fftfield();

        // Verify that the order of the engine is correctly set
        assert!(engine.order.is_power_of_two());

        // Verify that the root of unity is correctly initialized
        let expected_root = BabyBear::two_adic_generator(BabyBear::TWO_ADICITY);
        let computed_root = engine.root(engine.order);
        assert_eq!(computed_root.exp_u64(engine.order as u64), BabyBear::ONE);
        assert_eq!(computed_root, expected_root);
    }

    #[test]
    fn test_new_from_cache_singleton() {
        // Retrieve two instances of the engine
        let engine1 = NttEngine::<BabyBear>::new_from_cache();
        let engine2 = NttEngine::<BabyBear>::new_from_cache();

        // Both instances should point to the same object in memory
        assert!(Arc::ptr_eq(&engine1, &engine2));

        // Verify that the cached instance has the expected properties
        assert!(engine1.order.is_power_of_two());

        let expected_root = BabyBear::two_adic_generator(BabyBear::TWO_ADICITY);
        assert_eq!(engine1.root(engine1.order), expected_root);
    }

    #[test]
    fn test_ntt_batch_size_2() {
        let omega = BabyBear::two_adic_generator(1); // 2nd root of unity
        let engine = NttEngine::new(2, omega);

        // Input values: f(x) = [1, 2]
        let f0 = BabyBear::from_u64(1);
        let f1 = BabyBear::from_u64(2);
        let mut values = vec![f0, f1];

        // Compute the expected NTT manually:
        //
        //   F(0)  =  f0 + f1
        //   F(1)  =  f0 - f1
        //
        // ω is the 2nd root of unity: ω² = 1.

        let expected_f0 = f0 + f1;
        let expected_f1 = f0 - f1;

        let expected_values = vec![expected_f0, expected_f1];

        engine.ntt_batch(&mut values, 2);

        assert_eq!(values, expected_values);
    }

    #[test]
    fn test_ntt_batch_size_4() {
        let omega = BabyBear::two_adic_generator(2); // 4th root of unity
        let engine = NttEngine::new(4, omega);

        // Input values: f(x) = [1, 2, 3, 4]
        let f0 = BabyBear::from_u64(1);
        let f1 = BabyBear::from_u64(2);
        let f2 = BabyBear::from_u64(3);
        let f3 = BabyBear::from_u64(4);
        let mut values = vec![f0, f1, f2, f3];

        // Compute the expected NTT manually:
        //
        //   F(0)  =  f0 + f1 + f2 + f3
        //   F(1)  =  f0 + f1 * ω + f2 * ω² + f3 * ω³
        //   F(2)  =  f0 + f1 * ω² + f2 * ω⁴ + f3 * ω⁶
        //   F(3)  =  f0 + f1 * ω³ + f2 * ω⁶ + f3 * ω⁹
        //
        // ω is the 4th root of unity: ω⁴ = 1, ω² = -1.

        let omega1 = omega; // ω
        let omega2 = omega * omega; // ω² = -1
        let omega3 = omega * omega2; // ω³ = -ω
        let omega4 = omega * omega3; // ω⁴ = 1

        let expected_f0 = f0 + f1 + f2 + f3;
        let expected_f1 = f0 + f1 * omega1 + f2 * omega2 + f3 * omega3;
        let expected_f2 = f0 + f1 * omega2 + f2 * omega4 + f3 * omega2;
        let expected_f3 = f0 + f1 * omega3 + f2 * omega2 + f3 * omega1;

        let expected_values = vec![expected_f0, expected_f1, expected_f2, expected_f3];

        engine.ntt_batch(&mut values, 4);

        assert_eq!(values, expected_values);
    }

    #[test]
    fn test_ntt_batch_size_8() {
        let omega = BabyBear::two_adic_generator(3); // 8th root of unity
        let engine = NttEngine::new(8, omega);

        // Input values: f(x) = [1, 2, 3, 4, 5, 6, 7, 8]
        let f0 = BabyBear::from_u64(1);
        let f1 = BabyBear::from_u64(2);
        let f2 = BabyBear::from_u64(3);
        let f3 = BabyBear::from_u64(4);
        let f4 = BabyBear::from_u64(5);
        let f5 = BabyBear::from_u64(6);
        let f6 = BabyBear::from_u64(7);
        let f7 = BabyBear::from_u64(8);
        let mut values = vec![f0, f1, f2, f3, f4, f5, f6, f7];

        // Compute the expected NTT manually:
        //
        //   F(k) = ∑ f_j * ω^(j*k)  for k ∈ {0, ..., 7}
        //
        // ω is the 8th root of unity: ω⁸ = 1.

        let omega1 = omega; // ω
        let omega2 = omega * omega; // ω²
        let omega3 = omega * omega2; // ω³
        let omega4 = omega * omega3; // ω⁴
        let omega5 = omega * omega4; // ω⁵
        let omega6 = omega * omega5; // ω⁶
        let omega7 = omega * omega6; // ω⁷

        let expected_f0 = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7;
        let expected_f1 = f0 +
            f1 * omega1 +
            f2 * omega2 +
            f3 * omega3 +
            f4 * omega4 +
            f5 * omega5 +
            f6 * omega6 +
            f7 * omega7;
        let expected_f2 = f0 +
            f1 * omega2 +
            f2 * omega4 +
            f3 * omega6 +
            f4 * BabyBear::ONE +
            f5 * omega2 +
            f6 * omega4 +
            f7 * omega6;
        let expected_f3 = f0 +
            f1 * omega3 +
            f2 * omega6 +
            f3 * omega1 +
            f4 * omega4 +
            f5 * omega7 +
            f6 * omega2 +
            f7 * omega5;
        let expected_f4 = f0 +
            f1 * omega4 +
            f2 * BabyBear::ONE +
            f3 * omega4 +
            f4 * BabyBear::ONE +
            f5 * omega4 +
            f6 * BabyBear::ONE +
            f7 * omega4;
        let expected_f5 = f0 +
            f1 * omega5 +
            f2 * omega2 +
            f3 * omega7 +
            f4 * omega4 +
            f5 * omega1 +
            f6 * omega6 +
            f7 * omega3;
        let expected_f6 = f0 +
            f1 * omega6 +
            f2 * omega4 +
            f3 * omega2 +
            f4 * BabyBear::ONE +
            f5 * omega6 +
            f6 * omega4 +
            f7 * omega2;
        let expected_f7 = f0 +
            f1 * omega7 +
            f2 * omega6 +
            f3 * omega5 +
            f4 * omega4 +
            f5 * omega3 +
            f6 * omega2 +
            f7 * omega1;

        let expected_values = vec![
            expected_f0,
            expected_f1,
            expected_f2,
            expected_f3,
            expected_f4,
            expected_f5,
            expected_f6,
            expected_f7,
        ];

        engine.ntt_batch(&mut values, 8);

        assert_eq!(values, expected_values);
    }

    #[test]
    fn test_ntt_batch_size_16() {
        let omega = BabyBear::two_adic_generator(4); // 16th root of unity
        let engine = NttEngine::new(16, omega);

        // Input values: f(x) = [1, 2, ..., 16]
        let values: Vec<BabyBear> = (1..=16).map(BabyBear::from_u64).collect();
        let mut values_ntt = values.clone();

        // Compute the expected NTT manually:
        //
        //   F(k) = ∑ f_j * ω^(j*k)  for k ∈ {0, ..., 15}
        //
        // ω is the 16th root of unity: ω¹⁶ = 1.

        let mut expected_values = vec![BabyBear::ZERO; 16];
        for (k, expected_value) in expected_values.iter_mut().enumerate().take(16) {
            let omega_k = omega.exp_u64(k as u64);
            *expected_value =
                values.iter().enumerate().map(|(j, &f_j)| f_j * omega_k.exp_u64(j as u64)).sum();
        }

        engine.ntt_batch(&mut values_ntt, 16);

        assert_eq!(values_ntt, expected_values);
    }

    #[test]
    fn test_ntt_batch_size_32() {
        let omega = BabyBear::two_adic_generator(5); // 32nd root of unity
        let engine = NttEngine::new(32, omega);

        // Input values: f(x) = [1, 2, ..., 32]
        let values: Vec<BabyBear> = (1..=32).map(BabyBear::from_u64).collect();
        let mut values_ntt = values.clone();

        // Compute the expected NTT manually:
        //
        //   F(k) = ∑ f_j * ω^(j*k)  for k ∈ {0, ..., 31}
        //
        // ω is the 32nd root of unity: ω³² = 1.

        let mut expected_values = vec![BabyBear::ZERO; 32];
        for (k, expected_value) in expected_values.iter_mut().enumerate().take(32) {
            let omega_k = omega.exp_u64(k as u64);
            *expected_value =
                values.iter().enumerate().map(|(j, &f_j)| f_j * omega_k.exp_u64(j as u64)).sum();
        }

        engine.ntt_batch(&mut values_ntt, 32);

        assert_eq!(values_ntt, expected_values);
    }
}
