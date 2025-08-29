use p3_field::{ExtensionField, Field};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;

pub fn parallel_clone<A>(src: &[A], dst: &mut [A])
where
    A: Clone + Send + Sync,
{
    assert_eq!(src.len(), dst.len());
    if src.len() < 1 << 15 {
        // sequential copy
        dst.clone_from_slice(src);
    } else {
        let chunk_size = src.len() / current_num_threads().max(1);
        dst.par_chunks_mut(chunk_size)
            .zip(src.par_chunks(chunk_size))
            .for_each(|(d, s)| {
                d.clone_from_slice(s);
            });
    }
}

pub fn parallel_repeat<A>(src: &[A], n: usize) -> Vec<A>
where
    A: Copy + Send + Sync,
{
    if src.len() * n < 1 << 15 {
        // sequential repeat
        src.repeat(n)
    } else {
        let res = unsafe { uninitialized_vec::<A>(src.len() * n) };
        src.par_iter().enumerate().for_each(|(i, &v)| {
            for j in 0..n {
                unsafe {
                    std::ptr::write(res.as_ptr().cast_mut().add(i + j * src.len()), v);
                }
            }
        });
        res
    }
}

/// Returns a vector of uninitialized elements of type `A` with the specified length.
/// # Safety
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

/// Computes the evaluations of the select(p, X) polynomial and adds them to the weights.
///
/// This is the replacement for `eval_eq` that enables the "select trick".
///
/// ```text
/// select(pow(z), b) = z^int(b)
/// ```
///
/// # Arguments
/// * `point`: The multilinear point `p` (which corresponds to `pow(z)`).
/// * `weights`: The slice of weights to be updated.
/// * `combination_randomness`: The random scalar `alpha` to multiply by.
pub fn eval_select<F, EF>(point: &[F], weights: &mut [EF], combination_randomness: EF)
where
    F: Field,
    EF: ExtensionField<F>,
{
    let n = log2_strict_usize(weights.len());
    assert_eq!(
        point.len(),
        n,
        "Point dimension must match number of variables"
    );

    // Get the powers of the point `z` in the boolean hypercube.
    let z_powers = &point;

    // Iterate through each point `b` on the boolean hypercube, represented by its integer value `i`.
    for (i, weight) in weights.iter_mut().enumerate() {
        let mut res = combination_randomness;
        for j in 0..n {
            if (i >> j) & 1 == 1 {
                res *= z_powers[j];
            }
        }
        // Add the computed value to the weights vector.
        *weight += res;
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, extension::BinomialExtensionField};

    use super::*;

    type F = BabyBear;
    type EF4 = BinomialExtensionField<F, 4>;

    #[test]
    fn test_parallel_clone() {
        let src = (0..(1 << 23) + 7).map(F::from_u64).collect::<Vec<_>>();
        let mut dst_seq = F::zero_vec(src.len());
        let time = std::time::Instant::now();
        dst_seq.copy_from_slice(&src);
        println!("Sequential clone took: {:?}", time.elapsed());
        let mut dst_parallel = F::zero_vec(src.len());
        let time = std::time::Instant::now();
        parallel_clone(&src, &mut dst_parallel);
        println!("Parallel clone took: {:?}", time.elapsed());
        assert_eq!(dst_seq, dst_parallel);
    }

    #[test]
    fn test_parallel_repeat() {
        let src = (0..(1 << 23) + 7).map(F::from_u64).collect::<Vec<_>>();
        let n = 3;
        let time = std::time::Instant::now();
        let dst_seq = src.repeat(n);
        println!("Sequential repeat took: {:?}", time.elapsed());
        let time = std::time::Instant::now();
        let dst_parallel = parallel_repeat(&src, n);
        println!("Parallel repeat took: {:?}", time.elapsed());
        assert_eq!(dst_seq, dst_parallel);
    }

    #[test]
    fn test_eval_select_basic_2_vars() {
        // ARRANGE
        // The number of variables is 2. The domain size is 2^2 = 4.
        let n_vars = 2;
        // Initialize a weights vector to all zeros. This is where results will be accumulated.
        let mut weights = vec![F::ZERO; 1 << n_vars];
        // Define the point 'z' for the pow(z) map. Let z = 3.
        let z = F::from_u64(3);
        // The point passed to `eval_select` is pow(z) = (z^(2^0), z^(2^1)) = (3, 9).
        let point = &[z, z.square()];
        // The combination randomness (alpha) is set to 1 for simplicity.
        let combination_randomness = F::ONE;

        // ACT
        // Call the function under test.
        eval_select(point, &mut weights, combination_randomness);

        // ASSERT
        // Manually calculate the expected result for each point `b` on the hypercube.
        // The formula is `alpha * z^int(b)`.
        // b = (0,0), int(b) = 0. Expected: 1 * 3^0 = 1.
        assert_eq!(weights[0b00], F::from_u64(1));
        // b = (0,1), int(b) = 1. Expected: 1 * 3^1 = 3.
        assert_eq!(weights[0b01], F::from_u64(3));
        // b = (1,0), int(b) = 2. Expected: 1 * 3^2 = 9.
        assert_eq!(weights[0b10], F::from_u64(9));
        // b = (1,1), int(b) = 3. Expected: 1 * 3^3 = 27.
        assert_eq!(weights[0b11], F::from_u64(27));
    }

    #[test]
    fn test_eval_select_with_nonzero_alpha() {
        // ARRANGE
        // Same setup as the basic 2-variable test.
        let n_vars = 2;
        let mut weights = vec![F::ZERO; 1 << n_vars];
        let z = F::from_u64(2);
        let point = &[z, z.square()]; // pow(z) = (2, 4)
        // Use a different randomness value, alpha = 5.
        let combination_randomness = F::from_u64(5);

        // ACT
        eval_select(point, &mut weights, combination_randomness);

        // ASSERT
        // The expected results are the same as the basic case, but scaled by alpha.
        // b = (0,0), int(b) = 0. Expected: 5 * 2^0 = 5.
        assert_eq!(weights[0b00], F::from_u64(5));
        // b = (0,1), int(b) = 1. Expected: 5 * 2^1 = 10.
        assert_eq!(weights[0b01], F::from_u64(10));
        // b = (1,0), int(b) = 2. Expected: 5 * 2^2 = 20.
        assert_eq!(weights[0b10], F::from_u64(20));
        // b = (1,1), int(b) = 3. Expected: 5 * 2^3 = 40.
        assert_eq!(weights[0b11], F::from_u64(40));
    }

    #[test]
    fn test_eval_select_accumulates_correctly() {
        // ARRANGE
        // Initialize weights with pre-existing values [10, 100].
        let mut weights = vec![F::from_u64(10), F::from_u64(100)];
        let z1 = F::from_u64(2);
        let point1 = &[z1]; // pow(z1) = (2)
        let alpha1 = F::from_u64(3);

        // ACT (First Call)
        // First accumulation: add 3*2^int(b) to the weights.
        eval_select(point1, &mut weights, alpha1);

        // ASSERT (First Call)
        // b=0, int(b)=0. Expected: 10 + 3*2^0 = 13.
        assert_eq!(weights[0b0], F::from_u64(13));
        // b=1, int(b)=1. Expected: 100 + 3*2^1 = 106.
        assert_eq!(weights[0b1], F::from_u64(106));

        // ARRANGE (Second Call)
        let z2 = F::from_u64(4);
        let point2 = &[z2]; // pow(z2) = (4)
        let alpha2 = F::from_u64(5);

        // ACT (Second Call)
        // Second accumulation: add 5*4^int(b) to the updated weights.
        eval_select(point2, &mut weights, alpha2);

        // ASSERT (Second Call)
        // b=0, int(b)=0. Expected: 13 + 5*4^0 = 18.
        assert_eq!(weights[0b0], F::from_u64(18));
        // b=1, int(b)=1. Expected: 106 + 5*4^1 = 126.
        assert_eq!(weights[0b1], F::from_u64(126));
    }

    #[test]
    fn test_eval_select_2_vars() {
        // ARRANGE
        // The setup is identical to the first test, but types are adjusted.
        let n_vars = 2;
        // The weights buffer is now in the extension field.
        let mut weights = vec![EF4::ZERO; 1 << n_vars];
        // The point 'z' is in the base field.
        let z = F::from_u64(3);
        // The pow(z) map is also in the base field.
        let point = &[z, z.square()];
        // The combination randomness is in the extension field.
        let combination_randomness = EF4::from_u64(1);

        // ACT
        eval_select(point, &mut weights, combination_randomness);

        // ASSERT
        // The expected results are the same values, but lifted into the extension field.
        assert_eq!(weights[0b00], EF4::from_u64(1));
        assert_eq!(weights[0b01], EF4::from_u64(3));
        assert_eq!(weights[0b10], EF4::from_u64(9));
        assert_eq!(weights[0b11], EF4::from_u64(27));
    }

    #[test]
    fn test_eval_select_with_extension_alpha() {
        // ARRANGE
        let n_vars = 1;
        let mut weights = vec![EF4::ZERO; 1 << n_vars];
        // The point is from the base field.
        let point = &[F::from_u64(10)];
        // Create a non-trivial extension field element for alpha.
        let combination_randomness =
            EF4::from_basis_coefficients_slice(&[F::from_u64(2), F::from_u64(3), F::ZERO, F::ZERO])
                .unwrap();

        // ACT
        eval_select(point, &mut weights, combination_randomness);

        // ASSERT
        // b=0, int(b)=0. Expected: alpha * 10^0 = alpha.
        assert_eq!(weights[0b0], combination_randomness);
        // b=1, int(b)=1. Expected: alpha * 10^1 = 10 * alpha.
        assert_eq!(weights[0b1], combination_randomness * EF4::from_u64(10));
    }

    #[test]
    fn test_eval_select_high_dimension() {
        // ARRANGE
        // Use a higher number of variables, e.g., 4. The domain size is 2^4 = 16.
        let n_vars = 4;
        // Initialize a weights vector to all zeros.
        let mut weights = vec![F::ZERO; 1 << n_vars];
        // Define the base point 'z'.
        let z = F::from_u64(3);
        // The point passed to `eval_select` is (z, z^2, z^4, z^8).
        let point: Vec<F> = (0..n_vars).map(|i| z.exp_u64(1 << i)).collect();
        // The combination randomness (alpha) is set to 1 for simplicity.
        let combination_randomness = F::ONE;

        // ACT
        // Call the function under test.
        eval_select(&point, &mut weights, combination_randomness);

        // ASSERT
        // Manually calculate the expected result for each point `b` on the hypercube.
        // The formula is `alpha * z^int(b)`. Since alpha is 1, this is just `z^i`.
        let mut expected_val = F::ONE;
        for i in 0..(1 << n_vars) {
            // The expected value at index `i` is `z^i`.
            assert_eq!(weights[i], expected_val, "Mismatch at index {}", i);
            // Update the expected value for the next iteration.
            expected_val *= z;
        }
    }
}
