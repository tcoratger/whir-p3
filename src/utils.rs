use p3_field::{
    Algebra, ExtensionField, Field, PackedFieldExtension, PackedValue, PrimeCharacteristicRing,
};
use p3_util::log2_strict_usize;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

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
#[inline]
pub(crate) fn eval_eq<F: Field, EF: ExtensionField<F>>(eval: &[EF], out: &mut [EF], scalar: EF) {
    // Number of threads to spawn.
    // Long term this should be a modifiable parameter.
    // I've chosen 32 here as my machine has 20 logical cores. Plausibly we might want to hard code this
    // to be a little larger than this.
    #[cfg(feature = "parallel")]
    const LOG_NUM_THREADS: usize = 5;
    #[cfg(not(feature = "parallel"))]
    const LOG_NUM_THREADS: usize = 0;

    const NUM_THREADS: usize = 1 << LOG_NUM_THREADS;

    // It's possible for this to be called with F = EF (Despite F actually being an extension field).
    // This check ensures this is not the case unless F is a prime field with non-trivial packing.
    let packing_width = F::Packing::WIDTH;
    debug_assert!(packing_width > 1);

    // Ensure that the output buffer size is correct:
    // It should be of size `2^n`, where `n` is the number of variables.
    debug_assert_eq!(out.len(), 1 << eval.len());

    // If the number of variables is small, there is no need to use
    // parallelization or packings.
    if eval.len() <= packing_width + 1 + LOG_NUM_THREADS {
        // A basic recursive approach.
        eval_eq_basic(eval, out, scalar);
    } else {
        let log_packing_width = log2_strict_usize(packing_width);
        let eval_len_min_packing = eval.len() - log_packing_width;

        // We split eval into three parts:
        // - eval[..LOG_NUM_THREADS] (the first LOG_NUM_THREADS elements)
        // - eval[LOG_NUM_THREADS..eval_len_min_packing] (the middle elements)
        // - eval[eval_len_min_packing..] (the last log_packing_width elements)

        // The middle elements are the ones which will be computed in parallel.
        // The last log_packing_width elements are the ones which will be packed.

        // We make a buffer of elements of size `NUM_THREADS`.
        let mut parallel_buffer = EF::ExtensionPacking::zero_vec(NUM_THREADS);
        let out_chunk_size = out.len() / NUM_THREADS;

        // Compute the equality polynomial corresponding to the last log_packing_width elements
        // and pack these.
        parallel_buffer[0] = packed_eq_poly(&eval[eval_len_min_packing..], scalar);

        // Update the buffer so it contains the evaluations of the equality polynomial
        // with respect to parts one and three.
        for (ind, entry) in eval[..LOG_NUM_THREADS].iter().rev().enumerate() {
            let stride = 1 << ind;

            for index in 0..stride {
                let val = parallel_buffer[index];
                let scaled_val = val * *entry;
                let new_val = val - scaled_val;

                parallel_buffer[index] = new_val;
                parallel_buffer[index + stride] = scaled_val;
            }
        }

        // Finally do all computations involving the middle elements in parallel.
        #[cfg(feature = "parallel")]
        out.par_chunks_exact_mut(out_chunk_size)
            .zip(parallel_buffer.par_iter())
            .for_each(|(out_chunk, buffer_val)| {
                eval_eq_packed(
                    &eval[LOG_NUM_THREADS..(eval.len() - log_packing_width)],
                    out_chunk,
                    *buffer_val,
                );
            });

        // Or not in parallel if the feature is unavailable.
        #[cfg(not(feature = "parallel"))]
        out.chunks_exact_mut(out_chunk_size)
            .zip(parallel_buffer.iter())
            .for_each(|(out_chunk, buffer_val)| {
                eval_eq_packed(
                    &eval[LOG_NUM_THREADS..(eval.len() - log_packing_width)],
                    out_chunk,
                    *buffer_val,
                );
            });
    }
}

/// Hard coded base case for the equality polynomial evaluation on 1 variable.
#[allow(clippy::inline_always)] // Adding inline(always) seems to give a small performance boost.
#[inline(always)]
fn eval_eq_1<F: Field, FP: Algebra<F> + Copy>(eval: &[F], scalar: FP) -> [FP; 2] {
    let x0 = eval[0];

    let s1 = scalar * x0;
    let s0 = scalar - s1;

    [s0, s1]
}

/// Hard coded base case for the equality polynomial evaluation on 2 variables.
#[allow(clippy::inline_always)] // Adding inline(always) seems to give a small performance boost.
#[inline(always)]
fn eval_eq_2<F: Field, FP: Algebra<F> + Copy>(eval: &[F], scalar: FP) -> [FP; 4] {
    let x0 = eval[0];
    let x1 = eval[1];

    let s1 = scalar * x0;
    let s0 = scalar - s1;

    let s01 = s0 * x1;
    let s00 = s0 - s01;
    let s11 = s1 * x1;
    let s10 = s1 - s11;

    [s00, s01, s10, s11]
}

/// Hard coded base case for the equality polynomial evaluation on 3 variables.
#[allow(clippy::inline_always)] // Adding inline(always) seems to give a small performance boost.
#[inline(always)]
fn eval_eq_3<F: Field, FP: Algebra<F> + Copy>(eval: &[F], scalar: FP) -> [FP; 8] {
    let x0 = eval[0];
    let x1 = eval[1];
    let x2 = eval[2];

    let s1 = scalar * x0;
    let s0 = scalar - s1;

    let s01 = s0 * x1;
    let s00 = s0 - s01;
    let s11 = s1 * x1;
    let s10 = s1 - s11;

    let s001 = s00 * x2;
    let s000 = s00 - s001;
    let s011 = s01 * x2;
    let s010 = s01 - s011;
    let s101 = s10 * x2;
    let s100 = s10 - s101;
    let s111 = s11 * x2;
    let s110 = s11 - s111;

    [s000, s001, s010, s011, s100, s101, s110, s111]
}

/// Computes the equality polynomial evaluations via a simple recursive algorithm.
///
/// Given an evaluation point vector `eval`, the function computes
/// the equality polynomial recursively using the formula:
///
/// ```text
/// eq(X) = scalar * ∏ (1 - X_i + 2X_i z_i)
/// ```
///
/// where `z_i` are the constraint points. It then updates the output buffer `out`
/// with the computed values by adding them in.
#[allow(clippy::too_many_lines)]
#[inline]
fn eval_eq_basic<F: Field, EF: ExtensionField<F>>(eval: &[EF], out: &mut [EF], scalar: EF) {
    // Ensure that the output buffer size is correct:
    // It should be of size `2^n`, where `n` is the number of variables.
    debug_assert_eq!(out.len(), 1 << eval.len());

    match eval.len() {
        0 => {
            out[0] += scalar;
        }
        1 => {
            // Manually unroll for single variable case
            let eq_evaluations = eval_eq_1(eval, scalar);

            out[0] += eq_evaluations[0];
            out[1] += eq_evaluations[1];
        }
        2 => {
            // Manually unroll for two variable case
            let eq_evaluations = eval_eq_2(eval, scalar);

            out.iter_mut()
                .zip(eq_evaluations.iter())
                .for_each(|(out, eq_eval)| {
                    *out += *eq_eval;
                });
        }
        3 => {
            // Manually unroll for three variable case
            let eq_evaluations = eval_eq_3(eval, scalar);

            out.iter_mut()
                .zip(eq_evaluations.iter())
                .for_each(|(out, eq_eval)| {
                    *out += *eq_eval;
                });
        }
        _ => {
            let (&x, tail) = eval.split_first().unwrap();

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

            // The recursive approach turns out to be faster than the iterative one here.
            // Probably related to nice cache locality.
            eval_eq_basic(tail, low, s0);
            eval_eq_basic(tail, high, s1);
        }
    }
}

/// Computes the equality polynomial evaluations via a simple recursive algorithm.
///
/// Unlike [`eval_eq_basic`], this function makes heavy use of packed values to speed up computations.
/// In particular `scalar` should be passed in as a packed value coming from [`packed_eq_poly`].
///
/// Essentially using packings this functions computes
///
/// ```text
/// eq(X) = scalar[j] * ∏ (1 - X_i + 2X_i z_i)
/// ```
///
/// for a collection of `i` at the same time. Here `scalar[j]` should be though of as evaluations of an equality
/// polynomial over different variables so `eq(X)` ends up being the evaluation of the equality polynomial over
/// the combined set of variables.
///
/// It then updates the output buffer `out` with the computed values by adding them in.
#[allow(clippy::too_many_lines)]
#[inline]
fn eval_eq_packed<F: Field, EF: ExtensionField<F>>(
    eval: &[EF],
    out: &mut [EF],
    scalar: EF::ExtensionPacking,
) {
    // Ensure that the output buffer size is correct:
    // It should be of size `2^n`, where `n` is the number of variables.
    let width = F::Packing::WIDTH;
    debug_assert_eq!(out.len(), width << eval.len());

    match eval.len() {
        0 => {
            // TODO: Going to add something to Plonky3 to make this more efficient.
            // Will let us add using the in-built packings.
            EF::ExtensionPacking::to_ext_iter([scalar])
                .zip(out.iter_mut())
                .for_each(|(scalar, out)| {
                    *out += scalar;
                });
        }
        1 => {
            // Manually unroll for single variable case
            let eq_evaluations = eval_eq_1(eval, scalar);

            // TODO: Going to add something to Plonky3 to make this more efficient.
            // Will let us add using the in-built packings.
            EF::ExtensionPacking::to_ext_iter(eq_evaluations)
                .zip(out.iter_mut())
                .for_each(|(scalar, out)| {
                    *out += scalar;
                });
        }
        2 => {
            // Manually unroll for two variables case
            let eq_evaluations = eval_eq_2(eval, scalar);

            // TODO: Going to add something to Plonky3 to make this more efficient.
            // Will let us add using the in-built packings.
            EF::ExtensionPacking::to_ext_iter(eq_evaluations)
                .zip(out.iter_mut())
                .for_each(|(scalar, out)| {
                    *out += scalar;
                });
        }
        3 => {
            // Manually unroll for single variable case
            let eq_evaluations = eval_eq_3(eval, scalar);

            // TODO: Going to add something to Plonky3 to make this more efficient.
            // Will let us add using the in-built packings.
            EF::ExtensionPacking::to_ext_iter(eq_evaluations)
                .zip(out.iter_mut())
                .for_each(|(scalar, out)| {
                    *out += scalar;
                });
        }
        _ => {
            let (&x, tail) = eval.split_first().unwrap();

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

            // The recursive approach turns out to be faster than the iterative one here.
            // Probably related to nice cache locality.
            eval_eq_packed(tail, low, s0);
            eval_eq_packed(tail, high, s1);
        }
    }
}

/// Computes equality polynomial evaluations and packs them into a `PackedFieldExtension`.
///
/// The length of `eval` must be equal to the `log2` of `F::Packing::WIDTH`.
#[allow(clippy::inline_always)] // Adding inline(always) seems to give a small performance boost.
#[inline(always)]
fn packed_eq_poly<F: Field, EF: ExtensionField<F>>(
    eval: &[EF],
    scalar: EF,
) -> EF::ExtensionPacking {
    // As this function is only available in this file, debug_assert should be fine here.
    // If this function becomes public, this should be changed to an assert.
    debug_assert_eq!(F::Packing::WIDTH, 1 << eval.len());

    // We build up the evaluations of the equality polynomial in buffer.
    let mut buffer = EF::zero_vec(1 << eval.len());
    buffer[0] = scalar;

    for (ind, entry) in eval.iter().rev().enumerate() {
        // After round 1 buffer should look like:
        // [(1 - xn)*scalar, xn*scalar, 0, 0, ...]
        // After round 2 buffer should look like:
        // [(1 - xn)*(1 - x_{n-1})*scalar, xn*(1 - x_{n-1})*scalar,
        //      (1 - xn)*x_{n-1}*scalar, xn*x_{n-1}*scalar, 0, 0, ...]
        let stride = 1 << ind;
        for index in 0..stride {
            let val = buffer[index];
            let scaled_val = val * *entry;
            let new_val = val - scaled_val;

            buffer[index] = new_val;
            buffer[index + stride] = scaled_val;
        }
    }

    // Finally we need to do a "transpose" to get a `PackedFieldExtension` element.
    EF::ExtensionPacking::from_ext_slice(&buffer)
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
}
