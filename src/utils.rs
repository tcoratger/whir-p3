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
///
/// If INITIALIZED is:
/// - false: the result is directly set to the `out` buffer
/// - true: the result is added to the `out` buffer
#[inline]
pub(crate) fn eval_eq<F: Field, EF: ExtensionField<F>, const INITIALIZED: bool>(
    eval: &[EF],
    out: &mut [EF],
    scalar: EF,
) {
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
    //
    // IMPORTANT: We previously checked here that `packing_width > 1`,
    // but this check is **not viable** for Goldilocks on Neon or when not using `target-cpu=native`.
    //
    // Why? Because Neon SIMD vectors are 128 bits and Goldilocks elements are already 64 bits,
    // so no packing happens (width stays 1), and there's no performance advantage.
    //
    // Be careful: this means code relying on packing optimizations should **not assume**
    // `packing_width > 1` is always true.
    let packing_width = F::Packing::WIDTH;
    // debug_assert!(packing_width > 1);

    // Ensure that the output buffer size is correct:
    // It should be of size `2^n`, where `n` is the number of variables.
    debug_assert_eq!(out.len(), 1 << eval.len());

    // If the number of variables is small, there is no need to use
    // parallelization or packings.
    if eval.len() <= packing_width + 1 + LOG_NUM_THREADS {
        // A basic recursive approach.
        eval_eq_basic::<_, _, INITIALIZED>(eval, out, scalar);
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
                eval_eq_packed::<_, _, INITIALIZED>(
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
                eval_eq_packed::<_, _, INITIALIZED>(
                    &eval[LOG_NUM_THREADS..(eval.len() - log_packing_width)],
                    out_chunk,
                    *buffer_val,
                );
            });
    }
}

/// Compute the scaled multilinear equality polynomial over `{0,1}` for a single variable.
///
/// This is the hardcoded base case for the equality polynomial `eq(x, z)`
/// in the case of a single variable `z = [z_0] ∈ 𝔽`, and returns:
///
/// \begin{equation}
/// [α ⋅ (1 - z_0), α ⋅ z_0]
/// \end{equation}
///
/// corresponding to the evaluations:
///
/// \begin{equation}
/// [α ⋅ eq(0, z), α ⋅ eq(1, z)]
/// \end{equation}
///
/// where the multilinear equality function is:
///
/// \begin{equation}
/// eq(x, z) = x ⋅ z + (1 - x)(1 - z)
/// \end{equation}
///
/// Concretely:
/// - For `x = 0`, we have:
///   \begin{equation}
///   eq(0, z_0) = 0 ⋅ z_0 + (1 - 0)(1 - z_0) = 1 - z_0
///   \end{equation}
/// - For `x = 1`, we have:
///   \begin{equation}
///   eq(1, z_0) = 1 ⋅ z_0 + (1 - 1)(1 - z_0) = z_0
///   \end{equation}
///
/// So the return value is:
/// - `[α ⋅ (1 - z_0), α ⋅ z_0]`
///
/// # Arguments
/// - `eval`: Slice containing the evaluation point `[z_0]` (must have length 1)
/// - `scalar`: A scalar multiplier `α` to scale the result by
///
/// # Returns
/// An array `[α ⋅ (1 - z_0), α ⋅ z_0]` representing the scaled evaluations
/// of `eq(x, z)` for `x ∈ {0,1}`.
#[allow(clippy::inline_always)] // Adding inline(always) seems to give a small performance boost.
#[inline(always)]
fn eval_eq_1<F: Field, FP: Algebra<F> + Copy>(eval: &[F], scalar: FP) -> [FP; 2] {
    // Extract the evaluation point z_0
    let z_0 = eval[0];

    // Compute α ⋅ z_0 = α ⋅ eq(1, z)
    let s1 = scalar * z_0;

    // Compute α ⋅ (1 - z_0) = α - α ⋅ z_0 = eq(0, z)
    let s0 = scalar - s1;

    // [eq(0, z), eq(1, z)]
    [s0, s1]
}

/// Compute the scaled multilinear equality polynomial over `{0,1}^2`.
///
/// This is the hardcoded base case for the multilinear equality polynomial `eq(x, z)`
/// when the evaluation point has 2 variables: `z = [z_0, z_1] ∈ 𝔽²`.
///
/// It computes and returns the vector:
///
/// \begin{equation}
/// [α ⋅ eq((0,0), z), α ⋅ eq((0,1), z), α ⋅ eq((1,0), z), α ⋅ eq((1,1), z)]
/// \end{equation}
///
/// where the multilinear equality polynomial is:
///
/// \begin{equation}
/// eq(x, z) = ∏_{i=0}^{1} (x_i ⋅ z_i + (1 - x_i)(1 - z_i))
/// \end{equation}
///
/// Concretely, this gives:
/// - `eq((0,0), z) = (1 - z_0)(1 - z_1)`
/// - `eq((0,1), z) = (1 - z_0)(z_1)`
/// - `eq((1,0), z) = z_0(1 - z_1)`
/// - `eq((1,1), z) = z_0(z_1)`
///
/// Then all outputs are scaled by `α`.
///
/// # Arguments
/// - `eval`: Slice `[z_0, z_1]`, the evaluation point in `𝔽²`
/// - `scalar`: The scalar multiplier `α ∈ 𝔽`
///
/// # Returns
/// An array `[α ⋅ eq((0,0), z), α ⋅ eq((0,1), z), α ⋅ eq((1,0), z), α ⋅ eq((1,1), z)]`
#[allow(clippy::inline_always)] // Helps with performance in tight loops
#[inline(always)]
fn eval_eq_2<F: Field, FP: Algebra<F> + Copy>(eval: &[F], scalar: FP) -> [FP; 4] {
    // First variable z_0
    let z_0 = eval[0];

    // Second variable z_1
    let z_1 = eval[1];

    // Compute s1 = α ⋅ z_0 = α ⋅ eq(x_0 = 1, -)
    let s1 = scalar * z_0;

    // Compute s0 = α - s1 = α ⋅ (1 - z_0) = α ⋅ eq(x_0 = 0, -)
    let s0 = scalar - s1;

    // For x_0 = 0:
    // - s01 = s0 ⋅ z_1 = α ⋅ (1 - z_0) ⋅ z_1 = α ⋅ eq((0,1), z)
    // - s00 = s0 - s01 = α ⋅ (1 - z_0)(1 - z_1) = α ⋅ eq((0,0), z)
    let s01 = s0 * z_1;
    let s00 = s0 - s01;

    // For x_0 = 1:
    // - s11 = s1 ⋅ z_1 = α ⋅ z_0 ⋅ z_1 = α ⋅ eq((1,1), z)
    // - s10 = s1 - s11 = α ⋅ z_0(1 - z_1) = α ⋅ eq((1,0), z)
    let s11 = s1 * z_1;
    let s10 = s1 - s11;

    // Return values in lexicographic order of x = (x_0, x_1)
    [s00, s01, s10, s11]
}

/// Compute the scaled multilinear equality polynomial over `{0,1}³` for 3 variables.
///
/// This is the hardcoded base case for the equality polynomial `eq(x, z)`
/// in the case of three variables `z = [z_0, z_1, z_2] ∈ 𝔽³`, and returns:
///
/// \begin{equation}
/// [α ⋅ eq((0,0,0), z), α ⋅ eq((0,0,1), z), ..., α ⋅ eq((1,1,1), z)]
/// \end{equation}
///
/// where the multilinear equality function is defined as:
///
/// \begin{equation}
/// \mathrm{eq}(x, z) = \prod_{i=0}^{2} \left( x_i z_i + (1 - x_i)(1 - z_i) \right)
/// \end{equation}
///
/// For each binary vector `x ∈ {0,1}³`, this returns the scaled evaluation `α ⋅ eq(x, z)`,
/// in lexicographic order: `(0,0,0), (0,0,1), ..., (1,1,1)`.
///
/// # Arguments
/// - `eval`: A slice containing `[z_0, z_1, z_2]`, the evaluation point.
/// - `scalar`: A scalar multiplier `α` to apply to all results.
///
/// # Returns
/// An array of 8 values `[α ⋅ eq(x, z)]` for all `x ∈ {0,1}³`, in lex order.
#[allow(clippy::inline_always)] // Adding inline(always) seems to give a small performance boost.
#[inline(always)]
fn eval_eq_3<F: Field, FP: Algebra<F> + Copy>(eval: &[F], scalar: FP) -> [FP; 8] {
    // Extract z_0, z_1, z_2 from the evaluation point
    let z_0 = eval[0];
    let z_1 = eval[1];
    let z_2 = eval[2];

    // First dimension split: scalar * z_0 and scalar * (1 - z_0)
    let s1 = scalar * z_0; // α ⋅ z_0
    let s0 = scalar - s1; // α ⋅ (1 - z_0)

    // Second dimension split:
    // Group (0, x1) branch using s0 = α ⋅ (1 - z_0)
    let s01 = s0 * z_1; // α ⋅ (1 - z_0) ⋅ z_1
    let s00 = s0 - s01; // α ⋅ (1 - z_0) ⋅ (1 - z_1)

    // Group (1, x1) branch using s1 = α ⋅ z_0
    let s11 = s1 * z_1; // α ⋅ z_0 ⋅ z_1
    let s10 = s1 - s11; // α ⋅ z_0 ⋅ (1 - z_1)

    // Third dimension split:
    // For (0,0,x2) branch
    let s001 = s00 * z_2; // α ⋅ (1 - z_0)(1 - z_1) ⋅ z_2
    let s000 = s00 - s001; // α ⋅ (1 - z_0)(1 - z_1) ⋅ (1 - z_2)

    // For (0,1,x2) branch
    let s011 = s01 * z_2; // α ⋅ (1 - z_0) ⋅ z_1 ⋅ z_2
    let s010 = s01 - s011; // α ⋅ (1 - z_0) ⋅ z_1 ⋅ (1 - z_2)

    // For (1,0,x2) branch
    let s101 = s10 * z_2; // α ⋅ z_0 ⋅ (1 - z_1) ⋅ z_2
    let s100 = s10 - s101; // α ⋅ z_0 ⋅ (1 - z_1) ⋅ (1 - z_2)

    // For (1,1,x2) branch
    let s111 = s11 * z_2; // α ⋅ z_0 ⋅ z_1 ⋅ z_2
    let s110 = s11 - s111; // α ⋅ z_0 ⋅ z_1 ⋅ (1 - z_2)

    // Return all 8 evaluations in lexicographic order of x ∈ {0,1}³
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
/// where `z_i` are the constraint points.
///
/// If INITIALIZED is:
/// - false: the result is directly set to the `out` buffer
/// - true: the result is added to the `out` buffer
#[allow(clippy::too_many_lines)]
#[inline]
fn eval_eq_basic<F: Field, EF: ExtensionField<F>, const INITIALIZED: bool>(
    eval: &[EF],
    out: &mut [EF],
    scalar: EF,
) {
    // Ensure that the output buffer size is correct:
    // It should be of size `2^n`, where `n` is the number of variables.
    debug_assert_eq!(out.len(), 1 << eval.len());

    match eval.len() {
        0 => {
            if INITIALIZED {
                out[0] += scalar;
            } else {
                out[0] = scalar;
            }
        }
        1 => {
            // Manually unroll for single variable case
            let eq_evaluations = eval_eq_1(eval, scalar);

            if INITIALIZED {
                out[0] += eq_evaluations[0];
                out[1] += eq_evaluations[1];
            } else {
                out[0] = eq_evaluations[0];
                out[1] = eq_evaluations[1];
            }
        }
        2 => {
            // Manually unroll for two variable case
            let eq_evaluations = eval_eq_2(eval, scalar);

            out.iter_mut()
                .zip(eq_evaluations.iter())
                .for_each(|(out, eq_eval)| {
                    if INITIALIZED {
                        *out += *eq_eval;
                    } else {
                        *out = *eq_eval;
                    }
                });
        }
        3 => {
            // Manually unroll for three variable case
            let eq_evaluations = eval_eq_3(eval, scalar);

            out.iter_mut()
                .zip(eq_evaluations.iter())
                .for_each(|(out, eq_eval)| {
                    if INITIALIZED {
                        *out += *eq_eval;
                    } else {
                        *out = *eq_eval;
                    }
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
            eval_eq_basic::<_, _, INITIALIZED>(tail, low, s0);
            eval_eq_basic::<_, _, INITIALIZED>(tail, high, s1);
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
/// for a collection of `i` at the same time. Here `scalar[j]` should be thought of as evaluations of an equality
/// polynomial over different variables so `eq(X)` ends up being the evaluation of the equality polynomial over
/// the combined set of variables.
///
/// It then updates the output buffer `out` with the computed values by adding them in.
#[allow(clippy::too_many_lines)]
#[inline]
fn eval_eq_packed<F: Field, EF: ExtensionField<F>, const INITIALIZED: bool>(
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
                    if INITIALIZED {
                        *out += scalar;
                    } else {
                        *out = scalar;
                    }
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
                    if INITIALIZED {
                        *out += scalar;
                    } else {
                        *out = scalar;
                    }
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
                    if INITIALIZED {
                        *out += scalar;
                    } else {
                        *out = scalar;
                    }
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
                    if INITIALIZED {
                        *out += scalar;
                    } else {
                        *out = scalar;
                    }
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
            eval_eq_packed::<_, _, INITIALIZED>(tail, low, s0);
            eval_eq_packed::<_, _, INITIALIZED>(tail, high, s1);
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

pub fn parallel_clone<A: Clone + Send + Sync>(src: &[A], dst: &mut [A]) {
    #[cfg(feature = "parallel")]
    if src.len() < 1 << 15 {
        // sequential copy
        dst.clone_from_slice(src);
    } else {
        assert_eq!(src.len(), dst.len());
        let chunk_size = src.len() / rayon::current_num_threads().max(1);
        dst.par_chunks_mut(chunk_size)
            .zip(src.par_chunks(chunk_size))
            .for_each(|(d, s)| {
                d.clone_from_slice(s);
            });
    }

    #[cfg(not(feature = "parallel"))]
    dst.clone_from_slice(src);
}

/// Recursively computes a chunk of the scaled multilinear equality polynomial over the Boolean hypercube.
///
/// Given an evaluation point $z ∈ 𝔽^n$ and a scalar multiplier $α ∈ 𝔽$, this function computes the values:
///
/// \begin{equation}
/// [α ⋅ eq(x, z)]_{x ∈ \{0,1\}^n}
/// \end{equation}
///
/// for all Boolean inputs $x$ within a **specific chunk** of the full hypercube, defined by `start_index` and `out.len()`.
///
/// The multilinear equality polynomial is defined as:
///
/// \begin{equation}
/// eq(x, z) = \prod_{i=0}^{n-1} \left(x_i z_i + (1 - x_i)(1 - z_i)\right)
/// \end{equation}
///
/// This recursive function updates a sub-slice `out` of a larger buffer with the correct scaled evaluations.
///
/// # Arguments
/// - `eval`: The evaluation point $z = (z_0, z_1, \dots, z_{n-1}) ∈ 𝔽^n$
/// - `out`: The mutable slice of the result buffer, containing a contiguous chunk of the Boolean hypercube
/// - `scalar`: The scalar multiplier $α ∈ 𝔽$
/// - `start_index`: The global starting index of `out` within the full $2^n$-sized hypercube
///
/// # Behavior
/// For a given chunk of the output buffer, this function computes:
///
/// \begin{equation}
/// out[i] += α ⋅ eq(x, z)
/// \end{equation}
///
/// where $x$ is the Boolean vector corresponding to index $i + \text{start_index}$ in lexicographic order.
///
/// # Recursive structure
/// - At each level of recursion, the function considers one variable $z_i$
/// - It determines whether the current chunk lies entirely in the $x_i = 0$ subcube, $x_i = 1$ subcube, or spans both
/// - It updates the `scalar` for each branch accordingly:
///
///   \begin{align}
///   s_1 &= α ⋅ z_i (for x_i = 1) \\
///   s_0 &= α ⋅ (1 - z_i) = α - s_1 (for x_i = 0 )
///   \end{align}
///
/// - It then recurses on the appropriate part(s) of `out`
pub(crate) fn eval_eq_chunked<F>(eval: &[F], out: &mut [F], scalar: F, start_index: usize)
where
    F: Field,
{
    // Early exit: Nothing to process if the output chunk is empty
    if out.is_empty() {
        return;
    }

    // Base case: all variables consumed → we’re at a leaf node of the recursion tree
    // Every point in the current chunk gets incremented by the scalar α
    if eval.is_empty() {
        for v in out.iter_mut() {
            *v += scalar;
        }
        return;
    }

    // --- Recursive step begins here ---

    // Split the input: extract current variable z_0 and the tail z_1..z_{n-1}
    let (&z_i, tail) = eval.split_first().unwrap();

    // The number of remaining variables after removing z_i
    let remaining_vars = tail.len();

    // The midpoint divides the current cube into two equal parts:
    //   - Lower half: x_i = 0 (indices 0..half)
    //   - Upper half: x_i = 1 (indices half..2^remaining_vars)
    let half = 1 << remaining_vars;

    // Compute branch scalars:
    //   - s1: contribution from x_i = 1
    //   - s0: contribution from x_i = 0
    //
    // These correspond to:
    //   s1 = α ⋅ z_i
    //   s0 = α ⋅ (1 - z_i) = α - s1
    let s1 = scalar * z_i;
    let s0 = scalar - s1;

    // Decide whether the current chunk falls entirely in one half or spans both halves

    if start_index + out.len() <= half {
        // Case 1: Entire chunk lies in the lower half (x_i = 0)
        // We recurse only into the s0 (x_i = 0) branch
        eval_eq_chunked(tail, out, s0, start_index);
    } else if start_index >= half {
        // Case 2: Entire chunk lies in the upper half (x_i = 1)
        // We recurse only into the s1 (x_i = 1) branch
        // We subtract `half` to make the index relative to the upper subcube
        eval_eq_chunked(tail, out, s1, start_index - half);
    } else {
        // Case 3: The chunk spans both subcubes
        // We split it at the midpoint and recurse into both branches

        // Number of elements in the lower half of the chunk
        let mid_point = half - start_index;

        // Split `out` into chunks for the x_i = 0 and x_i = 1 subcubes
        let (low_chunk, high_chunk) = out.split_at_mut(mid_point);

        // Optional parallelism for deep recursion trees
        #[cfg(feature = "parallel")]
        {
            const PARALLEL_THRESHOLD: usize = 10;

            if remaining_vars > PARALLEL_THRESHOLD {
                rayon::join(
                    || eval_eq_chunked(tail, low_chunk, s0, start_index),
                    || eval_eq_chunked(tail, high_chunk, s1, 0),
                );
                return;
            }
        }

        // Sequential fallback: recurse on both branches
        // x_i = 0 part
        eval_eq_chunked(tail, low_chunk, s0, start_index);
        // x_i = 1 part (new subproblem starts at 0)
        eval_eq_chunked(tail, high_chunk, s1, 0);
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

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::{PrimeCharacteristicRing, PrimeField64, extension::BinomialExtensionField};
    use proptest::prelude::*;
    use rand::{Rng, SeedableRng, distr::StandardUniform, rngs::StdRng};

    use super::*;

    type F = BabyBear;
    type EF4 = BinomialExtensionField<F, 4>;

    #[test]
    fn test_parallel_clone() {
        let src = (0..(1 << 25) + 7).map(F::from_u64).collect::<Vec<_>>();
        let mut dst_seq = F::zero_vec(src.len());
        let time = std::time::Instant::now();
        dst_seq.copy_from_slice(&src);
        println!("Sequential copy took: {:?}", time.elapsed());
        let mut dst_parallel = F::zero_vec(src.len());
        let time = std::time::Instant::now();
        parallel_clone(&src, &mut dst_parallel);
        println!("Parallel copy took: {:?}", time.elapsed());
        assert_eq!(dst_seq, dst_parallel);
    }

    #[test]
    fn test_eval_eq_functionality() {
        let mut output = vec![F::ZERO; 4]; // n=2 → 2^2 = 4 elements
        let eval = vec![F::from_u64(1), F::from_u64(0)]; // (X1, X2) = (1,0)
        let scalar = F::from_u64(2);

        eval_eq::<_, _, true>(&eval, &mut output, scalar);

        // Expected results for (X1, X2) = (1,0)
        let expected_output = vec![F::ZERO, F::ZERO, F::from_u64(2), F::ZERO];

        assert_eq!(output, expected_output);
    }

    /// Compute the multilinear equality polynomial over the boolean hypercube.
    ///
    /// Given an evaluation point `z ∈ 𝔽ⁿ` and a scalar `α ∈ 𝔽`, this function returns the vector of
    /// evaluations of the equality polynomial `eq(x, z)` over all boolean inputs `x ∈ {0,1}ⁿ`,
    /// scaled by the scalar.
    ///
    /// The equality polynomial is defined as:
    ///
    /// \begin{equation}
    /// \mathrm{eq}(x, z) = \prod_{i=0}^{n-1} \left( x_i z_i + (1 - x_i)(1 - z_i) \right)
    /// \end{equation}
    ///
    /// This function evaluates:
    ///
    /// \begin{equation}
    /// α \cdot \mathrm{eq}(x, z)
    /// \end{equation}
    ///
    /// for all `x ∈ {0,1}ⁿ`, and returns a vector of size `2ⁿ` containing these values in lexicographic order.
    ///
    /// # Arguments
    /// - `eval`: The vector `z ∈ 𝔽ⁿ`, representing the evaluation point.
    /// - `scalar`: The scalar `α ∈ 𝔽` to scale the result by.
    ///
    /// # Returns
    /// A vector `v` of length `2ⁿ`, where `v[i] = α ⋅ eq(xᵢ, z)`, and `xᵢ` is the binary vector corresponding
    /// to the `i`-th index in lex order (i.e., big-endian bit decomposition of `i`).
    fn naive_eq(eval: &[EF4], scalar: EF4) -> Vec<EF4> {
        // Number of boolean variables `n` = length of evaluation point
        let n = eval.len();

        // Allocate result vector of size 2^n, initialized to zero
        let mut result = vec![EF4::ZERO; 1 << n];

        // Iterate over each binary input `x ∈ {0,1}ⁿ`, indexed by `i`
        for (i, out) in result.iter_mut().enumerate() {
            // Convert index `i` to a binary vector `x ∈ {0,1}ⁿ` in big-endian order
            let x: Vec<EF4> = (0..n)
                .map(|j| {
                    let bit = (i >> (n - 1 - j)) & 1;
                    if bit == 1 { EF4::ONE } else { EF4::ZERO }
                })
                .collect();

            // Compute the equality polynomial:
            // eq(x, z) = ∏_{i=0}^{n-1} (xᵢ ⋅ zᵢ + (1 - xᵢ)(1 - zᵢ))
            let eq = x
                .iter()
                .zip(eval.iter())
                .map(|(xi, zi)| {
                    // Each term: xᵢ zᵢ + (1 - xᵢ)(1 - zᵢ)
                    *xi * *zi + (EF4::ONE - *xi) * (EF4::ONE - *zi)
                })
                .product::<EF4>(); // Take product over all coordinates

            // Store the scaled result: α ⋅ eq(x, z)
            *out = scalar * eq;
        }

        result
    }

    proptest! {
        #[test]
        fn prop_eval_eq_matches_naive(
            n in 1usize..6, // number of variables
            evals in prop::collection::vec(0u64..F::ORDER_U64, 1..6),
            scalar_val in 0u64..F::ORDER_U64,
        ) {
            // Take exactly n elements and map to EF4
            let evals: Vec<EF4> = evals.into_iter().take(n).map(EF4::from_u64).collect();
            let scalar = EF4::from_u64(scalar_val);

            // Make sure output has correct size: 2^n
            let out_len = 1 << evals.len();
            let mut output = vec![EF4::ZERO; out_len];

            eval_eq::<F, EF4, true>(&evals, &mut output, scalar);

            let expected = naive_eq(&evals, scalar);

            prop_assert_eq!(output, expected);
        }
    }

    #[test]
    fn test_eval_eq_1_against_naive() {
        let rng = &mut StdRng::seed_from_u64(0);

        // Choose a few values of z_0 and α to test
        let test_cases = vec![
            (rng.sample(StandardUniform), rng.sample(StandardUniform)),
            (rng.sample(StandardUniform), rng.sample(StandardUniform)),
            (rng.sample(StandardUniform), rng.sample(StandardUniform)),
            (rng.sample(StandardUniform), rng.sample(StandardUniform)),
            (rng.sample(StandardUniform), rng.sample(StandardUniform)),
        ];

        for (z_0, alpha) in test_cases {
            // Compute using the optimized eval_eq_1 function
            let result = eval_eq_1::<F, F>(&[z_0], alpha);

            // Compute eq(0, z_0) and eq(1, z_0) naively using the full formula:
            //
            // eq(x, z_0) = x * z_0 + (1 - x) * (1 - z_0)
            //
            let x0 = F::ZERO;
            let x1 = F::ONE;

            let eq_0 = x0 * z_0 + (F::ONE - x0) * (F::ONE - z_0);
            let eq_1 = x1 * z_0 + (F::ONE - x1) * (F::ONE - z_0);

            // Scale by α
            let expected_0 = alpha * eq_0;
            let expected_1 = alpha * eq_1;

            assert_eq!(
                result[0], expected_0,
                "eq(0, z_0) mismatch for z_0 = {z_0:?}"
            );
            assert_eq!(
                result[1], expected_1,
                "eq(1, z_0) mismatch for z_0 = {z_0:?}"
            );
        }
    }

    #[test]
    fn test_eval_eq_2_against_naive() {
        let rng = &mut StdRng::seed_from_u64(42);

        // Generate a few random test cases for (z_0, z_1) and α
        let test_cases = (0..5)
            .map(|_| {
                let z_0: F = rng.sample(StandardUniform);
                let z_1: F = rng.sample(StandardUniform);
                let alpha: F = rng.sample(StandardUniform);
                ([z_0, z_1], alpha)
            })
            .collect::<Vec<_>>();

        for ([z_0, z_1], alpha) in test_cases {
            // Optimized output
            let result = eval_eq_2::<F, F>(&[z_0, z_1], alpha);

            // Naive computation using the full formula:
            //
            // eq(x, z) = ∏ (x_i z_i + (1 - x_i)(1 - z_i))
            // for x ∈ { (0,0), (0,1), (1,0), (1,1) }

            let inputs = [
                (F::ZERO, F::ZERO), // x = (0,0)
                (F::ZERO, F::ONE),  // x = (0,1)
                (F::ONE, F::ZERO),  // x = (1,0)
                (F::ONE, F::ONE),   // x = (1,1)
            ];

            for (i, (x0, x1)) in inputs.iter().enumerate() {
                let eq_val = (*x0 * z_0 + (F::ONE - *x0) * (F::ONE - z_0))
                    * (*x1 * z_1 + (F::ONE - *x1) * (F::ONE - z_1));
                let expected = alpha * eq_val;

                assert_eq!(
                    result[i], expected,
                    "Mismatch at x = ({x0:?}, {x1:?}), z = ({z_0:?}, {z_1:?})"
                );
            }
        }
    }

    #[test]
    fn test_eval_eq_3_against_naive() {
        let rng = &mut StdRng::seed_from_u64(123);

        // Generate random test cases for (z_0, z_1, z_2) and α
        let test_cases = (0..5)
            .map(|_| {
                let z_0: F = rng.sample(StandardUniform);
                let z_1: F = rng.sample(StandardUniform);
                let z_2: F = rng.sample(StandardUniform);
                let alpha: F = rng.sample(StandardUniform);
                ([z_0, z_1, z_2], alpha)
            })
            .collect::<Vec<_>>();

        for ([z_0, z_1, z_2], alpha) in test_cases {
            // Optimized computation
            let result = eval_eq_3::<F, F>(&[z_0, z_1, z_2], alpha);

            // Naive computation using:
            // eq(x, z) = ∏ (x_i z_i + (1 - x_i)(1 - z_i))
            let inputs = [
                (F::ZERO, F::ZERO, F::ZERO), // (0,0,0)
                (F::ZERO, F::ZERO, F::ONE),  // (0,0,1)
                (F::ZERO, F::ONE, F::ZERO),  // (0,1,0)
                (F::ZERO, F::ONE, F::ONE),   // (0,1,1)
                (F::ONE, F::ZERO, F::ZERO),  // (1,0,0)
                (F::ONE, F::ZERO, F::ONE),   // (1,0,1)
                (F::ONE, F::ONE, F::ZERO),   // (1,1,0)
                (F::ONE, F::ONE, F::ONE),    // (1,1,1)
            ];

            for (i, (x0, x1, x2)) in inputs.iter().enumerate() {
                let eq_val = (*x0 * z_0 + (F::ONE - *x0) * (F::ONE - z_0))
                    * (*x1 * z_1 + (F::ONE - *x1) * (F::ONE - z_1))
                    * (*x2 * z_2 + (F::ONE - *x2) * (F::ONE - z_2));
                let expected = alpha * eq_val;

                assert_eq!(
                    result[i], expected,
                    "Mismatch at x = ({x0:?}, {x1:?}, {x2:?}), z = ({z_0:?}, {z_1:?}, {z_2:?})"
                );
            }
        }
    }

    proptest! {
        #[test]
        fn prop_chunked_eval_eq_matches_monolithic(
            n in 1usize..6, // number of variables
            evals in prop::collection::vec(0u64..F::ORDER_U64, 1..6),
            scalar_val in 0u64..F::ORDER_U64,
            num_chunks in 1usize..8, // how many chunks to split into
        ) {
            // Truncate evals to size `n` and convert to EF4
            let evals: Vec<EF4> = evals.into_iter().take(n).map(EF4::from_u64).collect();
            let scalar = EF4::from_u64(scalar_val);

            let size = 1 << evals.len();

            // Compute baseline result using monolithic `eval_eq`
            let mut expected = vec![EF4::ZERO; size];
            eval_eq::<F, EF4, false>(&evals, &mut expected, scalar);

            // Compute using chunked version
            let mut chunked = vec![EF4::ZERO; size];
            let chunk_size = size.div_ceil(num_chunks);

            for i in 0..num_chunks {
                let start = i * chunk_size;
                if start >= size {
                    break;
                }
                let end = (start + chunk_size).min(size);
                eval_eq_chunked::<EF4>(&evals, &mut chunked[start..end], scalar, start);
            }

            prop_assert_eq!(chunked, expected);
        }
    }
}
