use p3_field::{BasedVectorSpace, ExtensionField, Field};
use p3_maybe_rayon::prelude::*;

/// Scales the evaluations by scalar and either adds the result to the output buffer or
/// sets the output buffer directly depending on the `INITIALIZED` flag.
///
/// If the output buffer is already initialized, it adds the evaluations otherwise
/// it copies the evaluations into the buffer directly.
#[inline]
fn scale_and_add<F: Field, EF: ExtensionField<F>, const INITIALIZED: bool>(
    out: &mut [EF],
    base_vals: &[F],
    scalar: EF,
) {
    // TODO: We can probably add a custom method to Plonky3 to handle this more efficiently (and use packings).
    // This approach is faster than collecting `scalar * eq_eval` into a vector and using `add_slices`. Presumably
    // this is because we avoid the allocation.
    if INITIALIZED {
        out.iter_mut().zip(base_vals).for_each(|(out, &eq_eval)| {
            *out += scalar * eq_eval;
        });
    } else {
        out.iter_mut().zip(base_vals).for_each(|(out, &eq_eval)| {
            *out = scalar * eq_eval;
        });
    }
}

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
    // Parallelism for deep recursion trees
    const PARALLEL_THRESHOLD: usize = 10;

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

        if remaining_vars > PARALLEL_THRESHOLD {
            join(
                || eval_eq_chunked(tail, low_chunk, s0, start_index),
                || eval_eq_chunked(tail, high_chunk, s1, 0),
            );
        } else {
            eval_eq_chunked(tail, low_chunk, s0, start_index);
            eval_eq_chunked(tail, high_chunk, s1, 0);
        }
    }
}

pub fn flatten_scalars_to_base<F, EF>(scalars: &[EF]) -> Vec<F>
where
    F: Field,
    EF: ExtensionField<F>,
{
    scalars
        .iter()
        .flat_map(BasedVectorSpace::as_basis_coefficients_slice)
        .copied()
        .collect()
}

pub fn pack_scalars_to_extension<F, EF>(scalars: &[F]) -> Vec<EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let extension_size = <EF as BasedVectorSpace<F>>::DIMENSION;
    assert!(
        scalars.len() % extension_size == 0,
        "Scalars length must be a multiple of the extension size"
    );
    scalars
        .chunks_exact(extension_size)
        .map(|chunk| EF::from_basis_coefficients_slice(chunk).unwrap())
        .collect()
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
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};

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
}
