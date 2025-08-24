use p3_maybe_rayon::prelude::*;

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
