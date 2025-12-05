use alloc::vec::Vec;

use p3_field::{BasedVectorSpace, ExtensionField, Field, PackedFieldExtension, PackedValue};
use p3_maybe_rayon::prelude::*;

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

/// Unpack packed extension field elements into the standard representation.
#[inline]
pub fn unpack_slice_into<F: Field, Ext: ExtensionField<F>>(
    out: &mut [Ext],
    packed: &[Ext::ExtensionPacking],
) {
    const PARALLEL_THRESHOLD: usize = 4096;
    assert_eq!(out.len(), packed.len() * F::Packing::WIDTH);
    if packed.len() < PARALLEL_THRESHOLD {
        packed
            .iter()
            .zip(out.chunks_mut(F::Packing::WIDTH))
            .for_each(|(packed, out_chunk)| {
                let packed_coeffs = packed.as_basis_coefficients_slice();
                for (i, out) in out_chunk.iter_mut().enumerate().take(F::Packing::WIDTH) {
                    *out = Ext::from_basis_coefficients_fn(|j| packed_coeffs[j].as_slice()[i]);
                }
            });
    } else {
        packed
            .par_iter()
            .zip(out.par_chunks_mut(F::Packing::WIDTH))
            .for_each(|(packed, out_chunk)| {
                let packed_coeffs = packed.as_basis_coefficients_slice();
                for (i, out) in out_chunk.iter_mut().enumerate().take(F::Packing::WIDTH) {
                    *out = Ext::from_basis_coefficients_fn(|j| packed_coeffs[j].as_slice()[i]);
                }
            });
    }
}

/// Unpack packed extension field elements to the standard representation.
#[inline]
pub fn unpack_slice<F: Field, EF: ExtensionField<F>>(
    packed: &[EF::ExtensionPacking],
) -> Vec<EF> {
    let mut out = EF::zero_vec(packed.len() * F::Packing::WIDTH);
    unpack_slice_into(&mut out, packed);
    out
}

#[inline]
/// Pack extension field elements into their packed representation.
pub fn pack_slice<F: Field, EF: ExtensionField<F>>(slice: &[EF]) -> Vec<EF::ExtensionPacking> {
    const PARALLEL_THRESHOLD: usize = 4096;
    if slice.len() < PARALLEL_THRESHOLD {
        slice
            .chunks(F::Packing::WIDTH)
            .map(|ext| EF::ExtensionPacking::from_ext_slice(ext))
            .collect()
    } else {
        slice
            .par_chunks(F::Packing::WIDTH)
            .map(|ext| EF::ExtensionPacking::from_ext_slice(ext))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};
    use p3_util::log2_strict_usize;
    use rand::{Rng, SeedableRng, rngs::SmallRng};

    use super::*;

    type F = BabyBear;
    type PackedF = <F as Field>::Packing;
    type EF = BinomialExtensionField<F, 4>;
    type PackedEF = <EF as ExtensionField<F>>::ExtensionPacking;

    #[test]
    fn test_packing_roundtrip() {
        let mut rng = SmallRng::seed_from_u64(1);
        let width = PackedF::WIDTH;
        let k_packed = log2_strict_usize(width);
        for k in log2_strict_usize(PackedF::WIDTH)..10 {
            let unpacked0 = (0..1 << k).map(|_| rng.random()).collect::<Vec<_>>();
            let packed0 = pack_slice::<F, EF>(&unpacked0);
            assert_eq!(log2_strict_usize(packed0.len()), k - k_packed);
            let unpacked1 = unpack_slice::<F, EF>(&packed0);
            assert_eq!(unpacked0, unpacked1);
            let mut unpacked1 = EF::zero_vec(1 << k);
            unpack_slice_into::<F, EF>(&mut unpacked1, &packed0);
            assert_eq!(unpacked0, unpacked1);
        }
    }
}
