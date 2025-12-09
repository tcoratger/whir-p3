use alloc::vec::Vec;

// use p3_field::{BasedVectorSpace, ExtensionField, Field, PackedFieldExtension, PackedValue};
// use p3_maybe_rayon::prelude::*;

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
