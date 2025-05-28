use p3_symmetric::Permutation;

/// A [`DuplexInterface`] is an abstract interface for absorbing and squeezing data.
/// The type parameter `U` represents basic unit that the sponge works with.
///
/// We require [`DuplexInterface`] implementations to have a [`std::default::Default`]
/// implementation, that initializes to zero the hash function state, and a [`zeroize::Zeroize`]
/// implementation for secure deletion.
///
/// **HAZARD**: Don't implement this trait unless you know what you are doing.
/// Consider using the sponges already provided by this library.
pub trait DuplexSpongeInterface<C: Permutation<[u8; 200]>>: zeroize::Zeroize {
    /// Initializes a new sponge, setting up the state.
    fn new(permutation: C, iv: [u8; 32]) -> Self;

    /// Absorbs new elements in the sponge.
    fn absorb_unchecked(&mut self, input: &[u8]) -> &mut Self;

    /// Squeezes out new elements.
    fn squeeze_unchecked(&mut self, output: &mut [u8]) -> &mut Self;
}
