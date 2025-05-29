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
pub trait DuplexSpongeInterface<C: Permutation<[u8; 200]>, U = u8>: zeroize::Zeroize
where
    U: Unit,
{
    /// Initializes a new sponge, setting up the state.
    fn new(permutation: C, iv: [u8; 32]) -> Self;

    /// Absorbs new elements in the sponge.
    fn absorb_unchecked(&mut self, input: &[U]) -> &mut Self;

    /// Squeezes out new elements.
    fn squeeze_unchecked(&mut self, output: &mut [U]) -> &mut Self;
}

/// Basic units over which a sponge operates.
///
/// We require the units to have a precise size in memory, to be cloneable,
/// and that we can zeroize them.
pub trait Unit: Clone + Sized + zeroize::Zeroize {
    /// Write a bunch of units in the wire.
    fn write(bunch: &[Self], w: &mut impl std::io::Write) -> Result<(), std::io::Error>;
    /// Read a bunch of units from the wire
    fn read(r: &mut impl std::io::Read, bunch: &mut [Self]) -> Result<(), std::io::Error>;
}

impl Unit for u8 {
    fn write(bunch: &[Self], w: &mut impl std::io::Write) -> Result<(), std::io::Error> {
        w.write_all(bunch)
    }

    fn read(r: &mut impl std::io::Read, bunch: &mut [Self]) -> Result<(), std::io::Error> {
        r.read_exact(bunch)
    }
}
