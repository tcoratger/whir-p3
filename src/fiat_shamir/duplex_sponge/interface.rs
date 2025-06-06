/// Basic units over which a sponge operates.
///
/// We require the units to have a precise size in memory, to be cloneable,
/// and that we can zeroize them.
pub trait Unit: Clone + Sized + zeroize::Zeroize {
    /// Write a bunch of units in the wire.
    fn write(bunch: &[Self], w: &mut impl std::io::Write) -> Result<(), std::io::Error>;

    /// Read a bunch of units from the wire
    fn read(r: &mut impl std::io::Read, bunch: &mut [Self]) -> Result<(), std::io::Error>;

    /// Convert a `u8` to a unit.
    fn from_u8(x: u8) -> Self;

    /// Convert a `Unit` to a `u8`.
    fn to_u8(x: Self) -> u8;

    /// Convert a `[u8]` slice to a `Vec<Self>`.
    #[must_use]
    fn slice_from_u8_slice(src: &[u8]) -> Vec<Self> {
        src.iter().map(|&b| Self::from_u8(b)).collect()
    }

    /// Convert [U; N] → [u8; N] (only if U is u8-width).
    fn array_to_u8_array<const N: usize>(src: &[Self; N]) -> [u8; N] {
        assert_eq!(std::mem::size_of::<Self>(), 1, "Unit must be 1 byte");
        let src_bytes: &[u8] = unsafe { std::slice::from_raw_parts(src.as_ptr().cast::<u8>(), N) };
        let mut out = [0u8; N];
        out.copy_from_slice(src_bytes);
        out
    }

    /// Zero-copy convert &[Self] → &[u8] (only if U is u8-width).
    fn slice_to_u8_slice(src: &[Self]) -> &[u8] {
        assert_eq!(std::mem::size_of::<Self>(), 1, "Unit must be 1 byte");
        unsafe { std::slice::from_raw_parts(src.as_ptr().cast::<u8>(), src.len()) }
    }
}

impl Unit for u8 {
    fn write(bunch: &[Self], w: &mut impl std::io::Write) -> Result<(), std::io::Error> {
        w.write_all(bunch)
    }

    fn read(r: &mut impl std::io::Read, bunch: &mut [Self]) -> Result<(), std::io::Error> {
        r.read_exact(bunch)
    }

    fn from_u8(x: u8) -> Self {
        x
    }

    fn to_u8(x: Self) -> u8 {
        x
    }

    fn slice_from_u8_slice(src: &[u8]) -> Vec<Self> {
        src.to_vec()
    }
}
