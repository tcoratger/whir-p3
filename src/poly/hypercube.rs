use core::ops::Range;
use std::ops::{Deref, DerefMut};

/// Represents a point on the binary hypercube `{0,1}^n`.
///
/// The point is encoded via the `n` least significant bits of a `usize` in big-endian order.
/// The struct does not store `n`; interpretation relies on context.
#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BinaryHypercubePoint(pub usize);

impl Deref for BinaryHypercubePoint {
    type Target = usize;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for BinaryHypercubePoint {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Iterator over all points of the binary hypercube `{0,1}^n` in lexicographic order.
///
/// Yields `2^n` points from `0..(1<<n)`.
#[derive(Debug, Clone)]
pub struct BinaryHypercube {
    /// Range of points to yield.
    range: Range<usize>,
}

impl BinaryHypercube {
    /// Constructs a new iterator for `{0,1}^num_variables`.
    #[must_use]
    #[inline]
    pub const fn new(num_variables: usize) -> Self {
        // keep the debug guard; shifting by >= word size is UB
        debug_assert!(num_variables < usize::BITS as usize);
        // `1usize << n` is fine because of the assert above
        let end = 1usize << num_variables;
        Self { range: 0..end }
    }

    /// Remaining number of points that will be yielded.
    #[inline]
    pub fn len(&self) -> usize {
        self.range.end.saturating_sub(self.range.start)
    }

    /// Whether iteration is finished.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.range.is_empty()
    }
}

impl Iterator for BinaryHypercube {
    type Item = BinaryHypercubePoint;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.range.next().map(BinaryHypercubePoint)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.len();
        (rem, Some(rem))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_hypercube_iterator() {
        let mut hypercube = BinaryHypercube::new(2);

        assert_eq!(hypercube.next(), Some(BinaryHypercubePoint(0)));
        assert_eq!(hypercube.next(), Some(BinaryHypercubePoint(1)));
        assert_eq!(hypercube.next(), Some(BinaryHypercubePoint(2)));
        assert_eq!(hypercube.next(), Some(BinaryHypercubePoint(3)));
        assert_eq!(hypercube.next(), None);
    }

    #[test]
    fn test_binary_hypercube_single_dimension() {
        let mut hypercube = BinaryHypercube::new(1);

        assert_eq!(hypercube.next(), Some(BinaryHypercubePoint(0)));
        assert_eq!(hypercube.next(), Some(BinaryHypercubePoint(1)));
        assert_eq!(hypercube.next(), None);
    }

    #[test]
    fn test_binary_hypercube_zero_dimensions() {
        let mut hypercube = BinaryHypercube::new(0);

        assert_eq!(hypercube.next(), Some(BinaryHypercubePoint(0)));
        assert_eq!(hypercube.next(), None);
    }

    #[test]
    fn test_binary_hypercube_large_dimensions() {
        let n = 5;
        let hypercube = BinaryHypercube::new(n);
        let expected_size = 1 << n;

        let count = hypercube.count();
        assert_eq!(count, expected_size);
    }

    #[test]
    fn test_binary_hypercube_point_order() {
        let mut hypercube = BinaryHypercube::new(3);
        let expected_points = [
            BinaryHypercubePoint(0),
            BinaryHypercubePoint(1),
            BinaryHypercubePoint(2),
            BinaryHypercubePoint(3),
            BinaryHypercubePoint(4),
            BinaryHypercubePoint(5),
            BinaryHypercubePoint(6),
            BinaryHypercubePoint(7),
        ];

        for &expected in &expected_points {
            assert_eq!(hypercube.next(), Some(expected));
        }
        assert_eq!(hypercube.next(), None);
    }
}
