use p3_field::Field;

use super::{hypercube::BinaryHypercubePoint, multilinear::MultilinearPoint};

/// Iterator for evaluating the Lagrange polynomial over the hypercube `{0,1}^n`.
///
/// This efficiently computes values of the equality polynomial at every binary point.
///
/// Given a multilinear point `(c_1, ..., c_n)`, it iterates over all binary vectors `(x_1, ...,
/// x_n)` and computes:
///
/// \begin{equation}
/// y = \prod_{i=1}^{n} \left( x_i c_i + (1 - x_i) (1 - c_i) \right)
/// \end{equation}
///
/// This means `y = eq_poly(c, x)`, where `eq_poly` is the **equality polynomial**.
///
/// # Properties
/// - **Precomputed negations**: We store `1 - c_i` to avoid recomputation.
#[derive(Debug)]
pub struct LagrangePolynomialIterator<F> {
    /// The last binary point output (`None` before the first step).
    last_position: Option<usize>,
    /// The point `(c_1, ..., c_n)` stored in **reverse order** for efficient access.
    point: Vec<F>,
    /// Precomputed values `1 - c_i` stored in **reverse order**.
    point_negated: Vec<F>,
    /// Stack containing **partial products**:
    /// - Before first iteration: `[1, y_1, y_1 y_2, ..., y_1 ... y_n]`
    /// - After each iteration: updated values corresponding to the next `x`
    stack: Vec<F>,
    /// The number of variables `n` (i.e., dimension of the hypercube).
    num_variables: usize,
}

impl<F: Field> From<&MultilinearPoint<F>> for LagrangePolynomialIterator<F> {
    /// Initializes the iterator from a multilinear point `(c_1, ..., c_n)`.
    ///
    /// # Initialization:
    /// - Stores `c_i` in reverse order for **efficient bit processing**.
    /// - Precomputes `1 - c_i` for each coordinate to avoid recomputation.
    /// - Constructs a **stack** for incremental computation.
    fn from(multilinear_point: &MultilinearPoint<F>) -> Self {
        let num_variables = multilinear_point.num_variables();

        // Clone the original point (c_1, ..., c_n)
        let mut point = multilinear_point.0.clone();

        // Compute point_negated = (1 - c_1, ..., 1 - c_n)
        let mut point_negated: Vec<_> = point.iter().map(|&x| F::ONE - x).collect();

        // Compute the stack of partial products (1, (1 - c_1), ..., ∏_{i=1}^{n} (1 - c_i))
        let mut stack = Vec::with_capacity(num_variables + 1);
        let mut running_product = F::ONE;
        stack.push(running_product); // stack[0] = 1

        for &neg in &point_negated {
            running_product *= neg;
            stack.push(running_product);
        }

        // Reverse the point and its negation for bit-friendly access
        point.reverse();
        point_negated.reverse();

        Self {
            last_position: None,
            point,
            point_negated,
            stack,
            num_variables,
        }
    }
}

impl<F: Field> Iterator for LagrangePolynomialIterator<F> {
    type Item = (BinaryHypercubePoint, F);

    /// Computes the next `(x, y)` pair where `y = eq_poly(c, x)`.
    ///
    /// - The first iteration **outputs** `(0, y_1 ... y_n)`, where `y_i = (1 - c_i)`.
    /// - Subsequent iterations **update** `y` using binary code ordering, minimizing
    ///   recomputations.
    fn next(&mut self) -> Option<Self::Item> {
        let pos = self.last_position.map_or(0, |p| p + 1);
        if pos >= (1 << self.num_variables) {
            return None;
        }

        // First iteration: return the initial product
        if self.last_position.is_none() {
            self.last_position = Some(0);
            return Some((BinaryHypercubePoint(0), *self.stack.last().unwrap()));
        }

        // Compute bit difference (position of the changed bit)
        let bit_diff = self.last_position.unwrap() ^ pos;
        let low_idx = (bit_diff + 1).trailing_zeros() as usize;

        // Discard unused values from the stack
        self.stack.truncate(self.stack.len() - low_idx);

        // Compute new values up to `low_idx`
        for i in (0..low_idx).rev() {
            let last = *self.stack.last().unwrap();
            let next_bit = (pos & (1 << i)) != 0;
            self.stack.push(
                last * if next_bit {
                    self.point[i]
                } else {
                    self.point_negated[i]
                },
            );
        }

        self.last_position = Some(pos);
        Some((BinaryHypercubePoint(pos), *self.stack.last().unwrap()))
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    use super::*;

    #[test]
    fn test_lagrange_iterator_single_variable() {
        let point = MultilinearPoint(vec![BabyBear::from_u64(3)]);
        let mut iter = LagrangePolynomialIterator::from(&point);

        // Expected values: (0, 1 - p) and (1, p)
        assert_eq!(
            iter.next(),
            Some((BinaryHypercubePoint(0), BabyBear::ONE - point.0[0]))
        );
        assert_eq!(iter.next(), Some((BinaryHypercubePoint(1), point.0[0])));
        assert_eq!(iter.next(), None); // No more elements should be present
    }

    #[test]
    fn test_lagrange_iterator_two_variables() {
        let (a, b) = (BabyBear::from_u64(2), BabyBear::from_u64(3));
        let point = MultilinearPoint(vec![a, b]);
        let mut iter = LagrangePolynomialIterator::from(&point);

        // Expected values based on binary enumeration (big-endian)
        assert_eq!(
            iter.next(),
            Some((
                BinaryHypercubePoint(0b00),
                (BabyBear::ONE - a) * (BabyBear::ONE - b)
            ))
        );
        assert_eq!(
            iter.next(),
            Some((BinaryHypercubePoint(0b01), (BabyBear::ONE - a) * b))
        );
        assert_eq!(
            iter.next(),
            Some((BinaryHypercubePoint(0b10), a * (BabyBear::ONE - b)))
        );
        assert_eq!(iter.next(), Some((BinaryHypercubePoint(0b11), a * b)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_lagrange_iterator_all_zeros() {
        let point = MultilinearPoint(vec![BabyBear::ZERO; 3]);
        let mut iter = LagrangePolynomialIterator::from(&point);

        // Expect all outputs to be 1 when x is all zeros and 0 otherwise
        assert_eq!(
            iter.next(),
            Some((BinaryHypercubePoint(0b000), BabyBear::ONE))
        );
        assert_eq!(
            iter.next(),
            Some((BinaryHypercubePoint(0b001), BabyBear::ZERO))
        );
        assert_eq!(
            iter.next(),
            Some((BinaryHypercubePoint(0b010), BabyBear::ZERO))
        );
        assert_eq!(
            iter.next(),
            Some((BinaryHypercubePoint(0b011), BabyBear::ZERO))
        );
        assert_eq!(
            iter.next(),
            Some((BinaryHypercubePoint(0b100), BabyBear::ZERO))
        );
        assert_eq!(
            iter.next(),
            Some((BinaryHypercubePoint(0b101), BabyBear::ZERO))
        );
        assert_eq!(
            iter.next(),
            Some((BinaryHypercubePoint(0b110), BabyBear::ZERO))
        );
        assert_eq!(
            iter.next(),
            Some((BinaryHypercubePoint(0b111), BabyBear::ZERO))
        );
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_lagrange_iterator_all_ones() {
        let point = MultilinearPoint(vec![BabyBear::ONE; 3]);
        let mut iter = LagrangePolynomialIterator::from(&point);

        // Expect all outputs to be 1 when x is all ones and 0 otherwise
        assert_eq!(
            iter.next(),
            Some((BinaryHypercubePoint(0b000), BabyBear::ZERO))
        );
        assert_eq!(
            iter.next(),
            Some((BinaryHypercubePoint(0b001), BabyBear::ZERO))
        );
        assert_eq!(
            iter.next(),
            Some((BinaryHypercubePoint(0b010), BabyBear::ZERO))
        );
        assert_eq!(
            iter.next(),
            Some((BinaryHypercubePoint(0b011), BabyBear::ZERO))
        );
        assert_eq!(
            iter.next(),
            Some((BinaryHypercubePoint(0b100), BabyBear::ZERO))
        );
        assert_eq!(
            iter.next(),
            Some((BinaryHypercubePoint(0b101), BabyBear::ZERO))
        );
        assert_eq!(
            iter.next(),
            Some((BinaryHypercubePoint(0b110), BabyBear::ZERO))
        );
        assert_eq!(
            iter.next(),
            Some((BinaryHypercubePoint(0b111), BabyBear::ONE))
        );
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_lagrange_iterator_mixed_values() {
        let (a, b, c) = (
            BabyBear::from_u64(2),
            BabyBear::from_u64(3),
            BabyBear::from_u64(4),
        );
        let point = MultilinearPoint(vec![a, b, c]);
        let mut iter = LagrangePolynomialIterator::from(&point);

        // Verify correctness against eq_poly function
        for (b, lag) in LagrangePolynomialIterator::from(&point) {
            assert_eq!(point.eq_poly(b), lag);
        }

        // Ensure the iterator completes all 2^3 = 8 elements
        let mut count = 0;
        while iter.next().is_some() {
            count += 1;
        }
        assert_eq!(count, 8);
    }

    #[test]
    fn test_lagrange_iterator_four_variables() {
        let point = MultilinearPoint(vec![
            BabyBear::from_u64(1),
            BabyBear::from_u64(2),
            BabyBear::from_u64(3),
            BabyBear::from_u64(4),
        ]);
        let mut iter = LagrangePolynomialIterator::from(&point);

        // Ensure the iterator completes all 2^4 = 16 elements
        let mut count = 0;
        while iter.next().is_some() {
            count += 1;
        }
        assert_eq!(count, 16);
    }

    #[test]
    fn test_lagrange_iterator_correct_order() {
        let point = MultilinearPoint(vec![BabyBear::from_u64(5), BabyBear::from_u64(7)]);
        let mut iter = LagrangePolynomialIterator::from(&point);

        // Expect values in **binary order**: 0b00, 0b01, 0b10, 0b11
        let expected_order = [0b00, 0b01, 0b10, 0b11];

        for &expected in &expected_order {
            let (b, _) = iter.next().unwrap();
            assert_eq!(b.0, expected);
        }

        // Ensure no extra values are generated
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_lagrange_iterator_output_count() {
        let num_vars = 5;
        let point = MultilinearPoint(vec![BabyBear::from_u64(3); num_vars]);
        let iter = LagrangePolynomialIterator::from(&point);

        // The iterator should yield exactly 2^num_vars elements
        assert_eq!(iter.count(), 1 << num_vars);
    }

    #[test]
    fn test_lagrange_iterator_empty() {
        let point = MultilinearPoint::<BabyBear>(vec![]);
        let mut iter = LagrangePolynomialIterator::from(&point);

        // Only a single output should be generated
        assert_eq!(iter.next(), Some((BinaryHypercubePoint(0), BabyBear::ONE)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_blendy() {
        let one = BabyBear::from_u64(1);
        let (a, b) = (BabyBear::from_u64(2), BabyBear::from_u64(3));
        let point_1 = MultilinearPoint(vec![a, b]);

        let mut lag_iterator = LagrangePolynomialIterator::from(&point_1);

        assert_eq!(
            lag_iterator.next().unwrap(),
            (BinaryHypercubePoint(0), (one - a) * (one - b))
        );
        assert_eq!(
            lag_iterator.next().unwrap(),
            (BinaryHypercubePoint(1), (one - a) * b)
        );
        assert_eq!(
            lag_iterator.next().unwrap(),
            (BinaryHypercubePoint(2), a * (one - b))
        );
        assert_eq!(
            lag_iterator.next().unwrap(),
            (BinaryHypercubePoint(3), a * b)
        );
        assert_eq!(lag_iterator.next(), None);
    }

    #[test]
    fn test_blendy_2() {
        let point = MultilinearPoint(vec![
            BabyBear::from_u64(12),
            BabyBear::from_u64(13),
            BabyBear::from_u64(32),
        ]);

        let mut last_b = None;
        for (b, lag) in LagrangePolynomialIterator::from(&point) {
            assert_eq!(point.eq_poly(b), lag);
            assert!(b.0 < 1 << 3);
            last_b = Some(b);
        }
        assert_eq!(last_b, Some(BinaryHypercubePoint(7)));
    }

    #[test]
    fn test_blendy_3() {
        let point = MultilinearPoint(vec![
            BabyBear::from_u64(414_151),
            BabyBear::from_u64(109_849_018),
            BabyBear::from_u64(33_184_190),
            BabyBear::from_u64(33_184_190),
            BabyBear::from_u64(33_184_190),
        ]);

        let mut last_b = None;
        for (b, lag) in LagrangePolynomialIterator::from(&point) {
            assert_eq!(point.eq_poly(b), lag);
            last_b = Some(b);
        }
        assert_eq!(last_b, Some(BinaryHypercubePoint(31)));
    }
}
