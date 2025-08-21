use p3_field::Field;
use p3_multilinear_util::eq::eval_eq;

use crate::poly::multilinear::MultilinearPoint;

/// A query to a multilinear polynomial over an extension field.
///
/// This enum represents the type of query made to a multilinear polynomial.
/// It supports:
/// - Exact evaluation at a point (`Eq`)
/// - Evaluation at a rotated version of a point (`EqRotateRight`)
///
/// These queries are typically used in polynomial commitment schemes for verifying
/// proximity to multilinear codes or enforcing algebraic constraints.
///
/// # Type Parameters
/// - `Challenge`: The field element type used in queries (typically an extension field).
#[derive(Clone, Debug)]
pub enum MlQuery<F> {
    /// A standard query asking for evaluation of the polynomial at the given point.
    ///
    /// The point is represented as a vector of field elements corresponding to the
    /// $m$-dimensional input of a multilinear function $\hat{f} : \{0,1\}^m \to F$.
    ///
    /// # Example
    /// ```text
    /// Eq([α_0, α_1, α_2]) → query at (α_0, α_1, α_2)
    /// ```
    Eq(MultilinearPoint<F>),

    /// A rotated query: evaluate the polynomial at a rotated version of the point.
    ///
    /// This is useful when enforcing symmetries or constraints that involve bit rotations.
    /// The second argument `r` indicates a right rotation by `r` bits of the input vector.
    ///
    /// The rotation is **circular** (wraps around), and is applied *before* evaluation.
    ///
    /// # Example
    /// ```text
    /// EqRotateRight([α_0, α_1, α_2], 1) → query at (α_2, α_0, α_1)
    /// ```
    EqRotateRight(MultilinearPoint<F>, usize),
}

impl<F> MlQuery<F>
where
    F: Field,
{
    /// Returns the number of variables (`log_b`) in the multilinear query point.
    ///
    /// This corresponds to the dimensionality of the Boolean hypercube over which
    /// the multilinear polynomial is evaluated.
    ///
    /// The value `log_b` is the length of the underlying vector `z`, regardless
    /// of whether the query is a direct evaluation (`Eq`) or a rotated evaluation (`EqRotateRight`).
    ///
    /// # Example
    /// ```text
    /// let q = MlQuery::Eq(MultilinearPoint(vec![x0, x1, x2]));
    /// assert_eq!(q.log_b(), 3);
    /// ```
    ///
    /// For `EqRotateRight(z, r)`, the rotation does not affect `log_b`.
    ///
    /// # Returns
    /// The number of variables (length of the `z` vector).
    #[must_use]
    pub fn log_b(&self) -> usize {
        match self {
            Self::Eq(z) | Self::EqRotateRight(z, _) => z.len(),
        }
    }

    /// Evaluate the multilinear equality polynomial corresponding to the query.
    ///
    /// This returns the evaluation of the multilinear equality polynomial
    /// over all inputs `x ∈ {0,1}^n` with respect to a fixed evaluation point `z`,
    /// scaled by a scalar factor `α`.
    ///
    /// The result is a vector of size `2^n` representing:
    ///
    /// \begin{equation}
    /// \text{mle}[x] = α ⋅ \mathrm{eq}(x, z)
    /// \end{equation}
    ///
    /// For `EqRotateRight(z, r)`, the output is:
    ///
    /// \begin{equation}
    /// \text{mle}[x] = α ⋅ \mathrm{eq}(x, z) \quad \text{then rotate right by } r
    /// \end{equation}
    ///
    /// # Arguments
    /// - `scalar`: A scaling factor α to apply to the result.
    ///
    /// # Returns
    /// A vector of `2^n` elements (for `n = z.len()`), representing
    /// evaluations of the scaled equality polynomial.
    pub fn to_mle(&self, scalar: F) -> Vec<F> {
        match self {
            // Standard case: evaluate at z
            Self::Eq(z) => {
                // Allocate output buffer of size 2^n
                let mut mle = F::zero_vec(1 << z.len());

                // Fill with α ⋅ eq(x, z) for all x ∈ {0,1}^n
                eval_eq::<_, _, false>(z, &mut mle, scalar);

                mle
            }

            // Rotated case: evaluate at z, then rotate output
            Self::EqRotateRight(z, mid) => {
                // Allocate output buffer of size 2^n
                let mut mle = F::zero_vec(1 << z.len());

                // Compute α ⋅ eq(x, z)
                eval_eq::<_, _, false>(z, &mut mle, scalar);

                // Apply circular rotation to output buffer
                mle.rotate_right(*mid);

                mle
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    use super::*;

    type F = BabyBear;

    /// Naive equality polynomial evaluation.
    ///
    /// For a given point `z ∈ 𝔽ⁿ` and scalar `α`, compute:
    ///
    /// \[
    /// [α ⋅ eq(x, z)]_{x ∈ {0,1}ⁿ}
    /// \]
    ///
    /// in lexicographic order of `x`.
    fn naive_eq(z: &[F], scalar: F) -> Vec<F> {
        let n = z.len();
        let mut out = vec![F::ZERO; 1 << n];

        for (i, o) in out.iter_mut().enumerate().take(1 << n) {
            // Decompose index i into binary vector x
            let x_bits: Vec<u8> = (0..n).map(|j| ((i >> (n - 1 - j)) & 1) as u8).collect();

            // Compute eq(x, z) = ∏ (x_i z_i + (1 - x_i)(1 - z_i))
            let mut acc = F::ONE;
            for (&bit, &zi) in x_bits.iter().zip(z.iter()) {
                let xi = F::from_u8(bit);
                acc *= xi * zi + (F::ONE - xi) * (F::ONE - zi);
            }

            *o = scalar * acc;
        }

        out
    }

    #[test]
    fn test_eq_query_mle_length_2() {
        // z = [z0, z1] = [1, 0]
        let z = vec![F::ONE, F::ZERO];
        let scalar = F::from_u8(3);

        let mle = MlQuery::Eq(MultilinearPoint(z)).to_mle(scalar);

        // Manually compute:
        // eq((0,0), z) = (1 - z0)(1 - z1) = 0
        // eq((0,1), z) = (1 - z0)(z1)     = 0
        // eq((1,0), z) = z0(1 - z1)       = 1
        // eq((1,1), z) = z0(z1)           = 0
        //
        // So the evaluations are:
        // [0, 0, 1, 0] * 3 = [0, 0, 3, 0]
        let expected = vec![F::ZERO, F::ZERO, F::from_u8(3), F::ZERO];

        assert_eq!(mle, expected);
    }

    #[test]
    fn test_eq_query_mle_length_3() {
        // z = [1, 1, 0], α = 2
        let z = vec![F::ONE, F::ONE, F::ZERO];
        let scalar = F::from_u8(2);

        let mle = MlQuery::Eq(MultilinearPoint(z.clone())).to_mle(scalar);
        let expected = naive_eq(&z, scalar);

        assert_eq!(mle, expected);
    }

    #[test]
    fn test_eq_rotate_right_3_vars() {
        // Setup:
        // z = [2, 3, 4], scalar = 5
        // mid = 1 → output should be rotated once to the right after evaluating at z
        let z = vec![F::from_u8(2), F::from_u8(3), F::from_u8(4)];
        let scalar = F::from_u8(5);
        let mid = 1;

        // Compute using the actual implementation
        let mle = MlQuery::EqRotateRight(MultilinearPoint(z.clone()), mid).to_mle(scalar);

        // Expected behavior:
        // - First compute the equality polynomial: naive_eq(z, scalar)
        // - Then rotate that output vector to the right by `mid`
        let mut expected = naive_eq(&z, scalar);
        expected.rotate_right(mid);

        assert_eq!(mle, expected);
    }

    #[test]
    fn test_eq_query_mle_empty() {
        // Empty query: n = 0 → eq() = α for x = ()
        let z = vec![];
        let scalar = F::from_u8(7);

        let mle = MlQuery::Eq(MultilinearPoint(z)).to_mle(scalar);
        let expected = vec![scalar];

        assert_eq!(mle, expected);
    }

    #[test]
    fn test_eq_rotate_right_4_vars_rotate_2() {
        // Setup:
        // z = [1, 2, 3, 4], scalar = 6
        // mid = 2 → rotate output by 2 positions after evaluating at z
        let z = vec![F::from_u8(1), F::from_u8(2), F::from_u8(3), F::from_u8(4)];
        let scalar = F::from_u8(6);
        let mid = 2;

        // Actual mle computation with rotation
        let mle = MlQuery::EqRotateRight(MultilinearPoint(z.clone()), mid).to_mle(scalar);

        // Expected behavior:
        // - Compute eq(x, z) for all x ∈ {0,1}⁴ → output has 2⁴ = 16 values
        // - Then rotate the result right by 2 positions
        let mut expected = naive_eq(&z, scalar);
        expected.rotate_right(mid);

        assert_eq!(mle, expected);
    }

    #[test]
    fn test_log_b_all_supported_lengths() {
        for log_b in 0..=4 {
            let point: MultilinearPoint<F> = MultilinearPoint(vec![F::from_u32(1); log_b]);

            // Test regular Eq query
            let query_eq = MlQuery::Eq(point.clone());
            assert_eq!(
                query_eq.log_b(),
                log_b,
                "Eq query with length {log_b} should return log_b = {log_b}"
            );

            // Test rotated query
            let query_rot = MlQuery::EqRotateRight(point.clone(), 1);
            assert_eq!(
                query_rot.log_b(),
                log_b,
                "EqRotateRight query with length {log_b} should return log_b = {log_b}"
            );
        }
    }
}
