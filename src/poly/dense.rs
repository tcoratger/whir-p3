use std::{
    collections::HashSet,
    ops::{Add, AddAssign, Mul, MulAssign},
};

use p3_field::{ExtensionField, Field};
use rand::distr::{Distribution, StandardUniform};

/// A univariate polynomial represented in coefficient form.
///
/// The coefficient of `x^i` is stored at index `i`.
///
/// Designed for verifier use: avoids parallelism by enforcing sequential Horner evaluation.
/// The verifier should be run on a cheap device.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct WhirDensePolynomial<F: Field> {
    /// The coefficient of `x^i` is stored at location `i` in `self.coeffs`.
    pub coeffs: Vec<F>,
}

impl<F: Field> WhirDensePolynomial<F> {
    /// Constructs a new polynomial from a list of coefficients.
    #[must_use]
    pub fn from_coefficients_slice(coeffs: &[F]) -> Self {
        Self::from_coefficients_vec(coeffs.to_vec())
    }

    /// Constructs a new polynomial from a list of coefficients.
    #[must_use]
    pub const fn from_coefficients_vec(coeffs: Vec<F>) -> Self {
        Self { coeffs }
    }

    /// Removes trailing zero coefficients from the polynomial's coefficient vector.
    fn truncate_leading_zeros(&mut self) {
        while self.coeffs.last().is_some_and(Field::is_zero) {
            self.coeffs.pop();
        }
    }

    /// Checks if the given polynomial is zero.
    fn is_zero(&self) -> bool {
        self.coeffs.is_empty() || self.coeffs.iter().all(Field::is_zero)
    }

    /// Evaluates `self` at the given `point` in `Self::Point`.
    pub fn evaluate<EF: ExtensionField<F>>(&self, point: EF) -> EF {
        if self.is_zero() {
            return EF::ZERO;
        } else if point.is_zero() {
            return EF::from(self.coeffs[0]);
        }
        self.horner_evaluate(point)
    }

    // Horner's method for polynomial evaluation
    fn horner_evaluate<EF: ExtensionField<F>>(&self, point: EF) -> EF {
        self.coeffs
            .iter()
            .rfold(EF::ZERO, move |result, coeff| result * point + *coeff)
    }

    /// Constructs a new polynomial from a list of coefficients
    ///  which are random elements mapped by closed interval [0, degree]
    pub fn random<R: rand::Rng>(rng: &mut R, degree: usize) -> Self
    where
        StandardUniform: Distribution<F>,
    {
        Self::from_coefficients_vec((0..=degree).map(|_| rng.random()).collect())
    }

    /// Constructs the unique interpolating polynomial `P(x)` such that:
    ///
    /// \begin{equation}
    /// P(x_i) = y_i \quad \text{for each } (x_i, y_i) \text{ in the input set}
    /// \end{equation}
    ///
    /// The result is a dense univariate polynomial of degree at most `n - 1`, where `n` is the number of input points.
    ///
    /// # Parameters
    ///
    /// - `values`: A slice of `(x_i, y_i)` pairs, where each `x_i` is a field element from `S`,
    ///   and each `y_i` is an extension field element from `F`.
    ///
    /// # Returns
    ///
    /// - `Some(P)`: The interpolating polynomial if all `x_i` are distinct.
    /// - `None`: If any duplicate `x_i` values exist (even with equal `y_i`).
    pub fn lagrange_interpolation<S>(values: &[(S, F)]) -> Option<Self>
    where
        S: Field,
        F: ExtensionField<S>,
    {
        // The number of (x, y) pairs to interpolate.
        let n = values.len();

        // Special case: no points => return the zero polynomial.
        if n == 0 {
            return Some(Self::default());
        }

        // Check for duplicate x-coordinates
        //
        // We use a HashSet to track x_i values. If any duplicates exist, we cannot interpolate.
        let mut unique_x = HashSet::with_capacity(n);
        if !values.iter().all(|(x, _)| unique_x.insert(*x)) {
            return None; // Found a duplicate x_i
        }

        // Initialize result and basis polynomials

        // The result polynomial P(x) starts at zero and is updated iteratively.
        let mut result_poly = Self::default();

        // The basis polynomial B(x) starts at 1.
        // After i steps, B(x) = (x - x_0)(x - x_1)...(x - x_{i-1})
        let mut basis_poly = Self::from_coefficients_vec(vec![F::ONE]);

        // Newton-style interpolation loop
        for (x_i, y_i) in values.iter().take(n) {
            // Promote x_i to the extension field for evaluation
            let x_i_ext = F::from(*x_i);

            // Compute current prediction: P(x_i)
            let current_y = result_poly.evaluate(x_i_ext);

            // Compute the discrepancy: how far off our polynomial is
            let delta = *y_i - current_y;

            // Compute B(x_i), the value of the basis at x_i
            let basis_eval = basis_poly.evaluate(x_i_ext);

            // The scalar coefficient c_i is the correction needed:
            // c_i = (y_i - P(x_i)) / B(x_i)
            let c_i = delta
                * basis_eval
                    .try_inverse()
                    .expect("x_i was checked to be unique");

            // Form term = c_i · B(x)

            // Multiply all coefficients of B(x) by c_i
            let mut term_coeffs = basis_poly.coeffs.clone();
            for coeff in &mut term_coeffs {
                *coeff *= c_i;
            }

            // Convert the coefficient vector into a polynomial
            let term = Self::from_coefficients_vec(term_coeffs);

            // Add the term to the result: P(x) := P(x) + c_i · B(x)
            result_poly += &term;

            // Update B(x) := B(x) · (x - x_i)

            // Monomial: (x - x_i) = -x_i + x = [-x_i, 1]
            let monomial = Self::from_coefficients_slice(&[-x_i_ext, F::ONE]);

            // Multiply current basis polynomial by the monomial
            basis_poly = &basis_poly * &monomial;
        }

        Some(result_poly)
    }
}

impl<F: Field> Add for &WhirDensePolynomial<F> {
    type Output = WhirDensePolynomial<F>;

    /// Adds two dense polynomials and returns the resulting polynomial.
    ///
    /// This function computes the sum of `self` and `other` by adding their
    /// coefficients term by term. If the polynomials have different lengths,
    /// the coefficients of the longer polynomial that do not have a corresponding
    /// term in the shorter polynomial are left unchanged in the result.
    ///
    /// # Arguments
    ///
    /// * `other` - The polynomial to add to `self`.
    ///
    /// # Returns
    ///
    /// A new `WhirDensePolynomial<F>` representing the sum of the two input polynomials.
    fn add(self, other: Self) -> WhirDensePolynomial<F> {
        let (big, small) = if self.coeffs.len() >= other.coeffs.len() {
            (self, other)
        } else {
            (other, self)
        };
        let mut sum = big.coeffs.clone();
        for (i, coeff) in small.coeffs.iter().enumerate() {
            sum[i] += *coeff;
        }
        WhirDensePolynomial::from_coefficients_vec(sum)
    }
}

impl<F: Field> AddAssign<&Self> for WhirDensePolynomial<F> {
    fn add_assign(&mut self, other: &Self) {
        *self = &*self + other;
    }
}

impl<F: Field> Mul for &WhirDensePolynomial<F> {
    type Output = WhirDensePolynomial<F>;

    /// Multiplies two dense polynomials and returns the resulting polynomial.
    ///
    /// This function computes the product of `self` and `other` using the standard
    /// schoolbook (naive) polynomial multiplication algorithm. If either polynomial
    /// is zero, the result is the zero polynomial. The resulting polynomial's
    /// coefficients are computed by summing the products of all pairs of coefficients
    /// whose degrees add up to the same value.
    ///
    /// # Arguments
    ///
    /// * `other` - The polynomial to multiply with `self`.
    ///
    /// # Returns
    ///
    /// A new `WhirDensePolynomial<F>` representing the product of the two input polynomials.
    fn mul(self, other: Self) -> WhirDensePolynomial<F> {
        if self.is_zero() || other.is_zero() {
            return WhirDensePolynomial::default();
        }
        let mut prod = vec![F::ZERO; self.coeffs.len() + other.coeffs.len() - 1];
        for i in 0..self.coeffs.len() {
            for j in 0..other.coeffs.len() {
                prod[i + j] += self.coeffs[i] * other.coeffs[j];
            }
        }
        WhirDensePolynomial::from_coefficients_vec(prod)
    }
}

impl<F: Field> MulAssign<&Self> for WhirDensePolynomial<F> {
    fn mul_assign(&mut self, other: &Self) {
        *self = &*self * other;
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};
    use proptest::prelude::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use super::*;

    type F = BabyBear;
    type EF4 = BinomialExtensionField<F, 4>;

    #[test]
    fn test_zero_polynomial() {
        // A zero polynomial has no coefficients
        let poly = WhirDensePolynomial::<F>::from_coefficients_vec(vec![]);
        assert!(poly.is_zero());
        assert_eq!(poly.evaluate(F::from_u64(42)), F::ZERO);
    }

    #[test]
    fn test_constant_polynomial() {
        // Polynomial: f(x) = 7
        let c0 = F::from_u64(7);
        let poly = WhirDensePolynomial::from_coefficients_vec(vec![c0]);

        // f(0)
        assert_eq!(poly.evaluate(F::ZERO), c0);
        // f(1)
        assert_eq!(poly.evaluate(F::ONE), c0);
        // f(42)
        assert_eq!(poly.evaluate(F::from_u64(42)), c0);
    }

    #[test]
    fn test_linear_polynomial() {
        // Polynomial: f(x) = 3 + 4x
        let c0 = F::from_u64(3);
        let c1 = F::from_u64(4);
        let poly = WhirDensePolynomial::from_coefficients_vec(vec![c0, c1]);

        // f(0)
        assert_eq!(poly.evaluate(F::ZERO), c0);
        // f(1)
        assert_eq!(poly.evaluate(F::ONE), c0 + c1 * F::ONE);
        // f(2)
        assert_eq!(poly.evaluate(F::from_u64(2)), c0 + c1 * F::from_u64(2));
    }

    #[test]
    fn test_quadratic_polynomial() {
        // Polynomial: f(x) = 2 + 0x + 5x²
        let c0 = F::from_u64(2);
        let c1 = F::from_u64(0);
        let c2 = F::from_u64(5);
        let poly = WhirDensePolynomial::from_coefficients_vec(vec![c0, c1, c2]);

        // f(0)
        assert_eq!(poly.evaluate(F::ZERO), c0);
        // f(1)
        assert_eq!(poly.evaluate(F::ONE), c0 + c2);
        // f(2)
        assert_eq!(poly.evaluate(F::from_u64(2)), c0 + c2 * F::from_u64(4));
    }

    #[test]
    fn test_cubic_polynomial() {
        // Polynomial: f(x) = 1 + 2x + 3x² + 4x³
        let c0 = F::from_u64(1);
        let c1 = F::from_u64(2);
        let c2 = F::from_u64(3);
        let c3 = F::from_u64(4);
        let poly = WhirDensePolynomial::from_coefficients_vec(vec![c0, c1, c2, c3]);

        // f(0)
        assert_eq!(poly.evaluate(F::ZERO), c0);
        // f(1)
        assert_eq!(poly.evaluate(F::ONE), c0 + c1 + c2 + c3);

        // f(2)
        assert_eq!(
            poly.evaluate(F::from_u64(2)),
            c0 + c1 * F::from_u64(2) + c2 * F::from_u64(4) + c3 * F::from_u64(8)
        );
    }

    #[test]
    fn test_is_zero_various_cases() {
        let zero_poly = WhirDensePolynomial::<F>::from_coefficients_vec(vec![]);
        assert!(zero_poly.is_zero());

        let zero_poly_all_zeros = WhirDensePolynomial::<F>::from_coefficients_vec(vec![F::ZERO; 5]);
        assert!(zero_poly_all_zeros.is_zero());

        let non_zero_poly = WhirDensePolynomial::<F>::from_coefficients_vec(vec![F::ONE]);
        assert!(!non_zero_poly.is_zero());
    }

    #[test]
    fn test_mul() {
        let rng = &mut StdRng::seed_from_u64(0);
        let pol1 = WhirDensePolynomial::<F>::random(rng, 5);
        let pol2 = WhirDensePolynomial::<F>::random(rng, 7);
        let point: EF4 = rng.random();
        assert_eq!(
            (&pol1 * &pol2).evaluate(point),
            pol1.evaluate(point) * pol2.evaluate(point)
        );

        let zero_poly = WhirDensePolynomial::<F>::from_coefficients_vec(vec![]);
        let non_zero_poly = WhirDensePolynomial::<F>::from_coefficients_vec(vec![F::ONE, F::ZERO]);
        assert!(zero_poly.is_zero());
        assert_eq!(&non_zero_poly * &zero_poly, zero_poly.clone());
        assert_eq!(&zero_poly * &non_zero_poly, zero_poly);
        assert_eq!(&zero_poly * &zero_poly, zero_poly.clone());
    }

    #[test]
    fn test_add() {
        let rng = &mut StdRng::seed_from_u64(0);
        let pol1 = WhirDensePolynomial::<F>::random(rng, 5);
        let pol2 = WhirDensePolynomial::<F>::random(rng, 7);
        let point: EF4 = rng.random();
        assert_eq!(
            (&pol1 + &pol2).evaluate(point),
            pol1.evaluate(point) + pol2.evaluate(point)
        );

        let zero_poly = WhirDensePolynomial::<F>::from_coefficients_vec(vec![]);
        let non_zero_poly = WhirDensePolynomial::<F>::from_coefficients_vec(vec![F::ONE, F::ZERO]);
        assert!(zero_poly.is_zero());
        assert_eq!(&non_zero_poly + &zero_poly, non_zero_poly);
        assert_eq!(&zero_poly + &non_zero_poly, non_zero_poly);
        assert_eq!(&zero_poly + &zero_poly, zero_poly);
    }

    #[test]
    fn test_lagrange_interpolation() {
        let mut rng = StdRng::seed_from_u64(0);
        let degree = 5;
        let pol = WhirDensePolynomial::random(&mut rng, 5);
        let points = (0..=degree)
            .map(|_| {
                let point = rng.random::<F>();
                (point, pol.evaluate(point))
            })
            .collect::<Vec<_>>();
        let interpolated = WhirDensePolynomial::lagrange_interpolation(&points).unwrap();
        assert_eq!(pol, interpolated);

        assert!(
            WhirDensePolynomial::<F>::lagrange_interpolation(&[])
                .unwrap()
                .is_zero()
        );
    }

    #[test]
    fn test_lagrange_interpolation_duplicated_point() {
        let points = vec![
            (F::from_u64(1), F::from_u64(2)),
            (F::from_u64(1), F::from_u64(3)), // Duplicate x, different y
            (F::from_u64(2), F::from_u64(4)),
        ];
        assert!(WhirDensePolynomial::<F>::lagrange_interpolation(&points).is_none());

        let points = vec![
            (F::from_u64(1), F::from_u64(2)),
            (F::from_u64(2), F::from_u64(3)),
            (F::from_u64(2), F::from_u64(3)), // Duplicate x, same y
        ];
        assert!(WhirDensePolynomial::<F>::lagrange_interpolation(&points).is_none());
    }

    proptest! {
        #[test]
        fn proptest_lagrange_interpolation(degree in 1usize..10) {
            // Initialize a deterministic RNG with a fixed seed for reproducibility.
            let mut rng = StdRng::seed_from_u64(0);

            // STEP 1: Generate a random polynomial `pol` of the selected degree.
            //
            // This produces random coefficients c₀, c₁, ..., c_degree, forming:
            //     pol(x) = c₀ + c₁·x + c₂·x² + ... + c_degree·x^degree
            let pol = WhirDensePolynomial::<F>::random(&mut rng, degree);

            // STEP 2: Prepare a set of (x, y) pairs such that:
            //     - x values are unique (no duplicates)
            //     - y = pol(x) is the correct evaluation at that point.
            //
            // We will collect `degree + 1` points, enough to fully determine the polynomial.
            let mut points = vec![];
            let mut used_x = std::collections::HashSet::new();

            while points.len() <= degree {
                // Randomly generate an x value.
                let x: F = rng.random();

                // Only keep it if we haven't used this x before.
                if used_x.insert(x) {
                    // Evaluate the polynomial at x.
                    let y = pol.evaluate(x);

                    // Store the (x, y) pair.
                    points.push((x, y));
                }
            }

            // STEP 3: Run Lagrange interpolation on the collected points.
            //
            // This reconstructs a new polynomial `interpolated` that satisfies:
            //     interpolated(x_i) = y_i  for all provided (x_i, y_i).
            let interpolated = WhirDensePolynomial::lagrange_interpolation(&points).unwrap();

            // STEP 4: Assert that the interpolated polynomial matches the original.
            //
            // This checks that:
            //     interpolated.coeffs == pol.coeffs
            //
            // If the Lagrange interpolation is correct, the reconstructed polynomial
            // should be exactly the same as the randomly generated one.
            prop_assert_eq!(interpolated, pol);
        }
    }

    /// Builds a strategy that yields random dense polynomials over `F`.
    ///
    /// Each polynomial’s degree is chosen uniformly from `max_deg` (exclusive),
    /// then its coefficients are sampled as `u64` values and lifted into `F`
    /// via `F::from_u64`.
    ///
    /// # Parameters
    ///
    /// - `max_deg`: an exclusive upper bound on the degree.
    ///   Degrees will be drawn from `0..max_deg`; a sampled degree `d`
    ///   yields a polynomial of degree exactly `d` (with `d+1` coefficients).
    ///
    /// # Returns
    ///
    /// A `proptest::Strategy` producing `WhirDensePolynomial<F>` instances.
    fn any_polynomial(
        max_deg: std::ops::Range<usize>,
    ) -> impl Strategy<Value = WhirDensePolynomial<F>> {
        max_deg
            // 1. Pick a random degree `deg` in 0..max_deg
            .prop_flat_map(move |deg| {
                // 2. Generate exactly `deg + 1` raw `u64` values
                prop::collection::vec(any::<u64>(), deg + 1)
            })
            // 3. Convert each `u64` into `F` and build the polynomial
            .prop_map(|coeffs_u64| {
                let coeffs_f = coeffs_u64.into_iter().map(F::from_u64).collect();
                WhirDensePolynomial::from_coefficients_vec(coeffs_f)
            })
    }

    proptest! {
        /// (a + b) + c == a + (b + c)
        #[test]
        fn prop_add_associative(
            // three random polys of small degree for speed
            a in any_polynomial(0..5),
            b in any_polynomial(0..5),
            c in any_polynomial(0..5),
        ) {
            let lhs = &(&a + &b) + &c;
            let rhs = &a + &(&b + &c);
            prop_assert_eq!(lhs, rhs);
        }

        /// a + b == b + a
        #[test]
        fn prop_add_commutative(
            a in any_polynomial(0..5),
            b in any_polynomial(0..5),
        ) {
            prop_assert_eq!(&a + &b, &b + &a);
        }

        /// 0 is the additive identity: a + 0 == a
        #[test]
        #[allow(clippy::redundant_clone)]
        fn prop_add_identity(
            a in any_polynomial(0..5),
        ) {
            let zero = WhirDensePolynomial::<F>::default();
            prop_assert_eq!(&a + &zero, a.clone());
            prop_assert_eq!(&zero + &a, a);
        }

        /// (a * b) * c == a * (b * c)
        #[test]
        fn prop_mul_associative(
            a in any_polynomial(0..5),
            b in any_polynomial(0..5),
            c in any_polynomial(0..5),
        ) {
            let lhs = &(&a * &b) * &c;
            let rhs = &a * &(&b * &c);
            prop_assert_eq!(lhs, rhs);
        }

        /// a * b == b * a
        #[test]
        fn prop_mul_commutative(
            a in any_polynomial(0..5),
            b in any_polynomial(0..5),
        ) {
            prop_assert_eq!(&a * &b, &b * &a);
        }

        /// 1 is the multiplicative identity: a * 1 == a
        #[test]
        #[allow(clippy::redundant_clone)]
        fn prop_mul_identity(
            a in any_polynomial(0..5),
        ) {
            // Build the constant-1 polynomial
            let one = WhirDensePolynomial::from_coefficients_vec(vec![F::ONE]);
            prop_assert_eq!(&a * &one, a.clone());
            prop_assert_eq!(&one * &a, a);
        }

        /// Distributivity: a * (b + c) == a*b + a*c
        #[test]
        fn prop_distributive(
            a in any_polynomial(0..5),
            b in any_polynomial(0..5),
            c in any_polynomial(0..5),
        ) {
            let lhs = &a * &(&b + &c);
            let rhs = &(&a * &b) + &(&a * &c);
            prop_assert_eq!(lhs, rhs);
        }
    }
}
