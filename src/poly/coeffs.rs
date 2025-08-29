use std::ops::Deref;

use p3_field::{ExtensionField, Field, PackedFieldExtension, PackedValue};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use tracing::instrument;

use super::{
    dense::WhirDensePolynomial, evals::EvaluationsList, multilinear::MultilinearPoint,
    wavelet::Radix2WaveletKernel,
};

/// Represents a multilinear polynomial `f` in `n` variables, stored by its coefficients.
///
/// The inner vector stores the coefficients in lexicographic order of monomials. The number of
/// variables `n` is inferred from the length of this vector, where `self.len() = 2^n`.
///
/// The coefficient for the monomial `X_0^{b_{n-1}} * ... * X_{n-1}^{b_0}` is stored at index `j`
/// where the binary representation of `j` is `(b_{n-1}, ..., b_0)`.
///
/// ### Example (n = 3, variables X₀, X₁, X₂)
///
/// - `coeffs[0]` (binary 000) → Constant term (1)
/// - `coeffs[1]` (binary 001) → Coefficient of `X₂`
/// - `coeffs[2]` (binary 010) → Coefficient of `X₁`
/// - `coeffs[4]` (binary 100) → Coefficient of `X₀`
#[derive(Default, Debug, Clone, Eq, PartialEq)]
pub struct CoefficientList<F>(Vec<F>);

impl<F> CoefficientList<F>
where
    F: Field,
{
    /// Interprets self as a univariate polynomial (with coefficients of X^i in order of ascending
    /// i) and evaluates it at each point in `points`. We return the vector of evaluations.
    ///
    /// NOTE: For the `usual` mapping between univariate and multilinear polynomials, the
    /// coefficient ordering is such that for a single point x, we have (extending notation to a
    /// single point) self.evaluate_at_univariate(x) == self.evaluate([x^(2^n), x^(2^{n-1}),
    /// ..., x^2, x])
    pub fn evaluate_at_univariate(&self, points: &[F]) -> Vec<F> {
        // WhirDensePolynomial::from_coefficients_slice converts to a dense univariate polynomial.
        // The coefficient order is "coefficient of 1 first".
        let univariate = WhirDensePolynomial::from_coefficients_slice(self);
        points
            .iter()
            .map(|point| univariate.evaluate(*point))
            .collect()
    }

    /// Folds the polynomial along high-indexed variables, reducing its dimensionality.
    ///
    /// Given a multilinear polynomial `f(X₀, ..., X_{n-1})`, this partially evaluates it at
    /// `folding_randomness`, returning a new polynomial in fewer variables:
    ///
    /// ```ignore
    /// f(X₀, ..., X_{m-1}, r₀, r₁, ..., r_k) → g(X₀, ..., X_{m-1})
    /// ```
    /// where `r₀, ..., r_k` are values from `folding_randomness`.
    ///
    /// - The number of variables decreases: `m = n - k`
    /// - Uses multivariate evaluation over chunks of coefficients.
    #[must_use]
    pub fn fold<EF>(&self, folding_randomness: &MultilinearPoint<EF>) -> CoefficientList<EF>
    where
        EF: ExtensionField<F>,
    {
        let folding_factor = folding_randomness.num_variables();
        CoefficientList(
            self.par_chunks_exact(1 << folding_factor)
                .map(|coeffs| eval_multivariate(coeffs, folding_randomness))
                .collect(),
        )
    }

    /// Evaluate self at `point`, where `point` is from a field extension extending the field over
    /// which the polynomial `self` is defined.
    ///
    /// Note that we only support the case where F is a prime field.
    #[instrument(skip_all, fields(size = point.num_variables()), level = "debug")]
    pub fn evaluate<EF: ExtensionField<F>>(&self, point: &MultilinearPoint<EF>) -> EF {
        assert_eq!(self.num_variables(), point.num_variables());
        eval_extension_par(self, point)
    }
}

impl<F> Deref for CoefficientList<F> {
    type Target = [F];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Multivariate Horner’s method for evaluating a polynomial at a general point.
///
/// Given `coeffs` (coefficients in lexicographic order) and `point = (x₀, x₁, ..., x_{n-1})`,
/// this recursively evaluates:
///
/// ```ignore
/// P(x₀, ..., x_{n-1}) = P_0(x₁, ..., x_{n-1}) + x₀ * P_1(x₁, ..., x_{n-1})
/// ```
///
/// where `P_0` and `P_1` are the even-indexed and odd-indexed coefficient subsets.
fn eval_multivariate<F, EF>(coeffs: &[F], point: &MultilinearPoint<EF>) -> EF
where
    F: Field,
    EF: ExtensionField<F>,
{
    debug_assert_eq!(coeffs.len(), 1 << point.num_variables());
    match point.as_slice() {
        [] => coeffs[0].into(),
        [x] => *x * coeffs[1] + coeffs[0],
        [x0, x1] => {
            let b0 = *x1 * coeffs[1] + coeffs[0];
            let b1 = *x1 * coeffs[3] + coeffs[2];
            b0 + b1 * *x0
        }
        [x0, x1, x2] => {
            let b00 = *x2 * coeffs[1] + coeffs[0];
            let b01 = *x2 * coeffs[3] + coeffs[2];
            let b10 = *x2 * coeffs[5] + coeffs[4];
            let b11 = *x2 * coeffs[7] + coeffs[6];
            let b0 = b00 + b01 * *x1;
            let b1 = b10 + b11 * *x1;
            b0 + b1 * *x0
        }
        [x0, x1, x2, x3] => {
            let b000 = *x3 * coeffs[1] + coeffs[0];
            let b001 = *x3 * coeffs[3] + coeffs[2];
            let b010 = *x3 * coeffs[5] + coeffs[4];
            let b011 = *x3 * coeffs[7] + coeffs[6];
            let b100 = *x3 * coeffs[9] + coeffs[8];
            let b101 = *x3 * coeffs[11] + coeffs[10];
            let b110 = *x3 * coeffs[13] + coeffs[12];
            let b111 = *x3 * coeffs[15] + coeffs[14];
            let b00 = b000 + b001 * *x2;
            let b01 = b010 + b011 * *x2;
            let b10 = b100 + b101 * *x2;
            let b11 = b110 + b111 * *x2;
            let b0 = b00 + b01 * *x1;
            let b1 = b10 + b11 * *x1;
            b0 + b1 * *x0
        }
        [x, tail @ ..] => {
            let sub_point = MultilinearPoint::new(tail.to_vec());
            let (b0t, b1t) = coeffs.split_at(coeffs.len() / 2);
            let (b0t, b1t) = {
                let work_size: usize = (1 << 15) / size_of::<F>();
                if coeffs.len() > work_size {
                    join(
                        || eval_multivariate(b0t, &sub_point),
                        || eval_multivariate(b1t, &sub_point),
                    )
                } else {
                    (
                        eval_multivariate(b0t, &sub_point),
                        eval_multivariate(b1t, &sub_point),
                    )
                }
            };
            b0t + b1t * *x
        }
    }
}

impl<F> CoefficientList<F> {
    /// Creates a `CoefficientList` from a vector of coefficients.
    ///
    /// Ensures that:
    /// - The length is a power of two (`2^n` for some `n`).
    /// - Computes `num_variables` as `log₂(coeffs.len())`.
    #[must_use]
    pub fn new(coeffs: Vec<F>) -> Self {
        let len = coeffs.len();
        assert!(len.is_power_of_two());
        Self(coeffs)
    }

    /// Returns the number of variables (`n`).
    ///
    /// Since `coeffs.len() = 2^n`, this returns `log₂(coeffs.len())`.
    #[must_use]
    pub const fn num_variables(&self) -> usize {
        // Safety: The length is guaranteed to be a power of two.
        self.0.len().ilog2() as usize
    }

    /// Returns the total number of coefficients (`2^n`).
    #[must_use]
    pub const fn num_coeffs(&self) -> usize {
        self.0.len()
    }

    /// Convert from a list of multilinear coefficients to a list of
    /// evaluations over the hypercube.
    pub fn to_evaluations<B: Field>(self) -> EvaluationsList<F>
    where
        F: ExtensionField<B>,
    {
        let kernel = Radix2WaveletKernel::<B>::default();
        let evals = kernel.wavelet_transform_algebra(self.to_vec());
        EvaluationsList::new(evals)
    }
}

/// Evaluates a multilinear polynomial at an extension field point making use of parallelism.
///
/// Given `coeffs` in lexicographic order, this computes:
/// ```ignore
/// eval_poly(X_0, ..., X_n) = sum(coeffs[i] * product(X_j for j in S(i)))
/// ```
/// where `S(i)` is the set of variables active in term `i` (based on its binary representation).
///
/// - Splits into `num_threads` many subtasks with the `j`'th subtask computing:
/// ```ignore
/// eval_poly_{L, j}(X_L, ..., X_n) = sum(coeffs[shift * j + i] * product(X_j for j in S(i)))
/// ```
/// where `shift = 2^n / num_threads` and `L = log_num_threads`.
/// - Then the results are combined using:
/// ```ignore
/// eval_poly = sum(eval_poly_{L, j} * product(X_j for j in S(i)))
/// ```
/// where the product is now only over the first `L` variables.
#[inline]
fn eval_extension_par<F, E>(coeff: &[F], point: &MultilinearPoint<E>) -> E
where
    F: Field,
    E: ExtensionField<F>,
{
    let num_threads = current_num_threads().next_power_of_two();
    let log_num_threads = log2_strict_usize(num_threads);

    debug_assert_eq!(coeff.len(), 1 << point.num_variables());

    let size = coeff.len();
    // While we could run the following code for any size > LOG_NUM_THREADS, there isn't
    // much point. In particular, we would lose some of the benefits of packing tricks in
    // eval_extension_packed. Instead we set a (slightly arbitrary) threshold of 15.
    if size > (1 << 15) {
        let chunk_size = size / num_threads;
        let (head, tail) = point.as_slice().split_at(log_num_threads);
        let head_point = MultilinearPoint::new(head.to_vec());
        let tail_point = MultilinearPoint::new(tail.to_vec());
        let partial_sum = coeff
            .par_chunks_exact(chunk_size)
            .map(|chunk| eval_extension_packed(chunk, &tail_point))
            .collect::<Vec<_>>();
        return eval_extension_packed(&partial_sum, &head_point);
    }

    eval_extension_packed(coeff, point)
}

/// Recursively evaluates a multilinear polynomial at an extension field point.
///
/// Given `coeffs` in lexicographic order, this computes:
/// ```ignore
/// eval_poly(X_0, ..., X_n) = sum(coeffs[i] * product(X_j for j in S(i)))
/// ```
/// where `S(i)` is the set of variables active in term `i` (based on its binary representation).
///
/// - Small cases are passed to `eval_extension_unpacked`.
/// - For larger cases, we pack extension field elements into `PackedFieldExtension` elements and
///   do as many rounds as possible in the packed case which is reasonably faster.
///   Eventually we unpack and pass to `eval_extension_unpacked`
#[inline]
fn eval_extension_packed<F, E>(coeff: &[F], point: &MultilinearPoint<E>) -> E
where
    F: Field,
    E: ExtensionField<F>,
{
    debug_assert_eq!(coeff.len(), 1 << point.num_variables());
    let log_packing_width = log2_strict_usize(F::Packing::WIDTH);

    // There is some overhead when packing extension field elements so we only want to do it
    // when we have a large number of coefficients. I chose 2 here basically arbitrarily, it might
    // be worth benchmarking this.
    if point.num_variables() <= (log_packing_width + 2) {
        eval_extension_unpacked(coeff, point)
    } else {
        // If the size of eval is > log_packing_width + 2, it makes sense to start using packings.
        // As coeffs lie in F, we do the first round manually as it's a little cheaper.
        let packed_coeff = F::Packing::pack_slice(coeff);
        let packed_eval: E::ExtensionPacking = point[0].into();
        let (lo, hi) = packed_coeff.split_at(packed_coeff.len() / 2);
        let mut buffer = lo
            .iter()
            .zip(hi.iter())
            .map(|(l, h)| packed_eval * *h + *l)
            .collect::<Vec<_>>();

        for &ef_eval in &point.as_slice()[1..(point.num_variables() - log_packing_width)] {
            let half_buffer_len = buffer.len() / 2;
            let (lo, hi) = buffer.split_at_mut(half_buffer_len);
            lo.iter_mut().zip(hi.iter()).for_each(|(l, &h)| {
                *l += h * ef_eval;
            });
            buffer.truncate(half_buffer_len);
        }

        // After this, buffer should have been reduced to a single element.
        // Further reductions will need to be done in the non-packed extension field.
        let base_elems = E::ExtensionPacking::to_ext_iter(buffer).collect::<Vec<_>>();
        eval_extension_unpacked(
            &base_elems,
            &MultilinearPoint::new(
                point.as_slice()[point.num_variables() - log_packing_width..].to_vec(),
            ),
        )
    }
}

/// Recursively evaluates a multilinear polynomial at an extension field point.
///
/// Given `coeffs` in lexicographic order, this computes:
/// ```ignore
/// eval_poly(X_0, ..., X_n) = sum(coeffs[i] * product(X_j for j in S(i)))
/// ```
/// where `S(i)` is the set of variables active in term `i` (based on its binary representation).
///
/// - The cases `n = 0, ..., 4` are hard coded.
/// - Uses divide-and-conquer recursion for larger cases: (Note this is slow and should usually be avoided for large n)
///   - Splits `coeffs` into two halves for `X_0 = 0` and `X_0 = 1`.
///   - Recursively evaluates each half.
#[inline]
fn eval_extension_unpacked<F, E>(coeff: &[F], point: &MultilinearPoint<E>) -> E
where
    F: Field,
    E: ExtensionField<F>,
{
    let num_vars = point.num_variables();
    debug_assert_eq!(coeff.len(), 1 << num_vars);

    let point = point.as_slice();

    match num_vars {
        0 => coeff[0].into(),
        1 => point[0] * coeff[1] + coeff[0],
        2 => point[0] * (point[1] * coeff[3] + coeff[2]) + point[1] * coeff[1] + coeff[0],
        3 => {
            point[0]
                * (point[1] * (point[2] * coeff[7] + coeff[6]) + point[2] * coeff[5] + coeff[4])
                + point[1] * (point[2] * coeff[3] + coeff[2])
                + point[2] * coeff[1]
                + coeff[0]
        }
        4 => {
            point[0]
                * (point[1]
                    * (point[2] * (point[3] * coeff[15] + coeff[14])
                        + point[3] * coeff[13]
                        + coeff[12])
                    + point[2] * (point[3] * coeff[11] + coeff[10])
                    + point[3] * coeff[9]
                    + coeff[8])
                + point[1]
                    * (point[2] * (point[3] * coeff[7] + coeff[6]) + point[3] * coeff[5] + coeff[4])
                + point[2] * (point[3] * coeff[3] + coeff[2])
                + point[3] * coeff[1]
                + coeff[0]
        }
        _ => {
            let (lo, hi) = coeff.split_at(coeff.len() / 2);
            eval_extension_unpacked(lo, &MultilinearPoint::new(point[1..].to_vec()))
                + point[0]
                    * eval_extension_unpacked(hi, &MultilinearPoint::new(point[1..].to_vec()))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, extension::BinomialExtensionField};
    use proptest::prelude::*;
    use rand::{Rng, rng};

    use super::*;
    use crate::poly::evals::EvaluationsList;

    type F = BabyBear;
    type EF4 = BinomialExtensionField<F, 4>;

    #[test]
    fn test_coefficient_list_initialization() {
        let coeffs = vec![
            F::from_u64(3),
            F::from_u64(1),
            F::from_u64(4),
            F::from_u64(1),
        ];
        let coeff_list = CoefficientList::new(coeffs.clone());

        // Check that the coefficients are stored correctly
        assert_eq!(&*coeff_list, &coeffs);
        // Since len = 4 = 2^2, we expect num_variables = 2
        assert_eq!(coeff_list.num_variables(), 2);
    }

    #[test]
    fn test_evaluate_multilinear() {
        let coeff0 = F::from_u64(8);
        let coeff1 = F::from_u64(2);
        let coeff2 = F::from_u64(3);
        let coeff3 = F::from_u64(1);

        let coeffs = vec![coeff0, coeff1, coeff2, coeff3];
        let coeff_list = CoefficientList::new(coeffs);

        let x0 = F::from_u64(2);
        let x1 = F::from_u64(3);
        let point = MultilinearPoint::new(vec![x0, x1]);

        // Expected value based on multilinear evaluation
        let expected_value = coeff0 + coeff1 * x1 + coeff2 * x0 + coeff3 * x0 * x1;
        assert_eq!(coeff_list.evaluate(&point), expected_value);
    }

    #[test]
    fn test_folding() {
        let coeff0 = F::from_u64(6);
        let coeff1 = F::from_u64(2);
        let coeff2 = F::from_u64(4);
        let coeff3 = F::from_u64(8);

        let coeffs = vec![coeff0, coeff1, coeff2, coeff3];
        let coeff_list = CoefficientList::new(coeffs);

        let folding_value = F::from_u64(3);
        let folding_point = MultilinearPoint::new(vec![folding_value]);
        let folded = coeff_list.fold(&folding_point);

        let eval_value = F::from_u64(5);
        let expected_eval =
            coeff_list.evaluate(&MultilinearPoint::new(vec![eval_value, folding_value]));

        // Ensure folding preserves evaluation correctness
        assert_eq!(
            folded.evaluate(&MultilinearPoint::new(vec![eval_value])),
            expected_eval
        );
    }

    #[test]
    fn test_folding_multiple_variables() {
        let num_variables = 3;
        let coeffs: Vec<_> = (0..(1 << num_variables)).map(F::from_u64).collect();
        let coeff_list = CoefficientList::new(coeffs);

        let fold_x1 = F::from_u64(4);
        let fold_x2 = F::from_u64(2);
        let folding_point = MultilinearPoint::new(vec![fold_x1, fold_x2]);

        let folded = coeff_list.fold(&folding_point);

        let eval_x0 = F::from_u64(6);
        let full_point = MultilinearPoint::new(vec![eval_x0, fold_x1, fold_x2]);
        let expected_eval = coeff_list.evaluate(&full_point);

        // Ensure correctness of folding and evaluation
        assert_eq!(
            folded.evaluate(&MultilinearPoint::new(vec![eval_x0])),
            expected_eval
        );
    }

    #[test]
    fn test_coefficient_to_evaluations_conversion() {
        let coeff0 = F::from_u64(5);
        let coeff1 = F::from_u64(3);
        let coeff2 = F::from_u64(7);
        let coeff3 = F::from_u64(2);

        let coeffs = vec![coeff0, coeff1, coeff2, coeff3];
        let coeff_list = CoefficientList::new(coeffs);

        let evaluations = coeff_list.to_evaluations();
        let evaluations = evaluations.as_slice();

        // Expected results after wavelet transform (manually derived)
        assert_eq!(evaluations[0], coeff0);
        assert_eq!(evaluations[1], coeff0 + coeff1);
        assert_eq!(evaluations[2], coeff0 + coeff2);
        assert_eq!(evaluations[3], coeff0 + coeff1 + coeff2 + coeff3);
    }

    #[test]
    fn test_num_variables_and_coeffs() {
        // 8 = 2^3, so num_variables = 3
        let coeffs = vec![F::from_u64(1); 8];
        let coeff_list = CoefficientList::new(coeffs);

        assert_eq!(coeff_list.num_variables(), 3);
        assert_eq!(coeff_list.num_coeffs(), 8);
    }

    #[test]
    #[should_panic]
    fn test_coefficient_list_empty() {
        let _coeff_list = CoefficientList::<F>::new(vec![]);
    }

    #[test]
    #[should_panic]
    fn test_coefficient_list_invalid_size() {
        // 7 is not a power of two
        let _coeff_list = CoefficientList::new(vec![F::from_u64(1); 7]);
    }

    #[test]
    fn test_folding_and_evaluation() {
        // Set the number of Boolean variables (so the polynomial has 2^num_variables entries).
        let num_variables = 10;

        // Build a polynomial with coefficients f(x) = [0, 1, 2, ..., 2^10 - 1] ∈ F
        let coeffs = (0..(1 << num_variables)).map(F::from_u64).collect();

        // Wrap the raw coefficients into a CoefficientList for multilinear operations
        let coeffs_list = CoefficientList::new(coeffs);

        // Build a fixed evaluation point in F^10 by setting the i-th variable to 35 * i
        let randomness: Vec<_> = (0..num_variables)
            .map(|i| F::from_u64(35 * i as u64))
            .collect();

        // Try folding at every prefix length k = 0..10
        for k in 0..num_variables {
            // Take the first `k` coordinates as folding randomness
            let fold_part = randomness[0..k].to_vec();

            // The remaining coordinates will form the evaluation point for the folded poly
            let eval_part = randomness[k..].to_vec();

            // Convert `fold_part` into a MultilinearPoint to fold the polynomial
            let fold_random = MultilinearPoint::new(fold_part.clone());

            // Evaluate the original polynomial at the point [eval_part || fold_part]
            // to check folding + evaluation match full evaluation
            let eval_point = MultilinearPoint::new([eval_part.clone(), fold_part].concat());

            // Perform the folding step: reduce the polynomial to fewer variables
            let folded = coeffs_list.fold(&fold_random);

            // Check that folding followed by evaluation matches direct evaluation
            assert_eq!(
                folded.evaluate(&MultilinearPoint::new(eval_part)),
                coeffs_list.evaluate(&eval_point)
            );
        }
    }

    #[test]
    fn test_evaluate_at_extension_single_variable() {
        // Polynomial f(X) = 3 + 7X in base field
        let coeff0 = F::from_u64(3);
        let coeff1 = F::from_u64(7);
        let coeffs = vec![coeff0, coeff1];
        let coeff_list = CoefficientList::new(coeffs);

        let x = EF4::from_u64(2); // Evaluation at x = 2 in extension field
        let expected_value = EF4::from_u64(3) + EF4::from_u64(7) * x; // f(2) = 3 + 7 * 2
        let eval_result = coeff_list.evaluate(&MultilinearPoint::new(vec![x]));

        assert_eq!(eval_result, expected_value);
    }

    #[test]
    fn test_evaluate_at_extension_two_variables() {
        // Polynomial f(X₀, X₁) = 2 + 5X₀ + 3X₁ + 7X₀X₁
        let coeffs = vec![
            F::from_u64(2), // Constant term
            F::from_u64(5), // X₁ term
            F::from_u64(3), // X₀ term
            F::from_u64(7), // X₀X₁ term
        ];
        let coeff_list = CoefficientList::new(coeffs);

        let x0 = EF4::from_u64(2);
        let x1 = EF4::from_u64(3);
        let expected_value = EF4::from_u64(2)
            + EF4::from_u64(5) * x1
            + EF4::from_u64(3) * x0
            + EF4::from_u64(7) * x0 * x1;
        let eval_result = coeff_list.evaluate(&MultilinearPoint::new(vec![x0, x1]));

        assert_eq!(eval_result, expected_value);
    }

    #[test]
    fn test_evaluate_at_extension_three_variables() {
        // Polynomial: f(X₀, X₁, X₂) = 1 + 2X₂ + 3X₁ + 4X₁X₂ + 5X₀ + 6X₀X₂ + 7X₀X₁ + 8X₀X₁X₂
        let coeffs = vec![
            F::from_u64(1), // Constant term (000)
            F::from_u64(2), // X₂ (001)
            F::from_u64(3), // X₁ (010)
            F::from_u64(5), // X₁X₂ (011)
            F::from_u64(4), // X₀ (100)
            F::from_u64(6), // X₀X₂ (101)
            F::from_u64(7), // X₀X₁ (110)
            F::from_u64(8), // X₀X₁X₂ (111)
        ];
        let coeff_list = CoefficientList::new(coeffs);

        let x0 = EF4::from_u64(2);
        let x1 = EF4::from_u64(3);
        let x2 = EF4::from_u64(4);

        // Correct expected value based on the coefficient order
        let expected_value = EF4::from_u64(1)
            + EF4::from_u64(2) * x2
            + EF4::from_u64(3) * x1
            + EF4::from_u64(5) * x1 * x2
            + EF4::from_u64(4) * x0
            + EF4::from_u64(6) * x0 * x2
            + EF4::from_u64(7) * x0 * x1
            + EF4::from_u64(8) * x0 * x1 * x2;

        let eval_result = coeff_list.evaluate(&MultilinearPoint::new(vec![x0, x1, x2]));

        assert_eq!(eval_result, expected_value);
    }

    #[test]
    fn test_evaluate_at_extension_zero_polynomial() {
        // Zero polynomial f(X) = 0
        let coeff_list = CoefficientList::new(vec![F::ZERO; 4]); // f(X₀, X₁) = 0

        let x0 = EF4::from_u64(5);
        let x1 = EF4::from_u64(7);
        let eval_result = coeff_list.evaluate(&MultilinearPoint::new(vec![x0, x1]));

        assert_eq!(eval_result, EF4::ZERO);
    }

    #[test]
    fn test_evaluate_at_univariate_degree_one() {
        // Polynomial: f(x) = 3 + 4x
        let c0 = F::from_u64(3);
        let c1 = F::from_u64(4);
        let coeffs = vec![c0, c1];
        let poly = CoefficientList::new(coeffs);

        let p0 = F::from_u64(0);
        let p1 = F::from_u64(1);
        let p2 = F::from_u64(2);
        let p3 = F::from_u64(5);
        let points = vec![p0, p1, p2, p3];

        // Manually compute expected values from coeffs
        // f(x) = coeffs[0] + coeffs[1] * x
        let expected = vec![
            c0 + c1 * p0, // 3 + 4 * 0
            c0 + c1 * p1, // 3 + 4 * 1
            c0 + c1 * p2, // 3 + 4 * 2
            c0 + c1 * p3, // 3 + 4 * 5
        ];

        let result = poly.evaluate_at_univariate(&points);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_evaluate_at_univariate_degree_three_multiple_points() {
        // Polynomial: f(x) = 1 + 2x + 3x² + 4x³
        let c0 = F::from_u64(1);
        let c1 = F::from_u64(2);
        let c2 = F::from_u64(3);
        let c3 = F::from_u64(4);
        let coeffs = vec![c0, c1, c2, c3];
        let poly = CoefficientList::new(coeffs);

        let p0 = F::from_u64(0);
        let p1 = F::from_u64(1);
        let p2 = F::from_u64(2);
        let points = vec![p0, p1, p2];

        // f(x) = c0 + c1*x + c2*x² + c3*x³
        let expected = vec![
            c0 + c1 * p0 + c2 * p0.square() + c3 * p0.square() * p0,
            c0 + c1 * p1 + c2 * p1.square() + c3 * p1.square() * p1,
            c0 + c1 * p2 + c2 * p2.square() + c3 * p2.square() * p2,
        ];

        let result = poly.evaluate_at_univariate(&points);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_evaluate_at_univariate_equivalence_to_multilinear() {
        // Polynomial: f(x) = 5 + 6x + 7x² + 8x³
        let c0 = F::from_u64(5);
        let c1 = F::from_u64(6);
        let c2 = F::from_u64(7);
        let c3 = F::from_u64(8);
        let coeffs = vec![c0, c1, c2, c3];
        let poly = CoefficientList::new(coeffs);

        let x = F::from_u64(2);

        let expected = c0 + c1 * x + c2 * x.square() + c3 * x.square() * x;

        let result_univariate = poly.evaluate_at_univariate(&[x])[0];

        let ml_point = MultilinearPoint::expand_from_univariate(x, 2);
        let result_multilinear = poly.evaluate(&ml_point);

        assert_eq!(result_univariate, expected);
        assert_eq!(result_multilinear, expected);
    }

    #[test]
    fn test_eval_multivariate_with_extension_field_points() {
        let mut rng = rng();

        let c0 = F::from_u64(1);
        let c1 = F::from_u64(2);
        let c2 = F::from_u64(3);
        let c3 = F::from_u64(4);

        let coeffs = vec![c0, c1, c2, c3];
        assert_eq!(coeffs.len(), 4);

        let x0 = F::from_u64(5);
        let x1 = F::from_u64(7);

        let eval_extension = eval_multivariate(
            &coeffs,
            &MultilinearPoint::new(vec![EF4::from(x0), EF4::from(x1)]),
        );
        let expected = c0 + c1 * x1 + c2 * x0 + c3 * x0 * x1;

        assert_eq!(eval_extension, EF4::from(expected));

        // Compare with base field evaluation
        let eval_base = eval_multivariate(&coeffs, &MultilinearPoint::new(vec![x0, x1]));
        assert_eq!(eval_extension, EF4::from(eval_base));

        // Now test with some random extension points
        let e: EF4 = rng.random();
        let f: EF4 = rng.random();

        let eval = eval_multivariate(&coeffs, &MultilinearPoint::new(vec![e, f]));

        let expected =
            EF4::from(c0) + EF4::from(c1) * f + EF4::from(c2) * e + EF4::from(c3) * e * f;

        assert_eq!(eval, expected);
    }

    #[test]
    fn test_eval_multivariate_constant_poly() {
        let c = F::from_u64(42);
        let coeffs = vec![c];

        let points: Vec<EF4> = vec![]; // Zero variables

        let result = eval_multivariate(&coeffs, &MultilinearPoint::new(points));
        assert_eq!(result, EF4::from(c));
    }

    #[test]
    fn test_eval_multivariate_all_zero_point() {
        let coeffs = vec![
            F::from_u64(11), // constant term
            F::from_u64(22), // x₁
            F::from_u64(33), // x₀
            F::from_u64(44), // x₀x₁
        ];
        let zeros = vec![EF4::ZERO, EF4::ZERO];

        let result = eval_multivariate(&coeffs, &MultilinearPoint::new(zeros));
        assert_eq!(result, EF4::from(coeffs[0])); // Only constant survives
    }

    #[test]
    fn test_eval_multivariate_all_ones_point() {
        let coeffs = vec![
            F::from_u64(5), // 1
            F::from_u64(6), // x₁
            F::from_u64(7), // x₀
            F::from_u64(8), // x₀x₁
        ];

        let one = EF4::ONE;
        let result = eval_multivariate(&coeffs, &MultilinearPoint::new(vec![one, one]));

        let expected = EF4::from(coeffs[0])
            + EF4::from(coeffs[1])
            + EF4::from(coeffs[2])
            + EF4::from(coeffs[3]);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_eval_multivariate_three_vars_mixed() {
        // We define a multilinear polynomial in three variables (X₀, X₁, X₂):
        //
        // f(X₀, X₁, X₂) =
        //    1                     (constant term)
        //  + 2·X₂                  (degree-1 term in X₂)
        //  + 3·X₁                  (degree-1 term in X₁)
        //  + 4·X₁·X₂               (degree-2 term X₁X₂)
        //  + 5·X₀                  (degree-1 term in X₀)
        //  + 6·X₀·X₂               (degree-2 term X₀X₂)
        //  + 7·X₀·X₁               (degree-2 term X₀X₁)
        //  + 8·X₀·X₁·X₂            (degree-3 term X₀X₁X₂)

        let coeffs = vec![
            F::from_u64(1), // coeff for constant term
            F::from_u64(2), // coeff for X₂
            F::from_u64(3), // coeff for X₁
            F::from_u64(4), // coeff for X₁·X₂
            F::from_u64(5), // coeff for X₀
            F::from_u64(6), // coeff for X₀·X₂
            F::from_u64(7), // coeff for X₀·X₁
            F::from_u64(8), // coeff for X₀·X₁·X₂
        ];

        // Now define evaluation point (x₀, x₁, x₂) ∈ EF4³:
        let x0 = EF4::from_u64(2); // x₀ = 2 (embedded from base field)
        let x1 = EF4::from_u64(3); // x₁ = 3
        let x2 = EF4::from_basis_coefficients_iter(
            [F::new(9), F::new(0), F::new(1), F::new(0)].into_iter(),
        )
        .unwrap(); // x₂ is a custom EF4 element

        // Evaluate the multilinear polynomial at (x₀, x₁, x₂)
        let result = eval_multivariate(&coeffs, &MultilinearPoint::new(vec![x0, x1, x2]));

        // Manually expand the expected result:
        //
        // f(x₀,x₁,x₂) =
        //    1
        //  + 2 * x₂
        //  + 3 * x₁
        //  + 4 * (x₁ * x₂)
        //  + 5 * x₀
        //  + 6 * (x₀ * x₂)
        //  + 7 * (x₀ * x₁)
        //  + 8 * (x₀ * x₁ * x₂)
        //
        // Each coefficient is promoted from F to EF4 before multiplication.

        let expected = EF4::from(coeffs[0])
            + EF4::from(coeffs[1]) * x2
            + EF4::from(coeffs[2]) * x1
            + EF4::from(coeffs[3]) * x1 * x2
            + EF4::from(coeffs[4]) * x0
            + EF4::from(coeffs[5]) * x0 * x2
            + EF4::from(coeffs[6]) * x0 * x1
            + EF4::from(coeffs[7]) * x0 * x1 * x2;

        // Final assertion: evaluated result must match the manual expansion
        assert_eq!(result, expected);
    }

    #[test]
    fn test_fold_with_extension_one_var() {
        // Polynomial: f(X₀, X₁) = 1 + 2X₁ + 3X₀ + 4X₀X₁
        let coeffs = vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ];
        let poly = CoefficientList::new(coeffs);

        // Fold with X₁ = 5 (in EF4)
        let r1 = EF4::from_u64(5);
        let folded = poly.fold(&MultilinearPoint::new(vec![r1]));

        // Should produce polynomial in X₀ only
        for x0_f in 0..10 {
            let x0 = EF4::from_u64(x0_f);
            let full_point = MultilinearPoint::new(vec![x0, r1]);
            let folded_point = MultilinearPoint::new(vec![x0]);

            let expected = poly.evaluate(&full_point);
            let actual = folded.evaluate(&folded_point);
            assert_eq!(expected, actual);
        }
    }

    #[test]
    fn test_fold_with_extension_two_vars() {
        // f(X₀, X₁, X₂) = 1 + 2X₂ + 3X₁ + 4X₁X₂ + 5X₀ + 6X₀X₂ + 7X₀X₁ + 8X₀X₁X₂
        let coeffs = (1..=8).map(F::from_u64).collect::<Vec<_>>();
        let poly = CoefficientList::new(coeffs);

        let r1 = EF4::from_u64(9);
        let r2 = EF4::from_basis_coefficients_iter(
            [F::new(2), F::new(0), F::new(3), F::new(1)].into_iter(),
        )
        .unwrap();

        let folded = poly.fold(&MultilinearPoint::new(vec![r1, r2]));

        for x0_f in 0..10 {
            let x0 = EF4::from_u64(x0_f);
            let full_point = MultilinearPoint::new(vec![x0, r1, r2]);
            let folded_point = MultilinearPoint::new(vec![x0]);

            let expected = poly.evaluate(&full_point);
            let actual = folded.evaluate(&folded_point);
            assert_eq!(expected, actual);
        }
    }

    #[test]
    fn test_fold_with_zero_extension_randomness() {
        // f(X₀, X₁) = 1 + 2X₁ + 3X₀ + 4X₀X₁
        let coeffs = vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ];
        let poly = CoefficientList::new(coeffs);

        let zero = EF4::ZERO;
        let folded = poly.fold(&MultilinearPoint::new(vec![zero]));

        // Should be equivalent to evaluating X₁ = 0 in original poly
        for x0_f in 0..5 {
            let x0 = EF4::from_u64(x0_f);
            let full_point = MultilinearPoint::new(vec![x0, zero]);
            let folded_point = MultilinearPoint::new(vec![x0]);

            let expected = poly.evaluate(&full_point);
            let actual = folded.evaluate(&folded_point);
            assert_eq!(expected, actual);
        }
    }

    #[test]
    fn test_coefficient_list_to_evaluations_list_three_variables() {
        // Define a multilinear polynomial:
        //
        // f(X₀, X₁, X₂) =
        //    1                  (constant term)
        //  + 2·X₂               (degree-1 term in X₂)
        //  + 3·X₁               (degree-1 term in X₁)
        //  + 4·X₁·X₂            (degree-2 term X₁X₂)
        //  + 5·X₀               (degree-1 term in X₀)
        //  + 6·X₀·X₂            (degree-2 term X₀X₂)
        //  + 7·X₀·X₁            (degree-2 term X₀X₁)
        //  + 8·X₀·X₁·X₂         (degree-3 term X₀X₁X₂)

        let coeffs = vec![
            F::from_u64(1), // index 0: constant
            F::from_u64(2), // index 1: X₂
            F::from_u64(3), // index 2: X₁
            F::from_u64(4), // index 3: X₁·X₂
            F::from_u64(5), // index 4: X₀
            F::from_u64(6), // index 5: X₀·X₂
            F::from_u64(7), // index 6: X₀·X₁
            F::from_u64(8), // index 7: X₀·X₁·X₂
        ];

        let coeff_list = CoefficientList::new(coeffs);

        // Convert to evaluations list via wavelet transform
        let eval_list = coeff_list.to_evaluations();

        // Manually compute expected evaluations.
        //
        // Evaluations correspond to values of f(x₀,x₁,x₂) at all (x₀,x₁,x₂) ∈ {0,1}³
        //
        // Order of points is lex smallest bit last:
        // [ (x₂, x₁, x₀) ] ← binary counting
        //
        // | Index | (X₀, X₁, X₂) | f(X₀,X₁,X₂) |
        // |:-----:|:------------:|:-----------:|
        // |   0   | (0, 0, 0)    | 1
        // |   1   | (0, 0, 1)    | 1 + 2
        // |   2   | (0, 1, 0)    | 1 + 3
        // |   3   | (0, 1, 1)    | 1 + 2 + 3 + 4
        // |   4   | (1, 0, 0)    | 1 + 5
        // |   5   | (1, 0, 1)    | 1 + 2 + 5 + 6
        // |   6   | (1, 1, 0)    | 1 + 3 + 5 + 7
        // |   7   | (1, 1, 1)    | 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8

        let expected = vec![
            F::from_u64(1),
            F::from_u64(1 + 2),
            F::from_u64(1 + 3),
            F::from_u64(1 + 2 + 3 + 4),
            F::from_u64(1 + 5),
            F::from_u64(1 + 2 + 5 + 6),
            F::from_u64(1 + 3 + 5 + 7),
            F::from_u64(1 + 2 + 3 + 4 + 5 + 6 + 7 + 8),
        ];

        assert_eq!(eval_list.as_slice(), &expected);
    }

    proptest! {
        #[test]
        fn prop_eval_coeff_roundtrip_varying_size(
            // Random power-of-two length exponent: generates sizes 2^0 to 2^10
            log_len in 0usize..=10,
            // Overallocate to 1024, we'll truncate later
            coeffs in prop::collection::vec(0u64..100_000, 1 << 10)
        ) {
            // Compute length as power of two
            let len = 1 << log_len;

            // Take the first `len` elements and convert to field elements
            let input: Vec<F> = coeffs.into_iter().take(len).map(F::from_u64).collect();

            // Wrap input as an EvaluationsList
            let evals = EvaluationsList::new(input);

            // Apply inverse wavelet transform to get coefficients followed
            // by forward wavelet transform to get evaluations back.
            let roundtrip = evals.clone().to_coefficients().to_evaluations();

            // Final assertion: roundtrip must be exact
            prop_assert_eq!(roundtrip, evals);
        }
    }

    #[test]
    fn test_evaluations_to_coefficients_roundtrip() {
        // Construct evaluations in lex order for a 3-variable polynomial
        // Example polynomial: f(x₀, x₁, x₂) = x₂ + 2x₁ + 3x₀ + 4x₀x₁ + 5x₀x₂ + ...
        let evals = vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
            F::from_u64(5),
            F::from_u64(6),
            F::from_u64(7),
            F::from_u64(8),
        ];
        let original = EvaluationsList::new(evals);

        // Apply inverse wavelet transform to get coefficients followed
        // by forward wavelet transform to get evaluations back.
        let roundtrip = original.clone().to_coefficients().to_evaluations();

        // The recovered evaluations must exactly match the original
        assert_eq!(roundtrip, original);
    }
}
