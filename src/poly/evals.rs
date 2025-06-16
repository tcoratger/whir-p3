use std::ops::Deref;

use p3_field::{ExtensionField, Field};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use tracing::instrument;

use super::{coeffs::CoefficientList, multilinear::MultilinearPoint, wavelet::Radix2WaveletKernel};
use crate::utils::{eval_eq, parallel_clone, uninitialized_vec};

/// A wrapper enum that holds evaluation data for a multilinear polynomial,
/// either over the base field `F` or an extension field `EF`.
///
/// This abstraction allows operating generically on both base and extension
/// field evaluations, as used in sumcheck protocols and other polynomial
/// computations.
#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) enum EvaluationStorage<F, EF> {
    /// Evaluation data over the base field `F`.
    Base(EvaluationsList<F>),
    /// Evaluation data over the extension field `EF`.
    Extension(EvaluationsList<EF>),
}

impl<F, EF> EvaluationStorage<F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    /// Returns the number of variables in the stored evaluation list.
    ///
    /// This corresponds to the logarithm base 2 of the number of evaluation points.
    /// It is assumed that the number of evaluations is a power of two.
    ///
    /// # Returns
    /// - `usize`: The number of input variables of the underlying multilinear polynomial.
    pub(crate) const fn num_variables(&self) -> usize {
        match self {
            Self::Base(evals) => evals.num_variables(),
            Self::Extension(evals) => evals.num_variables(),
        }
    }

    /// Folds the stored polynomial using the provided `folding_randomness`, returning a new
    /// `EvaluationsList<EF>` with fewer variables.
    ///
    /// Works generically on both base and extension field representations.
    #[instrument(skip_all)]
    pub(crate) fn fold(&self, folding_randomness: &MultilinearPoint<EF>) -> EvaluationsList<EF> {
        match self {
            Self::Base(cl) => cl.fold(folding_randomness),
            Self::Extension(cl) => cl.fold(folding_randomness),
        }
    }
}

/// Represents a multilinear polynomial `f` in `num_variables` unknowns, stored via its evaluations
/// over the hypercube `{0,1}^{num_variables}`.
///
/// The vector `evals` contains function evaluations at **lexicographically ordered** points.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct EvaluationsList<F> {
    /// Stores evaluations in **lexicographic order**.
    evals: Vec<F>,
    /// Number of variables in the multilinear polynomial.
    /// Ensures `evals.len() = 2^{num_variables}`.
    num_variables: usize,
}

impl<F> EvaluationsList<F>
where
    F: Field,
{
    /// Constructs an `EvaluationsList` from a given vector of evaluations.
    ///
    /// - The `evals` vector must have a **length that is a power of two** since it represents
    ///   evaluations over an `n`-dimensional binary hypercube.
    /// - The ordering of evaluation points follows **lexicographic order**.
    ///
    /// **Mathematical Constraint:**
    /// If `evals.len() = 2^n`, then `num_variables = n`, ensuring correct indexing.
    ///
    /// **Panics:**
    /// - If `evals.len()` is **not** a power of two.
    #[must_use]
    pub fn new(evals: Vec<F>) -> Self {
        let len = evals.len();
        assert!(
            len.is_power_of_two(),
            "Evaluation list length must be a power of two."
        );

        Self {
            evals,
            num_variables: len.ilog2() as usize,
        }
    }

    /// Given `evals` = (α_1, ..., α_n), returns a multilinear polynomial P in n variables,
    /// defined on the boolean hypercube by: ∀ (x_1, ..., x_n) ∈ {0, 1}^n,
    /// P(x_1, ..., x_n) = Π_{i=1}^{n} (x_i.α_i + (1 - x_i).(1 - α_i))
    /// (often denoted as P(x) = eq(x, evals))
    pub fn eval_eq(eval: &[F]) -> Self {
        // Alloc memory without initializing it to zero.
        // This is safe because we overwrite it inside `eval_eq`.
        let mut out: Vec<F> = Vec::with_capacity(1 << eval.len());
        #[allow(clippy::uninit_vec)]
        unsafe {
            out.set_len(1 << eval.len());
        }
        eval_eq::<_, _, false>(eval, &mut out, F::ONE);
        Self {
            evals: out,
            num_variables: eval.len(),
        }
    }

    /// Returns an immutable reference to the evaluations vector.
    #[must_use]
    pub fn evals(&self) -> &[F] {
        &self.evals
    }

    /// Returns a mutable reference to the evaluations vector.
    pub fn evals_mut(&mut self) -> &mut [F] {
        &mut self.evals
    }

    /// Returns the total number of stored evaluations.
    ///
    /// Mathematical Invariant:
    /// ```ignore
    /// num_evals = 2^{num_variables}
    /// ```
    #[must_use]
    pub fn num_evals(&self) -> usize {
        self.evals.len()
    }

    /// Returns the number of variables in the multilinear polynomial.
    #[must_use]
    pub const fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// Evaluates the multilinear polynomial at `point ∈ [0,1]^n`.
    ///
    /// - If `point ∈ {0,1}^n`, returns the precomputed evaluation `f(point)`.
    /// - Otherwise, computes `f(point) = ∑_{x ∈ {0,1}^n} eq(x, point) * f(x)`, where `eq(x, point)
    ///   = ∏_{i=1}^{n} (1 - p_i + 2 p_i x_i)`.
    /// - Uses fast multilinear interpolation for efficiency.
    #[must_use]
    pub fn evaluate<EF>(&self, point: &MultilinearPoint<EF>) -> EF
    where
        EF: ExtensionField<F>,
    {
        if let Some(point) = point.to_hypercube() {
            return self.evals[point.0].into();
        }
        eval_multilinear(&self.evals, &point.0)
    }

    /// Folds a multilinear polynomial stored in evaluation form along the last `k` variables.
    ///
    /// Given evaluations `f: {0,1}^n → F`, this method returns a new evaluation list `g` such that:
    ///
    /// \[
    /// g(x_0, ..., x_{n-k-1}) = f(x_0, ..., x_{n-k-1}, r_0, ..., r_{k-1})
    /// \]
    ///
    /// where `folding_randomness = (r_0, ..., r_{k-1})` is a fixed assignment to the last `k` variables.
    ///
    /// This operation reduces the dimensionality of the polynomial:
    ///
    /// - Input: `f ∈ F^{2^n}`
    /// - Output: `g ∈ EF^{2^{n-k}}`, where `EF` is an extension field of `F`
    ///
    /// # Arguments
    /// - `folding_randomness`: The extension-field values to substitute for the last `k` variables.
    ///
    /// # Returns
    /// - A new `EvaluationsList<EF>` representing the folded function over the remaining `n - k` variables.
    ///
    /// # Panics
    /// - If the evaluation list is not sized `2^n` for some `n`.
    #[instrument(skip_all)]
    #[must_use]
    pub fn fold<EF>(&self, folding_randomness: &MultilinearPoint<EF>) -> EvaluationsList<EF>
    where
        EF: ExtensionField<F>,
    {
        let folding_factor = folding_randomness.num_variables();
        #[cfg(not(feature = "parallel"))]
        let evals = self
            .evals
            .chunks_exact(1 << folding_factor)
            .map(|ev| eval_multilinear(ev, &folding_randomness.0))
            .collect();
        #[cfg(feature = "parallel")]
        let evals = self
            .evals
            .par_chunks_exact(1 << folding_factor)
            .map(|ev| eval_multilinear(ev, &folding_randomness.0))
            .collect();

        EvaluationsList {
            evals,
            num_variables: self.num_variables() - folding_factor,
        }
    }

    #[must_use]
    #[instrument(skip_all)]
    pub fn parallel_clone(&self) -> Self {
        let mut evals = unsafe { uninitialized_vec(self.evals.len()) };
        parallel_clone(&self.evals, &mut evals);
        Self {
            evals,
            num_variables: self.num_variables,
        }
    }

    /// Multiply the polynomial by a scalar factor.
    #[must_use]
    pub fn scale<EF: ExtensionField<F>>(&self, factor: EF) -> EvaluationsList<EF> {
        #[cfg(not(feature = "parallel"))]
        let evals = self.evals.iter().map(|&e| factor * e).collect();
        #[cfg(feature = "parallel")]
        let evals = self.evals.par_iter().map(|&e| factor * e).collect();
        EvaluationsList {
            evals,
            num_variables: self.num_variables(),
        }
    }

    /// Convert from a list of evaluations to a list of
    /// multilinear coefficients.
    #[must_use]
    #[instrument(skip_all)]
    pub fn to_coefficients<B: Field>(self) -> CoefficientList<F>
    where
        F: ExtensionField<B>,
    {
        let kernel = Radix2WaveletKernel::<B>::default();
        let evals = kernel.inverse_wavelet_transform_algebra(self.evals);
        CoefficientList::new(evals)
    }
}

impl<F> Deref for EvaluationsList<F> {
    type Target = [F];

    fn deref(&self) -> &Self::Target {
        &self.evals
    }
}

/// Evaluates a multilinear polynomial at `point ∈ [0,1]^n` using fast interpolation.
///
/// - Given evaluations `evals` over `{0,1}^n`, computes `f(point)` via iterative interpolation.
/// - Uses the recurrence: `f(x_1, ..., x_n) = (1 - x_1) f_0 + x_1 f_1`, reducing dimension at each
///   step.
/// - Ensures `evals.len() = 2^n` to match the number of variables.
fn eval_multilinear<F, EF>(evals: &[F], point: &[EF]) -> EF
where
    F: Field,
    EF: ExtensionField<F>,
{
    debug_assert_eq!(evals.len(), 1 << point.len());
    match point {
        [] => evals[0].into(),
        [x] => *x * (evals[1] - evals[0]) + evals[0],
        [x0, x1] => {
            let a0 = *x1 * (evals[1] - evals[0]) + evals[0];
            let a1 = *x1 * (evals[3] - evals[2]) + evals[2];
            a0 + (a1 - a0) * *x0
        }
        [x0, x1, x2] => {
            let a00 = *x2 * (evals[1] - evals[0]) + evals[0];
            let a01 = *x2 * (evals[3] - evals[2]) + evals[2];
            let a10 = *x2 * (evals[5] - evals[4]) + evals[4];
            let a11 = *x2 * (evals[7] - evals[6]) + evals[6];
            let a0 = a00 + *x1 * (a01 - a00);
            let a1 = a10 + *x1 * (a11 - a10);
            a0 + (a1 - a0) * *x0
        }
        [x0, x1, x2, x3] => {
            let a000 = *x3 * (evals[1] - evals[0]) + evals[0];
            let a001 = *x3 * (evals[3] - evals[2]) + evals[2];
            let a010 = *x3 * (evals[5] - evals[4]) + evals[4];
            let a011 = *x3 * (evals[7] - evals[6]) + evals[6];
            let a100 = *x3 * (evals[9] - evals[8]) + evals[8];
            let a101 = *x3 * (evals[11] - evals[10]) + evals[10];
            let a110 = *x3 * (evals[13] - evals[12]) + evals[12];
            let a111 = *x3 * (evals[15] - evals[14]) + evals[14];
            let a00 = a000 + *x2 * (a001 - a000);
            let a01 = a010 + *x2 * (a011 - a010);
            let a10 = a100 + *x2 * (a101 - a100);
            let a11 = a110 + *x2 * (a111 - a110);
            let a0 = a00 + *x1 * (a01 - a00);
            let a1 = a10 + *x1 * (a11 - a10);
            a0 + (a1 - a0) * *x0
        }
        [x, tail @ ..] => {
            let (f0, f1) = evals.split_at(evals.len() / 2);
            #[cfg(not(feature = "parallel"))]
            let (f0, f1) = (eval_multilinear(f0, tail), eval_multilinear(f1, tail));
            #[cfg(feature = "parallel")]
            let (f0, f1) = {
                let work_size: usize = (1 << 15) / std::mem::size_of::<F>();
                if evals.len() > work_size {
                    rayon::join(|| eval_multilinear(f0, tail), || eval_multilinear(f1, tail))
                } else {
                    (eval_multilinear(f0, tail), eval_multilinear(f1, tail))
                }
            };
            f0 + (f1 - f0) * *x
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use p3_baby_bear::BabyBear;
    use p3_field::{PrimeCharacteristicRing, PrimeField64, extension::BinomialExtensionField};
    use proptest::prelude::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use super::*;
    use crate::poly::{coeffs::CoefficientList, hypercube::BinaryHypercube};

    type F = BabyBear;
    type EF4 = BinomialExtensionField<F, 4>;

    #[test]
    #[allow(clippy::redundant_clone)]
    fn test_parallel_clone() {
        let mut rng = StdRng::seed_from_u64(0);
        let evals =
            EvaluationsList::<F>::new((0..1 << 25).map(|_| F::from_u64(rng.random())).collect());
        let time = Instant::now();
        let seq_clone = evals.clone();
        println!("Sequential clone took: {:?}", time.elapsed());
        let time = Instant::now();
        let par_clone = evals.parallel_clone();
        println!("Parallel clone took: {:?}", time.elapsed());
        assert_eq!(seq_clone, par_clone);
    }

    #[test]
    fn test_new_evaluations_list() {
        let evals = vec![F::ZERO, F::ONE, F::ZERO, F::ONE];
        let evaluations_list = EvaluationsList::new(evals.clone());

        assert_eq!(evaluations_list.num_evals(), evals.len());
        assert_eq!(evaluations_list.num_variables(), 2);
        assert_eq!(evaluations_list.evals(), &evals);
    }

    #[test]
    #[should_panic]
    fn test_new_evaluations_list_invalid_length() {
        // Length is not a power of two, should panic
        let _ = EvaluationsList::new(vec![F::ONE, F::ZERO, F::ONE]);
    }

    #[test]
    fn test_indexing() {
        let evals = vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ];
        let evaluations_list = EvaluationsList::new(evals.clone());

        assert_eq!(evaluations_list[0], evals[0]);
        assert_eq!(evaluations_list[1], evals[1]);
        assert_eq!(evaluations_list[2], evals[2]);
        assert_eq!(evaluations_list[3], evals[3]);
    }

    #[test]
    #[should_panic]
    fn test_index_out_of_bounds() {
        let evals = vec![F::ZERO, F::ONE, F::ZERO, F::ONE];
        let evaluations_list = EvaluationsList::new(evals);

        let _ = evaluations_list[4]; // Index out of range, should panic
    }

    #[test]
    fn test_mutability_of_evals() {
        let mut evals = EvaluationsList::new(vec![F::ZERO, F::ONE, F::ZERO, F::ONE]);

        assert_eq!(evals.evals()[1], F::ONE);

        evals.evals_mut()[1] = F::from_u64(5);

        assert_eq!(evals.evals()[1], F::from_u64(5));
    }

    #[test]
    fn test_evaluate_on_hypercube_points() {
        let evaluations_vec = vec![F::ZERO, F::ONE, F::ZERO, F::ONE];
        let evals = EvaluationsList::new(evaluations_vec.clone());

        for i in BinaryHypercube::new(2) {
            assert_eq!(
                evaluations_vec[i.0],
                evals.evaluate(&MultilinearPoint::from_binary_hypercube_point(i, 2))
            );
        }
    }

    #[test]
    fn test_evaluate_edge_cases() {
        let e1 = F::from_u64(7);
        let e2 = F::from_u64(8);
        let e3 = F::from_u64(9);
        let e4 = F::from_u64(10);

        let evals = EvaluationsList::new(vec![e1, e2, e3, e4]);

        // Evaluating at a binary hypercube point should return the direct value
        assert_eq!(
            evals.evaluate(&MultilinearPoint(vec![F::ZERO, F::ZERO])),
            e1
        );
        assert_eq!(evals.evaluate(&MultilinearPoint(vec![F::ZERO, F::ONE])), e2);
        assert_eq!(evals.evaluate(&MultilinearPoint(vec![F::ONE, F::ZERO])), e3);
        assert_eq!(evals.evaluate(&MultilinearPoint(vec![F::ONE, F::ONE])), e4);
    }

    #[test]
    fn test_num_evals() {
        let evals = EvaluationsList::new(vec![F::ONE, F::ZERO, F::ONE, F::ZERO]);
        assert_eq!(evals.num_evals(), 4);
    }

    #[test]
    fn test_num_variables() {
        let evals = EvaluationsList::new(vec![F::ONE, F::ZERO, F::ONE, F::ZERO]);
        assert_eq!(evals.num_variables(), 2);
    }

    #[test]
    fn test_eval_extension_on_hypercube_points() {
        let evals = vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ];
        let eval_list = EvaluationsList::new(evals.clone());

        for i in BinaryHypercube::new(2) {
            assert_eq!(
                eval_list.evaluate(&MultilinearPoint::<F>::from_binary_hypercube_point(i, 2)),
                evals[i.0]
            );
        }
    }

    #[test]
    fn test_eval_extension_on_non_hypercube_points() {
        let evals = EvaluationsList::new(vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ]);

        let point = MultilinearPoint(vec![F::from_u64(2), F::from_u64(3)]);

        let result = evals.evaluate(&point);

        // Expected result using `eval_multilinear`
        let expected = eval_multilinear(evals.evals(), &point.0);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_eval_multilinear_1d() {
        let a = F::from_u64(5);
        let b = F::from_u64(10);
        let evals = vec![a, b];

        // Evaluate at midpoint `x = 1/2`
        let x = F::from_u64(1) / F::from_u64(2);
        let expected = a + (b - a) * x;

        assert_eq!(eval_multilinear(&evals, &[x]), expected);
    }

    #[test]
    fn test_eval_multilinear_2d() {
        let a = F::from_u64(1);
        let b = F::from_u64(2);
        let c = F::from_u64(3);
        let d = F::from_u64(4);

        // The evaluations are stored in lexicographic order for (x, y)
        // f(0,0) = a, f(0,1) = c, f(1,0) = b, f(1,1) = d
        let evals = vec![a, b, c, d];

        // Evaluate at `(x, y) = (1/2, 1/2)`
        let x = F::from_u64(1) / F::from_u64(2);
        let y = F::from_u64(1) / F::from_u64(2);

        // Interpolation formula:
        // f(x, y) = (1-x)(1-y) * f(0,0) + (1-x)y * f(0,1) + x(1-y) * f(1,0) + xy * f(1,1)
        let expected = (F::ONE - x) * (F::ONE - y) * a
            + (F::ONE - x) * y * c
            + x * (F::ONE - y) * b
            + x * y * d;

        assert_eq!(eval_multilinear(&evals, &[x, y]), expected);
    }

    #[test]
    fn test_eval_multilinear_3d() {
        let a = F::from_u64(1);
        let b = F::from_u64(2);
        let c = F::from_u64(3);
        let d = F::from_u64(4);
        let e = F::from_u64(5);
        let f = F::from_u64(6);
        let g = F::from_u64(7);
        let h = F::from_u64(8);

        // The evaluations are stored in lexicographic order for (x, y, z)
        // f(0,0,0) = a, f(0,0,1) = c, f(0,1,0) = b, f(0,1,1) = e
        // f(1,0,0) = d, f(1,0,1) = f, f(1,1,0) = g, f(1,1,1) = h
        let evals = vec![a, b, c, e, d, f, g, h];

        let x = F::from_u64(1) / F::from_u64(3);
        let y = F::from_u64(1) / F::from_u64(3);
        let z = F::from_u64(1) / F::from_u64(3);

        // Using trilinear interpolation formula:
        let expected = (F::ONE - x) * (F::ONE - y) * (F::ONE - z) * a
            + (F::ONE - x) * (F::ONE - y) * z * c
            + (F::ONE - x) * y * (F::ONE - z) * b
            + (F::ONE - x) * y * z * e
            + x * (F::ONE - y) * (F::ONE - z) * d
            + x * (F::ONE - y) * z * f
            + x * y * (F::ONE - z) * g
            + x * y * z * h;

        assert_eq!(eval_multilinear(&evals, &[x, y, z]), expected);
    }

    #[test]
    fn test_eval_multilinear_4d() {
        let a = F::from_u64(1);
        let b = F::from_u64(2);
        let c = F::from_u64(3);
        let d = F::from_u64(4);
        let e = F::from_u64(5);
        let f = F::from_u64(6);
        let g = F::from_u64(7);
        let h = F::from_u64(8);
        let i = F::from_u64(9);
        let j = F::from_u64(10);
        let k = F::from_u64(11);
        let l = F::from_u64(12);
        let m = F::from_u64(13);
        let n = F::from_u64(14);
        let o = F::from_u64(15);
        let p = F::from_u64(16);

        // Evaluations stored in lexicographic order for (x, y, z, w)
        let evals = vec![a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p];

        let x = F::from_u64(1) / F::from_u64(2);
        let y = F::from_u64(2) / F::from_u64(3);
        let z = F::from_u64(1) / F::from_u64(4);
        let w = F::from_u64(3) / F::from_u64(5);

        // Quadlinear interpolation formula
        let expected = (F::ONE - x) * (F::ONE - y) * (F::ONE - z) * (F::ONE - w) * a
            + (F::ONE - x) * (F::ONE - y) * (F::ONE - z) * w * b
            + (F::ONE - x) * (F::ONE - y) * z * (F::ONE - w) * c
            + (F::ONE - x) * (F::ONE - y) * z * w * d
            + (F::ONE - x) * y * (F::ONE - z) * (F::ONE - w) * e
            + (F::ONE - x) * y * (F::ONE - z) * w * f
            + (F::ONE - x) * y * z * (F::ONE - w) * g
            + (F::ONE - x) * y * z * w * h
            + x * (F::ONE - y) * (F::ONE - z) * (F::ONE - w) * i
            + x * (F::ONE - y) * (F::ONE - z) * w * j
            + x * (F::ONE - y) * z * (F::ONE - w) * k
            + x * (F::ONE - y) * z * w * l
            + x * y * (F::ONE - z) * (F::ONE - w) * m
            + x * y * (F::ONE - z) * w * n
            + x * y * z * (F::ONE - w) * o
            + x * y * z * w * p;

        // Validate against the function output
        assert_eq!(eval_multilinear(&evals, &[x, y, z, w]), expected);
    }

    #[test]
    fn test_num_variables_base_storage() {
        // Polynomial with 2 variables: 4 evaluation points
        let values = vec![F::ONE, F::ZERO, F::ONE, F::ZERO];
        let evals = EvaluationsList::new(values);

        // Wrap in EvaluationStorage::Base
        let storage = EvaluationStorage::<F, EF4>::Base(evals);

        // 4 points = 2 variables (log2(4) = 2)
        assert_eq!(storage.num_variables(), 2);
    }

    #[test]
    fn test_num_variables_extension_storage() {
        // Polynomial with 3 variables: 8 evaluation points
        let values = vec![
            EF4::ONE,
            EF4::ZERO,
            EF4::ONE,
            EF4::ZERO,
            EF4::ONE,
            EF4::ZERO,
            EF4::ONE,
            EF4::ZERO,
        ];
        let evals = EvaluationsList::new(values);

        // Wrap in EvaluationStorage::Extension
        let storage = EvaluationStorage::<F, EF4>::Extension(evals);

        // 8 points = 3 variables (log2(8) = 3)
        assert_eq!(storage.num_variables(), 3);
    }

    proptest! {
        #[test]
        fn prop_eval_multilinear_equiv_between_f_and_ef4(
            values in prop::collection::vec(0u64..100, 8),
            x0 in 0u64..100,
            x1 in 0u64..100,
            x2 in 0u64..100,
        ) {
            // Base field evaluations
            let coeffs_f: Vec<F> = values.iter().copied().map(F::from_u64).collect();
            let poly_f = EvaluationsList::new(coeffs_f);

            // Lift to extension field EF4
            let coeffs_ef: Vec<EF4> = values.iter().copied().map(EF4::from_u64).collect();
            let poly_ef = EvaluationsList::new(coeffs_ef);

            // Evaluation point in EF4
            let point_ef = MultilinearPoint(vec![
                EF4::from_u64(x0),
                EF4::from_u64(x1),
                EF4::from_u64(x2),
            ]);

            // Evaluate using both base and extension representations
            let eval_f = poly_f.evaluate(&point_ef);
            let eval_ef = poly_ef.evaluate(&point_ef);

            prop_assert_eq!(eval_f, eval_ef);
        }
    }

    #[test]
    fn test_multilinear_eval_two_vars() {
        // Define a simple 2-variable multilinear polynomial:
        //
        // Variables: X₁, X₂
        // Coefficients ordered in lexicographic order: (X₁, X₂)
        //
        // - coeffs[0] → constant term
        // - coeffs[1] → X₂ term
        // - coeffs[2] → X₁ term
        // - coeffs[3] → X₁·X₂ term
        //
        // Thus, the polynomial is:
        //
        //   f(X₁, X₂) = c0 + c1·X₂ + c2·X₁ + c3·X₁·X₂
        //
        // where:
        let c0 = F::from_u64(5); // constant
        let c1 = F::from_u64(6); // X₂ coefficient
        let c2 = F::from_u64(7); // X₁ coefficient
        let c3 = F::from_u64(8); // X₁·X₂ coefficient
        //
        // So concretely:
        //
        //   f(X₁, X₂) = 5 + 6·X₂ + 7·X₁ + 8·X₁·X₂
        let coeffs = CoefficientList::new(vec![c0, c1, c2, c3]);

        // Convert coefficients to evaluations via the wavelet transform
        let evals = coeffs.to_evaluations();

        // Choose evaluation point:
        //
        // Let's pick (x₁, x₂) = (2, 1)
        let x1 = F::from_u64(2);
        let x2 = F::from_u64(1);
        let coords = MultilinearPoint(vec![x1, x2]);

        // Manually compute the expected value step-by-step:
        //
        // Reminder:
        //   f(X₁, X₂) = 5 + 6·X₂ + 7·X₁ + 8·X₁·X₂
        //
        // Substituting (X₁, X₂):
        let expected = c0 + c1 * x2 + c2 * x1 + c3 * x1 * x2;

        // Now evaluate using the function under test
        let result = evals.evaluate(&coords);

        // Check that it matches the manual computation
        assert_eq!(result, expected);
    }

    #[test]
    fn test_evaluate_3_variables() {
        // Define a multilinear polynomial in 3 variables: X₀, X₁, X₂
        //
        // Coefficients ordered lex in index order:
        //
        // - coeffs[0] → constant term
        // - coeffs[1] → X₂
        // - coeffs[2] → X₁
        // - coeffs[3] → X₁·X₂
        // - coeffs[4] → X₀
        // - coeffs[5] → X₀·X₂
        // - coeffs[6] → X₀·X₁
        // - coeffs[7] → X₀·X₁·X₂
        //
        // Thus:
        //    f(X₀,X₁,X₂) = c0 + c1·X₂ + c2·X₁ + c3·X₁·X₂
        //                + c4·X₀ + c5·X₀·X₂ + c6·X₀·X₁ + c7·X₀·X₁·X₂
        let c0 = F::from_u64(1);
        let c1 = F::from_u64(2);
        let c2 = F::from_u64(3);
        let c3 = F::from_u64(4);
        let c4 = F::from_u64(5);
        let c5 = F::from_u64(6);
        let c6 = F::from_u64(7);
        let c7 = F::from_u64(8);

        let coeffs = CoefficientList::new(vec![c0, c1, c2, c3, c4, c5, c6, c7]);
        let evals = coeffs.to_evaluations();

        // Pick point: (x₀,x₁,x₂) = (2, 3, 4)
        let x0 = F::from_u64(2);
        let x1 = F::from_u64(3);
        let x2 = F::from_u64(4);

        let point = MultilinearPoint(vec![x0, x1, x2]);

        // Manually compute:
        //
        // expected = 1
        //          + 2·4
        //          + 3·3
        //          + 4·3·4
        //          + 5·2
        //          + 6·2·4
        //          + 7·2·3
        //          + 8·2·3·4
        let expected = c0
            + c1 * x2
            + c2 * x1
            + c3 * x1 * x2
            + c4 * x0
            + c5 * x0 * x2
            + c6 * x0 * x1
            + c7 * x0 * x1 * x2;

        let result = evals.evaluate(&point);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_eval_extension_3_variables() {
        // Define a multilinear polynomial in 3 variables: X₀, X₁, X₂
        //
        // Coefficients ordered lex in index order:
        //
        // - coeffs[0] → constant term
        // - coeffs[1] → X₂ term
        // - coeffs[2] → X₁ term
        // - coeffs[3] → X₁·X₂ term
        // - coeffs[4] → X₀ term
        // - coeffs[5] → X₀·X₂ term
        // - coeffs[6] → X₀·X₁ term
        // - coeffs[7] → X₀·X₁·X₂ term
        //
        // Thus:
        //    f(X₀,X₁,X₂) = c0 + c1·X₂ + c2·X₁ + c3·X₁·X₂
        //                + c4·X₀ + c5·X₀·X₂ + c6·X₀·X₁ + c7·X₀·X₁·X₂
        let c0 = F::from_u64(1);
        let c1 = F::from_u64(2);
        let c2 = F::from_u64(3);
        let c3 = F::from_u64(4);
        let c4 = F::from_u64(5);
        let c5 = F::from_u64(6);
        let c6 = F::from_u64(7);
        let c7 = F::from_u64(8);

        let coeffs = CoefficientList::new(vec![c0, c1, c2, c3, c4, c5, c6, c7]);
        let evals = coeffs.to_evaluations();

        // Choose evaluation point: (x₀, x₁, x₂) = (2, 3, 4)
        //
        // Here we lift into the extension field EF4
        let x0 = EF4::from_u64(2);
        let x1 = EF4::from_u64(3);
        let x2 = EF4::from_u64(4);

        let point = MultilinearPoint(vec![x0, x1, x2]);

        // Manually compute expected value
        //
        // Substituting (X₀,X₁,X₂) = (2,3,4) into:
        //
        //   f(X₀,X₁,X₂) = 1
        //               + 2·4
        //               + 3·3
        //               + 4·3·4
        //               + 5·2
        //               + 6·2·4
        //               + 7·2·3
        //               + 8·2·3·4
        //
        // and lifting each constant into EF4 for correct typing
        let expected = EF4::from(c0)
            + EF4::from(c1) * x2
            + EF4::from(c2) * x1
            + EF4::from(c3) * x1 * x2
            + EF4::from(c4) * x0
            + EF4::from(c5) * x0 * x2
            + EF4::from(c6) * x0 * x1
            + EF4::from(c7) * x0 * x1 * x2;

        // Evaluate via `evaluate` method
        let result = evals.evaluate(&point);

        // Verify that result matches manual computation
        assert_eq!(result, expected);
    }

    #[test]
    fn test_folding_and_evaluation() {
        // Set number of Boolean input variables n = 10.
        let num_variables = 10;

        // Build a multilinear polynomial f(x) = x with coefficients 0, 1, ..., 1023 in F
        let coeffs = (0..(1 << num_variables)).map(F::from_u64).collect();

        // Wrap into CoefficientList to access polynomial logic
        let coeffs_list = CoefficientList::new(coeffs);

        // Convert to EvaluationsList using a wavelet transform
        let evals_list: EvaluationsList<F> = coeffs_list.clone().to_evaluations();

        // Define a fixed evaluation point in F^n: [0, 35, 70, ..., 35*(n-1)]
        let randomness: Vec<_> = (0..num_variables)
            .map(|i| F::from_u64(35 * i as u64))
            .collect();

        // Try folding at every possible prefix of the randomness vector: k = 0 to n-1
        for k in 0..num_variables {
            // Use the first k values as the fold coordinates (we will substitute those)
            let fold_part = randomness[0..k].to_vec();

            // The remaining coordinates are used as the evaluation input into the folded poly
            let eval_part = randomness[k..randomness.len()].to_vec();

            // Convert to a MultilinearPoint (in EF) for folding
            let fold_random = MultilinearPoint(fold_part.clone());

            // Reconstruct the full point (x₀, ..., xₙ₋₁) = [eval_part || fold_part]
            // Used to evaluate the original uncompressed polynomial
            let eval_point = MultilinearPoint([eval_part.clone(), fold_part].concat());

            // Fold the evaluation list over the last `k` variables
            let folded_evals = evals_list.fold(&fold_random);

            // Verify that the number of variables has been folded correctly
            assert_eq!(folded_evals.num_variables(), num_variables - k);

            // Fold the coefficients list over the last `k` variables
            let folded_coeffs = coeffs_list.fold(&fold_random);

            // Verify that the number of variables has been folded correctly
            assert_eq!(folded_coeffs.num_variables(), num_variables - k);

            // Verify correctness:
            // folded(e) == original([e, r]) for all k
            assert_eq!(
                folded_evals.evaluate(&MultilinearPoint(eval_part.clone())),
                evals_list.evaluate(&eval_point)
            );

            // Compare with the coefficient list equivalent
            assert_eq!(
                folded_coeffs.evaluate(&MultilinearPoint(eval_part)),
                evals_list.evaluate(&eval_point)
            );
        }
    }

    #[test]
    fn test_fold_with_extension_one_var() {
        // Define a 2-variable polynomial:
        // f(X₀, X₁) = 1 + 2·X₁ + 3·X₀ + 4·X₀·X₁
        let coeffs = vec![
            F::from_u64(1), // constant
            F::from_u64(2), // X₁
            F::from_u64(3), // X₀
            F::from_u64(4), // X₀·X₁
        ];
        let poly = CoefficientList::new(coeffs);

        // Convert coefficients into an EvaluationsList (for testing the fold on evals)
        let evals_list: EvaluationsList<F> = poly.clone().to_evaluations();

        // We fold over the last variable (X₁) by setting X₁ = 5 in EF4
        let r1 = EF4::from_u64(5);

        // Perform the fold: f(X₀, 5) becomes a new function g(X₀)
        let folded = evals_list.fold(&MultilinearPoint(vec![r1]));

        // For 10 test points x₀ = 0, 1, ..., 9
        for x0_f in 0..10 {
            // Lift to EF4 for extension-field evaluation
            let x0 = EF4::from_u64(x0_f);

            // Construct the full point (x₀, X₁ = 5)
            let full_point = MultilinearPoint(vec![x0, r1]);

            // Construct folded point (x₀)
            let folded_point = MultilinearPoint(vec![x0]);

            // Evaluate original poly at (x₀, 5)
            let expected = poly.evaluate(&full_point);

            // Evaluate folded poly at x₀
            let actual = folded.evaluate(&folded_point);

            // Ensure the results agree
            assert_eq!(expected, actual);
        }
    }

    #[test]
    fn test_eval_eq() {
        let c0 = F::from_u64(11);
        let c1 = F::from_u64(22);
        let evals = [c0, c1];
        let poly = EvaluationsList::eval_eq(&evals);
        assert_eq!(poly.evals()[0], (F::ONE - c0) * (F::ONE - c1));
        assert_eq!(poly.evals()[1], (F::ONE - c0) * c1);
        assert_eq!(poly.evals()[2], c0 * (F::ONE - c1));
        assert_eq!(poly.evals()[3], c0 * c1);
    }

    proptest! {
        #[test]
        fn prop_eval_eq_matches_naive_for_eval_list(
            // number of variables (keep small to avoid blowup)
            n in 1usize..5,
             // always at least 5 elements
            evals_raw in prop::collection::vec(0u64..F::ORDER_U64, 5),
        ) {
            use crate::utils::eval_eq;

            // Slice out exactly n elements, guaranteed present
            let evals: Vec<F> = evals_raw[..n].iter().map(|&x| F::from_u64(x)).collect();

            // Allocate output buffer of size 2^n
            let mut out = vec![F::ZERO; 1 << n];

            // Run eval_eq with scalar = 1
            eval_eq::<F, F, false>(&evals, &mut out, F::ONE);

            // Naively compute expected values for each binary assignment
            let mut expected = vec![F::ZERO; 1 << n];
            for (i, e) in expected.iter_mut().enumerate().take(1 << n) {
                let mut weight = F::ONE;
                for (j, &val) in evals.iter().enumerate() {
                    let bit = (i >> (n - 1 - j)) & 1;
                    if bit == 1 {
                        weight *= val;
                    } else {
                        weight *= F::ONE - val;
                    }
                }
                *e = weight;
            }

            prop_assert_eq!(out, expected);
        }
    }

    #[test]
    fn test_scale_empty() {
        let list = EvaluationsList::<F> {
            evals: vec![],
            num_variables: 0,
        };

        let factor = EF4::from_u64(3);
        let result = list.scale(factor);

        assert_eq!(result.evals, vec![]);
        assert_eq!(result.num_variables, 0);
    }

    #[test]
    fn test_scale_by_zero() {
        let list = EvaluationsList::<F> {
            evals: vec![F::from_u64(1), F::from_u64(2), F::from_u64(3)],
            num_variables: 2,
        };

        let result = list.scale(EF4::ZERO);

        assert_eq!(result.evals.len(), 3);
        for val in result.evals {
            assert_eq!(val, EF4::ZERO);
        }
        assert_eq!(result.num_variables, 2);
    }

    #[test]
    fn test_scale_by_one() {
        let list = EvaluationsList::<F> {
            evals: vec![F::from_u64(10), F::from_u64(20), F::from_u64(30)],
            num_variables: 1,
        };

        let result = list.scale(EF4::ONE);

        assert_eq!(result.evals.len(), 3);
        for (i, val) in result.evals.iter().enumerate() {
            assert_eq!(*val, EF4::from(list.evals[i]));
        }
        assert_eq!(result.num_variables, 1);
    }

    #[test]
    fn test_scale_by_nontrivial_scalar() {
        let list = EvaluationsList::<F> {
            evals: vec![F::from_u64(2), F::from_u64(5), F::from_u64(7)],
            num_variables: 2,
        };

        let factor = EF4::from_u64(9);
        let result = list.scale(factor);

        assert_eq!(result.evals.len(), 3);
        for (i, val) in result.evals.iter().enumerate() {
            assert_eq!(*val, EF4::from(list.evals[i]) * factor);
        }
        assert_eq!(result.num_variables, 2);
    }

    #[test]
    fn test_scale_preserves_length_and_variables() {
        let list = EvaluationsList::<F> {
            evals: vec![
                F::from_u64(0),
                F::from_u64(1),
                F::from_u64(2),
                F::from_u64(3),
            ],
            num_variables: 2,
        };

        let result = list.scale(EF4::from_u64(7));

        assert_eq!(result.evals.len(), 4);
        assert_eq!(result.num_variables, 2);
    }
}
