use std::ops::Index;

use p3_field::{ExtensionField, Field};

use super::{lagrange_iterator::LagrangePolynomialIterator, multilinear::MultilinearPoint};

/// A wrapper enum that holds evaluation data for a multilinear polynomial,
/// either over the base field `F` or an extension field `EF`.
///
/// This abstraction allows operating generically on both base and extension
/// field evaluations, as used in sumcheck protocols and other polynomial
/// computations.
#[derive(Debug, Clone)]
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
}

/// Represents a multilinear polynomial `f` in `num_variables` unknowns, stored via its evaluations
/// over the hypercube `{0,1}^{num_variables}`.
///
/// The vector `evals` contains function evaluations at **lexicographically ordered** points.
#[derive(Debug, Clone)]
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

    /// Evaluates the polynomial at a given multilinear point.
    ///
    /// - If `point` belongs to the binary hypercube `{0,1}^n`, we directly return the precomputed
    ///   evaluation.
    /// - Otherwise, we **reconstruct** the evaluation using Lagrange interpolation.
    ///
    /// Mathematical definition:
    /// Given evaluations `f(x)` stored in `evals`, we compute:
    ///
    /// ```ignore
    /// f(p) = Σ_{x ∈ {0,1}^n} eq(x, p) * f(x)
    /// ```
    ///
    /// where `eq(x, p)` is the Lagrange basis polynomial.
    pub fn evaluate(&self, point: &MultilinearPoint<F>) -> F {
        if let Some(binary_index) = point.to_hypercube() {
            return self.evals[binary_index.0];
        }

        self.evals
            .iter()
            .zip(LagrangePolynomialIterator::from(point))
            .map(|(eval, (_, lag))| *eval * lag)
            .sum()
    }

    /// Returns an immutable reference to the evaluations vector.
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
    pub fn num_evals(&self) -> usize {
        self.evals.len()
    }

    /// Returns the number of variables in the multilinear polynomial.
    pub const fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// Evaluates the multilinear polynomial at `point ∈ [0,1]^n`.
    ///
    /// - If `point ∈ {0,1}^n`, returns the precomputed evaluation `f(point)`.
    /// - Otherwise, computes `f(point) = ∑_{x ∈ {0,1}^n} eq(x, point) * f(x)`, where `eq(x, point)
    ///   = ∏_{i=1}^{n} (1 - p_i + 2 p_i x_i)`.
    /// - Uses fast multilinear interpolation for efficiency.
    pub fn eval_extension<EF>(&self, point: &MultilinearPoint<EF>) -> EF
    where
        EF: ExtensionField<F>,
    {
        if let Some(point) = point.to_hypercube() {
            return self.evals[point.0].into();
        }
        eval_multilinear(&self.evals, &point.0)
    }
}

impl<F> Index<usize> for EvaluationsList<F> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        &self.evals[index]
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
    use p3_baby_bear::BabyBear;
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};
    use proptest::prelude::*;

    use super::*;
    use crate::poly::hypercube::BinaryHypercube;

    type F = BabyBear;
    type EF4 = BinomialExtensionField<F, 4>;

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
    fn test_evaluate_on_non_hypercube_points() {
        let evals = EvaluationsList::new(vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ]);

        let point = MultilinearPoint(vec![F::from_u64(2), F::from_u64(3)]);

        let result = evals.evaluate(&point);

        // The result should be computed using Lagrange interpolation.
        let expected = LagrangePolynomialIterator::from(&point)
            .map(|(b, lag)| lag * evals[b.0])
            .sum();

        assert_eq!(result, expected);
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
                eval_list.eval_extension(&MultilinearPoint::<F>::from_binary_hypercube_point(i, 2)),
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

        let result = evals.eval_extension(&point);

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
            let eval_f = poly_f.eval_extension(&point_ef);
            let eval_ef = poly_ef.eval_extension(&point_ef);

            prop_assert_eq!(eval_f, eval_ef);
        }
    }
}
