use super::{multilinear::MultilinearPoint, sequential_lag_poly::LagrangePolynomialIterator};
use p3_field::Field;
use std::ops::Index;

/// Represents a multilinear polynomial `f` in `num_variables` unknowns, stored via its evaluations
/// over the hypercube `{0,1}^{num_variables}`.
///
/// The vector `evals` contains function evaluations at **lexicographically ordered** points.
#[derive(Debug)]
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
        assert!(len.is_power_of_two(), "Evaluation list length must be a power of two.");

        Self { evals, num_variables: len.ilog2() as usize }
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
    #[allow(clippy::missing_const_for_fn)]
    pub fn evals(&self) -> &[F] {
        &self.evals
    }

    /// Returns a mutable reference to the evaluations vector.
    #[allow(clippy::missing_const_for_fn)]
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
}

impl<F> Index<usize> for EvaluationsList<F> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        &self.evals[index]
    }
}

#[cfg(test)]
#[allow(clippy::should_panic_without_expect)]
mod tests {
    use super::*;
    use crate::poly::hypercube::BinaryHypercube;
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    #[test]
    fn test_new_evaluations_list() {
        let evals = vec![BabyBear::ZERO, BabyBear::ONE, BabyBear::ZERO, BabyBear::ONE];
        let evaluations_list = EvaluationsList::new(evals.clone());

        assert_eq!(evaluations_list.num_evals(), evals.len());
        assert_eq!(evaluations_list.num_variables(), 2);
        assert_eq!(evaluations_list.evals(), &evals);
    }

    #[test]
    #[should_panic]
    fn test_new_evaluations_list_invalid_length() {
        // Length is not a power of two, should panic
        let _ = EvaluationsList::new(vec![BabyBear::ONE, BabyBear::ZERO, BabyBear::ONE]);
    }

    #[test]
    fn test_indexing() {
        let evals = vec![
            BabyBear::from_u64(1),
            BabyBear::from_u64(2),
            BabyBear::from_u64(3),
            BabyBear::from_u64(4),
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
        let evals = vec![BabyBear::ZERO, BabyBear::ONE, BabyBear::ZERO, BabyBear::ONE];
        let evaluations_list = EvaluationsList::new(evals);

        let _ = evaluations_list[4]; // Index out of range, should panic
    }

    #[test]
    fn test_mutability_of_evals() {
        let mut evals = EvaluationsList::new(vec![
            BabyBear::ZERO,
            BabyBear::ONE,
            BabyBear::ZERO,
            BabyBear::ONE,
        ]);

        assert_eq!(evals.evals()[1], BabyBear::ONE);

        evals.evals_mut()[1] = BabyBear::from_u64(5);

        assert_eq!(evals.evals()[1], BabyBear::from_u64(5));
    }

    #[test]
    fn test_evaluate_on_hypercube_points() {
        let evaluations_vec = vec![BabyBear::ZERO, BabyBear::ONE, BabyBear::ZERO, BabyBear::ONE];
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
            BabyBear::from_u64(1),
            BabyBear::from_u64(2),
            BabyBear::from_u64(3),
            BabyBear::from_u64(4),
        ]);

        let point = MultilinearPoint(vec![BabyBear::from_u64(2), BabyBear::from_u64(3)]);

        let result = evals.evaluate(&point);

        // The result should be computed using Lagrange interpolation.
        let expected =
            LagrangePolynomialIterator::from(&point).map(|(b, lag)| lag * evals[b.0]).sum();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_evaluate_edge_cases() {
        let e1 = BabyBear::from_u64(7);
        let e2 = BabyBear::from_u64(8);
        let e3 = BabyBear::from_u64(9);
        let e4 = BabyBear::from_u64(10);

        let evals = EvaluationsList::new(vec![e1, e2, e3, e4]);

        // Evaluating at a binary hypercube point should return the direct value
        assert_eq!(evals.evaluate(&MultilinearPoint(vec![BabyBear::ZERO, BabyBear::ZERO])), e1);
        assert_eq!(evals.evaluate(&MultilinearPoint(vec![BabyBear::ZERO, BabyBear::ONE])), e2);
        assert_eq!(evals.evaluate(&MultilinearPoint(vec![BabyBear::ONE, BabyBear::ZERO])), e3);
        assert_eq!(evals.evaluate(&MultilinearPoint(vec![BabyBear::ONE, BabyBear::ONE])), e4);
    }

    #[test]
    fn test_num_evals() {
        let evals = EvaluationsList::new(vec![
            BabyBear::ONE,
            BabyBear::ZERO,
            BabyBear::ONE,
            BabyBear::ZERO,
        ]);
        assert_eq!(evals.num_evals(), 4);
    }

    #[test]
    fn test_num_variables() {
        let evals = EvaluationsList::new(vec![
            BabyBear::ONE,
            BabyBear::ZERO,
            BabyBear::ONE,
            BabyBear::ZERO,
        ]);
        assert_eq!(evals.num_variables(), 2);
    }
}
