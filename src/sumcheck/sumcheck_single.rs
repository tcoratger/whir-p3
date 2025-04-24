use p3_field::{ExtensionField, Field, PrimeField64, TwoAdicField};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::sumcheck_polynomial::SumcheckPolynomial;
use crate::{
    fiat_shamir::{errors::ProofResult, pow::traits::PowStrategy, prover::ProverState},
    poly::{coeffs::CoefficientList, evals::EvaluationsList, multilinear::MultilinearPoint},
    utils::eval_eq,
    whir::statement::Statement,
};

/// A wrapper enum that holds evaluation data for a multilinear polynomial,
/// either over the base field `F` or an extension field `EF`.
///
/// This abstraction allows operating generically on both base and extension
/// field evaluations, as used in sumcheck protocols and other polynomial
/// computations.
#[derive(Debug, Clone)]
enum EvaluationStorage<F, EF> {
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
    const fn num_variables(&self) -> usize {
        match self {
            Self::Base(evals) => evals.num_variables(),
            Self::Extension(evals) => evals.num_variables(),
        }
    }
}

/// Implements the single-round sumcheck protocol for verifying a multilinear polynomial evaluation.
///
/// This struct is responsible for:
/// - Transforming a polynomial from coefficient representation into evaluation form.
/// - Constructing and evaluating weighted constraints.
/// - Computing the sumcheck polynomial, which is a quadratic polynomial in a single variable.
///
/// Given a multilinear polynomial `p(X1, ..., Xn)`, the sumcheck polynomial is computed as:
///
/// \begin{equation}
/// h(X) = \sum_b p(b, X) \cdot w(b, X)
/// \end{equation}
///
/// where:
/// - `b` ranges over evaluation points in `{0,1,2}^k` (with `k=1` in this implementation).
/// - `w(b, X)` represents generic weights applied to `p(b, X)`.
/// - The result `h(X)` is a quadratic polynomial in `X`.
///
/// The sumcheck protocol ensures that the claimed sum is correct.
#[derive(Debug, Clone)]
pub struct SumcheckSingle<F, EF> {
    /// Evaluations of the polynomial `p(X)`.
    evaluation_of_p: EvaluationStorage<F, EF>,
    /// Evaluations of the equality polynomial used for enforcing constraints.
    weights: EvaluationsList<EF>,
    /// Accumulated sum incorporating equality constraints.
    sum: EF,
    /// Marker for phantom type parameter `F`.
    phantom: std::marker::PhantomData<F>,
}

impl<F, EF> SumcheckSingle<F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    /// Constructs a new `SumcheckSingle` instance from polynomial coefficients in base field.
    ///
    /// This function:
    /// - Converts `coeffs` into evaluation form.
    /// - Initializes an empty constraint table.
    /// - Applies weighted constraints if provided.
    ///
    /// The provided `Statement` encodes constraints that contribute to the final sumcheck equation.
    pub fn from_base_coeffs(
        coeffs: CoefficientList<F>,
        statement: &Statement<EF>,
        combination_randomness: EF,
    ) -> Self {
        let (weights, sum) = statement.combine(combination_randomness);
        Self {
            evaluation_of_p: EvaluationStorage::Base(coeffs.into()),
            weights,
            sum,
            phantom: std::marker::PhantomData,
        }
    }

    /// Constructs a new `SumcheckSingle` instance from polynomial coefficients in extension field.
    ///
    /// This function:
    /// - Converts `coeffs` into evaluation form.
    /// - Initializes an empty constraint table.
    /// - Applies weighted constraints if provided.
    ///
    /// The provided `Statement` encodes constraints that contribute to the final sumcheck equation.
    pub fn from_extension_coeffs(
        coeffs: CoefficientList<EF>,
        statement: &Statement<EF>,
        combination_randomness: EF,
    ) -> Self {
        let (weights, sum) = statement.combine(combination_randomness);
        Self {
            evaluation_of_p: EvaluationStorage::Extension(coeffs.into()),
            weights,
            sum,
            phantom: std::marker::PhantomData,
        }
    }

    /// Returns the number of variables in the polynomial.
    pub const fn num_variables(&self) -> usize {
        self.evaluation_of_p.num_variables()
    }

    /// Adds new weighted constraints to the polynomial.
    ///
    /// This function updates the weight evaluations and sum by incorporating new constraints.
    ///
    /// Given points `z_i`, weights `ε_i`, and evaluation values `f(z_i)`, it updates:
    ///
    /// \begin{equation}
    /// w(X) = w(X) + \sum \epsilon_i \cdot w_{z_i}(X)
    /// \end{equation}
    ///
    /// and updates the sum as:
    ///
    /// \begin{equation}
    /// S = S + \sum \epsilon_i \cdot f(z_i)
    /// \end{equation}
    ///
    /// where `w_{z_i}(X)` represents the constraint encoding at point `z_i`.
    pub fn add_new_equality(
        &mut self,
        points: &[MultilinearPoint<EF>],
        evaluations: &[EF],
        combination_randomness: &[EF],
    ) {
        assert_eq!(combination_randomness.len(), points.len());
        assert_eq!(combination_randomness.len(), evaluations.len());

        // Accumulate the sum while applying all constraints simultaneously
        points
            .iter()
            .zip(combination_randomness.iter().zip(evaluations.iter()))
            .for_each(|(point, (&rand, &eval))| {
                eval_eq(&point.0, self.weights.evals_mut(), rand);
                self.sum += rand * eval;
            });
    }

    /// Computes the sumcheck polynomial `h(X)`, which is quadratic.
    ///
    /// The sumcheck polynomial is given by:
    ///
    /// \begin{equation}
    /// h(X) = \sum_b p(b, X) \cdot w(b, X)
    /// \end{equation}
    ///
    /// where:
    /// - `b` represents points in `{0,1,2}^1`.
    /// - `w(b, X)` are the generic weights applied to `p(b, X)`.
    /// - `h(X)` is a quadratic polynomial.
    pub fn compute_sumcheck_polynomial(&self) -> SumcheckPolynomial<EF> {
        assert!(self.num_variables() >= 1);

        #[cfg(feature = "parallel")]
        let (c0, c2) = match &self.evaluation_of_p {
            EvaluationStorage::Base(evals_f) => evals_f
                .evals()
                .par_chunks_exact(2)
                .zip(self.weights.evals().par_chunks_exact(2))
                .map(|(p_at, eq_at)| {
                    // Convert evaluations to coefficients for the linear fns p and eq.
                    let (p_0, p_1) = (p_at[0], p_at[1] - p_at[0]);
                    let (eq_0, eq_1) = (eq_at[0], eq_at[1] - eq_at[0]);

                    (eq_0 * p_0, eq_1 * p_1)
                })
                .reduce(
                    || (EF::ZERO, EF::ZERO),
                    |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
                ),
            EvaluationStorage::Extension(evals_ef) => evals_ef
                .evals()
                .par_chunks_exact(2)
                .zip(self.weights.evals().par_chunks_exact(2))
                .map(|(p_at, eq_at)| {
                    // Convert evaluations to coefficients for the linear fns p and eq.
                    let (p_0, p_1) = (p_at[0], p_at[1] - p_at[0]);
                    let (eq_0, eq_1) = (eq_at[0], eq_at[1] - eq_at[0]);

                    (p_0 * eq_0, p_1 * eq_1)
                })
                .reduce(
                    || (EF::ZERO, EF::ZERO),
                    |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
                ),
        };

        #[cfg(not(feature = "parallel"))]
        let (c0, c2) = match &self.evaluation_of_p {
            EvaluationStorage::Base(evals_f) => evals_f
                .evals()
                .chunks_exact(2)
                .zip(self.weights.evals().chunks_exact(2))
                .map(|(p_at, eq_at)| {
                    // Convert evaluations to coefficients for the linear fns p and eq.
                    let (p_0, p_1) = (p_at[0], p_at[1] - p_at[0]);
                    let (eq_0, eq_1) = (eq_at[0], eq_at[1] - eq_at[0]);

                    (eq_0 * p_0, eq_1 * p_1)
                })
                .fold((EF::ZERO, EF::ZERO), |(a0, a2), (b0, b2)| {
                    (a0 + b0, a2 + b2)
                }),

            EvaluationStorage::Extension(evals_ef) => evals_ef
                .evals()
                .chunks_exact(2)
                .zip(self.weights.evals().chunks_exact(2))
                .map(|(p_at, eq_at)| {
                    // Convert evaluations to coefficients for the linear fns p and eq.
                    let (p_0, p_1) = (p_at[0], p_at[1] - p_at[0]);
                    let (eq_0, eq_1) = (eq_at[0], eq_at[1] - eq_at[0]);

                    (p_0 * eq_0, p_1 * eq_1)
                })
                .fold((EF::ZERO, EF::ZERO), |(a0, a2), (b0, b2)| {
                    (a0 + b0, a2 + b2)
                }),
        };

        // Compute the middle coefficient using sum rule: sum = 2 * c0 + c1 + c2
        let c1 = self.sum - c0.double() - c2;

        // Evaluate the quadratic polynomial at 0, 1, 2
        let eval_0 = c0;
        let eval_1 = c0 + c1 + c2;
        let eval_2 = eval_1 + c1 + c2 + c2.double();

        SumcheckPolynomial::new(vec![eval_0, eval_1, eval_2], 1)
    }

    /// Implements `folding_factor` rounds of the sumcheck protocol.
    ///
    /// The sumcheck protocol progressively reduces the number of variables in a multilinear
    /// polynomial. At each step, a quadratic polynomial is derived and verified.
    ///
    /// Given a polynomial \( p(X_1, \dots, X_n) \), this function iteratively applies the
    /// transformation:
    ///
    /// \begin{equation}
    /// h(X) = \sum_b p(b, X) \cdot w(b, X)
    /// \end{equation}
    ///
    /// where:
    /// - \( b \) are points in \{0,1,2\}.
    /// - \( w(b, X) \) represents generic weights applied to \( p(b, X) \).
    /// - \( h(X) \) is a quadratic polynomial in \( X \).
    ///
    /// This function:
    /// - Samples random values to progressively reduce the polynomial.
    /// - Applies proof-of-work grinding if required.
    /// - Returns the sampled folding randomness values used in each reduction step.
    pub fn compute_sumcheck_polynomials<S>(
        &mut self,
        prover_state: &mut ProverState<EF, F>,
        folding_factor: usize,
        pow_bits: f64,
    ) -> ProofResult<MultilinearPoint<EF>>
    where
        F: PrimeField64 + TwoAdicField,
        EF: ExtensionField<F> + TwoAdicField,
        S: PowStrategy,
    {
        let mut res = Vec::with_capacity(folding_factor);

        for _ in 0..folding_factor {
            let sumcheck_poly = self.compute_sumcheck_polynomial();
            prover_state.add_scalars(sumcheck_poly.evaluations())?;
            let [folding_randomness]: [EF; 1] = prover_state.challenge_scalars()?;
            res.push(folding_randomness);

            // Do PoW if needed
            if pow_bits > 0. {
                prover_state.challenge_pow::<S>(pow_bits)?;
            }

            self.compress(EF::ONE, &folding_randomness.into(), &sumcheck_poly);
        }

        res.reverse();
        Ok(MultilinearPoint(res))
    }

    /// Compresses the polynomial and weight evaluations by reducing the number of variables.
    ///
    /// Given a multilinear polynomial `p(X1, ..., Xn)`, this function eliminates `X1` using the
    /// folding randomness `r`:
    /// \begin{equation}
    ///     p'(X_2, ..., X_n) = (p(1, X_2, ..., X_n) - p(0, X_2, ..., X_n)) \cdot r
    ///     + p(0, X_2, ...,X_n)
    /// \end{equation}
    ///
    /// The same transformation applies to the weights `w(X)`, and the sum is updated as:
    ///
    /// \begin{equation}
    ///     S' = \rho \cdot h(r)
    /// \end{equation}
    ///
    /// where `h(r)` is the sumcheck polynomial evaluated at `r`, and `\rho` is
    /// `combination_randomness`.
    ///
    /// # Effects
    /// - Shrinks `p(X)` and `w(X)` by half.
    /// - Updates `sum` using `sumcheck_poly`.
    pub fn compress(
        &mut self,
        combination_randomness: EF, // Scale the initial point
        folding_randomness: &MultilinearPoint<EF>,
        sumcheck_poly: &SumcheckPolynomial<EF>,
    ) {
        assert_eq!(folding_randomness.num_variables(), 1);
        assert!(self.num_variables() >= 1);

        let randomness = folding_randomness.0[0];

        // Fold between extension field elements
        let fold_extension = |slice: &[EF]| -> EF { randomness * (slice[1] - slice[0]) + slice[0] };
        // Fold between base and extension field elements
        let fold_base = |slice: &[F]| -> EF { randomness * (slice[1] - slice[0]) + slice[0] };

        #[cfg(feature = "parallel")]
        let (evaluations_of_p, evaluations_of_eq) = {
            // Threshold below which sequential computation is faster
            //
            // This was chosen based on experiments with the `compress` function.
            // It is possible that the threshold can be tuned further.
            const PARALLEL_THRESHOLD: usize = 4096;

            match &self.evaluation_of_p {
                EvaluationStorage::Base(evals_f) => {
                    if evals_f.evals().len() >= PARALLEL_THRESHOLD
                        && self.weights.evals().len() >= PARALLEL_THRESHOLD
                    {
                        rayon::join(
                            || evals_f.evals().par_chunks_exact(2).map(fold_base).collect(),
                            || {
                                self.weights
                                    .evals()
                                    .par_chunks_exact(2)
                                    .map(fold_extension)
                                    .collect()
                            },
                        )
                    } else {
                        (
                            evals_f.evals().chunks_exact(2).map(fold_base).collect(),
                            self.weights
                                .evals()
                                .chunks_exact(2)
                                .map(fold_extension)
                                .collect(),
                        )
                    }
                }
                EvaluationStorage::Extension(evals_ef) => {
                    if evals_ef.evals().len() >= PARALLEL_THRESHOLD
                        && self.weights.evals().len() >= PARALLEL_THRESHOLD
                    {
                        rayon::join(
                            || {
                                evals_ef
                                    .evals()
                                    .par_chunks_exact(2)
                                    .map(fold_extension)
                                    .collect()
                            },
                            || {
                                self.weights
                                    .evals()
                                    .par_chunks_exact(2)
                                    .map(fold_extension)
                                    .collect()
                            },
                        )
                    } else {
                        (
                            evals_ef
                                .evals()
                                .chunks_exact(2)
                                .map(fold_extension)
                                .collect(),
                            self.weights
                                .evals()
                                .chunks_exact(2)
                                .map(fold_extension)
                                .collect(),
                        )
                    }
                }
            }
        };

        #[cfg(not(feature = "parallel"))]
        let (evaluations_of_p, evaluations_of_eq) = match &self.evaluation_of_p {
            EvaluationStorage::Base(evals_f) => (
                evals_f.evals().chunks_exact(2).map(fold_base).collect(),
                self.weights
                    .evals()
                    .chunks_exact(2)
                    .map(fold_extension)
                    .collect(),
            ),
            EvaluationStorage::Extension(evals_ef) => (
                evals_ef
                    .evals()
                    .chunks_exact(2)
                    .map(fold_extension)
                    .collect(),
                self.weights
                    .evals()
                    .chunks_exact(2)
                    .map(fold_extension)
                    .collect(),
            ),
        };

        // Update internal state
        self.evaluation_of_p = EvaluationStorage::Extension(EvaluationsList::new(evaluations_of_p));
        self.weights = EvaluationsList::new(evaluations_of_eq);
        self.sum = combination_randomness * sumcheck_poly.evaluate_at_point(folding_randomness);
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, extension::BinomialExtensionField};
    use proptest::prelude::*;

    use super::*;
    use crate::{
        fiat_shamir::{DefaultHash, domain_separator::DomainSeparator, pow::blake3::Blake3PoW},
        poly::{coeffs::CoefficientList, multilinear::MultilinearPoint},
        whir::statement::Weights,
    };

    type F = BabyBear;
    type EF4 = BinomialExtensionField<BabyBear, 4>;

    #[test]
    fn test_sumcheck_single_initialization() {
        // Polynomial with 2 variables: f(X1, X2) = 1 + 2*X1 + 3*X2 + 4*X1*X2
        let c1 = F::from_u64(1);
        let c2 = F::from_u64(2);
        let c3 = F::from_u64(3);
        let c4 = F::from_u64(4);

        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4]);
        let statement = Statement::new(2);

        let prover = SumcheckSingle::from_base_coeffs(coeffs, &statement, F::ONE);

        // Expected evaluation table after wavelet transform
        let expected_evaluation_of_p = vec![c1, c1 + c2, c1 + c3, c1 + c2 + c3 + c4];

        assert_eq!(
            match prover.evaluation_of_p {
                EvaluationStorage::Base(ref evals_f) => evals_f.evals(),
                EvaluationStorage::Extension(ref evals_ef) => evals_ef.evals(),
            },
            &expected_evaluation_of_p
        );
        assert_eq!(prover.weights.evals(), &vec![F::ZERO; 4]);
        assert_eq!(prover.sum, F::ZERO);
        assert_eq!(prover.num_variables(), 2);
    }

    #[test]
    fn test_sumcheck_single_one_variable() {
        // Polynomial with 1 variable: f(X1) = 1 + 3*X1
        let c1 = F::from_u64(1);
        let c2 = F::from_u64(3);

        // Convert the polynomial into coefficient form
        let coeffs = CoefficientList::new(vec![c1, c2]);

        // Create an empty statement (no equality constraints)
        let statement = Statement::new(1);

        // Instantiate the Sumcheck prover
        let prover = SumcheckSingle::from_base_coeffs(coeffs, &statement, F::ONE);

        // Expected evaluations of the polynomial in evaluation form
        let expected_evaluation_of_p = vec![c1, c1 + c2];

        assert_eq!(
            match prover.evaluation_of_p {
                EvaluationStorage::Base(ref evals_f) => evals_f.evals(),
                EvaluationStorage::Extension(ref evals_ef) => evals_ef.evals(),
            },
            &expected_evaluation_of_p
        );
        assert_eq!(prover.weights.evals(), &vec![F::ZERO; 2]);
        assert_eq!(prover.sum, F::ZERO);
        assert_eq!(prover.num_variables(), 1);
    }

    #[test]
    fn test_sumcheck_single_three_variables() {
        // Polynomial with 3 variables:
        // f(X1, X2, X3) = 1 + 2*X1 + 3*X2 + 4*X1*X2 + 5*X3 + 6*X1*X3 + 7*X2*X3 + 8*X1*X2*X3
        let c1 = F::from_u64(1);
        let c2 = F::from_u64(2);
        let c3 = F::from_u64(3);
        let c4 = F::from_u64(4);
        let c5 = F::from_u64(5);
        let c6 = F::from_u64(6);
        let c7 = F::from_u64(7);
        let c8 = F::from_u64(8);

        // Convert the polynomial into coefficient form
        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4, c5, c6, c7, c8]);

        // Create an empty statement (no equality constraints)
        let statement = Statement::new(3);

        // Instantiate the Sumcheck prover
        let prover = SumcheckSingle::from_base_coeffs(coeffs, &statement, F::ONE);

        // Expected evaluations of the polynomial in evaluation form
        let expected_evaluation_of_p = vec![
            c1,
            c1 + c2,
            c1 + c3,
            c1 + c2 + c3 + c4,
            c1 + c5,
            c1 + c2 + c5 + c6,
            c1 + c3 + c5 + c7,
            c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8,
        ];

        assert_eq!(
            match prover.evaluation_of_p {
                EvaluationStorage::Base(ref evals_f) => evals_f.evals(),
                EvaluationStorage::Extension(ref evals_ef) => evals_ef.evals(),
            },
            &expected_evaluation_of_p
        );
        assert_eq!(prover.weights.evals(), &vec![F::ZERO; 8]);
        assert_eq!(prover.sum, F::ZERO);
        assert_eq!(prover.num_variables(), 3);
    }

    #[test]
    fn test_sumcheck_single_with_equality_constraints() {
        // Define a polynomial with 2 variables:
        // f(X1, X2) = 1 + 2*X1 + 3*X2 + 4*X1*X2
        let c1 = F::from_u64(1);
        let c2 = F::from_u64(2);
        let c3 = F::from_u64(3);
        let c4 = F::from_u64(4);

        // Convert the polynomial into coefficient form
        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4]);

        // Create a statement and introduce an equality constraint at (X1, X2) = (1,0)
        let mut statement = Statement::new(2);
        let point = MultilinearPoint(vec![F::ONE, F::ZERO]);
        let weights = Weights::evaluation(point);
        let eval = F::from_u64(5);
        statement.add_constraint(weights, eval);

        // Instantiate the Sumcheck prover
        let prover = SumcheckSingle::from_base_coeffs(coeffs, &statement, F::ONE);

        // Expected sum update: sum = 5
        assert_eq!(prover.sum, eval);

        // Expected evaluation table after wavelet transform
        let expected_evaluation_of_p = vec![c1, c1 + c2, c1 + c3, c1 + c2 + c3 + c4];
        assert_eq!(
            match prover.evaluation_of_p {
                EvaluationStorage::Base(ref evals_f) => evals_f.evals(),
                EvaluationStorage::Extension(ref evals_ef) => evals_ef.evals(),
            },
            &expected_evaluation_of_p
        );
        assert_eq!(prover.num_variables(), 2);
    }

    #[test]
    fn test_sumcheck_single_multiple_constraints() {
        // Define a polynomial with 3 variables:
        // f(X1, X2, X3) = c1 + c2*X1 + c3*X2 + c4*X3 + c5*X1*X2 + c6*X1*X3 + c7*X2*X3 + c8*X1*X2*X3
        let c1 = F::from_u64(1);
        let c2 = F::from_u64(2);
        let c3 = F::from_u64(3);
        let c4 = F::from_u64(4);
        let c5 = F::from_u64(5);
        let c6 = F::from_u64(6);
        let c7 = F::from_u64(7);
        let c8 = F::from_u64(8);

        // Convert the polynomial into coefficient form
        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4, c5, c6, c7, c8]);

        // Create a statement and introduce multiple equality constraints
        let mut statement = Statement::new(3);

        // Constraints: (X1, X2, X3) = (1,0,1) with weight 2, (0,1,0) with weight 3
        let point1 = MultilinearPoint(vec![F::ONE, F::ZERO, F::ONE]);
        let point2 = MultilinearPoint(vec![F::ZERO, F::ONE, F::ZERO]);

        let weights1 = Weights::evaluation(point1);
        let weights2 = Weights::evaluation(point2);

        let eval1 = F::from_u64(5);
        let eval2 = F::from_u64(4);

        statement.add_constraint(weights1, eval1);
        statement.add_constraint(weights2, eval2);

        // Instantiate the Sumcheck prover
        let prover = SumcheckSingle::from_base_coeffs(coeffs, &statement, F::ONE);

        // Expected sum update: sum = (5) + (4)
        let expected_sum = eval1 + eval2;
        assert_eq!(prover.sum, expected_sum);
    }

    #[test]
    fn test_compute_sumcheck_polynomial_basic() {
        // Polynomial with 2 variables: f(X1, X2) = c1 + c2*X1 + c3*X2 + c4*X1*X2
        let c1 = F::from_u64(1);
        let c2 = F::from_u64(2);
        let c3 = F::from_u64(3);
        let c4 = F::from_u64(4);

        // Convert the polynomial into coefficient form
        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4]);

        // Create an empty statement (no constraints)
        let statement = Statement::new(2);

        // Instantiate the Sumcheck prover
        let prover = SumcheckSingle::from_base_coeffs(coeffs, &statement, F::ONE);
        let sumcheck_poly = prover.compute_sumcheck_polynomial();

        // Since no equality constraints, sumcheck_poly should be **zero**
        let expected_evaluations = vec![F::ZERO; 3];
        assert_eq!(sumcheck_poly.evaluations(), &expected_evaluations);
    }

    #[test]
    fn test_compute_sumcheck_polynomial_with_equality_constraints() {
        // Define a multilinear polynomial with two variables:
        // f(X1, X2) = c1 + c2*X1 + c3*X2 + c4*X1*X2
        // This polynomial is represented in coefficient form.
        let c1 = F::from_u64(1);
        let c2 = F::from_u64(2);
        let c3 = F::from_u64(3);
        let c4 = F::from_u64(4);

        // Convert the polynomial into coefficient list representation
        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4]);

        // Create a statement and introduce an equality constraint at (X1, X2) = (1,0)
        // The constraint enforces that f(1,0) must evaluate to 5 with weight 2.
        let mut statement = Statement::new(2);
        let point = MultilinearPoint(vec![F::ONE, F::ZERO]);
        let weights = Weights::evaluation(point);
        let eval = F::from_u64(5);
        statement.add_constraint(weights, eval);

        // Instantiate the Sumcheck prover with the polynomial and equality constraints
        let prover = SumcheckSingle::from_base_coeffs(coeffs, &statement, F::ONE);
        let sumcheck_poly = prover.compute_sumcheck_polynomial();

        // The constraint directly contributes to the sum, hence sum = 5
        assert_eq!(prover.sum, eval);

        // Compute the polynomial evaluations at the four possible binary inputs
        let ep_00 = c1; // f(0,0) = c1
        let ep_01 = c1 + c2; // f(0,1) = c1 + c2
        let ep_10 = c1 + c3; // f(1,0) = c1 + c3
        let ep_11 = c1 + c3 + c2 + c4; // f(1,1) = c1 + c3 + c2 + c4

        // Compute the evaluations of the equality constraint polynomial at each binary input
        // Given that the constraint is at (1,0) with weight 2, the equality function is:
        //
        // \begin{equation}
        // eq(X1, X2) = 2 * (X1 - 1) * (X2 - 0)
        // \end{equation}
        let f_00 = F::ZERO; // eq(0,0) = 0
        let f_01 = F::ZERO; // eq(0,1) = 0
        let f_10 = F::ONE; // eq(1,0) = 1
        let f_11 = F::ZERO; // eq(1,1) = 0

        // Compute the coefficients of the sumcheck polynomial S(X)
        let e0 = ep_00 * f_00 + ep_10 * f_10; // Constant term (X = 0)
        let e2 = (ep_01 - ep_00) * (f_01 - f_00) + (ep_11 - ep_10) * (f_11 - f_10); // Quadratic coefficient
        let e1 = prover.sum - e0.double() - e2; // Middle coefficient using sum rule

        // Compute the evaluations of the sumcheck polynomial at X ∈ {0,1,2}
        let eval_0 = e0;
        let eval_1 = e0 + e1 + e2;
        let eval_2 = eval_1 + e1 + e2 + e2.double();
        let expected_evaluations = vec![eval_0, eval_1, eval_2];

        // Ensure that the computed sumcheck polynomial matches expectations
        assert_eq!(sumcheck_poly.evaluations(), &expected_evaluations);
    }

    #[test]
    fn test_compute_sumcheck_polynomial_with_equality_constraints_3vars() {
        // Define a multilinear polynomial with three variables:
        // f(X1, X2, X3) = c1 + c2*X1 + c3*X2 + c4*X3 + c5*X1*X2 + c6*X1*X3 + c7*X2*X3 + c8*X1*X2*X3
        let c1 = F::from_u64(1);
        let c2 = F::from_u64(2);
        let c3 = F::from_u64(3);
        let c4 = F::from_u64(4);
        let c5 = F::from_u64(5);
        let c6 = F::from_u64(6);
        let c7 = F::from_u64(7);
        let c8 = F::from_u64(8);

        // Convert the polynomial into coefficient form
        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4, c5, c6, c7, c8]);

        // Create a statement and introduce an equality constraint at (X1, X2, X3) = (1,0,1)
        let mut statement = Statement::new(3);
        let point = MultilinearPoint(vec![F::ONE, F::ZERO, F::ONE]);
        let weights = Weights::evaluation(point);
        let eval = F::from_u64(5);
        statement.add_constraint(weights, eval);

        // Instantiate the Sumcheck prover with the polynomial and equality constraints
        let prover = SumcheckSingle::from_base_coeffs(coeffs, &statement, F::ONE);
        let sumcheck_poly = prover.compute_sumcheck_polynomial();

        // Expected sum update: sum = 5
        assert_eq!(prover.sum, eval);

        // Compute polynomial evaluations at the eight possible binary inputs
        let ep_000 = c1; // f(0,0,0)
        let ep_001 = c1 + c2; // f(0,0,1)
        let ep_010 = c1 + c3; // f(0,1,0)
        let ep_011 = c1 + c2 + c3 + c4; // f(0,1,1)
        let ep_100 = c1 + c5; // f(1,0,0)
        let ep_101 = c1 + c2 + c5 + c6; // f(1,0,1)
        let ep_110 = c1 + c3 + c5 + c7; // f(1,1,0)
        let ep_111 = c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8; // f(1,1,1)

        // Compute the evaluations of the equality constraint polynomial at each binary input
        // Given that the constraint is at (1,0,1) with weight 2, the equality function is:
        //
        // \begin{equation}
        // eq(X1, X2, X3) = 2 * (X1 - 1) * (X2 - 0) * (X3 - 1)
        // \end{equation}
        let f_000 = F::ZERO; // eq(0,0,0) = 0
        let f_001 = F::ZERO; // eq(0,0,1) = 0
        let f_010 = F::ZERO; // eq(0,1,0) = 0
        let f_011 = F::ZERO; // eq(0,1,1) = 0
        let f_100 = F::ZERO; // eq(1,0,0) = 0
        let f_101 = F::ONE; // eq(1,0,1) = 1
        let f_110 = F::ZERO; // eq(1,1,0) = 0
        let f_111 = F::ZERO; // eq(1,1,1) = 0

        // Compute the coefficients of the sumcheck polynomial S(X)
        let e0 = ep_000 * f_000 + ep_010 * f_010 + ep_100 * f_100 + ep_110 * f_110; // Contribution at X = 0
        let e2 = (ep_001 - ep_000) * (f_001 - f_000)
            + (ep_011 - ep_010) * (f_011 - f_010)
            + (ep_101 - ep_100) * (f_101 - f_100)
            + (ep_111 - ep_110) * (f_111 - f_110); // Quadratic coefficient
        let e1 = prover.sum - e0.double() - e2; // Middle coefficient using sum rule

        // Compute sumcheck polynomial evaluations at {0,1,2}
        let eval_0 = e0;
        let eval_1 = e0 + e1 + e2;
        let eval_2 = eval_1 + e1 + e2 + e2.double();
        let expected_evaluations = vec![eval_0, eval_1, eval_2];

        // Assert that computed sumcheck polynomial matches expectations
        assert_eq!(sumcheck_poly.evaluations(), &expected_evaluations);
    }

    #[test]
    fn test_add_new_equality_single_constraint() {
        // Polynomial with 2 variables:
        // f(X1, X2) = c1 + c2*X1 + c3*X2 + c4*X1*X2
        let c1 = F::from_u64(1);
        let c2 = F::from_u64(2);
        let c3 = F::from_u64(3);
        let c4 = F::from_u64(4);
        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4]);

        // Create an empty statement (no constraints initially)
        let statement = Statement::new(2);

        // Instantiate the Sumcheck prover
        let mut prover = SumcheckSingle::from_base_coeffs(coeffs, &statement, F::ONE);

        // Add a single constraint at (X1, X2) = (1,0) with weight 2
        let point = MultilinearPoint(vec![F::ONE, F::ZERO]);
        let weight = F::from_u64(2);

        // Compute f(1,0) **without simplifications**
        //
        // f(1,0) = c1 + c2*X1 + c3*X2 + c4*X1*X2
        //        = c1 + c2*(1) + c3*(0) + c4*(1)*(0)
        let eval = c1 + c2 * F::ONE + c3 * F::ZERO + c4 * F::ONE * F::ZERO;

        prover.add_new_equality(&[point.clone()], &[eval], &[weight]);

        // Compute expected sum explicitly:
        //
        // sum = weight * eval
        //     = (2 * (c1 + c2*(1) + c3*(0) + c4*(1)*(0)))
        //
        let expected_sum = weight * eval;
        assert_eq!(prover.sum, expected_sum);

        // Compute the expected weight updates:
        // The equality function at point (X1, X2) = (1,0) updates the weights.
        let mut expected_weights = vec![F::ZERO; 4];
        eval_eq(&point.0, &mut expected_weights, weight);

        assert_eq!(prover.weights.evals(), &expected_weights);
    }

    #[test]
    fn test_add_new_equality_multiple_constraints() {
        // Polynomial with 3 variables:
        // f(X1, X2, X3) = c1 + c2*X1 + c3*X2 + c4*X1*X2 + c5*X3 + c6*X1*X3 + c7*X2*X3 + c8*X1*X2*X3
        let c1 = F::from_u64(1);
        let c2 = F::from_u64(2);
        let c3 = F::from_u64(3);
        let c4 = F::from_u64(4);
        let c5 = F::from_u64(5);
        let c6 = F::from_u64(6);
        let c7 = F::from_u64(7);
        let c8 = F::from_u64(8);
        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4, c5, c6, c7, c8]);

        // Create an empty statement (no constraints initially)
        let statement = Statement::new(3);

        // Instantiate the Sumcheck prover
        let mut prover = SumcheckSingle::from_base_coeffs(coeffs, &statement, F::ONE);

        // Add constraints at (X1, X2, X3) = (1,0,1) with weight 2 and (0,1,0) with weight 3
        let point1 = MultilinearPoint(vec![F::ONE, F::ZERO, F::ONE]);
        let point2 = MultilinearPoint(vec![F::ZERO, F::ONE, F::ZERO]);

        let weight1 = F::from_u64(2);
        let weight2 = F::from_u64(3);

        // Compute f(1,0,1) using the polynomial definition:
        //
        // f(1,0,1) = c1 + c2*X1 + c3*X2 + c4*X1*X2 + c5*X3 + c6*X1*X3 + c7*X2*X3 + c8*X1*X2*X3
        //          = c1 + c2*(1) + c3*(0) + c4*(1)*(0) + c5*(1) + c6*(1)*(1) + c7*(0)*(1) +
        // c8*(1)*(0)*(1)
        let eval1 = c1
            + c2 * F::ONE
            + c3 * F::ZERO
            + c4 * F::ONE * F::ZERO
            + c5 * F::ONE
            + c6 * F::ONE * F::ONE
            + c7 * F::ZERO * F::ONE
            + c8 * F::ONE * F::ZERO * F::ONE;

        // Compute f(0,1,0) using the polynomial definition:
        //
        // f(0,1,0) = c1 + c2*X1 + c3*X2 + c4*X1*X2 + c5*X3 + c6*X1*X3 + c7*X2*X3 + c8*X1*X2*X3
        //          = c1 + c2*(0) + c3*(1) + c4*(0)*(1) + c5*(0) + c6*(0)*(0) + c7*(1)*(0) +
        // c8*(0)*(1)*(0)
        let eval2 = c1
            + c2 * F::ZERO
            + c3 * F::ONE
            + c4 * F::ZERO * F::ONE
            + c5 * F::ZERO
            + c6 * F::ZERO * F::ZERO
            + c7 * F::ONE * F::ZERO
            + c8 * F::ZERO * F::ONE * F::ZERO;

        prover.add_new_equality(
            &[point1.clone(), point2.clone()],
            &[eval1, eval2],
            &[weight1, weight2],
        );

        // Compute the expected sum manually:
        //
        // sum = (weight1 * eval1) + (weight2 * eval2)
        let expected_sum = weight1 * eval1 + weight2 * eval2;
        assert_eq!(prover.sum, expected_sum);

        // Expected weight updates
        let mut expected_weights = vec![F::ZERO; 8];
        eval_eq(&point1.0, &mut expected_weights, weight1);
        eval_eq(&point2.0, &mut expected_weights, weight2);

        assert_eq!(prover.weights.evals(), &expected_weights);
    }

    #[test]
    fn test_add_new_equality_with_zero_weight() {
        let c1 = F::from_u64(1);
        let c2 = F::from_u64(2);
        let coeffs = CoefficientList::new(vec![c1, c2]);

        let statement = Statement::new(1);
        let mut prover = SumcheckSingle::from_base_coeffs(coeffs, &statement, F::ONE);

        let point = MultilinearPoint(vec![F::ONE]);
        let weight = F::ZERO;
        let eval = F::from_u64(5);

        prover.add_new_equality(&[point], &[eval], &[weight]);

        // The sum should remain unchanged since the weight is zero
        assert_eq!(prover.sum, F::ZERO);

        // The weights should remain unchanged
        let expected_weights = vec![F::ZERO; 2];
        assert_eq!(prover.weights.evals(), &expected_weights);
    }

    #[test]
    fn test_compress_basic() {
        // Polynomial with 2 variables:
        // f(X1, X2) = c1 + c2*X1 + c3*X2 + c4*X1*X2
        let c1 = F::from_u64(1);
        let c2 = F::from_u64(2);
        let c3 = F::from_u64(3);
        let c4 = F::from_u64(4);
        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4]);

        // Create an empty statement (no constraints initially)
        let statement = Statement::new(2);

        // Instantiate the Sumcheck prover
        let mut prover = SumcheckSingle::from_base_coeffs(coeffs, &statement, F::ONE);

        // Define random values for compression
        let combination_randomness = F::from_u64(3);
        let folding_randomness = MultilinearPoint(vec![F::from_u64(2)]);

        // Compute sumcheck polynomial manually:
        let sumcheck_poly = prover.compute_sumcheck_polynomial();

        // Apply compression
        prover.compress(combination_randomness, &folding_randomness, &sumcheck_poly);

        // Compute expected evaluations after compression
        //
        // Compression follows the formula:
        //
        // p'(X2) = (p(X1=1, X2) - p(X1=0, X2)) * r + p(X1=0, X2)
        //
        // where r = folding_randomness
        let r = folding_randomness.0[0];

        let eval_00 = c1; // f(0,0) = c1
        let eval_01 = c1 + c3; // f(0,1) = c1 + c3
        let eval_10 = c1 + c2; // f(1,0) = c1 + c2
        let eval_11 = c1 + c2 + c3 + c4; // f(1,1) = c1 + c2 + c3 + c4

        // Compute new evaluations after compression:
        let compressed_eval_0 = (eval_10 - eval_00) * r + eval_00;
        let compressed_eval_1 = (eval_11 - eval_01) * r + eval_01;

        let expected_compressed_evaluations = vec![compressed_eval_0, compressed_eval_1];

        assert_eq!(
            match prover.evaluation_of_p {
                EvaluationStorage::Base(ref evals_f) => evals_f.evals(),
                EvaluationStorage::Extension(ref evals_ef) => evals_ef.evals(),
            },
            &expected_compressed_evaluations
        );

        // Compute the expected sum update:
        //
        // sum' = combination_randomness * sumcheck_poly.evaluate_at_point(folding_randomness)
        let expected_sum =
            combination_randomness * sumcheck_poly.evaluate_at_point(&folding_randomness);
        assert_eq!(prover.sum, expected_sum);

        // Check weights after compression
        let weight_0 = prover.weights.evals()[0]; // w(X1=0, X2=0)
        let weight_1 = prover.weights.evals()[1]; // w(X1=0, X2=1)

        // Compute compressed weights
        let compressed_weight_0 = weight_0; // No change as X1=0 remains
        let compressed_weight_1 = weight_1; // No change as X1=0 remains

        // The expected compressed weights after applying the transformation
        let expected_compressed_weights = vec![compressed_weight_0, compressed_weight_1];

        assert_eq!(prover.weights.evals(), &expected_compressed_weights);
    }

    #[test]
    fn test_compress_three_variables() {
        // Polynomial with 3 variables:
        // f(X1, X2, X3) = c1 + c2*X1 + c3*X2 + c4*X1*X2 + c5*X3 + c6*X1*X3 + c7*X2*X3 + c8*X1*X2*X3
        let c1 = F::from_u64(1);
        let c2 = F::from_u64(2);
        let c3 = F::from_u64(3);
        let c4 = F::from_u64(4);
        let c5 = F::from_u64(5);
        let c6 = F::from_u64(6);
        let c7 = F::from_u64(7);
        let c8 = F::from_u64(8);
        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4, c5, c6, c7, c8]);

        // Create an empty statement (no constraints initially)
        let statement = Statement::new(3);

        // Instantiate the Sumcheck prover
        let mut prover = SumcheckSingle::from_base_coeffs(coeffs, &statement, F::ONE);

        // Define random values for compression
        let combination_randomness = F::from_u64(2);
        let folding_randomness = MultilinearPoint(vec![F::from_u64(3)]);

        // Compute sumcheck polynomial manually:
        let sumcheck_poly = prover.compute_sumcheck_polynomial();

        // Apply compression
        prover.compress(combination_randomness, &folding_randomness, &sumcheck_poly);

        // Compute expected evaluations after compression
        //
        // Compression formula:
        //
        // p'(X2, X3) = (p(X1=1, X2, X3) - p(X1=0, X2, X3)) * r + p(X1=0, X2, X3)
        //
        // where r = folding_randomness
        let r = folding_randomness.0[0];

        let eval_000 = c1;
        let eval_001 = c1 + c5;
        let eval_010 = c1 + c3;
        let eval_011 = c1 + c3 + c5 + c7;
        let eval_100 = c1 + c2;
        let eval_101 = c1 + c2 + c5 + c6;
        let eval_110 = c1 + c2 + c3 + c4;
        let eval_111 = c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8;

        // Compute compressed evaluations
        let compressed_eval_00 = (eval_100 - eval_000) * r + eval_000;
        let compressed_eval_01 = (eval_101 - eval_001) * r + eval_001;
        let compressed_eval_10 = (eval_110 - eval_010) * r + eval_010;
        let compressed_eval_11 = (eval_111 - eval_011) * r + eval_011;

        let expected_compressed_evaluations = vec![
            compressed_eval_00,
            compressed_eval_10,
            compressed_eval_01,
            compressed_eval_11,
        ];

        assert_eq!(
            match prover.evaluation_of_p {
                EvaluationStorage::Base(ref evals_f) => evals_f.evals(),
                EvaluationStorage::Extension(ref evals_ef) => evals_ef.evals(),
            },
            &expected_compressed_evaluations
        );

        // Compute the expected sum update:
        let expected_sum =
            combination_randomness * sumcheck_poly.evaluate_at_point(&folding_randomness);
        assert_eq!(prover.sum, expected_sum);

        // Check weights after compression
        //
        // Compression formula:
        //
        // w'(X2, X3) = (w(X1=1, X2, X3) - w(X1=0, X2, X3)) * r + w(X1=0, X2, X3)
        //
        let r = folding_randomness.0[0];

        let weight_00 = prover.weights.evals()[0];
        let weight_01 = prover.weights.evals()[1];
        let weight_10 = prover.weights.evals()[2];
        let weight_11 = prover.weights.evals()[3];

        // Apply the same compression rule
        let compressed_weight_00 = (weight_10 - weight_00) * r + weight_00;
        let compressed_weight_01 = (weight_11 - weight_01) * r + weight_01;

        // The compressed weights should match expected values
        let expected_compressed_weights = vec![
            compressed_weight_00,
            compressed_weight_01,
            compressed_weight_00,
            compressed_weight_01,
        ];

        assert_eq!(prover.weights.evals(), &expected_compressed_weights);
    }

    #[test]
    fn test_compress_with_zero_randomness() {
        // Polynomial with 2 variables:
        // f(X1, X2) = c1 + c2*X1 + c3*X2 + c4*X1*X2
        let c1 = F::from_u64(1);
        let c2 = F::from_u64(2);
        let c3 = F::from_u64(3);
        let c4 = F::from_u64(4);
        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4]);

        // Create an empty statement (no constraints initially)
        let statement = Statement::new(2);

        // Instantiate the Sumcheck prover
        let mut prover = SumcheckSingle::from_base_coeffs(coeffs, &statement, F::ONE);

        // Define zero folding randomness
        let combination_randomness = F::from_u64(2);
        let folding_randomness = MultilinearPoint(vec![F::ZERO]);

        // Compute sumcheck polynomial manually:
        let sumcheck_poly = prover.compute_sumcheck_polynomial();

        // Apply compression
        prover.compress(combination_randomness, &folding_randomness, &sumcheck_poly);

        // Since folding randomness is zero, the compressed evaluations should be:
        //
        // p'(X2) = (p(X1=1, X2) - p(X1=0, X2)) * 0 + p(X1=0, X2)
        //        = p(X1=0, X2)
        let expected_compressed_evaluations = vec![c1, c1 + c3];

        assert_eq!(
            match prover.evaluation_of_p {
                EvaluationStorage::Base(ref evals_f) => evals_f.evals(),
                EvaluationStorage::Extension(ref evals_ef) => evals_ef.evals(),
            },
            &expected_compressed_evaluations
        );

        // Compute the expected sum update:
        let expected_sum =
            combination_randomness * sumcheck_poly.evaluate_at_point(&folding_randomness);
        assert_eq!(prover.sum, expected_sum);

        // Check weights after compression
        //
        // Compression formula:
        //
        // w'(X2) = (w(X1=1, X2) - w(X1=0, X2)) * 0 + w(X1=0, X2)
        //        = w(X1=0, X2)
        //
        // Since `r = 0`, this means the weights remain the same as `X1=0` slice.

        let weight_0 = prover.weights.evals()[0]; // w(X1=0, X2=0)
        let weight_1 = prover.weights.evals()[1]; // w(X1=0, X2=1)

        let expected_compressed_weights = vec![weight_0, weight_1];

        assert_eq!(prover.weights.evals(), &expected_compressed_weights);
    }

    #[test]
    fn test_compute_sumcheck_polynomials_basic_case() {
        // Polynomial with 1 variable: f(X1) = 1 + 2*X1
        let c1 = F::from_u64(1);
        let c2 = F::from_u64(2);
        let coeffs = CoefficientList::new(vec![c1, c2]);

        // Create a statement with no equality constraints
        let statement = Statement::new(1);

        // Instantiate the Sumcheck prover
        let mut prover = SumcheckSingle::from_base_coeffs(coeffs, &statement, F::ONE);

        // Domain separator setup
        // Step 1: Initialize domain separator with a context label
        let mut domsep: DomainSeparator<F, F, DefaultHash> = DomainSeparator::new("test");

        // Step 2: Register the fact that we’re about to absorb 3 field elements
        domsep.add_scalars(3, "test");

        // Step 3: Sample 1 challenge scalar from the transcript
        domsep.challenge_scalars(1, "test");

        // Convert the domain separator to a prover state
        let mut prover_state = domsep.to_prover_state();

        let folding_factor = 1; // Minimum folding factor
        let pow_bits = 0.; // No grinding

        // Compute sumcheck polynomials
        let result = prover
            .compute_sumcheck_polynomials::<Blake3PoW>(&mut prover_state, folding_factor, pow_bits)
            .unwrap();

        // The result should contain `folding_factor` elements
        assert_eq!(result.0.len(), folding_factor);
    }

    #[test]
    fn test_compute_sumcheck_polynomials_with_multiple_folding_factors() {
        // Polynomial with 2 variables: f(X1, X2) = 1 + 2*X1 + 3*X2 + 4*X1*X2
        let c1 = F::from_u64(1);
        let c2 = F::from_u64(2);
        let c3 = F::from_u64(3);
        let c4 = F::from_u64(3);
        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4]);

        let statement = Statement::new(2);
        let mut prover = SumcheckSingle::from_base_coeffs(coeffs, &statement, F::ONE);

        let folding_factor = 2; // Increase folding factor
        let pow_bits = 1.; // Minimal grinding

        // Setup the domain separator
        let mut domsep: DomainSeparator<F, F, DefaultHash> = DomainSeparator::new("test");

        // For each folding round, we must absorb values, sample challenge, and apply PoW
        for _ in 0..folding_factor {
            // Absorb 3 field elements (evaluations of sumcheck polynomial)
            domsep.add_scalars(3, "tag");

            // Sample 1 challenge scalar from the Fiat-Shamir transcript
            domsep.challenge_scalars(1, "tag");

            // Apply optional PoW grinding to ensure randomness
            domsep.challenge_pow("tag");
        }

        // Convert the domain separator to a prover state
        let mut prover_state = domsep.to_prover_state();

        let result = prover
            .compute_sumcheck_polynomials::<Blake3PoW>(&mut prover_state, folding_factor, pow_bits)
            .unwrap();

        // Ensure we get `folding_factor` sampled randomness values
        assert_eq!(result.0.len(), folding_factor);
    }

    #[test]
    fn test_compute_sumcheck_polynomials_with_three_variables() {
        // Multilinear polynomial with 3 variables:
        // f(X1, X2, X3) = 1 + 2*X1 + 3*X2 + 4*X3 + 5*X1*X2 + 6*X1*X3 + 7*X2*X3 + 8*X1*X2*X3
        let c1 = F::from_u64(1);
        let c2 = F::from_u64(2);
        let c3 = F::from_u64(3);
        let c4 = F::from_u64(4);
        let c5 = F::from_u64(5);
        let c6 = F::from_u64(6);
        let c7 = F::from_u64(7);
        let c8 = F::from_u64(8);
        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4, c5, c6, c7, c8]);

        let statement = Statement::new(3);
        let mut prover = SumcheckSingle::from_base_coeffs(coeffs, &statement, F::ONE);

        let folding_factor = 3;
        let pow_bits = 2.;

        // Setup the domain separator
        let mut domsep: DomainSeparator<F, F, DefaultHash> = DomainSeparator::new("test");

        // Register interactions with the transcript for each round
        for _ in 0..folding_factor {
            // Absorb 3 field values (sumcheck evaluations at X = 0, 1, 2)
            domsep.add_scalars(3, "tag");

            // Sample 1 field challenge (folding randomness)
            domsep.challenge_scalars(1, "tag");

            // Apply challenge PoW (grinding) to enhance soundness
            domsep.challenge_pow("tag");
        }

        // Convert the domain separator to a prover state
        let mut prover_state = domsep.to_prover_state();

        let result = prover
            .compute_sumcheck_polynomials::<Blake3PoW>(&mut prover_state, folding_factor, pow_bits)
            .unwrap();

        assert_eq!(result.0.len(), folding_factor);
    }

    #[test]
    fn test_compute_sumcheck_polynomials_edge_case_zero_folding() {
        // Multilinear polynomial with 2 variables:
        // f(X1, X2) = 1 + 2*X1 + 3*X2 + 4*X1*X2
        let c1 = F::from_u64(1);
        let c2 = F::from_u64(2);
        let c3 = F::from_u64(3);
        let c4 = F::from_u64(4);
        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4]);

        let statement = Statement::new(2);
        let mut prover = SumcheckSingle::from_base_coeffs(coeffs, &statement, F::ONE);

        let folding_factor = 0; // Edge case: No folding
        let pow_bits = 1.;

        // No domain separator logic needed since we don't fold
        let domsep: DomainSeparator<F, F, DefaultHash> = DomainSeparator::new("test");
        let mut prover_state = domsep.to_prover_state();

        let result = prover
            .compute_sumcheck_polynomials::<Blake3PoW>(&mut prover_state, folding_factor, pow_bits)
            .unwrap();

        assert_eq!(result.0.len(), 0);
    }

    #[test]
    fn test_sumcheck_single_extension_coeffs_basic() {
        // Define a polynomial f(X1) = 1 + 2·X1 in EF4
        let c1 = EF4::from(F::from_u64(1)); // Constant term
        let c2 = EF4::from(F::from_u64(2)); // Coefficient of X1

        // Coefficients in multilinear form: [c1, c2]
        let coeffs = CoefficientList::new(vec![c1, c2]);

        // Empty statement with no constraints
        let statement = Statement::new(1);

        // Initialize the sumcheck prover with extension field coefficients
        let prover = SumcheckSingle::<F, EF4>::from_extension_coeffs(coeffs, &statement, EF4::ONE);

        // The polynomial has 1 variable
        assert_eq!(prover.num_variables(), 1);

        // No constraints means the initial sum should be 0
        assert_eq!(prover.sum, EF4::ZERO);

        // The wavelet transform of [c1, c2] gives [c1, c1 + c2]
        match &prover.evaluation_of_p {
            EvaluationStorage::Extension(evals) => {
                assert_eq!(evals.evals(), &vec![c1, c1 + c2]);
            }
            EvaluationStorage::Base(_) => panic!("Expected extension evaluations"),
        }
    }

    #[test]
    fn test_add_new_equality_mixed_inputs() {
        // Define a base field polynomial f(X1) = 1 + 2·X1
        let c1 = F::from_u64(1); // Constant term
        let c2 = F::from_u64(2); // Coefficient of X1

        // Coefficients in base field
        let coeffs = CoefficientList::new(vec![c1, c2]);

        // No equality constraints at the start
        let statement = Statement::new(1);

        // Initialize the sumcheck prover with base field coefficients
        let mut prover = SumcheckSingle::<F, EF4>::from_base_coeffs(coeffs, &statement, EF4::ONE);

        // Add an equality constraint at point X1 = 1 with weight 2 and expected value 5
        let point = MultilinearPoint(vec![EF4::from(F::ONE)]); // (X1 = 1)
        let eval = EF4::from(F::from_u64(5)); // f(1) = 5
        let weight = EF4::from(F::from_u64(2)); // Constraint applied with weight 2

        // Apply the equality constraint
        prover.add_new_equality(&[point.clone()], &[eval], &[weight]);

        // The sum should now be weight × eval = 2 × 5 = 10
        assert_eq!(prover.sum, weight * eval);

        // Expected weight table updated using eq(X) = weight × eq_at_point(point)
        let mut expected_weights = vec![EF4::ZERO; 2];
        eval_eq(&point.0, &mut expected_weights, weight);
        assert_eq!(prover.weights.evals(), &expected_weights);
    }

    #[test]
    fn test_compress_mixed_fields() {
        // Define a 2-variable multilinear polynomial with base field coefficients:
        // f(X1, X2) = 1 + 2*X1 + 3*X2 + 4*X1*X2
        let c1 = F::from_u64(1);
        let c2 = F::from_u64(2);
        let c3 = F::from_u64(3);
        let c4 = F::from_u64(4);
        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4]);

        // Create a statement with no equality constraints
        let statement = Statement::new(2);

        // Create a Sumcheck prover using base field coefficients and EF4 as extension
        let mut prover = SumcheckSingle::<F, EF4>::from_base_coeffs(coeffs, &statement, EF4::ONE);

        // Define combination randomness for compress (scaling the new sum)
        let combination_randomness = EF4::from(F::from_u64(5));

        // Define folding randomness used to eliminate the first variable X1
        let folding_randomness = MultilinearPoint(vec![EF4::from(F::from_u64(2))]);

        // Compute the sumcheck polynomial h(X) = a + b*X + c*X^2
        let sumcheck_poly = prover.compute_sumcheck_polynomial();

        // Apply the compress function which reduces number of variables by 1
        prover.compress(combination_randomness, &folding_randomness, &sumcheck_poly);

        // Evaluate the original polynomial at all binary points to get expected evaluations
        // Wavelet transform yields: [f(0,0), f(1,0), f(0,1), f(1,1)]
        let eval_00 = EF4::from(c1); // f(0,0)
        let eval_10 = EF4::from(c1 + c2); // f(1,0)
        let eval_01 = EF4::from(c1 + c3); // f(0,1)
        let eval_11 = EF4::from(c1 + c2 + c3 + c4); // f(1,1)

        // Folding randomness r
        let r = folding_randomness.0[0];

        // Compute expected compressed evaluations using:
        // f'(X2) = (f(1, X2) - f(0, X2)) * r + f(0, X2)
        let compressed_0 = (eval_10 - eval_00) * r + eval_00;
        let compressed_1 = (eval_11 - eval_01) * r + eval_01;
        let expected_compressed_evals = vec![compressed_0, compressed_1];

        // Check that the evaluations after compression are correct
        assert_eq!(
            match prover.evaluation_of_p {
                EvaluationStorage::Base(_) => panic!("Should be extension after compression"),
                EvaluationStorage::Extension(ref evals) => evals.evals(),
            },
            &expected_compressed_evals
        );

        // The new sum should be combination_randomness * h(r)
        let expected_sum =
            combination_randomness * sumcheck_poly.evaluate_at_point(&folding_randomness);
        assert_eq!(prover.sum, expected_sum);

        // Initial weights were all zero, so folded weights should remain zero
        let expected_weights = vec![EF4::ZERO, EF4::ZERO];
        assert_eq!(prover.weights.evals(), &expected_weights);
    }

    #[test]
    fn test_compute_sumcheck_polynomials_mixed_fields_three_vars() {
        // Define a multilinear polynomial with 3 variables:
        // f(X1, X2, X3) = 1 + 2*X1 + 3*X2 + 4*X3 + 5*X1*X2 + 6*X1*X3 + 7*X2*X3 + 8*X1*X2*X3
        let c1 = F::from_u64(1);
        let c2 = F::from_u64(2);
        let c3 = F::from_u64(3);
        let c4 = F::from_u64(4);
        let c5 = F::from_u64(5);
        let c6 = F::from_u64(6);
        let c7 = F::from_u64(7);
        let c8 = F::from_u64(8);
        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4, c5, c6, c7, c8]);

        // Empty constraint system (no added equalities)
        let statement = Statement::new(3);

        // Instantiate the Sumcheck prover using base field coefficients and extension field EF4
        let mut prover = SumcheckSingle::<F, EF4>::from_base_coeffs(coeffs, &statement, EF4::ONE);

        // Number of folding rounds (equal to number of variables)
        let folding_factor = 3;

        // PoW challenge difficulty (controls grinding)
        let pow_bits = 2.;

        // Create domain separator for Fiat-Shamir transcript simulation
        let mut domsep: DomainSeparator<EF4, F, DefaultHash> = DomainSeparator::new("test");

        // Register expected Fiat-Shamir interactions for each round
        for _ in 0..folding_factor {
            // Step 1: absorb 3 evaluations of the sumcheck polynomial h(X)
            domsep.add_scalars(3, "tag");

            // Step 2: derive a folding challenge scalar from transcript
            domsep.challenge_scalars(1, "tag");

            // Step 3: optionally apply PoW grinding (if pow_bits > 0)
            domsep.challenge_pow("tag");
        }

        // Convert domain separator into prover state object
        let mut prover_state = domsep.to_prover_state();

        // Perform sumcheck folding using Fiat-Shamir-derived randomness and PoW
        let result = prover
            .compute_sumcheck_polynomials::<Blake3PoW>(&mut prover_state, folding_factor, pow_bits)
            .unwrap();

        // Ensure we received the expected number of folding randomness values
        assert_eq!(result.0.len(), folding_factor);
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
        fn prop_compute_sumcheck_polynomial_consistency(
            // Ensure at least 8 base coefficients (2 EF4 elements)
            raw_coeffs in prop::collection::vec(0u64..F::ORDER_U64, 8..=64)
                .prop_filter("len must be power of two and at least 8", |v| v.len().is_power_of_two()),

            // Random coefficients to form a true EF4 combination randomness
            rand_scalars in prop::array::uniform4(0u64..F::ORDER_U64),
        ) {
            // Convert u64s into base field elements
            let coeffs: Vec<F> = raw_coeffs.iter().map(|&x| F::from_u64(x)).collect();

            // Convert base field coeffs into true EF4 elements (4 base elems → 1 EF4)
            let coeffs_ext: Vec<EF4> = coeffs
                .chunks_exact(4)
                .map(|chunk| {
                    let basis = [chunk[0], chunk[1], chunk[2], chunk[3]];
                    EF4::from_basis_coefficients_iter(basis.into_iter()).unwrap()
                })
                .collect();

            // Build coefficient lists
            let base_cl = CoefficientList::new(coeffs.clone());
            let ext_cl = CoefficientList::new(coeffs_ext.clone());

            // Determine number of variables (log₂(length))
            let n_vars = base_cl.num_variables();
            prop_assume!(n_vars >= 1); // Safeguard for edge cases

            // Build empty constraint system
            let statement = Statement::new(n_vars);

            // Construct random combination EF4 element from 4 base values
            let combination_randomness = EF4::from_basis_coefficients_iter(
                rand_scalars.map(F::from_u64).into_iter()
            ).unwrap();

            // Initialize sumcheck provers for both base and extension representations
            let base_prover = SumcheckSingle::<F, EF4>::from_base_coeffs(base_cl, &statement, combination_randomness);
            let ext_prover = SumcheckSingle::<F, EF4>::from_extension_coeffs(ext_cl, &statement, combination_randomness);

            // Compute the sumcheck polynomial in both cases
            let poly_base = base_prover.compute_sumcheck_polynomial();
            let poly_ext = ext_prover.compute_sumcheck_polynomial();

            // Assert consistency between both representations
            prop_assert_eq!(poly_base.evaluations(), poly_ext.evaluations());
        }

        #[test]
        fn prop_compress_consistency(
            // Generate a longer list (at least 4, and power of two) for testing compression
            raw_coeffs in prop::collection::vec(0u64..F::ORDER_U64, 4_usize..=64)
                .prop_filter("len must be power of two and >= 4", |v| v.len().is_power_of_two() && v.len() >= 4),
            // Random folding randomness value
            fold_scalar in 0u64..F::ORDER_U64,
            // Random scalar for compression scaling
            rand_scalar in 0u64..F::ORDER_U64,
            // Random scalar for combination randomness
            combo_scalar in 0u64..F::ORDER_U64,
        ) {
            // Convert to base and extension field representations
            let coeffs: Vec<F> = raw_coeffs.iter().map(|&x| F::from_u64(x)).collect();
            let coeffs_ext: Vec<EF4> = coeffs.iter().copied().map(EF4::from).collect();

            // Wrap as coefficient lists
            let base_cl = CoefficientList::new(coeffs.clone());
            let ext_cl = CoefficientList::new(coeffs_ext.clone());

            // Determine number of variables
            let n_vars = base_cl.num_variables();

            // Create a dummy statement (no constraints)
            let statement = Statement::new(n_vars);

            // Use a random combination randomness for initializing the provers
            let init_randomness = EF4::from(F::from_u64(combo_scalar));
            let mut base_prover = SumcheckSingle::<F, EF4>::from_base_coeffs(base_cl, &statement, init_randomness);
            let mut ext_prover = SumcheckSingle::<F, EF4>::from_extension_coeffs(ext_cl, &statement, init_randomness);

            // Construct folding point and compression randomness
            let fold_point = MultilinearPoint(vec![EF4::from(F::from_u64(fold_scalar))]);
            let combination_randomness = EF4::from(F::from_u64(rand_scalar));

            // Compute the sumcheck polynomials for both
            let poly_base = base_prover.compute_sumcheck_polynomial();
            let poly_ext = ext_prover.compute_sumcheck_polynomial();

            // Apply compression step to both provers
            base_prover.compress(combination_randomness, &fold_point, &poly_base);
            ext_prover.compress(combination_randomness, &fold_point, &poly_ext);

            // Assert that the sum is identical post-compression
            prop_assert_eq!(base_prover.sum, ext_prover.sum);

            // Assert that the evaluations match
            prop_assert_eq!(
                match base_prover.evaluation_of_p {
                    EvaluationStorage::Extension(ref evals_b) => evals_b.evals(),
                    _ => panic!("Expected extension evaluations"),
                },
                match ext_prover.evaluation_of_p {
                    EvaluationStorage::Extension(ref evals_e) => evals_e.evals(),
                    _ => panic!("Expected extension evaluations"),
                }
            );

            // Assert that the weights match
            prop_assert_eq!(base_prover.weights.evals(), ext_prover.weights.evals());
        }
    }
}
