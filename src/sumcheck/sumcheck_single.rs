use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, PrimeField64, TwoAdicField};
use p3_interpolation::interpolate_subgroup;
use p3_matrix::dense::RowMajorMatrix;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::sumcheck_polynomial::SumcheckPolynomial;
use crate::{
    fiat_shamir::{errors::ProofResult, pow::traits::PowStrategy, prover::ProverState},
    poly::{
        coeffs::CoefficientList,
        evals::{EvaluationStorage, EvaluationsList},
        multilinear::MultilinearPoint,
    },
    sumcheck::{sumcheck_single_skip::fold_k_times, utils::sumcheck_quadratic},
    utils::eval_eq,
    whir::statement::Statement,
};

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
    pub(crate) evaluation_of_p: EvaluationStorage<F, EF>,
    /// Evaluations of the equality polynomial used for enforcing constraints.
    pub(crate) weights: EvaluationsList<EF>,
    /// Accumulated sum incorporating equality constraints.
    pub(crate) sum: EF,
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

    /// Computes the sumcheck polynomial `h(X)`, a quadratic polynomial resulting from the folding step.
    ///
    /// The sumcheck polynomial is computed as:
    ///
    /// \[
    /// h(X) = \sum_b p(b, X) \cdot w(b, X)
    /// \]
    ///
    /// where:
    /// - `b` ranges over evaluation points in `{0,1,2}^1` (i.e., two points per fold).
    /// - `p(b, X)` is the polynomial evaluation at `b` as a function of `X`.
    /// - `w(b, X)` is the associated weight applied at `b` as a function of `X`.
    ///
    /// **Mathematical model:**
    /// - Each chunk of two evaluations encodes a linear polynomial in `X`.
    /// - The product `p(X) * w(X)` is a quadratic polynomial.
    /// - We compute the constant and quadratic coefficients first, then infer the linear coefficient using:
    ///
    /// \[
    /// \text{sum} = 2 \cdot c_0 + c_1 + c_2
    /// \]
    ///
    /// where `sum` is the accumulated constraint sum.
    ///
    /// Returns a `SumcheckPolynomial` with evaluations at `X = 0, 1, 2`.
    pub fn compute_sumcheck_polynomial(&self) -> SumcheckPolynomial<EF> {
        assert!(self.num_variables() >= 1);

        #[cfg(feature = "parallel")]
        let (c0, c2) = match &self.evaluation_of_p {
            EvaluationStorage::Base(evals_f) => evals_f
                .evals()
                .par_chunks_exact(2)
                .zip(self.weights.evals().par_chunks_exact(2))
                .map(sumcheck_quadratic::<F, EF>)
                .reduce(
                    || (EF::ZERO, EF::ZERO),
                    |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
                ),
            EvaluationStorage::Extension(evals_ef) => evals_ef
                .evals()
                .par_chunks_exact(2)
                .zip(self.weights.evals().par_chunks_exact(2))
                .map(sumcheck_quadratic::<EF, EF>)
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
                .map(sumcheck_quadratic::<F, EF>)
                .fold((EF::ZERO, EF::ZERO), |(a0, a2), (b0, b2)| {
                    (a0 + b0, a2 + b2)
                }),

            EvaluationStorage::Extension(evals_ef) => evals_ef
                .evals()
                .chunks_exact(2)
                .zip(self.weights.evals().chunks_exact(2))
                .map(sumcheck_quadratic::<EF, EF>)
                .fold((EF::ZERO, EF::ZERO), |(a0, a2), (b0, b2)| {
                    (a0 + b0, a2 + b2)
                }),
        };

        // Compute the middle (linear) coefficient
        //
        // The quadratic polynomial h(X) has the form:
        //     h(X) = c0 + c1 * X + c2 * X^2
        //
        // We already computed:
        // - c0: the constant coefficient (contribution at X=0)
        // - c2: the quadratic coefficient (contribution at X^2)
        //
        // To recover c1 (linear term), we use the known sum rule:
        //     sum = h(0) + h(1)
        // Expand h(0) and h(1):
        //     h(0) = c0
        //     h(1) = c0 + c1 + c2
        // Therefore:
        //     sum = c0 + (c0 + c1 + c2) = 2*c0 + c1 + c2
        //
        // Rearranging for c1 gives:
        //     c1 = sum - 2*c0 - c2
        let c1 = self.sum - c0.double() - c2;

        // Evaluate the quadratic polynomial at points 0, 1, 2
        //
        // Evaluate:
        //     h(0) = c0
        //     h(1) = c0 + c1 + c2
        //     h(2) = c0 + 2*c1 + 4*c2
        //
        // To compute h(2) efficiently, observe:
        //     h(2) = h(1) + (c1 + 2*c2)
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
    pub fn compute_sumcheck_polynomials<S, DFT>(
        &mut self,
        prover_state: &mut ProverState<EF, F>,
        folding_factor: usize,
        pow_bits: f64,
        k_skip: Option<usize>,
        dft: &DFT,
    ) -> ProofResult<MultilinearPoint<EF>>
    where
        F: PrimeField64 + TwoAdicField,
        EF: ExtensionField<F> + TwoAdicField,
        S: PowStrategy,
        DFT: TwoAdicSubgroupDft<F>,
    {
        assert!(
            self.num_variables() >= folding_factor,
            "Not enough variables to fold"
        );
        let mut res = Vec::with_capacity(folding_factor);

        // Optional univariate skip
        let skip = match k_skip {
            // We apply a k-skip if:
            // - k >= 2 (to avoid using the univariate skip uselessly)
            // - k <= folding_factor (to be sure we don't skip more rounds than folding factor)
            Some(k) if k >= 2 && k <= folding_factor => {
                // Apply univariate skip over the first k variables
                let sumcheck_poly = self.compute_skipping_sumcheck_polynomial(dft, k);
                prover_state.add_scalars(sumcheck_poly.evaluations())?;

                for _ in 0..k {
                    let [folding_randomness] = prover_state.challenge_scalars()?;
                    res.push(folding_randomness);

                    if pow_bits > 0. {
                        prover_state.challenge_pow::<S>(pow_bits)?;
                    }
                }

                // Apply k rounds of folding to p and weights
                let new_p = fold_k_times(
                    match &self.evaluation_of_p {
                        EvaluationStorage::Base(evals_f) => evals_f.evals(),
                        EvaluationStorage::Extension(_) => panic!(
                            "The univariate skip optimization should only occur in base field"
                        ),
                    },
                    &res,
                    k,
                );
                let new_weights = fold_k_times(self.weights.evals(), &res, k);

                self.evaluation_of_p = EvaluationStorage::Extension(EvaluationsList::new(new_p));
                self.weights = EvaluationsList::new(new_weights);

                let evals_mat = RowMajorMatrix::new(sumcheck_poly.evaluations().to_vec(), 1);
                let next_sum = interpolate_subgroup(&evals_mat, res[0])[0];

                // Update the sum state variable
                self.sum = next_sum;

                k
            }
            _ => 0, // No skip
        };

        for _ in skip..folding_factor {
            let sumcheck_poly = self.compute_sumcheck_polynomial();
            prover_state.add_scalars(sumcheck_poly.evaluations())?;
            let [folding_randomness] = prover_state.challenge_scalars()?;
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
    use p3_dft::NaiveDft;
    use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, extension::BinomialExtensionField};
    use p3_interpolation::interpolate_subgroup;
    use p3_matrix::dense::RowMajorMatrix;
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
        // ----------------------------------------------------------------
        // Step 0: Define the multilinear polynomial
        //
        // f(X₀, X₁) = c₁ + c₂·X₁ + c₃·X₀ + c₄·X₀·X₁
        //
        // with coefficients:
        //   c₁ = 1
        //   c₂ = 2
        //   c₃ = 3
        //   c₄ = 4
        // ----------------------------------------------------------------
        let c1 = F::from_u64(1);
        let c2 = F::from_u64(2);
        let c3 = F::from_u64(3);
        let c4 = F::from_u64(4);

        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4]);

        // ----------------------------------------------------------------
        // Step 1: Define an equality constraint
        //
        // Constraint: f(1,0) = 5
        // Using a Weights::evaluation at the point (X₀, X₁) = (1,0).
        // ----------------------------------------------------------------
        let mut statement = Statement::new(2);
        let point = MultilinearPoint(vec![F::ONE, F::ZERO]);
        let weights = Weights::evaluation(point);
        let eval = F::from_u64(5);
        statement.add_constraint(weights, eval);

        // Instantiate prover
        let prover = SumcheckSingle::from_base_coeffs(coeffs, &statement, F::ONE);

        // Compute the sumcheck polynomial h(X)
        let sumcheck_poly = prover.compute_sumcheck_polynomial();

        // Check sum consistency
        assert_eq!(prover.sum, eval);

        // ----------------------------------------------------------------
        // Step 2: Define the polynomials f and w manually
        //
        // - f(x₀,x₁) = c₁ + c₂·x₁ + c₃·x₀ + c₄·x₀·x₁
        //
        // - w(x₀,x₁) is the Lagrange interpolation enforcing (x₀,x₁) = (1,0):
        //   w(x₀,x₁) = x₀ · (1 - x₁)
        //
        // This satisfies:
        //  - w(1,0) = 1
        //  - w(x) = 0 elsewhere
        // ----------------------------------------------------------------
        let f = |x0: F, x1: F| c1 + c2 * x1 + c3 * x0 + c4 * x0 * x1;
        let w = |x0: F, x1: F| x0 * (F::ONE - x1);

        // ----------------------------------------------------------------
        // Step 3: Evaluate f and w at all binary points (0/1 for x₀, x₁)
        // ----------------------------------------------------------------
        let f_00 = f(F::ZERO, F::ZERO);
        let f_01 = f(F::ZERO, F::ONE);
        let f_10 = f(F::ONE, F::ZERO);
        let f_11 = f(F::ONE, F::ONE);

        let w_00 = w(F::ZERO, F::ZERO);
        let w_01 = w(F::ZERO, F::ONE);
        let w_10 = w(F::ONE, F::ZERO);
        let w_11 = w(F::ONE, F::ONE);

        // ----------------------------------------------------------------
        // Step 4: Manually reconstruct the quadratic sumcheck polynomial
        //
        // We want h(X) = quadratic polynomial satisfying:
        //
        //      h(X₀) = ∑_{X₁ ∈ {0,1}} f(X₀,X₁) · w(X₀,X₁)
        //
        // which can be interpolated from:
        //
        // - c₀ = constant coefficient
        // - c₂ = quadratic coefficient
        // - c₁ = determined from sum rule: prover.sum = 2·c₀ + c₁ + c₂
        //
        // More precisely:
        //
        // \[
        // c₀ = (f₀₀ × w₀₀) + (f₁₀ × w₁₀)
        // \]
        //
        // \[
        // c₂ = (f₀₁ - f₀₀) × (w₀₁ - w₀₀) + (f₁₁ - f₁₀) × (w₁₁ - w₁₀)
        // \]
        //
        // \[
        // c₁ = sum - 2·c₀ - c₂
        // \]
        // ----------------------------------------------------------------

        let e0 = f_00 * w_00 + f_10 * w_10; // Constant term
        let e2 = (f_01 - f_00) * (w_01 - w_00) + (f_11 - f_10) * (w_11 - w_10); // Quadratic term
        let e1 = prover.sum - e0.double() - e2; // Middle coefficient using sum rule

        // ----------------------------------------------------------------
        // Step 5: Evaluate the quadratic polynomial at {0,1,2}
        //
        // - h(0) = c₀
        // - h(1) = c₀ + c₁ + c₂
        // - h(2) = h(1) + c₁ + c₂ + 2c₂
        // ----------------------------------------------------------------

        let eval_0 = e0;
        let eval_1 = e0 + e1 + e2;
        let eval_2 = eval_1 + e1 + e2 + e2.double();

        let expected_evaluations = vec![eval_0, eval_1, eval_2];

        // ----------------------------------------------------------------
        // Step 6: Assert final match
        // ----------------------------------------------------------------
        assert_eq!(sumcheck_poly.evaluations(), &expected_evaluations);
    }

    #[test]
    fn test_compute_sumcheck_polynomial_with_equality_constraints_3vars() {
        // Step 1: Define a multilinear polynomial with three variables:
        //
        //   f(X₀, X₁, X₂) = c₀
        //                 + c₁·X₂
        //                 + c₂·X₁
        //                 + c₃·X₁·X₂
        //                 + c₄·X₀
        //                 + c₅·X₀·X₂
        //                 + c₆·X₀·X₁
        //                 + c₇·X₀·X₁·X₂
        //
        // Coefficients:
        //   - c₀ = 1
        //   - c₁ = 2
        //   - c₂ = 3
        //   - c₃ = 4
        //   - c₄ = 5
        //   - c₅ = 6
        //   - c₆ = 7
        //   - c₇ = 8
        let c0 = F::from_u64(1);
        let c1 = F::from_u64(2);
        let c2 = F::from_u64(3);
        let c3 = F::from_u64(4);
        let c4 = F::from_u64(5);
        let c5 = F::from_u64(6);
        let c6 = F::from_u64(7);
        let c7 = F::from_u64(8);

        let coeffs = CoefficientList::new(vec![c0, c1, c2, c3, c4, c5, c6, c7]);

        // Step 2: Set up a sumcheck statement with a **single equality constraint**.
        //
        // The constraint is at the point (X₀,X₁,X₂) = (1,0,1),
        // with expected evaluation value = 5.
        let mut statement = Statement::new(3);
        let point = MultilinearPoint(vec![F::ONE, F::ZERO, F::ONE]);
        let weights = Weights::evaluation(point);

        let expected_sum = F::from_u64(5);
        statement.add_constraint(weights, expected_sum);

        // Build prover and compute sumcheck polynomial.
        let prover = SumcheckSingle::from_base_coeffs(coeffs, &statement, F::ONE);
        let sumcheck_poly = prover.compute_sumcheck_polynomial();

        // Sanity check: sum must match the constraint
        assert_eq!(prover.sum, expected_sum);

        // Step 3: Define f(x₀,x₁,x₂) explicitly as a function
        let f = |x0: F, x1: F, x2: F| {
            c0 + c1 * x2
                + c2 * x1
                + c3 * x1 * x2
                + c4 * x0
                + c5 * x0 * x2
                + c6 * x0 * x1
                + c7 * x0 * x1 * x2
        };

        // Step 4: Manually compute f at all 8 binary points (0,1)^3
        let ep_000 = f(F::ZERO, F::ZERO, F::ZERO);
        let ep_001 = f(F::ZERO, F::ZERO, F::ONE);
        let ep_010 = f(F::ZERO, F::ONE, F::ZERO);
        let ep_011 = f(F::ZERO, F::ONE, F::ONE);
        let ep_100 = f(F::ONE, F::ZERO, F::ZERO);
        let ep_101 = f(F::ONE, F::ZERO, F::ONE);
        let ep_110 = f(F::ONE, F::ONE, F::ZERO);
        let ep_111 = f(F::ONE, F::ONE, F::ONE);

        // Step 5: Compute the evaluations of the **equality constraint polynomial**.
        //
        // The equality constraint enforces X = (1,0,1).
        // The equality polynomial eq_{(1,0,1)}(X₀,X₁,X₂) evaluates to:
        //   - 1 when (X₀,X₁,X₂) == (1,0,1)
        //   - 0 elsewhere
        //
        // Thus:
        //   - Only point (1,0,1) gets a 1
        //   - All other 7 points are 0
        let w_000 = F::ZERO; // eq(0,0,0) = 0
        let w_001 = F::ZERO; // eq(0,0,1) = 0
        let w_010 = F::ZERO; // eq(0,1,0) = 0
        let w_011 = F::ZERO; // eq(0,1,1) = 0
        let w_100 = F::ZERO; // eq(1,0,0) = 0
        let w_101 = F::ONE; // eq(1,0,1) = 1 (this is the constraint point)
        let w_110 = F::ZERO; // eq(1,1,0) = 0
        let w_111 = F::ZERO; // eq(1,1,1) = 0

        // ----------------------------------------------------------------
        // Step 6: Manually compute the coefficients (e₀, e₁, e₂) of the sumcheck polynomial
        //
        // Recall:
        // - e₀ = sum of f(x)·w(x) where x₀ = 0
        // - e₁ and e₂ derived from interpolation using prover.sum
        //
        // Detailed formulas:
        //   e₀ = (ep_000 * w_000) + (ep_010 * w_010) + (ep_100 * w_100) + (ep_110 * w_110)
        //       = 0
        //
        //   e₂ = (ep_001 - ep_000) * (w_001 - w_000)
        //       + (ep_011 - ep_010) * (w_011 - w_010)
        //       + (ep_101 - ep_100) * (w_101 - w_100)
        //       + (ep_111 - ep_110) * (w_111 - w_110)
        //
        //   e₁ = sum - 2*e₀ - e₂
        // ----------------------------------------------------------------
        let e0 = ep_000 * w_000 + ep_010 * w_010 + ep_100 * w_100 + ep_110 * w_110; // = 0
        let e2 = (ep_001 - ep_000) * (w_001 - w_000)
            + (ep_011 - ep_010) * (w_011 - w_010)
            + (ep_101 - ep_100) * (w_101 - w_100)
            + (ep_111 - ep_110) * (w_111 - w_110);
        let e1 = prover.sum - e0.double() - e2;

        // ----------------------------------------------------------------
        // Step 7: Manually compute evaluations of sumcheck polynomial at {0,1,2}
        //
        // Recall:
        //   h(0) = e₀
        //   h(1) = e₀ + e₁ + e₂
        //   h(2) = h(1) + e₁ + e₂ + 2e₂
        // ----------------------------------------------------------------
        let eval_0 = e0;
        let eval_1 = e0 + e1 + e2;
        let eval_2 = eval_1 + e1 + e2 + e2.double();

        let expected_evaluations = vec![eval_0, eval_1, eval_2];

        // Final assertion: check that the computed polynomial matches manual expansion
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

        // Create a statement with a single equality constraint: f(1) = 5
        let mut statement = Statement::new(1);
        let point = MultilinearPoint(vec![F::ONE]);
        let weights = Weights::evaluation(point);
        let eval = F::from_u64(5);
        statement.add_constraint(weights, eval);

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

        // Check sum BEFORE running protocol
        assert_eq!(prover.sum, eval);

        // Compute sumcheck polynomials
        let result = prover
            .compute_sumcheck_polynomials::<Blake3PoW, _>(
                &mut prover_state,
                folding_factor,
                pow_bits,
                None,
                &NaiveDft,
            )
            .unwrap();

        // The result should contain `folding_factor` elements
        assert_eq!(result.0.len(), folding_factor);

        // Reconstruct verifier state to manually validate the sumcheck round
        let mut verifier_state = domsep.to_verifier_state(prover_state.narg_string());

        // Read the sumcheck polynomial evaluations: h(0), h(1), h(2)
        let sumcheck_poly_evals: [_; 3] = verifier_state.next_scalars().unwrap();
        let sumcheck_poly = SumcheckPolynomial::new(sumcheck_poly_evals.to_vec(), 1);

        // Read the folding randomness challenge
        let [folding_randomness] = verifier_state.challenge_scalars().unwrap();

        // Check that sumcheck polynomial satisfies the sum rule:
        //  h(0) + h(1) = claimed initial sum = eval = 5
        let sum = sumcheck_poly_evals[0] + sumcheck_poly_evals[1];
        assert_eq!(sum, eval);

        // Check that the folded sum stored in prover matches h(r)
        let expected_folded_sum = sumcheck_poly.evaluate_at_point(&folding_randomness.into());
        assert_eq!(prover.sum, expected_folded_sum);
    }

    #[test]
    fn test_compute_sumcheck_polynomials_with_multiple_folding_factors() {
        // Polynomial with 2 variables: f(X1, X2) = 1 + 2*X1 + 3*X2 + 4*X1*X2
        let c1 = F::from_u64(1);
        let c2 = F::from_u64(2);
        let c3 = F::from_u64(3);
        let c4 = F::from_u64(3);
        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4]);

        // Add two equality constraints:
        //  - f(0, 1) = 4
        //  - f(1, 0) = 5
        let mut statement = Statement::new(2);
        let point1 = MultilinearPoint(vec![F::ZERO, F::ONE]); // (X1=0, X2=1)
        let point2 = MultilinearPoint(vec![F::ONE, F::ZERO]); // (X1=1, X2=0)
        let eval1 = F::from_u64(4);
        let eval2 = F::from_u64(5);
        statement.add_constraint(Weights::evaluation(point1), eval1);
        statement.add_constraint(Weights::evaluation(point2), eval2);

        // Instantiate prover
        let mut prover = SumcheckSingle::from_base_coeffs(coeffs, &statement, F::ONE);

        // Record the initial sum = expected combination of constraints
        let expected_initial_sum = eval1 + eval2;
        assert_eq!(prover.sum, expected_initial_sum);

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
            .compute_sumcheck_polynomials::<Blake3PoW, _>(
                &mut prover_state,
                folding_factor,
                pow_bits,
                None,
                &NaiveDft,
            )
            .unwrap();

        // Ensure we get `folding_factor` sampled randomness values
        assert_eq!(result.0.len(), folding_factor);

        // Reconstruct verifier state for round-by-round checks
        let mut verifier_state = domsep.to_verifier_state(prover_state.narg_string());

        // Initialize claimed sum with the expected initial value from constraints (before any folding)
        let mut current_sum = expected_initial_sum;

        for i in 0..folding_factor {
            // Step 1: Read the polynomial sent in this round

            // The prover sends 3 evaluations of a degree-1 polynomial h_i over {0,1,2}
            // These are evaluations at points 0, 1, 2, stored in lexicographic ternary order
            let sumcheck_evals: [_; 3] = verifier_state.next_scalars().unwrap();

            // Create a SumcheckPolynomial over 1 variable with those 3 values
            let poly = SumcheckPolynomial::new(sumcheck_evals.to_vec(), 1);

            // Step 2: Verifier checks sum over Boolean hypercube {0,1}^1
            // This ensures that:
            //     h_i(0) + h_i(1) == current_sum
            // where h_i is evaluated at x = 0 and x = 1 (not 2!)
            let sum = poly.evaluations()[0] + poly.evaluations()[1];
            assert_eq!(
                sum, current_sum,
                "Sumcheck round {i}: sum rule failed (h(0) + h(1) != current_sum)"
            );

            // Step 3: Verifier samples next challenge r_i ∈ F to fold
            let [r] = verifier_state.challenge_scalars().unwrap();

            // Step 4: Evaluate the sumcheck polynomial at r_i to compute new folded sum
            // The polynomial h_i is evaluated at x = r_i ∈ F (can be non-{0,1,2})
            current_sum = poly.evaluate_at_point(&r.into());

            // Step 5: Optional proof-of-work grinding
            // If `pow_bits > 0`, we enforce entropy in Fiat-Shamir via grinding
            verifier_state.challenge_pow::<Blake3PoW>(pow_bits).unwrap();

            // End of round i
        }

        // Final check: the sum stored by the prover must match the last folded sum value
        assert_eq!(
            prover.sum, current_sum,
            "Final folded sum does not match prover's claimed value"
        );
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

        // Add two equality constraints:
        // - f(0, 0, 1) = 4
        // - f(1, 1, 0) = 25
        let mut statement = Statement::new(3);
        let point1 = MultilinearPoint(vec![F::ZERO, F::ZERO, F::ONE]); // (X1=0, X2=0, X3=1)
        let point2 = MultilinearPoint(vec![F::ONE, F::ONE, F::ZERO]); // (X1=1, X2=1, X3=0)
        let eval1 = F::from_u64(4);
        let eval2 = F::from_u64(25);
        statement.add_constraint(Weights::evaluation(point1), eval1);
        statement.add_constraint(Weights::evaluation(point2), eval2);

        // Instantiate prover
        let mut prover = SumcheckSingle::from_base_coeffs(coeffs, &statement, F::ONE);

        let expected_initial_sum = eval1 + eval2;
        assert_eq!(prover.sum, expected_initial_sum);

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
            .compute_sumcheck_polynomials::<Blake3PoW, _>(
                &mut prover_state,
                folding_factor,
                pow_bits,
                None,
                &NaiveDft,
            )
            .unwrap();

        // There should be exactly `folding_factor` sumcheck polynomials
        assert_eq!(result.0.len(), folding_factor);

        // Initialize the verifier state for checking round-by-round
        let mut verifier_state = domsep.to_verifier_state(prover_state.narg_string());

        // Initialize the sum to be verified round-by-round
        let mut current_sum = expected_initial_sum;

        for i in 0..folding_factor {
            // Read the 3 evaluations of the sumcheck polynomial for this round
            let sumcheck_evals: [_; 3] = verifier_state.next_scalars().unwrap();

            // Construct the polynomial h_i(X) over 1 variable with those evaluations
            let poly = SumcheckPolynomial::new(sumcheck_evals.to_vec(), 1);

            // Check that h_i(0) + h_i(1) equals the claimed current sum
            let sum = poly.evaluations()[0] + poly.evaluations()[1];
            assert_eq!(
                sum, current_sum,
                "Sumcheck round {i}: sum rule failed (h(0) + h(1) != current_sum)"
            );

            // Sample the next folding challenge r_i ∈ F
            let [r] = verifier_state.challenge_scalars().unwrap();

            // Fold the polynomial at r_i to get new claimed sum
            current_sum = poly.evaluate_at_point(&r.into());

            // Perform proof-of-work grinding check
            verifier_state.challenge_pow::<Blake3PoW>(pow_bits).unwrap();
        }

        // After all rounds, the prover’s stored folded sum must match what verifier has
        assert_eq!(
            prover.sum, current_sum,
            "Final folded sum does not match prover's claimed value"
        );
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

        // Construct a statement with two equality constraints:
        // f(0, 1) = 1 + 0 + 3*1 + 0 = 4
        // f(1, 1) = 1 + 2*1 + 3*1 + 4*1 = 10
        let mut statement = Statement::new(2);
        let point1 = MultilinearPoint(vec![F::ZERO, F::ONE]);
        let point2 = MultilinearPoint(vec![F::ONE, F::ONE]);
        let eval1 = F::from_u64(4);
        let eval2 = F::from_u64(10);
        statement.add_constraint(Weights::evaluation(point1), eval1);
        statement.add_constraint(Weights::evaluation(point2), eval2);

        // Instantiate the prover with the polynomial and constraint statement
        let mut prover = SumcheckSingle::from_base_coeffs(coeffs, &statement, F::ONE);

        // With 0 folding rounds, this is an edge case: no polynomial rounds will be generated
        let folding_factor = 0;
        let pow_bits = 1.;

        // No domain separator logic needed since we don't fold
        let domsep: DomainSeparator<F, F, DefaultHash> = DomainSeparator::new("test");
        let mut prover_state = domsep.to_prover_state();

        let result = prover
            .compute_sumcheck_polynomials::<Blake3PoW, _>(
                &mut prover_state,
                folding_factor,
                pow_bits,
                None,
                &NaiveDft,
            )
            .unwrap();

        assert_eq!(result.0.len(), 0);

        // Verify that the prover's initial sum equals the expected constraint sum:
        // Since the two constraints are independent, the expected sum is:
        let expected_sum = eval1 + eval2;
        assert_eq!(
            prover.sum, expected_sum,
            "Prover's initial sum does not match expected constraint sum"
        );
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
    fn test_compute_sumcheck_polynomials_mixed_fields_three_vars1() {
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

        // Create a statement with 5 equality constraints
        let mut statement = Statement::new(3);
        let points = vec![
            MultilinearPoint(vec![EF4::ZERO, EF4::ZERO, EF4::ZERO]), // f(0,0,0) = 1
            MultilinearPoint(vec![EF4::ONE, EF4::ZERO, EF4::ZERO]),  // f(1,0,0) // f(1,0,0) = 3
            MultilinearPoint(vec![EF4::ONE, EF4::ONE, EF4::ZERO]),   // f(1,1,0) = // f(1,1,0) = 11
            MultilinearPoint(vec![EF4::ONE, EF4::ONE, EF4::ONE]),    // f(1,1,1) = 36
            MultilinearPoint(vec![EF4::ZERO, EF4::ONE, EF4::ONE]),   // f(0,1,1) = 14
        ];
        let expected_evals: Vec<_> = vec![1, 3, 11, 36, 14]
            .into_iter()
            .map(EF4::from_u64)
            .collect();
        for (pt, ev) in points.into_iter().zip(expected_evals) {
            statement.add_constraint(Weights::evaluation(pt), ev);
        }

        // Instantiate the Sumcheck prover using base field coefficients and extension field EF4
        let mut prover = SumcheckSingle::<F, EF4>::from_base_coeffs(coeffs, &statement, EF4::ONE);

        // Compute expected sum of evaluations from constraints
        let expected_initial_sum: EF4 = EF4::from_u64(1)
            + EF4::from_u64(3)
            + EF4::from_u64(11)
            + EF4::from_u64(36)
            + EF4::from_u64(14);
        assert_eq!(prover.sum, expected_initial_sum);

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
            .compute_sumcheck_polynomials::<Blake3PoW, _>(
                &mut prover_state,
                folding_factor,
                pow_bits,
                None,
                &NaiveDft,
            )
            .unwrap();

        // Ensure we received the expected number of folding randomness values
        assert_eq!(result.0.len(), folding_factor);

        // Reconstruct verifier state to simulate the rounds
        let mut verifier_state = domsep.to_verifier_state(prover_state.narg_string());

        // Start with the claimed sum before folding
        let mut current_sum = expected_initial_sum;

        for i in 0..folding_factor {
            // Get the 3 evaluations of sumcheck polynomial h_i(X) at X = 0, 1, 2
            let sumcheck_evals: [_; 3] = verifier_state.next_scalars().unwrap();
            let poly = SumcheckPolynomial::new(sumcheck_evals.to_vec(), 1);

            // Verify sum over Boolean points {0,1} matches current sum
            let sum = poly.evaluations()[0] + poly.evaluations()[1];
            assert_eq!(
                sum, current_sum,
                "Sumcheck round {i}: sum rule failed (h(0) + h(1) != current_sum)"
            );

            // Sample random challenge r_i ∈ F and evaluate h_i(r_i)
            let [r] = verifier_state.challenge_scalars().unwrap();
            current_sum = poly.evaluate_at_point(&r.into());

            // Apply grinding check if required
            verifier_state.challenge_pow::<Blake3PoW>(pow_bits).unwrap();
        }

        // Final consistency check: last folded sum must match prover's final claimed sum
        assert_eq!(
            prover.sum, current_sum,
            "Final folded sum does not match prover's claimed value"
        );
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
            let base_cl = CoefficientList::new(coeffs);
            let ext_cl = CoefficientList::new(coeffs_ext);

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
            let base_cl = CoefficientList::new(coeffs);
            let ext_cl = CoefficientList::new(coeffs_ext);

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
                    EvaluationStorage::Base(_) => panic!("Expected extension evaluations"),
                },
                match ext_prover.evaluation_of_p {
                    EvaluationStorage::Extension(ref evals_e) => evals_e.evals(),
                    EvaluationStorage::Base(_) => panic!("Expected extension evaluations"),
                }
            );

            // Assert that the weights match
            prop_assert_eq!(base_prover.weights.evals(), ext_prover.weights.evals());
        }


        #[test]
        fn prop_compute_sumcheck_polynomials_roundtrip(
            raw_coeffs in prop::collection::vec(0u64..F::ORDER_U64, 8..=64)
                .prop_filter("len must be power of two and >= 8", |v| v.len().is_power_of_two()),
            combo_scalars in prop::array::uniform4(0u64..F::ORDER_U64),
        ) {
            // Convert base coefficients from u64 to BabyBear elements
            let coeffs: Vec<F> = raw_coeffs.iter().copied().map(F::from_u64).collect();

            // Convert base field coeffs into EF4 (element by element)
            let coeffs_ext: Vec<EF4> = coeffs.clone()
                .into_iter()
                .map(EF4::from)
                .collect();

            //  Wrap both coefficient representations
            let base_cl = CoefficientList::new(coeffs);
            let ext_cl = CoefficientList::new(coeffs_ext);

            // Determine how many variables exist (log₂ of length)
            let n_vars = base_cl.num_variables();
            prop_assume!(n_vars >= 1);
            let folding_rounds = n_vars;

            // Construct an empty constraint system
            let statement= Statement::new(n_vars);

            // Construct EF4 combination randomness from 4 base field values
            let combination_randomness = EF4::from_basis_coefficients_iter(
                combo_scalars.map(F::from_u64).into_iter()
            ).unwrap();

            // Create both provers
            let mut prover_base = SumcheckSingle::<F, EF4>::from_base_coeffs(base_cl, &statement, combination_randomness);
            let mut prover_ext = SumcheckSingle::<F, EF4>::from_extension_coeffs(ext_cl, &statement, combination_randomness);

            // Use a single shared DomainSeparator and clone it (identical transcript!)
            let mut domsep_base: DomainSeparator<EF4, F, DefaultHash> = DomainSeparator::new("tag");
            let mut domsep_ext:DomainSeparator<EF4, F, DefaultHash> = DomainSeparator::new("tag");

            // Register the same interactions for each folding round
            for _ in 0..folding_rounds {
                domsep_base.add_scalars(3, "tag");
                domsep_base.challenge_scalars(1, "tag");

                domsep_ext.add_scalars(3, "tag");
                domsep_ext.challenge_scalars(1, "tag");
            }


            // Convert into prover states
            let mut state_base = domsep_base.to_prover_state();
            let mut state_ext = domsep_ext.to_prover_state();

            // Run sumcheck with zero grinding (no challenge_pow)
            let final_point_base = prover_base
                .compute_sumcheck_polynomials::<Blake3PoW,_>(&mut state_base, folding_rounds, 0.0, None, &NaiveDft)
                .unwrap();

            let final_point_ext = prover_ext
                .compute_sumcheck_polynomials::<Blake3PoW,_>(&mut state_ext, folding_rounds, 0.0, None, &NaiveDft)
                .unwrap();

            // Ensure roundtrip consistency
            prop_assert_eq!(final_point_base.0, final_point_ext.0);
        }
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_compute_sumcheck_polynomials_mixed_fields_three_vars_with_skip() {
        // -------------------------------------------------------------
        // Define a multilinear polynomial in 3 variables:
        // f(X0, X1, X2) = 1 + 2*X2 + 3*X1 + 4*X1*X2
        //              + 5*X0 + 6*X0*X2 + 7*X0*X1 + 8*X0*X1*X2
        // -------------------------------------------------------------
        let c0 = F::from_u64(1);
        let c1 = F::from_u64(2);
        let c2 = F::from_u64(3);
        let c3 = F::from_u64(4);
        let c4 = F::from_u64(5);
        let c5 = F::from_u64(6);
        let c6 = F::from_u64(7);
        let c7 = F::from_u64(8);
        let coeffs = CoefficientList::new(vec![c0, c1, c2, c3, c4, c5, c6, c7]);

        // A closure representing the polynomial for evaluation at points
        let f = |x0: F, x1: F, x2: F| {
            c0 + c1 * x2
                + c2 * x1
                + c3 * x1 * x2
                + c4 * x0
                + c5 * x0 * x2
                + c6 * x0 * x1
                + c7 * x0 * x1 * x2
        };

        // -------------------------------------------------------------
        // Construct an evaluation statement by specifying equality constraints
        // Each constraint is of the form f(x) = value for x in {0,1}^3
        // -------------------------------------------------------------
        let mut statement = Statement::new(3);
        statement.add_constraint(
            Weights::evaluation(MultilinearPoint(vec![EF4::ZERO, EF4::ZERO, EF4::ZERO])),
            EF4::from_u64(1),
        );
        statement.add_constraint(
            Weights::evaluation(MultilinearPoint(vec![EF4::ZERO, EF4::ZERO, EF4::ONE])),
            EF4::from_u64(3),
        );
        statement.add_constraint(
            Weights::evaluation(MultilinearPoint(vec![EF4::ZERO, EF4::ONE, EF4::ZERO])),
            EF4::from_u64(11),
        );
        statement.add_constraint(
            Weights::evaluation(MultilinearPoint(vec![EF4::ZERO, EF4::ONE, EF4::ONE])),
            EF4::from_u64(36),
        );
        statement.add_constraint(
            Weights::evaluation(MultilinearPoint(vec![EF4::ONE, EF4::ZERO, EF4::ZERO])),
            EF4::from_u64(14),
        );
        statement.add_constraint(
            Weights::evaluation(MultilinearPoint(vec![EF4::ONE, EF4::ZERO, EF4::ONE])),
            EF4::from_u64(13),
        );
        statement.add_constraint(
            Weights::evaluation(MultilinearPoint(vec![EF4::ONE, EF4::ONE, EF4::ZERO])),
            EF4::from_u64(4),
        );
        statement.add_constraint(
            Weights::evaluation(MultilinearPoint(vec![EF4::ONE, EF4::ONE, EF4::ONE])),
            EF4::from_u64(34),
        );

        // -------------------------------------------------------------
        // Create the prover instance using the coefficients and constraints
        // The prover evaluates the polynomial at all 8 Boolean points and stores results
        // -------------------------------------------------------------
        let mut prover = SumcheckSingle::<F, EF4>::from_base_coeffs(coeffs, &statement, EF4::ONE);

        // -------------------------------------------------------------
        // Evaluate the polynomial manually at all 8 input points
        // -------------------------------------------------------------
        let f_000 = f(F::ZERO, F::ZERO, F::ZERO);
        let f_001 = f(F::ZERO, F::ZERO, F::ONE);
        let f_010 = f(F::ZERO, F::ONE, F::ZERO);
        let f_011 = f(F::ZERO, F::ONE, F::ONE);
        let f_100 = f(F::ONE, F::ZERO, F::ZERO);
        let f_101 = f(F::ONE, F::ZERO, F::ONE);
        let f_110 = f(F::ONE, F::ONE, F::ZERO);
        let f_111 = f(F::ONE, F::ONE, F::ONE);

        // -------------------------------------------------------------
        // Check that prover internally stores evaluations correctly and weights are consistent
        // Each evaluation f(b) is scaled by its weight to enforce a constraint
        // -------------------------------------------------------------
        match prover.evaluation_of_p {
            EvaluationStorage::Base(ref eval_f) => {
                let f_evals = eval_f.evals();
                let weights_evals = prover.weights.evals();

                assert_eq!(
                    f_evals,
                    vec![f_000, f_001, f_010, f_011, f_100, f_101, f_110, f_111]
                );

                for i in 0..8 {
                    assert_eq!(weights_evals[i] * f_evals[i], EF4::from(f_evals[i]));
                }
            }
            EvaluationStorage::Extension(_) => {
                panic!("We should be in base field here");
            }
        }

        // -------------------------------------------------------------
        // Manually compute the expected weighted sum (constraint enforcement)
        // The sumcheck protocol must maintain this sum across folds
        // -------------------------------------------------------------
        let expected_sum = EF4::from_u64(1)
            + EF4::from_u64(3)
            + EF4::from_u64(11)
            + EF4::from_u64(36)
            + EF4::from_u64(14)
            + EF4::from_u64(13)
            + EF4::from_u64(4)
            + EF4::from_u64(34);
        assert_eq!(prover.sum, expected_sum);

        // -------------------------------------------------------------
        // Set up sumcheck protocol with:
        // - 3 rounds of folding (equal to 3 variables)
        // - 2-round univariate skip enabled
        // -------------------------------------------------------------
        let folding_factor = 3;
        let pow_bits = 0.;

        // Create domain separator for Fiat-Shamir transcript simulation
        let mut domsep: DomainSeparator<EF4, F, DefaultHash> = DomainSeparator::new("test");

        // Step 1: absorb 3 evaluations of the sumcheck polynomial h(X)
        domsep.add_scalars(8, "tag");

        // Step 2: derive a folding challenge scalar from transcript
        domsep.challenge_scalars(1, "tag");
        domsep.challenge_scalars(1, "tag");

        // Step 1: absorb 3 evaluations of the sumcheck polynomial h(X)
        domsep.add_scalars(3, "tag");

        // Step 2: derive a folding challenge scalar from transcript
        domsep.challenge_scalars(1, "tag");

        // Convert domain separator into prover state object
        let mut prover_state = domsep.to_prover_state();

        // Run sumcheck with k = 2 skipped rounds and 1 regular round
        let result = prover
            .compute_sumcheck_polynomials::<Blake3PoW, _>(
                &mut prover_state,
                folding_factor,
                pow_bits,
                Some(2), // skip 2 variables at once
                &NaiveDft,
            )
            .unwrap();

        // -------------------------------------------------------------
        // Ensure we received exactly 3 challenge points (folding_factor)
        // -------------------------------------------------------------
        assert_eq!(result.0.len(), 3);

        // -------------------------------------------------------------
        // Replay verifier's side using same Fiat-Shamir transcript
        // -------------------------------------------------------------
        let mut verifier_state = domsep.to_verifier_state(prover_state.narg_string());
        let mut current_sum = expected_sum;

        // Get the 8 evaluations of the skipping polynomial h₀(X)
        let sumcheck_evals: [_; 8] = verifier_state.next_scalars().unwrap();
        let poly = SumcheckPolynomial::new(sumcheck_evals.to_vec(), 1);

        // Here we can take a multiplicative subgroup of size 4 (omega^4)
        let omega4 = F::two_adic_generator(2);

        // Interpolate the polynomial over the subgroup
        let evals_mat = RowMajorMatrix::new(poly.evaluations().to_vec(), 1);
        let interpolation_0 = interpolate_subgroup(&evals_mat, EF4::from(omega4.exp_u64(0)))[0];
        let interpolation_1 = interpolate_subgroup(&evals_mat, EF4::from(omega4.exp_u64(1)))[0];
        let interpolation_2 = interpolate_subgroup(&evals_mat, EF4::from(omega4.exp_u64(2)))[0];
        let interpolation_3 = interpolate_subgroup(&evals_mat, EF4::from(omega4.exp_u64(3)))[0];

        // Compute the sum from the interpolations and compare with the expected sum
        let sum_interpolation =
            interpolation_0 + interpolation_1 + interpolation_2 + interpolation_3;
        assert_eq!(sum_interpolation, current_sum);

        // Interpolate h₀(X) and update current sum using first challenge r₀
        let evals_mat = RowMajorMatrix::new(poly.evaluations().to_vec(), 1);
        let [r] = verifier_state.challenge_scalars().unwrap();
        let [_] = verifier_state.challenge_scalars().unwrap(); // skip dummy

        current_sum = interpolate_subgroup(&evals_mat, r)[0];

        // -------------------------------------------------------------
        // Continue with round 2: regular quadratic sumcheck step
        // h₁(X) must satisfy h₁(0) + h₁(1) == current_sum
        // -------------------------------------------------------------
        for i in 2..folding_factor {
            let sumcheck_evals: [_; 3] = verifier_state.next_scalars().unwrap();
            let poly = SumcheckPolynomial::new(sumcheck_evals.to_vec(), 1);

            let sum = poly.evaluations()[0] + poly.evaluations()[1];
            assert_eq!(
                sum, current_sum,
                "Sumcheck round {i}: h(0) + h(1) != current_sum"
            );

            let [r] = verifier_state.challenge_scalars().unwrap();
            current_sum = poly.evaluate_at_point(&r.into());
        }

        // Final consistency check: does prover's internal `sum` match verifier’s result?
        assert_eq!(
            prover.sum, current_sum,
            "Final prover sum doesn't match verifier folding result"
        );
    }
}
