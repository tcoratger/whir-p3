use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
#[cfg(feature = "parallel")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};

use super::{sumcheck_polynomial::SumcheckPolynomial, sumcheck_single::SumcheckSingle};
use crate::poly::evals::EvaluationStorage;

impl<F, EF> SumcheckSingle<F, EF>
where
    F: Field + TwoAdicField,
    EF: ExtensionField<F>,
{
    /// Computes the sumcheck polynomial using the **univariate skip** optimization,
    /// folding the first `k` variables in a single step, and returns intermediate matrices for reuse.
    ///
    /// This method skips `k` rounds of classical sumcheck by evaluating the polynomial over
    /// a structured domain of the form `D × H^{n-k}`, where `D` is a multiplicative coset of size `2^{k+1}`.
    ///
    /// It returns a univariate polynomial `h(X)` of degree ≤ `2^k - 1`, defined as:
    ///
    /// \begin{equation}
    /// h(X) = \sum_{b \in \{0,1\}^{n-k}} p(X, b) \cdot w(X, b)
    /// \end{equation}
    ///
    /// where `X` ranges over the coset and each `b` represents an assignment to the remaining variables.
    ///
    /// # Inputs
    /// - `dft`: A two-adic DFT backend used to perform low-degree extension over a coset domain.
    /// - `k`: Number of initial variables to skip via univariate folding.
    ///
    /// # Output
    /// - A tuple:
    ///   - `SumcheckPolynomial<EF>`: The folded univariate polynomial evaluated over the coset.
    ///   - `RowMajorMatrix<F>`: Matrix view of base field evaluations reshaped as `(2^k, 2^{n-k})`.
    ///   - `RowMajorMatrix<EF>`: Matrix view of extension field weights, reshaped similarly.
    ///
    /// # Constraints
    /// - This method requires base field evaluations (`EvaluationStorage::Base`).
    ///   It panics if the polynomial is already stored in the extension field.
    pub fn compute_skipping_sumcheck_polynomial<DFT>(
        &self,
        dft: &DFT,
        k: usize,
    ) -> (
        SumcheckPolynomial<EF>,
        RowMajorMatrix<F>,
        RowMajorMatrix<EF>,
    )
    where
        DFT: TwoAdicSubgroupDft<F>,
    {
        // Ensure we have enough variables to perform k-fold skipping.
        assert!(
            self.num_variables() >= k,
            "Need at least k variables to apply univariate skip on k variables"
        );

        // Evaluate based on the storage format of the polynomial:
        // Only base field evaluations are supported for skipping.
        let (out_vec, f, w) = match &self.evaluation_of_p {
            EvaluationStorage::Base(evals_f) => {
                // Reinterpret the flat evaluation vector as a matrix of shape:
                // - Rows: 2^k (assignments to the first k variables, which we skip)
                // - Columns: 2^{n-k} (assignments to the remaining variables)
                //
                // The transpose operation converts from row-major evaluations over {0,1}^n
                // into a layout where each row corresponds to a fixed setting of the remaining variables.
                let f_mat = RowMajorMatrix::new(evals_f.evals().to_vec(), 1 << k).transpose();
                // Similarly reshape weights to align with p
                let weights_mat =
                    RowMajorMatrix::new(self.weights.evals().to_vec(), 1 << k).transpose();

                // Apply low-degree extension (LDE) over a multiplicative coset of size 2^{k+1}
                // to both the function and weights matrices:
                // - Each row in f_mat is extended from size 2^k to 2^{k+1} over the coset domain.
                // - The same is done for weights_mat, but with values in the extension field EF.
                let f_on_coset = dft.lde_batch(f_mat.clone(), 1).to_row_major_matrix();
                let weights_on_coset = dft
                    .lde_algebra_batch(weights_mat.clone(), 1)
                    .to_row_major_matrix();

                // After LDE, each row corresponds to a fixed assignment to the remaining variables,
                // and contains the univariate evaluations over the coset (length 2^{k+1}).
                //
                // We now compute the sumcheck polynomial:
                // - For each row, compute the pointwise product f(x) * w(x)
                // - Then sum across x in the coset to collapse that row into a scalar
                //
                // This yields one evaluation of the final univariate sumcheck polynomial per row.
                let result: Vec<EF> = f_on_coset
                    .par_row_slices()
                    .zip(weights_on_coset.par_row_slices())
                    .map(|(coeffs_row, weights_row)| {
                        coeffs_row
                            .iter()
                            .zip(weights_row.iter())
                            .map(|(&c, &w)| w * c)
                            .sum()
                    })
                    .collect();

                (result, f_mat, weights_mat)
            }

            // If the polynomial is already in extension form, univariate skip is not valid.
            EvaluationStorage::Extension(_) => {
                panic!("The univariate skip optimization should only occur in base field")
            }
        };

        // Return the sumcheck polynomial and the intermediate pre-LDE matrices

        (SumcheckPolynomial::new(out_vec, 1), f, w)
    }
}

#[cfg(test)]
#[allow(clippy::erasing_op, clippy::identity_op)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_dft::NaiveDft;
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};

    use super::*;
    use crate::{
        poly::{coeffs::CoefficientList, multilinear::MultilinearPoint},
        whir::statement::{Statement, Weights},
    };

    type F = BabyBear;
    type EF4 = BinomialExtensionField<BabyBear, 4>;
    type Dft = NaiveDft;

    #[test]
    fn test_skipping_sumcheck_polynomial_k1() {
        // ----------------------------------------------------------------
        // Polynomial f(X0, X1) = 1 + 2*X1 + 3*X0 + 4*X0*X1
        // Coefficient order: [1, 2, 3, 4] for monomials:
        // [1, X1, X0, X0*X1]
        //
        // Interpretation:
        // This is a bilinear polynomial (degree ≤ 1 in each variable).
        // We'll use it to test the univariate skip sumcheck with k = 1.
        // ----------------------------------------------------------------

        let c0 = F::from_u64(1);
        let c1 = F::from_u64(2);
        let c2 = F::from_u64(3);
        let c3 = F::from_u64(4);
        let coeffs = CoefficientList::new(vec![c0, c1, c2, c3]);

        // ----------------------------------------------------------------
        // Statement has no constraints (zero constraint polynomial).
        // So the weight polynomial w(X) = 0 for all X ∈ {0,1}^2.
        //
        // That means f(X) · w(X) = 0 everywhere → result is the zero polynomial.
        // ----------------------------------------------------------------
        let statement = Statement::<EF4>::new(2);
        let prover = SumcheckSingle::<F, EF4>::from_base_coeffs(coeffs, &statement, EF4::ONE);

        // ----------------------------------------------------------------
        // We perform the univariate skip with k = 1:
        // This skips 1 variable (X0), leaving a univariate polynomial in X1.
        //
        // Instead of evaluating over {0,1} (Boolean domain),
        // we extend the evaluation domain to a **multiplicative coset of size 4**
        // using `coset_lde_batch` (low-degree extension via DFT).
        // ----------------------------------------------------------------
        let dft = Dft::default();
        let (poly, _, _) = prover.compute_skipping_sumcheck_polynomial(&dft, 1);

        // ----------------------------------------------------------------
        // Sum over the Boolean hypercube {0,1}:
        // - Only includes values at X1 = 0 and X1 = 1
        // - Since the polynomial is zero everywhere, sum = 0
        // ----------------------------------------------------------------
        assert_eq!(poly.sum_over_boolean_hypercube(), EF4::ZERO);
    }

    #[test]
    fn test_skipping_sumcheck_polynomial_k2() {
        // ----------------------------------------------------------------
        // Polynomial f(X0, X1, X2) =
        //       1
        //     + 2·X2
        //     + 3·X1
        //     + 4·X1·X2
        //     + 5·X0
        //
        // Coefficients are ordered by lexicographic order of (X0, X1, X2):
        //   [1, X2, X1, X1·X2, X0, X0·X2, X0·X1, X0·X1·X2]
        // Index-to-monomial mapping:
        //   - c0 = 1
        //   - c1 = 2        ← X2
        //   - c2 = 3        ← X1
        //   - c3 = 4        ← X1·X2
        //   - c4 = 5        ← X0
        //   - c5 = 0        ← X0·X2
        //   - c6 = 0        ← X0·X1
        //   - c7 = 0        ← X0·X1·X2
        // ----------------------------------------------------------------

        let coeffs = CoefficientList::new(vec![
            F::from_u64(1), // 1
            F::from_u64(2), // X2
            F::from_u64(3), // X1
            F::from_u64(4), // X1·X2
            F::from_u64(5), // X0
            F::ZERO,        // X0·X2
            F::ZERO,        // X0·X1
            F::ZERO,        // X0·X1·X2
        ]);

        // ----------------------------------------------------------------
        // Statement has no constraints → weight polynomial w(X) = 0
        // Therefore the product f(X)·w(X) = 0 for all X ∈ {0,1}³
        // So the resulting sumcheck polynomial must be identically zero.
        // ----------------------------------------------------------------
        let statement = Statement::<EF4>::new(3);
        let prover = SumcheckSingle::<F, EF4>::from_base_coeffs(coeffs, &statement, EF4::ONE);

        // ----------------------------------------------------------------
        // Apply the univariate skip optimization with k = 2:
        //
        //   - This skips 2 variables: X0 and X1
        //   - The result is a univariate polynomial
        //
        // To evaluate this polynomial, we perform a low-degree extension
        // using DFT on a multiplicative coset of size 2^{k+1} = 8.
        // ----------------------------------------------------------------
        let dft = Dft::default();
        let (poly, _, _) = prover.compute_skipping_sumcheck_polynomial(&dft, 2);

        // ----------------------------------------------------------------
        // Finally, the sum over {0,1} values of X2 must also be zero
        // because the polynomial is identically zero on the full domain.
        // ----------------------------------------------------------------
        assert_eq!(poly.sum_over_boolean_hypercube(), EF4::ZERO);
    }

    #[test]
    #[should_panic]
    fn test_skipping_sumcheck_polynomial_panics_on_extension_input() {
        let ef1 = EF4::from(F::from_u64(1));
        let ef2 = EF4::from(F::from_u64(2));

        let coeffs = CoefficientList::new(vec![ef1, ef2]);
        let statement = Statement::<EF4>::new(1);
        let prover = SumcheckSingle::<F, EF4>::from_extension_coeffs(coeffs, &statement, EF4::ONE);
        let dft = Dft::default();

        // This should panic because the input is not in the base field
        let _ = prover.compute_skipping_sumcheck_polynomial(&dft, 1);
    }

    #[test]
    #[should_panic]
    fn test_skipping_sumcheck_polynomial_panics_on_invalid_k() {
        // Polynomial with only 1 variable, can't skip 2 variables
        let c0 = F::from_u64(1);
        let c1 = F::from_u64(2);
        let coeffs = CoefficientList::new(vec![c0, c1]);

        let statement = Statement::<EF4>::new(1);
        let prover = SumcheckSingle::<F, EF4>::from_base_coeffs(coeffs, &statement, EF4::ONE);
        let dft = Dft::default();

        // This should panic because:
        // - the polynomial has only 1 variable
        // - we try to skip 2 variables
        let _ = prover.compute_skipping_sumcheck_polynomial(&dft, 2);
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_skipping_sumcheck_polynomial_k2_weight() {
        // ----------------------------------------------------------------
        // Define multilinear polynomial f(X0, X1, X2):
        //
        //   f(X0, X1, X2) = 1
        //                 + 2·X2
        //                 + 3·X1
        //                 + 4·X1·X2
        //                 + 5·X0
        //                 + 6·X0·X2
        //                 + 7·X0·X1
        //                 + 8·X0·X1·X2
        //
        // Coefficient order: [1, X2, X1, X1·X2, X0, X0·X2, X0·X1, X0·X1·X2]
        // ----------------------------------------------------------------
        let c0 = F::from_u64(1);
        let c1 = F::from_u64(2);
        let c2 = F::from_u64(3);
        let c3 = F::from_u64(4);
        let c4 = F::from_u64(5);
        let c5 = F::from_u64(6);
        let c6 = F::from_u64(7);
        let c7 = F::from_u64(8);
        let coeffs = CoefficientList::new(vec![c0, c1, c2, c3, c4, c5, c6, c7]);

        // ------------------------------------------------------------
        // Manually evaluate f at each binary point
        // f(X0, X1, X2) = c0 + c1·X2 + c2·X1 + c3·X1·X2
        //              + c4·X0 + c5·X0·X2 + c6·X0·X1 + c7·X0·X1·X2
        // ------------------------------------------------------------
        let f_base = |x0: F, x1: F, x2: F| {
            c0 + c1 * x2
                + c2 * x1
                + c3 * x1 * x2
                + c4 * x0
                + c5 * x0 * x2
                + c6 * x0 * x1
                + c7 * x0 * x1 * x2
        };

        let f_extension = |x0: EF4, x1: EF4, x2: EF4| {
            x2 * c1
                + x1 * c2
                + x1 * x2 * c3
                + x0 * c4
                + x0 * x2 * c5
                + x0 * x1 * c6
                + x0 * x1 * x2 * c7
                + c0
        };

        // Constraints
        let mut statement = Statement::new(3);
        statement.add_constraint(
            Weights::evaluation(MultilinearPoint(vec![EF4::ZERO, EF4::ZERO, EF4::ZERO])),
            f_extension(EF4::ZERO, EF4::ZERO, EF4::ZERO),
        );
        statement.add_constraint(
            Weights::evaluation(MultilinearPoint(vec![EF4::ONE, EF4::ZERO, EF4::ONE])),
            f_extension(EF4::ONE, EF4::ZERO, EF4::ONE),
        );

        let prover = SumcheckSingle::<F, EF4>::from_base_coeffs(coeffs, &statement, EF4::ONE);

        // Get the f evaluations
        let evals_f = match prover.evaluation_of_p {
            EvaluationStorage::Base(ref evals_f) => evals_f.evals(),
            EvaluationStorage::Extension(_) => panic!("We should be in the base field"),
        };
        // Get the w evaluations
        let evals_w = prover.weights.evals();

        // Compute the expected sum manually via dot product
        let expected_sum = evals_w[0] * evals_f[0]
            + evals_w[1] * evals_f[1]
            + evals_w[2] * evals_f[2]
            + evals_w[3] * evals_f[3]
            + evals_w[4] * evals_f[4]
            + evals_w[5] * evals_f[5]
            + evals_w[6] * evals_f[6]
            + evals_w[7] * evals_f[7];

        // ------------------------------------------------------------
        // Apply univariate skip optimization with k = 2:
        // - This folds over variables X0 and X1
        // - The result is a univariate polynomial
        // - It will be evaluated over a multiplicative coset of size 8
        // ------------------------------------------------------------
        // Parameters for skipping sumcheck
        let dft = Dft::default();
        // Skip first 2 variables (X0, X1)
        let k = 2;
        let n = prover.num_variables();
        assert_eq!(n, 3);
        // j = 1 (remaining variables X2)
        let j = n - k;
        assert_eq!(j, 1);
        let n_evals_func = 1 << (k + 1); // 8

        // ------------------------------------------------------------
        // Compute the polynomial using the function under test
        // ------------------------------------------------------------
        let (poly, _, _) = prover.compute_skipping_sumcheck_polynomial(&dft, k);
        assert_eq!(poly.evaluations().len(), n_evals_func);

        // Manually compute f at all 8 binary points (0,1)^3
        let f_000 = f_base(F::ZERO, F::ZERO, F::ZERO);
        let f_001 = f_base(F::ZERO, F::ZERO, F::ONE);
        let f_010 = f_base(F::ZERO, F::ONE, F::ZERO);
        let f_011 = f_base(F::ZERO, F::ONE, F::ONE);
        let f_100 = f_base(F::ONE, F::ZERO, F::ZERO);
        let f_101 = f_base(F::ONE, F::ZERO, F::ONE);
        let f_110 = f_base(F::ONE, F::ONE, F::ZERO);
        let f_111 = f_base(F::ONE, F::ONE, F::ONE);

        // Compute the evaluations of the **equality constraint polynomial**.
        //
        // The equality constraint enforces X = (1,0,1).
        // The equality polynomial eq_{(1,0,1)}(X0,X1,X2) evaluates to:
        //   - 1 when (X₀,X₁,X₂) == (1,0,1)
        //   - 0 elsewhere
        //
        // Thus:
        //   - Only point (1,0,1) gets a 1
        //   - All other 7 points are 0
        let w_000 = EF4::ONE; // eq(0,0,0) = 1 (this is the constraint point)
        let w_001 = EF4::ZERO; // eq(0,0,1) = 0
        let w_010 = EF4::ZERO; // eq(0,1,0) = 0
        let w_011 = EF4::ZERO; // eq(0,1,1) = 0
        let w_100 = EF4::ZERO; // eq(1,0,0) = 0
        let w_101 = EF4::ONE; // eq(1,0,1) = 1 (this is the constraint point)
        let w_110 = EF4::ZERO; // eq(1,1,0) = 0
        let w_111 = EF4::ZERO; // eq(1,1,1) = 0

        assert_eq!(
            w_000 * f_000
                + w_001 * f_001
                + w_010 * f_010
                + w_011 * f_011
                + w_100 * f_100
                + w_101 * f_101
                + w_110 * f_110
                + w_111 * f_111,
            expected_sum
        );

        // Construct a matrix representing f(X0, X1, X2) values.
        // Each row corresponds to a fixed (X1, X2) pair,
        // and each column to X0 = 0 and X0 = 1.
        // The matrix is stored in row-major order, but transposed.
        // This layout aligns with batch DFTs over the "folded" variable X0.
        let f_mat_transpose = RowMajorMatrix::new(
            vec![
                f_000, f_100, // X0 = 0, 1 | (X1, X2) = (0, 0)
                f_001, f_101, // X0 = 0, 1 | (X1, X2) = (0, 1)
                f_010, f_110, // X0 = 0, 1 | (X1, X2) = (1, 0)
                f_011, f_111, // X0 = 0, 1 | (X1, X2) = (1, 1)
            ],
            2, // num columns = 2 (X0=0, X0=1)
        );

        // Do the same for the equality weights w(X0, X1, X2)
        let weights_mat_transpose = RowMajorMatrix::new(
            vec![
                w_000, w_100, // X0 = 0, 1 | (X1, X2) = (0, 0)
                w_001, w_101, // X0 = 0, 1 | (X1, X2) = (0, 1)
                w_010, w_110, // X0 = 0, 1 | (X1, X2) = (1, 0)
                w_011, w_111, // X0 = 0, 1 | (X1, X2) = (1, 1)
            ],
            2,
        );

        // We recover the coefficients of f by doing the inverse DFT of each column.
        //
        // We do this to be able to calculate manually the LDE.
        let f_coeffs_on_coset = dft.idft_batch(f_mat_transpose);
        let cf00 = f_coeffs_on_coset.get(0, 0).unwrap();
        let cf01 = f_coeffs_on_coset.get(1, 0).unwrap();
        let cf02 = f_coeffs_on_coset.get(2, 0).unwrap();
        let cf03 = f_coeffs_on_coset.get(3, 0).unwrap();

        let cf10 = f_coeffs_on_coset.get(0, 1).unwrap();
        let cf11 = f_coeffs_on_coset.get(1, 1).unwrap();
        let cf12 = f_coeffs_on_coset.get(2, 1).unwrap();
        let cf13 = f_coeffs_on_coset.get(3, 1).unwrap();

        // We recover the coefficients of w by doing the inverse DFT of each column.
        //
        // We do this to be able to calculate manually the LDE.
        let w_coeffs_on_coset = dft.idft_batch(weights_mat_transpose);
        let cw00 = w_coeffs_on_coset.get(0, 0).unwrap();
        let cw01 = w_coeffs_on_coset.get(1, 0).unwrap();
        let cw02 = w_coeffs_on_coset.get(2, 0).unwrap();
        let cw03 = w_coeffs_on_coset.get(3, 0).unwrap();

        let cw10 = w_coeffs_on_coset.get(0, 1).unwrap();
        let cw11 = w_coeffs_on_coset.get(1, 1).unwrap();
        let cw12 = w_coeffs_on_coset.get(2, 1).unwrap();
        let cw13 = w_coeffs_on_coset.get(3, 1).unwrap();

        // Evaluate on:
        //   [ shift·ω^0, shift·ω^1, ..., shift·ω^7 ]
        let omega8 = F::two_adic_generator(3);

        let f00 = cf00 * omega8.exp_u64(0 * 0)
            + cf01 * omega8.exp_u64(0 * 1)
            + cf02 * omega8.exp_u64(0 * 2)
            + cf03 * omega8.exp_u64(0 * 3);
        let f01 = cf00 * omega8.exp_u64(1 * 0)
            + cf01 * omega8.exp_u64(1 * 1)
            + cf02 * omega8.exp_u64(1 * 2)
            + cf03 * omega8.exp_u64(1 * 3);
        let f02 = cf00 * omega8.exp_u64(2 * 0)
            + cf01 * omega8.exp_u64(2 * 1)
            + cf02 * omega8.exp_u64(2 * 2)
            + cf03 * omega8.exp_u64(2 * 3);
        let f03 = cf00 * omega8.exp_u64(3 * 0)
            + cf01 * omega8.exp_u64(3 * 1)
            + cf02 * omega8.exp_u64(3 * 2)
            + cf03 * omega8.exp_u64(3 * 3);
        let f04 = cf00 * omega8.exp_u64(4 * 0)
            + cf01 * omega8.exp_u64(4 * 1)
            + cf02 * omega8.exp_u64(4 * 2)
            + cf03 * omega8.exp_u64(4 * 3);
        let f05 = cf00 * omega8.exp_u64(5 * 0)
            + cf01 * omega8.exp_u64(5 * 1)
            + cf02 * omega8.exp_u64(5 * 2)
            + cf03 * omega8.exp_u64(5 * 3);
        let f06 = cf00 * omega8.exp_u64(6 * 0)
            + cf01 * omega8.exp_u64(6 * 1)
            + cf02 * omega8.exp_u64(6 * 2)
            + cf03 * omega8.exp_u64(6 * 3);
        let f07 = cf00 * omega8.exp_u64(7 * 0)
            + cf01 * omega8.exp_u64(7 * 1)
            + cf02 * omega8.exp_u64(7 * 2)
            + cf03 * omega8.exp_u64(7 * 3);

        let f10 = cf10 * omega8.exp_u64(0 * 0)
            + cf11 * omega8.exp_u64(0 * 1)
            + cf12 * omega8.exp_u64(0 * 2)
            + cf13 * omega8.exp_u64(0 * 3);
        let f11 = cf10 * omega8.exp_u64(1 * 0)
            + cf11 * omega8.exp_u64(1 * 1)
            + cf12 * omega8.exp_u64(1 * 2)
            + cf13 * omega8.exp_u64(1 * 3);
        let f12 = cf10 * omega8.exp_u64(2 * 0)
            + cf11 * omega8.exp_u64(2 * 1)
            + cf12 * omega8.exp_u64(2 * 2)
            + cf13 * omega8.exp_u64(2 * 3);
        let f13 = cf10 * omega8.exp_u64(3 * 0)
            + cf11 * omega8.exp_u64(3 * 1)
            + cf12 * omega8.exp_u64(3 * 2)
            + cf13 * omega8.exp_u64(3 * 3);
        let f14 = cf10 * omega8.exp_u64(4 * 0)
            + cf11 * omega8.exp_u64(4 * 1)
            + cf12 * omega8.exp_u64(4 * 2)
            + cf13 * omega8.exp_u64(4 * 3);
        let f15 = cf10 * omega8.exp_u64(5 * 0)
            + cf11 * omega8.exp_u64(5 * 1)
            + cf12 * omega8.exp_u64(5 * 2)
            + cf13 * omega8.exp_u64(5 * 3);
        let f16 = cf10 * omega8.exp_u64(6 * 0)
            + cf11 * omega8.exp_u64(6 * 1)
            + cf12 * omega8.exp_u64(6 * 2)
            + cf13 * omega8.exp_u64(6 * 3);
        let f17 = cf10 * omega8.exp_u64(7 * 0)
            + cf11 * omega8.exp_u64(7 * 1)
            + cf12 * omega8.exp_u64(7 * 2)
            + cf13 * omega8.exp_u64(7 * 3);

        // w
        let w00 = cw00 * omega8.exp_u64(0 * 0)
            + cw01 * omega8.exp_u64(0 * 1)
            + cw02 * omega8.exp_u64(0 * 2)
            + cw03 * omega8.exp_u64(0 * 3);
        let w01 = cw00 * omega8.exp_u64(1 * 0)
            + cw01 * omega8.exp_u64(1 * 1)
            + cw02 * omega8.exp_u64(1 * 2)
            + cw03 * omega8.exp_u64(1 * 3);
        let w02 = cw00 * omega8.exp_u64(2 * 0)
            + cw01 * omega8.exp_u64(2 * 1)
            + cw02 * omega8.exp_u64(2 * 2)
            + cw03 * omega8.exp_u64(2 * 3);
        let w03 = cw00 * omega8.exp_u64(3 * 0)
            + cw01 * omega8.exp_u64(3 * 1)
            + cw02 * omega8.exp_u64(3 * 2)
            + cw03 * omega8.exp_u64(3 * 3);
        let w04 = cw00 * omega8.exp_u64(4 * 0)
            + cw01 * omega8.exp_u64(4 * 1)
            + cw02 * omega8.exp_u64(4 * 2)
            + cw03 * omega8.exp_u64(4 * 3);
        let w05 = cw00 * omega8.exp_u64(5 * 0)
            + cw01 * omega8.exp_u64(5 * 1)
            + cw02 * omega8.exp_u64(5 * 2)
            + cw03 * omega8.exp_u64(5 * 3);
        let w06 = cw00 * omega8.exp_u64(6 * 0)
            + cw01 * omega8.exp_u64(6 * 1)
            + cw02 * omega8.exp_u64(6 * 2)
            + cw03 * omega8.exp_u64(6 * 3);
        let w07 = cw00 * omega8.exp_u64(7 * 0)
            + cw01 * omega8.exp_u64(7 * 1)
            + cw02 * omega8.exp_u64(7 * 2)
            + cw03 * omega8.exp_u64(7 * 3);

        let w10 = cw10 * omega8.exp_u64(0 * 0)
            + cw11 * omega8.exp_u64(0 * 1)
            + cw12 * omega8.exp_u64(0 * 2)
            + cw13 * omega8.exp_u64(0 * 3);
        let w11 = cw10 * omega8.exp_u64(1 * 0)
            + cw11 * omega8.exp_u64(1 * 1)
            + cw12 * omega8.exp_u64(1 * 2)
            + cw13 * omega8.exp_u64(1 * 3);
        let w12 = cw10 * omega8.exp_u64(2 * 0)
            + cw11 * omega8.exp_u64(2 * 1)
            + cw12 * omega8.exp_u64(2 * 2)
            + cw13 * omega8.exp_u64(2 * 3);
        let w13 = cw10 * omega8.exp_u64(3 * 0)
            + cw11 * omega8.exp_u64(3 * 1)
            + cw12 * omega8.exp_u64(3 * 2)
            + cw13 * omega8.exp_u64(3 * 3);
        let w14 = cw10 * omega8.exp_u64(4 * 0)
            + cw11 * omega8.exp_u64(4 * 1)
            + cw12 * omega8.exp_u64(4 * 2)
            + cw13 * omega8.exp_u64(4 * 3);
        let w15 = cw10 * omega8.exp_u64(5 * 0)
            + cw11 * omega8.exp_u64(5 * 1)
            + cw12 * omega8.exp_u64(5 * 2)
            + cw13 * omega8.exp_u64(5 * 3);
        let w16 = cw10 * omega8.exp_u64(6 * 0)
            + cw11 * omega8.exp_u64(6 * 1)
            + cw12 * omega8.exp_u64(6 * 2)
            + cw13 * omega8.exp_u64(6 * 3);
        let w17 = cw10 * omega8.exp_u64(7 * 0)
            + cw11 * omega8.exp_u64(7 * 1)
            + cw12 * omega8.exp_u64(7 * 2)
            + cw13 * omega8.exp_u64(7 * 3);

        let r0 = w00 * f00 + w10 * f10;
        let r1 = w01 * f01 + w11 * f11;
        let r2 = w02 * f02 + w12 * f12;
        let r3 = w03 * f03 + w13 * f13;
        let r4 = w04 * f04 + w14 * f14;
        let r5 = w05 * f05 + w15 * f15;
        let r6 = w06 * f06 + w16 * f16;
        let r7 = w07 * f07 + w17 * f17;

        // Check the polynomial evaluations are correct
        assert_eq!(poly.evaluations(), vec![r0, r1, r2, r3, r4, r5, r6, r7]);

        // Check the sum of the polynomial evaluations is correct
        assert_eq!(
            poly.evaluations().iter().step_by(2).copied().sum::<EF4>(),
            expected_sum
        );
    }
}
