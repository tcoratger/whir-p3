use p3_dft::{NaiveDft, TwoAdicSubgroupDft};
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
#[cfg(feature = "parallel")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};

use super::sumcheck_polynomial::SumcheckPolynomial;
use crate::poly::evals::EvaluationsList;

/// Computes the sumcheck polynomial using the **univariate skip** optimization,
/// which folds the first `k` variables in one step via low-degree extension (LDE).
///
/// The goal is to reduce a multilinear polynomial $f(x_1, \dots, x_n)$
/// and a weight polynomial $w(x_1, \dots, x_n)$ defined over the Boolean hypercube
/// $\{0,1\}^n$ into a univariate polynomial $h(X)$ via partial evaluation and DFT-based extension.
///
/// This function interprets the original evaluations over $\{0,1\}^n$ as a matrix of shape
/// $(2^k \times 2^{n-k})$, where:
/// - Each row corresponds to a distinct assignment to the first $k$ variables (which we skip/fold),
/// - Each column corresponds to a Boolean assignment to the remaining $n - k$ variables.
///
/// It then applies LDE to each row over a multiplicative coset $D$ of size $2^{k+1}$ and computes:
///
/// \begin{equation}
/// h(X) = \sum_{b \in \{0,1\}^{n-k}} f(X, b) \cdot w(X, b)
/// \end{equation}
///
/// where:
/// - $X$ ranges over $D$, a multiplicative coset used to evaluate the first $k$ variables,
/// - $b$ ranges over $\{0,1\}^{n-k}$, the Boolean values of the remaining variables.
///
/// # Arguments
/// - `k`: Number of initial variables to skip and fold into a univariate extension.
/// - `evals`: Evaluations of the multilinear polynomial $f$ over $\{0,1\}^n$, in the base field `F`.
/// - `weights`: Evaluations of the weight polynomial $w$ over $\{0,1\}^n$, in the extension field `EF`.
///
/// # Returns
/// A tuple containing:
/// - `SumcheckPolynomial<EF>`: The resulting univariate polynomial $h(X)$ evaluated over coset $D$.
/// - `RowMajorMatrix<F>`: The original evaluations of $f$, reshaped to $(2^k \times 2^{n-k})$.
/// - `RowMajorMatrix<EF>`: The original evaluations of $w$, reshaped to $(2^k \times 2^{n-k})$.
///
/// # Panics
/// Panics if `k > evals.num_variables()`.
///
/// # Notes
/// - This method assumes that `f` is represented using base field values (`F`)
///   and that `w` is represented using extension field values (`EF`).
/// - The LDE step extends each row from $2^k$ to $2^{k+1}$ using a coset FFT,
///   enabling efficient computation of the univariate sumcheck polynomial.
#[must_use]
pub(crate) fn compute_skipping_sumcheck_polynomial<F: TwoAdicField, EF: ExtensionField<F>>(
    k: usize,
    evals: &EvaluationsList<F>,
    weights: &EvaluationsList<EF>,
) -> (
    SumcheckPolynomial<EF>,
    RowMajorMatrix<F>,
    RowMajorMatrix<EF>,
) {
    // Ensure we have enough variables to skip.
    // We can only skip if the number of variables n ≥ k.
    assert!(
        evals.num_variables() >= k,
        "Need at least k variables to apply univariate skip on k variables"
    );

    // Main logic block that computes the univariate sumcheck polynomial h(X)
    // and returns intermediate matrices of shape (2^k × 2^{n-k}).
    let (out_vec, f, w) = {
        // Number of variables for the multilinear polynomial f(X)
        let n = evals.num_variables();
        // Number of remaining variables after skipping k
        let num_remaining_vars = n - k;
        // Number of columns = 2^{n-k}
        let width = 1 << num_remaining_vars;

        // Reshape the evaluation vector of f (over {0,1}^n) into a matrix:
        // - Each row corresponds to one of the 2^k assignments to the skipped variables.
        // - Each column corresponds to a Boolean assignment to the remaining n−k variables.
        //
        // This aligns with the goal of computing:
        //   h(X) = ∑_{b ∈ {0,1}^{n−k}} f(X, b) · w(X, b)
        let f_mat = RowMajorMatrix::new(evals.evals().to_vec(), width);

        // Do the same for the weight polynomial w(X): shape = (2^k × 2^{n-k})
        let weights_mat = RowMajorMatrix::new(weights.evals().to_vec(), width);

        // Apply a low-degree extension (LDE) to each row of f_mat and weights_mat.
        // The LDE maps each row of length 2^k to 2^{k+1} evaluations over a multiplicative coset.
        //
        // This gives us access to evaluations of f(X, b) and w(X, b)
        // for non-Boolean values of X ∈ D (coset of size 2^{k+1}).
        let dft = NaiveDft;

        // Apply base-field LDE to each row of f_mat: F^2^k → F^2^{k+1}
        let f_on_coset = dft.lde_batch(f_mat.clone(), 1).to_row_major_matrix();

        // Apply extension-field LDE to each row of weights_mat: EF^2^k → EF^2^{k+1}
        let weights_on_coset = dft
            .lde_algebra_batch(weights_mat.clone(), 1)
            .to_row_major_matrix();

        // For each column (i.e., each value X in the coset domain),
        // compute: sum over all b ∈ {0,1}^{n−k} of f(X, b) · w(X, b)
        //
        // This is done by pointwise multiplying the f and w values in each row,
        // then summing across the row. Each output corresponds to one X in the coset.
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

        // Return:
        // - result: evaluations of the univariate sumcheck polynomial h(X)
        // - f_mat: original (2^k × 2^{n−k}) matrix of f(X) before LDE
        // - weights_mat: original (2^k × 2^{n−k}) matrix of w(X) before LDE
        (result, f_mat, weights_mat)
    };

    // Return h(X) as a SumcheckPolynomial, along with the raw pre-LDE matrices
    (SumcheckPolynomial::new(out_vec, 1), f, w)
}

#[cfg(test)]
#[allow(clippy::erasing_op, clippy::identity_op)]
mod tests {
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_dft::NaiveDft;
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};
    use rand::{SeedableRng, rngs::SmallRng};

    use super::*;
    use crate::{
        fiat_shamir::{domain_separator::DomainSeparator, prover::ProverState},
        poly::{coeffs::CoefficientList, multilinear::MultilinearPoint},
        whir::statement::{Statement, weights::Weights},
    };

    type F = BabyBear;
    type EF4 = BinomialExtensionField<BabyBear, 4>;
    type Dft = NaiveDft;
    type Perm = Poseidon2BabyBear<16>;
    type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

    /// Creates a fresh domain separator and challenger with fixed RNG seed.
    fn domainsep_and_challenger() -> (DomainSeparator<EF4, F>, MyChallenger) {
        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);
        let challenger = MyChallenger::new(perm);
        (DomainSeparator::new(vec![]), challenger)
    }

    fn prover() -> ProverState<F, EF4, MyChallenger> {
        let (domsep, challenger) = domainsep_and_challenger();
        domsep.to_prover_state(challenger)
    }

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
        let (weights, _sum) = statement.combine::<F>(EF4::ONE);

        // ----------------------------------------------------------------
        // We perform the univariate skip with k = 1:
        // This skips 1 variable (X0), leaving a univariate polynomial in X1.
        //
        // Instead of evaluating over {0,1} (Boolean domain),
        // we extend the evaluation domain to a **multiplicative coset of size 4**
        // using `coset_lde_batch` (low-degree extension via DFT).
        // ----------------------------------------------------------------
        let (poly, _, _) =
            compute_skipping_sumcheck_polynomial::<F, EF4>(1, &coeffs.to_evaluations(), &weights);

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
        let (weights, _sum) = statement.combine::<F>(EF4::ONE);

        // ----------------------------------------------------------------
        // Apply the univariate skip optimization with k = 2:
        //
        //   - This skips 2 variables: X0 and X1
        //   - The result is a univariate polynomial
        //
        // To evaluate this polynomial, we perform a low-degree extension
        // using DFT on a multiplicative coset of size 2^{k+1} = 8.
        // ----------------------------------------------------------------
        let (poly, _, _) =
            compute_skipping_sumcheck_polynomial::<F, EF4>(2, &coeffs.to_evaluations(), &weights);

        // ----------------------------------------------------------------
        // Finally, the sum over {0,1} values of X2 must also be zero
        // because the polynomial is identically zero on the full domain.
        // ----------------------------------------------------------------
        assert_eq!(poly.sum_over_boolean_hypercube(), EF4::ZERO);
    }

    #[test]
    #[should_panic]
    fn test_skipping_sumcheck_polynomial_panics_on_invalid_k() {
        // Polynomial with only 1 variable, can't skip 2 variables
        let c0 = F::from_u64(1);
        let c1 = F::from_u64(2);
        let coeffs = CoefficientList::new(vec![c0, c1]);

        let statement = Statement::<EF4>::new(1);
        let (weights, _sum) = statement.combine::<F>(EF4::ONE);

        // This should panic because:
        // - the polynomial has only 1 variable
        // - we try to skip 2 variables
        let _ =
            compute_skipping_sumcheck_polynomial::<F, EF4>(2, &coeffs.to_evaluations(), &weights);
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

        let (weights, expected_sum) = statement.combine::<F>(EF4::ONE);

        // Get the f evaluations
        let evals_f = coeffs.to_evaluations();

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
        let n = evals_f.num_variables();
        assert_eq!(n, 3);
        // j = 1 (remaining variables X2)
        let j = n - k;
        assert_eq!(j, 1);
        let n_evals_func = 1 << (k + 1); // 8

        // ------------------------------------------------------------
        // Compute the polynomial using the function under test
        // ------------------------------------------------------------
        let (poly, f_mat, w_mat) = compute_skipping_sumcheck_polynomial(k, &evals_f, &weights);
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
        // - Each row corresponds to a fixed (X0, X1) pair (variables we fold over),
        // - Each column corresponds to a fixed X2 value (remaining variable).
        let f_mat_expected = RowMajorMatrix::new(
            vec![
                f_000, f_001, // (0, 0, 0) | (0, 0, 1)
                f_010, f_011, // (0, 1, 0) | (0, 1, 1)
                f_100, f_101, // (1, 0, 0) | (1, 0, 1)
                f_110, f_111, // (1, 1, 0) | (1, 1, 1)
            ],
            2, // num columns = 2 -> We want to fold over X0 and X1
        );

        // Do the same for the equality weights w(X0, X1, X2)
        let weights_mat_expected = RowMajorMatrix::new(
            vec![
                w_000, w_001, // (0, 0, 0) | (0, 0, 1)
                w_010, w_011, // (0, 1, 0) | (0, 1, 1)
                w_100, w_101, // (1, 0, 0) | (1, 0, 1)
                w_110, w_111, // (1, 1, 0) | (1, 1, 1)
            ],
            2, // num columns = 2 -> We want to fold over X0 and X1
        );

        // Verify the f and w matrices
        assert_eq!(f_mat, f_mat_expected);
        assert_eq!(w_mat, weights_mat_expected);

        // We recover the coefficients of f by doing the inverse DFT of each column.
        //
        // We do this to be able to calculate manually the LDE.
        let f_coeffs_on_coset = dft.idft_batch(f_mat);
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
        let w_coeffs_on_coset = dft.idft_batch(w_mat);
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
