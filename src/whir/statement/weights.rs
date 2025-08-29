use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_interpolation::interpolate_subgroup;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::{eq::eval_eq, point::MultilinearPoint};
use tracing::instrument;

use crate::poly::evals::EvaluationsList;

/// Represents a weight function used in polynomial evaluations.
///
/// A `Weights<F>` instance allows evaluating or accumulating weighted contributions
/// to a multilinear polynomial stored in evaluation form. It supports two modes:
///
/// - Evaluation mode: Represents an equality constraint at a specific `MultilinearPoint<F>`.
/// - Linear mode: Represents a set of per-corner weights stored as `EvaluationsList<F>`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Weights<F> {
    /// Represents a weight function that enforces equality constraints at a specific point.
    Evaluation { point: MultilinearPoint<F> },
    /// Represents a weight function defined as a precomputed set of evaluations.
    Linear { weight: EvaluationsList<F> },
}

impl<F: Field> Weights<F> {
    /// Constructs a weight in evaluation mode, enforcing an equality constraint at `point`.
    ///
    /// Given a multilinear polynomial `p(X)`, this weight evaluates:
    ///
    /// \begin{equation}
    /// w(X) = eq_{z}(X)
    /// \end{equation}
    ///
    /// where `eq_z(X)` is the Lagrange interpolation polynomial enforcing `X = z`.
    #[must_use]
    pub const fn evaluation(point: MultilinearPoint<F>) -> Self {
        Self::Evaluation { point }
    }

    /// Constructs a weight in linear mode, applying a set of precomputed weights.
    ///
    /// This mode allows applying a function `w(X)` stored in `EvaluationsList<F>`:
    ///
    /// \begin{equation}
    /// w(X) = \sum_{i} w_i \cdot X_i
    /// \end{equation}
    ///
    /// where `w_i` are the predefined weight values for each corner of the hypercube.
    #[must_use]
    pub const fn linear(weight: EvaluationsList<F>) -> Self {
        Self::Linear { weight }
    }

    /// Returns the number of variables involved in the weight function.
    #[must_use]
    pub const fn num_variables(&self) -> usize {
        match self {
            Self::Evaluation { point } => point.num_variables(),
            Self::Linear { weight } => weight.num_variables(),
        }
    }

    /// Construct weights for a univariate evaluation
    pub fn univariate(point: F, size: usize) -> Self {
        Self::Evaluation {
            point: MultilinearPoint::expand_from_univariate(point, size),
        }
    }

    /// Computes the weighted sum of a polynomial `p(X)` under the current weight function.
    ///
    /// - In linear mode, computes the inner product between the polynomial values and weights:
    ///
    /// \begin{equation}
    /// \sum_{i} p_i \cdot w_i
    /// \end{equation}
    ///
    /// - In evaluation mode, evaluates `p(X)` at the equality constraint point.
    ///
    /// **Precondition:**
    /// If `self` is in linear mode, `poly.num_variables()` must match `weight.num_variables()`.
    #[must_use]
    pub fn evaluate_evals<BF>(&self, poly: &EvaluationsList<BF>) -> F
    where
        BF: Field,
        F: ExtensionField<BF>,
    {
        match self {
            Self::Linear { weight } => {
                assert_eq!(poly.num_variables(), weight.num_variables());
                poly.as_slice()
                    .par_iter()
                    .zip(weight.as_slice().par_iter())
                    .map(|(p, w)| *w * *p)
                    .sum()
            }
            Self::Evaluation { point } => poly.evaluate(point),
        }
    }

    /// Accumulates the contribution of the weight function into `accumulator`, scaled by `factor`.
    ///
    /// - In evaluation mode, updates `accumulator` using an equality constraint.
    /// - In linear mode, scales the weight function by `factor` and accumulates it.
    ///
    /// Given a weight function `w(X)` and a factor `λ`, this updates `accumulator` as:
    ///
    /// ```math
    /// a(X) <- a(X) + \lambda \cdot w(X)
    /// ```
    ///
    /// where `a(X)` is the accumulator polynomial.
    ///
    /// **Precondition:**
    /// `accumulator.num_variables()` must match `self.num_variables()`.
    ///
    /// **Warning:**
    /// If INITIALIZED is `false`, the accumulator must be overwritten with the new values.
    #[instrument(skip_all)]
    pub fn accumulate<Base, const INITIALIZED: bool>(
        &self,
        accumulator: &mut EvaluationsList<F>,
        factor: F,
    ) where
        Base: Field,
        F: ExtensionField<Base>,
    {
        assert_eq!(accumulator.num_variables(), self.num_variables());
        match self {
            Self::Evaluation { point } => {
                eval_eq::<Base, F, INITIALIZED>(
                    point.as_slice(),
                    accumulator.as_mut_slice(),
                    factor,
                );
            }
            Self::Linear { weight } => {
                let weight_slice = weight.as_slice();
                accumulator
                    .as_mut_slice()
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(corner, acc)| {
                        if INITIALIZED {
                            *acc += factor * weight_slice[corner];
                        } else {
                            *acc = factor * weight_slice[corner];
                        }
                    });
            }
        }
    }

    /// Evaluates the weight function at a given folding point.
    ///
    /// - In evaluation mode, computes the equality polynomial at the provided point:
    ///
    /// \begin{equation}
    /// w(X) = eq_{\text{point}}(X)
    /// \end{equation}
    ///
    /// This enforces that the polynomial is evaluated exactly at a specific input.
    ///
    /// - In linear mode, interprets the stored weight function as a multilinear polynomial
    ///   and evaluates it at the provided point using multilinear extension.
    ///
    /// **Precondition:**
    /// The input point must have the same number of variables as the weight function.
    ///
    /// **Returns:**
    /// A field element representing the weight function applied to the given point.
    #[must_use]
    pub fn compute(&self, folding_randomness: &MultilinearPoint<F>) -> F {
        match self {
            Self::Evaluation { point } => point.eq_poly(folding_randomness),
            Self::Linear { weight } => weight.evaluate(folding_randomness),
        }
    }

    /// Evaluates the weight polynomial `W(X)` at a challenge point `r` using univariate skip semantics.
    ///
    /// This verifier-side operation is designed to mirror the prover's special evaluation method for
    /// the main polynomial `f(X)`. To ensure the final protocol check is valid, the verifier must
    /// use the exact same evaluation process for the weights.
    ///
    /// This method is computationally intensive ("heavy") because protocol consistency requires it to
    /// fully materialize the `2^n` evaluation table of the weight polynomial.
    ///
    /// TODO: simplify this but for now we just want this to work.
    ///
    /// # Arguments
    /// * `r_all`: The verifier's challenge object. This is a special structure containing
    ///   `(n - k_skip) + 1` field elements, not a full n-dimensional point.
    /// * `k_skip`: The number of variables that were folded in the first protocol round.
    ///
    /// # Returns
    /// The resulting evaluation `W(r)` as a single field element.
    #[must_use]
    pub fn compute_with_skip<EF>(&self, r_all: &MultilinearPoint<EF>, k_skip: usize) -> EF
    where
        F: TwoAdicField,
        EF: TwoAdicField + ExtensionField<F>,
    {
        // Determine the full domain size `n`
        //
        // The total number of variables `n` is inferred from the structure of the challenge object `r_all`
        // and the number of skipped variables `k_skip`.
        let n = r_all.num_variables() + k_skip - 1;

        // Materialize the 2^n evaluation table for the weight polynomial
        //
        // We build the complete table of the weight polynomial `W(X)` over the n-dimensional hypercube.
        let evals = match self {
            // Case 1: The weight is defined by a pre-computed evaluation table.
            Self::Linear { weight } => {
                // Ensure the provided table matches the full domain size.
                assert_eq!(
                    weight.num_variables(),
                    n,
                    "Linear weight must match domain size"
                );
                weight.clone()
            }
            // Case 2: The weight is defined by an equality constraint at a point `z`.
            Self::Evaluation { point } => {
                // The constraint point `z` must be defined over the full n-variable domain.
                let k = point.num_variables();
                assert!(
                    k <= n,
                    "Constraint point cannot have more variables than the domain"
                );

                // Construct the evaluation table for the polynomial eq_z(X).
                let mut evals = EvaluationsList::new(F::zero_vec(1 << n));
                eval_eq::<_, _, false>(point.as_slice(), evals.as_mut_slice(), F::ONE);
                evals
            }
        };

        // Reshape the evaluation table into a matrix for folding
        //
        // The flat list of 2^n evaluations is viewed as a `2^k_skip x 2^(n-k_skip)` matrix.
        // Rows correspond to the skipped variables (X0, ..., Xk-1).
        // Columns correspond to the remaining variables (Xk, ..., Xn-1).
        let num_remaining_vars = n - k_skip;
        let width = 1 << num_remaining_vars;
        let mat = RowMajorMatrix::new(evals.as_slice().to_vec(), width);

        // Deconstruct the challenge object `r_all`
        //
        // The last element is the challenge for the `k_skip` variables being folded.
        let r_skip = *r_all
            .last_variable()
            .expect("skip challenge must be present");
        // The first `n - k_skip` elements are the challenges for the remaining variables.
        let r_rest = MultilinearPoint::new(r_all.as_slice()[..num_remaining_vars].to_vec());

        // Perform the two-stage evaluation
        //
        // First, "fold" the skipped variables by interpolating each column at `r_skip`.
        // This produces a new, smaller polynomial over the remaining variables.
        let folded_row = interpolate_subgroup(&mat, r_skip);

        // Second, evaluate this new polynomial at the remaining challenges `r_rest`.
        EvaluationsList::new(folded_row).evaluate(&r_rest)
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};
    use proptest::prelude::*;

    use super::*;

    type F = BabyBear;
    type EF4 = BinomialExtensionField<F, 4>;

    #[test]
    fn test_weights_evaluation() {
        // Define a point in the multilinear space
        let point = MultilinearPoint::new(vec![F::ONE, F::ZERO]);
        let weight = Weights::evaluation(point);

        // The number of variables in the weight should match the number of variables in the point
        assert_eq!(weight.num_variables(), 2);
    }

    #[test]
    fn test_weights_linear() {
        // Define a list of evaluation values
        let evals = EvaluationsList::new(vec![F::ONE, F::TWO, F::from_u64(3), F::from_u64(3)]);
        let weight = Weights::linear(evals);

        // The number of variables in the weight should match the number of variables in evals
        assert_eq!(weight.num_variables(), 2);
    }

    #[test]
    fn test_weighted_sum_evaluation() {
        // Define polynomial evaluations at different points
        let e0 = F::from_u64(3);
        let e1 = F::from_u64(5);
        let evals = EvaluationsList::new(vec![e0, e1]);

        // Define an evaluation weight at a specific point
        let point = MultilinearPoint::new(vec![F::ONE]);
        let weight = Weights::evaluation(point);

        // Expected result: polynomial evaluation at the given point
        let expected = e1;

        assert_eq!(weight.evaluate_evals(&evals), expected);
    }

    #[test]
    fn test_weighted_sum_linear() {
        // Define polynomial evaluations
        let e0 = F::ONE;
        let e1 = F::TWO;
        let evals = EvaluationsList::new(vec![e0, e1]);

        // Define linear weights
        let w0 = F::TWO;
        let w1 = F::from_u64(3);
        let weight_list = EvaluationsList::new(vec![w0, w1]);
        let weight = Weights::linear(weight_list);

        // Compute expected result manually
        //
        // \begin{equation}
        // \sum_{i} e_i \cdot w_i = e_0 \cdot w_0 + e_1 \cdot w_1
        // \end{equation}
        let expected = e0 * w0 + e1 * w1;

        assert_eq!(weight.evaluate_evals(&evals), expected);
    }

    #[test]
    fn test_accumulate_linear() {
        // Initialize an empty accumulator
        let mut accumulator = EvaluationsList::new(vec![F::ZERO, F::ZERO]);

        // Define weights
        let w0 = F::from_u64(2);
        let w1 = F::from_u64(3);
        let weight_list = EvaluationsList::new(vec![w0, w1]);
        let weight = Weights::linear(weight_list);

        // Define a multiplication factor
        let factor = F::from_u64(4);

        // Accumulate weighted values
        weight.accumulate::<_, true>(&mut accumulator, factor);

        // Expected result:
        //
        // \begin{equation}
        // acc_i = factor \cdot w_i
        // \end{equation}
        let expected = vec![
            w0 * factor, // 2 * 4 = 8
            w1 * factor, // 3 * 4 = 12
        ];

        assert_eq!(accumulator.as_slice(), &expected);
    }

    #[test]
    fn test_accumulate_evaluation() {
        // Initialize an empty accumulator
        let mut accumulator = EvaluationsList::new(vec![F::ZERO, F::ZERO]);

        // Define an evaluation point
        let point = MultilinearPoint::new(vec![F::ONE]);
        let weight = Weights::evaluation(point.clone());

        // Define a multiplication factor
        let factor = F::from_u64(5);

        // Accumulate weighted values
        weight.accumulate::<_, true>(&mut accumulator, factor);

        // Compute expected result manually
        let mut expected = vec![F::ZERO, F::ZERO];
        eval_eq::<_, _, true>(point.as_slice(), &mut expected, factor);

        assert_eq!(accumulator.as_slice(), &expected);
    }

    #[test]
    fn test_univariate_weights_one_variable() {
        // y = 3, n = 1 → [3]
        let y = F::from_u64(3);
        let weight = Weights::univariate(y, 1);

        // Expect point to be [3]
        let expected = MultilinearPoint::new(vec![y]);
        assert_eq!(weight, Weights::evaluation(expected));
    }

    #[test]
    fn test_univariate_weights_two_variables() {
        // y = 4, n = 2 → [y^2, y] = [16, 4]
        let y = F::from_u64(4);
        let weight = Weights::univariate(y, 2);

        let expected = MultilinearPoint::new(vec![y.square(), y]);
        assert_eq!(weight, Weights::evaluation(expected));
    }

    #[test]
    fn test_univariate_weights_four_variables() {
        // y = 3, n = 4 → [3^8, 3^4, 3^2, 3]
        let y = F::from_u64(3);
        let weight = Weights::univariate(y, 4);

        let expected = MultilinearPoint::new(vec![y.exp_u64(8), y.exp_u64(4), y.square(), y]);

        assert_eq!(weight, Weights::evaluation(expected));
    }

    #[test]
    fn test_univariate_weights_zero_variables() {
        let y = F::from_u64(10);
        let weight = Weights::univariate(y, 0);

        // Expect empty point
        let expected = MultilinearPoint::new(vec![]);
        assert_eq!(weight, Weights::evaluation(expected));
    }

    #[test]
    fn test_compute_with_skip_linear() {
        // SETUP:
        // - n = 3 total variables: (X0, X1, X2).
        // - k_skip = 2 variables to skip: X0, X1.
        // - n - k_skip = 1 variable remaining: X2.
        let n = 3;
        let k_skip = 2;

        // A pre-defined weight polynomial W(X) over the 3-variable hypercube.
        // Values are [w(000), w(001), w(010), w(011), w(100), w(101), w(110), w(111)]
        let w_000 = F::from_u32(0);
        let w_001 = F::from_u32(1);
        let w_010 = F::from_u32(2);
        let w_011 = F::from_u32(3);
        let w_100 = F::from_u32(4);
        let w_101 = F::from_u32(5);
        let w_110 = F::from_u32(6);
        let w_111 = F::from_u32(7);
        let weight_evals = vec![w_000, w_001, w_010, w_011, w_100, w_101, w_110, w_111];
        let weights = Weights::Linear {
            weight: EvaluationsList::new(weight_evals),
        };

        // The verifier's full challenge point `r_all`. The prover reverses challenges,
        // so the layout is [r_rest..., r_skip].
        // - r_rest corresponds to the remaining variables (X2).
        // - r_skip is the single challenge for the combined (X0, X1) domain.
        let r_rest = MultilinearPoint::new(vec![EF4::from_u32(5)]);
        let r_skip = EF4::from_u32(7);
        let r_all = MultilinearPoint::new([r_rest.as_slice(), &[r_skip]].concat());

        // ACTION: Compute W(r) using the function under test.
        let result = weights.compute_with_skip(&r_all, k_skip);

        // MANUAL VERIFICATION:
        // 1. Reshape W(X) into a 2^k x 2^(n-k) = 4x2 matrix.
        //    Rows are indexed by (X0, X1), columns by (X2).
        //    mat[row, col] corresponds to W(x0, x1, x2) where row=(x0,x1) and col=x2.
        //    [[w(0,0,0), w(0,0,1)],  <- X0=0, X1=0
        //     [w(0,1,0), w(0,1,1)],  <- X0=0, X1=1
        //     [w(1,0,0), w(1,0,1)],  <- X0=1, X1=0
        //     [w(1,1,0), w(1,1,1)]]  <- X0=1, X1=1
        let num_remaining = n - k_skip;
        let mat = RowMajorMatrix::new(
            vec![
                EF4::from(w_000),
                EF4::from(w_001), // w(0,0,0), w(0,0,1)
                EF4::from(w_010),
                EF4::from(w_011), // w(0,1,0), w(0,1,1)
                EF4::from(w_100),
                EF4::from(w_101), // w(1,0,0), w(1,0,1)
                EF4::from(w_110),
                EF4::from(w_111), // w(1,1,0), w(1,1,1)
            ],
            1 << num_remaining,
        );

        // 2. Interpolate the polynomial represented by each *column* at `r_skip`.
        //    This evaluates the part of the polynomial depending on (X0,X1) at the
        //    challenge `r_skip`, effectively "folding" the first 4 rows into 1.
        //    The result is a single row vector of 2 elements.
        let folded_row = interpolate_subgroup(&mat, r_skip);

        // 3. The `folded_row` represents a new 1-variable polynomial, W'(X2),
        //    evaluated at the points X2=0 and X2=1.
        let final_poly = EvaluationsList::new(folded_row);

        // 4. Evaluate this final polynomial at the remaining challenge, r_rest = [5].
        let expected = final_poly.evaluate(&r_rest);

        assert_eq!(
            result, expected,
            "Manual skip evaluation should match function"
        );
    }

    #[test]
    fn test_compute_with_skip_evaluation() {
        // SETUP:
        // - n = 3 total variables: (X0, X1, X2).
        // - The constraint point `z` is defined over the full n=3 variables.
        // - k_skip = 2 variables to skip: X0, X1.
        let n = 3;
        let k_skip = 2;

        // The weight polynomial is W(X) = eq_z(X0, X1, X2), where z=(2,3,4).
        // The constraint point MUST be full-dimensional.
        let point = MultilinearPoint::new(vec![F::from_u32(2), F::from_u32(3), F::from_u32(4)]);
        let weights = Weights::Evaluation {
            point: point.clone(),
        };

        // The verifier's full challenge object `r_all`.
        // It has (n - k_skip) + 1 = (3 - 2) + 1 = 2 elements.
        // - r_rest for remaining variable (X2).
        // - r_skip for the combined (X0, X1) domain.
        let r_rest = MultilinearPoint::new(vec![EF4::from_u32(5)]);
        let r_skip = EF4::from_u32(7);
        let r_all = MultilinearPoint::new([r_rest.as_slice(), &[r_skip]].concat());

        // ACTION: Compute W(r) using the function under test.
        let result = weights.compute_with_skip(&r_all, k_skip);

        // MANUAL VERIFICATION:
        // 1. Manually construct the full 8-element table for W(X) = eq_z(X0, X1, X2).
        let z0 = EF4::from(point.as_slice()[0]);
        let z1 = EF4::from(point.as_slice()[1]);
        let z2 = EF4::from(point.as_slice()[2]);
        let mut full_evals_vec = Vec::with_capacity(1 << n);
        for i in 0..(1 << n) {
            // Index `i` corresponds to the point (x0, x1, x2)
            let x0 = EF4::from_u32((i >> 2) & 1);
            let x1 = EF4::from_u32((i >> 1) & 1);
            let x2 = EF4::from_u32(i & 1);
            let term0 = z0 * x0 + (EF4::ONE - z0) * (EF4::ONE - x0);
            let term1 = z1 * x1 + (EF4::ONE - z1) * (EF4::ONE - x1);
            let term2 = z2 * x2 + (EF4::ONE - z2) * (EF4::ONE - x2);
            full_evals_vec.push(term0 * term1 * term2);
        }

        // Reshape into a 4x2 matrix (Rows: (X0,X1), Cols: X2).
        let num_remaining = n - k_skip;
        let mat = RowMajorMatrix::new(full_evals_vec, 1 << num_remaining);

        // Interpolate each column at r_skip to fold the (X0, X1) variables.
        let folded_row = interpolate_subgroup(&mat, r_skip);

        // The `folded_row` is a new 1-variable polynomial, W'(X2).
        let final_poly = EvaluationsList::new(folded_row);

        // Evaluate this final polynomial at the remaining challenge, r_rest.
        let expected = final_poly.evaluate(&r_rest);

        assert_eq!(
            result, expected,
            "Manual skip evaluation for Evaluation weight should match"
        );
    }

    #[test]
    fn test_compute_with_skip_evaluation_all_vars() {
        // SETUP:
        // - n = 5 total variables: (X0, X1, X2, X3, X4).
        // - The constraint point `z` is defined over the full n=5 variables.
        // - k_skip = 5 variables to skip (all of them).
        // - This leaves 0 remaining variables.
        let n = 5;
        let k_skip = 5;

        // The weight polynomial is W(X) = eq_z(X0..X4), where z is a random 5-element point.
        let point = MultilinearPoint::new(vec![
            F::from_u32(2),
            F::from_u32(3),
            F::from_u32(5),
            F::from_u32(7),
            F::from_u32(11),
        ]);
        let weights = Weights::Evaluation {
            point: point.clone(),
        };

        // The verifier's challenge object `r_all`.
        // It has (n - k_skip) + 1 = (5 - 5) + 1 = 1 element.
        // - r_rest is an empty vector for the 0 remaining variables.
        // - r_skip is the single challenge for the combined (X0..X4) domain.
        let r_rest = MultilinearPoint::new(vec![]);
        let r_skip = EF4::from_u32(13);
        let r_all = MultilinearPoint::new(vec![r_skip]);

        // Compute W(r) using the function under test.
        let result = weights.compute_with_skip(&r_all, k_skip);

        // MANUAL VERIFICATION:
        // Manually construct the full 2^5=32 element table for W(X) = eq_z(X0..X4).
        let mut full_evals_vec = Vec::with_capacity(1 << n);
        for i in 0..(1 << n) {
            // The evaluation of eq_z(x) is the product of n individual terms.
            let eq_val: EF4 = (0..n)
                .map(|j| {
                    // Get the j-th coordinate of the constraint point z.
                    let z_j = EF4::from(point.as_slice()[j]);
                    // Get the j-th coordinate of the hypercube point x by checking the j-th bit of i.
                    let x_j = EF4::from_u32((i >> (n - 1 - j)) & 1);
                    // Calculate the j-th term of the product.
                    z_j * x_j + (EF4::ONE - z_j) * (EF4::ONE - x_j)
                })
                .product();
            full_evals_vec.push(eq_val);
        }

        // Reshape into a 32x1 matrix (Rows: (X0..X4), Cols: empty).
        let num_remaining = n - k_skip;
        let mat = RowMajorMatrix::new(full_evals_vec, 1 << num_remaining);

        // Interpolate the single column at r_skip to fold all 5 variables.
        let folded_row = interpolate_subgroup(&mat, r_skip);

        // The `folded_row` is a new 0-variable polynomial (a constant).
        let final_poly = EvaluationsList::new(folded_row);

        // Evaluate this constant polynomial. The point `r_rest` is empty.
        let expected = final_poly.evaluate(&r_rest);

        // The result of interpolation should be a single scalar.
        assert_eq!(
            final_poly.num_evals(),
            1,
            "Folding all variables should result in a single value"
        );

        assert_eq!(
            result, expected,
            "Manual skip evaluation for n=k_skip should match"
        );
    }

    proptest! {
        /// The test is set up with randomly generated dimensions and field elements. The generation
        /// is chained to ensure all inputs are valid and correctly sized:
        ///
        /// 1.  First, it generates `n`, the **total number of variables**, as a random integer from 2 to 6.
        /// 2.  Using that `n`, it generates `k_skip`, the **number of variables to skip**, from 1 to `n`.
        /// 3.  Finally, it creates two random vectors based on these dimensions:
        ///     - `point_vals`: A vector of `n` elements for the constraint point `z`.
        ///     - `r_all_vals`: A vector of `(n - k_skip) + 1` elements for the verifier's challenge object `r_all`.
        #[test]
        fn test_evaluation_and_linear_equivalence(
            (n, k_skip, point_vals, r_all_vals) in (2..=6usize)
                .prop_flat_map(|n| (
                    Just(n),
                    1..=n
                ))
                .prop_flat_map(|(n, k_skip)| (
                    Just(n),
                    Just(k_skip),
                    prop::collection::vec(any::<u32>(), n),
                    prop::collection::vec(any::<u32>(), (n - k_skip) + 1)
                ))
        ) {
            // --- SETUP ---

            // Define the random constraint point `z` from the generated values.
            let point = MultilinearPoint::new(point_vals.into_iter().map(F::from_u32).collect());

            // Define the random challenge object `r_all`.
            let r_all = MultilinearPoint::new(r_all_vals.into_iter().map(EF4::from_u32).collect());

            // --- WEIGHT 1: The symbolic `Evaluation` variant ---
            // This represents the constraint `eq_z(X)` symbolically.
            let w_eval = Weights::Evaluation { point: point.clone() };

            // --- WEIGHT 2: The materialized `Linear` variant ---
            // Manually construct the full 2^n evaluation table for `eq_z(X)`.
            let mut eq_evals_vec = Vec::with_capacity(1 << n);

            // Iterate through every point `x = (x_0, ..., x_{n-1})` on the n-dimensional hypercube.
            for i in 0..(1 << n) {
                // The evaluation of eq_z(x) is the product of n individual terms.
                // eq_z(x) = Π [z_j * x_j + (1-z_j)*(1-x_j)] for j=0..n-1
                let eq_val: EF4 = (0..n).map(|j| {
                    // Get the j-th coordinate of the constraint point z.
                    let z_j = EF4::from(point.as_slice()[j]);
                    // Get the j-th coordinate of the hypercube point x by checking the j-th bit of i.
                    // We use (n - 1 - j) to match the (X0, X1, ...) significance order.
                    let x_j = EF4::from_u32((i >> (n - 1 - j)) & 1);
                    // Calculate the j-th term of the product.
                    z_j * x_j + (EF4::ONE - z_j) * (EF4::ONE - x_j)
                }).product(); // Multiply all n terms together.
                eq_evals_vec.push(eq_val);
            }

            // The `Linear` variant stores base field elements, so we convert back.
            let eq_evals_base_vec : Vec<_> = eq_evals_vec.iter().map(|e| e.as_base().unwrap()).collect();
            let w_linear = Weights::<F>::Linear { weight: EvaluationsList::new(eq_evals_base_vec) };

            // --- ACTION ---
            // Compute the skip-aware evaluation for both symbolic and materialized variants.
            let result_eval = w_eval.compute_with_skip(&r_all, k_skip);
            let result_linear = w_linear.compute_with_skip(&r_all, k_skip);

            // --- ASSERT ---
            // The results must be identical, proving the branches are equivalent for any valid n and k_skip.
            prop_assert_eq!(result_eval, result_linear);
        }
    }
}
