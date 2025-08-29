use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_interpolation::interpolate_subgroup;
use p3_multilinear_util::{point::MultilinearPoint};

use crate::poly::evals::EvaluationsList;

/// Represents a multilinear point `z` used in an evaluation constraint of the form `p(z) = s`.
///
/// This structure provides a symbolic representation of an equality constraint at a
/// specific multilinear point. The actual weight polynomial `w(X) = eq_z(X)` is
/// materialized from this point on-demand when needed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConstraintPoint<F>(pub MultilinearPoint<F>);

impl<F: Field> ConstraintPoint<F> {
    /// Constructs a new evaluation point constraint.
    #[must_use]
    pub const fn new(point: MultilinearPoint<F>) -> Self {
        Self(point)
    }

    /// Returns the number of variables in the evaluation point's domain.
    #[must_use]
    pub const fn num_variables(&self) -> usize {
        self.0.num_variables()
    }

    /// Constructs an evaluation point for a univariate check, expanded to a multilinear domain.
    ///
    /// This is a special case of `new` used for univariate sumchecks, where
    /// a point `y` is mapped to a multilinear point `(y, y^2, y^4, ...)`.
    pub fn univariate(point: F, size: usize) -> Self {
        Self(MultilinearPoint::expand_from_univariate(point, size))
    }

    /// Evaluates the weight function `eq_z(X)` at a given folding point.
    #[must_use]
    pub fn compute(&self, folding_randomness: &MultilinearPoint<F>) -> F {
        self.0.eq_poly(folding_randomness)
    }

    /// Evaluates the weight polynomial `eq_z(X)` at a challenge point `r` using univariate skip semantics.
    ///
    /// TODO: simplify this but for now we just want this to work.
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
        // The constraint point `z` must be defined over the full n-variable domain.
        let k = self.0.num_variables();
        assert!(
            k <= n,
            "Constraint point cannot have more variables than the domain"
        );

        // Construct the evaluation table for the polynomial eq_z(X).
        let evals = EvaluationsList::new_from_point(&self.0, F::ONE);

        // Reshape the evaluation table into a matrix for folding
        //
        // The flat list of 2^n evaluations is viewed as a `2^k_skip x 2^(n-k_skip)` matrix.
        // Rows correspond to the skipped variables (X0, ..., Xk-1).
        // Columns correspond to the remaining variables (Xk, ..., Xn-1).
        let num_remaining_vars = n - k_skip;
        let width = 1 << num_remaining_vars;
        let mat = evals.into_mat(width);

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
    use p3_matrix::dense::RowMajorMatrix;

    use super::*;

    type F = BabyBear;
    type EF4 = BinomialExtensionField<F, 4>;

    #[test]
    fn test_evaluation_point_new() {
        // Define a point in the multilinear space
        let point = MultilinearPoint::new(vec![F::ONE, F::ZERO]);
        let constraint_point = ConstraintPoint::new(point);

        // The number of variables in the weight should match the number of variables in the point
        assert_eq!(constraint_point.num_variables(), 2);
    }

    #[test]
    fn test_accumulate() {
        // Initialize an empty accumulator
        let mut accumulator = EvaluationsList::new(vec![F::ZERO, F::ZERO]);

        // Define an evaluation point
        let point = MultilinearPoint::new(vec![F::ONE]);

        // Define a multiplication factor
        let factor = F::from_u64(5);

        // Accumulate weighted values
        accumulator.accumulate(&point, factor);

        // Compute expected result manually
        let expected = EvaluationsList::new_from_point(&point, factor);

        assert_eq!(accumulator, expected);
    }

    #[test]
    fn test_univariate_constraint_point_one_variable() {
        // y = 3, n = 1 → [3]
        let y = F::from_u64(3);
        let constraint_point = ConstraintPoint::univariate(y, 1);

        // Expect point to be [3]
        let expected = MultilinearPoint::new(vec![y]);
        assert_eq!(constraint_point, ConstraintPoint::new(expected));
    }

    #[test]
    fn test_univariate_constraint_point_two_variables() {
        // y = 4, n = 2 → [y^2, y] = [16, 4]
        let y = F::from_u64(4);
        let constraint_point = ConstraintPoint::univariate(y, 2);

        let expected = MultilinearPoint::new(vec![y.square(), y]);
        assert_eq!(constraint_point, ConstraintPoint::new(expected));
    }

    #[test]
    fn test_univariate_constraint_point_four_variables() {
        // y = 3, n = 4 → [3^8, 3^4, 3^2, 3]
        let y = F::from_u64(3);
        let constraint_point = ConstraintPoint::univariate(y, 4);

        let expected = MultilinearPoint::new(vec![y.exp_u64(8), y.exp_u64(4), y.square(), y]);

        assert_eq!(constraint_point, ConstraintPoint::new(expected));
    }

    #[test]
    fn test_univariate_constraint_point_zero_variables() {
        let y = F::from_u64(10);
        let constraint_point = ConstraintPoint::univariate(y, 0);

        // Expect empty point
        let expected = MultilinearPoint::new(vec![]);
        assert_eq!(constraint_point, ConstraintPoint::new(expected));
    }

    #[test]
    fn test_compute_with_skip() {
        // SETUP:
        // - n = 3 total variables: (X0, X1, X2).
        // - The constraint point `z` is defined over the full n=3 variables.
        // - k_skip = 2 variables to skip: X0, X1.
        let n = 3;
        let k_skip = 2;

        // The weight polynomial is W(X) = eq_z(X0, X1, X2), where z=(2,3,4).
        // The constraint point MUST be full-dimensional.
        let point = MultilinearPoint::new(vec![F::from_u32(2), F::from_u32(3), F::from_u32(4)]);
        let constraint_point = ConstraintPoint::new(point.clone());

        // The verifier's full challenge object `r_all`.
        // It has (n - k_skip) + 1 = (3 - 2) + 1 = 2 elements.
        // - r_rest for remaining variable (X2).
        // - r_skip for the combined (X0, X1) domain.
        let r_rest = MultilinearPoint::new(vec![EF4::from_u32(5)]);
        let r_skip = EF4::from_u32(7);
        let r_all = MultilinearPoint::new([r_rest.as_slice(), &[r_skip]].concat());

        // ACTION: Compute W(r) using the function under test.
        let result = constraint_point.compute_with_skip(&r_all, k_skip);

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
        let constraint_point = ConstraintPoint::new(point.clone());

        // The verifier's challenge object `r_all`.
        // It has (n - k_skip) + 1 = (5 - 5) + 1 = 1 element.
        // - r_rest is an empty vector for the 0 remaining variables.
        // - r_skip is the single challenge for the combined (X0..X4) domain.
        let r_rest = MultilinearPoint::new(vec![]);
        let r_skip = EF4::from_u32(13);
        let r_all = MultilinearPoint::new(vec![r_skip]);

        // Compute W(r) using the function under test.
        let result = constraint_point.compute_with_skip(&r_all, k_skip);

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
}
