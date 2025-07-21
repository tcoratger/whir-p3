use p3_field::{Field, TwoAdicField};
use p3_interpolation::interpolate_subgroup;
use p3_matrix::dense::RowMajorMatrix;

use crate::poly::multilinear::MultilinearPoint;

/// A helper for mapping a univariate point to a multilinear point.
///
/// This struct precomputes the evaluations of the "bit selector" polynomials
/// over a multiplicative subgroup. It then uses the existing `interpolate_subgroup`
/// function to efficiently evaluate the public mapping `M(Y)` at any point `y`.
#[derive(Debug, Clone)]
pub struct DomainMapper<F: Field> {
    // A matrix where column `i` contains the evaluations of the mapping
    // polynomial `M_i(Y)` over the subgroup.
    bit_selector_evals: RowMajorMatrix<F>,
}

impl<F: TwoAdicField> DomainMapper<F> {
    /// Creates a new `DomainMapper` for a given number of skipped variables `k`.
    pub fn new(k: usize) -> Self {
        let n_points = 1 << k;
        let mut row_major_evals = vec![F::ZERO; k * n_points];

        // We are building a matrix of size n_points x k.
        // - Each row `i` corresponds to a point `w^i` in the subgroup.
        // - Each column `j` corresponds to the mapping polynomial `M_j(Y)`.
        //
        // The value at `(i, j)` should be the j-th bit of the integer i.
        for i in 0..n_points {
            // For each point w^i in the subgroup
            for j in 0..k {
                // For each mapping polynomial M_j
                if (i >> j) & 1 == 1 {
                    // The j-th bit of i is 1, so M_j(w^i) should be 1.
                    row_major_evals[i * k + j] = F::ONE;
                }
            }
        }

        Self {
            bit_selector_evals: RowMajorMatrix::new(row_major_evals, k),
        }
    }

    /// Maps a univariate point `y` to a k-variable multilinear point.
    ///
    /// This implements `M(y)` by interpolating the precomputed evaluations
    /// of the bit selector polynomials.
    pub fn map_point(&self, y: F) -> MultilinearPoint<F> {
        let multilinear_coords = interpolate_subgroup(&self.bit_selector_evals, y);
        MultilinearPoint(multilinear_coords)
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    use super::*;

    type F = BabyBear;

    #[test]
    fn test_domain_mapper_k2_at_random_point() {
        // GOAL: Verify M(y) for a random y, where k=2.
        // We will manually compute the expected result and compare.

        let k = 2;
        let mapper = DomainMapper::<F>::new(k);
        let y = F::from_u64(10); // Our random evaluation point.

        //------------------------------------------------------------------
        // 1. SETUP & MANUAL CALCULATION
        //------------------------------------------------------------------

        // The subgroup H has 4 elements. The generator w is the 4th root of unity.
        let w = F::two_adic_generator(k);
        let w0 = w.exp_u64(0);
        let w1 = w.exp_u64(1);
        let w2 = w.exp_u64(2);
        let w3 = w.exp_u64(3);

        let w4 = w.exp_u64(4);
        assert_eq!(w4, F::ONE);

        // The standard mapping from H to {0,1}^2 uses the bits of the exponent i for w^i.
        // The point is (b₀, b₁), where b₀ is the LSB.
        // w^0 (i=0, bin=00) -> (0,0)
        // w^1 (i=1, bin=01) -> (1,0)
        // w^2 (i=2, bin=10) -> (0,1)
        // w^3 (i=3, bin=11) -> (1,1)

        // The mapping polynomials are built from Lagrange basis polynomials L_i(Y).
        // M₀(Y) selects the first bit (b₀), which is 1 for w¹ and w³.
        // M₁(Y) selects the second bit (b₁), which is 1 for w² and w³.
        //
        // M_0(Y) = L_1(Y) + L_3(Y)
        // M_1(Y) = L_2(Y) + L_3(Y)

        // Denominators (pre-inverted for division)
        let d0_inv = ((w0 - w1) * (w0 - w2) * (w0 - w3)).inverse(); // (1-w)(1-w²)(1-w³)⁻¹
        let d1_inv = ((w1 - w0) * (w1 - w2) * (w1 - w3)).inverse(); // (w-1)(w-w²)(w-w³)⁻¹
        let d2_inv = ((w2 - w0) * (w2 - w1) * (w2 - w3)).inverse(); // (w²-1)(w²-w)(w²-w³)⁻¹
        let d3_inv = ((w3 - w0) * (w3 - w1) * (w3 - w2)).inverse(); // (w³-1)(w³-w)(w³-w²)⁻¹

        // Lagrange polynomial evaluations at y = 10
        let _l0_at_y = (y - w1) * (y - w2) * (y - w3) * d0_inv;
        let l1_at_y = (y - w0) * (y - w2) * (y - w3) * d1_inv;
        let l2_at_y = (y - w0) * (y - w1) * (y - w3) * d2_inv;
        let l3_at_y = (y - w0) * (y - w1) * (y - w2) * d3_inv;

        // Now, compute the expected coordinates using the CORRECTED mapping polynomial formulas.
        let expected_r0 = l1_at_y + l3_at_y; // M_0(10)
        let expected_r1 = l2_at_y + l3_at_y; // M_1(10)

        let expected_point = MultilinearPoint(vec![expected_r0, expected_r1]);

        //------------------------------------------------------------------
        // 2. ACTUAL COMPUTATION & ASSERTION
        //------------------------------------------------------------------
        let result_point = mapper.map_point(y);

        assert_eq!(
            result_point, expected_point,
            "Mapped point did not match manual calculation for k=2"
        );
    }

    #[test]
    fn test_domain_mapper_k3_at_random_point() {
        // GOAL: Verify M(y) for a random y, where k=3.
        // We will manually compute the expected result and compare.

        let k = 3;
        let mapper = DomainMapper::<F>::new(k);
        let y = F::from_u64(7); // Chosen random evaluation point.

        //------------------------------------------------------------------
        // 1. SETUP & MANUAL CALCULATION
        //------------------------------------------------------------------

        // The subgroup H has 8 elements: w^0, w^1, ..., w^7.
        let w = F::two_adic_generator(k);
        let ws: Vec<F> = (0..8).map(|i| w.exp_u64(i)).collect();

        // H = [w^0, w^1, ..., w^7] corresponds to:
        //  i | bits | (b0, b1, b2)
        // ---|------|--------------
        //  0 | 000  | (0, 0, 0)
        //  1 | 001  | (1, 0, 0)
        //  2 | 010  | (0, 1, 0)
        //  3 | 011  | (1, 1, 0)
        //  4 | 100  | (0, 0, 1)
        //  5 | 101  | (1, 0, 1)
        //  6 | 110  | (0, 1, 1)
        //  7 | 111  | (1, 1, 1)

        // We define 3 mapping polynomials:
        // M_0(Y) = sum of L_i(Y) for i where bit 0 is 1 => i = 1, 3, 5, 7
        // M_1(Y) = sum of L_i(Y) for i where bit 1 is 1 => i = 2, 3, 6, 7
        // M_2(Y) = sum of L_i(Y) for i where bit 2 is 1 => i = 4, 5, 6, 7

        // Compute all Lagrange basis polynomials L_i(Y) evaluated at y = 7.
        let mut lagrange_evals = vec![F::ZERO; 8];

        for i in 0..8 {
            let xi = ws[i];
            // Denominator: product of (xi - xj) for j ≠ i
            let denom = (0..8)
                .filter(|&j| j != i)
                .map(|j| xi - ws[j])
                .product::<F>();
            let denom_inv = denom.inverse();

            // Numerator: product of (y - xj) for j ≠ i
            let numer = (0..8).filter(|&j| j != i).map(|j| y - ws[j]).product::<F>();

            lagrange_evals[i] = numer * denom_inv;
        }

        // Now construct:
        // M_0(7) = L_1 + L_3 + L_5 + L_7
        // M_1(7) = L_2 + L_3 + L_6 + L_7
        // M_2(7) = L_4 + L_5 + L_6 + L_7
        let expected_r0 =
            lagrange_evals[1] + lagrange_evals[3] + lagrange_evals[5] + lagrange_evals[7];
        let expected_r1 =
            lagrange_evals[2] + lagrange_evals[3] + lagrange_evals[6] + lagrange_evals[7];
        let expected_r2 =
            lagrange_evals[4] + lagrange_evals[5] + lagrange_evals[6] + lagrange_evals[7];

        let expected_point = MultilinearPoint(vec![expected_r0, expected_r1, expected_r2]);

        //------------------------------------------------------------------
        // 2. ACTUAL COMPUTATION & ASSERTION
        //------------------------------------------------------------------
        let result_point = mapper.map_point(y);

        assert_eq!(
            result_point, expected_point,
            "Mapped point did not match manual calculation for k=3"
        );
    }
}
