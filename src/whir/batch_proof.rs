//! Batch opening proof for WHIR protocol using the selector variable approach.
//!
//! This module implements batch polynomial opening where two polynomials f_a and f_b
//! are opened simultaneously at their respective evaluation points z_a and z_b.
//!
//! # Selector Variable Approach
//!
//! Given two polynomials f_a and f_b (both with m variables), we construct a combined
//! polynomial f_c with m+1 variables:
//!
//! ```text
//! f_c(X, x_1, ..., x_m) = X·f_a(x_1, ..., x_m) + (1-X)·f_b(x_1, ..., x_m)
//! ```
//!
//! The selector variable X chooses between f_a (when X=1) and f_b (when X=0).
//!
//! The combined weight polynomial is:
//!
//! ```text
//! w(X, b) = X·eq(b, z_a) + α(1-X)·eq(b, z_b)
//! ```
//!
//! The first sumcheck round folds the selector variable, producing:
//! - Folded polynomial: g(b) = r_0·f_a(b) + (1-r_0)·f_b(b)
//! - Folded weights: w'(b) = r_0·eq(b, z_a) + α(1-r_0)·eq(b, z_b)
//! - Folded claim: σ' = r_0·v_a + α(1-r_0)·v_b
//!
//! The protocol then continues with standard WHIR on the folded polynomial.

use alloc::vec::Vec;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::dense::DenseMatrix;
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use serde::{Deserialize, Serialize};

use super::proof::{SumcheckData, WhirProof};
use crate::{
    fiat_shamir::errors::FiatShamirError,
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    sumcheck::{product_polynomial::ProductPolynomial, sumcheck_single::SumcheckSingle},
    whir::{committer::Witness, constraints::statement::EqStatement, prover::Prover},
};

/// Batch opening proof wrapper
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound(
    serialize = "F: Serialize, EF: Serialize, [F; DIGEST_ELEMS]: Serialize",
    deserialize = "F: Deserialize<'de>, EF: Deserialize<'de>, [F; DIGEST_ELEMS]: Deserialize<'de>"
))]
pub struct BatchWhirProof<F, EF, const DIGEST_ELEMS: usize> {
    /// Commitment to first polynomial (f_a)
    pub commitment_a: [F; DIGEST_ELEMS],

    /// Commitment to second polynomial (f_b)
    pub commitment_b: [F; DIGEST_ELEMS],

    /// Selector sumcheck data: stores [c0, c2] for h(X) = c0 + c1·X + c2·X²
    /// c0 = h(0) = α·v_b
    /// c2 = quadratic coefficient
    /// Verifier derives c1
    pub selector_sumcheck: SumcheckData<EF, F>,

    /// Inner WHIR proof on the folded polynomial g = r_0·f_a + (1-r_0)·f_b
    pub inner_proof: WhirProof<F, EF, DIGEST_ELEMS>,
}

impl<EF, F, H, C, Challenger> Prover<'_, EF, F, H, C, Challenger>
where
    F: TwoAdicField + Ord,
    EF: ExtensionField<F> + TwoAdicField,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    /// Performs batch opening of two polynomials using the selector variable approach.
    ///
    /// This function executes the batch WHIR protocol:
    /// 1. Sample batching randomness α
    /// 2. Run selector round (first round of sumcheck on selector variable)
    /// 3. Fold to get combined polynomial g
    /// 4. Continue with standard WHIR on g
    ///
    /// # Arguments
    ///
    /// * `dft` - DFT backend for polynomial operations
    /// * `proof` - Mutable proof structure to fill in
    /// * `challenger` - Fiat-Shamir transcript
    /// * `statement_a` - Evaluation constraints for polynomial A (point z_a, value v_a)
    /// * `witness_a` - Polynomial A with its Merkle commitment
    /// * `statement_b` - Evaluation constraints for polynomial B (point z_b, value v_b)
    /// * `witness_b` - Polynomial B with its Merkle commitment
    ///
    /// # Errors
    ///
    /// Returns an error if the protocol fails.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_prove<Dft: TwoAdicSubgroupDft<F>, const DIGEST_ELEMS: usize>(
        &self,
        dft: &Dft,
        proof: &mut BatchWhirProof<F, EF, DIGEST_ELEMS>,
        challenger: &mut Challenger,
        statement_a: &EqStatement<EF>,
        witness_a: &Witness<EF, F, DenseMatrix<F>, DIGEST_ELEMS>,
        statement_b: &EqStatement<EF>,
        witness_b: &Witness<EF, F, DenseMatrix<F>, DIGEST_ELEMS>,
    ) -> Result<(), FiatShamirError>
    where
        H: CryptographicHasher<F, [F; DIGEST_ELEMS]>
            + CryptographicHasher<F::Packing, [F::Packing; DIGEST_ELEMS]>
            + Sync,
        C: PseudoCompressionFunction<[F; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[F::Packing; DIGEST_ELEMS], 2>
            + Sync,
        [F; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        // Validate that both polynomials have the same number of variables
        assert_eq!(
            witness_a.polynomial.num_variables(),
            witness_b.polynomial.num_variables(),
            "Batch opening requires same-degree polynomials"
        );

        // ==== Step 1: Store commitments ====
        proof.commitment_a = witness_a.prover_data.root().into();
        proof.commitment_b = witness_b.prover_data.root().into();

        // ==== Step 2: Sample batching randomness α ====
        let alpha: EF = challenger.sample_algebra_element();

        // ==== Step 3: Extract claims from statements ====
        // For now, we assume single-constraint statements
        let (z_a, v_a) = extract_single_constraint(statement_a);
        let (z_b, v_b) = extract_single_constraint(statement_b);

        // ==== Step 4: Run selector round ====
        let (sumcheck_prover, r_0) = self.selector_round(
            &mut proof.selector_sumcheck,
            challenger,
            &witness_a.polynomial,
            &witness_b.polynomial,
            &z_a,
            &z_b,
            v_a,
            v_b,
            alpha,
        );

        // ==== Step 5: Continue with inner WHIR ====
        // The sumcheck_prover now contains the folded polynomial g and weights w'
        // We need to continue with the WHIR rounds on this folded state

        // TODO: Continue with inner WHIR rounds
        // This requires setting up a RoundState with the folded polynomial
        // and running the remaining rounds

        // For now, store the folding challenge for verification
        let _ = (sumcheck_prover, r_0, dft);

        Ok(())
    }

    /// Executes the selector round of the batch opening protocol.
    ///
    /// This is the first round of sumcheck over the selector variable X,
    /// which combines the two polynomials into one folded polynomial.
    ///
    /// # Arguments
    ///
    /// * `selector_data` - Sumcheck data structure to fill in
    /// * `challenger` - Fiat-Shamir transcript
    /// * `f_a` - Evaluations of the first polynomial
    /// * `f_b` - Evaluations of the second polynomial
    /// * `z_a` - Evaluation point for f_a
    /// * `z_b` - Evaluation point for f_b
    /// * `v_a` - Claimed evaluation f_a(z_a)
    /// * `v_b` - Claimed evaluation f_b(z_b)
    /// * `alpha` - Batching randomness
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// * `SumcheckSingle<F, EF>` - The folded sumcheck state
    /// * `EF` - The folding challenge r_0
    #[allow(clippy::too_many_arguments)]
    fn selector_round(
        &self,
        selector_data: &mut SumcheckData<EF, F>,
        challenger: &mut Challenger,
        f_a: &EvaluationsList<F>,
        f_b: &EvaluationsList<F>,
        z_a: &MultilinearPoint<EF>,
        z_b: &MultilinearPoint<EF>,
        v_a: EF,
        v_b: EF,
        alpha: EF,
    ) -> (SumcheckSingle<F, EF>, EF) {
        // ==== Create combined polynomial and weights for selector round ====
        //
        // The combined polynomial f_c(X, b) = X·f_a(b) + (1-X)·f_b(b) over {0,1}^{m+1}
        // has evaluations [f_b | f_a] (first half is f_b at X=0, second half is f_a at X=1)
        //
        // The combined weight w(X, b) = X·eq(b, z_a) + α(1-X)·eq(b, z_b)
        // has evaluations [α·eq(·, z_b) | eq(·, z_a)]

        // Build combined polynomial: [f_b | f_a]
        let combined_evals: Vec<F> = f_b
            .as_slice()
            .iter()
            .chain(f_a.as_slice().iter())
            .copied()
            .collect();
        let combined_poly = EvaluationsList::new(combined_evals);

        // Build combined weights: [α·eq(·, z_b) | eq(·, z_a)]
        let eq_z_b = EvaluationsList::new_from_point(z_b.as_slice(), alpha);
        let eq_z_a = EvaluationsList::new_from_point(z_a.as_slice(), EF::ONE);
        let combined_weights_vec: Vec<EF> = eq_z_b
            .as_slice()
            .iter()
            .chain(eq_z_a.as_slice().iter())
            .copied()
            .collect();
        let combined_weights = EvaluationsList::new(combined_weights_vec);

        // ==== Compute sumcheck polynomial coefficients ====
        //
        // h(X) = Σ_b f_c(X, b) · w(X, b)
        // We compute c0 = h(0) and c2 (quadratic coefficient)
        let (c0, c2) = combined_poly.sumcheck_coefficients(&combined_weights);

        // Sanity check: c0 should equal α·v_b
        debug_assert_eq!(c0, alpha * v_b, "c0 = h(0) should equal α·v_b");

        // ==== Fiat-Shamir: observe and sample ====
        let pow_bits = self.starting_folding_pow_bits;
        let r_0 = selector_data.observe_and_sample::<Challenger, F>(challenger, c0, c2, pow_bits);

        // ==== Fold the polynomial and weights ====
        //
        // g(b) = r_0·f_a(b) + (1-r_0)·f_b(b)
        // w'(b) = r_0·eq(b, z_a) + α(1-r_0)·eq(b, z_b)
        // σ' = h(r_0)

        // Folded polynomial in extension field
        let g = EvaluationsList::linear_combination(f_a, r_0, f_b, EF::ONE - r_0);

        // Folded weights: r_0·eq(·, z_a) + (1-r_0)·α·eq(·, z_b)
        let w_prime: Vec<EF> = eq_z_a
            .iter()
            .zip(eq_z_b.iter())
            .map(|(&a, &b)| r_0 * a + (EF::ONE - r_0) * b)
            .collect();
        let w_prime = EvaluationsList::new(w_prime);

        // Folded sum: σ' = h(r_0)
        let sigma = v_a + alpha * v_b;
        let h_1 = sigma - c0;
        let c1 = h_1 - c0 - c2;
        let sigma_prime = c0 + c1 * r_0 + c2 * r_0.square();

        // ==== Create SumcheckSingle for continuation ====
        let poly = ProductPolynomial::new(g, w_prime);
        debug_assert_eq!(poly.dot_product(), sigma_prime);

        let sumcheck_prover = SumcheckSingle {
            poly,
            sum: sigma_prime,
        };

        (sumcheck_prover, r_0)
    }
}

/// Extracts a single constraint (point, evaluation) from an EqStatement.
///
/// # Panics
///
/// Panics if the statement is empty.
fn extract_single_constraint<EF: Field>(statement: &EqStatement<EF>) -> (MultilinearPoint<EF>, EF) {
    assert!(
        !statement.is_empty(),
        "Statement must contain at least one constraint"
    );
    let (point, &eval) = statement.iter().next().unwrap();
    (point.clone(), eval)
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};
    use rand::{Rng, SeedableRng, rngs::SmallRng};

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    /// Test that the selector round produces correct coefficients.
    #[test]
    fn test_selector_coefficients() {
        let num_vars = 3;
        let num_evals = 1 << num_vars;

        let mut rng = SmallRng::seed_from_u64(42);

        // Random polynomials
        let f_a_evals: Vec<F> = (0..num_evals).map(|_| rng.random()).collect();
        let f_b_evals: Vec<F> = (0..num_evals).map(|_| rng.random()).collect();
        let f_a = EvaluationsList::new(f_a_evals);
        let f_b = EvaluationsList::new(f_b_evals);

        // Random evaluation points
        let z_a = MultilinearPoint::<EF>::rand(&mut rng, num_vars);
        let z_b = MultilinearPoint::<EF>::rand(&mut rng, num_vars);

        // Compute actual evaluations
        let v_a = f_a.evaluate_hypercube_base(&z_a);
        let v_b = f_b.evaluate_hypercube_base(&z_b);

        let alpha = EF::from_u64(7);

        // Build combined structures
        let combined_evals: Vec<F> = f_b
            .as_slice()
            .iter()
            .chain(f_a.as_slice().iter())
            .copied()
            .collect();
        let combined_poly = EvaluationsList::new(combined_evals);

        let eq_z_b = EvaluationsList::new_from_point(z_b.as_slice(), alpha);
        let eq_z_a = EvaluationsList::<EF>::new_from_point(z_a.as_slice(), EF::ONE);
        let combined_weights_vec: Vec<EF> = eq_z_b
            .as_slice()
            .iter()
            .chain(eq_z_a.as_slice().iter())
            .copied()
            .collect();
        let combined_weights = EvaluationsList::new(combined_weights_vec);

        let (c0, _c2) = combined_poly.sumcheck_coefficients(&combined_weights);

        // Verify c0 = h(0) = α·v_b
        assert_eq!(c0, alpha * v_b);

        // Verify σ = h(0) + h(1) = v_a + α·v_b
        let sigma = v_a + alpha * v_b;
        let h_1 = sigma - c0;
        assert_eq!(h_1, v_a);
    }

    /// Test that folding produces the correct polynomial.
    #[test]
    fn test_folding() {
        let num_vars = 2;
        let num_evals = 1 << num_vars;

        let f_a = EvaluationsList::new(vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ]);
        let f_b = EvaluationsList::new(vec![
            F::from_u64(5),
            F::from_u64(6),
            F::from_u64(7),
            F::from_u64(8),
        ]);

        let r_0 = EF::from_u64(3);
        let g = EvaluationsList::linear_combination(&f_a, r_0, &f_b, EF::ONE - r_0);

        for i in 0..num_evals {
            let expected =
                r_0 * EF::from(f_a.as_slice()[i]) + (EF::ONE - r_0) * EF::from(f_b.as_slice()[i]);
            assert_eq!(g.as_slice()[i], expected);
        }
    }

    /// Test that the folded sum matches h(r_0) from the sumcheck polynomial.
    ///
    /// This verifies the key sumcheck invariant: after folding with challenge r_0,
    /// the new sum σ' = Σ_b g(b)·w'(b) equals h(r_0).
    #[test]
    fn test_folded_sum() {
        let num_vars = 3;

        let mut rng = SmallRng::seed_from_u64(123);

        let f_a_evals: Vec<F> = (0..(1 << num_vars)).map(|_| rng.random()).collect();
        let f_b_evals: Vec<F> = (0..(1 << num_vars)).map(|_| rng.random()).collect();
        let f_a = EvaluationsList::new(f_a_evals);
        let f_b = EvaluationsList::new(f_b_evals);

        let z_a = MultilinearPoint::<EF>::rand(&mut rng, num_vars);
        let z_b = MultilinearPoint::<EF>::rand(&mut rng, num_vars);

        let v_a = f_a.evaluate_hypercube_base(&z_a);
        let v_b = f_b.evaluate_hypercube_base(&z_b);

        let alpha = EF::from_u64(5);
        let r_0 = EF::from_u64(7);

        // Build combined polynomial and weights (as in selector_round)
        let combined_evals: Vec<F> = f_b
            .as_slice()
            .iter()
            .chain(f_a.as_slice().iter())
            .copied()
            .collect();
        let combined_poly = EvaluationsList::new(combined_evals);

        let eq_z_b = EvaluationsList::<EF>::new_from_point(z_b.as_slice(), alpha);
        let eq_z_a = EvaluationsList::<EF>::new_from_point(z_a.as_slice(), EF::ONE);
        let combined_weights_vec: Vec<EF> = eq_z_b
            .as_slice()
            .iter()
            .chain(eq_z_a.as_slice().iter())
            .copied()
            .collect();
        let combined_weights = EvaluationsList::new(combined_weights_vec);

        // Compute sumcheck polynomial coefficients
        let (c0, c2) = combined_poly.sumcheck_coefficients(&combined_weights);
        let sigma = v_a + alpha * v_b;
        let h_1 = sigma - c0;
        let c1 = h_1 - c0 - c2;

        // Expected sum: h(r_0) via the quadratic polynomial
        let expected_sum = c0 + c1 * r_0 + c2 * r_0.square();

        // Compute folded polynomial and weights
        let g = EvaluationsList::linear_combination(&f_a, r_0, &f_b, EF::ONE - r_0);
        let w_prime: Vec<EF> = eq_z_a
            .iter()
            .zip(eq_z_b.iter())
            .map(|(&a, &b)| r_0 * a + (EF::ONE - r_0) * b)
            .collect();
        let w_prime = EvaluationsList::new(w_prime);

        // Actual sum: Σ_b g(b)·w'(b)
        let actual_sum: EF = g
            .iter()
            .zip(w_prime.iter())
            .map(|(&g_b, &w_b)| g_b * w_b)
            .sum();

        // Verify the sumcheck invariant
        assert_eq!(actual_sum, expected_sum);
    }
}
