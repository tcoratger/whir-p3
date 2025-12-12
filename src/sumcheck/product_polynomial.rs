//! SIMD-aware polynomial pair for quadratic sumcheck.
//!
//! This module implements a data structure that manages two multilinear polynomials.
//!
//! Evaluations and weights—whose pointwise product is required for the sumcheck protocol.
//!
//! # Mathematical Background
//!
//! In the sumcheck protocol, we prove knowledge of a claimed sum:
//!
//! ```text
//! S = \sum_{x \in \{0,1\}^n} f(x) \cdot w(x)
//! ```
//!
//! where:
//! - `f(x)` is the multilinear polynomial being sumchecked (evaluations).
//! - `w(x)` is the weight polynomial, typically derived from equality constraints.
//!
//! At each round, we compute a univariate polynomial `h(X)` that represents the partial sum
//! over remaining variables. For quadratic sumcheck, `h(X)` is degree-2.

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field, PackedFieldExtension, PackedValue, dot_product};
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::{
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    whir::{constraints::Constraint, proof::SumcheckData},
};

/// A paired representation of evaluation and weight polynomials for quadratic sumcheck.
///
/// This enum stores two multilinear polynomials:
/// - `evals` (the polynomial being sumchecked)
/// - `weights` (the constraint weights),
///
/// in either SIMD-packed or scalar format, depending on the polynomial size.
///
/// # Memory Layout
///
/// In packed mode, evaluations are organized as:
///
/// ```text
/// Logical view:   [f(0,0,...,0), f(0,0,...,1), ..., f(1,1,...,1)]
///                 \---------------- 2^n elements ---------------/
///
/// Packed view:    [Pack_0, Pack_1, ..., Pack_{2^n/W - 1}]
///                 \--2^{n - log_2(W)} packed elements--/
///
/// where W = SIMD_WIDTH and each Pack_i contains W consecutive field elements.
/// ```
///
/// # Variants
///
/// - Packed: Uses SIMD-packed extension field elements for large polynomials.
///   Each packed element contains `F::Packing::WIDTH` consecutive evaluations, enabling
///   parallel arithmetic across SIMD lanes.
///
/// - Small: Uses scalar extension field elements for small polynomials
///   where SIMD overhead would exceed the benefit.
///
/// # Transition Logic
///
/// The representation transitions from `Packed` to `Small` when:
///
/// ```text
/// num_variables <= log_2(SIMD_WIDTH)
/// ```
///
/// This occurs after sufficient rounds of folding reduce the polynomial size below the
/// SIMD efficiency threshold.
#[derive(Debug, Clone)]
pub(crate) enum ProductPolynomial<F: Field, EF: ExtensionField<F>> {
    /// SIMD-packed representation for large polynomials.
    ///
    /// Each element in `evals` and `weights` is an `EF::ExtensionPacking`, which holds
    /// `F::Packing::WIDTH` extension field elements packed into SIMD lanes.
    ///
    /// # Memory Efficiency
    ///
    /// For a polynomial with `2^n` evaluations and SIMD width `W`:
    /// - Stored elements: `2^{n - log_2(W)}`
    /// - Memory per element: `sizeof(EF) * W`
    /// - Total memory: `2^n * sizeof(EF)` (same as scalar, but with SIMD alignment)
    Packed {
        /// Packed evaluations of the polynomial `f(x)` being sumchecked.
        ///
        /// Layout: `evals[i]` contains logical evaluations at indices `[i*W, (i+1)*W)`.
        evals: EvaluationsList<EF::ExtensionPacking>,

        /// Packed evaluations of the weight polynomial `w(x)`.
        ///
        /// Derived from equality constraints and challenge batching.
        weights: EvaluationsList<EF::ExtensionPacking>,
    },

    /// Scalar representation for small polynomials.
    ///
    /// Each element in `evals` and `weights` is a single extension field element.
    ///
    /// Used when the polynomial is too small for SIMD packing to be beneficial.
    Small {
        /// Scalar evaluations of the polynomial `f(x)` being sumchecked.
        evals: EvaluationsList<EF>,

        /// Scalar evaluations of the weight polynomial `w(x)`.
        weights: EvaluationsList<EF>,
    },
}

impl<F: Field, EF: ExtensionField<F>> ProductPolynomial<F, EF> {
    /// Creates a new [`ProductPolynomial`] from extension field evaluations.
    ///
    /// Automatically selects the optimal representation (packed or scalar) based on
    /// the polynomial size relative to the SIMD width.
    ///
    /// # Decision Criteria
    ///
    /// ```text
    /// if num_variables > log_2(SIMD_WIDTH):
    ///     -> Packed (SIMD benefits outweigh overhead)
    /// else:
    ///     -> Small (scalar operations are more efficient)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `evals` - Evaluations of the polynomial `f(x)` over the boolean hypercube.
    /// * `weights` - Evaluations of the weight polynomial `w(x)`.
    ///
    /// # Panics
    ///
    /// Panics if `evals` and `weights` have different numbers of variables.
    pub(crate) fn new(evals: EvaluationsList<EF>, weights: EvaluationsList<EF>) -> Self {
        // Validate that both polynomials have the same number of variables.
        //
        // This is essential since they must be multiplied pointwise.
        assert_eq!(evals.num_variables(), weights.num_variables());

        // Determine the SIMD threshold: log_2(packing width).
        //
        // If num_variables > threshold, SIMD packing is beneficial.
        let simd_threshold = log2_strict_usize(F::Packing::WIDTH);

        if evals.num_variables() > simd_threshold {
            // Convert scalar evaluations to packed format.
            //
            // We chunk consecutive evaluations into groups of SIMD_WIDTH and pack them.
            // This enables parallel arithmetic on all elements within a chunk.
            let evals = EvaluationsList::new(
                evals
                    .0
                    .chunks(F::Packing::WIDTH)
                    .map(EF::ExtensionPacking::from_ext_slice)
                    .collect(),
            );
            let weights = EvaluationsList::new(
                weights
                    .0
                    .chunks(F::Packing::WIDTH)
                    .map(EF::ExtensionPacking::from_ext_slice)
                    .collect(),
            );
            Self::new_packed(evals, weights)
        } else {
            // Polynomial is small enough that scalar operations are more efficient.
            Self::new_small(evals, weights)
        }
    }

    /// Creates a packed variant and checks for immediate transition.
    ///
    /// This constructor is used when we know the data is already in packed format.
    /// It performs a transition check in case the packed data has been folded down
    /// to a single packed element.
    ///
    /// # Arguments
    ///
    /// * `evals` - Packed evaluations of `f(x)`.
    /// * `weights` - Packed evaluations of `w(x)`.
    pub(crate) fn new_packed(
        evals: EvaluationsList<EF::ExtensionPacking>,
        weights: EvaluationsList<EF::ExtensionPacking>,
    ) -> Self {
        let mut poly = Self::Packed { evals, weights };

        // Check if we should immediately transition to scalar mode.
        // This handles edge cases where the input is already small.
        poly.transition();
        poly
    }

    /// Creates a scalar variant.
    ///
    /// Used when the polynomial is too small for SIMD packing.
    ///
    /// # Arguments
    ///
    /// * `evals` - Scalar evaluations of `f(x)`.
    /// * `weights` - Scalar evaluations of `w(x)`.
    pub(crate) const fn new_small(
        evals: EvaluationsList<EF>,
        weights: EvaluationsList<EF>,
    ) -> Self {
        Self::Small { evals, weights }
    }

    /// Returns the number of variables in the multilinear polynomials.
    ///
    /// This is the logical number of variables, accounting for SIMD packing.
    ///
    /// # Computation
    ///
    /// - **Packed**: `stored_variables + log_2(SIMD_WIDTH)`
    /// - **Small**: `stored_variables`
    pub(crate) fn num_variables(&self) -> usize {
        match self {
            Self::Packed { evals, weights } => {
                // Get the number of variables in the packed representation.
                let k = evals.num_variables();
                assert_eq!(k, weights.num_variables());

                // Add back the variables absorbed by SIMD packing.
                k + log2_strict_usize(F::Packing::WIDTH)
            }
            Self::Small { evals, weights } => {
                let k = evals.num_variables();
                assert_eq!(k, weights.num_variables());
                k
            }
        }
    }

    /// Evaluates the polynomial `f(x)` at a given multilinear point.
    ///
    /// This computes `f(point)` where `point \in EF^n`.
    ///
    /// # Arguments
    ///
    /// * `point` - The evaluation point as a [`MultilinearPoint`].
    pub(crate) fn eval(&self, point: &MultilinearPoint<EF>) -> EF {
        match self {
            Self::Packed { evals, .. } => evals.eval_hypercube_packed(point),
            Self::Small { evals, .. } => evals.evaluate_hypercube_ext(point),
        }
    }

    /// Folds both polynomials by binding the first variable to a challenge.
    ///
    /// This is the core operation of each sumcheck round. After receiving a challenge `r`,
    /// we reduce the polynomial from `n` variables to `n-1` variables by setting `X_1 = r`.
    ///
    /// # Mathematical Operation
    ///
    /// For a multilinear polynomial `p(X_1, X_2, ..., X_n)`:
    ///
    /// ```text
    /// p'(X_2, ..., X_n) = p(r, X_2, ..., X_n)
    ///                   = p(0, X_2, ..., X_n) + r * (p(1, X_2, ..., X_n) - p(0, X_2, ..., X_n))
    /// ```
    ///
    /// This linear interpolation is applied independently to both `evals` and `weights`.
    ///
    /// # Arguments
    ///
    /// * `r` - The verifier's challenge for this round.
    fn compress(&mut self, r: EF) {
        match self {
            // Apply folding to both packed polynomials.
            //
            // The compress operation handles SIMD lanes correctly.
            Self::Packed { evals, weights } => {
                evals.compress(r);
                weights.compress(r);
            }
            // Apply folding to both scalar polynomials.
            Self::Small { evals, weights } => {
                evals.compress(r);
                weights.compress(r);
            }
        }
    }

    /// Returns the total number of evaluations in the polynomial.
    ///
    /// This is `2^n` where `n` is the number of variables.
    pub(crate) const fn num_evals(&self) -> usize {
        match self {
            Self::Packed { evals, .. } => evals.num_evals() * F::Packing::WIDTH,
            Self::Small { evals, .. } => evals.num_evals(),
        }
    }

    /// Transitions from packed to scalar mode if the polynomial is small enough.
    ///
    /// This is called after each fold operation to check if we should switch representations.
    /// The transition occurs when the packed representation has only a single element
    /// (i.e., `num_variables() == 0` in the packed view).
    ///
    /// # Transition Condition
    ///
    /// ```text
    /// if packed_num_variables == 0:
    ///     -> Unpack to scalar and switch to Small variant
    /// ```
    ///
    /// # Why Transition?
    ///
    /// When only one packed element remains, SIMD operations become pure overhead:
    /// - No parallelism benefit (only one "lane group" of work)
    /// - Extra unpacking/repacking costs per operation
    ///
    /// Scalar mode eliminates this overhead for the final rounds.
    fn transition(&mut self) {
        if let Self::Packed { evals, weights } = self {
            // Check if we've folded down to a single packed element.
            let k = evals.num_variables();
            assert_eq!(k, weights.num_variables());

            if k == 0 {
                // Unpack the single packed element into SIMD_WIDTH scalar elements.
                //
                // Extract individual extension field elements from the packed representation.
                let evals =
                    EF::ExtensionPacking::to_ext_iter(evals.as_slice().iter().copied()).collect();
                let weights =
                    EF::ExtensionPacking::to_ext_iter(weights.as_slice().iter().copied()).collect();

                // Replace self with the scalar variant.
                *self = Self::Small {
                    evals: EvaluationsList::new(evals),
                    weights: EvaluationsList::new(weights),
                };
            }
        }
    }

    /// Executes one round of the quadratic sumcheck protocol.
    ///
    /// This is the main method that:
    /// 1. Computes the sumcheck polynomial coefficients `(c_0, c_2)`.
    /// 2. Commits them to the Fiat-Shamir transcript.
    /// 3. Receives a challenge from the verifier.
    /// 4. Folds both polynomials using the challenge.
    /// 5. Updates the running sum.
    ///
    /// # Sumcheck Polynomial
    ///
    /// At each round, we send a univariate quadratic polynomial:
    ///
    /// ```text
    ///     h(X) = c_0 + c_1 * X + c_2 * X^2
    /// ```
    ///
    /// where:
    /// - `c_0 = h(0)` = sum of products where first variable is 0
    /// - `c_1` = derived from the constraint `h(0) + h(1) = claimed_sum`
    /// - `c_2` = quadratic coefficient from cross-terms
    ///
    /// We only send `(c_0, c_2)` since `c_1` is derivable by the verifier.
    ///
    /// # Arguments
    ///
    /// * `sumcheck_data` - Storage for polynomial evaluations sent to verifier.
    /// * `challenger` - Fiat-Shamir challenger for transcript operations.
    /// * `sum` - Current claimed sum (updated after this round).
    /// * `pow_bits` - Proof-of-work difficulty (0 to disable).
    ///
    /// # Returns
    ///
    /// The verifier's challenge `r \in EF` for this round.
    #[instrument(skip_all)]
    pub(crate) fn round<Challenger>(
        &mut self,
        sumcheck_data: &mut SumcheckData<EF, F>,
        challenger: &mut Challenger,
        sum: &mut EF,
        pow_bits: usize,
    ) -> EF
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Step 1: Compute sumcheck polynomial coefficients.
        //
        // The strategy differs based on representation to maximize SIMD utilization.
        let (c0, c2) = match self {
            Self::Packed { evals, weights } => {
                // Compute coefficients using packed arithmetic.
                // Each operation processes SIMD_WIDTH elements in parallel.
                let (c0, c2) = evals.sumcheck_coefficients(weights);

                // Horizontal reduction: sum across all SIMD lanes to get scalar result.
                //
                // The packed computation gives us one result per lane.
                // We need the sum across all lanes as the final coefficient.
                (
                    EF::ExtensionPacking::to_ext_iter([c0]).sum(),
                    EF::ExtensionPacking::to_ext_iter([c2]).sum(),
                )
            }
            Self::Small { evals, weights } => {
                // Compute coefficients directly on scalar elements.
                evals.sumcheck_coefficients(weights)
            }
        };

        // Step 2-4: Commit to transcript, do PoW, and receive challenge.
        let r = sumcheck_data.observe_and_sample(challenger, c0, c2, pow_bits);

        // Step 5: Fold both polynomials using the challenge.
        self.compress(r);

        // Step 6: Update the claimed sum using the quadratic formula.
        //
        // Recall: h(X) = c_0 + c_1 * X + c_2 * X^2
        //
        // From the sumcheck constraint: h(0) + h(1) = claimed_sum
        //   -> c_0 + (c_0 + c_1 + c_2) = claimed_sum
        //   -> c_1 = claimed_sum - 2 * c_0 - c_2
        //
        // Therefore: h(r) = c_0 + c_1 * r + c_2 * r^2
        //                 = c_0 + (claimed_sum - 2 * c_0 - c_2) * r + c_2 * r^2
        //                 = c_2 * r^2 + (h(1) - c_0 - c_2) * r + c_0
        //
        // where h(1) = claimed_sum - c_0.
        let h_1 = *sum - c0;
        *sum = c2 * r.square() + (h_1 - c0 - c2) * r + c0;

        // Sanity check: the updated sum should equal the inner product after folding.
        debug_assert_eq!(*sum, self.dot_product());

        // Step 7: Check if we should transition to scalar mode.
        //
        // After folding, the polynomial may be small enough that scalar operations
        // are more efficient than packed operations.
        self.transition();

        r
    }

    /// Extracts the evaluation polynomial as a scalar [`EvaluationsList`].
    ///
    /// This unpacks the evaluations if in packed mode.
    ///
    /// # Returns
    ///
    /// A copy of the evaluations in scalar extension field format.
    pub(crate) fn evals(&self) -> EvaluationsList<EF> {
        match self {
            Self::Packed { evals, .. } => EvaluationsList::new(
                EF::ExtensionPacking::to_ext_iter(evals.as_slice().iter().copied()).collect(),
            ),
            Self::Small { evals, .. } => evals.clone(),
        }
    }

    /// Incorporates new constraints into the weight polynomial.
    ///
    /// This is used when additional constraints need to be folded into the sumcheck
    /// after initial construction (e.g., from STIR challenges).
    ///
    /// # Arguments
    ///
    /// * `sum` - Running sum to update with new constraint contributions.
    /// * `constraint` - The constraint to combine into weights.
    pub(crate) fn combine(&mut self, sum: &mut EF, constraint: &Constraint<F, EF>) {
        match self {
            Self::Packed { weights, .. } => {
                constraint.combine_packed(weights, sum);
            }
            Self::Small { weights, .. } => {
                constraint.combine(weights, sum);
            }
        }
    }

    /// Extracts the weight polynomial as a scalar [`EvaluationsList`].
    ///
    /// This unpacks the weights if in packed mode. Only available in tests.
    #[cfg(test)]
    pub(crate) fn weights(&self) -> EvaluationsList<EF> {
        match &self {
            Self::Packed { weights, .. } => EvaluationsList::new(
                EF::ExtensionPacking::to_ext_iter(weights.as_slice().iter().copied()).collect(),
            ),
            Self::Small { weights, .. } => weights.clone(),
        }
    }

    /// Computes the dot product of evaluations and weights.
    ///
    /// This computes:
    ///
    /// ```text
    ///     \sum_{x \in \{0,1\}^n} evals(x) * weights(x)
    /// ```
    ///
    /// which should equal the current claimed sum at any point in the protocol.
    ///
    /// # Returns
    ///
    /// The dot product of `evals` and `weights`.
    pub(crate) fn dot_product(&self) -> EF {
        match self {
            Self::Packed { evals, weights } => {
                // Compute packed dot product (SIMD parallel multiply-accumulate).
                let sum_packed = dot_product(evals.iter().copied(), weights.iter().copied());

                // Horizontal sum to reduce packed result to scalar.
                EF::ExtensionPacking::to_ext_iter([sum_packed]).sum()
            }
            Self::Small { evals, weights } => {
                // Direct scalar dot product.
                dot_product(evals.iter().copied(), weights.iter().copied())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::{vec, vec::Vec};

    use p3_baby_bear::BabyBear;
    use p3_field::{Field, PrimeCharacteristicRing, extension::BinomialExtensionField};
    use rand::{Rng, SeedableRng, rngs::SmallRng};

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    #[test]
    fn test_new_selects_small_variant_for_small_polynomials() {
        // Create polynomials with 2 variables (4 evaluations).
        //
        // For BabyBear, SIMD width is 16, so log_2(16) = 4.
        // Since 2 <= 4, this should use the Small variant.
        let e0 = EF::from_u64(1);
        let e1 = EF::from_u64(2);
        let e2 = EF::from_u64(3);
        let e3 = EF::from_u64(4);
        let w0 = EF::from_u64(5);
        let w1 = EF::from_u64(6);
        let w2 = EF::from_u64(7);
        let w3 = EF::from_u64(8);

        let evals = EvaluationsList::new(vec![e0, e1, e2, e3]);
        let weights = EvaluationsList::new(vec![w0, w1, w2, w3]);

        let poly = ProductPolynomial::<F, EF>::new(evals, weights);

        // Verify it selected the Small variant and check internal state.
        match &poly {
            ProductPolynomial::Small { evals, weights } => {
                // Verify stored evaluations match input.
                assert_eq!(evals.as_slice(), &[e0, e1, e2, e3]);
                assert_eq!(weights.as_slice(), &[w0, w1, w2, w3]);
            }
            ProductPolynomial::Packed { .. } => {
                panic!("Expected Small variant for 2-variable polynomial");
            }
        }
    }

    #[test]
    fn test_new_selects_packed_variant_for_large_polynomials() {
        // Create polynomials with 6 variables (64 evaluations).
        //
        // For BabyBear, SIMD width is 16, so log_2(16) = 4.
        // Since 6 > 4, this should use the Packed variant.
        let evals_vec: Vec<EF> = (0..64).map(|i| EF::from_u64(i as u64)).collect();
        let weights_vec: Vec<EF> = (0..64).map(|i| EF::from_u64(100 + i as u64)).collect();

        let evals = EvaluationsList::new(evals_vec.clone());
        let weights = EvaluationsList::new(weights_vec.clone());

        let poly = ProductPolynomial::<F, EF>::new(evals, weights);

        // Verify it selected the Packed variant.
        match &poly {
            ProductPolynomial::Packed {
                evals: packed_evals,
                weights: packed_weights,
            } => {
                // With 64 elements and SIMD width 16, we should have 64/16 = 4 packed elements.
                let simd_width = <F as Field>::Packing::WIDTH;
                let expected_packed_len = 64 / simd_width;
                assert_eq!(packed_evals.num_evals(), expected_packed_len);
                assert_eq!(packed_weights.num_evals(), expected_packed_len);
            }
            ProductPolynomial::Small { .. } => {
                panic!("Expected Packed variant for 6-variable polynomial");
            }
        }

        // Verify evals() correctly unpacks to original values.
        assert_eq!(poly.evals().as_slice(), &evals_vec);
        assert_eq!(poly.weights().as_slice(), &weights_vec);
    }

    #[test]
    #[should_panic]
    fn test_new_panics_on_mismatched_sizes() {
        // Evals has 4 elements (2 variables), weights has 8 elements (3 variables).
        let evals = EvaluationsList::new(vec![EF::ONE; 4]);
        let weights = EvaluationsList::new(vec![EF::TWO; 8]);

        // This should panic because evals and weights have different num_variables.
        let _ = ProductPolynomial::<F, EF>::new(evals, weights);
    }

    #[test]
    fn test_num_variables_small_variant() {
        // Create a Small variant with 3 variables (8 evaluations).
        let evals = EvaluationsList::new(vec![EF::ONE; 8]);
        let weights = EvaluationsList::new(vec![EF::TWO; 8]);

        // Force Small variant by using new_small directly.
        let poly = ProductPolynomial::<F, EF>::new_small(evals, weights);

        // The logical number of variables should be 3 (since 2^3 = 8).
        assert_eq!(poly.num_variables(), 3);
    }

    #[test]
    fn test_num_variables_packed_variant() {
        // Create a Packed variant with 6 variables (64 evaluations).
        //
        // After packing with SIMD width 16, we have:
        //   - 64 / 16 = 4 packed elements
        //   - stored_variables = log_2(4) = 2
        //   - total_variables = stored_variables + log_2(16) = 2 + 4 = 6
        let evals = EvaluationsList::new(vec![EF::ONE; 64]);
        let weights = EvaluationsList::new(vec![EF::TWO; 64]);

        let poly = ProductPolynomial::<F, EF>::new(evals, weights);

        // Verify it's Packed and has correct num_variables.
        assert!(matches!(poly, ProductPolynomial::Packed { .. }));
        assert_eq!(poly.num_variables(), 6);
    }

    #[test]
    fn test_dot_product_manual_calculation() {
        // Create Small variant with known values and verify dot product.
        //
        // dot_product = Σ_i evals[i] * weights[i]
        //             = e0*w0 + e1*w1 + e2*w2 + e3*w3
        let e0 = EF::from_u64(1);
        let e1 = EF::from_u64(2);
        let e2 = EF::from_u64(3);
        let e3 = EF::from_u64(4);
        let w0 = EF::from_u64(5);
        let w1 = EF::from_u64(6);
        let w2 = EF::from_u64(7);
        let w3 = EF::from_u64(8);

        let evals = EvaluationsList::new(vec![e0, e1, e2, e3]);
        let weights = EvaluationsList::new(vec![w0, w1, w2, w3]);

        let poly = ProductPolynomial::<F, EF>::new_small(evals, weights);

        // dot_product = e0*w0 + e1*w1 + e2*w2 + e3*w3
        let expected = e0 * w0 + e1 * w1 + e2 * w2 + e3 * w3;
        assert_eq!(poly.dot_product(), expected);
    }

    #[test]
    fn test_sumcheck_coefficients_manual_calculation() {
        // Test the sumcheck coefficient computation with manual verification.
        //
        // For a 1-variable polynomial (2 evaluations):
        //   evals   = [e0, e1] where f(0) = e0, f(1) = e1
        //   weights = [w0, w1] where g(0) = w0, g(1) = w1
        //
        // The sumcheck polynomial h(X) = f(X) * g(X) where:
        //   f(X) = e0 + (e1 - e0)*X
        //   g(X) = w0 + (w1 - w0)*X
        //
        // h(X) = [e0 + (e1-e0)*X] * [w0 + (w1-w0)*X]
        //      = e0*w0 + [e0*(w1-w0) + (e1-e0)*w0]*X + (e1-e0)*(w1-w0)*X^2
        //      = c0 + c1*X + c2*X^2
        //
        // where:
        //   c0 = e0 * w0
        //   c2 = (e1 - e0) * (w1 - w0)
        //
        // The sumcheck_coefficients function returns (c0, c2).
        let e0 = EF::from_u64(3);
        let e1 = EF::from_u64(7);
        let w0 = EF::from_u64(2);
        let w1 = EF::from_u64(5);

        let evals = EvaluationsList::new(vec![e0, e1]);
        let weights = EvaluationsList::new(vec![w0, w1]);

        let (c0, c2) = evals.sumcheck_coefficients(&weights);

        // c0 = e0 * w0
        let expected_c0 = e0 * w0;
        assert_eq!(c0, expected_c0);

        // c2 = (e1 - e0) * (w1 - w0)
        let expected_c2 = (e1 - e0) * (w1 - w0);
        assert_eq!(c2, expected_c2);

        // Verify consistency: h(0) + h(1) should equal the claimed sum.
        // h(0) = c0
        // h(1) = e1 * w1
        // sum = e0*w0 + e1*w1
        let h_0 = c0;
        let h_1 = e1 * w1;
        let sum = e0 * w0 + e1 * w1;
        assert_eq!(h_0 + h_1, sum);
    }

    #[test]
    fn test_compress_manual_calculation() {
        // Test the compress (folding) operation with manual verification.
        //
        // Initial state: 2-variable polynomial (4 evaluations)
        //   evals   = [e0, e1, e2, e3] representing f(x0, x1)
        //   weights = [w0, w1, w2, w3] representing g(x0, x1)
        //
        // Memory layout:
        //   f(0,0) = e0, f(0,1) = e1 (lo half, x0 = 0)
        //   f(1,0) = e2, f(1,1) = e3 (hi half, x0 = 1)
        //
        // Folding binds x0 to challenge r:
        //   f'(x1) = f(0,x1) + r * (f(1,x1) - f(0,x1))
        //
        // So:
        //   e'0 = e0 + r * (e2 - e0)
        //   e'1 = e1 + r * (e3 - e1)
        //   w'0 = w0 + r * (w2 - w0)
        //   w'1 = w1 + r * (w3 - w1)
        let e0 = EF::from_u64(1);
        let e1 = EF::from_u64(2);
        let e2 = EF::from_u64(5);
        let e3 = EF::from_u64(8);
        let w0 = EF::from_u64(3);
        let w1 = EF::from_u64(4);
        let w2 = EF::from_u64(6);
        let w3 = EF::from_u64(7);

        let evals = EvaluationsList::new(vec![e0, e1, e2, e3]);
        let weights = EvaluationsList::new(vec![w0, w1, w2, w3]);

        let mut poly = ProductPolynomial::<F, EF>::new_small(evals, weights);

        // Initial dot product: sum = e0*w0 + e1*w1 + e2*w2 + e3*w3
        let initial_sum = e0 * w0 + e1 * w1 + e2 * w2 + e3 * w3;
        assert_eq!(poly.dot_product(), initial_sum);

        // Fold with challenge r.
        let r = EF::from_u64(2);
        poly.compress(r);

        let folded_evals = poly.evals();
        let folded_weights = poly.weights();

        // e'0 = e0 + r * (e2 - e0)
        // e'1 = e1 + r * (e3 - e1)
        // w'0 = w0 + r * (w2 - w0)
        // w'1 = w1 + r * (w3 - w1)
        let expected_e0 = e0 + r * (e2 - e0);
        let expected_e1 = e1 + r * (e3 - e1);
        let expected_w0 = w0 + r * (w2 - w0);
        let expected_w1 = w1 + r * (w3 - w1);

        assert_eq!(folded_evals.as_slice(), &[expected_e0, expected_e1]);
        assert_eq!(folded_weights.as_slice(), &[expected_w0, expected_w1]);

        // After folding, dot_product equals h(r) where h is the sumcheck polynomial:
        //   h(X) = c0 + c1*X + c2*X^2
        //   c0 = e0*w0 + e1*w1
        //   c2 = (e2 - e0)*(w2 - w0) + (e3 - e1)*(w3 - w1)
        //   h(1) = e2*w2 + e3*w3
        //   c1 = h(1) - c0 - c2
        //   h(r) = c0 + c1*r + c2*r^2
        let c0 = e0 * w0 + e1 * w1;
        let c2 = (e2 - e0) * (w2 - w0) + (e3 - e1) * (w3 - w1);
        let h_1 = e2 * w2 + e3 * w3;
        let c1 = h_1 - c0 - c2;
        let h_r = c0 + c1 * r + c2 * r.square();

        assert_eq!(poly.dot_product(), h_r);
    }

    #[test]
    fn test_eval_multilinear_interpolation() {
        // Test eval() with non-boolean points using multilinear interpolation.
        //
        // For a 2-variable polynomial f(x0, x1):
        //   f(x0, x1) = f(0,0)*(1-x0)*(1-x1) + f(0,1)*(1-x0)*x1
        //             + f(1,0)*x0*(1-x1)     + f(1,1)*x0*x1
        //
        // With evals = [e0, e1, e2, e3]:
        //   f(0,0) = e0, f(0,1) = e1, f(1,0) = e2, f(1,1) = e3
        let e0 = EF::from_u64(2);
        let e1 = EF::from_u64(5);
        let e2 = EF::from_u64(3);
        let e3 = EF::from_u64(11);

        let evals = EvaluationsList::new(vec![e0, e1, e2, e3]);
        let weights = EvaluationsList::new(vec![EF::ONE; 4]);

        let poly = ProductPolynomial::<F, EF>::new_small(evals, weights);

        // Evaluate at (x0, x1):
        //   f(x0, x1) = e0*(1-x0)*(1-x1) + e1*(1-x0)*x1 + e2*x0*(1-x1) + e3*x0*x1
        let x0 = EF::from_u64(3);
        let x1 = EF::from_u64(4);
        let point = MultilinearPoint::new(vec![x0, x1]);

        let one = EF::ONE;
        let expected = e0 * (one - x0) * (one - x1)
            + e1 * (one - x0) * x1
            + e2 * x0 * (one - x1)
            + e3 * x0 * x1;

        assert_eq!(poly.eval(&point), expected);
    }

    #[test]
    fn test_transition_from_packed_to_small() {
        // Create a Packed variant that will transition to Small after sufficient folding.
        //
        // The SIMD threshold is log_2(F::Packing::WIDTH).
        // We need a polynomial large enough to start in Packed mode.
        let simd_width = <F as Field>::Packing::WIDTH;
        let simd_log = log2_strict_usize(simd_width);

        // Start with simd_log + 2 variables (e.g., if simd_width=16, start with 6 vars = 64 evals).
        // This gives us 2 packed elements initially (1 stored variable).
        let num_vars = simd_log + 2;
        let num_evals = 1 << num_vars;

        let evals = EvaluationsList::new(vec![EF::ONE; num_evals]);
        let weights = EvaluationsList::new(vec![EF::ONE; num_evals]);

        let mut poly = ProductPolynomial::<F, EF>::new(evals, weights);

        // Initially should be Packed with correct internal structure.
        match &poly {
            ProductPolynomial::Packed {
                evals: packed_evals,
                weights: packed_weights,
            } => {
                // Should have num_evals / simd_width = 4 packed elements.
                let expected_packed_len = num_evals / simd_width;
                assert_eq!(packed_evals.num_evals(), expected_packed_len);
                assert_eq!(packed_weights.num_evals(), expected_packed_len);
            }
            ProductPolynomial::Small { .. } => {
                panic!("Expected Packed variant initially");
            }
        }
        assert_eq!(poly.num_variables(), num_vars);

        // Fold twice to reduce to simd_log variables (threshold for transition).
        for _ in 0..2 {
            let challenge = EF::from_u64(7);
            poly.compress(challenge);
            poly.transition();
        }

        // After two folds: simd_log variables, which triggers transition to Small.
        match &poly {
            ProductPolynomial::Small { evals, weights } => {
                // Should have 2^simd_log = simd_width scalar elements.
                assert_eq!(evals.num_evals(), simd_width);
                assert_eq!(weights.num_evals(), simd_width);
            }
            ProductPolynomial::Packed { .. } => {
                panic!("Expected Small variant after transition");
            }
        }
        assert_eq!(poly.num_variables(), simd_log);
    }

    #[test]
    fn test_new_packed_with_single_element_transitions() {
        // If we create a Packed variant with just 1 packed element (0 stored variables),
        // it should immediately transition to Small.
        //
        // This happens when packed evals has exactly 1 element.
        type EP = <EF as ExtensionField<F>>::ExtensionPacking;

        // Get the actual SIMD width to create properly sized arrays.
        let simd_width = <F as Field>::Packing::WIDTH;

        // Create a single packed element containing simd_width extension field elements.
        let evals_scalar: Vec<EF> = (0..simd_width).map(|i| EF::from_u64(i as u64)).collect();
        let weights_scalar: Vec<EF> = (0..simd_width)
            .map(|i| EF::from_u64(100 + i as u64))
            .collect();

        let evals = EvaluationsList::new(vec![EP::from_ext_slice(&evals_scalar)]);
        let weights = EvaluationsList::new(vec![EP::from_ext_slice(&weights_scalar)]);

        let poly = ProductPolynomial::<F, EF>::new_packed(evals, weights);

        // Should have transitioned to Small with correct values.
        match &poly {
            ProductPolynomial::Small {
                evals: small_evals,
                weights: small_weights,
            } => {
                // Verify the unpacked values match the original scalars.
                assert_eq!(small_evals.as_slice(), &evals_scalar);
                assert_eq!(small_weights.as_slice(), &weights_scalar);
            }
            ProductPolynomial::Packed { .. } => {
                panic!("Expected Small variant after transition from single packed element");
            }
        }

        // Should have log_2(simd_width) variables.
        assert_eq!(poly.num_variables(), log2_strict_usize(simd_width));
    }

    #[test]
    fn test_consistency_between_variants() {
        // Create the same logical polynomial in both variants and verify they behave identically.
        let mut rng = SmallRng::seed_from_u64(999);

        // Use 4 variables (16 elements) - at SIMD boundary.
        let evals_vec: Vec<EF> = (0..16).map(|_| rng.random()).collect();
        let weights_vec: Vec<EF> = (0..16).map(|_| rng.random()).collect();

        // Create Small variant directly.
        let small = ProductPolynomial::<F, EF>::new_small(
            EvaluationsList::new(evals_vec.clone()),
            EvaluationsList::new(weights_vec.clone()),
        );

        // The auto-selection might choose either variant depending on SIMD width.
        let auto = ProductPolynomial::<F, EF>::new(
            EvaluationsList::new(evals_vec.clone()),
            EvaluationsList::new(weights_vec.clone()),
        );

        // Both should have the same num_variables.
        assert_eq!(small.num_variables(), auto.num_variables());

        // Both should have the same num_evals.
        assert_eq!(small.num_evals(), auto.num_evals());

        // Both should have the same dot_product.
        assert_eq!(small.dot_product(), auto.dot_product());

        // Both should evaluate to the same value at the same point.
        let point = MultilinearPoint::new(vec![EF::from_u64(3); 4]);
        assert_eq!(small.eval(&point), auto.eval(&point));

        // Both should extract the same evals.
        assert_eq!(small.evals().as_slice(), auto.evals().as_slice());
    }
}
