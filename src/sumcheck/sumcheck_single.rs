//! Single-round sumcheck protocol implementation.
//!
//! This module provides [`SumcheckSingle`], the core data structure for managing
//! the prover state during the quadratic sumcheck protocol.

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field, PackedFieldExtension, PackedValue, TwoAdicField};
use p3_interpolation::interpolate_subgroup;
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::{
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    sumcheck::{
        product_polynomial::ProductPolynomial,
        sumcheck_single_skip::compute_skipping_sumcheck_polynomial,
    },
    whir::{
        constraints::{Constraint, statement::EqStatement},
        proof::{SumcheckData, SumcheckSkipData},
    },
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
pub struct SumcheckSingle<F: Field, EF: ExtensionField<F>> {
    /// Paired evaluation and weight polynomials for the quadratic sumcheck.
    ///
    /// This holds both `f(x)` (the polynomial being sumchecked) and `w(x)` (the constraint
    /// weights) in a SIMD-optimized representation.
    pub(crate) poly: ProductPolynomial<F, EF>,

    /// Accumulated sum incorporating equality constraints.
    ///
    /// This is the current claimed value of `Σ_{x ∈ {0,1}^n} f(x) · w(x)`.
    /// It is updated after each round of folding.
    pub(crate) sum: EF,
}

/// Executes the initial round of the sumcheck protocol.
///
/// This function executes the initial round of the sumcheck protocol, which is unique because it
/// transitions the polynomial evaluations from the base field `F` to the extension field `EF`.
/// It computes the sumcheck polynomial, incorporates it into the prover's state, derives a challenge,
/// and then uses that challenge to compress both the polynomial evaluations and the constraint weights.
///
/// # Arguments
///
/// * `evals`: A reference to the polynomial's evaluations in the base field `F`.
/// * `sumcheck_data`: A mutable reference to the sumcheck data structure for storing polynomial evaluations.
/// * `challenger`: A mutable reference to the Fiat-Shamir challenger for transcript management.
/// * `constraint`: Constraint to combine into the sumcheck weights.
/// * `pow_bits`: The number of proof-of-work bits for the grinding protocol.
///
/// # Returns
///
/// A tuple containing:
/// * The verifier's challenge `r` as an `EF` element.
/// * [`ProductPolynomial`] with new compressed polynomial evaluations and weights in the extension field.
/// * Updated sum.
fn initial_round<F: Field, EF: ExtensionField<F>, Challenger>(
    evals: &EvaluationsList<F>,
    sumcheck_data: &mut SumcheckData<EF, F>,
    challenger: &mut Challenger,
    constraint: &Constraint<F, EF>,
    pow_bits: usize,
) -> (ProductPolynomial<F, EF>, EF, EF)
where
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    let num_vars = evals.num_variables();

    // Closure to update the sum after receiving a challenge.
    //
    // Formula: h(r) = c₀ + c₁·r + c₂·r²
    //
    // Since c₁ = h(1) - c₀ - c₂ and h(1) = claimed_sum - c₀, we get:
    // h(r) = c₂·r² + (h(1) - c₀ - c₂)·r + c₀
    let update_sum = |sum: &mut EF, c0: EF, c2: EF, r: EF| {
        let h_1 = *sum - c0;
        *sum = c2 * r.square() + (h_1 - c0 - c2) * r + c0;
    };

    // Choose between packed (SIMD) and scalar paths based on polynomial size.
    if num_vars > log2_strict_usize(F::Packing::WIDTH) {
        // Packed path: Use SIMD operations for large polynomials.
        let (mut weights, mut sum) = constraint.combine_new_packed();
        let evals_packed = EvaluationsList::new(F::Packing::pack_slice(evals.as_slice()).to_vec());

        // Compute the constant (c₀) and quadratic (c₂) coefficients of h(X).
        let (c0, c2) = evals_packed.sumcheck_coefficients(&weights);

        // Reduce packed results to scalar via horizontal sum.
        let c0 = EF::ExtensionPacking::to_ext_iter([c0]).sum();
        let c2 = EF::ExtensionPacking::to_ext_iter([c2]).sum();

        // Commit to transcript, perform PoW, and receive challenge.
        let r = sumcheck_data.observe_and_sample(challenger, c0, c2, pow_bits);

        // Fold both polynomials and update the sum.
        weights.compress(r);
        let evals = evals.compress_into_packed(r);
        update_sum(&mut sum, c0, c2, r);

        let poly = ProductPolynomial::<F, EF>::new_packed(evals, weights);
        debug_assert_eq!(poly.dot_product(), sum);
        (poly, r, sum)
    } else {
        // Scalar path: Direct computation for small polynomials.
        let (mut weights, mut sum) = constraint.combine_new();

        // Compute the constant (c₀) and quadratic (c₂) coefficients of h(X).
        let (c0, c2) = evals.sumcheck_coefficients(&weights);

        // Commit to transcript, perform PoW, and receive challenge.
        let r = sumcheck_data.observe_and_sample(challenger, c0, c2, pow_bits);

        // Fold both polynomials and update the sum.
        weights.compress(r);
        let evals = evals.compress_ext(r);
        update_sum(&mut sum, c0, c2, r);

        let poly = ProductPolynomial::<F, EF>::new_small(evals, weights);
        debug_assert_eq!(poly.dot_product(), sum);
        (poly, r, sum)
    }
}

impl<F, EF> SumcheckSingle<F, EF>
where
    F: Field + Ord,
    EF: ExtensionField<F>,
{
    /// Constructs a new `SumcheckSingle` instance from evaluations in the extension field.
    ///
    /// This function:
    /// - Uses precomputed evaluations of the polynomial `p` over the Boolean hypercube,
    ///   where `p` is already represented over the extension field `EF`.
    /// - Applies the provided `Statement` to compute equality weights and the expected sum.
    /// - Initializes the internal state used in the sumcheck protocol.
    ///
    /// This is the entry point when the polynomial is defined directly over `EF`.
    pub fn from_extension_evals(
        evals: EvaluationsList<EF>,
        statement: EqStatement<EF>,
        challenge: EF,
    ) -> Self {
        let k = evals.num_variables();

        let constraint = Constraint::new_eq_only(challenge, statement);
        if k > log2_strict_usize(F::Packing::WIDTH) {
            let (weights, sum) = constraint.combine_new_packed();
            let evals = EvaluationsList::new(
                evals
                    .0
                    .chunks(<F as Field>::Packing::WIDTH)
                    .map(EF::ExtensionPacking::from_ext_slice)
                    .collect(),
            );
            Self {
                poly: ProductPolynomial::<F, EF>::new_packed(evals, weights),
                sum,
            }
        } else {
            let (weights, sum) = constraint.combine_new();
            Self {
                poly: ProductPolynomial::<F, EF>::new_small(evals, weights),
                sum,
            }
        }
    }

    /// Constructs a new `SumcheckSingle` instance from evaluations in the base field.
    ///
    /// This function:
    /// - Uses precomputed evaluations of the polynomial `p` over the Boolean hypercube.
    /// - Applies the given constraint `Statement` using a random linear combination.
    /// - Initializes internal sumcheck state with weights and expected sum.
    /// - Applies first set of sumcheck rounds
    #[instrument(skip_all)]
    pub fn from_base_evals<Challenger>(
        evals: &EvaluationsList<F>,
        sumcheck: &mut SumcheckData<EF, F>,
        challenger: &mut Challenger,
        folding_factor: usize,
        pow_bits: usize,
        constraint: &Constraint<F, EF>,
    ) -> (Self, MultilinearPoint<EF>)
    where
        F: TwoAdicField,
        EF: TwoAdicField,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        assert_ne!(folding_factor, 0);

        let k = evals.num_variables();
        assert!(
            folding_factor > 0,
            "must be initialize with at least one round for base to ext transition"
        );
        assert!(
            folding_factor <= k,
            "number of rounds must be less than or equal to instance size"
        );

        let (mut poly, r, mut sum) =
            initial_round(evals, sumcheck, challenger, constraint, pow_bits);

        let rs = core::iter::once(r)
            .chain(
                (1..folding_factor).map(|_| poly.round(sumcheck, challenger, &mut sum, pow_bits)),
            )
            .collect();

        (Self { poly, sum }, MultilinearPoint::new(rs))
    }

    /// Constructs a new `SumcheckSingle` instance from evaluations in the base field.
    ///
    /// This function:
    /// - Uses precomputed evaluations of the polynomial `p` over the Boolean hypercube.
    /// - Applies the given constraint `Statement` using a random linear combination.
    /// - Initializes internal sumcheck state with weights and expected sum.
    /// - Applies first set of sumcheck rounds with univariate skip optimization.
    #[instrument(skip_all)]
    #[allow(clippy::too_many_arguments)]
    pub fn with_skip<Challenger>(
        evals: &EvaluationsList<F>,
        skip_data: &mut SumcheckSkipData<EF, F>,
        challenger: &mut Challenger,
        folding_factor: usize,
        pow_bits: usize,
        k_skip: usize,
        constraint: &Constraint<F, EF>,
    ) -> (Self, MultilinearPoint<EF>)
    where
        F: TwoAdicField,
        EF: TwoAdicField,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        assert_ne!(folding_factor, 0);
        assert!(k_skip > 1);
        assert!(k_skip <= folding_factor);

        constraint.validate_for_skip_case();
        let (weights, sum) = constraint.combine_new();

        // Compute the skipped-round polynomial h and the rectangular views f̂, ŵ.
        //
        // - `sumcheck_poly`: The univariate polynomial sent to the verifier for this round.
        // - `f_mat`, `w_mat`: The original evaluations reshaped into matrices of size 2^k x 2^(n-k).
        let num_remaining_vars = evals.num_variables() - k_skip;
        let width = 1 << num_remaining_vars;

        // Create the matrices that we'll need for interpolation later.
        let f_mat = evals.clone().into_mat(width);
        let w_mat = weights.into_mat(width);

        // Compute the sumcheck polynomial.
        let sumcheck_poly = compute_skipping_sumcheck_polynomial(f_mat.clone(), w_mat.clone());

        debug_assert_eq!(
            sumcheck_poly
                .evaluations()
                .iter()
                .step_by(2)
                .copied()
                .sum::<EF>(),
            sum
        );
        let polynomial_skip_evaluation = sumcheck_poly.evaluations();

        // Fiat–Shamir: commit to h by absorbing its M evaluations into the transcript.
        challenger.observe_algebra_slice(polynomial_skip_evaluation);

        // Store skip evaluations
        skip_data
            .evaluations
            .extend_from_slice(polynomial_skip_evaluation);

        // Proof-of-work challenge to delay prover (only if pow_bits > 0).
        if pow_bits > 0 {
            skip_data.pow = challenger.grind(pow_bits);
        }

        // Receive the verifier challenge for this entire collapsed round.
        let r: EF = challenger.sample_algebra_element();

        // Interpolate the LDE matrices at the folding randomness to get the new "folded" polynomial state.
        let new_p = EvaluationsList::new(interpolate_subgroup(&f_mat, r));
        let new_w = EvaluationsList::new(interpolate_subgroup(&w_mat, r));
        let mut poly = ProductPolynomial::new(new_p, new_w);

        // While we could interpolate sumcheck_poly, it's cheaper and easier to just use
        // the new_p and new_w evaluations.
        let mut sum = poly.dot_product();

        // Apply rest of sumcheck rounds
        let rs = core::iter::once(r)
            .chain(
                (k_skip..folding_factor)
                    .map(|_| poly.round(&mut skip_data.sumcheck, challenger, &mut sum, pow_bits)),
            )
            .collect();

        (Self { poly, sum }, MultilinearPoint::new(rs))
    }

    /// Returns the number of variables in the polynomial.
    pub fn num_variables(&self) -> usize {
        self.poly.num_variables()
    }

    /// Returns the polynomial evaluations.
    #[instrument(skip_all)]
    pub fn evals(&self) -> EvaluationsList<EF> {
        self.poly.evals()
    }

    /// Returns the weight polynomial evaluations (test only).
    #[cfg(test)]
    pub fn weights(&self) -> EvaluationsList<EF> {
        self.poly.weights()
    }

    /// Evaluates the sumcheck polynomial at a given multilinear point.
    pub fn eval(&self, point: &MultilinearPoint<EF>) -> EF {
        self.poly.eval(point)
    }

    /// Returns the number of evaluations in the polynomial.
    pub const fn num_evals(&self) -> usize {
        self.poly.num_evals()
    }

    /// Executes the sumcheck protocol for a multilinear polynomial with optional **univariate skip**.
    ///
    /// This function performs `folding_factor` rounds of the sumcheck protocol:
    ///
    /// - At each round, a univariate polynomial is sent representing a partial sum over a subset of variables.
    /// - The verifier responds with a random challenge that is used to fix one variable.
    /// - Optionally, the first `k` rounds can be skipped using the **univariate skip** optimization,
    ///   which collapses multiple Boolean variables at once over a multiplicative subgroup.
    ///
    /// The univariate skip is performed entirely in the base field and reduces expensive extension field
    /// computations, improving prover efficiency.
    ///
    /// # Arguments
    /// - `sumcheck_data`: A mutable reference to the sumcheck data structure for storing polynomial evaluations.
    /// - `challenger`: A mutable reference to the Fiat-Shamir challenger for transcript management and PoW grinding.
    /// - `folding_factor`: Number of variables to fold in total.
    /// - `pow_bits`: Number of PoW bits used to delay the prover (0.0 to disable).
    /// - `constraint`: Optional constraint to combine into the sumcheck weights.
    ///
    /// # Returns
    /// A `MultilinearPoint<EF>` representing the verifier's challenges across all folded variables.
    ///
    /// # Panics
    /// - If `folding_factor > num_variables()`
    /// - If univariate skip is attempted with evaluations in the extension field.
    #[instrument(skip_all)]
    pub fn compute_sumcheck_polynomials<Challenger>(
        &mut self,
        sumcheck_data: &mut SumcheckData<EF, F>,
        challenger: &mut Challenger,
        folding_factor: usize,
        pow_bits: usize,
        constraint: Option<Constraint<F, EF>>,
    ) -> MultilinearPoint<EF>
    where
        F: TwoAdicField,
        EF: TwoAdicField,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        if let Some(constraint) = constraint {
            self.poly.combine(&mut self.sum, &constraint);
        }

        // Standard round-by-round folding
        // Proceed with one-variable-per-round folding for remaining variables.
        let res = (0..folding_factor)
            .map(|_| {
                self.poly
                    .round(sumcheck_data, challenger, &mut self.sum, pow_bits)
            })
            .collect();

        // Return the full vector of verifier challenges as a multilinear point.
        MultilinearPoint::new(res)
    }
}
