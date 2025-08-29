use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_interpolation::interpolate_subgroup;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::{
    eq::{eval_eq, eval_eq_base},
    point::MultilinearPoint,
};
use tracing::instrument;

use super::sumcheck_polynomial::SumcheckPolynomial;
use crate::{
    fiat_shamir::prover::ProverState,
    poly::evals::EvaluationsList,
    sumcheck::{
        sumcheck_single_skip::compute_skipping_sumcheck_polynomial, utils::sumcheck_quadratic,
    },
    whir::statement::Statement,
};

const PARALLEL_THRESHOLD: usize = 4096;

/// Folds a list of evaluations from a base field `F` into an extension field `EF`.
///
/// This function performs an out-of-place compression of a polynomial's evaluations. It takes evaluations
/// over a base field `F`, folds them using a random value `r` from an extension field `EF`, and returns a new
/// list of evaluations in `EF`. This operation effectively reduces the number of variables in the
/// represented multilinear polynomial by one.
///
/// ## Arguments
/// * `evals`: A reference to an `EvaluationsList<F>` containing the evaluations of a multilinear
///   polynomial over the boolean hypercube in the base field `F`.
/// * `r`: A value `r` from the extension field `EF`, used as the random challenge for folding.
///
/// ## Returns
/// A new `EvaluationsList<EF>` containing the compressed evaluations in the extension field.
///
/// The compression is achieved by applying the following formula to pairs of evaluations:
///
/// The compression is achieved by applying the following formula to pairs of evaluations:
/// $p'(X_2, ..., X_n) = (p(1, X_2, ..., X_n) - p(0, X_2, ..., X_n)) \cdot r + p(0, X_2, ..., X_n)$
#[instrument(skip_all)]
pub fn compress_ext<F: Field, EF: ExtensionField<F>>(
    evals: &EvaluationsList<F>,
    r: EF,
) -> EvaluationsList<EF> {
    assert_ne!(evals.num_variables(), 0);

    // Fold between base and extension field elements
    let fold = |slice: &[F]| -> EF { r * (slice[1] - slice[0]) + slice[0] };

    // Threshold below which sequential computation is faster
    //
    // This was chosen based on experiments with the `compress` function.
    // It is possible that the threshold can be tuned further.
    let folded = if evals.num_evals() >= PARALLEL_THRESHOLD {
        evals.as_slice().par_chunks_exact(2).map(fold).collect()
    } else {
        evals.as_slice().chunks_exact(2).map(fold).collect()
    };

    EvaluationsList::new(folded)
}

/// Compresses a list of evaluations in-place using a random challenge.
///
/// This function performs an in-place memory optimization for the folding step
/// in the sumcheck protocol.
///
/// ## Algorithm
/// For smaller inputs, this function avoids allocating a new vector by overwriting
/// the first half of the existing evaluation list and then truncating it.
/// For larger inputs (determined by `PARALLEL_THRESHOLD`), it uses a parallel,
/// out-of-place method to maximize performance, consistent with the original implementation.
///
/// ## Arguments
/// * `evals`: A mutable reference to an `EvaluationsList<F>`, which will be modified in-place.
/// * `r`: A value from the field `F`, used as the random folding challenge.
///
/// ## Mathematical Formula
/// The compression is achieved by applying the following formula to pairs of evaluations:
/// $p'(X_2, ..., X_n) = (p(1, X_2, ..., X_n) - p(0, X_2, ..., X_n)) \cdot r + p(0, X_2, ..., X_n)$
pub fn compress<F: Field>(evals: &mut EvaluationsList<F>, r: F) {
    // Ensure the polynomial is not a constant (i.e., has variables to fold).
    assert_ne!(evals.num_variables(), 0);

    // For large inputs, we use the original parallel, out-of-place strategy for maximum speed.
    if evals.num_evals() >= PARALLEL_THRESHOLD {
        // Define the folding operation for a pair of elements.
        let fold = |slice: &[F]| -> F { r * (slice[1] - slice[0]) + slice[0] };
        // Execute the fold in parallel and collect into a new vector.
        let folded = evals.as_slice().par_chunks_exact(2).map(fold).collect();
        // Replace the old evaluations with the new, folded evaluations.
        *evals = EvaluationsList::new(folded);
    } else {
        // For smaller inputs, we use the sequential, in-place strategy to save memory.
        let mid = evals.num_evals() / 2;
        let evals_slice = evals.as_mut_slice();
        for i in 0..mid {
            let p0 = evals_slice[2 * i];
            let p1 = evals_slice[2 * i + 1];
            evals_slice[i] = r * (p1 - p0) + p0;
        }
        evals.truncate(mid);
    }
}

/// Executes the initial round of the sumcheck protocol.
///
/// This function executes the initial round of the sumcheck protocol, which is unique because it
/// transitions the polynomial evaluations from the base field `F` to the extension field `EF`.
/// It computes the sumcheck polynomial, incorporates it into the prover's state, derives a challenge,
/// and then uses that challenge to compress both the polynomial evaluations and the constraint weights.
///
/// ## Arguments
/// * `prover_state`: A mutable reference to the `ProverState`, which manages the Fiat-Shamir transcript.
/// * `evals`: A reference to the polynomial's evaluations in the base field `F`.
/// * `weights`: A mutable reference to the weight evaluations in the extension field `EF`.
/// * `sum`: A mutable reference to the claimed sum, which is updated with the new value after folding.
/// * `pow_bits`: The number of proof-of-work bits for the grinding protocol.
///
/// ## Returns
/// A tuple containing:
/// * The verifier's challenge `r` as an `EF` element.
/// * The new, compressed polynomial evaluations as an `EvaluationsList<EF>`.
fn initial_round<Challenger, F: Field, EF: ExtensionField<F>>(
    prover_state: &mut ProverState<F, EF, Challenger>,
    evals: &EvaluationsList<F>,
    weights: &mut EvaluationsList<EF>,
    sum: &mut EF,
    pow_bits: usize,
) -> (EF, EvaluationsList<EF>)
where
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    // Compute the quadratic sumcheck polynomial for the current variable.
    let sumcheck_poly = compute_sumcheck_polynomial(evals, weights, *sum);
    prover_state.add_extension_scalar(sumcheck_poly.evaluations()[0]);
    prover_state.add_extension_scalar(sumcheck_poly.evaluations()[2]);

    // Sample verifier challenge.
    let r: EF = prover_state.sample();

    prover_state.pow_grinding(pow_bits);

    // Compress polynomials and update the sum.
    let evals = join(|| compress(weights, r), || compress_ext(evals, r)).1;

    *sum = sumcheck_poly.evaluate_on_standard_domain(&MultilinearPoint::new(vec![r]));

    (r, evals)
}

/// Executes a standard, intermediate round of the sumcheck protocol.
///
/// This function executes a standard, intermediate round of the sumcheck protocol. Unlike the initial round,
/// it operates entirely within the extension field `EF`. It computes the sumcheck polynomial from the
/// current evaluations and weights, adds it to the transcript, gets a new challenge from the verifier,
/// and then compresses both the polynomial and weight evaluations in-place.
///
/// ## Arguments
/// * `prover_state` - A mutable reference to the `ProverState`, managing the Fiat-Shamir transcript.
/// * `evals` - A mutable reference to the polynomial's evaluations in `EF`, which will be compressed.
/// * `weights` - A mutable reference to the weight evaluations in `EF`, which will also be compressed.
/// * `sum` - A mutable reference to the claimed sum, updated after folding.
/// * `pow_bits` - The number of proof-of-work bits for grinding.
///
/// ## Returns
/// The verifier's challenge `r` as an `EF` element.
fn round<Challenger, F: Field, EF: ExtensionField<F>>(
    prover_state: &mut ProverState<F, EF, Challenger>,
    evals: &mut EvaluationsList<EF>,
    weights: &mut EvaluationsList<EF>,
    sum: &mut EF,
    pow_bits: usize,
) -> EF
where
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    // Compute the quadratic sumcheck polynomial for the current variable.
    let sumcheck_poly = compute_sumcheck_polynomial(evals, weights, *sum);
    prover_state.add_extension_scalar(sumcheck_poly.evaluations()[0]);
    prover_state.add_extension_scalar(sumcheck_poly.evaluations()[2]);

    // Sample verifier challenge.
    let r: EF = prover_state.sample();

    prover_state.pow_grinding(pow_bits);

    // Compress polynomials and update the sum.
    join(|| compress(evals, r), || compress(weights, r));

    *sum = sumcheck_poly.evaluate_on_standard_domain(&MultilinearPoint::new(vec![r]));

    r
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
#[instrument(skip_all, level = "debug")]
pub(crate) fn compute_sumcheck_polynomial<F: Field, EF: ExtensionField<F>>(
    evals: &EvaluationsList<F>,
    weights: &EvaluationsList<EF>,
    sum: EF,
) -> SumcheckPolynomial<EF> {
    assert!(evals.num_variables() >= 1);

    let (c0, c2) = evals
        .as_slice()
        .par_chunks_exact(2)
        .zip(weights.as_slice().par_chunks_exact(2))
        .map(sumcheck_quadratic::<F, EF>)
        .par_fold_reduce(
            || (EF::ZERO, EF::ZERO),
            |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
            |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
        );

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
    let c1 = sum - c0.double() - c2;

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
    pub(crate) evals: EvaluationsList<EF>,
    /// Evaluations of the equality polynomial used for enforcing constraints.
    pub(crate) weights: EvaluationsList<EF>,
    /// Accumulated sum incorporating equality constraints.
    pub(crate) sum: EF,
    /// Marker for phantom type parameter `F`.
    phantom: std::marker::PhantomData<F>,
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
        statement: &Statement<EF>,
        combination_randomness: EF,
    ) -> Self {
        let (weights, sum) = statement.combine::<F>(combination_randomness);

        Self {
            evals,
            weights,
            sum,
            phantom: std::marker::PhantomData,
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
        statement: &Statement<EF>,
        combination_randomness: EF,
        prover_state: &mut ProverState<F, EF, Challenger>,
        folding_factor: usize,
        pow_bits: usize,
    ) -> (Self, MultilinearPoint<EF>)
    where
        F: TwoAdicField,
        EF: TwoAdicField,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        assert_ne!(folding_factor, 0);
        let mut res = Vec::with_capacity(folding_factor);

        let (mut weights, mut sum) = statement.combine::<F>(combination_randomness);
        // In the first round base field evaluations are folded into extension field elements
        let (r, mut evals) = initial_round(prover_state, evals, &mut weights, &mut sum, pow_bits);
        res.push(r);

        // Apply rest of sumcheck rounds
        res.extend(
            (1..folding_factor)
                .map(|_| round(prover_state, &mut evals, &mut weights, &mut sum, pow_bits)),
        );

        // Reverse challenges to maintain order from X₀ to Xₙ.
        res.reverse();

        let sumcheck = Self {
            evals,
            weights,
            sum,
            phantom: std::marker::PhantomData,
        };

        (sumcheck, MultilinearPoint::new(res))
    }

    /// Constructs a new `SumcheckSingle` instance from evaluations in the base field.
    ///
    /// This function:
    /// - Uses precomputed evaluations of the polynomial `p` over the Boolean hypercube.
    /// - Applies the given constraint `Statement` using a random linear combination.
    /// - Initializes internal sumcheck state with weights and expected sum.
    /// - Applies first set of sumcheck rounds with univariate skip optimization.
    #[instrument(skip_all)]
    pub fn with_skip<Challenger>(
        evals: &EvaluationsList<F>,
        statement: &Statement<EF>,
        combination_randomness: EF,
        prover_state: &mut ProverState<F, EF, Challenger>,
        folding_factor: usize,
        pow_bits: usize,
        k_skip: usize,
    ) -> (Self, MultilinearPoint<EF>)
    where
        F: TwoAdicField,
        EF: TwoAdicField,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        assert_ne!(folding_factor, 0);
        let mut res = Vec::with_capacity(folding_factor);

        assert!(k_skip > 1);
        assert!(k_skip <= folding_factor);

        let (weights, _sum) = statement.combine::<F>(combination_randomness);

        // Compute the skipped-round polynomial h and the rectangular views f̂, ŵ.
        //
        // - `sumcheck_poly`: The univariate polynomial sent to the verifier for this round.
        // - `f_mat`, `w_mat`: The original evaluations reshaped into matrices of size 2^k x 2^(n-k).
        let (sumcheck_poly, f_mat, w_mat) =
            compute_skipping_sumcheck_polynomial(k_skip, evals, &weights);

        // Fiat–Shamir: commit to h by absorbing its M evaluations into the transcript.
        prover_state.add_extension_scalars(sumcheck_poly.evaluations());

        // Receive the verifier challenge for this entire collapsed round.
        let r: EF = prover_state.sample();
        res.push(r);

        // Proof-of-work challenge to delay prover.
        prover_state.pow_grinding(pow_bits);

        // Interpolate the LDE matrices at the folding randomness to get the new "folded" polynomial state.
        let new_p = interpolate_subgroup(&f_mat, r);
        let new_w = interpolate_subgroup(&w_mat, r);

        // Update polynomial and weights with reduced dimensionality.
        let mut evals = EvaluationsList::new(new_p);
        let mut weights = EvaluationsList::new(new_w);

        // Compute the new target sum after folding.
        let folded_poly_eval = interpolate_subgroup(
            &RowMajorMatrix::new_col(sumcheck_poly.evaluations().to_vec()),
            r,
        );
        let mut sum = folded_poly_eval[0];

        // Apply rest of sumcheck rounds
        res.extend(
            (k_skip..folding_factor)
                .map(|_| round(prover_state, &mut evals, &mut weights, &mut sum, pow_bits)),
        );

        // Reverse challenges to maintain order from X₀ to Xₙ.
        res.reverse();

        let sumcheck = Self {
            evals,
            weights,
            sum,
            phantom: std::marker::PhantomData,
        };

        (sumcheck, MultilinearPoint::new(res))
    }

    /// Returns the number of variables in the polynomial.
    pub const fn num_variables(&self) -> usize {
        self.evals.num_variables()
    }

    /// Adds new weighted constraints to the polynomial.
    ///
    /// This function updates the weight evaluations and sum by incorporating new constraints.
    ///
    /// Given points `z_i`, weights `ε_i`, and evaluation values `f(z_i)`, it updates:
    ///
    /// \begin{equation}
    ///     w(X) = w(X) + \sum ε_i \cdot w_{z_i}(X)
    /// \end{equation}
    ///
    /// and updates the sum as:
    ///
    /// \begin{equation}
    ///     S = S + \sum ε_i \cdot f(z_i)
    /// \end{equation}
    ///
    /// where `w_{z_i}(X)` represents the constraint encoding at point `z_i`.
    #[instrument(skip_all, fields(
        num_points = points.len(),
    ))]
    pub fn add_new_equality(
        &mut self,
        points: &[MultilinearPoint<EF>],
        evaluations: &[EF],
        combination_randomness: &[EF],
    ) {
        assert_eq!(combination_randomness.len(), points.len());
        assert_eq!(evaluations.len(), points.len());

        tracing::info_span!("accumulate_weight_buffer").in_scope(|| {
            points
                .iter()
                .zip(combination_randomness.iter())
                .for_each(|(point, &rand)| {
                    eval_eq::<_, _, true>(point.as_slice(), self.weights.as_mut_slice(), rand);
                });
        });

        // Accumulate the weighted sum
        self.sum += combination_randomness
            .iter()
            .zip(evaluations.iter())
            .map(|(&rand, &eval)| rand * eval)
            .sum::<EF>();
    }

    /// Adds new weighted constraints to the polynomial.
    ///
    /// Similar to `add_new_equality`, but specifically for constraints involving points
    /// in the base field.
    ///
    /// This function updates the weight evaluations and sum by incorporating new constraints.
    ///
    /// Given points `z_i`, weights `ε_i`, and evaluation values `f(z_i)`, it updates:
    ///
    /// \begin{equation}
    ///     w(X) = w(X) + \sum ε_i \cdot w_{z_i}(X)
    /// \end{equation}
    ///
    /// and updates the sum as:
    ///
    /// \begin{equation}
    ///     S = S + \sum ε_i \cdot f(z_i)
    /// \end{equation}
    ///
    /// where `w_{z_i}(X)` represents the constraint encoding at point `z_i`.
    #[instrument(skip_all, fields(
        num_points = points.len(),
    ))]
    pub fn add_new_base_equality(
        &mut self,
        points: &[MultilinearPoint<F>],
        evaluations: &[EF],
        combination_randomness: &[EF],
    ) {
        assert_eq!(combination_randomness.len(), points.len());
        assert_eq!(evaluations.len(), points.len());

        tracing::info_span!("accumulate_weight_buffer_base").in_scope(|| {
            points
                .iter()
                .zip(combination_randomness.iter())
                .for_each(|(point, &rand)| {
                    eval_eq_base::<_, _, true>(point.as_slice(), self.weights.as_mut_slice(), rand);
                });
        });

        // Accumulate the weighted sum
        self.sum += combination_randomness
            .iter()
            .zip(evaluations.iter())
            .map(|(&rand, &eval)| rand * eval)
            .sum::<EF>();
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
    /// - `prover_state`: The state of the prover, managing Fiat-Shamir transcript and PoW grinding.
    /// - `folding_factor`: Number of variables to fold in total.
    /// - `pow_bits`: Number of PoW bits used to delay the prover (0.0 to disable).
    /// - `k_skip`: Optional number of initial variables to skip using the univariate optimization.
    /// - `dft`: A two-adic FFT backend used for low-degree extensions over cosets.
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
        prover_state: &mut ProverState<F, EF, Challenger>,
        folding_factor: usize,
        pow_bits: usize,
    ) -> MultilinearPoint<EF>
    where
        F: TwoAdicField,
        EF: TwoAdicField,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Standard round-by-round folding
        // Proceed with one-variable-per-round folding for remaining variables.
        let mut res = (0..folding_factor)
            .map(|_| {
                round(
                    prover_state,
                    &mut self.evals,
                    &mut self.weights,
                    &mut self.sum,
                    pow_bits,
                )
            })
            .collect::<Vec<_>>();

        // Reverse challenges to maintain order from X₀ to Xₙ.
        res.reverse();

        // Return the full vector of verifier challenges as a multilinear point.
        MultilinearPoint::new(res)
    }
}
