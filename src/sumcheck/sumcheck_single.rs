use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_interpolation::interpolate_subgroup;
use p3_matrix::dense::RowMajorMatrix;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use tracing::instrument;

use super::sumcheck_polynomial::SumcheckPolynomial;
use crate::{
    fiat_shamir::prover::ProverState,
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    sumcheck::{
        sumcheck_single_skip::compute_skipping_sumcheck_polynomial, utils::sumcheck_quadratic,
    },
    whir::statement::Statement,
};

#[cfg(feature = "parallel")]
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
    #[cfg(feature = "parallel")]
    let folded = if evals.evals().len() >= PARALLEL_THRESHOLD {
        evals.evals().par_chunks_exact(2).map(fold).collect()
    } else {
        evals.evals().chunks_exact(2).map(fold).collect()
    };

    #[cfg(not(feature = "parallel"))]
    let folded = evals.evals().chunks_exact(2).map(fold).collect();
    EvaluationsList::new(folded)
}

/// Compresses a list of evaluations in-place.
///
/// This function performs an in-place compression of a polynomial's evaluations. It takes a mutable
/// list of evaluations and a random challenge `r` from the same field, and updates the list with
/// the compressed evaluations. This is the step in the sumcheck protocol where we reduce the polynomial's
/// variable count by one in each round.
///
/// ## Arguments
/// * `evals`: A mutable reference to an `EvaluationsList<F>`, which will be updated with the
///   compressed evaluations.
/// * `r`: A value from the field `F`, used as the random folding challenge.
///
/// This function modifies `evals` in-place, halving the number of evaluations and thus reducing the
/// polynomial's variable count.
///
/// The compression formula is the same as for `compress_ext`:
/// $p'(X_2, ..., X_n) = (p(1, X_2, ..., X_n) - p(0, X_2, ..., X_n)) \cdot r + p(0, X_2, ..., X_n)$
#[instrument(skip_all)]
pub fn compress<F: Field>(evals: &mut EvaluationsList<F>, r: F) {
    assert_ne!(evals.num_variables(), 0);

    // Fold between base and extension field elements
    let fold = |slice: &[F]| -> F { r * (slice[1] - slice[0]) + slice[0] };

    // Threshold below which sequential computation is faster
    //
    // This was chosen based on experiments with the `compress` function.
    // It is possible that the threshold can be tuned further.
    #[cfg(feature = "parallel")]
    let folded = if evals.evals().len() >= PARALLEL_THRESHOLD {
        evals.evals().par_chunks_exact(2).map(fold).collect()
    } else {
        evals.evals().chunks_exact(2).map(fold).collect()
    };

    #[cfg(not(feature = "parallel"))]
    let folded = evals.evals().chunks_exact(2).map(fold).collect();

    *evals = EvaluationsList::new(folded);
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
    prover_state.add_extension_scalars(sumcheck_poly.evaluations());

    // Sample verifier challenge.
    let r: EF = prover_state.sample();

    prover_state.pow_grinding(pow_bits);

    // Compress polynomials and update the sum.

    let (evals, ()) = rayon::join(|| compress_ext(evals, r), || compress(weights, r));
    *sum = sumcheck_poly.evaluate_at_point(&r.into());

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
    prover_state.add_extension_scalars(sumcheck_poly.evaluations());

    // Sample verifier challenge.
    let r: EF = prover_state.sample();

    prover_state.pow_grinding(pow_bits);

    // Compress polynomials and update the sum.
    rayon::join(|| compress(evals, r), || compress(weights, r));
    *sum = sumcheck_poly.evaluate_at_point(&r.into());
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

    #[cfg(feature = "parallel")]
    let (c0, c2) = evals
        .evals()
        .par_chunks_exact(2)
        .zip(weights.evals().par_chunks_exact(2))
        .map(sumcheck_quadratic::<F, EF>)
        .reduce(
            || (EF::ZERO, EF::ZERO),
            |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
        );

    #[cfg(not(feature = "parallel"))]
    let (c0, c2) = evals
        .evals()
        .chunks_exact(2)
        .zip(weights.evals().chunks_exact(2))
        .map(sumcheck_quadratic::<F, EF>)
        .fold((EF::ZERO, EF::ZERO), |(a0, a2), (b0, b2)| {
            (a0 + b0, a2 + b2)
        });

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
    F: Field,
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
                .map(|_| round(prover_state, &mut evals, &mut weights, &mut sum, pow_bits))
                .collect::<Vec<_>>(),
        );

        // Reverse challenges to maintain order from X₀ to Xₙ.
        res.reverse();

        let sumcheck = Self {
            evals,
            weights,
            sum,
            phantom: std::marker::PhantomData,
        };

        (sumcheck, MultilinearPoint(res))
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
        // Collapse the first k variables via a univariate evaluation over a multiplicative coset.
        let (sumcheck_poly, f_mat, w_mat) =
            compute_skipping_sumcheck_polynomial(k_skip, evals, &weights);

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
                .map(|_| round(prover_state, &mut evals, &mut weights, &mut sum, pow_bits))
                .collect::<Vec<_>>(),
        );

        // Reverse challenges to maintain order from X₀ to Xₙ.
        res.reverse();

        let sumcheck = Self {
            evals,
            weights,
            sum,
            phantom: std::marker::PhantomData,
        };

        (sumcheck, MultilinearPoint(res))
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

        #[cfg(feature = "parallel")]
        {
            use tracing::info_span;

            // Parallel update of weight buffer
            info_span!("accumulate_weight_buffer").in_scope(|| {
                points
                    .iter()
                    .zip(combination_randomness.iter())
                    .for_each(|(point, &rand)| {
                        crate::utils::eval_eq::<_, _, true>(
                            &point.0,
                            self.weights.evals_mut(),
                            rand,
                        );
                    });
            });

            // Accumulate the weighted sum (cheap, done sequentially)
            self.sum += combination_randomness
                .iter()
                .zip(evaluations.iter())
                .map(|(&rand, &eval)| rand * eval)
                .sum::<EF>();
        }

        #[cfg(not(feature = "parallel"))]
        {
            // Accumulate the sum while applying all constraints simultaneously
            points
                .iter()
                .zip(combination_randomness.iter().zip(evaluations.iter()))
                .for_each(|(point, (&rand, &eval))| {
                    crate::utils::eval_eq::<F, EF, true>(&point.0, self.weights.evals_mut(), rand);
                    self.sum += rand * eval;
                });
        }
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
        MultilinearPoint(res)
    }
}

#[cfg(test)]
mod tests {

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::{Rng, SeedableRng, distr::StandardUniform, rngs::SmallRng};

    use super::*;
    use crate::{
        fiat_shamir::{domain_separator::DomainSeparator, verifier::VerifierState},
        poly::multilinear::MultilinearPoint,
        sumcheck::K_SKIP_SUMCHECK,
        whir::{
            statement::{constraint::Constraint, weights::Weights},
            verifier::sumcheck::verify_sumcheck_rounds,
        },
    };

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;
    type Perm = Poseidon2BabyBear<16>;

    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

    /// Creates a fresh domain separator and challenger with fixed RNG seed.
    fn domainsep_and_challenger() -> (DomainSeparator<EF, F>, MyChallenger) {
        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);
        let challenger = MyChallenger::new(perm);
        (DomainSeparator::new(vec![]), challenger)
    }

    fn prover() -> ProverState<F, EF, MyChallenger> {
        let (domsep, challenger) = domainsep_and_challenger();
        domsep.to_prover_state(challenger)
    }

    fn verifier(proof: Vec<F>) -> VerifierState<F, EF, MyChallenger> {
        let (domsep, challenger) = domainsep_and_challenger();
        domsep.to_verifier_state(proof, challenger)
    }

    fn rand_vec<F>(mut rng: impl Rng, n: usize) -> Vec<F>
    where
        StandardUniform: rand::distr::Distribution<F>,
    {
        (0..n).map(|_| rng.random()).collect()
    }

    fn rand_point<F>(rng: impl Rng, k: usize) -> MultilinearPoint<F>
    where
        StandardUniform: rand::distr::Distribution<F>,
    {
        MultilinearPoint(rand_vec(rng, k))
    }

    fn make_initial_statement<Challenger>(
        prover: &mut ProverState<F, EF, Challenger>,
        num_vars: usize,
        num_points: usize,
        poly: &EvaluationsList<F>,
    ) -> Statement<EF>
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        let (points, evals): (Vec<_>, Vec<_>) = (0..num_points)
            .map(|_| {
                let point = prover.sample();
                let point = MultilinearPoint::expand_from_univariate(point, num_vars);
                let eval = poly.evaluate(&point);
                prover.add_extension_scalar(eval);
                (point, eval)
            })
            .unzip();

        let mut statement = Statement::new(num_vars);
        points.iter().zip(evals.iter()).for_each(|(point, &eval)| {
            statement.add_constraint(Weights::evaluation(point.clone()), eval);
        });
        statement
    }

    fn make_inter_statement<Challenger>(
        prover: &mut ProverState<F, EF, Challenger>,
        num_points: usize,
        sumcheck: &mut SumcheckSingle<F, EF>,
    ) -> (Statement<EF>, EF)
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        let num_vars = sumcheck.num_variables();

        let mut statement = Statement::new(num_vars);
        let (points, evals): (Vec<_>, Vec<_>) = (0..num_points)
            .map(|_| {
                let point = prover.sample();
                let point = MultilinearPoint::expand_from_univariate(point, num_vars);
                let eval = sumcheck.evals.evaluate(&point);
                prover.add_extension_scalar(eval);
                statement.add_constraint(Weights::evaluation(point.clone()), eval);
                (point, eval)
            })
            .unzip();

        // Draw combination randomness `alpha` and add new equality constraints
        let alpha: EF = prover.sample();
        sumcheck.add_new_equality(&points, &evals, &alpha.powers().take(num_points).collect());

        (statement, alpha)
    }

    fn read_statement<Challenger>(
        verifier: &mut VerifierState<F, EF, Challenger>,
        num_vars: usize,
        num_points: usize,
    ) -> Statement<EF>
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        let mut statement = Statement::new(num_vars);
        (0..num_points).for_each(|_| {
            let point = verifier.sample();
            let point = MultilinearPoint::expand_from_univariate(point, num_vars);
            let eval = verifier.next_extension_scalar().unwrap();
            statement.add_constraint(Weights::evaluation(point), eval);
        });
        statement
    }

    impl<F> MultilinearPoint<F>
    where
        F: Field,
    {
        fn extend(&mut self, rest: &Self) {
            self.0 = rest
                .0
                .iter()
                .chain(self.0.iter())
                .copied()
                .collect::<Vec<_>>();
        }
    }

    fn combine_constraints<EF: Field>(
        claimed_sum: &mut EF,
        constraints: &[Constraint<EF>],
        alpha: EF,
    ) -> Vec<EF> {
        let alpha: Vec<_> = alpha.powers().take(constraints.len()).collect();
        *claimed_sum += constraints
            .iter()
            .zip(&alpha)
            .map(|(c, &rand)| rand * c.sum)
            .sum::<EF>();
        alpha
    }

    fn eval_constraints_poly<EF: Field>(
        mut num_variables: usize,
        folding_factor: &[usize],
        constraints: &[Vec<Constraint<EF>>],
        alphas: &[EF],
        mut point: MultilinearPoint<EF>,
    ) -> EF {
        let mut value = EF::ZERO;
        assert_eq!(alphas.len(), constraints.len());

        for (round, (alphas, constraints)) in alphas.iter().zip(constraints.iter()).enumerate() {
            let alphas: Vec<_> = alphas.powers().take(constraints.len()).collect();
            assert_eq!(alphas.len(), constraints.len());
            if round > 0 {
                num_variables -= folding_factor[round - 1];
                point = MultilinearPoint(point.0[..num_variables].to_vec());
            }
            value += constraints
                .iter()
                .zip(alphas)
                .map(|(constraint, alpha)| alpha * constraint.weights.compute(&point))
                .sum::<EF>();
        }
        value
    }

    fn run_sumcheck_test(folding_factors: &[usize], num_points: &[usize]) {
        assert_eq!(folding_factors.len(), num_points.len() + 1);
        let num_vars = folding_factors.iter().sum::<usize>();

        let poly = {
            let mut rng = SmallRng::seed_from_u64(1);
            EvaluationsList::new(
                (0..1 << num_vars)
                    .map(|_| rng.random::<F>())
                    .collect::<Vec<_>>(),
            )
        };

        let proof = {
            let prover = &mut prover();
            let statement = make_initial_statement(prover, num_vars, num_points[0], &poly);
            let alpha: EF = prover.sample();

            let folding = folding_factors[0];

            // Run first set of rounds
            let (mut sumcheck, mut r) =
                SumcheckSingle::from_base_evals(&poly, &statement, alpha, prover, folding, 0);
            let mut num_vars_inter = num_vars - folding;

            // Run intermediate rounds
            // With intermediate statements
            for (&folding, &num_points) in folding_factors
                .iter()
                .skip(1)
                .zip(num_points.iter().skip(1))
            {
                make_inter_statement(prover, num_points, &mut sumcheck);

                r.extend(&sumcheck.compute_sumcheck_polynomials(prover, folding_factors[1], 0));
                num_vars_inter -= folding;

                assert_eq!(sumcheck.evals.num_variables(), num_vars_inter);
                assert_eq!(sumcheck.evals.num_evals(), 1 << num_vars_inter);
            }

            // Run final rounds
            // No new statements
            let final_rounds = *folding_factors.last().unwrap();
            r.extend(&sumcheck.compute_sumcheck_polynomials(prover, final_rounds, 0));
            assert_eq!(num_vars_inter, final_rounds);
            assert_eq!(sumcheck.evals.num_variables(), 0);
            assert_eq!(sumcheck.evals.num_evals(), 1);

            // Final constant must be the evalaution of input poly F(x) at r
            let constant = sumcheck.evals[0];
            assert_eq!(poly.evaluate(&r), constant);
            prover.add_extension_scalar(constant);

            // Return proof data
            prover.proof_data().to_vec()
        };

        {
            let verifier = &mut verifier(proof);
            let mut sum = EF::ZERO;
            let mut r = MultilinearPoint(vec![]);
            let mut alphas = vec![];
            let mut constraints = vec![];
            let mut num_vars_inter = num_vars;

            for (&folding, &num_points) in folding_factors.iter().zip(num_points.iter()) {
                let statement = read_statement(verifier, num_vars_inter, num_points);
                alphas.push(verifier.sample());
                constraints.push(statement.constraints.clone());
                combine_constraints(&mut sum, &statement.constraints, *alphas.last().unwrap());
                r.extend(&verify_sumcheck_rounds(verifier, &mut sum, folding, 0, false).unwrap());
                num_vars_inter -= folding;
            }

            let final_rounds = *folding_factors.last().unwrap();
            r.extend(&verify_sumcheck_rounds(verifier, &mut sum, final_rounds, 0, false).unwrap());
            let constant = verifier.next_extension_scalar().unwrap();

            // Check if `sum == f(r) * eq(z, r)`
            let eq_eval =
                eval_constraints_poly(num_vars, folding_factors, &constraints, &alphas, r);
            assert_eq!(sum, constant * eq_eval);
        }
    }

    fn run_sumcheck_test_skips(folding_factors: &[usize], num_points: &[usize]) {
        assert_eq!(folding_factors.len(), num_points.len() + 1);
        let num_vars = folding_factors.iter().sum::<usize>();

        let poly = {
            let mut rng = SmallRng::seed_from_u64(1);
            EvaluationsList::new(
                (0..1 << num_vars)
                    .map(|_| rng.random::<F>())
                    .collect::<Vec<_>>(),
            )
        };

        let proof = {
            let prover = &mut prover();
            let statement = make_initial_statement(prover, num_vars, num_points[0], &poly);
            let alpha: EF = prover.sample();

            let folding = folding_factors[0];

            // Run first set of rounds
            let (mut sumcheck, mut r) = SumcheckSingle::with_skip(
                &poly,
                &statement,
                alpha,
                prover,
                folding,
                0,
                K_SKIP_SUMCHECK,
            );
            let mut num_vars_inter = num_vars - folding;

            // Run intermediate rounds
            // With intermediate statements
            for (&folding, &num_points) in folding_factors
                .iter()
                .skip(1)
                .zip(num_points.iter().skip(1))
            {
                make_inter_statement(prover, num_points, &mut sumcheck);

                r.extend(&sumcheck.compute_sumcheck_polynomials(prover, folding_factors[1], 0));
                num_vars_inter -= folding;

                assert_eq!(sumcheck.evals.num_variables(), num_vars_inter);
                assert_eq!(sumcheck.evals.num_evals(), 1 << num_vars_inter);
            }

            // Run final rounds
            // No new statements
            let final_rounds = *folding_factors.last().unwrap();
            r.extend(&sumcheck.compute_sumcheck_polynomials(prover, final_rounds, 0));
            assert_eq!(num_vars_inter, final_rounds);
            assert_eq!(sumcheck.evals.num_variables(), 0);
            assert_eq!(sumcheck.evals.num_evals(), 1);

            // Final constant must be the evalaution of input poly F(x) at r
            let constant = sumcheck.evals[0];
            assert_eq!(poly.evaluate(&r), constant);
            prover.add_extension_scalar(constant);

            // Return proof data
            prover.proof_data().to_vec()
        };

        {
            let verifier = &mut verifier(proof);
            let mut sum = EF::ZERO;
            let mut r = MultilinearPoint(vec![]);
            let mut alphas = vec![];
            let mut constraints = vec![];
            let mut num_vars_inter = num_vars;

            for (&folding, &num_points) in folding_factors.iter().zip(num_points.iter()) {
                let statement = read_statement(verifier, num_vars_inter, num_points);
                alphas.push(verifier.sample());
                constraints.push(statement.constraints.clone());
                combine_constraints(&mut sum, &statement.constraints, *alphas.last().unwrap());
                r.extend(&verify_sumcheck_rounds(verifier, &mut sum, folding, 0, true).unwrap());
                num_vars_inter -= folding;
            }

            let final_rounds = *folding_factors.last().unwrap();
            r.extend(&verify_sumcheck_rounds(verifier, &mut sum, final_rounds, 0, true).unwrap());
            let constant = verifier.next_extension_scalar().unwrap();

            // Check if `sum == f(r) * eq(z, r)`
            let eq_eval =
                eval_constraints_poly(num_vars, folding_factors, &constraints, &alphas, r);
            assert_eq!(sum, constant * eq_eval);
        }
    }

    #[test]
    fn test_sumcheck_prover() {
        run_sumcheck_test(&[1, 0], &[1]);
        run_sumcheck_test(&[1, 1], &[1]);
        run_sumcheck_test(&[1, 1], &[9]);
        run_sumcheck_test(&[4, 1], &[9]);
        run_sumcheck_test(&[1, 4], &[9]);
        run_sumcheck_test(&[1, 1, 1], &[1, 1]);
        run_sumcheck_test(&[4, 1, 4], &[9, 9]);
        run_sumcheck_test(&[4, 4, 1], &[9, 9]);
        run_sumcheck_test(&[1, 4, 4], &[9, 9]);

        // run_sumcheck_test_skips(&[6, 1], &[1]);
    }
}
