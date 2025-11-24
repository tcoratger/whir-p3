use alloc::{vec, vec::Vec};
use core::{array, iter};

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_interpolation::interpolate_subgroup;
use p3_maybe_rayon::prelude::*;
use tracing::instrument;

use super::sumcheck_polynomial::SumcheckPolynomial;
use crate::{
    fiat_shamir::{grinding::pow_grinding, prover::ProverState},
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    sumcheck::sumcheck_single_skip::compute_skipping_sumcheck_polynomial,
    whir::{
        constraints::{Constraint, statement::EqStatement},
        proof::{InitialPhase, SumcheckData, SumcheckRoundData::Classic, WhirProof},
    },
};

const PARALLEL_THRESHOLD: usize = 4096;

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
#[instrument(skip_all)]
fn initial_round<Challenger, F: Field, EF: ExtensionField<F>, const DIGEST_ELEMS: usize>(
    prover_state: &mut ProverState<F, EF, Challenger>,
    sumcheck_data: &mut SumcheckData<EF, F>,
    challenger: &mut Challenger,
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
    let polynomial_evaluation: [EF; 3] = array::from_fn(|i| sumcheck_poly.evaluations()[i]);

    // Store polynomial evaluations in proof
    sumcheck_data
        .polynomial_evaluations
        .push(Classic(polynomial_evaluation));

    // Observe polynomial evaluations on BOTH challengers before grinding
    prover_state.add_extension_scalars(&polynomial_evaluation);

    // Observe polynomial evaluations for Fiat-Shamir
    let flattened = EF::flatten_to_base(polynomial_evaluation.to_vec());
    challenger.observe_slice(&flattened);

    // Proof-of-work challenge to delay prover
    prover_state.pow_grinding(pow_bits);
    let witness = pow_grinding(challenger, pow_bits);

    // Store PoW witness if present
    sumcheck_data.push_pow_witness(witness);

    // Sample verifier challenge.
    let r: EF = prover_state.sample();

    // Compress polynomials and update the sum.
    let evals = join(|| weights.compress(r), || evals.compress_ext(r)).1;

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
#[instrument(skip_all)]
fn round<Challenger, F: Field, EF: ExtensionField<F>>(
    sumcheck_data: &mut SumcheckData<EF, F>,
    challenger: &mut Challenger,
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
    let polynomial_evaluation: [EF; 3] = array::from_fn(|i| sumcheck_poly.evaluations()[i]);

    // Store polynomial evaluations in sumcheck data
    sumcheck_data
        .polynomial_evaluations
        .push(Classic(polynomial_evaluation));

    // Observe polynomial evaluations on BOTH challengers before grinding
    prover_state.add_extension_scalars(&polynomial_evaluation);

    // Observe polynomial evaluations for Fiat-Shamir
    let flattened = EF::flatten_to_base(polynomial_evaluation.to_vec());
    challenger.observe_slice(&flattened);

    // Proof-of-work challenge to delay prover
    prover_state.pow_grinding(pow_bits);
    let witness = pow_grinding(challenger, pow_bits);

    // Store PoW witness if present
    sumcheck_data.push_pow_witness(witness);

    // Sample verifier challenge.
    let r: EF = prover_state.sample();

    // Compress polynomials and update the sum.
    join(|| evals.compress(r), || weights.compress(r));

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

    let mid = evals.num_evals() / 2;
    let (plo, phi) = evals.0.split_at(mid);
    let (elo, ehi) = weights.0.split_at(mid);

    let (c0, c2) = plo
        .par_iter()
        .zip(phi.par_iter())
        .zip(elo.par_iter().zip(ehi.par_iter()))
        .map(|((&p0, &p1), (&e0, &e1))| (e0 * p0, (e1 - e0) * (p1 - p0)))
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

    SumcheckPolynomial::new(vec![eval_0, eval_1, eval_2])
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
    phantom: core::marker::PhantomData<F>,
}

impl<F, EF> SumcheckSingle<F, EF>
where
    F: Field + Ord,
    EF: ExtensionField<F>,
{
    pub const fn new(evals: EvaluationsList<EF>, weights: EvaluationsList<EF>, sum: EF) -> Self {
        Self {
            evals,
            weights,
            sum,
            phantom: core::marker::PhantomData,
        }
    }
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
        statement: &EqStatement<EF>,
        challenge: EF,
    ) -> Self {
        let mut weights = EvaluationsList::zero(statement.num_variables());
        let mut sum = EF::ZERO;
        statement.combine_hypercube::<F, false>(&mut weights, &mut sum, challenge);

        Self {
            evals,
            weights,
            sum,
            phantom: core::marker::PhantomData,
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
    pub fn from_base_evals<Challenger, const DIGEST_ELEMS: usize>(
        evals: &EvaluationsList<F>,
        prover_state: &mut ProverState<F, EF, Challenger>,
        proof: &mut WhirProof<F, EF, DIGEST_ELEMS>,
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

        let InitialPhase::WithStatement { ref mut sumcheck } = proof.initial_phase else {
            panic!("initial_round called with incorrect InitialPhase variant");
        };

        let (mut weights, mut sum) = constraint.combine_new();

        let (first_round, mut evals) = initial_round::<Challenger, F, EF, DIGEST_ELEMS>(
            prover_state,
            sumcheck,
            challenger,
            evals,
            &mut weights,
            &mut sum,
            pow_bits,
        );

        let subsequent_rounds = (1..folding_factor).map(|_| {
            round::<Challenger, F, EF>(
                sumcheck,
                challenger,
                prover_state,
                &mut evals,
                &mut weights,
                &mut sum,
                pow_bits,
            )
        });

        let res = iter::once(first_round)
            .chain(subsequent_rounds)
            .collect::<Vec<_>>();

        let sumcheck = Self {
            evals,
            weights,
            sum,
            phantom: core::marker::PhantomData,
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
    #[allow(clippy::too_many_arguments)]
    pub fn with_skip<Challenger, const DIGEST_ELEMS: usize>(
        evals: &EvaluationsList<F>,
        prover_state: &mut ProverState<F, EF, Challenger>,
        proof: &mut WhirProof<F, EF, DIGEST_ELEMS>,
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
        prover_state.add_extension_scalars(sumcheck_poly.evaluations());
        let flattened = EF::flatten_to_base(polynomial_skip_evaluation.to_vec());
        challenger.observe_slice(&flattened);

        // Update the WhirProof structure
        let InitialPhase::WithStatementSkip {
            ref mut skip_evaluations,
            ref mut skip_pow,
            ref mut sumcheck,
        } = proof.initial_phase
        else {
            panic!("initial_round called with incorrect InitialPhase variant");
        };

        skip_evaluations.extend_from_slice(polynomial_skip_evaluation);

        // Proof-of-work challenge to delay prover.
        prover_state.pow_grinding(pow_bits);
        *skip_pow = pow_grinding(challenger, pow_bits);

        // Receive the verifier challenge for this entire collapsed round.
        let r: EF = prover_state.sample();
        let r_rf: EF = challenger.sample_algebra_element();
        assert_eq!(r, r_rf);

        // Interpolate the LDE matrices at the folding randomness to get the new "folded" polynomial state.
        let new_p = interpolate_subgroup(&f_mat, r);
        let new_w = interpolate_subgroup(&w_mat, r);

        // While we could interpolate sumcheck_poly, it's cheaper and easier to just use
        // the new_p and new_w evaluations.
        let mut sum = new_p
            .iter()
            .zip(new_w.iter())
            .map(|(&p, &w)| p * w)
            .sum::<EF>();

        // Update polynomial and weights with reduced dimensionality.
        let mut evals = EvaluationsList::new(new_p);
        let mut weights = EvaluationsList::new(new_w);

        // Apply rest of sumcheck rounds
        let res = iter::once(r)
            .chain((k_skip..folding_factor).map(|_| {
                round(
                    sumcheck,
                    challenger,
                    prover_state,
                    &mut evals,
                    &mut weights,
                    &mut sum,
                    pow_bits,
                )
            }))
            .collect::<Vec<_>>();

        let sumcheck = Self {
            evals,
            weights,
            sum,
            phantom: core::marker::PhantomData,
        };

        (sumcheck, MultilinearPoint::new(res))
    }

    /// Returns the number of variables in the polynomial.
    pub const fn num_variables(&self) -> usize {
        self.evals.num_variables()
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
    #[allow(clippy::too_many_arguments)]
    pub fn compute_sumcheck_polynomials<Challenger, const DIGEST_ELEMS: usize>(
        &mut self,
        prover_state: &mut ProverState<F, EF, Challenger>,
        proof: &mut WhirProof<F, EF, DIGEST_ELEMS>,
        challenger: &mut Challenger,
        folding_factor: usize,
        pow_bits: usize,
        is_final_round: bool,
        constraint: Option<Constraint<F, EF>>,
    ) -> MultilinearPoint<EF>
    where
        F: TwoAdicField,
        EF: TwoAdicField,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        if let Some(constraint) = constraint {
            constraint.combine(&mut self.weights, &mut self.sum);
        }

        let mut sumcheck_data: SumcheckData<EF, F> = SumcheckData::default();
        // Standard round-by-round folding
        // Proceed with one-variable-per-round folding for remaining variables.
        let res = (0..folding_factor)
            .map(|_| {
                round(
                    &mut sumcheck_data,
                    challenger,
                    prover_state,
                    &mut self.evals,
                    &mut self.weights,
                    &mut self.sum,
                    pow_bits,
                )
            })
            .collect();

        // Store sumcheck data in the appropriate location
        if is_final_round {
            // Final round: write sumcheck data to proof.final_sumcheck
            proof.final_sumcheck = Some(sumcheck_data);
        } else {
            // Regular round: update the last round in proof.rounds with sumcheck data
            if let Some(last_round) = proof.rounds.last_mut() {
                last_round.sumcheck = sumcheck_data;
            } else {
                panic!("no rounds exist yet!");
            }
        }

        // Return the full vector of verifier challenges as a multilinear point.
        MultilinearPoint::new(res)
    }
}
