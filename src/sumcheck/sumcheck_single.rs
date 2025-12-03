use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{Algebra, ExtensionField, Field, PackedFieldExtension, PackedValue, TwoAdicField};
use p3_interpolation::interpolate_subgroup;
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::{
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    sumcheck::sumcheck_single_skip::compute_skipping_sumcheck_polynomial,
    utils::{pack_slice, unpack_slice, unpack_slice_into},
    whir::{
        constraints::{Constraint, statement::EqStatement},
        proof::{SumcheckData, SumcheckSkipData},
    },
};

const PARALLEL_THRESHOLD: usize = 4096;
pub(crate) const PACK_THRESHOLD: usize = 4;

#[derive(Debug, Clone)]
pub(crate) enum Quad<F: Field, EF: ExtensionField<F>> {
    Packed {
        evals: EvaluationsList<EF::ExtensionPacking>,
        weights: EvaluationsList<EF::ExtensionPacking>,
    },
    Small {
        evals: EvaluationsList<EF>,
        weights: EvaluationsList<EF>,
    },
}

impl<F: Field, EF: ExtensionField<F>> Quad<F, EF> {
    pub(crate) fn new(evals: EvaluationsList<EF>, weights: EvaluationsList<EF>) -> Self {
        assert_eq!(evals.num_variables(), weights.num_variables());
        if evals.num_variables() > PACK_THRESHOLD {
            Self::new_packed(
                EvaluationsList::new(pack_slice(evals.as_slice())),
                EvaluationsList::new(pack_slice(weights.as_slice())),
            )
        } else {
            Self::new_small(evals, weights)
        }
    }

    const fn new_packed(
        evals: EvaluationsList<EF::ExtensionPacking>,
        weights: EvaluationsList<EF::ExtensionPacking>,
    ) -> Self {
        Self::Packed { evals, weights }
    }

    const fn new_small(evals: EvaluationsList<EF>, weights: EvaluationsList<EF>) -> Self {
        Self::Small { evals, weights }
    }

    fn k(&self) -> usize {
        let k_pack = log2_strict_usize(F::Packing::WIDTH);
        match self {
            Self::Packed { evals, weights } => {
                let k = evals.num_variables();
                assert_eq!(k, weights.num_variables());
                k + k_pack
            }
            Self::Small { evals, weights } => {
                let k = evals.num_variables();
                assert_eq!(k, weights.num_variables());
                k
            }
        }
    }

    fn eval(&self, point: &MultilinearPoint<EF>) -> EF {
        match self {
            Self::Packed { evals, .. } => evals.eval_hypercube_packed(point),
            Self::Small { evals, .. } => evals.evaluate_hypercube_ext(point),
        }
    }

    const fn num_evals(&self) -> usize {
        match self {
            Self::Packed { evals, .. } => evals.num_evals() * F::Packing::WIDTH,
            Self::Small { evals, .. } => evals.num_evals(),
        }
    }

    fn transition(&mut self) {
        let k = self.k();
        match self {
            Self::Packed { evals, weights } if k < PACK_THRESHOLD => {
                let evals = EvaluationsList::new(unpack_slice::<F, EF>(evals.as_slice()));
                let weights = EvaluationsList::new(unpack_slice::<F, EF>(weights.as_slice()));
                *self = Self::Small { evals, weights };
            }
            _ => {}
        }
    }

    fn round<Challenger>(
        &mut self,
        sumcheck_data: &mut SumcheckData<EF, F>,
        challenger: &mut Challenger,
        sum: &mut EF,
        pow_bits: usize,
    ) -> EF
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        let r = match self {
            Self::Packed { evals, weights } => {
                round_packed(sumcheck_data, challenger, evals, weights, sum, pow_bits)
            }
            Self::Small { evals, weights } => {
                round(sumcheck_data, challenger, evals, weights, sum, pow_bits)
            }
        };

        #[cfg(debug_assertions)]
        assert_eq!(*sum, self.prod());

        // If remaining number of vars are small, unpack polynomials
        self.transition();

        r
    }

    fn evals(&self) -> EvaluationsList<EF> {
        match self {
            Self::Packed { evals, .. } => {
                EvaluationsList::new(unpack_slice::<F, EF>(evals.as_slice()))
            }
            Self::Small { evals, .. } => evals.clone(),
        }
    }

    #[tracing::instrument(skip_all)]
    fn unpack_into(&self, unpacked: &mut [EF]) {
        match self {
            Self::Packed { evals, .. } => {
                assert_eq!(unpacked.len(), evals.num_evals() * F::Packing::WIDTH);
                unpack_slice_into::<F, EF>(unpacked, &evals.0);
            }
            Self::Small { evals, .. } => {
                unpacked.copy_from_slice(&evals.0);
            }
        }
    }

    fn combine(&mut self, sum: &mut EF, constraint: &Constraint<F, EF>) {
        match self {
            Self::Packed { weights, .. } => {
                constraint.combine_packed(weights, sum);
            }
            Self::Small { weights, .. } => {
                constraint.combine(weights, sum);
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn weights(&self) -> EvaluationsList<EF> {
        match &self {
            Self::Packed { weights, .. } => {
                EvaluationsList::new(unpack_slice::<F, EF>(weights.as_slice()))
            }
            Self::Small { weights, .. } => weights.clone(),
        }
    }

    #[cfg(debug_assertions)]
    fn eval_univariate(&self, var: EF) -> EF {
        use crate::poly::univariate::UnivariatePolynomial;
        let evals = UnivariatePolynomial::from_coefficients_vec(match self {
            Self::Packed { evals, .. } => unpack_slice(evals.as_slice()),
            Self::Small { evals, .. } => evals.0.clone(),
        });
        evals.evaluate(var)
    }

    pub(crate) fn prod(&self) -> EF {
        use p3_field::{PackedFieldExtension, dot_product};
        match self {
            Self::Packed { evals, weights } => {
                let sum_packed = dot_product(evals.iter().copied(), weights.iter().copied());
                EF::ExtensionPacking::to_ext_iter([sum_packed]).sum::<EF>()
            }
            Self::Small { evals, weights } => {
                dot_product(evals.iter().copied(), weights.iter().copied())
            }
        }
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
/// * `sumcheck_data`: A mutable reference to the sumcheck data structure for storing polynomial evaluations.
/// * `challenger`: A mutable reference to the Fiat-Shamir challenger for transcript management.
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
fn initial_round<Challenger, F: Field, EF: ExtensionField<F>>(
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
    // Compute the constant (c_0) and quadratic (c_2) coefficients of h(X).
    //
    // The polynomial h(X) = c_0 + c_1 * X + c_2 * X^2
    //
    // We store [c_0, c_2] and derive c_1 from the sum constraint: h(1) = claimed_sum - c_0
    let (c_0, c_2) = compute_sumcheck_coefficients(evals.as_slice(), weights.as_slice());
    sumcheck_data.polynomial_evaluations.push([c_0, c_2]);

    // Observe only c_0 and c_2 for Fiat-Shamir (c_1 is derived)
    challenger.observe_algebra_slice(&[c_0, c_2]);

    // Proof-of-work challenge to delay prover (only if pow_bits > 0)
    if pow_bits > 0 {
        sumcheck_data.push_pow_witness(challenger.grind(pow_bits));
    }

    // Sample verifier challenge.
    let r: EF = challenger.sample_algebra_element();

    // Compress polynomials and update the sum.
    let evals = join(|| weights.compress(r), || evals.compress_ext(r)).1;

    // Update sum: h(r) = c_0 + c_1 * r + c_2 * r^2 where c_1 = h(1) - c_0 - c_2
    // Since h(1) = claimed_sum - h(0) = claimed_sum - c_0, we have c_1 = claimed_sum - 2*c_0 - c_2
    // So h(r) = c_2 * r^2 + (claimed_sum - 2*c_0 - c_2) * r + c_0
    let h_1 = *sum - c_0;
    *sum = c_2 * r.square() + (h_1 - c_0 - c_2) * r + c_0;

    (r, evals)
}

#[instrument(skip_all)]
fn initial_round_packed<Challenger, F: Field, EF: ExtensionField<F>>(
    sumcheck_data: &mut SumcheckData<EF, F>,
    challenger: &mut Challenger,
    evals: &EvaluationsList<F>,
    weights: &mut EvaluationsList<EF::ExtensionPacking>,
    sum: &mut EF,
    pow_bits: usize,
) -> (EF, EvaluationsList<EF::ExtensionPacking>)
where
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    let evals_packed = F::Packing::pack_slice(evals.as_slice());
    // Compute the constant (c_0) and quadratic (c_2) coefficients of h(X).
    //
    // The polynomial h(X) = c_0 + c_1 * X + c_2 * X^2
    //
    // We store [c_0, c_2] and derive c_1 from the sum constraint: h(1) = claimed_sum - c_0
    let (c_0, c_2) = compute_sumcheck_coefficients(evals_packed, weights.as_slice());
    let c_0 = EF::ExtensionPacking::to_ext_iter([c_0]).sum::<EF>();
    let c_2 = EF::ExtensionPacking::to_ext_iter([c_2]).sum::<EF>();
    sumcheck_data.polynomial_evaluations.push([c_0, c_2]);

    // Observe only c_0 and c_2 for Fiat-Shamir (c_1 is derived)
    challenger.observe_algebra_element(c_0);
    challenger.observe_algebra_element(c_2);

    // Proof-of-work challenge to delay prover
    let witness = pow_grinding(challenger, pow_bits);

    // Store PoW witness if present
    sumcheck_data.push_pow_witness(witness);

    // Sample verifier challenge.
    let r: EF = challenger.sample_algebra_element();

    // Compress polynomials and update the sum.
    weights.compress(r);
    let evals = evals.compress_packed(r);

    // Update sum: h(r) = c_0 + c_1 * r + c_2 * r^2 where c_1 = h(1) - c_0 - c_2
    // Since h(1) = claimed_sum - h(0) = claimed_sum - c_0, we have c_1 = claimed_sum - 2*c_0 - c_2
    // So h(r) = c_2 * r^2 + (claimed_sum - 2*c_0 - c_2) * r + c_0
    let h_1 = *sum - c_0;
    *sum = c_2 * r.square() + (h_1 - c_0 - c_2) * r + c_0;

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
/// * `sumcheck_data` - A mutable reference to the sumcheck data structure for storing polynomial evaluations.
/// * `challenger` - A mutable reference to the Fiat-Shamir challenger for transcript management.
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
    evals: &mut EvaluationsList<EF>,
    weights: &mut EvaluationsList<EF>,
    sum: &mut EF,
    pow_bits: usize,
) -> EF
where
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    // Compute the constant (c_0) and quadratic (c_2) coefficients of h(X).
    //
    // The polynomial h(X) = c_0 + c_1 * X + c_2 * X^2
    //
    // We store [c_0, c_2] and derive c_1 from the sum constraint: h(1) = claimed_sum - c_0
    let (c_0, c_2) = compute_sumcheck_coefficients(evals.as_slice(), weights.as_slice());
    sumcheck_data.polynomial_evaluations.push([c_0, c_2]);

    // Observe only c_0 and c_2 for Fiat-Shamir (c_1 is derived)
    challenger.observe_algebra_slice(&[c_0, c_2]);

    // Proof-of-work challenge to delay prover (only if pow_bits > 0)
    if pow_bits > 0 {
        sumcheck_data.push_pow_witness(challenger.grind(pow_bits));
    }

    // Sample verifier challenge.
    let r: EF = challenger.sample_algebra_element();

    // Compress polynomials and update the sum.
    evals.compress(r);
    weights.compress(r);

    // Update sum: h(r) = c_0 + c_1 * r + c_2 * r^2 where c_1 = h(1) - c_0 - c_2
    //
    // Since h(1) = claimed_sum - h(0) = claimed_sum - c_0, we have c_1 = claimed_sum - 2*c_0 - c_2
    //
    // So h(r) = c_2 * r^2 + (claimed_sum - 2*c_0 - c_2) * r + c_0
    let h_1 = *sum - c_0;
    *sum = c_2 * r.square() + (h_1 - c_0 - c_2) * r + c_0;

    r
}

#[instrument(skip_all)]
fn round_packed<Challenger, F: Field, EF: ExtensionField<F>>(
    sumcheck_data: &mut SumcheckData<EF, F>,
    challenger: &mut Challenger,
    evals: &mut EvaluationsList<EF::ExtensionPacking>,
    weights: &mut EvaluationsList<EF::ExtensionPacking>,
    sum: &mut EF,
    pow_bits: usize,
) -> EF
where
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    // Compute the constant (c_0) and quadratic (c_2) coefficients of h(X).
    //
    // The polynomial h(X) = c_0 + c_1 * X + c_2 * X^2
    //
    // We store [c_0, c_2] and derive c_1 from the sum constraint: h(1) = claimed_sum - c_0
    let (c_0, c_2) = compute_sumcheck_coefficients(evals.as_slice(), weights.as_slice());
    let c_0 = EF::ExtensionPacking::to_ext_iter([c_0]).sum::<EF>();
    let c_2 = EF::ExtensionPacking::to_ext_iter([c_2]).sum::<EF>();
    sumcheck_data.polynomial_evaluations.push([c_0, c_2]);

    // Observe only c_0 and c_2 for Fiat-Shamir (c_1 is derived)
    challenger.observe_algebra_element(c_0);
    challenger.observe_algebra_element(c_2);

    // Proof-of-work challenge to delay prover
    let witness = pow_grinding(challenger, pow_bits);

    // Store PoW witness if present
    sumcheck_data.push_pow_witness(witness);

    // Sample verifier challenge.
    let r: EF = challenger.sample_algebra_element();

    // Compress polynomials and update the sum.
    evals.compress(r);
    weights.compress(r);

    // Update sum: h(r) = c_0 + c_1 * r + c_2 * r^2 where c_1 = h(1) - c_0 - c_2
    //
    // Since h(1) = claimed_sum - h(0) = claimed_sum - c_0, we have c_1 = claimed_sum - 2*c_0 - c_2
    //
    // So h(r) = c_2 * r^2 + (claimed_sum - 2*c_0 - c_2) * r + c_0
    let h_1 = *sum - c_0;
    *sum = c_2 * r.square() + (h_1 - c_0 - c_2) * r + c_0;

    r
}

/// Computes the constant and quadratic coefficients of the sumcheck polynomial.
///
/// For the quadratic polynomial h(X) = c_0 + c_1 * X + c_2 * X^2, this function
/// computes (c_0, c_2) directly:
/// - c_0 = h(0) = constant term = Σ e0 * p0
/// - c_2 = quadratic coefficient = Σ (e1 - e0) * (p1 - p0)
///
/// The linear coefficient c_1 is derived by the verifier using the sum constraint:
/// h(0) + h(1) = claimed_sum, so h(1) = claimed_sum - c_0.
#[instrument(skip_all, level = "debug")]
fn compute_sumcheck_coefficients<A, B>(evals: &[A], weights: &[B]) -> (B, B)
where
    A: Copy + Send + Sync + Algebra<A>,
    B: Copy + Send + Sync + Algebra<A> + Algebra<B>,
{
    assert!(log2_strict_usize(evals.len()) >= 1);
    assert_eq!(evals.len(), weights.len());

    let mid = evals.len() / 2;
    let (plo, phi) = evals.split_at(mid);
    let (elo, ehi) = weights.split_at(mid);

    let (t0, t1) = plo
        .par_iter()
        .zip(phi.par_iter())
        .zip(elo.par_iter().zip(ehi.par_iter()))
        .map(|((&p0, &p1), (&e0, &e1))| (e0 * p0, (e1 - e0) * (p1 - p0)))
        .par_fold_reduce(
            || (B::ZERO, B::ZERO),
            |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
            |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
        );
    (t0, t1)
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
pub struct SumcheckSingle<F: Field, EF: ExtensionField<F>> {
    /// Evaluations of the polynomial `p(X)` and evaluations of the equality polynomial used for enforcing constraints.
    pub(crate) quad: Quad<F, EF>,
    /// Accumulated sum incorporating equality constraints.
    pub(crate) sum: EF,
}

impl<F, EF> SumcheckSingle<F, EF>
where
    F: Field + Ord,
    EF: ExtensionField<F>,
{
    /// Constructs a new `SumcheckMingle` instance from evaluations in the extension field.
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
        if k > PACK_THRESHOLD {
            let (weights, sum) = constraint.combine_new_packed();
            Self {
                quad: Quad::<F, EF>::new_packed(evals.pack_ext(), weights),
                sum,
            }
        } else {
            let (weights, sum) = constraint.combine_new();
            Self {
                quad: Quad::<F, EF>::new_small(evals, weights),
                sum,
            }
        }
    }

    /// Constructs a new `SumcheckMingle` instance from evaluations in the base field.
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

        let (mut quad, r, mut sum) = if k > PACK_THRESHOLD {
            let (mut weights, mut sum) = constraint.combine_new_packed();
            let (r, evals) = initial_round_packed(
                sumcheck,
                challenger,
                evals,
                &mut weights,
                &mut sum,
                pow_bits,
            );
            (Quad::<F, EF>::new_packed(evals, weights), r, sum)
        } else {
            let (mut weights, mut sum) = constraint.combine_new();
            let (r, evals) = initial_round(
                sumcheck,
                challenger,
                evals,
                &mut weights,
                &mut sum,
                pow_bits,
            );
            (Quad::<F, EF>::new_small(evals, weights), r, sum)
        };

        #[cfg(debug_assertions)]
        assert_eq!(quad.prod(), sum);

        let rs = core::iter::once(r)
            .chain(
                (1..folding_factor).map(|_| quad.round(sumcheck, challenger, &mut sum, pow_bits)),
            )
            .collect();

        (Self { quad, sum }, MultilinearPoint::new(rs))
    }

    /// Constructs a new `SumcheckMingle` instance from evaluations in the base field.
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
        let mut quad = Quad::new(new_p, new_w);

        // While we could interpolate sumcheck_poly, it's cheaper and easier to just use
        // the new_p and new_w evaluations.
        let mut sum = quad.prod();

        // Apply rest of sumcheck rounds
        let rs = core::iter::once(r)
            .chain(
                (k_skip..folding_factor)
                    .map(|_| quad.round(&mut skip_data.sumcheck, challenger, &mut sum, pow_bits)),
            )
            .collect();

        (Self { quad, sum }, MultilinearPoint::new(rs))
    }

    /// Returns the number of variables in the polynomial.
    pub fn num_variables(&self) -> usize {
        self.quad.k()
    }

    /// Returns the polynomial.
    #[instrument(skip_all)]
    pub fn evals(&self) -> EvaluationsList<EF> {
        self.quad.evals()
    }

    #[cfg(test)]
    pub fn weights(&self) -> EvaluationsList<EF> {
        self.quad.weights()
    }

    /// Evaluates the sumcheck polynomial at a given multilinear point.
    pub fn eval(&self, point: &MultilinearPoint<EF>) -> EF {
        self.quad.eval(point)
    }

    /// Returns the number of evaluations in the polynomial.
    pub const fn num_evals(&self) -> usize {
        self.quad.num_evals()
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
            self.quad.combine(&mut self.sum, &constraint);
        }

        // Standard round-by-round folding
        // Proceed with one-variable-per-round folding for remaining variables.
        let res = (0..folding_factor)
            .map(|_| {
                let r = match &mut self.quad {
                    Quad::Packed { evals, weights } => round_packed(
                        sumcheck_data,
                        challenger,
                        evals,
                        weights,
                        &mut self.sum,
                        pow_bits,
                    ),
                    Quad::Small { evals, weights } => round(
                        sumcheck_data,
                        challenger,
                        evals,
                        weights,
                        &mut self.sum,
                        pow_bits,
                    ),
                };

                #[cfg(debug_assertions)]
                assert_eq!(self.sum, self.quad.prod());

                // If remaining number of vars are small, unpack polynomials
                self.quad.transition();

                r
            })
            .collect();

        // Return the full vector of verifier challenges as a multilinear point.
        MultilinearPoint::new(res)
    }
}
