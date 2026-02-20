use alloc::vec::Vec;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field, PackedFieldExtension, PackedValue, dot_product};
use p3_util::log2_strict_usize;

use crate::{
    poly::{evals::EvaluationsList as Poly, multilinear::MultilinearPoint as Point},
    sumcheck::{
        extrapolate_012, lagrange::lagrange_weights_012_multi,
        product_polynomial::ProductPolynomial, svo::SplitEq,
    },
    whir::{
        constraints::{
            Constraint,
            statement::{
                EqStatement,
                initial::{InitialStatement, InitialStatementInner},
            },
        },
        proof::SumcheckData,
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
pub struct Sumcheck<F: Field, EF: ExtensionField<F>> {
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

impl<F, EF> Sumcheck<F, EF>
where
    F: Field + Ord,
    EF: ExtensionField<F>,
{
    #[tracing::instrument(skip_all)]
    fn new_classic_small<Challenger>(
        poly: &Poly<F>,
        sumcheck_data: &mut SumcheckData<F, EF>,
        challenger: &mut Challenger,
        folding_factor: usize,
        pow_bits: usize,
        statement: &EqStatement<EF>,
    ) -> (Self, Point<EF>)
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        let alpha: EF = challenger.sample_algebra_element();
        let k = poly.num_variables();
        // Initialize fresh accumulators for the weight polynomial and expected evaluation.
        // The weight polynomial needs 2^k entries for the full Boolean hypercube.
        let mut weights = Poly::zero(k);
        let mut sum = EF::ZERO;

        // Combine equality constraints without accumulation (INITIALIZED=false).
        // This directly writes the equality portion of W(X) to `weights`.
        statement.combine_hypercube::<F, false>(&mut weights, &mut sum, alpha);

        // Compute the constant (c₀) and quadratic (c₂) coefficients of h(X).
        let (c0, c2) = poly.sumcheck_coefficients(&weights);

        // Commit to transcript, perform PoW, and receive challenge.
        let r = sumcheck_data.observe_and_sample(challenger, c0, c2, pow_bits);

        // Fold both polynomials and update the sum.
        weights.compress(r);
        let evals = poly.compress_ext(r);
        sum = extrapolate_012(c0, sum - c0, c2, r);

        let mut poly = ProductPolynomial::<F, EF>::new_small(evals, weights);
        debug_assert_eq!(poly.dot_product(), sum);

        let rs = core::iter::once(r)
            .chain(
                (1..folding_factor)
                    .map(|_| poly.round(sumcheck_data, challenger, &mut sum, pow_bits)),
            )
            .collect();

        (Self { poly, sum }, Point::new(rs))
    }

    #[tracing::instrument(skip_all)]
    fn new_classic_packed<Challenger>(
        poly: &Poly<F>,
        sumcheck_data: &mut SumcheckData<F, EF>,
        challenger: &mut Challenger,
        folding_factor: usize,
        pow_bits: usize,
        statement: &EqStatement<EF>,
    ) -> (Self, Point<EF>)
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        let alpha: EF = challenger.sample_algebra_element();
        let k = poly.num_variables();
        let k_pack = log2_strict_usize(F::Packing::WIDTH);
        // Initialize fresh accumulators for the weight polynomial and expected evaluation.
        // The weight polynomial needs 2^(k - k_pack) packed entries.
        let mut weights = Poly::zero(k - k_pack);
        let mut sum = EF::ZERO;

        // Combine equality constraints without accumulation (INITIALIZED=false).
        // This directly writes the equality portion of W(X) to `weights`.
        statement.combine_hypercube_packed::<F, false>(&mut weights, &mut sum, alpha);

        let poly_packed = Poly::new(F::Packing::pack_slice(poly.as_slice()).to_vec());
        // Compute the constant (c₀) and quadratic (c₂) coefficients of h(X).
        let (c0, c2) = poly_packed.sumcheck_coefficients(&weights);
        // Reduce packed results to scalar via horizontal sum.
        let c0 = EF::ExtensionPacking::to_ext_iter([c0]).sum();
        let c2 = EF::ExtensionPacking::to_ext_iter([c2]).sum();

        // Commit to transcript, perform PoW, and receive challenge.
        let r = sumcheck_data.observe_and_sample(challenger, c0, c2, pow_bits);

        // Fold both polynomials and update the sum.
        weights.compress(r);
        let evals = poly.compress_into_packed(r);
        sum = extrapolate_012(c0, sum - c0, c2, r);

        let mut poly = ProductPolynomial::<F, EF>::new_packed(evals, weights);
        debug_assert_eq!(poly.dot_product(), sum);

        let rs = core::iter::once(r)
            .chain(
                (1..folding_factor)
                    .map(|_| poly.round(sumcheck_data, challenger, &mut sum, pow_bits)),
            )
            .collect();

        (Self { poly, sum }, Point::new(rs))
    }

    #[tracing::instrument(skip_all)]
    pub(super) fn new_svo<Challenger>(
        poly: &Poly<F>,
        sumcheck_data: &mut SumcheckData<F, EF>,
        challenger: &mut Challenger,
        folding_factor: usize,
        pow_bits: usize,
        statements: &[SplitEq<F, EF>],
    ) -> (Self, Point<EF>)
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        assert_ne!(folding_factor, 0);
        let alpha: EF = challenger.sample_algebra_element();

        let k = poly.num_variables();
        assert!(
            folding_factor <= k,
            "number of rounds must be less than or equal to instance size"
        );
        for statement in statements {
            assert_eq!(statement.num_variables(), k);
        }
        let k_pack = log2_strict_usize(F::Packing::WIDTH);
        assert!(k >= 2 * k_pack + folding_factor);

        let mut sum = statements
            .iter()
            .zip(alpha.powers())
            .map(|(statement, alpha)| statement.eval * alpha)
            .sum::<EF>();

        let accumulators = statements
            .iter()
            .map(SplitEq::accumulators)
            .collect::<Vec<_>>();

        let mut rs = Vec::with_capacity(folding_factor);
        tracing::info_span!("svo rounds").in_scope(|| {
            for round_idx in 0..folding_factor {
                let (mut c0, mut c2): (EF, EF) = Default::default();
                let weights = lagrange_weights_012_multi(rs.as_slice());
                for (accumulators, alpha) in accumulators.iter().zip(alpha.powers()) {
                    let acc0 = &accumulators[round_idx][0];
                    let acc2 = &accumulators[round_idx][1];
                    c0 += alpha
                        * dot_product::<EF, _, _>(acc0.iter().copied(), weights.iter().copied());
                    c2 += alpha
                        * dot_product::<EF, _, _>(acc2.iter().copied(), weights.iter().copied());
                }

                let r = sumcheck_data.observe_and_sample(challenger, c0, c2, pow_bits);
                sum = extrapolate_012(c0, sum - c0, c2, r);
                rs.push(r);
            }
        });

        let poly = poly.compress_multi_into_packed(&rs);
        let mut weights = Poly::<EF::ExtensionPacking>::zero(poly.num_variables());
        SplitEq::combine_into_packed(&mut weights.0, statements, alpha, &rs);
        let poly = ProductPolynomial::<F, EF>::new_packed(poly, weights);

        assert_eq!(poly.dot_product(), sum);
        (Self { poly, sum }, Point::new(rs))
    }

    /// Constructs a new `SumcheckSingle` instance from evaluations in the base field.
    ///
    /// This function:
    /// - Uses precomputed evaluations of the polynomial `p` over the Boolean hypercube.
    /// - Applies the given constraint `Statement` using a random linear combination.
    /// - Initializes internal sumcheck state with weights and expected sum.
    /// - Applies first set of sumcheck rounds
    #[tracing::instrument(skip_all)]
    pub fn from_base_evals<Challenger>(
        sumcheck_data: &mut SumcheckData<F, EF>,
        challenger: &mut Challenger,
        folding_factor: usize,
        pow_bits: usize,
        statement: &InitialStatement<F, EF>,
    ) -> (Self, Point<EF>)
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        let k = statement.num_variables();
        assert_ne!(folding_factor, 0, "number of rounds must be non-zero");
        assert!(
            folding_factor <= k,
            "number of rounds must be less than or equal to instance size"
        );

        let k_pack = log2_strict_usize(F::Packing::WIDTH);
        let poly = &statement.poly;
        match &statement.inner {
            InitialStatementInner::Svo { split_eqs, l0 } => {
                assert_eq!(*l0, folding_factor);
                assert!(k > 2 * k_pack + folding_factor);
                Self::new_svo(
                    poly,
                    sumcheck_data,
                    challenger,
                    folding_factor,
                    pow_bits,
                    split_eqs,
                )
            }
            InitialStatementInner::Classic(statement) => {
                if k > k_pack {
                    Self::new_classic_packed(
                        poly,
                        sumcheck_data,
                        challenger,
                        folding_factor,
                        pow_bits,
                        statement,
                    )
                } else {
                    // Fallback to classic for small instances
                    Self::new_classic_small(
                        poly,
                        sumcheck_data,
                        challenger,
                        folding_factor,
                        pow_bits,
                        statement,
                    )
                }
            }
        }
    }

    /// Returns the number of variables in the polynomial.
    pub fn num_variables(&self) -> usize {
        self.poly.num_variables()
    }

    /// Returns the polynomial evaluations.
    #[tracing::instrument(skip_all)]
    pub fn evals(&self) -> Poly<EF> {
        self.poly.evals()
    }

    /// Evaluates the sumcheck polynomial at a given multilinear point.
    pub fn eval(&self, point: &Point<EF>) -> EF {
        self.poly.eval(point)
    }

    /// Executes the sumcheck protocol for a multilinear polynomial
    ///
    /// This function performs `folding_factor` rounds of the sumcheck protocol:
    ///
    /// - At each round, a univariate polynomial is sent representing a partial sum over a subset of variables.
    /// - The verifier responds with a random challenge that is used to fix one variable.
    ///
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
    #[tracing::instrument(skip_all)]
    pub fn compute_sumcheck_polynomials<Challenger>(
        &mut self,
        sumcheck_data: &mut SumcheckData<F, EF>,
        challenger: &mut Challenger,
        folding_factor: usize,
        pow_bits: usize,
        constraint: Option<Constraint<F, EF>>,
    ) -> Point<EF>
    where
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
        Point::new(res)
    }
}
