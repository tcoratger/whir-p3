use crate::{
    poly::multilinear::MultilinearPoint, sumcheck::sumcheck_polynomial::SumcheckPolynomial,
};

/// Represents a single folding round in the WHIR protocol.
///
/// This structure enables recursive compression and verification of a Reed–Solomon
/// proximity test under algebraic constraints.
#[derive(Default, Debug, Clone)]
pub(crate) struct ParsedRound<F> {
    /// Folding randomness vector used in this round.
    pub(crate) folding_randomness: MultilinearPoint<F>,
    /// Out-of-domain query points.
    pub(crate) ood_points: Vec<F>,
    /// OOD answers at each query point.
    pub(crate) ood_answers: Vec<F>,
    /// Indexes of STIR constraint polynomials used in this round.
    pub(crate) stir_challenges_indexes: Vec<usize>,
    /// STIR constraint evaluation points.
    pub(crate) stir_challenges_points: Vec<F>,
    /// Answers to the STIR constraints at each evaluation point.
    pub(crate) stir_challenges_answers: Vec<Vec<F>>,
    /// Randomness used to linearly combine constraints.
    pub(crate) combination_randomness: Vec<F>,
    /// Sumcheck messages and challenge values for verifying correctness.
    pub(crate) sumcheck_rounds: Vec<(SumcheckPolynomial<F>, F)>,
    /// Inverse of the domain generator used in this round.
    pub(crate) domain_gen_inv: F,
}
