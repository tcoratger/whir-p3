use p3_field::{ExtensionField, Field, TwoAdicField};

use super::{Leafs, Proof};
use crate::{
    domain::Domain,
    poly::{coeffs::CoefficientStorage, multilinear::MultilinearPoint},
    sumcheck::sumcheck_single::SumcheckSingle,
    whir::{
        committer::{CommitmentMerkleTree, RoundMerkleTree},
        statement::Statement,
    },
};

/// Holds all per-round prover state required during the execution of the WHIR protocol.
///
/// Each round involves:
/// - A domain extension and folding step,
/// - Merkle commitments and openings,
/// - A sumcheck polynomial generation and folding randomness sampling,
/// - Bookkeeping of constraints and evaluation points.
///
/// The `RoundState` evolves with each round and captures all intermediate data required
/// to continue proving or to verify challenges from the verifier.
#[derive(Debug)]
pub(crate) struct RoundState<EF, F, const DIGEST_ELEMS: usize>
where
    F: Field + TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
{
    /// Current round index (starts from 0).
    pub(crate) round: usize,

    /// The domain used in this round, including the size and generator.
    /// This is typically a scaled version of the previous round’s domain.
    pub(crate) domain: Domain<EF, F>,

    /// The sumcheck prover responsible for managing constraint accumulation and sumcheck rounds.
    /// Initialized in the first round (if applicable), and reused/updated in each subsequent round.
    pub(crate) sumcheck_prover: Option<SumcheckSingle<F, EF>>,

    /// The sampled folding randomness for this round, used to collapse a subset of variables.
    /// Length equals the folding factor at this round.
    pub(crate) folding_randomness: MultilinearPoint<EF>,

    /// The multilinear polynomial coefficients at the start of this round.
    /// These are updated by folding the previous round’s coefficients using `folding_randomness`.
    pub(crate) coefficients: CoefficientStorage<F, EF>,

    /// Merkle commitment prover data for the **base field** polynomial from the first round.
    /// This is used to open values at queried locations.
    pub(crate) commitment_merkle_prover_data: CommitmentMerkleTree<F, DIGEST_ELEMS>,

    /// Merkle commitment prover data for the **extension field** polynomials (folded rounds).
    /// Present only after the first round.
    pub(crate) merkle_prover_data: Option<RoundMerkleTree<F, EF, DIGEST_ELEMS>>,

    /// Merkle proof from the initial commitment round.
    /// - First: list of opened leaf values in the base field.
    /// - Second: corresponding Merkle authentication paths.
    /// - Empty during setup; populated during final query phase.
    pub(crate) commitment_merkle_proof: Option<(Leafs<F>, Proof<DIGEST_ELEMS>)>,

    /// Merkle proofs for intermediate folded rounds.
    /// Each entry contains:
    /// - The opened values at verifier-chosen locations,
    /// - The corresponding authentication paths.
    pub(crate) merkle_proofs: Vec<(Leafs<EF>, Proof<DIGEST_ELEMS>)>,

    /// Flat vector of challenge values used across all rounds.
    /// Populated progressively as folding randomness is sampled.
    /// The `i`-th index corresponds to variable `X_{n - 1 - i}`.
    pub(crate) randomness_vec: Vec<EF>,

    /// The accumulated set of linear equality constraints for this round.
    /// Used in computing the weighted sum for the sumcheck polynomial.
    pub(crate) statement: Statement<EF>,
}
