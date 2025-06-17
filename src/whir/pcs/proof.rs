use serde::{Deserialize, Serialize};

/// A WHIR proof object containing verifier-checkable information for a single round.
///
/// In the WHIR protocol, each round of the IOP constructs a folded constraint system,
/// performs out-of-domain (OOD) sampling to pin down the committed polynomial, and runs
/// a sumcheck protocol to verify batched multilinear constraints. The resulting proof
/// is stored as a `WhirProof`.
///
/// Each `WhirProof` corresponds to one matrix batch (or "round") in the commitment scheme,
/// and a full proof consists of a vector of such rounds.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WhirProof<Challenge> {
    /// Evaluations of the committed multilinear polynomial at verifier-sampled
    /// out-of-domain (OOD) challenge points. These values pin down the polynomial
    /// to a unique low-degree extension, enforcing consistency after folding.
    pub ood_answers: Vec<Challenge>,

    /// Fiat–Shamir transcript state after all prover messages for this round.
    pub narg_string: Vec<u8>,
}
