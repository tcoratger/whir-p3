use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::whir::{Leafs, Proof};

/// The `WhirProof` struct encapsulates the verifier-facing data
/// produced during a WHIR proof, specifically:
/// - Authentication paths for Merkle commitments,
/// - Openings of the committed polynomials at verifier-chosen points,
/// - Evaluations of the public statement at the final random point.
///
/// This struct contains only the data required for **verification**.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize, EF: Serialize, [u8; DIGEST_ELEMS]: Serialize",
    deserialize = "F: DeserializeOwned, EF: DeserializeOwned, [u8; DIGEST_ELEMS]: DeserializeOwned",
))]
pub struct WhirProof<F, EF, const DIGEST_ELEMS: usize> {
    /// Merkle opening for the **base field** polynomial commitment.
    ///
    /// This corresponds to the initial (round 0) polynomial, and includes:
    /// - `Leafs<F>`: The opened leaf values of the committed base polynomial,
    /// - `Proof<DIGEST_ELEMS>`: The Merkle authentication path for those leaves.
    pub commitment_merkle_paths: (Leafs<F>, Proof<DIGEST_ELEMS>),

    /// Merkle openings for **extension field** commitments in folded rounds.
    ///
    /// Each entry corresponds to one folding round beyond the initial,
    /// and includes:
    /// - `Leafs<EF>`: The values of the folded polynomial at verifier-specified locations,
    /// - `Proof<DIGEST_ELEMS>`: The authentication path for those values.
    pub merkle_paths: Vec<(Leafs<EF>, Proof<DIGEST_ELEMS>)>,

    /// Evaluation of each public statement constraint at the final verifier-chosen point.
    ///
    /// This vector contains the actual values of the public inputs (or constraints)
    /// evaluated at the point derived from Fiat-Shamir. It is used to check that
    /// the claimed result matches the actual value of the constraints.
    pub statement_values_at_random_point: Vec<EF>,
}
