use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field};
use serde::{Deserialize, Serialize};

/// Complete WHIR proof
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound(
    serialize = "M::Commitment: Serialize, M::Proof: Serialize",
    deserialize = "M::Commitment: Deserialize<'de>, M::Proof: Deserialize<'de>"
))]
pub struct WhirProof<F, EF, M: Mmcs<F>> {
    /// Initial polynomial commitment (Merkle root)
    pub initial_commitment: M::Commitment,

    /// Initial OOD evaluations
    pub initial_ood_answers: Vec<EF>,

    /// Initial phase data - captures the protocol variant
    pub initial_phase: InitialPhase<EF, F>,

    /// One proof per WHIR round
    pub rounds: Vec<WhirRoundProof<F, EF, M>>,

    /// Final polynomial coefficients
    pub final_poly: Vec<EF>,

    /// Final round PoW witness
    pub final_pow_witness: F,

    /// Final round query openings
    pub final_queries: Vec<QueryOpening<F, EF, M>>,

    /// Final sumcheck (if final_sumcheck_rounds > 0)
    pub final_sumcheck: Option<SumcheckData<F, EF>>,
}

/// Initial phase of WHIR protocol
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "type")]
pub enum InitialPhase<EF, F> {
    /// Protocol with statement and univariate skip optimization
    #[serde(rename = "with_statement_skip")]
    WithStatementSkip {
        /// Skip round polynomial evaluations
        skip_evaluations: Vec<EF>,

        /// PoW witness after skip round
        skip_pow: F,

        /// Standard sumcheck for remaining variables
        remaining_sumcheck: SumcheckData<EF, F>,
    },

    /// Protocol with statement (standard sumcheck, no skip)
    #[serde(rename = "with_statement")]
    WithStatement {
        /// Standard sumcheck data
        sumcheck: SumcheckData<EF, F>,
    },

    /// Protocol without statement (direct folding)
    #[serde(rename = "without_statement")]
    WithoutStatement {
        /// Single PoW witness
        pow_witness: F,
    },
}

/// Data for a single WHIR round
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound(
    serialize = "M::Commitment: Serialize, M::Proof: Serialize",
    deserialize = "M::Commitment: Deserialize<'de>, M::Proof: Deserialize<'de>"
))]
pub struct WhirRoundProof<F, EF, M: Mmcs<F>> {
    /// Round commitment (Merkle root)
    pub commitment: M::Commitment,

    /// OOD evaluations for this round
    pub ood_answers: Vec<EF>,

    /// PoW witness after commitment
    pub pow_witness: F,

    /// STIR query openings
    pub queries: Vec<QueryOpening<F, EF, M>>,

    /// Sumcheck data for this round
    pub sumcheck: SumcheckData<EF, F>,
}

/// Query opening 
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(
    bound(
        serialize = "M::Proof: Serialize",
        deserialize = "M::Proof: Deserialize<'de>"
    ),
    tag = "type"
)]
pub enum QueryOpening<F, EF, M: Mmcs<F>> {
    /// Base field query (round_index == 0)
    #[serde(rename = "base")]
    Base {
        /// Merkle leaf values in F
        values: Vec<F>,
        /// Merkle authentication path
        proof: M::Proof,
    },
    /// Extension field query (round_index > 0)
    #[serde(rename = "extension")]
    Extension {
        /// Merkle leaf values in EF
        values: Vec<EF>,
        /// Merkle authentication path
        proof: M::Proof,
    },
}

/// Sumcheck polynomial data
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SumcheckData<EF, F> {
    /// Polynomial evaluations for each sumcheck round
    /// Each entry contains [h(0), h(1), h(2)] for that round
    /// Length: folding_factor
    pub polynomial_evaluations: Vec<[EF; 3]>,

    /// PoW witnesses for each sumcheck round
    /// Length: folding_factor
    pub pow_witnesses: Vec<F>,
}