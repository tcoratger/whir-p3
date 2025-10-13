use serde::{Deserialize, Serialize};

use crate::parameters::ProtocolParameters;

/// Complete WHIR proof
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound(
    serialize = "F: Serialize, EF: Serialize, [F; DIGEST_ELEMS]: Serialize",
    deserialize = "F: Deserialize<'de>, EF: Deserialize<'de>, [F; DIGEST_ELEMS]: Deserialize<'de>"
))]
pub struct WhirProof<F, EF, const DIGEST_ELEMS: usize> {
    /// Initial polynomial commitment (Merkle root)
    pub initial_commitment: [F; DIGEST_ELEMS],

    /// Initial OOD evaluations
    pub initial_ood_answers: Vec<EF>,

    /// Initial phase data - captures the protocol variant
    pub initial_phase: InitialPhase<EF, F>,

    /// One proof per WHIR round
    pub rounds: Vec<WhirRoundProof<F, EF, DIGEST_ELEMS>>,

    /// Final polynomial coefficients
    pub final_poly: Vec<EF>,

    /// Final round PoW witness
    pub final_pow_witness: F,

    /// Final round query openings
    pub final_queries: Vec<QueryOpening<F, EF, DIGEST_ELEMS>>,

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
    serialize = "F: Serialize, EF: Serialize, [F; DIGEST_ELEMS]: Serialize",
    deserialize = "F: Deserialize<'de>, EF: Deserialize<'de>, [F; DIGEST_ELEMS]: Deserialize<'de>"
))]
pub struct WhirRoundProof<F, EF, const DIGEST_ELEMS: usize> {
    /// Round commitment (Merkle root)
    pub commitment: [F; DIGEST_ELEMS],

    /// OOD evaluations for this round
    pub ood_answers: Vec<EF>,

    /// PoW witness after commitment
    pub pow_witness: F,

    /// STIR query openings
    pub queries: Vec<QueryOpening<F, EF, DIGEST_ELEMS>>,

    /// Sumcheck data for this round
    pub sumcheck: SumcheckData<EF, F>,
}

/// Query opening
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(
    bound(
        serialize = "F: Serialize, EF: Serialize, [F; DIGEST_ELEMS]: Serialize",
        deserialize = "F: Deserialize<'de>, EF: Deserialize<'de>, [F; DIGEST_ELEMS]: Deserialize<'de>"
    ),
    tag = "type"
)]
pub enum QueryOpening<F, EF, const DIGEST_ELEMS: usize> {
    /// Base field query (round_index == 0)
    #[serde(rename = "base")]
    Base {
        /// Merkle leaf values in F
        values: Vec<F>,
        /// Merkle authentication path
        proof: Vec<[F; DIGEST_ELEMS]>,
    },
    /// Extension field query (round_index > 0)
    #[serde(rename = "extension")]
    Extension {
        /// Merkle leaf values in EF
        values: Vec<EF>,
        /// Merkle authentication path
        proof: Vec<[F; DIGEST_ELEMS]>
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
    pub pow_witnesses: Option<Vec<F>>,
}

impl<EF, F> Default for SumcheckData<EF, F> {
    fn default() -> Self {
        Self {
            polynomial_evaluations: Vec::new(),
            pow_witnesses: None,
        }
    }
}

impl<F: Default, EF: Default, const DIGEST_ELEMS: usize> WhirProof<F, EF, DIGEST_ELEMS> {
    /// Create a new WhirProof from protocol parameters and configuration
    ///
    /// This initializes an empty proof structure with appropriate capacity allocations
    /// based on the protocol parameters. The actual proof data will be populated during
    /// the proving process.
    ///
    /// # Parameters
    /// - `params`: The protocol parameters containing security settings and folding configuration
    /// - `num_variables`: The number of variables in the multilinear polynomial
    ///
    /// # Returns
    /// A new `WhirProof` with pre-allocated vectors sized according to the protocol parameters
    pub fn from_protocol_parameters<H, C>(
        params: &ProtocolParameters<H, C>,
        num_variables: usize,
    ) -> Self {
        let initial_phase = match (params.initial_statement, params.univariate_skip) {
            (true, true) => InitialPhase::with_statement_skip(
                Vec::new(),
                F::default(),
                SumcheckData::default(),
            ),
            (true, false) => InitialPhase::with_statement(SumcheckData::default()),
            (false, _) => InitialPhase::without_statement(F::default()),
        };

        // Use the actual FoldingFactor method to calculate rounds correctly
        let (num_rounds, _final_sumcheck_rounds) = params
            .folding_factor
            .compute_number_of_rounds(num_variables);

        // Calculate protocol security level (after subtracting PoW bits)
        let protocol_security_level = params.security_level.saturating_sub(params.pow_bits);

        // Compute the number of queries
        let num_queries = params.soundness_type.queries(
            protocol_security_level,
            params.starting_log_inv_rate,
        );

        Self {
            initial_commitment: std::array::from_fn(|_| F::default()),
            initial_ood_answers: Vec::new(),
            initial_phase,
            rounds: Vec::with_capacity(num_rounds),
            final_poly: Vec::new(),
            final_pow_witness: F::default(),
            final_queries: Vec::with_capacity(num_queries),
            final_sumcheck: None,
        }
    }
}

impl<EF, F> InitialPhase<EF, F> {
    /// Create initial phase with statement and skip optimization
    pub fn with_statement_skip(
        skip_evaluations: Vec<EF>,
        skip_pow: F,
        remaining_sumcheck: SumcheckData<EF, F>,
    ) -> Self {
        Self::WithStatementSkip {
            skip_evaluations,
            skip_pow,
        }
    }

    /// Create initial phase with statement (no skip)
    pub fn with_statement(sumcheck: SumcheckData<EF, F>) -> Self {
        Self::WithStatement { sumcheck }
    }

    /// Create initial phase without statement
    pub fn without_statement(pow_witness: F) -> Self {
        Self::WithoutStatement { pow_witness }
    }
}