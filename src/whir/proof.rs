use serde::{Deserialize, Serialize};

use crate::constant::K_SKIP_SUMCHECK;
use crate::parameters::ProtocolParameters;
use crate::poly::evals::EvaluationsList;

/// Complete WHIR proof
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound(
    serialize = "F: Serialize, EF: Serialize, [F; DIGEST_ELEMS]: Serialize",
    deserialize = "F: Deserialize<'de>, EF: Deserialize<'de>, [F; DIGEST_ELEMS]: Deserialize<'de>"
))]
pub struct WhirProof<F, EF, const DIGEST_ELEMS: usize> {
    /// Initial polynomial commitment (Merkle root)
    pub initial_commitment: [F; DIGEST_ELEMS],

    // Initial PoW witness after the commitment
    pub initial_pow_witness: Option<F>,

    /// Initial OOD evaluations
    pub initial_ood_answers: Vec<EF>,

    /// Initial phase data - captures the protocol variant
    pub initial_phase: InitialPhase<EF, F>,

    /// One proof per WHIR round
    pub rounds: Vec<WhirRoundProof<F, EF, DIGEST_ELEMS>>,

    /// Final polynomial evaluations
    pub final_poly: Option<EvaluationsList<EF>>,

    /// Final round PoW witness
    pub final_pow_witness: F,

    /// Final round query openings
    pub final_queries: Vec<QueryOpening<F, EF, DIGEST_ELEMS>>,

    /// Final sumcheck (if final_sumcheck_rounds > 0)
    pub final_sumcheck: Option<SumcheckData<EF, F>>,
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
        skip_pow: Option<Vec<F>>,

        /// Remaining sumcheck rounds after the skip (for folding_factor > k_skip)
        sumcheck: SumcheckData<EF, F>,
    },

    /// Protocol with statement (standard sumcheck, no skip)
    #[serde(rename = "with_statement")]
    WithStatement {
        /// Standard sumcheck data
        sumcheck: SumcheckData<EF, F>,
    },

    /// Protocol without statement (direct folding)
    #[serde(rename = "without_statement")]
    WithoutStatement,
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
    pub pow_witness: Option<F>,

    /// STIR query openings
    pub queries: Vec<QueryOpening<F, EF, DIGEST_ELEMS>>,

    /// Sumcheck data for this round
    pub sumcheck: SumcheckData<EF, F>,
}

impl<F: Default, EF: Default, const DIGEST_ELEMS: usize> Default for WhirRoundProof<F, EF, DIGEST_ELEMS> {
    fn default() -> Self {
        Self {
            commitment: std::array::from_fn(|_| F::default()),
            ood_answers: Vec::new(),
            pow_witness: None,
            queries: Vec::new(),
            sumcheck: SumcheckData::default(),
        }
    }
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
        // The initial phase must match the prover's branching logic:
        // - WithStatementSkip is only used when univariate_skip is enabled AND
        //   the folding factor is large enough (>= K_SKIP_SUMCHECK)
        let initial_phase = match (
            params.initial_statement,
            params.univariate_skip && K_SKIP_SUMCHECK <= params.folding_factor.at_round(0),
        ) {
            (true, true) => InitialPhase::with_statement_skip(
                Vec::new(),
                None,
                SumcheckData::default(),
            ),
            (true, false) => InitialPhase::with_statement(SumcheckData::default()),
            (false, _) => InitialPhase::without_statement(),
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
            initial_pow_witness: None,
            initial_ood_answers: Vec::new(),
            initial_phase,
            rounds: Vec::with_capacity(num_rounds),
            final_poly: None,
            final_pow_witness: F::default(),
            final_queries: Vec::with_capacity(num_queries),
            final_sumcheck: None,
        }
    }
}

impl<F: Clone, EF, const DIGEST_ELEMS: usize> WhirProof<F, EF, DIGEST_ELEMS> {
    /// Extract the PoW witness after the commitment at the given round index
    ///
    /// * round_index = 0: Returns the PoW witness after the initial commitment
    /// * round_index > 0: Returns the PoW witness after the commitment in round (round_index - 1)
    pub fn get_pow_after_commitment(&self, round_index: usize) -> Option<F> {
        if round_index == 0 {
            // Initial commitment PoW
            self.initial_pow_witness.clone()
        } else {
            // Ordinary round PoW (round_index - 1 because rounds are 0-indexed)
            self.rounds.get(round_index - 1).and_then(|round| round.pow_witness.clone())
        }
    }

    /*
    /// Extract the PoW witness from the sumcheck rounds
    pub fn get_sumcheck_pow(&self, round_index: usize) -> Option<Vec<F>> {
        if round_index == 0 {
            // Initial commitment PoW
            match &self.initial_phase {
                InitialPhase::WithStatementSkip { skip_pow, .. } => {
                    skip_pow.clone()
                }
                InitialPhase::WithStatement { sumcheck, .. } => {
                    sumcheck.pow_witnesses.clone()
                }
                InitialPhase::WithoutStatement => None,
            }
        } else {
            // Ordinary round PoW
            self.rounds
                .get(round_index )
                .and_then(|round| round.sumcheck.pow_witnesses.clone())
        }
    }
     */
}

impl<EF, F> InitialPhase<EF, F> {
    /// Create initial phase with statement and skip optimization
    pub fn with_statement_skip(
        skip_evaluations: Vec<EF>,
        skip_pow: Option<Vec<F>>,
        sumcheck: SumcheckData<EF, F>,
    ) -> Self {
        Self::WithStatementSkip {
            skip_evaluations,
            skip_pow,
            sumcheck,
        }
    }

    /// Create initial phase with statement (no skip)
    pub fn with_statement(sumcheck: SumcheckData<EF, F>) -> Self {
        Self::WithStatement { sumcheck }
    }

    /// Create initial phase without statement
    pub fn without_statement() -> Self {
        Self::WithoutStatement
    }
}