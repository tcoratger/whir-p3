use alloc::vec::Vec;
use core::array;

use serde::{Deserialize, Serialize};

use crate::{
    constant::K_SKIP_SUMCHECK, parameters::ProtocolParameters, poly::evals::EvaluationsList,
    whir::parameters::SumcheckOptimization,
};

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
        skip_pow: Option<F>,

        /// Remaining sumcheck rounds after the skip (for folding_factor > k_skip)
        sumcheck: SumcheckData<EF, F>,
    },

    /// Protocol with statement and svo optimization.
    /// First 3 rounds of svo optimization, the remaing rounds from algorithm_5
    /// (which have the same structure) stored in the subsequents WhirRoundProof elements.
    #[serde(rename = "with_statement_svo")]
    WithStatementSvo {
        /// Svo sumcheck data
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

impl<F: Default, EF: Default, const DIGEST_ELEMS: usize> Default
    for WhirRoundProof<F, EF, DIGEST_ELEMS>
{
    fn default() -> Self {
        Self {
            commitment: array::from_fn(|_| F::default()),
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
        proof: Vec<[F; DIGEST_ELEMS]>,
    },
}

/// Sumcheck round data that can represent different optimization formats
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "format")]
pub enum SumcheckRoundData<EF> {
    /// Classic format: h(0), h(1), h(2)
    #[serde(rename = "classic")]
    Classic([EF; 3]),

    /// SVO format: S(0), S(inf)
    /// S(1) is derived as claimed_sum - S(0)
    #[serde(rename = "svo")]
    Svo([EF; 2]),
}

/// Sumcheck polynomial data
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SumcheckData<EF, F> {
    /// Polynomial evaluations for each sumcheck round
    /// Format depends on optimization used
    /// Length: folding_factor
    pub polynomial_evaluations: Vec<SumcheckRoundData<EF>>,

    /// PoW witnesses for each sumcheck round (optional)
    /// Length: folding_factor
    pub pow_witnesses: Option<Vec<F>>,
}

impl<EF: Default, F: Default> Default for SumcheckData<EF, F> {
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
        // Determine which initial phase variant based on protocol configuration
        let initial_phase = match (params.initial_statement, params.sumcheck_optimization) {
            // No initial statement: direct folding path
            (false, _) => InitialPhase::without_statement(),

            // With statement + UnivariateSkip optimization
            (true, SumcheckOptimization::UnivariateSkip)
                if K_SKIP_SUMCHECK <= params.folding_factor.at_round(0) =>
            {
                InitialPhase::with_statement_skip(Vec::new(), None, SumcheckData::default())
            }

            // With statement + SVO optimization
            (true, SumcheckOptimization::Svo) => {
                InitialPhase::with_statement_svo(SumcheckData::default())
            }

            // With statement + Classic (or UnivariateSkip with insufficient folding factor)
            (true, _) => InitialPhase::with_statement(SumcheckData::default()),
        };

        // Use the actual FoldingFactor method to calculate rounds correctly
        let (num_rounds, _final_sumcheck_rounds) = params
            .folding_factor
            .compute_number_of_rounds(num_variables);

        // Calculate protocol security level (after subtracting PoW bits)
        let protocol_security_level = params.security_level.saturating_sub(params.pow_bits);

        // Compute the number of queries
        let num_queries = params
            .soundness_type
            .queries(protocol_security_level, params.starting_log_inv_rate);

        Self {
            initial_commitment: array::from_fn(|_| F::default()),
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
    /// Returns the PoW witness from the round at the given index.
    /// The PoW witness is stored in proof.rounds[round_index].pow_witness.
    pub fn get_pow_after_commitment(&self, round_index: usize) -> Option<F> {
        self.rounds
            .get(round_index)
            .and_then(|round| round.pow_witness.clone())
    }
}

impl<EF, F> InitialPhase<EF, F> {
    /// Create initial phase with statement and skip optimization
    pub const fn with_statement_skip(
        skip_evaluations: Vec<EF>,
        skip_pow: Option<F>,
        sumcheck: SumcheckData<EF, F>,
    ) -> Self {
        Self::WithStatementSkip {
            skip_evaluations,
            skip_pow,
            sumcheck,
        }
    }

    /// Create initial phase with statement and SVO optimization
    #[must_use]
    pub const fn with_statement_svo(sumcheck: SumcheckData<EF, F>) -> Self {
        Self::WithStatementSvo { sumcheck }
    }

    /// Create initial phase with statement (no skip)
    #[must_use]
    pub const fn with_statement(sumcheck: SumcheckData<EF, F>) -> Self {
        Self::WithStatement { sumcheck }
    }

    /// Create initial phase without statement
    #[must_use]
    pub const fn without_statement() -> Self {
        Self::WithoutStatement
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::SeedableRng;

    use super::*;
    use crate::{
        parameters::{FoldingFactor, errors::SecurityAssumption},
        whir::parameters::SumcheckOptimization,
    };

    /// Type alias for the base field used in tests
    type F = BabyBear;

    /// Type alias for the extension field used in tests
    type EF = BinomialExtensionField<F, 4>;

    /// Type alias for the permutation used in Merkle tree
    type Perm = Poseidon2BabyBear<16>;

    /// Type alias for the hash function
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;

    /// Type alias for the compression function
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;

    /// Digest size for Merkle tree commitments
    const DIGEST_ELEMS: usize = 8;

    /// Helper function to create minimal protocol parameters for testing
    ///
    /// This creates a `ProtocolParameters` instance with specified configuration
    /// for testing different proof initialization scenarios.
    ///
    /// # Parameters
    /// - `initial_statement`: Whether the protocol includes an initial statement
    /// - `folding_factor`: The folding strategy for the protocol
    /// - `sumcheck_optimization`: The sumcheck optimization strategy to use
    ///
    /// # Returns
    /// A `ProtocolParameters` instance configured for testing
    fn create_test_params(
        initial_statement: bool,
        folding_factor: FoldingFactor,
        sumcheck_optimization: SumcheckOptimization,
    ) -> ProtocolParameters<MyHash, MyCompress> {
        // Create the permutation for hash and compress
        let perm = Perm::new_from_rng_128(&mut rand::rngs::SmallRng::seed_from_u64(42));

        ProtocolParameters {
            initial_statement,
            starting_log_inv_rate: 2,
            rs_domain_initial_reduction_factor: 1,
            folding_factor,
            soundness_type: SecurityAssumption::UniqueDecoding,
            security_level: 100,
            pow_bits: 10,
            merkle_hash: PaddingFreeSponge::new(perm.clone()),
            merkle_compress: TruncatedPermutation::new(perm),
            sumcheck_optimization,
        }
    }

    #[test]
    fn test_whir_proof_from_params_with_univariate_skip() {
        // Declare test parameters explicitly

        // Use initial statement with univariate skip optimization
        let initial_statement = true;

        // Set folding factor to 6, which is >= K_SKIP_SUMCHECK (5)
        // This ensures univariate skip optimization is enabled
        let folding_factor_value = 6;
        let folding_factor = FoldingFactor::Constant(folding_factor_value);

        // Enable univariate skip optimization
        let sumcheck_optimization = SumcheckOptimization::UnivariateSkip;

        // Use 20 variables for testing
        let num_variables = 20;

        // Create protocol parameters with univariate skip enabled
        let params = create_test_params(initial_statement, folding_factor, sumcheck_optimization);

        // Create proof structure from parameters
        let proof: WhirProof<F, EF, DIGEST_ELEMS> =
            WhirProof::from_protocol_parameters(&params, num_variables);

        // Verify that initial_phase is WithStatementSkip variant
        // This should be true because:
        // - initial_statement = true
        // - sumcheck_optimization = UnivariateSkip
        // - folding_factor (6) >= K_SKIP_SUMCHECK (5)
        match proof.initial_phase {
            InitialPhase::WithStatementSkip {
                skip_evaluations,
                skip_pow,
                sumcheck,
            } => {
                // skip_evaluations should be empty (not populated yet)
                assert_eq!(skip_evaluations.len(), 0);
                // skip_pow should be None (not populated yet)
                assert!(skip_pow.is_none());
                // sumcheck should have empty polynomial_evaluations
                assert_eq!(sumcheck.polynomial_evaluations.len(), 0);
                // sumcheck should have no PoW witnesses
                assert!(sumcheck.pow_witnesses.is_none());
            }
            _ => panic!("Expected WithStatementSkip variant"),
        }

        // Verify rounds capacity
        // Formula: ((num_variables - MAX_NUM_VARIABLES_TO_SEND_COEFFS) / folding_factor) - 1
        // MAX_NUM_VARIABLES_TO_SEND_COEFFS = 6 (threshold for sending coefficients directly)
        // For 20 variables with folding_factor 6:
        //   (20 - 6).div_ceil(6) - 1 = 14.div_ceil(6) - 1 = 3 - 1 = 2
        let expected_rounds = 2;
        assert_eq!(proof.rounds.capacity(), expected_rounds);
    }

    #[test]
    fn test_whir_proof_from_params_without_univariate_skip() {
        // Declare test parameters explicitly

        // Use initial statement without univariate skip optimization
        let initial_statement = true;

        // Set folding factor to 4, which is < K_SKIP_SUMCHECK (5)
        // This ensures univariate skip optimization is NOT enabled
        let folding_factor_value = 4;
        let folding_factor = FoldingFactor::Constant(folding_factor_value);

        // Even if we request UnivariateSkip, it won't be used
        // because folding_factor < K_SKIP_SUMCHECK
        let sumcheck_optimization = SumcheckOptimization::UnivariateSkip;

        // Use 16 variables for testing
        let num_variables = 16;

        // Create protocol parameters (skip won't be enabled due to folding_factor < 5)
        let params = create_test_params(initial_statement, folding_factor, sumcheck_optimization);

        // Create proof structure from parameters
        let proof: WhirProof<F, EF, DIGEST_ELEMS> =
            WhirProof::from_protocol_parameters(&params, num_variables);

        // Verify that initial_phase is WithStatement variant (NOT WithStatementSkip)
        // This is because folding_factor (4) < K_SKIP_SUMCHECK (5)
        match proof.initial_phase {
            InitialPhase::WithStatement { sumcheck } => {
                // sumcheck should have empty polynomial_evaluations
                assert_eq!(sumcheck.polynomial_evaluations.len(), 0);
                // sumcheck should have no PoW witnesses
                assert!(sumcheck.pow_witnesses.is_none());
            }
            _ => panic!("Expected WithStatement variant, not WithStatementSkip"),
        }

        // Verify rounds capacity
        // Formula: ((num_variables - MAX_NUM_VARIABLES_TO_SEND_COEFFS) / folding_factor) - 1
        // MAX_NUM_VARIABLES_TO_SEND_COEFFS = 6
        // For 16 variables with folding_factor 4:
        //   (16 - 6).div_ceil(4) - 1 = 10.div_ceil(4) - 1 = 3 - 1 = 2
        let expected_rounds = 2;
        assert_eq!(proof.rounds.capacity(), expected_rounds);
    }

    #[test]
    fn test_whir_proof_from_params_without_initial_statement() {
        // Declare test parameters explicitly

        // Disable initial statement
        let initial_statement = false;

        // Folding factor doesn't matter for initial_phase when initial_statement=false
        let folding_factor_value = 6;
        let folding_factor = FoldingFactor::Constant(folding_factor_value);

        // Optimization doesn't matter when there's no initial statement
        let sumcheck_optimization = SumcheckOptimization::UnivariateSkip;

        // Use 18 variables for testing
        let num_variables = 18;

        // Create protocol parameters without initial statement
        let params = create_test_params(initial_statement, folding_factor, sumcheck_optimization);

        // Create proof structure from parameters
        let proof: WhirProof<F, EF, DIGEST_ELEMS> =
            WhirProof::from_protocol_parameters(&params, num_variables);

        // Verify that initial_phase is WithoutStatement variant
        // This is because initial_statement = false
        match proof.initial_phase {
            InitialPhase::WithoutStatement => {
                // This is the expected variant, nothing to check inside
            }
            _ => panic!("Expected WithoutStatement variant"),
        }

        // Verify rounds capacity
        // Formula: ((num_variables - MAX_NUM_VARIABLES_TO_SEND_COEFFS) / folding_factor) - 1
        // MAX_NUM_VARIABLES_TO_SEND_COEFFS = 6
        // For 18 variables with folding_factor 6:
        //   (18 - 6).div_ceil(6) - 1 = 12.div_ceil(6) - 1 = 2 - 1 = 1
        let expected_rounds = 1;
        assert_eq!(proof.rounds.capacity(), expected_rounds);
    }

    #[test]
    fn test_get_pow_after_commitment_with_witness() {
        // Create an explicit PoW witness value for testing
        let pow_witness_value = F::from_u64(42);

        // Create a proof with one round containing a PoW witness
        let proof: WhirProof<F, EF, DIGEST_ELEMS> = WhirProof {
            initial_commitment: array::from_fn(|_| F::default()),
            initial_pow_witness: None,
            initial_ood_answers: Vec::new(),
            initial_phase: InitialPhase::WithoutStatement,
            rounds: vec![WhirRoundProof {
                commitment: array::from_fn(|_| F::default()),
                ood_answers: Vec::new(),
                pow_witness: Some(pow_witness_value),
                queries: Vec::new(),
                sumcheck: SumcheckData::default(),
            }],
            final_poly: None,
            final_pow_witness: F::default(),
            final_queries: Vec::new(),
            final_sumcheck: None,
        };

        // Query round index 0, which exists and has a PoW witness
        let round_index = 0;

        // Get the PoW witness after commitment at round 0
        let result = proof.get_pow_after_commitment(round_index);

        // Verify that we get Some(pow_witness_value)
        assert_eq!(result, Some(pow_witness_value));
    }

    #[test]
    fn test_get_pow_after_commitment_without_witness() {
        // Create a proof with one round but no PoW witness
        let proof: WhirProof<F, EF, DIGEST_ELEMS> = WhirProof {
            initial_commitment: array::from_fn(|_| F::default()),
            initial_pow_witness: None,
            initial_ood_answers: Vec::new(),
            initial_phase: InitialPhase::WithoutStatement,
            rounds: vec![WhirRoundProof {
                commitment: array::from_fn(|_| F::default()),
                ood_answers: Vec::new(),
                pow_witness: None,
                queries: Vec::new(),
                sumcheck: SumcheckData::default(),
            }],
            final_poly: None,
            final_pow_witness: F::default(),
            final_queries: Vec::new(),
            final_sumcheck: None,
        };

        // Query round index 0, which exists but has no PoW witness
        let round_index = 0;

        // Get the PoW witness after commitment at round 0
        let result = proof.get_pow_after_commitment(round_index);

        // Verify that we get None because pow_witness is None
        assert_eq!(result, None);
    }

    #[test]
    fn test_get_pow_after_commitment_invalid_round() {
        // Create a proof with one round
        let proof: WhirProof<F, EF, DIGEST_ELEMS> = WhirProof {
            initial_commitment: array::from_fn(|_| F::default()),
            initial_pow_witness: None,
            initial_ood_answers: Vec::new(),
            initial_phase: InitialPhase::WithoutStatement,
            rounds: vec![WhirRoundProof {
                commitment: array::from_fn(|_| F::default()),
                ood_answers: Vec::new(),
                pow_witness: Some(F::from_u64(42)),
                queries: Vec::new(),
                sumcheck: SumcheckData::default(),
            }],
            final_poly: None,
            final_pow_witness: F::default(),
            final_queries: Vec::new(),
            final_sumcheck: None,
        };

        // Query round index 1, which doesn't exist (only round 0 exists)
        let invalid_round_index = 1;

        // Get the PoW witness after commitment at invalid round
        let result = proof.get_pow_after_commitment(invalid_round_index);

        // Verify that we get None because the round doesn't exist
        assert_eq!(result, None);
    }

    #[test]
    fn test_initial_phase_constructors() {
        // Test with_statement_skip constructor

        // Create empty skip evaluations
        let skip_evaluations: Vec<EF> = Vec::new();

        // Create skip PoW witness
        let skip_pow_value = F::from_u64(123);
        let skip_pow = Some(skip_pow_value);

        // Create empty sumcheck data
        let sumcheck: SumcheckData<EF, F> = SumcheckData::default();

        // Construct WithStatementSkip variant
        let phase_skip = InitialPhase::with_statement_skip(skip_evaluations, skip_pow, sumcheck);

        // Verify it's the correct variant
        match phase_skip {
            InitialPhase::WithStatementSkip {
                skip_evaluations,
                skip_pow,
                sumcheck,
            } => {
                assert_eq!(skip_evaluations.len(), 0);
                assert_eq!(skip_pow, Some(skip_pow_value));
                assert_eq!(sumcheck.polynomial_evaluations.len(), 0);
            }
            _ => panic!("Expected WithStatementSkip variant"),
        }

        // Test with_statement constructor

        // Create empty sumcheck data
        let sumcheck: SumcheckData<EF, F> = SumcheckData::default();

        // Construct WithStatement variant
        let phase_statement = InitialPhase::with_statement(sumcheck);

        // Verify it's the correct variant
        match phase_statement {
            InitialPhase::WithStatement { sumcheck } => {
                assert_eq!(sumcheck.polynomial_evaluations.len(), 0);
            }
            _ => panic!("Expected WithStatement variant"),
        }

        // Test without_statement constructor

        // Construct WithoutStatement variant
        let phase_without = InitialPhase::<EF, F>::without_statement();

        // Verify it's the correct variant
        match phase_without {
            InitialPhase::WithoutStatement => {
                // Correct variant, nothing to check
            }
            _ => panic!("Expected WithoutStatement variant"),
        }
    }

    #[test]
    fn test_whir_round_proof_default() {
        // Create a default WhirRoundProof
        let round: WhirRoundProof<F, EF, DIGEST_ELEMS> = WhirRoundProof::default();

        // Verify commitment is array of default F values
        assert_eq!(round.commitment.len(), DIGEST_ELEMS);
        for elem in round.commitment {
            assert_eq!(elem, F::default());
        }

        // Verify ood_answers is empty
        assert_eq!(round.ood_answers.len(), 0);

        // Verify pow_witness is None
        assert!(round.pow_witness.is_none());

        // Verify queries is empty
        assert_eq!(round.queries.len(), 0);

        // Verify sumcheck has default values
        assert_eq!(round.sumcheck.polynomial_evaluations.len(), 0);
        assert!(round.sumcheck.pow_witnesses.is_none());
    }

    #[test]
    fn test_sumcheck_data_default() {
        // Create a default SumcheckData
        let sumcheck: SumcheckData<EF, F> = SumcheckData::default();

        // Verify polynomial_evaluations is empty
        assert_eq!(sumcheck.polynomial_evaluations.len(), 0);

        // Verify pow_witnesses is None
        assert!(sumcheck.pow_witnesses.is_none());
    }

    #[test]
    fn test_query_opening_variants() {
        // Test Base variant

        // Create base field values
        let base_val_0 = F::from_u64(1);
        let base_val_1 = F::from_u64(2);
        let values = vec![base_val_0, base_val_1];

        // Create Merkle proof (authentication path)
        let proof_node = array::from_fn(|i| F::from_u64(i as u64));
        let proof = vec![proof_node];

        // Construct Base variant
        let base_opening: QueryOpening<F, EF, DIGEST_ELEMS> = QueryOpening::Base {
            values,
            proof: proof.clone(),
        };

        // Verify it's the correct variant
        match base_opening {
            QueryOpening::Base {
                values: v,
                proof: p,
            } => {
                assert_eq!(v.len(), 2);
                assert_eq!(v[0], base_val_0);
                assert_eq!(v[1], base_val_1);
                assert_eq!(p.len(), 1);
            }
            QueryOpening::Extension { .. } => panic!("Expected Base variant"),
        }

        // Test Extension variant

        // Create extension field values
        // Extension field values are created from base field using From trait
        let ext_val_0 = EF::from_u64(3);
        let ext_val_1 = EF::from_u64(4);
        let ext_values = vec![ext_val_0, ext_val_1];

        // Construct Extension variant
        let ext_opening: QueryOpening<F, EF, DIGEST_ELEMS> = QueryOpening::Extension {
            values: ext_values,
            proof,
        };

        // Verify it's the correct variant
        match ext_opening {
            QueryOpening::Extension {
                values: v,
                proof: p,
            } => {
                assert_eq!(v.len(), 2);
                assert_eq!(v[0], ext_val_0);
                assert_eq!(v[1], ext_val_1);
                assert_eq!(p.len(), 1);
            }
            QueryOpening::Base { .. } => panic!("Expected Extension variant"),
        }
    }
}
