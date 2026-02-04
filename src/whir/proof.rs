use alloc::vec::Vec;
use core::array;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field};
use serde::{Deserialize, Serialize};

use crate::{parameters::ProtocolParameters, poly::evals::EvaluationsList};

/// Complete WHIR proof
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound(
    serialize = "F: Serialize, EF: Serialize, W: Serialize, [W; DIGEST_ELEMS]: Serialize",
    deserialize = "F: Deserialize<'de>, EF: Deserialize<'de>, W: Deserialize<'de>, [W; DIGEST_ELEMS]: Deserialize<'de>"
))]
// TODO: add initial claims?
pub struct WhirProof<F, EF, W, const DIGEST_ELEMS: usize> {
    /// Initial polynomial commitment (Merkle root)
    pub initial_commitment: [W; DIGEST_ELEMS],

    /// Initial OOD evaluations
    pub initial_ood_answers: Vec<EF>,

    /// Initial phase data - captures the protocol variant
    pub initial_sumcheck: SumcheckData<F, EF>,

    /// One proof per WHIR round
    pub rounds: Vec<WhirRoundProof<F, EF, W, DIGEST_ELEMS>>,

    /// Final polynomial evaluations
    pub final_poly: Option<EvaluationsList<EF>>,

    /// Final round PoW witness
    pub final_pow_witness: F,

    /// Final round query openings
    pub final_queries: Vec<QueryOpening<F, EF, W, DIGEST_ELEMS>>,

    /// Final sumcheck (if final_sumcheck_rounds > 0)
    pub final_sumcheck: Option<SumcheckData<F, EF>>,
}

impl<F: Default, EF: Default, W: Default, const DIGEST_ELEMS: usize> Default
    for WhirProof<F, EF, W, DIGEST_ELEMS>
{
    fn default() -> Self {
        Self {
            initial_commitment: array::from_fn(|_| W::default()),
            initial_ood_answers: Vec::new(),
            initial_sumcheck: SumcheckData::default(),
            rounds: Vec::new(),
            final_poly: None,
            final_pow_witness: F::default(),
            final_queries: Vec::new(),
            final_sumcheck: None,
        }
    }
}

/// Data for a single WHIR round
///
/// The type parameter `W` is the digest element type (same as in `WhirProof`)
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound(
    serialize = "F: Serialize, EF: Serialize, W: Serialize, [W; DIGEST_ELEMS]: Serialize",
    deserialize = "F: Deserialize<'de>, EF: Deserialize<'de>, W: Deserialize<'de>, [W; DIGEST_ELEMS]: Deserialize<'de>"
))]
pub struct WhirRoundProof<F, EF, W, const DIGEST_ELEMS: usize> {
    /// Round commitment (Merkle root)
    pub commitment: [W; DIGEST_ELEMS],

    /// OOD evaluations for this round
    pub ood_answers: Vec<EF>,

    /// PoW witness after commitment
    pub pow_witness: F,

    /// STIR query openings
    pub queries: Vec<QueryOpening<F, EF, W, DIGEST_ELEMS>>,

    /// Sumcheck data for this round
    pub sumcheck: SumcheckData<F, EF>,
}

impl<F: Default, EF: Default, W: Default, const DIGEST_ELEMS: usize> Default
    for WhirRoundProof<F, EF, W, DIGEST_ELEMS>
{
    fn default() -> Self {
        Self {
            commitment: array::from_fn(|_| W::default()),
            ood_answers: Vec::new(),
            pow_witness: F::default(),
            queries: Vec::new(),
            sumcheck: SumcheckData::default(),
        }
    }
}

/// Query opening
///
/// The type parameter `W` is the digest element type (same as in `WhirProof`)
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(
    bound(
        serialize = "F: Serialize, EF: Serialize, W: Serialize, [W; DIGEST_ELEMS]: Serialize",
        deserialize = "F: Deserialize<'de>, EF: Deserialize<'de>, W: Deserialize<'de>, [W; DIGEST_ELEMS]: Deserialize<'de>"
    ),
    tag = "type"
)]
pub enum QueryOpening<F, EF, W, const DIGEST_ELEMS: usize> {
    /// Base field query (round_index == 0)
    #[serde(rename = "base")]
    Base {
        /// Merkle leaf values in F
        values: Vec<F>,
        /// Merkle authentication path
        proof: Vec<[W; DIGEST_ELEMS]>,
    },
    /// Extension field query (round_index > 0)
    #[serde(rename = "extension")]
    Extension {
        /// Merkle leaf values in EF
        values: Vec<EF>,
        /// Merkle authentication path
        proof: Vec<[W; DIGEST_ELEMS]>,
    },
}

/// Sumcheck polynomial data
///
/// Stores the polynomial evaluations for sumcheck rounds in a compact format.
/// Each round stores `[h(0), h(2)]` where `h(1)` is derived as `claimed_sum - h(0)`.
#[derive(Default, Serialize, Deserialize, Clone, Debug)]
pub struct SumcheckData<F, EF> {
    /// Polynomial evaluations for each sumcheck round
    ///
    /// Each entry is `[h(0), h(2)]` - the evaluations at 0 and 2
    ///
    /// `h(1)` is derived as `claimed_sum - h(0)` by the verifier
    ///
    /// Length: folding_factor
    pub polynomial_evaluations: Vec<[EF; 2]>,

    /// PoW witnesses for each sumcheck round
    /// Length: folding_factor
    pub pow_witnesses: Vec<F>,
}

impl<F, EF> SumcheckData<F, EF> {
    /// Appends a proof-of-work witness.
    pub fn push_pow_witness(&mut self, witness: F) {
        self.pow_witnesses.push(witness);
    }

    /// Commits polynomial coefficients to the transcript and returns a challenge.
    ///
    /// This helper function handles the Fiat-Shamir interaction for a sumcheck round.
    ///
    /// # Arguments
    ///
    /// * `challenger` - Fiat-Shamir transcript.
    /// * `c0` - Constant coefficient `h(0)`.
    /// * `c2` - Quadratic coefficient.
    /// * `pow_bits` - PoW difficulty (0 to skip grinding).
    ///
    /// # Returns
    ///
    /// The sampled challenge `r \in EF`.
    pub fn observe_and_sample<Challenger, BF>(
        &mut self,
        challenger: &mut Challenger,
        c0: EF,
        c2: EF,
        pow_bits: usize,
    ) -> EF
    where
        BF: Field,
        EF: ExtensionField<BF>,
        F: Clone,
        Challenger: FieldChallenger<BF> + GrindingChallenger<Witness = F>,
    {
        // Record the polynomial coefficients in the proof.
        self.polynomial_evaluations.push([c0, c2]);

        // Absorb coefficients into the transcript.
        //
        // Note: We only send (c_0, c_2). The verifier derives c_1 from the sum constraint.
        challenger.observe_algebra_slice(&[c0, c2]);

        // Optional proof-of-work to increase prover cost.
        //
        // This makes it expensive for a malicious prover to "mine" favorable challenges.
        if pow_bits > 0 {
            self.push_pow_witness(challenger.grind(pow_bits));
        }

        // Sample the verifier's challenge for this round.
        challenger.sample_algebra_element()
    }
}

impl<F: Default, EF: Default, W: Default, const DIGEST_ELEMS: usize>
    WhirProof<F, EF, W, DIGEST_ELEMS>
{
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
            initial_commitment: array::from_fn(|_| W::default()),
            initial_ood_answers: Vec::new(),
            initial_sumcheck: SumcheckData::default(),
            rounds: (0..num_rounds).map(|_| WhirRoundProof::default()).collect(),
            final_poly: None,
            final_pow_witness: F::default(),
            final_queries: Vec::with_capacity(num_queries),
            final_sumcheck: None,
        }
    }
}

impl<F: Clone, EF, W, const DIGEST_ELEMS: usize> WhirProof<F, EF, W, DIGEST_ELEMS> {
    /// Extract the PoW witness after the commitment at the given round index
    ///
    /// Returns the PoW witness from the round at the given index.
    /// The PoW witness is stored in proof.rounds[round_index].pow_witness.
    pub fn get_pow_after_commitment(&self, round_index: usize) -> Option<F> {
        self.rounds
            .get(round_index)
            .map(|round| round.pow_witness.clone())
    }

    /// Stores sumcheck data at a specific round index.
    ///
    /// # Parameters
    /// - `data`: The sumcheck data to store
    /// - `round_index`: The round index to store the data at
    ///
    /// # Panics
    /// Panics if `round_index` is out of bounds.
    pub fn set_sumcheck_data_at(&mut self, data: SumcheckData<F, EF>, round_index: usize) {
        self.rounds[round_index].sumcheck = data;
    }

    /// Stores sumcheck data in the final sumcheck field.
    ///
    /// # Parameters
    /// - `data`: The sumcheck data to store
    pub fn set_final_sumcheck_data(&mut self, data: SumcheckData<F, EF>) {
        self.final_sumcheck = Some(data);
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::SeedableRng;

    use super::*;
    use crate::parameters::{FoldingFactor, errors::SecurityAssumption};

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

    /// Type alias for the challenger used in observe_and_sample tests.
    type TestChallenger = DuplexChallenger<F, Perm, 16, 8>;

    /// Digest size for Merkle tree commitments
    const DIGEST_ELEMS: usize = 8;

    /// Helper function to create minimal protocol parameters for testing
    ///
    /// This creates a `ProtocolParameters` instance with specified configuration
    /// for testing different proof initialization scenarios.
    ///
    /// # Parameters
    /// - `folding_factor`: The folding strategy for the protocol
    ///
    /// # Returns
    /// A `ProtocolParameters` instance configured for testing
    fn create_test_params(folding_factor: FoldingFactor) -> ProtocolParameters<MyHash, MyCompress> {
        // Create the permutation for hash and compress
        let perm = Perm::new_from_rng_128(&mut rand::rngs::SmallRng::seed_from_u64(42));

        ProtocolParameters {
            starting_log_inv_rate: 2,
            rs_domain_initial_reduction_factor: 1,
            folding_factor,
            soundness_type: SecurityAssumption::UniqueDecoding,
            security_level: 100,
            pow_bits: 10,
            merkle_hash: PaddingFreeSponge::new(perm.clone()),
            merkle_compress: TruncatedPermutation::new(perm),
        }
    }

    #[test]
    fn test_whir_proof_from_params() {
        let folding_factor_value = 4;
        let folding_factor = FoldingFactor::Constant(folding_factor_value);

        // Use 16 variables for testing
        let num_variables = 16;

        // Create protocol parameters
        let params = create_test_params(folding_factor);

        // Create proof structure from parameters
        let proof: WhirProof<F, EF, F, DIGEST_ELEMS> =
            WhirProof::from_protocol_parameters(&params, num_variables);

        // Verify that initial_phase is WithStatement variant
        // match proof.initial_phase {
        //     InitialPhase::WithStatement { data } => {
        //         // sumcheck should have empty polynomial_evaluations
        //         assert_eq!(data.polynomial_evaluations.len(), 0);
        //         // sumcheck should have empty PoW witnesses
        //         assert!(data.pow_witnesses.is_empty());
        //     }
        //     InitialPhase::WithoutStatement { .. } => {
        //         panic!("Expected WithStatement variant, not WithStatementSkip")
        //     }
        // }

        // Verify rounds length
        // Formula: ((num_variables - MAX_NUM_VARIABLES_TO_SEND_COEFFS) / folding_factor) - 1
        // MAX_NUM_VARIABLES_TO_SEND_COEFFS = 6
        // For 16 variables with folding_factor 4:
        //   (16 - 6).div_ceil(4) - 1 = 10.div_ceil(4) - 1 = 3 - 1 = 2
        let expected_rounds = 2;
        assert_eq!(proof.rounds.len(), expected_rounds);
    }

    #[test]
    fn test_whir_proof_from_params_without_initial_statement() {
        // Declare test parameters explicitly

        // Folding factor doesn't matter for initial_phase when WithoutStatement
        let folding_factor_value = 6;
        let folding_factor = FoldingFactor::Constant(folding_factor_value);

        // Configure without initial statement
        // let sumcheck_strategy = SumcheckStrategy::WithoutStatement;

        // Use 18 variables for testing
        let num_variables = 18;

        // Create protocol parameters without initial statement
        let params = create_test_params(folding_factor);

        // Create proof structure from parameters
        let proof: WhirProof<F, EF, F, DIGEST_ELEMS> =
            WhirProof::from_protocol_parameters(&params, num_variables);

        // Verify that initial_phase is WithoutStatement variant
        // This is because initial_phase_config = WithoutStatement
        // match proof.initial_phase {
        //     InitialPhase::WithoutStatement { pow_witness } => {
        //         // pow_witness should be default (not populated yet)
        //         assert_eq!(pow_witness, F::default());
        //     }
        //     InitialPhase::WithStatement { .. } => {
        //         panic!("Expected WithoutStatement variant")
        //     }
        // }

        // Verify rounds length
        // Formula: ((num_variables - MAX_NUM_VARIABLES_TO_SEND_COEFFS) / folding_factor) - 1
        // MAX_NUM_VARIABLES_TO_SEND_COEFFS = 6
        // For 18 variables with folding_factor 6:
        //   (18 - 6).div_ceil(6) - 1 = 12.div_ceil(6) - 1 = 2 - 1 = 1
        let expected_rounds = 1;
        assert_eq!(proof.rounds.len(), expected_rounds);
    }

    #[test]
    fn test_get_pow_after_commitment_with_witness() {
        // Create an explicit PoW witness value for testing
        let pow_witness_value = F::from_u64(42);

        // Create a proof with one round containing a PoW witness
        let proof: WhirProof<F, EF, F, DIGEST_ELEMS> = WhirProof {
            initial_commitment: array::from_fn(|_| F::default()),
            initial_ood_answers: Vec::new(),
            initial_sumcheck: SumcheckData::default(),
            rounds: vec![WhirRoundProof {
                commitment: array::from_fn(|_| F::default()),
                ood_answers: Vec::new(),
                pow_witness: pow_witness_value,
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
    fn test_get_pow_after_commitment_invalid_round() {
        // Create a proof with one round
        let proof: WhirProof<F, EF, F, DIGEST_ELEMS> = WhirProof {
            initial_commitment: array::from_fn(|_| F::default()),
            initial_ood_answers: Vec::new(),
            initial_sumcheck: SumcheckData::default(),
            rounds: vec![WhirRoundProof {
                commitment: array::from_fn(|_| F::default()),
                ood_answers: Vec::new(),
                pow_witness: F::from_u64(42),
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
    fn test_whir_round_proof_default() {
        // Create a default WhirRoundProof
        let round: WhirRoundProof<F, EF, F, DIGEST_ELEMS> = WhirRoundProof::default();

        // Verify commitment is array of default F values
        assert_eq!(round.commitment.len(), DIGEST_ELEMS);
        for elem in round.commitment {
            assert_eq!(elem, F::default());
        }

        // Verify ood_answers is empty
        assert_eq!(round.ood_answers.len(), 0);

        // Verify pow_witness is default
        assert_eq!(round.pow_witness, F::default());

        // Verify queries is empty
        assert_eq!(round.queries.len(), 0);

        // Verify sumcheck has default values
        assert_eq!(round.sumcheck.polynomial_evaluations.len(), 0);
        assert!(round.sumcheck.pow_witnesses.is_empty());
    }

    #[test]
    fn test_sumcheck_data_default() {
        // Create a default SumcheckData
        let sumcheck: SumcheckData<F, EF> = SumcheckData::default();

        // Verify polynomial_evaluations is empty
        assert_eq!(sumcheck.polynomial_evaluations.len(), 0);

        // Verify pow_witnesses is empty
        assert!(sumcheck.pow_witnesses.is_empty());
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
        let base_opening: QueryOpening<F, EF, F, DIGEST_ELEMS> = QueryOpening::Base {
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
        let ext_opening: QueryOpening<F, EF, F, DIGEST_ELEMS> = QueryOpening::Extension {
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

    #[test]
    fn test_push_pow_witness() {
        let mut sumcheck: SumcheckData<F, EF> = SumcheckData::default();

        // First push
        let witness1 = F::from_u64(42);
        sumcheck.push_pow_witness(witness1);

        assert_eq!(sumcheck.pow_witnesses.len(), 1);
        assert_eq!(sumcheck.pow_witnesses[0], witness1);

        // Second push should append to existing vector
        let witness2 = F::from_u64(123);
        sumcheck.push_pow_witness(witness2);

        assert_eq!(sumcheck.pow_witnesses.len(), 2);
        assert_eq!(sumcheck.pow_witnesses[1], witness2);
    }

    #[test]
    fn test_set_final_sumcheck_data() {
        // Create a proof with no rounds
        let mut proof: WhirProof<F, EF, F, DIGEST_ELEMS> = WhirProof {
            initial_commitment: array::from_fn(|_| F::default()),
            initial_ood_answers: Vec::new(),
            initial_sumcheck: SumcheckData::default(),
            rounds: Vec::new(),
            final_poly: None,
            final_pow_witness: F::default(),
            final_queries: Vec::new(),
            final_sumcheck: None,
        };

        // Verify final_sumcheck is None initially
        assert!(proof.final_sumcheck.is_none());

        // Create sumcheck data with a distinguishable value
        let mut data: SumcheckData<F, EF> = SumcheckData::default();
        data.push_pow_witness(F::from_u64(999));

        // Set as final
        proof.set_final_sumcheck_data(data);

        // Verify it was stored in final_sumcheck
        assert!(proof.final_sumcheck.is_some());
        let stored = proof.final_sumcheck.as_ref().unwrap();
        assert_eq!(stored.pow_witnesses[0], F::from_u64(999));
    }

    #[test]
    fn test_set_sumcheck_data_at_round() {
        // Create a proof with two rounds
        let mut proof: WhirProof<F, EF, F, DIGEST_ELEMS> = WhirProof {
            initial_commitment: array::from_fn(|_| F::default()),
            initial_ood_answers: Vec::new(),
            initial_sumcheck: SumcheckData::default(),
            rounds: vec![WhirRoundProof::default(), WhirRoundProof::default()],
            final_poly: None,
            final_pow_witness: F::default(),
            final_queries: Vec::new(),
            final_sumcheck: None,
        };

        // Verify rounds' sumcheck is empty initially
        assert!(proof.rounds[0].sumcheck.pow_witnesses.is_empty());
        assert!(proof.rounds[1].sumcheck.pow_witnesses.is_empty());

        // Create sumcheck data with a distinguishable value for round 0
        let mut data0: SumcheckData<F, EF> = SumcheckData::default();
        data0.push_pow_witness(F::from_u64(777));
        proof.set_sumcheck_data_at(data0, 0);

        // Create sumcheck data with a distinguishable value for round 1
        let mut data1: SumcheckData<F, EF> = SumcheckData::default();
        data1.push_pow_witness(F::from_u64(888));
        proof.set_sumcheck_data_at(data1, 1);

        // Verify it was stored in the correct rounds
        assert_eq!(proof.rounds[0].sumcheck.pow_witnesses[0], F::from_u64(777));
        assert_eq!(proof.rounds[1].sumcheck.pow_witnesses[0], F::from_u64(888));

        // Verify final_sumcheck is still None
        assert!(proof.final_sumcheck.is_none());
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_set_sumcheck_data_at_no_rounds_panics() {
        // Create a proof with no rounds
        let mut proof: WhirProof<F, EF, F, DIGEST_ELEMS> = WhirProof {
            initial_commitment: array::from_fn(|_| F::default()),
            initial_ood_answers: Vec::new(),
            initial_sumcheck: SumcheckData::default(),
            rounds: Vec::new(),
            final_poly: None,
            final_pow_witness: F::default(),
            final_queries: Vec::new(),
            final_sumcheck: None,
        };

        // Try to set sumcheck data at index 0 with no rounds - should panic
        proof.set_sumcheck_data_at(SumcheckData::default(), 0);
    }

    /// Creates a fresh challenger for testing.
    ///
    /// The challenger is seeded deterministically so tests are reproducible.
    fn create_test_challenger() -> TestChallenger {
        let perm = Perm::new_from_rng_128(&mut rand::rngs::SmallRng::seed_from_u64(42));
        DuplexChallenger::new(perm)
    }

    #[test]
    fn test_observe_and_sample_records_coefficients() {
        // The method should push [c0, c2] to polynomial_evaluations.
        //
        // polynomial_evaluations stores the sumcheck polynomial coefficients
        // for each round: [h(0), h(2)] where h(1) is derived by the verifier.
        let c0 = EF::from_u64(5);
        let c2 = EF::from_u64(7);

        let mut sumcheck: SumcheckData<F, EF> = SumcheckData::default();
        let mut challenger = create_test_challenger();

        // polynomial_evaluations should be empty initially
        assert!(sumcheck.polynomial_evaluations.is_empty());

        // Call observe_and_sample with pow_bits = 0 (no grinding)
        let pow_bits = 0;
        let _r = sumcheck.observe_and_sample::<_, F>(&mut challenger, c0, c2, pow_bits);

        // polynomial_evaluations should now have one entry: [c0, c2]
        assert_eq!(sumcheck.polynomial_evaluations.len(), 1);
        assert_eq!(sumcheck.polynomial_evaluations[0][0], c0);
        assert_eq!(sumcheck.polynomial_evaluations[0][1], c2);
    }

    #[test]
    fn test_observe_and_sample_multiple_rounds() {
        // Multiple calls should accumulate coefficients in order.
        //
        // Round 0: push [c0_0, c2_0]
        // Round 1: push [c0_1, c2_1]
        // Round 2: push [c0_2, c2_2]
        let c0_0 = EF::from_u64(1);
        let c2_0 = EF::from_u64(2);
        let c0_1 = EF::from_u64(3);
        let c2_1 = EF::from_u64(4);
        let c0_2 = EF::from_u64(5);
        let c2_2 = EF::from_u64(6);

        let mut sumcheck: SumcheckData<F, EF> = SumcheckData::default();
        let mut challenger = create_test_challenger();
        let pow_bits = 0;

        // Round 0
        let _r0 = sumcheck.observe_and_sample::<_, F>(&mut challenger, c0_0, c2_0, pow_bits);
        assert_eq!(sumcheck.polynomial_evaluations.len(), 1);

        // Round 1
        let _r1 = sumcheck.observe_and_sample::<_, F>(&mut challenger, c0_1, c2_1, pow_bits);
        assert_eq!(sumcheck.polynomial_evaluations.len(), 2);

        // Round 2
        let _r2 = sumcheck.observe_and_sample::<_, F>(&mut challenger, c0_2, c2_2, pow_bits);
        assert_eq!(sumcheck.polynomial_evaluations.len(), 3);

        // Verify all stored coefficients match input order
        assert_eq!(sumcheck.polynomial_evaluations[0], [c0_0, c2_0]);
        assert_eq!(sumcheck.polynomial_evaluations[1], [c0_1, c2_1]);
        assert_eq!(sumcheck.polynomial_evaluations[2], [c0_2, c2_2]);
    }

    #[test]
    fn test_observe_and_sample_without_pow() {
        // When pow_bits = 0, no PoW witness should be recorded.
        //
        // The method skips the grinding step when pow_bits is zero,
        // so pow_witnesses should remain empty.
        let c0 = EF::from_u64(10);
        let c2 = EF::from_u64(20);

        let mut sumcheck: SumcheckData<F, EF> = SumcheckData::default();
        let mut challenger = create_test_challenger();

        // pow_witnesses should be empty initially
        assert!(sumcheck.pow_witnesses.is_empty());

        // Call with pow_bits = 0
        let pow_bits = 0;
        let _r = sumcheck.observe_and_sample::<_, F>(&mut challenger, c0, c2, pow_bits);

        // pow_witnesses should still be empty (no grinding performed)
        assert!(sumcheck.pow_witnesses.is_empty());
    }

    #[test]
    fn test_observe_and_sample_with_pow() {
        // When pow_bits > 0, a PoW witness should be recorded.
        //
        // The method calls challenger.grind(pow_bits) and pushes
        // the resulting witness to pow_witnesses.
        let c0 = EF::from_u64(10);
        let c2 = EF::from_u64(20);

        let mut sumcheck: SumcheckData<F, EF> = SumcheckData::default();
        let mut challenger = create_test_challenger();

        // pow_witnesses should be empty initially
        assert!(sumcheck.pow_witnesses.is_empty());

        // Call with pow_bits = 1 (minimal PoW)
        let pow_bits = 1;
        let _r = sumcheck.observe_and_sample::<_, F>(&mut challenger, c0, c2, pow_bits);

        // pow_witnesses should now have one entry
        assert_eq!(sumcheck.pow_witnesses.len(), 1);
    }

    #[test]
    fn test_observe_and_sample_pow_accumulates() {
        // Multiple rounds with PoW should accumulate witnesses.
        //
        // Each call with pow_bits > 0 should add one witness.
        let c0 = EF::from_u64(1);
        let c2 = EF::from_u64(2);

        let mut sumcheck: SumcheckData<F, EF> = SumcheckData::default();
        let mut challenger = create_test_challenger();
        let pow_bits = 1;

        // Three rounds with PoW
        let _r0 = sumcheck.observe_and_sample::<_, F>(&mut challenger, c0, c2, pow_bits);
        let _r1 = sumcheck.observe_and_sample::<_, F>(&mut challenger, c0, c2, pow_bits);
        let _r2 = sumcheck.observe_and_sample::<_, F>(&mut challenger, c0, c2, pow_bits);

        // Should have 3 witnesses
        assert_eq!(sumcheck.pow_witnesses.len(), 3);
        // And 3 polynomial evaluations
        assert_eq!(sumcheck.polynomial_evaluations.len(), 3);
    }

    #[test]
    fn test_observe_and_sample_deterministic_challenge() {
        // Fiat-Shamir property: same inputs produce same challenge.
        //
        // Two challengers with the same initial state, observing the same
        // coefficients, should sample the same challenge.
        let c0 = EF::from_u64(42);
        let c2 = EF::from_u64(99);
        let pow_bits = 0;

        // First run
        let mut sumcheck1: SumcheckData<F, EF> = SumcheckData::default();
        let mut challenger1 = create_test_challenger();
        let r1 = sumcheck1.observe_and_sample::<_, F>(&mut challenger1, c0, c2, pow_bits);

        // Second run with fresh but identically-seeded challenger
        let mut sumcheck2: SumcheckData<F, EF> = SumcheckData::default();
        let mut challenger2 = create_test_challenger();
        let r2 = sumcheck2.observe_and_sample::<_, F>(&mut challenger2, c0, c2, pow_bits);

        // Challenges should be identical
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_observe_and_sample_challenge_depends_on_history() {
        // The challenge at round i depends on all previous observations.
        //
        // Two sequences with different history should produce different
        // challenges even if the final round has the same coefficients.
        let c0 = EF::from_u64(100);
        let c2 = EF::from_u64(200);
        let pow_bits = 0;

        // Sequence A: observe once, then observe (c0, c2)
        let mut sumcheck_a: SumcheckData<F, EF> = SumcheckData::default();
        let mut challenger_a = create_test_challenger();
        let _r0_a =
            sumcheck_a.observe_and_sample::<_, F>(&mut challenger_a, EF::ONE, EF::TWO, pow_bits);
        let r1_a = sumcheck_a.observe_and_sample::<_, F>(&mut challenger_a, c0, c2, pow_bits);

        // Sequence B: directly observe (c0, c2) without prior round
        let mut sumcheck_b: SumcheckData<F, EF> = SumcheckData::default();
        let mut challenger_b = create_test_challenger();
        let r_b = sumcheck_b.observe_and_sample::<_, F>(&mut challenger_b, c0, c2, pow_bits);

        // Challenges should differ due to different transcript history
        assert_ne!(r1_a, r_b);
    }

    #[test]
    fn test_observe_and_sample_returns_extension_field_element() {
        // The returned challenge should be a valid extension field element.
        //
        // This is verified implicitly by the type system, but we can also
        // check that it's not trivially zero (with high probability).
        let c0 = EF::from_u64(7);
        let c2 = EF::from_u64(11);
        let pow_bits = 0;

        let mut sumcheck: SumcheckData<F, EF> = SumcheckData::default();
        let mut challenger = create_test_challenger();

        let r: EF = sumcheck.observe_and_sample::<_, F>(&mut challenger, c0, c2, pow_bits);

        // The challenge should (with overwhelming probability) be non-zero
        assert_ne!(r, EF::ZERO);
    }
}
