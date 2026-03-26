use alloc::vec::Vec;
use core::fmt;

use p3_commit::Mmcs;
use p3_field::Field;
use p3_multilinear_util::{evals::EvaluationsList, multilinear::MultilinearPoint};
use serde::{Deserialize, Serialize};

pub use crate::sumcheck::SumcheckData;
use crate::{constraints::statement::EqStatement, parameters::ProtocolParameters};

/// Complete WHIR proof
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound(
    serialize = "F: Serialize, EF: Serialize, MT::Commitment: Serialize, MT::Proof: Serialize",
    deserialize = "F: Deserialize<'de>, EF: Deserialize<'de>, MT::Commitment: Deserialize<'de>, MT::Proof: Deserialize<'de>"
))]
// TODO: add initial claims?
pub struct WhirProof<F: Send + Sync + Clone, EF, MT: Mmcs<F>> {
    /// Initial polynomial commitment (Merkle root)
    pub initial_commitment: Option<MT::Commitment>,

    /// Initial OOD evaluations
    pub initial_ood_answers: Vec<EF>,

    /// Initial phase data - captures the protocol variant
    pub initial_sumcheck: SumcheckData<F, EF>,

    /// One proof per WHIR round
    pub rounds: Vec<WhirRoundProof<F, EF, MT>>,

    /// Final polynomial evaluations
    pub final_poly: Option<EvaluationsList<EF>>,

    /// Final round PoW witness
    pub final_pow_witness: F,

    /// Final round query openings
    pub final_queries: Vec<QueryOpening<F, EF, MT::Proof>>,

    /// Final sumcheck (if final_sumcheck_rounds > 0)
    pub final_sumcheck: Option<SumcheckData<F, EF>>,
}

impl<F: Default + Send + Sync + Clone, EF: Default, MT: Mmcs<F>> Default for WhirProof<F, EF, MT> {
    fn default() -> Self {
        Self {
            initial_commitment: None,
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
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound(
    serialize = "F: Serialize, EF: Serialize, MT::Commitment: Serialize, MT::Proof: Serialize",
    deserialize = "F: Deserialize<'de>, EF: Deserialize<'de>, MT::Commitment: Deserialize<'de>, MT::Proof: Deserialize<'de>"
))]
pub struct WhirRoundProof<F: Send + Sync + Clone, EF, MT: Mmcs<F>> {
    /// Round commitment (Merkle root)
    pub commitment: Option<MT::Commitment>,

    /// OOD evaluations for this round
    pub ood_answers: Vec<EF>,

    /// PoW witness after commitment
    pub pow_witness: F,

    /// STIR query openings
    pub queries: Vec<QueryOpening<F, EF, MT::Proof>>,

    /// Sumcheck data for this round
    pub sumcheck: SumcheckData<F, EF>,
}

impl<F: Default + Send + Sync + Clone, EF: Default, MT: Mmcs<F>> Default
    for WhirRoundProof<F, EF, MT>
{
    fn default() -> Self {
        Self {
            commitment: None,
            ood_answers: Vec::new(),
            pow_witness: F::default(),
            queries: Vec::new(),
            sumcheck: SumcheckData::default(),
        }
    }
}

/// Query opening
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(
    bound(
        serialize = "F: Serialize, EF: Serialize, Proof: Serialize",
        deserialize = "F: Deserialize<'de>, EF: Deserialize<'de>, Proof: Deserialize<'de>"
    ),
    tag = "type"
)]
pub enum QueryOpening<F, EF, Proof> {
    /// Base field query (round_index == 0)
    #[serde(rename = "base")]
    Base {
        /// Merkle leaf values in F
        values: Vec<F>,
        /// Merkle authentication path
        proof: Proof,
    },
    /// Extension field query (round_index > 0)
    #[serde(rename = "extension")]
    Extension {
        /// Merkle leaf values in EF
        values: Vec<EF>,
        /// Merkle authentication path
        proof: Proof,
    },
    /// Batch opening: opens both polynomial trees at the same query index
    #[serde(rename = "batch")]
    Batch {
        /// Merkle leaf values from first tree (f_a)
        values_a: Vec<F>,
        /// Merkle authentication path for first tree
        proof_a: Proof,
        /// Merkle leaf values from second tree (f_b)
        values_b: Vec<F>,
        /// Merkle authentication path for second tree
        proof_b: Proof,
    },
}

/// Batch opening proof for two polynomials.
///
/// Wraps batch-specific data (selector sumcheck, OOD answers for both polynomials)
/// alongside the inner WHIR proof that operates on the folded polynomial.
#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "F: Serialize, EF: Serialize, MT::Commitment: Serialize, MT::Proof: Serialize",
    deserialize = "F: Deserialize<'de>, EF: Deserialize<'de>, MT::Commitment: Deserialize<'de>, MT::Proof: Deserialize<'de>"
))]
pub struct BatchWhirProof<F: Send + Sync + Clone, EF, MT: Mmcs<F>> {
    /// Commitment to first polynomial (f_a) — Merkle root
    pub commitment_a: Option<MT::Commitment>,

    /// Commitment to second polynomial (f_b) — Merkle root
    pub commitment_b: Option<MT::Commitment>,

    /// OOD answers for both polynomials: [ood_a, ood_b]
    pub initial_ood_answers: [Vec<EF>; 2],

    /// Selector sumcheck data: stores [c0, c2] for h(X)
    pub selector_sumcheck: SumcheckData<F, EF>,

    /// Inner WHIR proof on the folded polynomial g = r_0·f_a + (1-r_0)·f_b
    pub inner_proof: WhirProof<F, EF, MT>,
}

impl<F: Send + Sync + Clone, EF, MT: Mmcs<F>> fmt::Debug for BatchWhirProof<F, EF, MT> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BatchWhirProof").finish_non_exhaustive()
    }
}

/// Extracts a single evaluation constraint (point, value) from an EqStatement.
///
/// Panics if the statement does not contain exactly one constraint.
pub(crate) fn single_constraint<EF: Field>(
    statement: &EqStatement<EF>,
) -> (MultilinearPoint<EF>, EF) {
    assert_eq!(
        statement.len(),
        1,
        "Batch opening expects exactly one evaluation claim per polynomial"
    );
    let (point, &value) = statement.iter().next().unwrap();
    (point.clone(), value)
}

/// Folds OOD constraints from two polynomials using the selector challenge.
///
/// Both polynomials must have been evaluated at the same OOD challenge points.
/// The folded value is: `r_0 * ood_a + (1 - r_0) * ood_b`.
pub(crate) fn fold_ood_constraints<EF: Field>(
    ood_a: &EqStatement<EF>,
    ood_b: &EqStatement<EF>,
    r_0: EF,
) -> EqStatement<EF> {
    assert_eq!(ood_a.len(), ood_b.len());
    let num_variables = ood_a.num_variables();
    let mut folded = EqStatement::initialize(num_variables);

    let one_minus_r0 = EF::ONE - r_0;
    for ((point_a, &v_a), (point_b, &v_b)) in ood_a.iter().zip(ood_b.iter()) {
        debug_assert_eq!(point_a, point_b, "OOD points must match");
        let folded_value = r_0 * v_a + one_minus_r0 * v_b;
        folded.add_evaluated_constraint(point_a.clone(), folded_value);
    }

    folded
}

impl<F: Default + Send + Sync + Clone, EF: Default, MT: Mmcs<F>> WhirProof<F, EF, MT> {
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
    pub fn from_protocol_parameters(params: &ProtocolParameters<MT>, num_variables: usize) -> Self {
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
            initial_commitment: None,
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

impl<F: Clone + Send + Sync + Default, EF, MT: Mmcs<F>> WhirProof<F, EF, MT> {
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
    use core::array;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_commit::Mmcs;
    use p3_field::{Field, PrimeCharacteristicRing, extension::BinomialExtensionField};
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::SeedableRng;

    use super::*;
    use crate::parameters::{FoldingFactor, SecurityAssumption};

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
    type PackedF = <F as Field>::Packing;
    type MyMmcs = MerkleTreeMmcs<PackedF, PackedF, MyHash, MyCompress, 2, DIGEST_ELEMS>;

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
    fn create_test_params(folding_factor: FoldingFactor) -> ProtocolParameters<MyMmcs> {
        // Create the permutation for hash and compress
        let perm = Perm::new_from_rng_128(&mut rand::rngs::SmallRng::seed_from_u64(42));
        let mmcs = MyMmcs::new(
            PaddingFreeSponge::new(perm.clone()),
            TruncatedPermutation::new(perm),
            0,
        );

        ProtocolParameters {
            starting_log_inv_rate: 2,
            rs_domain_initial_reduction_factor: 1,
            folding_factor,
            soundness_type: SecurityAssumption::UniqueDecoding,
            security_level: 100,
            pow_bits: 10,
            mmcs,
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
        let proof: WhirProof<F, EF, MyMmcs> =
            WhirProof::from_protocol_parameters(&params, num_variables);

        // Verify that initial_sumcheck has empty polynomial_evaluations and PoW witnesses
        assert_eq!(proof.initial_sumcheck.polynomial_evaluations.len(), 0);
        assert!(proof.initial_sumcheck.pow_witnesses.is_empty());

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

        // Use 18 variables for testing
        let num_variables = 18;

        // Create protocol parameters without initial statement
        let params = create_test_params(folding_factor);

        // Create proof structure from parameters
        let proof: WhirProof<F, EF, MyMmcs> =
            WhirProof::from_protocol_parameters(&params, num_variables);

        // Verify that initial_sumcheck is empty (without initial statement)
        // The polynomial_evaluations and pow_witnesses should be empty
        assert!(proof.initial_sumcheck.polynomial_evaluations.is_empty());
        assert!(proof.initial_sumcheck.pow_witnesses.is_empty());

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
        let proof: WhirProof<F, EF, MyMmcs> = WhirProof {
            initial_commitment: None,
            initial_ood_answers: Vec::new(),
            initial_sumcheck: SumcheckData::default(),
            rounds: vec![WhirRoundProof {
                commitment: None,
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
        let proof: WhirProof<F, EF, MyMmcs> = WhirProof {
            initial_commitment: None,
            initial_ood_answers: Vec::new(),
            initial_sumcheck: SumcheckData::default(),
            rounds: vec![WhirRoundProof {
                commitment: None,
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
        let round: WhirRoundProof<F, EF, MyMmcs> = WhirRoundProof::default();

        // Verify default round has no commitment yet
        assert!(round.commitment.is_none());

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
        let base_opening: QueryOpening<F, EF, <MyMmcs as Mmcs<F>>::Proof> = QueryOpening::Base {
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
            QueryOpening::Extension { .. } | QueryOpening::Batch { .. } => {
                panic!("Expected Base variant")
            }
        }

        // Test Extension variant

        // Create extension field values
        // Extension field values are created from base field using From trait
        let ext_val_0 = EF::from_u64(3);
        let ext_val_1 = EF::from_u64(4);
        let ext_values = vec![ext_val_0, ext_val_1];

        // Construct Extension variant
        let ext_opening: QueryOpening<F, EF, <MyMmcs as Mmcs<F>>::Proof> =
            QueryOpening::Extension {
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
            QueryOpening::Base { .. } | QueryOpening::Batch { .. } => {
                panic!("Expected Extension variant")
            }
        }
    }

    #[test]
    fn test_set_final_sumcheck_data() {
        // Create a proof with no rounds
        let mut proof: WhirProof<F, EF, MyMmcs> = WhirProof {
            initial_commitment: None,
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
        data.pow_witnesses.push(F::from_u64(999));

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
        let mut proof: WhirProof<F, EF, MyMmcs> = WhirProof {
            initial_commitment: None,
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
        data0.pow_witnesses.push(F::from_u64(777));
        proof.set_sumcheck_data_at(data0, 0);

        // Create sumcheck data with a distinguishable value for round 1
        let mut data1: SumcheckData<F, EF> = SumcheckData::default();
        data1.pow_witnesses.push(F::from_u64(888));
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
        let mut proof: WhirProof<F, EF, MyMmcs> = WhirProof {
            initial_commitment: None,
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

    #[test]
    fn test_fold_ood_constraints() {
        let num_variables = 4;
        let r_0 = EF::from_u64(7);

        let point = MultilinearPoint::new(vec![
            EF::from_u64(1),
            EF::from_u64(2),
            EF::from_u64(3),
            EF::from_u64(4),
        ]);

        let v_a = EF::from_u64(10);
        let v_b = EF::from_u64(20);

        let mut ood_a = EqStatement::initialize(num_variables);
        ood_a.add_evaluated_constraint(point.clone(), v_a);
        let mut ood_b = EqStatement::initialize(num_variables);
        ood_b.add_evaluated_constraint(point, v_b);

        let folded = fold_ood_constraints(&ood_a, &ood_b, r_0);

        assert_eq!(folded.len(), 1);
        let (_, &folded_val) = folded.iter().next().unwrap();
        let expected = r_0 * v_a + (EF::ONE - r_0) * v_b;
        assert_eq!(folded_val, expected);
    }
}
