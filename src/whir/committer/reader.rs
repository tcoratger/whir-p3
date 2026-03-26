use core::{fmt::Debug, ops::Deref};

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field, PackedValue, TwoAdicField};
use p3_multilinear_util::multilinear::MultilinearPoint;

use crate::{
    constraints::statement::EqStatement,
    parameters::WhirConfig,
    whir::proof::{BatchWhirProof, WhirProof},
};

/// Represents a parsed commitment from the prover in the WHIR protocol.
///
/// This includes the Merkle root of the committed table and any out-of-domain (OOD)
/// query points and their corresponding answers, which are required for verifier checks.
#[derive(Debug, Clone)]
pub struct ParsedCommitment<F, D> {
    /// Merkle root of the committed evaluation table.
    ///
    /// This hash is used by the verifier to check Merkle proofs of queried evaluations.
    pub root: D,

    /// Out-of-domain statement with:
    /// - Out-of-domain challenge points used for polynomial verification.
    /// - The corresponding polynomial evaluations at the OOD challenge point
    pub ood_statement: EqStatement<F>,
}

impl<F, D> ParsedCommitment<F, D>
where
    F: Field,
{
    /// Parse a commitment from the verifier's transcript state.
    ///
    /// This function extracts a `ParsedCommitment` by reading the Merkle root,
    /// out-of-domain (OOD) challenge points, and corresponding claimed evaluations
    /// from the verifier's Fiat-Shamir transcript.
    ///
    /// # Arguments
    ///
    /// - `proof`: The proof data the verifier reads (currently unused, reserved for RF flow).
    /// - `challenger`: The verifier's challenger (currently unused, reserved for RF flow).
    /// - `num_variables`: Number of variables in the committed multilinear polynomial.
    /// - `ood_samples`: Number of out-of-domain points the verifier expects to query.
    ///
    /// # Returns
    ///
    /// A [`ParsedCommitment`] containing:
    /// - Number of variables in the committed multilinear polynomial
    /// - The Merkle root of the committed table,
    /// - The OOD challenge points,
    /// - The prover's claimed answers at those points.
    ///
    /// This is used to verify consistency of polynomial commitments in WHIR.
    pub fn parse<EF, MT: Mmcs<F>, Challenger>(
        proof: &WhirProof<F, EF, MT>,
        challenger: &mut Challenger,
        num_variables: usize,
        ood_samples: usize,
    ) -> ParsedCommitment<EF, MT::Commitment>
    where
        F: TwoAdicField,
        EF: ExtensionField<F> + TwoAdicField,
        Challenger:
            FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<MT::Commitment>,
    {
        Self::parse_with_round(proof, challenger, num_variables, ood_samples, None)
    }

    pub fn parse_with_round<EF, MT: Mmcs<F>, Challenger>(
        proof: &WhirProof<F, EF, MT>,
        challenger: &mut Challenger,
        num_variables: usize,
        ood_samples: usize,
        round_index: Option<usize>,
    ) -> ParsedCommitment<EF, MT::Commitment>
    where
        F: TwoAdicField,
        EF: ExtensionField<F> + TwoAdicField,
        Challenger:
            FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<MT::Commitment>,
    {
        let (root, ood_answers) = round_index.map_or_else(
            || {
                (
                    proof.initial_commitment.clone().unwrap(),
                    proof.initial_ood_answers.clone(),
                )
            },
            |idx| {
                let round_proof = &proof.rounds[idx];
                (
                    round_proof.commitment.clone().unwrap(),
                    round_proof.ood_answers.clone(),
                )
            },
        );

        // Observe the root in the challenger using generic CanObserve
        challenger.observe(root.clone());

        // Construct equality constraints for all out-of-domain (OOD) samples.
        // Each constraint enforces that the committed polynomial evaluates to the
        // claimed `ood_answer` at the corresponding `ood_point`, using a univariate
        // equality weight over `num_variables` inputs.
        let mut ood_statement = EqStatement::initialize(num_variables);
        (0..ood_samples).for_each(|i| {
            let point = challenger.sample_algebra_element();
            let point = MultilinearPoint::expand_from_univariate(point, num_variables);
            let eval = ood_answers[i];
            challenger.observe_algebra_element(eval);
            ood_statement.add_evaluated_constraint(point, eval);
        });

        // Return a structured representation of the commitment.
        ParsedCommitment {
            root,
            ood_statement,
        }
    }
}

/// Helper for parsing commitment data during verification.
///
/// The `CommitmentReader` wraps the WHIR configuration and provides a convenient
/// method to extract a `ParsedCommitment` by reading values from the Fiat-Shamir transcript.
#[derive(Debug)]
pub struct CommitmentReader<'a, EF, F, MT: Mmcs<F>, Challenger>(
    /// Reference to the verifier’s configuration object.
    ///
    /// This contains all parameters needed to parse the commitment,
    /// including how many out-of-domain samples are expected.
    &'a WhirConfig<EF, F, MT, Challenger>,
)
where
    F: Field,
    EF: ExtensionField<F>;

impl<'a, EF, F, MT, Challenger> CommitmentReader<'a, EF, F, MT, Challenger>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    MT: Mmcs<F>,
{
    /// Create a new commitment reader from a WHIR configuration.
    ///
    /// This allows the verifier to parse a commitment from the Fiat-Shamir transcript.
    pub const fn new(params: &'a WhirConfig<EF, F, MT, Challenger>) -> Self {
        Self(params)
    }

    /// Parse a commitment from the verifier's transcript state.
    ///
    /// Reads the Merkle root and out-of-domain (OOD) challenge points and answers
    /// expected for verifying the committed polynomial.
    pub fn parse_commitment<W, const DIGEST_ELEMS: usize>(
        &self,
        proof: &WhirProof<F, EF, MT>,
        challenger: &mut Challenger,
    ) -> ParsedCommitment<EF, MT::Commitment>
    where
        W: PackedValue<Value = W> + Eq + Copy,
        Challenger: CanObserve<MT::Commitment>,
    {
        ParsedCommitment::<_, MT::Commitment>::parse(
            proof,
            challenger,
            self.num_variables,
            self.commitment_ood_samples,
        )
    }
}

impl<EF, F, MT, Challenger> Deref for CommitmentReader<'_, EF, F, MT, Challenger>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
{
    type Target = WhirConfig<EF, F, MT, Challenger>;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

/// Parses batch commitment data from a `BatchWhirProof` during verification.
///
/// Reconstructs both `ParsedCommitment`s by replaying the transcript: observes
/// both roots, then reconstructs OOD statements from the proof's stored answers.
#[derive(Debug)]
pub struct BatchCommitmentReader<'a, EF, F, MT: Mmcs<F>, Challenger>(
    &'a WhirConfig<EF, F, MT, Challenger>,
)
where
    F: Field,
    EF: ExtensionField<F>;

impl<EF, F, MT, Challenger> BatchCommitmentReader<'_, EF, F, MT, Challenger>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    MT: Mmcs<F>,
{
    pub const fn new(
        params: &WhirConfig<EF, F, MT, Challenger>,
    ) -> BatchCommitmentReader<'_, EF, F, MT, Challenger> {
        BatchCommitmentReader(params)
    }

    /// Parse both commitments from a batch proof, replaying the transcript.
    pub fn parse_batch_commitment(
        &self,
        proof: &BatchWhirProof<F, EF, MT>,
        challenger: &mut Challenger,
    ) -> (
        ParsedCommitment<EF, MT::Commitment>,
        ParsedCommitment<EF, MT::Commitment>,
    )
    where
        Challenger: CanObserve<MT::Commitment>,
    {
        let num_variables = self.0.num_variables;

        let root_a = proof.commitment_a.clone().unwrap();
        let root_b = proof.commitment_b.clone().unwrap();
        challenger.observe(root_a.clone());
        challenger.observe(root_b.clone());

        let mut ood_a = EqStatement::initialize(num_variables);
        let mut ood_b = EqStatement::initialize(num_variables);
        for i in 0..self.0.commitment_ood_samples {
            let point = MultilinearPoint::expand_from_univariate(
                challenger.sample_algebra_element(),
                num_variables,
            );
            let eval_a = proof.initial_ood_answers[0][i];
            let eval_b = proof.initial_ood_answers[1][i];
            challenger.observe_algebra_element(eval_a);
            challenger.observe_algebra_element(eval_b);
            ood_a.add_evaluated_constraint(point.clone(), eval_a);
            ood_b.add_evaluated_constraint(point, eval_b);
        }

        (
            ParsedCommitment {
                root: root_a,
                ood_statement: ood_a,
            },
            ParsedCommitment {
                root: root_b,
                ood_statement: ood_b,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_dft::Radix2DFTSmallBatch;
    use p3_field::{Field, extension::BinomialExtensionField};
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_multilinear_util::evals::EvaluationsList;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::{RngExt, SeedableRng, rngs::SmallRng};

    use super::*;
    use crate::{
        fiat_shamir::domain_separator::DomainSeparator,
        parameters::{FoldingFactor, ProtocolParameters, SecurityAssumption, SumcheckStrategy},
        whir::{committer::writer::CommitmentWriter, proof::WhirProof},
    };

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;
    type PackedF = <F as Field>::Packing;
    type MyMmcs = MerkleTreeMmcs<PackedF, PackedF, MyHash, MyCompress, 2, 8>;

    /// Constructs a WHIR configuration and RNG for test purposes.
    ///
    /// This sets up the protocol parameters and multivariate polynomial settings,
    /// with control over number of variables and OOD samples.
    #[allow(clippy::type_complexity)]
    fn make_test_params(
        num_variables: usize,
        ood_samples: usize,
    ) -> (
        WhirConfig<EF, BabyBear, MyMmcs, MyChallenger>,
        SmallRng,
        WhirProof<F, EF, MyMmcs>,
    ) {
        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);

        // Wrap the byte hash in a field-level serializer for Merkle hashing.
        let merkle_hash = MyHash::new(perm.clone());

        // Set up the Merkle compression function using the same byte hash.
        let merkle_compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(merkle_hash, merkle_compress, 0);

        // Define core protocol parameters for WHIR.
        let whir_params = ProtocolParameters {
            security_level: 100,
            pow_bits: 10,
            rs_domain_initial_reduction_factor: 1,
            folding_factor: FoldingFactor::ConstantFromSecondRound(4, 4),
            mmcs,
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: 1,
        };

        // Construct full WHIR configuration with MV polynomial shape and protocol rules.
        let mut config = WhirConfig::new(num_variables, whir_params.clone());

        // Set the number of OOD samples for commitment testing.
        config.commitment_ood_samples = ood_samples;

        // Return the config and a thread-local random number generator.
        (
            config,
            SmallRng::seed_from_u64(1),
            WhirProof::from_protocol_parameters(&whir_params, num_variables),
        )
    }

    #[test]
    fn test_commitment_roundtrip_with_ood() {
        // Create WHIR config with 5 variables and 3 OOD samples, plus a random number generator.
        let (params, mut rng, mut proof) = make_test_params(5, 3);

        // Create a random degree-5 multilinear polynomial (32 coefficients).
        let polynomial = EvaluationsList::new((0..32).map(|_| rng.random()).collect());

        // Instantiate the committer using the test config.
        let committer = CommitmentWriter::new(&params);

        // Use a DFT engine to expand/fold the polynomial for evaluation.
        let dft = Radix2DFTSmallBatch::<F>::default();

        // Set up Fiat-Shamir transcript and commit the protocol parameters.
        let mut ds = DomainSeparator::new(vec![]);
        ds.commit_statement::<_, _, 8>(&params);

        // Create the prover state from the transcript.
        let mut rng = SmallRng::seed_from_u64(1);
        let challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));
        let mut prover_challenger = challenger.clone();
        ds.observe_domain_separator(&mut prover_challenger);

        let mut statement = params.initial_statement(polynomial, SumcheckStrategy::Classic);
        // Commit the polynomial and obtain a witness (root, Merkle proof, OOD evaluations).
        let prover_data = committer
            .commit(&dft, &mut proof, &mut prover_challenger, &mut statement)
            .unwrap();

        // Simulate verifier state using transcript view of prover's nonce string.
        let mut verifier_challenger = challenger;
        ds.observe_domain_separator(&mut verifier_challenger);

        // Create a commitment reader and parse the commitment from verifier state.
        let reader = CommitmentReader::new(&params);
        let parsed = reader.parse_commitment::<F, 8>(&proof, &mut verifier_challenger);

        // Ensure the Merkle root matches between prover and parsed result.
        assert_eq!(parsed.root, prover_data.root().into());

        // Ensure the out-of-domain points and their answers match what was committed.
        assert_eq!(parsed.ood_statement, statement.normalize());
    }

    #[test]
    fn test_commitment_roundtrip_no_ood() {
        // Create WHIR config with 4 variables and *no* OOD samples.
        let (params, mut rng, mut proof) = make_test_params(4, 0);

        // Generate a polynomial with 16 random coefficients.
        let polynomial = EvaluationsList::new((0..16).map(|_| rng.random()).collect());

        // Set up the committer and DFT engine.
        let committer = CommitmentWriter::new(&params);
        let dft = Radix2DFTSmallBatch::<F>::default();

        // Begin the transcript and commit to the statement parameters.
        let mut ds = DomainSeparator::new(vec![]);
        ds.commit_statement::<_, _, 8>(&params);

        // Create the prover state from the transcript.
        let mut rng = SmallRng::seed_from_u64(1);
        let challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));
        let mut prover_challenger = challenger.clone();
        ds.observe_domain_separator(&mut prover_challenger);

        let mut statement = params.initial_statement(polynomial, SumcheckStrategy::Classic);
        // Commit the polynomial to obtain the witness.
        let prover_data = committer
            .commit(&dft, &mut proof, &mut prover_challenger, &mut statement)
            .unwrap();

        // Initialize the verifier view of the transcript.
        let mut verifier_challenger = challenger;
        ds.observe_domain_separator(&mut verifier_challenger);

        // Parse the commitment from verifier transcript.
        let reader = CommitmentReader::new(&params);
        let parsed = reader.parse_commitment::<F, 8>(&proof, &mut verifier_challenger);

        // Validate the Merkle root matches.
        assert_eq!(parsed.root, prover_data.root().into());

        // OOD samples should be empty since none were requested.
        assert!(parsed.ood_statement.is_empty());
    }

    #[test]
    fn test_commitment_roundtrip_large_polynomial() {
        // Create config with 10 variables and 5 OOD samples.
        let (params, mut rng, mut proof) = make_test_params(10, 5);

        // Generate a large polynomial with 1024 random coefficients.
        let polynomial = EvaluationsList::new((0..1024).map(|_| rng.random()).collect());

        // Initialize the committer and DFT engine.
        let committer = CommitmentWriter::new(&params);
        let dft = Radix2DFTSmallBatch::<F>::default();

        // Start a new transcript and commit to the public parameters.
        let mut ds = DomainSeparator::new(vec![]);
        ds.commit_statement::<_, _, 8>(&params);

        // Create prover state from the transcript.
        let mut rng = SmallRng::seed_from_u64(1);
        let challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));
        let mut prover_challenger = challenger.clone();
        ds.observe_domain_separator(&mut prover_challenger);

        let mut statement = params.initial_statement(polynomial, SumcheckStrategy::Classic);
        // Commit the polynomial and obtain the witness.
        let prover_data = committer
            .commit(&dft, &mut proof, &mut prover_challenger, &mut statement)
            .unwrap();

        // Initialize verifier view from prover's transcript string.
        let mut verifier_challenger = challenger;
        ds.observe_domain_separator(&mut verifier_challenger);

        // Parse the commitment from verifier's transcript.
        let reader = CommitmentReader::new(&params);
        let parsed = reader.parse_commitment::<F, 8>(&proof, &mut verifier_challenger);

        // Check Merkle root and OOD answers match.
        assert_eq!(parsed.root, prover_data.root().into());
        assert_eq!(parsed.ood_statement, statement.normalize());
    }

    #[test]
    fn test_oods_constraints_correctness() {
        // Create WHIR config with 4 variables and 2 OOD samples.
        let (params, mut rng, mut proof) = make_test_params(4, 2);

        // Generate a multilinear polynomial with 16 coefficients.
        let polynomial = EvaluationsList::new((0..16).map(|_| rng.random()).collect());

        // Instantiate a committer and DFT backend.
        let committer = CommitmentWriter::new(&params);
        let dft = Radix2DFTSmallBatch::<F>::default();

        // Set up Fiat-Shamir transcript and commit to the public parameters.
        let mut ds = DomainSeparator::new(vec![]);
        ds.commit_statement::<_, _, 8>(&params);

        // Create the prover state from the transcript.
        let mut rng = SmallRng::seed_from_u64(1);
        let challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));
        let mut prover_challenger = challenger.clone();
        ds.observe_domain_separator(&mut prover_challenger);

        let mut statement = params.initial_statement(polynomial, SumcheckStrategy::Classic);
        let _ = committer
            .commit(&dft, &mut proof, &mut prover_challenger, &mut statement)
            .unwrap();

        // Initialize the verifier view of the transcript.
        let mut verifier_challenger = challenger;
        ds.observe_domain_separator(&mut verifier_challenger);

        // Parse the commitment from the verifier's state.
        let reader = CommitmentReader::new(&params);
        let parsed = reader.parse_commitment::<F, 8>(&proof, &mut verifier_challenger);

        // Each constraint should have correct univariate weight, sum, and flag.
        for (i, (point, &eval)) in parsed.ood_statement.iter().enumerate() {
            let statement = statement.normalize();
            let expected_point = statement.points[i].clone();
            let expected_eval = statement.evaluations[i];
            assert_eq!(
                point.clone(),
                expected_point,
                "Constraint {i} has incorrect weight"
            );
            assert_eq!(eval, expected_eval, "Constraint {i} has incorrect sum");
        }
    }
}
