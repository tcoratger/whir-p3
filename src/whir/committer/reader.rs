use std::ops::Deref;

use p3_challenger::{CanObserve, CanSample};
use p3_field::{ExtensionField, Field, PrimeField64, TwoAdicField};
use p3_symmetric::Hash;

use crate::{
    fiat_shamir::{errors::ProofResult, unit::Unit, verifier::VerifierState},
    whir::{
        parameters::WhirConfig,
        statement::{constraint::Constraint, weights::Weights},
    },
};

/// Represents a parsed commitment from the prover in the WHIR protocol.
///
/// This includes the Merkle root of the committed table and any out-of-domain (OOD)
/// query points and their corresponding answers, which are required for verifier checks.
#[derive(Debug, Clone)]
pub struct ParsedCommitment<F, D> {
    /// Number of variables in the committed polynomial.
    pub num_variables: usize,

    /// Merkle root of the committed evaluation table.
    ///
    /// This hash is used by the verifier to check Merkle proofs of queried evaluations.
    pub root: D,

    /// Points queried by the verifier outside the low-degree evaluation domain.
    ///
    /// These are chosen using Fiat-Shamir and used to test polynomial consistency.
    pub ood_points: Vec<F>,

    /// Answers (evaluations) of the committed polynomial at the corresponding `ood_points`.
    pub ood_answers: Vec<F>,
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
    /// - `verifier_state`: The verifier's Fiat-Shamir state from which data is read.
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
    pub fn parse<EF, Challenger, W, const DIGEST_ELEMS: usize>(
        verifier_state: &mut VerifierState<'_, EF, F, Challenger, W>,
        num_variables: usize,
        ood_samples: usize,
    ) -> ProofResult<ParsedCommitment<EF, Hash<F, W, DIGEST_ELEMS>>>
    where
        F: TwoAdicField + PrimeField64,
        EF: ExtensionField<F> + TwoAdicField,
        W: Unit + Default + Copy,
        Challenger: CanObserve<W> + CanSample<W>,
    {
        // Read the Merkle root hash committed by the prover.
        let root = verifier_state.read_digest()?;

        // Allocate space for the OOD challenge points and answers.
        let mut ood_points = EF::zero_vec(ood_samples);
        let mut ood_answers = EF::zero_vec(ood_samples);

        // If there are any OOD samples expected, read them from the transcript.
        if ood_samples > 0 {
            // Read challenge points chosen by Fiat-Shamir.
            verifier_state.fill_challenge_scalars(&mut ood_points)?;

            // Read the prover's claimed evaluations at those points.
            verifier_state.fill_next_scalars(&mut ood_answers)?;
        }

        // Return a structured representation of the commitment.
        Ok(ParsedCommitment {
            num_variables,
            root,
            ood_points,
            ood_answers,
        })
    }

    /// Construct equality constraints for all out-of-domain (OOD) samples.
    ///
    /// Each constraint enforces that the committed polynomial evaluates to the
    /// claimed `ood_answer` at the corresponding `ood_point`, using a univariate
    /// equality weight over `num_variables` inputs.
    pub fn oods_constraints(&self) -> Vec<Constraint<F>> {
        self.ood_points
            .iter()
            .zip(&self.ood_answers)
            .map(|(&point, &eval)| Constraint {
                weights: Weights::univariate(point, self.num_variables),
                sum: eval,
                defer_evaluation: false,
            })
            .collect()
    }
}

/// Helper for parsing commitment data during verification.
///
/// The `CommitmentReader` wraps the WHIR configuration and provides a convenient
/// method to extract a `ParsedCommitment` by reading values from the Fiat-Shamir transcript.
#[derive(Debug)]
pub struct CommitmentReader<'a, EF, F, H, C, PowStrategy, Challenger, W>(
    /// Reference to the verifier’s configuration object.
    ///
    /// This contains all parameters needed to parse the commitment,
    /// including how many out-of-domain samples are expected.
    &'a WhirConfig<EF, F, H, C, PowStrategy, Challenger, W>,
)
where
    F: Field,
    EF: ExtensionField<F>;

impl<'a, EF, F, H, C, PS, Challenger, W> CommitmentReader<'a, EF, F, H, C, PS, Challenger, W>
where
    F: Field + TwoAdicField + PrimeField64,
    EF: ExtensionField<F> + TwoAdicField,
    W: Unit + Default + Copy,
    Challenger: CanObserve<W> + CanSample<W>,
{
    /// Create a new commitment reader from a WHIR configuration.
    ///
    /// This allows the verifier to parse a commitment from the Fiat-Shamir transcript.
    pub const fn new(params: &'a WhirConfig<EF, F, H, C, PS, Challenger, W>) -> Self {
        Self(params)
    }

    /// Parse a commitment from the verifier's transcript state.
    ///
    /// Reads the Merkle root and out-of-domain (OOD) challenge points and answers
    /// expected for verifying the committed polynomial.
    pub fn parse_commitment<const DIGEST_ELEMS: usize>(
        &self,
        verifier_state: &mut VerifierState<'_, EF, F, Challenger, W>,
    ) -> ProofResult<ParsedCommitment<EF, Hash<F, W, DIGEST_ELEMS>>> {
        ParsedCommitment::<_, Hash<F, W, DIGEST_ELEMS>>::parse(
            verifier_state,
            self.mv_parameters.num_variables,
            self.committment_ood_samples,
        )
    }
}

impl<EF, F, H, C, PowStrategy, Challenger, W> Deref
    for CommitmentReader<'_, EF, F, H, C, PowStrategy, Challenger, W>
where
    F: Field,
    EF: ExtensionField<F>,
{
    type Target = WhirConfig<EF, F, H, C, PowStrategy, Challenger, W>;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_challenger::HashChallenger;
    use p3_field::PrimeCharacteristicRing;
    use p3_keccak::Keccak256Hash;
    use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
    use rand::Rng;

    use super::*;
    use crate::{
        dft::EvalsDft,
        fiat_shamir::pow::blake3::Blake3PoW,
        parameters::{
            FoldingFactor, MultivariateParameters, ProtocolParameters, errors::SecurityAssumption,
        },
        poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
        whir::{DomainSeparator, W, committer::writer::CommitmentWriter},
    };

    type F = BabyBear;
    type ByteHash = Keccak256Hash;
    type FieldHash = SerializingHasher<ByteHash>;
    type MyCompress = CompressionFunctionFromHasher<ByteHash, 2, 32>;
    type MyChallenger = HashChallenger<u8, Keccak256Hash, 32>;

    /// Constructs a WHIR configuration and RNG for test purposes.
    ///
    /// This sets up the protocol parameters and multivariate polynomial settings,
    /// with control over number of variables and OOD samples.
    #[allow(clippy::type_complexity)]
    fn make_test_params(
        num_variables: usize,
        ood_samples: usize,
    ) -> (
        WhirConfig<BabyBear, BabyBear, FieldHash, MyCompress, Blake3PoW, MyChallenger, W>,
        rand::rngs::ThreadRng,
    ) {
        // Initialize the underlying byte-level hash function (e.g., Keccak256).
        let byte_hash = ByteHash {};

        // Wrap the byte hash in a field-level serializer for Merkle hashing.
        let field_hash = FieldHash::new(byte_hash);

        // Set up the Merkle compression function using the same byte hash.
        let compress = MyCompress::new(byte_hash);

        // Define core protocol parameters for WHIR.
        let whir_params = ProtocolParameters {
            initial_statement: true,
            security_level: 100,
            pow_bits: 10,
            rs_domain_initial_reduction_factor: 1,
            folding_factor: FoldingFactor::ConstantFromSecondRound(4, 4),
            merkle_hash: field_hash,
            merkle_compress: compress,
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: 1,
        };

        // Construct full WHIR configuration with MV polynomial shape and protocol rules.
        let mut config = WhirConfig::<
            BabyBear,
            BabyBear,
            FieldHash,
            MyCompress,
            Blake3PoW,
            MyChallenger,
            W,
        >::new(MultivariateParameters::new(num_variables), whir_params);

        // Set the number of OOD samples for commitment testing.
        config.committment_ood_samples = ood_samples;

        // Return the config and a thread-local random number generator.
        (config, rand::rng())
    }

    #[test]
    fn test_commitment_roundtrip_with_ood() {
        // Create WHIR config with 5 variables and 3 OOD samples, plus a random number generator.
        let (params, mut rng) = make_test_params(5, 3);

        // Create a random degree-5 multilinear polynomial (32 coefficients).
        let polynomial = EvaluationsList::new((0..32).map(|_| rng.random()).collect());

        // Instantiate the committer using the test config.
        let committer = CommitmentWriter::new(&params);

        // Use a DFT engine to expand/fold the polynomial for evaluation.
        let dft = EvalsDft::default();

        // Set up Fiat-Shamir transcript and commit the protocol parameters.
        let mut ds = DomainSeparator::new("test");
        ds.commit_statement(&params);

        // Create the prover state from the transcript.
        let challenger = MyChallenger::new(vec![], Keccak256Hash);
        let mut prover_state = ds.to_prover_state(challenger.clone());

        // Commit the polynomial and obtain a witness (root, Merkle proof, OOD evaluations).
        let witness = committer
            .commit(&dft, &mut prover_state, polynomial)
            .unwrap();

        // Simulate verifier state using transcript view of prover’s nonce string.
        let mut verifier_state = ds.to_verifier_state(prover_state.narg_string(), challenger);

        // Create a commitment reader and parse the commitment from verifier state.
        let reader = CommitmentReader::new(&params);
        let parsed = reader.parse_commitment::<32>(&mut verifier_state).unwrap();

        // Ensure the Merkle root matches between prover and parsed result.
        assert_eq!(parsed.root, witness.prover_data.root());

        // Ensure the out-of-domain points and their answers match what was committed.
        assert_eq!(parsed.ood_points, witness.ood_points);
        assert_eq!(parsed.ood_answers, witness.ood_answers);
    }

    #[test]
    fn test_commitment_roundtrip_no_ood() {
        // Create WHIR config with 4 variables and *no* OOD samples.
        let (params, mut rng) = make_test_params(4, 0);

        // Generate a polynomial with 16 random coefficients.
        let polynomial = EvaluationsList::new((0..16).map(|_| rng.random()).collect());

        // Set up the committer and DFT engine.
        let committer = CommitmentWriter::new(&params);
        let dft = EvalsDft::default();

        // Begin the transcript and commit to the statement parameters.
        let mut ds = DomainSeparator::new("test");
        ds.commit_statement(&params);

        // Generate the prover state from the transcript.
        let challenger = MyChallenger::new(vec![], Keccak256Hash);
        let mut prover_state = ds.to_prover_state(challenger.clone());

        // Commit the polynomial to obtain the witness.
        let witness = committer
            .commit(&dft, &mut prover_state, polynomial)
            .unwrap();

        // Initialize the verifier view of the transcript.
        let mut verifier_state = ds.to_verifier_state(prover_state.narg_string(), challenger);

        // Parse the commitment from verifier transcript.
        let reader = CommitmentReader::new(&params);
        let parsed = reader.parse_commitment::<32>(&mut verifier_state).unwrap();

        // Validate the Merkle root matches.
        assert_eq!(parsed.root, witness.prover_data.root());

        // OOD samples should be empty since none were requested.
        assert!(parsed.ood_points.is_empty());
        assert!(parsed.ood_answers.is_empty());

        assert_eq!(parsed.ood_points, witness.ood_points);
        assert_eq!(parsed.ood_answers, witness.ood_answers);
    }

    #[test]
    fn test_commitment_roundtrip_large_polynomial() {
        // Create config with 10 variables and 5 OOD samples.
        let (params, mut rng) = make_test_params(10, 5);

        // Generate a large polynomial with 1024 random coefficients.
        let polynomial = EvaluationsList::new((0..1024).map(|_| rng.random()).collect());

        // Initialize the committer and DFT engine.
        let committer = CommitmentWriter::new(&params);
        let dft = EvalsDft::default();

        // Start a new transcript and commit to the public parameters.
        let mut ds = DomainSeparator::new("test");
        ds.commit_statement(&params);

        // Create prover state from the transcript.
        let challenger = MyChallenger::new(vec![], Keccak256Hash);
        let mut prover_state = ds.to_prover_state(challenger.clone());

        // Commit the polynomial and obtain the witness.
        let witness = committer
            .commit(&dft, &mut prover_state, polynomial)
            .unwrap();

        // Initialize verifier view from prover's transcript string.
        let mut verifier_state = ds.to_verifier_state(prover_state.narg_string(), challenger);

        // Parse the commitment from verifier’s transcript.
        let reader = CommitmentReader::new(&params);
        let parsed = reader.parse_commitment::<32>(&mut verifier_state).unwrap();

        // Check Merkle root and OOD answers match.
        assert_eq!(parsed.root, witness.prover_data.root());
        assert_eq!(parsed.ood_points, witness.ood_points);
        assert_eq!(parsed.ood_answers, witness.ood_answers);
    }

    #[test]
    fn test_oods_constraints_correctness() {
        // Create WHIR config with 4 variables and 2 OOD samples.
        let (params, mut rng) = make_test_params(4, 2);

        // Generate a multilinear polynomial with 16 coefficients.
        let polynomial = EvaluationsList::new((0..16).map(|_| rng.random()).collect());

        // Instantiate a committer and DFT backend.
        let committer = CommitmentWriter::new(&params);
        let dft = EvalsDft::default();

        // Set up Fiat-Shamir transcript and commit to the public parameters.
        let mut ds = DomainSeparator::new("oods_constraints_test");
        ds.commit_statement(&params);

        // Generate prover and verifier transcript states.
        let challenger = MyChallenger::new(vec![], Keccak256Hash);
        let mut prover_state = ds.to_prover_state(challenger.clone());
        let _ = committer
            .commit(&dft, &mut prover_state, polynomial)
            .unwrap();
        let mut verifier_state = ds.to_verifier_state(prover_state.narg_string(), challenger);

        // Parse the commitment from the verifier’s state.
        let reader = CommitmentReader::new(&params);
        let parsed = reader.parse_commitment::<32>(&mut verifier_state).unwrap();

        // Extract constraints from parsed commitment.
        let constraints = parsed.oods_constraints();

        // Ensure we got one constraint per OOD point.
        assert_eq!(constraints.len(), parsed.ood_points.len());

        // Each constraint should have correct univariate weight, sum, and flag.
        for (i, constraint) in constraints.iter().enumerate() {
            let point = parsed.ood_points[i];
            let expected_eval = parsed.ood_answers[i];

            // Manually compute the expanded univariate point
            let expanded = MultilinearPoint(vec![
                point.exp_u64(8),
                point.exp_u64(4),
                point.exp_u64(2),
                point,
            ]);

            let expected_weight = Weights::evaluation(expanded);

            assert_eq!(
                constraint.weights, expected_weight,
                "Constraint {i} has incorrect weight"
            );
            assert_eq!(
                constraint.sum, expected_eval,
                "Constraint {i} has incorrect sum"
            );
            assert!(
                !constraint.defer_evaluation,
                "Constraint {i} should not defer evaluation"
            );
        }
    }
}
