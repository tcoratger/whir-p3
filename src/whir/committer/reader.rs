use std::ops::Deref;

use p3_field::{ExtensionField, Field, PrimeField64, TwoAdicField};
use p3_symmetric::Hash;

use crate::{
    fiat_shamir::{errors::ProofResult, verifier::VerifierState},
    whir::parameters::WhirConfig,
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

    /// Points queried by the verifier outside the low-degree evaluation domain.
    ///
    /// These are chosen using Fiat-Shamir and used to test polynomial consistency.
    pub ood_points: Vec<F>,

    /// Answers (evaluations) of the committed polynomial at the corresponding `ood_points`.
    pub ood_answers: Vec<F>,
}

/// Helper for parsing commitment data during verification.
///
/// The `CommitmentReader` wraps the WHIR configuration and provides a convenient
/// method to extract a `ParsedCommitment` by reading values from the Fiat-Shamir transcript.
#[derive(Debug)]
pub struct CommitmentReader<'a, EF, F, H, C, PowStrategy>(
    /// Reference to the verifier’s configuration object.
    ///
    /// This contains all parameters needed to parse the commitment,
    /// including how many out-of-domain samples are expected.
    &'a WhirConfig<EF, F, H, C, PowStrategy>,
)
where
    F: Field + TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField;

impl<'a, EF, F, H, C, PS> CommitmentReader<'a, EF, F, H, C, PS>
where
    F: Field + TwoAdicField + PrimeField64,
    EF: ExtensionField<F> + TwoAdicField,
{
    /// Create a new commitment reader from a WHIR configuration.
    ///
    /// This allows the verifier to parse a commitment from the Fiat-Shamir transcript.
    pub const fn new(params: &'a WhirConfig<EF, F, H, C, PS>) -> Self {
        Self(params)
    }

    /// Parse a commitment from the verifier's transcript state.
    ///
    /// Reads the Merkle root and out-of-domain (OOD) challenge points and answers
    /// expected for verifying the committed polynomial.
    pub fn parse_commitment<const DIGEST_ELEMS: usize>(
        &self,
        verifier_state: &mut VerifierState<'_, EF, F>,
    ) -> ProofResult<ParsedCommitment<EF, Hash<F, u8, DIGEST_ELEMS>>> {
        // Read the Merkle root hash committed by the prover.
        let root = verifier_state.read_digest()?;

        // Allocate space for the OOD challenge points and answers.
        let mut ood_points = vec![EF::ZERO; self.committment_ood_samples];
        let mut ood_answers = vec![EF::ZERO; self.committment_ood_samples];

        // If there are any OOD samples expected, read them from the transcript.
        if self.committment_ood_samples > 0 {
            // Read challenge points chosen by Fiat-Shamir.
            verifier_state.fill_challenge_scalars(&mut ood_points)?;

            // Read the prover's claimed evaluations at those points.
            verifier_state.fill_next_scalars(&mut ood_answers)?;
        }

        // Return a structured representation of the commitment.
        Ok(ParsedCommitment {
            root,
            ood_points,
            ood_answers,
        })
    }
}

impl<EF, F, H, C, PowStrategy> Deref for CommitmentReader<'_, EF, F, H, C, PowStrategy>
where
    F: Field + TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
{
    type Target = WhirConfig<EF, F, H, C, PowStrategy>;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_dft::Radix2DitParallel;
    use p3_keccak::Keccak256Hash;
    use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
    use rand::Rng;

    use super::*;
    use crate::{
        fiat_shamir::pow::blake3::Blake3PoW,
        parameters::{
            FoldingFactor, MultivariateParameters, ProtocolParameters, errors::SecurityAssumption,
        },
        poly::coeffs::CoefficientList,
        whir::{DomainSeparator, committer::writer::CommitmentWriter},
    };

    type F = BabyBear;
    type ByteHash = Keccak256Hash;
    type FieldHash = SerializingHasher<ByteHash>;
    type MyCompress = CompressionFunctionFromHasher<ByteHash, 2, 32>;

    /// Constructs a WHIR configuration and RNG for test purposes.
    ///
    /// This sets up the protocol parameters and multivariate polynomial settings,
    /// with control over number of variables and OOD samples.
    fn make_test_params(
        num_variables: usize,
        ood_samples: usize,
    ) -> (
        WhirConfig<BabyBear, BabyBear, FieldHash, MyCompress, Blake3PoW>,
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
            folding_factor: FoldingFactor::ConstantFromSecondRound(4, 4),
            merkle_hash: field_hash,
            merkle_compress: compress,
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: 1,
        };

        // Construct full WHIR configuration with MV polynomial shape and protocol rules.
        let mut config = WhirConfig::<BabyBear, BabyBear, FieldHash, MyCompress, Blake3PoW>::new(
            MultivariateParameters::new(num_variables),
            whir_params,
        );

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
        let polynomial = CoefficientList::new((0..32).map(|_| rng.random()).collect());

        // Instantiate the committer using the test config.
        let committer = CommitmentWriter::new(params.clone());

        // Use a DFT engine to expand/fold the polynomial for evaluation.
        let dft = Radix2DitParallel::default();

        // Set up Fiat-Shamir transcript and commit the protocol parameters.
        let mut ds = DomainSeparator::new("test");
        ds.commit_statement(&params);

        // Create the prover state from the transcript.
        let mut prover_state = ds.to_prover_state();

        // Commit the polynomial and obtain a witness (root, Merkle proof, OOD evaluations).
        let witness = committer
            .commit(&dft, &mut prover_state, polynomial)
            .unwrap();

        // Simulate verifier state using transcript view of prover’s nonce string.
        let mut verifier_state = ds.to_verifier_state(prover_state.narg_string());

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
        let polynomial = CoefficientList::new((0..16).map(|_| rng.random()).collect());

        // Set up the committer and DFT engine.
        let committer = CommitmentWriter::new(params.clone());
        let dft = Radix2DitParallel::default();

        // Begin the transcript and commit to the statement parameters.
        let mut ds = DomainSeparator::new("test");
        ds.commit_statement(&params);

        // Generate the prover state from the transcript.
        let mut prover_state = ds.to_prover_state();

        // Commit the polynomial to obtain the witness.
        let witness = committer
            .commit(&dft, &mut prover_state, polynomial)
            .unwrap();

        // Initialize the verifier view of the transcript.
        let mut verifier_state = ds.to_verifier_state(prover_state.narg_string());

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
        let polynomial = CoefficientList::new((0..1024).map(|_| rng.random()).collect());

        // Initialize the committer and DFT engine.
        let committer = CommitmentWriter::new(params.clone());
        let dft = Radix2DitParallel::default();

        // Start a new transcript and commit to the public parameters.
        let mut ds = DomainSeparator::new("test");
        ds.commit_statement(&params);

        // Create prover state from the transcript.
        let mut prover_state = ds.to_prover_state();

        // Commit the polynomial and obtain the witness.
        let witness = committer
            .commit(&dft, &mut prover_state, polynomial)
            .unwrap();

        // Initialize verifier view from prover's transcript string.
        let mut verifier_state = ds.to_verifier_state(prover_state.narg_string());

        // Parse the commitment from verifier’s transcript.
        let reader = CommitmentReader::new(&params);
        let parsed = reader.parse_commitment::<32>(&mut verifier_state).unwrap();

        // Check Merkle root and OOD answers match.
        assert_eq!(parsed.root, witness.prover_data.root());
        assert_eq!(parsed.ood_points, witness.ood_points);
        assert_eq!(parsed.ood_answers, witness.ood_answers);
    }
}
