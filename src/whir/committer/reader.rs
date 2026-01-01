use core::{fmt::Debug, ops::Deref};

use p3_field::{ExtensionField, Field};
use p3_symmetric::Hash;

use crate::{
    fiat_shamir::{
        errors::FiatShamirError,
        transcript::{Challenge, Reader},
    },
    poly::multilinear::MultilinearPoint,
    whir::{constraints::statement::EqStatement, parameters::WhirConfig},
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
    /// - `transcript`: The verifier's Fiat-Shamir state from which data is read.
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
    pub fn parse<EF, Transcript, const DIGEST_ELEMS: usize>(
        transcript: &mut Transcript,
        num_variables: usize,
        ood_samples: usize,
    ) -> Result<ParsedCommitment<EF, Hash<F, F, DIGEST_ELEMS>>, FiatShamirError>
    where
        EF: ExtensionField<F>,
        Transcript: Reader<[F; DIGEST_ELEMS]> + Reader<EF> + Challenge<EF>,
    {
        // Read the Merkle root
        let root: [F; DIGEST_ELEMS] = transcript.read()?;
        // Construct equality constraints for all out-of-domain (OOD) samples.
        // Each constraint enforces that the committed polynomial evaluates to the
        // claimed `ood_answer` at the corresponding `ood_point`, using a univariate
        // equality weight over `num_variables` inputs.
        let mut ood_statement = EqStatement::initialize(num_variables);
        (0..ood_samples).try_for_each(|_| {
            let point = transcript.sample();
            let point = MultilinearPoint::expand_from_univariate(point, num_variables);
            let eval = transcript.read()?;
            ood_statement.add_evaluated_constraint(point, eval);
            Ok(())
        })?;

        // Return a structured representation of the commitment.
        Ok(ParsedCommitment {
            root: root.into(),
            ood_statement,
        })
    }
}

/// Helper for parsing commitment data during verification.
///
/// The `CommitmentReader` wraps the WHIR configuration and provides a convenient
/// method to extract a `ParsedCommitment` by reading values from the Fiat-Shamir transcript.
#[derive(Debug)]
pub struct CommitmentReader<'a, F, EF, Hasher, Compress>(
    /// Reference to the verifier’s configuration object.
    ///
    /// This contains all parameters needed to parse the commitment,
    /// including how many out-of-domain samples are expected.
    &'a WhirConfig<F, EF, Hasher, Compress>,
);

impl<'a, F: Field, EF: ExtensionField<F>, Hasher, Compress>
    CommitmentReader<'a, F, EF, Hasher, Compress>
{
    /// Create a new commitment reader from a WHIR configuration.
    ///
    /// This allows the verifier to parse a commitment from the Fiat-Shamir transcript.
    pub const fn new(params: &'a WhirConfig<F, EF, Hasher, Compress>) -> Self {
        Self(params)
    }

    /// Parse a commitment from the verifier's transcript state.
    ///
    /// Reads the Merkle root and out-of-domain (OOD) challenge points and answers
    /// expected for verifying the committed polynomial.
    pub fn parse_commitment<Transcript, const DIGEST_ELEMS: usize>(
        &self,
        transcript: &mut Transcript,
    ) -> Result<ParsedCommitment<EF, Hash<F, F, DIGEST_ELEMS>>, FiatShamirError>
    where
        Transcript: Reader<[F; DIGEST_ELEMS]> + Reader<EF> + Challenge<EF>,
    {
        ParsedCommitment::<_, [F; DIGEST_ELEMS]>::parse(
            transcript,
            self.num_variables,
            self.commitment_ood_samples,
        )
    }
}

impl<F, EF, Hasher, Compress> Deref for CommitmentReader<'_, F, EF, Hasher, Compress>
where
    F: Field,
    EF: ExtensionField<F>,
{
    type Target = WhirConfig<F, EF, Hasher, Compress>;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_dft::Radix2DFTSmallBatch;
    use p3_field::extension::BinomialExtensionField;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::{Rng, SeedableRng, rngs::SmallRng};

    use super::*;
    use crate::{
        fiat_shamir::transcript::{FiatShamirReader, FiatShamirWriter},
        parameters::{FoldingFactor, ProtocolParameters, errors::SecurityAssumption},
        poly::evals::EvaluationsList,
        whir::{DomainSeparator, committer::writer::CommitmentWriter, parameters::InitialPhase},
    };

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

    /// Constructs a WHIR configuration and RNG for test purposes.
    ///
    /// This sets up the protocol parameters and multivariate polynomial settings,
    /// with control over number of variables and OOD samples.
    #[allow(clippy::type_complexity)]
    fn make_test_params(
        num_variables: usize,
        ood_samples: usize,
    ) -> (WhirConfig<F, EF, MyHash, MyCompress>, SmallRng) {
        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);

        // Wrap the byte hash in a field-level serializer for Merkle hashing.
        let merkle_hash = MyHash::new(perm.clone());

        // Set up the Merkle compression function using the same byte hash.
        let merkle_compress = MyCompress::new(perm);

        // Define core protocol parameters for WHIR.
        let whir_params = ProtocolParameters {
            initial_phase: InitialPhase::WithStatementClassic,
            security_level: 100,
            pow_bits: 10,
            rs_domain_initial_reduction_factor: 1,
            folding_factor: FoldingFactor::ConstantFromSecondRound(4, 4),
            merkle_hash,
            merkle_compress,
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: 1,
        };

        // Construct full WHIR configuration with MV polynomial shape and protocol rules.
        let mut config = WhirConfig::new(num_variables, whir_params);

        // Set the number of OOD samples for commitment testing.
        config.commitment_ood_samples = ood_samples;

        // Return the config and a thread-local random number generator.
        (config, SmallRng::seed_from_u64(1))
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
        let dft = Radix2DFTSmallBatch::<F>::default();

        // Set up Fiat-Shamir transcript and commit the protocol parameters.
        let mut ds = DomainSeparator::new(vec![]);
        ds.commit_statement::<_, _, 8>(&params);

        // Create the prover state from the transcript.
        let mut rng = SmallRng::seed_from_u64(1);
        let mut challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));
        ds.observe_domain_separator(&mut challenger);
        let mut trancript = FiatShamirWriter::init(challenger.clone());

        // Commit the polynomial and obtain a witness (root, Merkle proof, OOD evaluations).
        let witness = committer.commit(&dft, &mut trancript, polynomial).unwrap();

        // Simulate verifier state using transcript view of prover's nonce string.
        let proof = trancript.finalize();
        let mut trancript = FiatShamirReader::init(proof, challenger);

        // Create a commitment reader and parse the commitment from verifier state.
        let reader = CommitmentReader::new(&params);
        let parsed = reader.parse_commitment::<_, 8>(&mut trancript).unwrap();

        // Ensure the Merkle root matches between prover and parsed result.
        assert_eq!(parsed.root, witness.prover_data.root());

        // Ensure the out-of-domain points and their answers match what was committed.
        assert_eq!(parsed.ood_statement, witness.ood_statement);
    }

    #[test]
    fn test_commitment_roundtrip_no_ood() {
        // Create WHIR config with 4 variables and *no* OOD samples.
        let (params, mut rng) = make_test_params(4, 0);

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
        let mut challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));
        ds.observe_domain_separator(&mut challenger);

        let mut transcript = FiatShamirWriter::init(challenger.clone());
        // Commit the polynomial to obtain the witness.
        let witness = committer.commit(&dft, &mut transcript, polynomial).unwrap();

        let proof = transcript.finalize();
        // Initialize the verifier view of the transcript.
        let mut transcript = FiatShamirReader::init(proof, challenger);

        // Parse the commitment from verifier transcript.
        let reader = CommitmentReader::new(&params);
        let parsed = reader.parse_commitment::<_, 8>(&mut transcript).unwrap();

        // Validate the Merkle root matches.
        assert_eq!(parsed.root, witness.prover_data.root());

        // OOD samples should be empty since none were requested.
        assert!(parsed.ood_statement.is_empty());
    }

    #[test]
    fn test_commitment_roundtrip_large_polynomial() {
        // Create config with 10 variables and 5 OOD samples.
        let (params, mut rng) = make_test_params(10, 5);

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
        let mut challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));
        ds.observe_domain_separator(&mut challenger);

        let mut transcript = FiatShamirWriter::init(challenger.clone());

        // Commit the polynomial and obtain the witness.
        let witness = committer.commit(&dft, &mut transcript, polynomial).unwrap();

        let proof = transcript.finalize();
        // Initialize verifier view from prover's transcript string.
        let mut transcript = FiatShamirReader::init(proof, challenger);

        // Parse the commitment from verifier's transcript.
        let reader = CommitmentReader::new(&params);
        let parsed = reader.parse_commitment::<_, 8>(&mut transcript).unwrap();

        // Check Merkle root and OOD answers match.
        assert_eq!(parsed.root, witness.prover_data.root());
        assert_eq!(parsed.ood_statement, witness.ood_statement);
    }

    #[test]
    fn test_oods_constraints_correctness() {
        // Create WHIR config with 4 variables and 2 OOD samples.
        let (params, mut rng) = make_test_params(4, 2);

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
        let mut challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));
        ds.observe_domain_separator(&mut challenger);

        let mut transcript = FiatShamirWriter::init(challenger.clone());
        let witness = committer.commit(&dft, &mut transcript, polynomial).unwrap();

        let proof = transcript.finalize();
        // Initialize the verifier view of the transcript.
        let mut transcript = FiatShamirReader::init(proof, challenger);

        // Parse the commitment from the verifier's state.
        let reader = CommitmentReader::new(&params);
        let parsed = reader.parse_commitment::<_, 8>(&mut transcript).unwrap();

        // Each constraint should have correct univariate weight, sum, and flag.
        for (i, (point, &eval)) in parsed.ood_statement.iter().enumerate() {
            let expected_point = witness.ood_statement.points[i].clone();
            let expected_eval = witness.ood_statement.evaluations[i];
            assert_eq!(
                point.clone(),
                expected_point,
                "Constraint {i} has incorrect weight"
            );
            assert_eq!(eval, expected_eval, "Constraint {i} has incorrect sum");
        }
    }
}
