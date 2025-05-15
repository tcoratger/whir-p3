use p3_commit::Mmcs;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, PrimeField64, TwoAdicField};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use serde::{Deserialize, Serialize};

use super::Witness;
use crate::{
    fiat_shamir::{errors::ProofResult, prover::ProverState},
    poly::coeffs::CoefficientList,
    whir::{parameters::WhirConfig, utils::sample_ood_points},
};

/// Responsible for committing polynomials using a Merkle-based scheme.
///
/// The `Committer` processes a polynomial, expands and folds its evaluations,
/// and constructs a Merkle tree from the resulting values.
///
/// It provides a commitment that can be used for proof generation and verification.
#[derive(Debug)]
pub struct CommitmentWriter<EF, F, H, C, PowStrategy>(WhirConfig<EF, F, H, C, PowStrategy>)
where
    F: Field + TwoAdicField + PrimeField64,
    EF: ExtensionField<F> + TwoAdicField;

impl<EF, F, H, C, PS> CommitmentWriter<EF, F, H, C, PS>
where
    F: Field + TwoAdicField + PrimeField64,
    EF: ExtensionField<F> + TwoAdicField,
{
    pub const fn new(params: WhirConfig<EF, F, H, C, PS>) -> Self {
        Self(params)
    }

    /// Commits a polynomial using a Merkle-based commitment scheme.
    ///
    /// This function:
    /// - Expands polynomial coefficients to evaluations.
    /// - Applies folding and restructuring optimizations.
    /// - Converts evaluations to an extension field.
    /// - Constructs a Merkle tree from the evaluations.
    /// - Computes out-of-domain (OOD) challenge points and their evaluations.
    /// - Returns a `Witness` containing the commitment data.
    pub fn commit<D, const DIGEST_ELEMS: usize>(
        &self,
        dft: &D,
        prover_state: &mut ProverState<EF, F>,
        polynomial: CoefficientList<F>,
    ) -> ProofResult<Witness<EF, F, DIGEST_ELEMS>>
    where
        H: CryptographicHasher<F, [u8; DIGEST_ELEMS]> + Sync,
        C: PseudoCompressionFunction<[u8; DIGEST_ELEMS], 2> + Sync,
        [u8; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
        D: TwoAdicSubgroupDft<F>,
    {
        // Retrieve the base domain, ensuring it is set.
        let base_domain = self.0.starting_domain.base_domain.unwrap();

        // Compute expansion factor based on the domain size and polynomial length.
        let expansion = base_domain.size() / polynomial.num_coeffs();

        // Expand polynomial coefficients into evaluations over the domain
        let folded_matrix = {
            let mut coeffs = polynomial.coeffs().to_vec();
            coeffs.resize(coeffs.len() * expansion, F::ZERO);
            // Do DFT on only interleaved polys to be folded.
            dft.dft_batch(RowMajorMatrix::new(
                coeffs,
                1 << self.0.folding_factor.at_round(0),
            ))
            // Get natural order of rows.
            .to_row_major_matrix()
        };

        // Commit to the Merkle tree
        let merkle_tree =
            MerkleTreeMmcs::new(self.0.merkle_hash.clone(), self.0.merkle_compress.clone());
        let (root, prover_data) = merkle_tree.commit_matrix(folded_matrix);

        // Observe Merkle root in challenger
        prover_state.add_digest(root)?;

        // Handle OOD (Out-Of-Domain) samples
        let (ood_points, ood_answers) = sample_ood_points(
            prover_state,
            self.0.committment_ood_samples,
            self.0.mv_parameters.num_variables,
            |point| polynomial.evaluate_at_extension(point),
        )?;

        // Return the witness containing the polynomial, Merkle tree, and OOD results.
        Ok(Witness {
            polynomial,
            prover_data,
            ood_points,
            ood_answers,
        })
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
        fiat_shamir::{domain_separator::DomainSeparator, pow::blake3::Blake3PoW},
        parameters::{
            FoldingFactor, MultivariateParameters, ProtocolParameters, errors::SecurityAssumption,
        },
        poly::multilinear::MultilinearPoint,
    };

    type F = BabyBear;
    type ByteHash = Keccak256Hash;
    type FieldHash = SerializingHasher<ByteHash>;
    type MyCompress = CompressionFunctionFromHasher<ByteHash, 2, 32>;

    #[test]
    fn test_basic_commitment() {
        // Set up Whir protocol parameters.
        let security_level = 100;
        let pow_bits = 20;
        let num_variables = 5;
        let starting_rate = 1;
        let folding_factor = 4;
        let first_round_folding_factor = 4;

        let byte_hash = ByteHash {};
        let field_hash = FieldHash::new(byte_hash);

        let compress = MyCompress::new(byte_hash);

        let whir_params = ProtocolParameters::<FieldHash, MyCompress> {
            initial_statement: true,
            security_level,
            pow_bits,
            folding_factor: FoldingFactor::ConstantFromSecondRound(
                first_round_folding_factor,
                folding_factor,
            ),
            // merkle_hash: Poseidon2Sponge::new(poseidon_p24),
            // merkle_compress: Poseidon2Compression::new(poseidon_p16),
            merkle_hash: field_hash,
            merkle_compress: compress,
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: starting_rate,
        };

        // Define multivariate parameters for the polynomial.
        let mv_params = MultivariateParameters::new(num_variables);
        let params =
            WhirConfig::<F, F, FieldHash, MyCompress, Blake3PoW>::new(mv_params, whir_params);

        // Generate a random polynomial with 32 coefficients.
        let mut rng = rand::rng();
        let polynomial = CoefficientList::<BabyBear>::new(vec![rng.random(); 32]);

        // Set up the DomainSeparator and initialize a ProverState narg_string.
        let mut domainsep: DomainSeparator<F, F> = DomainSeparator::new("🌪️");
        domainsep.commit_statement(&params);
        domainsep.add_whir_proof(&params);
        let mut prover_state = domainsep.to_prover_state();

        // Run the Commitment Phase
        let committer = CommitmentWriter::new(params.clone());
        let dft_committer = Radix2DitParallel::<F>::default();
        let witness = committer
            .commit(&dft_committer, &mut prover_state, polynomial.clone())
            .unwrap();

        // Ensure OOD (out-of-domain) points are generated.
        assert!(
            !witness.ood_points.is_empty(),
            "OOD points should be generated"
        );

        // Validate the number of generated OOD points.
        assert_eq!(
            witness.ood_points.len(),
            params.committment_ood_samples,
            "OOD points count should match expected samples"
        );

        // Ensure polynomial data is correctly stored
        assert_eq!(
            witness.polynomial.coeffs().len(),
            polynomial.coeffs().len(),
            "Stored polynomial should have the correct number of coefficients"
        );

        // Check that OOD answers match expected evaluations
        for (i, ood_point) in witness.ood_points.iter().enumerate() {
            let expected_eval = polynomial.evaluate_at_extension(
                &MultilinearPoint::expand_from_univariate(*ood_point, num_variables),
            );
            assert_eq!(
                witness.ood_answers[i], expected_eval,
                "OOD answer at index {i} should match expected evaluation"
            );
        }
    }

    #[test]
    fn test_large_polynomial() {
        let security_level = 100;
        let pow_bits = 20;
        let num_variables = 10;
        let starting_rate = 1;
        let folding_factor = 4;
        let first_round_folding_factor = 4;

        let byte_hash = ByteHash {};
        let field_hash = FieldHash::new(byte_hash);

        let compress = MyCompress::new(byte_hash);

        let whir_params = ProtocolParameters {
            initial_statement: true,
            security_level,
            pow_bits,
            folding_factor: FoldingFactor::ConstantFromSecondRound(
                first_round_folding_factor,
                folding_factor,
            ),
            merkle_hash: field_hash,
            merkle_compress: compress,
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: starting_rate,
        };

        let mv_params = MultivariateParameters::<F>::new(num_variables);
        let params =
            WhirConfig::<F, F, FieldHash, MyCompress, Blake3PoW>::new(mv_params, whir_params);

        let mut rng = rand::rng();
        let polynomial = CoefficientList::<BabyBear>::new(vec![rng.random(); 1024]);

        let mut domainsep = DomainSeparator::new("🌪️");
        domainsep.commit_statement(&params);

        let mut prover_state = domainsep.to_prover_state();

        let dft_committer = Radix2DitParallel::<F>::default();
        let committer = CommitmentWriter::new(params);
        let _ = committer
            .commit(&dft_committer, &mut prover_state, polynomial)
            .unwrap();
    }

    #[test]
    fn test_commitment_without_ood_samples() {
        let security_level = 100;
        let pow_bits = 20;
        let num_variables = 5;
        let starting_rate = 1;
        let folding_factor = 4;
        let first_round_folding_factor = 4;

        let byte_hash = ByteHash {};
        let field_hash = FieldHash::new(byte_hash);

        let compress = MyCompress::new(byte_hash);

        let whir_params = ProtocolParameters {
            initial_statement: true,
            security_level,
            pow_bits,
            folding_factor: FoldingFactor::ConstantFromSecondRound(
                first_round_folding_factor,
                folding_factor,
            ),
            merkle_hash: field_hash,
            merkle_compress: compress,
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: starting_rate,
        };

        let mv_params = MultivariateParameters::<F>::new(num_variables);
        let mut params =
            WhirConfig::<F, F, FieldHash, MyCompress, Blake3PoW>::new(mv_params, whir_params);

        // Explicitly set OOD samples to 0
        params.committment_ood_samples = 0;

        let mut rng = rand::rng();
        let polynomial = CoefficientList::<BabyBear>::new(vec![rng.random(); 32]);

        let mut domainsep = DomainSeparator::new("🌪️");
        domainsep.commit_statement(&params);
        let mut prover_state = domainsep.to_prover_state();

        let dft_committer = Radix2DitParallel::<F>::default();
        let committer = CommitmentWriter::new(params);
        let witness = committer
            .commit(&dft_committer, &mut prover_state, polynomial)
            .unwrap();

        assert!(
            witness.ood_points.is_empty(),
            "There should be no OOD points when committment_ood_samples is 0"
        );
    }
}
