use std::ops::Deref;

use p3_challenger::{CanObserve, CanSample};
use p3_commit::Mmcs;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, Packable, PrimeField64, TwoAdicField};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use serde::{Deserialize, Serialize};
use tracing::{info_span, instrument};

use super::Witness;
use crate::{
    fiat_shamir::{errors::ProofResult, prover::ProverState, unit::Unit},
    poly::{coeffs::CoefficientList, evals::EvaluationsList},
    utils::parallel_clone,
    whir::{committer::DenseMatrix, parameters::WhirConfig, utils::sample_ood_points},
};

/// Responsible for committing polynomials using a Merkle-based scheme.
///
/// The `Committer` processes a polynomial, expands and folds its evaluations,
/// and constructs a Merkle tree from the resulting values.
///
/// It provides a commitment that can be used for proof generation and verification.
#[derive(Debug)]
pub struct CommitmentWriter<'a, EF, F, H, C, PowStrategy, Challenger, W>(
    /// Reference to the WHIR protocol configuration.
    &'a WhirConfig<EF, F, H, C, PowStrategy, Challenger, W>,
)
where
    F: Field,
    EF: ExtensionField<F>;

impl<'a, EF, F, H, C, PS, Challenger, W> CommitmentWriter<'a, EF, F, H, C, PS, Challenger, W>
where
    F: Field + TwoAdicField + PrimeField64,
    EF: ExtensionField<F> + TwoAdicField,
    W: Unit + Default + Copy,
    Challenger: CanObserve<W> + CanSample<W>,
{
    /// Create a new writer that borrows the WHIR protocol configuration.
    pub const fn new(params: &'a WhirConfig<EF, F, H, C, PS, Challenger, W>) -> Self {
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
    #[instrument(skip_all)]
    pub fn commit<D, const DIGEST_ELEMS: usize>(
        &self,
        dft: &D,
        prover_state: &mut ProverState<EF, F, Challenger, W>,
        polynomial: EvaluationsList<F>,
    ) -> ProofResult<Witness<EF, F, W, DenseMatrix<F>, DIGEST_ELEMS>>
    where
        H: CryptographicHasher<F, [W; DIGEST_ELEMS]> + Sync,
        C: PseudoCompressionFunction<[W; DIGEST_ELEMS], 2> + Sync,
        [W; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
        D: TwoAdicSubgroupDft<F>,
        W: Eq + Packable,
    {
        // convert evaluations -> coefficients form
        let pol_coeffs: CoefficientList<F> = polynomial.parallel_clone().to_coefficients();

        // Compute expansion factor based on the domain size and polynomial length.
        let initial_size = polynomial.num_evals();
        let expanded_size = self.starting_domain.backing_domain.size();

        // Pad coefficients with zeros to match the domain size
        let coeffs = info_span!("copy_across_coeffs").in_scope(|| {
            let mut coeffs = F::zero_vec(expanded_size);
            parallel_clone(pol_coeffs.coeffs(), &mut coeffs[..initial_size]);
            coeffs
        });

        // Perform DFT on the padded coefficient matrix
        let width = 1 << self.folding_factor.at_round(0);
        let folded_matrix =
            info_span!("dft", height = coeffs.len() / width, width).in_scope(|| {
                dft.dft_batch(RowMajorMatrix::new(coeffs, width))
                    .to_row_major_matrix()
            });

        // Commit to the Merkle tree
        let merkle_tree =
            MerkleTreeMmcs::new(self.merkle_hash.clone(), self.merkle_compress.clone());
        let (root, prover_data) =
            info_span!("commit_matrix").in_scope(|| merkle_tree.commit_matrix(folded_matrix));

        // Observe Merkle root in challenger
        prover_state.add_digest(root)?;

        // Handle OOD (Out-Of-Domain) samples
        let (ood_points, ood_answers) = sample_ood_points(
            prover_state,
            self.committment_ood_samples,
            self.mv_parameters.num_variables,
            |point| info_span!("ood evaluation").in_scope(|| pol_coeffs.evaluate(point)),
        )?;

        // Return the witness containing the polynomial, Merkle tree, and OOD results.
        Ok(Witness {
            pol_coeffs,
            pol_evals: polynomial,
            prover_data,
            ood_points,
            ood_answers,
        })
    }
}

impl<EF, F, H, C, PowStrategy, Challenger, W> Deref
    for CommitmentWriter<'_, EF, F, H, C, PowStrategy, Challenger, W>
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
    use p3_dft::Radix2DitSmallBatch;
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
        whir::W,
    };

    type F = BabyBear;
    type ByteHash = Keccak256Hash;
    type FieldHash = SerializingHasher<ByteHash>;
    type MyCompress = CompressionFunctionFromHasher<ByteHash, 2, 32>;
    type MyChallenger = HashChallenger<u8, Keccak256Hash, 32>;

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
            rs_domain_initial_reduction_factor: 1,
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
        let params = WhirConfig::<F, F, FieldHash, MyCompress, Blake3PoW, MyChallenger, W>::new(
            mv_params,
            whir_params,
        );

        // Generate a random polynomial with 32 coefficients.
        let mut rng = rand::rng();
        let polynomial = EvaluationsList::<BabyBear>::new(vec![rng.random(); 32]);

        // Set up the DomainSeparator and initialize a ProverState narg_string.
        let mut domainsep: DomainSeparator<F, F, u8> = DomainSeparator::new("🌪️");
        domainsep.commit_statement(&params);
        domainsep.add_whir_proof(&params);

        let challenger = MyChallenger::new(vec![], Keccak256Hash);
        let mut prover_state = domainsep.to_prover_state(challenger);

        // Run the Commitment Phase
        let committer = CommitmentWriter::new(&params);
        let dft_committer = Radix2DitSmallBatch::<F>::default();
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
            witness.pol_coeffs.coeffs().len(),
            polynomial.num_evals(),
            "Stored polynomial should have the correct number of coefficients"
        );

        // Check that OOD answers match expected evaluations
        for (i, ood_point) in witness.ood_points.iter().enumerate() {
            let expected_eval = polynomial.evaluate(&MultilinearPoint::expand_from_univariate(
                *ood_point,
                num_variables,
            ));
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
            rs_domain_initial_reduction_factor: 1,
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
        let params = WhirConfig::<F, F, FieldHash, MyCompress, Blake3PoW, MyChallenger, W>::new(
            mv_params,
            whir_params,
        );

        let mut rng = rand::rng();
        let polynomial = EvaluationsList::<BabyBear>::new(vec![rng.random(); 1024]);

        let mut domainsep = DomainSeparator::new("🌪️");
        domainsep.commit_statement(&params);

        let challenger = MyChallenger::new(vec![], Keccak256Hash);
        let mut prover_state = domainsep.to_prover_state(challenger);

        let dft_committer = Radix2DitSmallBatch::<F>::default();
        let committer = CommitmentWriter::new(&params);
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
            rs_domain_initial_reduction_factor: 1,
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
        let mut params = WhirConfig::<F, F, FieldHash, MyCompress, Blake3PoW, MyChallenger, W>::new(
            mv_params,
            whir_params,
        );

        // Explicitly set OOD samples to 0
        params.committment_ood_samples = 0;

        let mut rng = rand::rng();
        let polynomial = EvaluationsList::<BabyBear>::new(vec![rng.random(); 32]);

        let mut domainsep = DomainSeparator::new("🌪️");
        domainsep.commit_statement(&params);

        let challenger = MyChallenger::new(vec![], Keccak256Hash);
        let mut prover_state = domainsep.to_prover_state(challenger);

        let dft_committer = Radix2DitSmallBatch::<F>::default();
        let committer = CommitmentWriter::new(&params);
        let witness = committer
            .commit(&dft_committer, &mut prover_state, polynomial)
            .unwrap();

        assert!(
            witness.ood_points.is_empty(),
            "There should be no OOD points when committment_ood_samples is 0"
        );
    }
}
