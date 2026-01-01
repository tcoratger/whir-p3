use alloc::sync::Arc;
use core::ops::Deref;

use p3_commit::Mmcs;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::{Matrix, dense::RowMajorMatrixView};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use serde::{Deserialize, Serialize};
use tracing::{info_span, instrument};

use super::Witness;
use crate::{
    fiat_shamir::{
        errors::FiatShamirError,
        transcript::{Challenge, Writer},
    },
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    whir::{committer::DenseMatrix, constraints::statement::EqStatement, parameters::WhirConfig},
};

/// Responsible for committing polynomials using a Merkle-based scheme.
///
/// The `Committer` processes a polynomial, expands and folds its evaluations,
/// and constructs a Merkle tree from the resulting values.
///
/// It provides a commitment that can be used for proof generation and verification.
#[derive(Debug)]
pub struct CommitmentWriter<'a, F, EF, Hash, Compress>(
    /// Reference to the WHIR protocol configuration.
    &'a WhirConfig<F, EF, Hash, Compress>,
);

impl<'a, F: TwoAdicField, EF: ExtensionField<F>, Hash, Compress>
    CommitmentWriter<'a, F, EF, Hash, Compress>
{
    /// Create a new writer that borrows the WHIR protocol configuration.
    pub const fn new(params: &'a WhirConfig<F, EF, Hash, Compress>) -> Self {
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
    pub fn commit<Dft: TwoAdicSubgroupDft<F>, Transcript, const DIGEST_ELEMS: usize>(
        &self,
        dft: &Dft,
        transcript: &mut Transcript,
        polynomial: EvaluationsList<F>,
    ) -> Result<Witness<EF, F, DenseMatrix<F>, DIGEST_ELEMS>, FiatShamirError>
    where
        Hash: CryptographicHasher<F, [F; DIGEST_ELEMS]>
            + CryptographicHasher<F::Packing, [F::Packing; DIGEST_ELEMS]>
            + Sync,
        Compress: PseudoCompressionFunction<[F; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[F::Packing; DIGEST_ELEMS], 2>
            + Sync,
        [F; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
        Transcript: Writer<[F; DIGEST_ELEMS]> + Writer<EF> + Challenge<EF>,
    {
        // Transpose for reverse variable order
        // And then pad with zeros

        let padded = info_span!("transpose & pad").in_scope(|| {
            let num_vars = polynomial.num_variables();
            let mut mat = RowMajorMatrixView::new(
                polynomial.as_slice(),
                1 << (num_vars - self.folding_factor.at_round(0)),
            )
            .transpose();
            mat.pad_to_height(
                1 << (num_vars + self.starting_log_inv_rate - self.folding_factor.at_round(0)),
                F::ZERO,
            );
            mat
        });

        // Perform DFT on the padded evaluations matrix
        let folded_matrix = info_span!("dft", height = padded.height(), width = padded.width())
            .in_scope(|| dft.dft_batch(padded).to_row_major_matrix());

        // Commit to the Merkle tree
        let merkle_tree =
            MerkleTreeMmcs::<F::Packing, F::Packing, Hash, Compress, DIGEST_ELEMS>::new(
                self.merkle_hash.clone(),
                self.merkle_compress.clone(),
            );
        let (root, prover_data) =
            info_span!("commit_matrix").in_scope(|| merkle_tree.commit_matrix(folded_matrix));
        Writer::<[F; DIGEST_ELEMS]>::write(transcript, *root.as_ref())?;

        let mut ood_statement = EqStatement::initialize(self.num_variables);
        (0..self.0.commitment_ood_samples).try_for_each(|_| {
            // Generate OOD points from ProverState randomness
            let var: EF = transcript.sample();
            let point = MultilinearPoint::expand_from_univariate(var, self.num_variables);
            let eval = info_span!("ood evaluation")
                .in_scope(|| polynomial.evaluate_hypercube_base(&point));
            transcript.write(eval)?;
            ood_statement.add_evaluated_constraint(point, eval);
            Ok(())
        })?;

        // Return the witness containing the polynomial, Merkle tree, and OOD results.
        Ok(Witness {
            polynomial,
            prover_data: Arc::new(prover_data),
            ood_statement,
        })
    }
}

impl<F, EF, Hash, Compress> Deref for CommitmentWriter<'_, F, EF, Hash, Compress>
where
    F: Field,
{
    type Target = WhirConfig<F, EF, Hash, Compress>;
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
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::{Rng, SeedableRng, rngs::SmallRng};

    use super::*;
    use crate::{
        fiat_shamir::{domain_separator::DomainSeparator, transcript::FiatShamirWriter},
        parameters::{FoldingFactor, ProtocolParameters, errors::SecurityAssumption},
        whir::parameters::InitialPhase,
    };

    type F = BabyBear;
    type Perm = Poseidon2BabyBear<16>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

    #[test]
    fn test_basic_commitment() {
        // Set up Whir protocol parameters.
        let security_level = 100;
        let pow_bits = 20;
        let num_variables = 5;
        let starting_rate = 1;
        let folding_factor = 4;
        let first_round_folding_factor = 4;

        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);

        let merkle_hash = MyHash::new(perm.clone());
        let merkle_compress = MyCompress::new(perm);

        let whir_params = ProtocolParameters {
            initial_phase: InitialPhase::WithStatementClassic,
            security_level,
            pow_bits,
            rs_domain_initial_reduction_factor: 1,
            folding_factor: FoldingFactor::ConstantFromSecondRound(
                first_round_folding_factor,
                folding_factor,
            ),
            merkle_hash,
            merkle_compress,
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: starting_rate,
        };

        // Define multivariate parameters for the polynomial.
        let params = WhirConfig::<F, F, MyHash, MyCompress>::new(num_variables, whir_params);

        // Generate a random polynomial with 32 coefficients.
        let mut rng = SmallRng::seed_from_u64(1);
        let polynomial = EvaluationsList::<BabyBear>::new(vec![rng.random(); 32]);

        // Set up the DomainSeparator and initialize a ProverState narg_string.
        let mut domainsep = DomainSeparator::new(vec![]);
        domainsep.commit_statement::<_, _, 8>(&params);
        domainsep.add_whir_proof::<_, _, 8>(&params);

        let mut rng = SmallRng::seed_from_u64(1);
        let mut challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));
        domainsep.observe_domain_separator(&mut challenger);
        let mut transcript: FiatShamirWriter<F, _> = FiatShamirWriter::init(challenger);

        // Run the Commitment Phase
        let committer = CommitmentWriter::new(&params);
        let dft = Radix2DFTSmallBatch::<F>::default();
        let witness = committer
            .commit(&dft, &mut transcript, polynomial.clone())
            .unwrap();

        // Ensure OOD (out-of-domain) points are generated.
        assert!(
            !witness.ood_statement.is_empty(),
            "OOD points should be generated"
        );

        // Validate the number of generated OOD points.
        assert_eq!(
            witness.ood_statement.len(),
            params.commitment_ood_samples,
            "OOD points count should match expected samples"
        );

        // Check that OOD answers match expected evaluations
        for (i, (ood_point, ood_eval)) in witness.ood_statement.iter().enumerate() {
            let expected_eval = polynomial.evaluate_hypercube_base::<F>(ood_point);
            assert_eq!(
                *ood_eval, expected_eval,
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

        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);

        let merkle_hash = MyHash::new(perm.clone());
        let merkle_compress = MyCompress::new(perm);

        let whir_params = ProtocolParameters {
            initial_phase: InitialPhase::WithStatementClassic,
            security_level,
            pow_bits,
            rs_domain_initial_reduction_factor: 1,
            folding_factor: FoldingFactor::ConstantFromSecondRound(
                first_round_folding_factor,
                folding_factor,
            ),
            merkle_hash,
            merkle_compress,
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: starting_rate,
        };

        let params = WhirConfig::<F, F, MyHash, MyCompress>::new(num_variables, whir_params);

        let mut rng = SmallRng::seed_from_u64(1);
        let polynomial = EvaluationsList::<BabyBear>::new(vec![rng.random(); 1024]);

        let mut domainsep = DomainSeparator::new(vec![]);
        domainsep.commit_statement::<_, _, 8>(&params);

        let mut rng = SmallRng::seed_from_u64(1);
        let mut challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));
        domainsep.observe_domain_separator(&mut challenger);
        let mut transcript = FiatShamirWriter::init(challenger);

        let dft = Radix2DFTSmallBatch::<F>::default();
        let committer = CommitmentWriter::new(&params);
        let _ = committer.commit(&dft, &mut transcript, polynomial).unwrap();
    }

    #[test]
    fn test_commitment_without_ood_samples() {
        let security_level = 100;
        let pow_bits = 20;
        let num_variables = 5;
        let starting_rate = 1;
        let folding_factor = 4;
        let first_round_folding_factor = 4;

        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);

        let merkle_hash = MyHash::new(perm.clone());
        let merkle_compress = MyCompress::new(perm);

        let whir_params = ProtocolParameters {
            initial_phase: InitialPhase::WithStatementClassic,
            security_level,
            pow_bits,
            rs_domain_initial_reduction_factor: 1,
            folding_factor: FoldingFactor::ConstantFromSecondRound(
                first_round_folding_factor,
                folding_factor,
            ),
            merkle_hash,
            merkle_compress,
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: starting_rate,
        };

        let mut params = WhirConfig::<F, F, MyHash, MyCompress>::new(num_variables, whir_params);

        // Explicitly set OOD samples to 0
        params.commitment_ood_samples = 0;

        let mut rng = SmallRng::seed_from_u64(1);
        let polynomial = EvaluationsList::<BabyBear>::new(vec![rng.random(); 32]);

        let mut domainsep = DomainSeparator::new(vec![]);
        domainsep.commit_statement::<_, _, 8>(&params);

        let mut rng = SmallRng::seed_from_u64(1);
        let mut challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));
        domainsep.observe_domain_separator(&mut challenger);
        let mut transcript = FiatShamirWriter::init(challenger);

        let dft = Radix2DFTSmallBatch::<F>::default();
        let committer = CommitmentWriter::new(&params);
        let witness = committer.commit(&dft, &mut transcript, polynomial).unwrap();

        assert!(
            witness.ood_statement.is_empty(),
            "There should be no OOD points when committment_ood_samples is 0"
        );
    }
}
