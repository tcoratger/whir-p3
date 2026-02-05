use core::ops::Deref;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, PackedValue, TwoAdicField};
use p3_matrix::{Matrix, dense::RowMajorMatrixView};
use p3_merkle_tree::{MerkleTree, MerkleTreeMmcs};
use p3_symmetric::{CryptographicHasher, Hash, PseudoCompressionFunction};
use serde::{Deserialize, Serialize};
use tracing::{info_span, instrument};
use crate::{
    fiat_shamir::errors::FiatShamirError,
    poly::multilinear::MultilinearPoint,
    whir::{
        committer::DenseMatrix,
        constraints::statement::{EqStatement, initial::InitialStatement},
        parameters::WhirConfig,
        proof::WhirProof,
    },
};

/// Responsible for committing polynomials using a Merkle-based scheme.
///
/// The `Committer` processes a polynomial, expands and folds its evaluations,
/// and constructs a Merkle tree from the resulting values.
///
/// It provides a commitment that can be used for proof generation and verification.
#[derive(Debug)]
pub struct CommitmentWriter<'a, EF, F, H, C, Challenger>(
    /// Reference to the WHIR protocol configuration.
    &'a WhirConfig<EF, F, H, C, Challenger>,
)
where
    F: Field,
    EF: ExtensionField<F>;

impl<'a, EF, F, H, C, Challenger> CommitmentWriter<'a, EF, F, H, C, Challenger>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    /// Create a new writer that borrows the WHIR protocol configuration.
    pub const fn new(params: &'a WhirConfig<EF, F, H, C, Challenger>) -> Self {
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
    pub fn commit<Dft, P, W, PW, const DIGEST_ELEMS: usize>(
        &self,
        dft: &Dft,
        proof: &mut WhirProof<F, EF, W, DIGEST_ELEMS>,
        challenger: &mut Challenger,
        statement: &mut InitialStatement<F, EF>,
    ) -> Result<MerkleTree<F, W, DenseMatrix<F>, DIGEST_ELEMS>, FiatShamirError>
    where
        Dft: TwoAdicSubgroupDft<F>,
        P: PackedValue<Value = F> + Eq + Send + Sync,
        W: PackedValue<Value = W> + Eq + Send + Sync,
        PW: PackedValue<Value = W> + Eq + Send + Sync,
        H: CryptographicHasher<F, [W; DIGEST_ELEMS]>
            + CryptographicHasher<P, [PW; DIGEST_ELEMS]>
            + Sync,
        C: PseudoCompressionFunction<[W; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[PW; DIGEST_ELEMS], 2>
            + Sync,
        Challenger: CanObserve<Hash<F, W, DIGEST_ELEMS>>,
        [W; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        // Transpose for reverse variable order
        // And then pad with zeros

        let padded = info_span!("transpose & pad").in_scope(|| {
            let num_vars = statement.num_variables();
            let mut mat = RowMajorMatrixView::new(
                statement.poly.as_slice(),
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

        // Commit to the Merkle tree (using P for leaves and PW for digest SIMD)
        let merkle_tree = MerkleTreeMmcs::<P, PW, H, C, DIGEST_ELEMS>::new(
            self.merkle_hash.clone(),
            self.merkle_compress.clone(),
        );
        let (root, prover_data) =
            info_span!("commit_matrix").in_scope(|| merkle_tree.commit_matrix(folded_matrix));

        proof.initial_commitment = *root.as_ref();
        // Use CanObserve<Hash<F, W, N>> which both DuplexChallenger and SerializingChallenger implement
        challenger.observe(root);

        // TODO: consider moving ood sampling to whir::Prover::prove
        let mut ood_statement = EqStatement::initialize(self.num_variables);
        (0..self.0.commitment_ood_samples).for_each(|_| {
            // Generate OOD points from ProverState randomness
            let point = MultilinearPoint::expand_from_univariate(
                challenger.sample_algebra_element(),
                self.num_variables,
            );
            let eval = info_span!("ood evaluation").in_scope(|| statement.evaluate(&point));
            proof.initial_ood_answers.push(eval);
            challenger.observe_algebra_element(eval);
            ood_statement.add_evaluated_constraint(point, eval);
        });

        // Return the prover data
        Ok(prover_data)
    }
}

impl<EF, F, H, C, Challenger> Deref for CommitmentWriter<'_, EF, F, H, C, Challenger>
where
    F: Field,
    EF: ExtensionField<F>,
{
    type Target = WhirConfig<EF, F, H, C, Challenger>;

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
        fiat_shamir::domain_separator::DomainSeparator,
        parameters::{FoldingFactor, ProtocolParameters, errors::SecurityAssumption},
        poly::evals::EvaluationsList,
        whir::parameters::SumcheckStrategy,
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
        let params = WhirConfig::<F, F, MyHash, MyCompress, MyChallenger>::new(
            num_variables,
            whir_params.clone(),
        );

        // Generate a random polynomial with 32 coefficients.
        let mut rng = SmallRng::seed_from_u64(1);
        let polynomial = EvaluationsList::<BabyBear>::new(vec![rng.random(); 32]);

        let mut proof =
            WhirProof::<F, F, F, 8>::from_protocol_parameters(&whir_params, num_variables);

        // Set up the DomainSeparator and initialize a ProverState narg_string.
        let mut domainsep: DomainSeparator<F, F> = DomainSeparator::new(vec![]);
        domainsep.commit_statement::<_, _, _, 8>(&params);
        domainsep.add_whir_proof::<_, _, _, 8>(&params);

        let mut rng = SmallRng::seed_from_u64(1);
        let mut challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));
        domainsep.observe_domain_separator(&mut challenger);

        let mut statement = params.initial_statement(polynomial, SumcheckStrategy::Classic);
        // Run the Commitment Phase
        let committer = CommitmentWriter::new(&params);
        let dft = Radix2DFTSmallBatch::<F>::default();
        let _ = committer
            .commit::<_, <F as Field>::Packing, F, <F as Field>::Packing, 8>(
                &dft,
                &mut proof,
                &mut challenger,
                &mut statement,
            )
            .unwrap();

        // Ensure OOD (out-of-domain) points are generated.
        assert!(!statement.is_empty(), "OOD points should be generated");

        // Validate the number of generated OOD points.
        assert_eq!(
            statement.len(),
            params.commitment_ood_samples,
            "OOD points count should match expected samples"
        );

        // Check that OOD answers match expected evaluations
        let poly = &statement.poly;
        let statement = statement.normalize();
        for (i, (ood_point, ood_eval)) in statement.iter().enumerate() {
            let expected_eval = poly.evaluate_hypercube_base(ood_point);
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

        let params = WhirConfig::<F, F, MyHash, MyCompress, MyChallenger>::new(
            num_variables,
            whir_params.clone(),
        );

        let mut rng = SmallRng::seed_from_u64(1);
        let polynomial = EvaluationsList::<BabyBear>::new(vec![rng.random(); 1024]);

        let mut proof =
            WhirProof::<F, F, F, 8>::from_protocol_parameters(&whir_params, num_variables);

        let mut domainsep = DomainSeparator::new(vec![]);
        domainsep.commit_statement::<_, _, _, 8>(&params);

        let mut rng = SmallRng::seed_from_u64(1);
        let mut challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));
        domainsep.observe_domain_separator(&mut challenger);

        let mut statement = params.initial_statement(polynomial, SumcheckStrategy::Classic);
        let dft = Radix2DFTSmallBatch::<F>::default();
        let committer = CommitmentWriter::new(&params);
        let _ = committer
            .commit::<_, <F as Field>::Packing, F, <F as Field>::Packing, 8>(
                &dft,
                &mut proof,
                &mut challenger,
                &mut statement,
            )
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

        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);

        let merkle_hash = MyHash::new(perm.clone());
        let merkle_compress = MyCompress::new(perm);

        let whir_params = ProtocolParameters {
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

        let mut params = WhirConfig::<F, F, MyHash, MyCompress, MyChallenger>::new(
            num_variables,
            whir_params.clone(),
        );

        // Explicitly set OOD samples to 0
        params.commitment_ood_samples = 0;

        let mut rng = SmallRng::seed_from_u64(1);
        let polynomial = EvaluationsList::<BabyBear>::new(vec![rng.random(); 32]);

        let mut proof =
            WhirProof::<F, F, F, 8>::from_protocol_parameters(&whir_params, num_variables);

        let mut domainsep = DomainSeparator::new(vec![]);
        domainsep.commit_statement::<_, _, _, 8>(&params);

        let mut rng = SmallRng::seed_from_u64(1);
        let mut challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));

        domainsep.observe_domain_separator(&mut challenger);

        let mut statement = params.initial_statement(polynomial, SumcheckStrategy::Classic);
        let dft = Radix2DFTSmallBatch::<F>::default();
        let committer = CommitmentWriter::new(&params);
        let _ = committer
            .commit::<_, <F as Field>::Packing, F, <F as Field>::Packing, 8>(
                &dft,
                &mut proof,
                &mut challenger,
                &mut statement,
            )
            .unwrap();

        assert!(
            statement.is_empty(),
            "There should be no OOD points when committment_ood_samples is 0"
        );
    }
}
