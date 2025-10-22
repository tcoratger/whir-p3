use std::{ops::Deref, sync::Arc};

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use serde::{Deserialize, Serialize};
use tracing::{info_span, instrument};

use super::Witness;
use crate::{
    fiat_shamir::{errors::FiatShamirError, prover::ProverState},
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    whir::{committer::DenseMatrix, constraints::statement::Statement, parameters::WhirConfig},
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
    pub fn commit<Dft: TwoAdicSubgroupDft<F>, const DIGEST_ELEMS: usize>(
        &self,
        dft: &Dft,
        prover_state: &mut ProverState<F, EF, Challenger>,
        polynomial: EvaluationsList<F>,
    ) -> Result<Witness<EF, F, DenseMatrix<F>, DIGEST_ELEMS>, FiatShamirError>
    where
        H: CryptographicHasher<F, [F; DIGEST_ELEMS]>
            + CryptographicHasher<F::Packing, [F::Packing; DIGEST_ELEMS]>
            + Sync,
        C: PseudoCompressionFunction<[F; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[F::Packing; DIGEST_ELEMS], 2>
            + Sync,
        [F; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        // Pad with zeros
        let n = polynomial.num_evals();
        let padded = info_span!("pad evals").in_scope(|| {
            let mut padded = F::zero_vec(n * (1 << self.starting_log_inv_rate));
            padded[..n].copy_from_slice(polynomial.as_slice());
            RowMajorMatrix::new(padded, 1 << self.folding_factor.at_round(0))
        });

        // Perform DFT on the padded evaluations matrix
        let folded_matrix = info_span!("dft", height = padded.height(), width = padded.width())
            .in_scope(|| dft.dft_batch(padded).to_row_major_matrix());

        // Commit to the Merkle tree
        let merkle_tree = MerkleTreeMmcs::<F::Packing, F::Packing, H, C, DIGEST_ELEMS>::new(
            self.merkle_hash.clone(),
            self.merkle_compress.clone(),
        );
        let (root, prover_data) =
            info_span!("commit_matrix").in_scope(|| merkle_tree.commit_matrix(folded_matrix));

        prover_state.add_base_scalars(root.as_ref());

        let mut ood_statement = Statement::initialize(self.num_variables);
        (0..self.0.commitment_ood_samples).for_each(|_| {
            // Generate OOD points from ProverState randomness
            let point =
                MultilinearPoint::expand_from_univariate(prover_state.sample(), self.num_variables);
            let eval = info_span!("ood evaluation").in_scope(|| polynomial.evaluate(&point));
            prover_state.add_extension_scalar(eval);
            ood_statement.add_evaluated_constraint(point, eval);
        });

        // Return the witness containing the polynomial, Merkle tree, and OOD results.
        Ok(Witness {
            polynomial,
            prover_data: Arc::new(prover_data),
            ood_statement,
        })
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
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_dft::Radix2DFTSmallBatch;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::{Rng, SeedableRng, rngs::SmallRng};

    use super::*;
    use crate::{
        fiat_shamir::domain_separator::DomainSeparator,
        parameters::{FoldingFactor, ProtocolParameters, errors::SecurityAssumption},
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
            initial_statement: true,
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
            univariate_skip: false,
        };

        // Define multivariate parameters for the polynomial.
        let params =
            WhirConfig::<F, F, MyHash, MyCompress, MyChallenger>::new(num_variables, whir_params);

        // Generate a random polynomial with 32 coefficients.
        let mut rng = rand::rng();
        let polynomial = EvaluationsList::<BabyBear>::new(vec![rng.random(); 32]);

        // Set up the DomainSeparator and initialize a ProverState narg_string.
        let mut domainsep: DomainSeparator<F, F> = DomainSeparator::new(vec![]);
        domainsep.commit_statement::<_, _, _, 8>(&params);
        domainsep.add_whir_proof::<_, _, _, 8>(&params);

        let mut rng = SmallRng::seed_from_u64(1);
        let challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));

        let mut prover_state = domainsep.to_prover_state(challenger);

        // Run the Commitment Phase
        let committer = CommitmentWriter::new(&params);
        let dft = Radix2DFTSmallBatch::<F>::default();
        let witness = committer
            .commit(&dft, &mut prover_state, polynomial.clone())
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
            let expected_eval = polynomial.evaluate(ood_point);
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
            initial_statement: true,
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
            univariate_skip: false,
        };

        let params =
            WhirConfig::<F, F, MyHash, MyCompress, MyChallenger>::new(num_variables, whir_params);

        let mut rng = rand::rng();
        let polynomial = EvaluationsList::<BabyBear>::new(vec![rng.random(); 1024]);

        let mut domainsep = DomainSeparator::new(vec![]);
        domainsep.commit_statement::<_, _, _, 8>(&params);

        let mut rng = SmallRng::seed_from_u64(1);
        let challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));

        let mut prover_state = domainsep.to_prover_state(challenger);

        let dft = Radix2DFTSmallBatch::<F>::default();
        let committer = CommitmentWriter::new(&params);
        let _ = committer
            .commit(&dft, &mut prover_state, polynomial)
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
            initial_statement: true,
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
            univariate_skip: false,
        };

        let mut params =
            WhirConfig::<F, F, MyHash, MyCompress, MyChallenger>::new(num_variables, whir_params);

        // Explicitly set OOD samples to 0
        params.commitment_ood_samples = 0;

        let mut rng = rand::rng();
        let polynomial = EvaluationsList::<BabyBear>::new(vec![rng.random(); 32]);

        let mut domainsep = DomainSeparator::new(vec![]);
        domainsep.commit_statement::<_, _, _, 8>(&params);

        let mut rng = SmallRng::seed_from_u64(1);
        let challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));

        let mut prover_state = domainsep.to_prover_state(challenger);

        let dft = Radix2DFTSmallBatch::<F>::default();
        let committer = CommitmentWriter::new(&params);
        let witness = committer
            .commit(&dft, &mut prover_state, polynomial)
            .unwrap();

        assert!(
            witness.ood_statement.is_empty(),
            "There should be no OOD points when committment_ood_samples is 0"
        );
    }
}
