use std::{ops::Deref, sync::Arc};

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use serde::{Deserialize, Serialize};
use tracing::{info_span, instrument};

use super::Witness;
use crate::{
    dft::EvalsDft,
    poly::evals::EvaluationsList,
    utils::parallel_repeat,
    whir::{committer::DenseMatrix,
           parameters::WhirConfig,
           utils::sample_ood_points,
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
    pub fn commit<const DIGEST_ELEMS: usize>(
        &self,
        dft: &EvalsDft<F>,
        proof: &mut WhirProof<F, EF, DIGEST_ELEMS>,
        challenger: &mut Challenger,
        polynomial: EvaluationsList<F>,
    ) -> Witness<EF, F, DenseMatrix<F>, DIGEST_ELEMS>
    where
        H: CryptographicHasher<F, [F; DIGEST_ELEMS]>
            + CryptographicHasher<F::Packing, [F::Packing; DIGEST_ELEMS]>
            + Sync,
        C: PseudoCompressionFunction<[F; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[F::Packing; DIGEST_ELEMS], 2>
            + Sync,
        [F; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        let evals_repeated = info_span!("repeating evals")
            .in_scope(|| parallel_repeat(polynomial.as_slice(), 1 << self.starting_log_inv_rate));

        // Perform DFT on the padded evaluations matrix
        let width = 1 << self.folding_factor.at_round(0);
        let folded_matrix = info_span!("dft", height = evals_repeated.len() / width, width)
            .in_scope(|| {
                dft.dft_batch_by_evals(RowMajorMatrix::new(evals_repeated, width))
                    .to_row_major_matrix()
            });

        // Commit to the Merkle tree
        let merkle_tree = MerkleTreeMmcs::<F::Packing, F::Packing, H, C, DIGEST_ELEMS>::new(
            self.merkle_hash.clone(),
            self.merkle_compress.clone(),
        );
        let (root, prover_data) =
            info_span!("commit_matrix").in_scope(|| merkle_tree.commit_matrix(folded_matrix));

        // Extend the proof data vector with the initial polynomial commitment
        proof.initial_commitment = *root.as_ref();

        // Notify the challenger that these scalars have been committed.
        challenger.observe_slice(root.as_ref());

        // Handle OOD (Out-Of-Domain) samples
        let (ood_points, ood_answers) = sample_ood_points(
            challenger,
            self.committment_ood_samples,
            self.num_variables,
            |point| info_span!("ood evaluation").in_scope(|| polynomial.evaluate(point)),
        );

        // Observe OOD answers to the transcript
        for answer in &ood_answers {
            challenger.observe_algebra_element(*answer);
        }


        // Return the witness containing the polynomial, Merkle tree, and OOD results.
        Witness {
            polynomial,
            prover_data: Arc::new(prover_data),
            ood_points,
            ood_answers,
        }
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
    use p3_field::PrimeCharacteristicRing;
    use crate::whir::proof::InitialPhase;
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::{CanObserve, DuplexChallenger};
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::{Rng, SeedableRng, rngs::SmallRng};
    use crate::whir::proof::WhirProof;

    use super::*;
    use crate::{
        parameters::{FoldingFactor, ProtocolParameters, errors::SecurityAssumption},
        poly::multilinear::MultilinearPoint,
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
            WhirConfig::<F, F, MyHash, MyCompress, MyChallenger>::new(num_variables, whir_params.clone());

        // Initialize the Whir Proof structure
        let mut proof = WhirProof::from_protocol_parameters(&whir_params, num_variables);

        // Generate a random polynomial with 32 coefficients.
        let mut rng = rand::rng();
        let polynomial = EvaluationsList::<BabyBear>::new(vec![rng.random(); 32]);

        // Set up the Domain separation by observing the current Whir Proof structure
        let mut rng = SmallRng::seed_from_u64(1);
        let mut challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));

        // Observe protocol parameters for domain separation
        challenger.observe(F::from_u64(proof.rounds.len() as u64));
        challenger.observe(F::from_u64(proof.final_queries.len() as u64));

        // Observe initial phase variant (0=WithoutStatement, 1=WithStatement, 2=WithStatementSkip)
        let phase_tag = match &proof.initial_phase {
            InitialPhase::WithoutStatement { .. } => 0,
            InitialPhase::WithStatement { .. } => 1,
            InitialPhase::WithStatementSkip { .. } => 2,
        };
        challenger.observe(F::from_u64(phase_tag as u64));

        // Run the Commitment Phase
        let committer = CommitmentWriter::new(&params);
        let dft_committer = EvalsDft::<F>::default();
        let witness = committer
            .commit(&dft_committer, &mut proof, &mut challenger, polynomial.clone());

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
            WhirConfig::<F, F, MyHash, MyCompress, MyChallenger>::new(num_variables, whir_params.clone());

        // Initialize the Whir Proof structure
        let mut proof = WhirProof::from_protocol_parameters(&whir_params, num_variables);

        let mut rng = rand::rng();
        let polynomial = EvaluationsList::<BabyBear>::new(vec![rng.random(); 1024]);

        // Set up the challenger
        let mut rng = SmallRng::seed_from_u64(1);
        let mut challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));

        // Observe protocol parameters for domain separation
        challenger.observe(F::from_u64(proof.rounds.len() as u64));
        challenger.observe(F::from_u64(proof.final_queries.len() as u64));

        // Observe initial phase variant
        let phase_tag = match &proof.initial_phase {
            InitialPhase::WithoutStatement { .. } => 0,
            InitialPhase::WithStatement { .. } => 1,
            InitialPhase::WithStatementSkip { .. } => 2,
        };
        challenger.observe(F::from_u64(phase_tag as u64));

        let dft_committer = EvalsDft::<F>::default();
        let committer = CommitmentWriter::new(&params);
        let witness = committer
            .commit(&dft_committer, &mut proof, &mut challenger, polynomial);

        // Ensure witness was created successfully
        assert_eq!(
            witness.ood_points.len(),
            params.committment_ood_samples,
            "OOD points count should match expected samples"
        );
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
            WhirConfig::<F, F, MyHash, MyCompress, MyChallenger>::new(num_variables, whir_params.clone());

        // Explicitly set OOD samples to 0
        params.committment_ood_samples = 0;

        // Initialize the Whir Proof structure
        let mut proof = WhirProof::from_protocol_parameters(&whir_params, num_variables);

        let mut rng = rand::rng();
        let polynomial = EvaluationsList::<BabyBear>::new(vec![rng.random(); 32]);

        // Set up the challenger
        let mut rng = SmallRng::seed_from_u64(1);
        let mut challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));

        // Observe protocol parameters for domain separation
        challenger.observe(F::from_u64(proof.rounds.len() as u64));
        challenger.observe(F::from_u64(proof.final_queries.len() as u64));

        // Observe initial phase variant
        let phase_tag = match &proof.initial_phase {
            InitialPhase::WithoutStatement { .. } => 0,
            InitialPhase::WithStatement { .. } => 1,
            InitialPhase::WithStatementSkip { .. } => 2,
        };
        challenger.observe(F::from_u64(phase_tag as u64));

        let dft_committer = EvalsDft::<F>::default();
        let committer = CommitmentWriter::new(&params);
        let witness = committer
            .commit(&dft_committer, &mut proof, &mut challenger, polynomial);

        assert!(
            witness.ood_points.is_empty(),
            "There should be no OOD points when committment_ood_samples is 0"
        );
        assert_eq!(params.committment_ood_samples, 0);
    }
}
