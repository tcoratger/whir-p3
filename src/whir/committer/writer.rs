use core::ops::Deref;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::{
    Matrix,
    dense::{DenseMatrix, RowMajorMatrixView},
};
use p3_multilinear_util::{evals::EvaluationsList, multilinear::MultilinearPoint};
use tracing::{info_span, instrument};

use crate::{
    constraints::statement::{EqStatement, initial::InitialStatement},
    fiat_shamir::errors::FiatShamirError,
    parameters::WhirConfig,
    whir::proof::{BatchWhirProof, WhirProof},
};

/// Responsible for committing polynomials using a Merkle-based scheme.
///
/// The `Committer` processes a polynomial, expands and folds its evaluations,
/// and constructs a Merkle tree from the resulting values.
///
/// It provides a commitment that can be used for proof generation and verification.
#[derive(Debug)]
pub struct CommitmentWriter<'a, EF, F, MT: Mmcs<F>, Challenger>(
    /// Reference to the WHIR protocol configuration.
    &'a WhirConfig<EF, F, MT, Challenger>,
)
where
    F: Field,
    EF: ExtensionField<F>;

impl<'a, EF, F, MT, Challenger> CommitmentWriter<'a, EF, F, MT, Challenger>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    MT: Mmcs<F>,
{
    /// Create a new writer that borrows the WHIR protocol configuration.
    pub const fn new(params: &'a WhirConfig<EF, F, MT, Challenger>) -> Self {
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
    pub fn commit<Dft>(
        &self,
        dft: &Dft,
        proof: &mut WhirProof<F, EF, MT>,
        challenger: &mut Challenger,
        statement: &mut InitialStatement<F, EF>,
    ) -> Result<MT::ProverData<DenseMatrix<F>>, FiatShamirError>
    where
        Dft: TwoAdicSubgroupDft<F>,
        Challenger: CanObserve<MT::Commitment>,
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

        let (root, prover_data) =
            info_span!("commit_matrix").in_scope(|| self.mmcs.commit_matrix(folded_matrix));

        proof.initial_commitment = Some(root.clone());
        // Use CanObserve<Hash<F, W, N>> which both DuplexChallenger and SerializingChallenger implement
        challenger.observe(root);

        // TODO: consider moving ood sampling to whir::Prover::prove
        (0..self.0.commitment_ood_samples).for_each(|_| {
            // Generate OOD points from ProverState randomness
            let point = MultilinearPoint::expand_from_univariate(
                challenger.sample_algebra_element(),
                self.num_variables,
            );
            let eval = info_span!("ood evaluation").in_scope(|| statement.evaluate(&point));
            proof.initial_ood_answers.push(eval);
            challenger.observe_algebra_element(eval);
        });

        // Return the prover data
        Ok(prover_data)
    }
}

impl<EF, F, MT: Mmcs<F>, Challenger> Deref for CommitmentWriter<'_, EF, F, MT, Challenger>
where
    F: Field,
    EF: ExtensionField<F>,
{
    type Target = WhirConfig<EF, F, MT, Challenger>;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

/// Batch commitment writer for committing two polynomials into separate Merkle trees
/// and sampling shared OOD points.
#[derive(Debug)]
pub struct BatchCommitmentWriter<'a, EF, F, MT: Mmcs<F>, Challenger>(
    &'a WhirConfig<EF, F, MT, Challenger>,
)
where
    F: Field,
    EF: ExtensionField<F>;

/// Result of batch commitment: prover data and OOD statements for both polynomials.
#[derive(Debug)]
pub struct BatchCommitmentData<EF, F: Send + Sync + Clone, MT: Mmcs<F>> {
    pub prover_data_a: MT::ProverData<DenseMatrix<F>>,
    pub prover_data_b: MT::ProverData<DenseMatrix<F>>,
    pub ood_statement_a: EqStatement<EF>,
    pub ood_statement_b: EqStatement<EF>,
}

impl<'a, EF, F, MT, Challenger> BatchCommitmentWriter<'a, EF, F, MT, Challenger>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    MT: Mmcs<F>,
{
    pub const fn new(params: &'a WhirConfig<EF, F, MT, Challenger>) -> Self {
        Self(params)
    }

    /// Commits two polynomials into separate Merkle trees and samples shared OOD points.
    ///
    /// Both roots are observed into the transcript before OOD sampling, ensuring
    /// the verifier can replay the same transcript.
    #[instrument(skip_all)]
    pub fn commit<Dft>(
        &self,
        dft: &Dft,
        proof: &mut BatchWhirProof<F, EF, MT>,
        challenger: &mut Challenger,
        poly_a: &EvaluationsList<F>,
        poly_b: &EvaluationsList<F>,
    ) -> BatchCommitmentData<EF, F, MT>
    where
        Dft: TwoAdicSubgroupDft<F>,
        Challenger: CanObserve<MT::Commitment>,
    {
        let num_variables = poly_a.num_variables();
        assert_eq!(num_variables, poly_b.num_variables());

        let (root_a, prover_data_a) = self.commit_single(dft, poly_a, challenger);
        let (root_b, prover_data_b) = self.commit_single(dft, poly_b, challenger);

        proof.commitment_a = Some(root_a);
        proof.commitment_b = Some(root_b);

        let (ood_statement_a, ood_statement_b) =
            self.sample_batch_ood(challenger, proof, poly_a, poly_b, num_variables);

        BatchCommitmentData {
            prover_data_a,
            prover_data_b,
            ood_statement_a,
            ood_statement_b,
        }
    }

    /// Commit a single polynomial: RS-encode, build Merkle tree, observe root.
    fn commit_single<Dft>(
        &self,
        dft: &Dft,
        poly: &EvaluationsList<F>,
        challenger: &mut Challenger,
    ) -> (MT::Commitment, MT::ProverData<DenseMatrix<F>>)
    where
        Dft: TwoAdicSubgroupDft<F>,
        Challenger: CanObserve<MT::Commitment>,
    {
        let num_vars = poly.num_variables();
        let mut mat = RowMajorMatrixView::new(
            poly.as_slice(),
            1 << (num_vars - self.0.folding_factor.at_round(0)),
        )
        .transpose();
        mat.pad_to_height(
            1 << (num_vars + self.0.starting_log_inv_rate - self.0.folding_factor.at_round(0)),
            F::ZERO,
        );

        let folded_matrix = dft.dft_batch(mat).to_row_major_matrix();
        let (root, prover_data) = self.0.mmcs.commit_matrix(folded_matrix);
        challenger.observe(root.clone());
        (root, prover_data)
    }

    /// Sample OOD points for both polynomials after both commitments are observed.
    fn sample_batch_ood(
        &self,
        challenger: &mut Challenger,
        proof: &mut BatchWhirProof<F, EF, MT>,
        poly_a: &EvaluationsList<F>,
        poly_b: &EvaluationsList<F>,
        num_variables: usize,
    ) -> (EqStatement<EF>, EqStatement<EF>) {
        let mut ood_a = EqStatement::initialize(num_variables);
        let mut ood_b = EqStatement::initialize(num_variables);

        for _ in 0..self.0.commitment_ood_samples {
            let point = MultilinearPoint::expand_from_univariate(
                challenger.sample_algebra_element(),
                num_variables,
            );
            let eval_a = poly_a.evaluate_hypercube_base(&point);
            let eval_b = poly_b.evaluate_hypercube_base(&point);
            challenger.observe_algebra_element(eval_a);
            challenger.observe_algebra_element(eval_b);

            proof.initial_ood_answers[0].push(eval_a);
            proof.initial_ood_answers[1].push(eval_b);
            ood_a.add_evaluated_constraint(point.clone(), eval_a);
            ood_b.add_evaluated_constraint(point, eval_b);
        }

        (ood_a, ood_b)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_dft::Radix2DFTSmallBatch;
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_multilinear_util::evals::EvaluationsList;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::{RngExt, SeedableRng, rngs::SmallRng};

    use super::*;
    use crate::{
        fiat_shamir::domain_separator::DomainSeparator,
        parameters::{FoldingFactor, ProtocolParameters, SecurityAssumption, SumcheckStrategy},
    };

    type F = BabyBear;
    type Perm = Poseidon2BabyBear<16>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;
    type PackedF = <F as Field>::Packing;
    type MyMmcs = MerkleTreeMmcs<PackedF, PackedF, MyHash, MyCompress, 2, 8>;

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
        let mmcs = MyMmcs::new(merkle_hash, merkle_compress, 0);

        let whir_params = ProtocolParameters {
            security_level,
            pow_bits,
            rs_domain_initial_reduction_factor: 1,
            folding_factor: FoldingFactor::ConstantFromSecondRound(
                first_round_folding_factor,
                folding_factor,
            ),
            mmcs,
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: starting_rate,
        };

        // Define multivariate parameters for the polynomial.
        let params =
            WhirConfig::<F, F, MyMmcs, MyChallenger>::new(num_variables, whir_params.clone());

        // Generate a random polynomial with 32 coefficients.
        let mut rng = SmallRng::seed_from_u64(1);
        let polynomial = EvaluationsList::<BabyBear>::new(vec![rng.random(); 32]);

        let mut proof =
            WhirProof::<F, F, MyMmcs>::from_protocol_parameters(&whir_params, num_variables);

        // Set up the DomainSeparator and initialize a ProverState narg_string.
        let mut domainsep: DomainSeparator<F, F> = DomainSeparator::new(vec![]);
        domainsep.commit_statement::<_, _, 8>(&params);
        domainsep.add_whir_proof::<_, _, 8>(&params);

        let mut rng = SmallRng::seed_from_u64(1);
        let mut challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));
        domainsep.observe_domain_separator(&mut challenger);

        let mut statement = params.initial_statement(polynomial, SumcheckStrategy::Classic);
        // Run the Commitment Phase
        let committer = CommitmentWriter::new(&params);
        let dft = Radix2DFTSmallBatch::<F>::default();
        let _ = committer
            .commit(&dft, &mut proof, &mut challenger, &mut statement)
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
        let mmcs = MyMmcs::new(merkle_hash, merkle_compress, 0);

        let whir_params = ProtocolParameters {
            security_level,
            pow_bits,
            rs_domain_initial_reduction_factor: 1,
            folding_factor: FoldingFactor::ConstantFromSecondRound(
                first_round_folding_factor,
                folding_factor,
            ),
            mmcs,
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: starting_rate,
        };

        let params =
            WhirConfig::<F, F, MyMmcs, MyChallenger>::new(num_variables, whir_params.clone());

        let mut rng = SmallRng::seed_from_u64(1);
        let polynomial = EvaluationsList::<BabyBear>::new(vec![rng.random(); 1024]);

        let mut proof =
            WhirProof::<F, F, MyMmcs>::from_protocol_parameters(&whir_params, num_variables);

        let mut domainsep = DomainSeparator::new(vec![]);
        domainsep.commit_statement::<_, _, 8>(&params);

        let mut rng = SmallRng::seed_from_u64(1);
        let mut challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));
        domainsep.observe_domain_separator(&mut challenger);

        let mut statement = params.initial_statement(polynomial, SumcheckStrategy::Classic);
        let dft = Radix2DFTSmallBatch::<F>::default();
        let committer = CommitmentWriter::new(&params);
        let _ = committer
            .commit(&dft, &mut proof, &mut challenger, &mut statement)
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
        let mmcs = MyMmcs::new(merkle_hash, merkle_compress, 0);

        let whir_params = ProtocolParameters {
            security_level,
            pow_bits,
            rs_domain_initial_reduction_factor: 1,
            folding_factor: FoldingFactor::ConstantFromSecondRound(
                first_round_folding_factor,
                folding_factor,
            ),
            mmcs,
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: starting_rate,
        };

        let mut params =
            WhirConfig::<F, F, MyMmcs, MyChallenger>::new(num_variables, whir_params.clone());

        // Explicitly set OOD samples to 0
        params.commitment_ood_samples = 0;

        let mut rng = SmallRng::seed_from_u64(1);
        let polynomial = EvaluationsList::<BabyBear>::new(vec![rng.random(); 32]);

        let mut proof =
            WhirProof::<F, F, MyMmcs>::from_protocol_parameters(&whir_params, num_variables);

        let mut domainsep = DomainSeparator::new(vec![]);
        domainsep.commit_statement::<_, _, 8>(&params);

        let mut rng = SmallRng::seed_from_u64(1);
        let mut challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));

        domainsep.observe_domain_separator(&mut challenger);

        let mut statement = params.initial_statement(polynomial, SumcheckStrategy::Classic);
        let dft = Radix2DFTSmallBatch::<F>::default();
        let committer = CommitmentWriter::new(&params);
        let _ = committer
            .commit(&dft, &mut proof, &mut challenger, &mut statement)
            .unwrap();

        assert!(
            statement.is_empty(),
            "There should be no OOD points when committment_ood_samples is 0"
        );
    }
}
