use p3_commit::Mmcs;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField32, TwoAdicField};
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix};
use p3_merkle_tree::{MerkleTree, MerkleTreeMmcs};
use p3_symmetric::{CryptographicHasher, Hash, PseudoCompressionFunction};
use serde::{Deserialize, Serialize};

use super::{
    parameters::WhirConfig,
    utils::{DigestWriter, sample_ood_points},
};
use crate::{
    fiat_shamir::{
        codecs::traits::{FieldToUnit, UnitToField},
        errors::ProofResult,
        traits::ByteWriter,
    },
    ntt::expand_from_coeff,
    poly::{coeffs::CoefficientList, fold::transform_evaluations},
};

#[derive(Debug)]
pub struct Witness<F: Field, H, C, const DIGEST_ELEMS: usize> {
    pub(crate) polynomial: CoefficientList<F>,
    pub(crate) merkle_tree:
        MerkleTreeMmcs<<F as Field>::Packing, <F as Field>::Packing, H, C, DIGEST_ELEMS>,
    pub(crate) prover_data: MerkleTree<F, F, DenseMatrix<F>, DIGEST_ELEMS>,
    pub(crate) merkle_leaves: Vec<F>,
    pub(crate) ood_points: Vec<F>,
    pub(crate) ood_answers: Vec<F>,
}

#[derive(Debug)]
pub struct Committer<F, H, C, PowStrategy>(WhirConfig<F, H, C, PowStrategy>)
where
    F: Field + TwoAdicField,
    <F as PrimeCharacteristicRing>::PrimeSubfield: TwoAdicField;

impl<F, H, C, PowStrategy> Committer<F, H, C, PowStrategy>
where
    F: Field + TwoAdicField + PrimeField32 + Eq,
    <F as PrimeCharacteristicRing>::PrimeSubfield: TwoAdicField,
{
    pub const fn new(config: WhirConfig<F, H, C, PowStrategy>) -> Self {
        Self(config)
    }

    pub fn commit<ProverState, const DIGEST_ELEMS: usize>(
        &self,
        prover_state: &mut ProverState,
        polynomial: CoefficientList<<F as PrimeCharacteristicRing>::PrimeSubfield>,
    ) -> ProofResult<Witness<F, H, C, DIGEST_ELEMS>>
    where
        H: CryptographicHasher<F, [F; DIGEST_ELEMS]>
            + CryptographicHasher<<F as Field>::Packing, [<F as Field>::Packing; DIGEST_ELEMS]>
            + Sync,
        C: PseudoCompressionFunction<[F; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[<F as Field>::Packing; DIGEST_ELEMS], 2>
            + Sync,
        [F; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
        ProverState: FieldToUnit<F> + UnitToField<F> + DigestWriter<Hash<F, F, DIGEST_ELEMS>>,
    {
        // Compute domain expansion factor
        let base_domain = self.0.starting_domain.base_domain.unwrap();
        let expansion = base_domain.size() / polynomial.num_coeffs();

        // Expand polynomial coefficients into evaluations over the domain
        let mut evals = expand_from_coeff(polynomial.coeffs(), expansion);

        // Apply structured folding transformation
        transform_evaluations(
            &mut evals,
            self.0.fold_optimisation,
            base_domain.group_gen(),
            base_domain.group_gen_inv(),
            self.0.folding_factor.at_round(0),
        );

        // Convert to extension field (for future rounds)
        let folded_evals: Vec<_> = evals.into_iter().map(F::from_prime_subfield).collect();

        // Determine leaf size based on folding factor.
        let fold_size = 1 << self.0.folding_factor.at_round(0);

        // Convert folded evaluations into a RowMajorMatrix to satisfy the `Matrix<F>` trait
        let folded_matrix = RowMajorMatrix::new(folded_evals.clone(), fold_size);

        // Commit to the Merkle tree
        let merkle_tree =
            MerkleTreeMmcs::new(self.0.merkle_hash.clone(), self.0.merkle_compress.clone());
        let (root, prover_data) = merkle_tree.commit(vec![folded_matrix]);

        // Observe Merkle root in challenger
        prover_state.add_digest(root)?;

        // Handle OOD (Out-Of-Domain) samples
        let (ood_points, ood_answers) = sample_ood_points(
            prover_state,
            self.0.committment_ood_samples,
            self.0.mv_parameters.num_variables,
            |point| polynomial.evaluate_at_extension(point),
        )?;

        Ok(Witness {
            polynomial: polynomial.to_extension(),
            merkle_tree,
            prover_data,
            merkle_leaves: folded_evals,
            ood_points,
            ood_answers,
        })
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::{
        BabyBear, Poseidon2BabyBear, default_babybear_poseidon2_16, default_babybear_poseidon2_24,
    };
    use p3_challenger::{HashChallenger, SerializingChallenger32};
    use p3_keccak::Keccak256Hash;
    use rand::Rng;

    use super::*;
    use crate::{
        fiat_shamir::domain_separator::DomainSeparator,
        merkle_tree::{Poseidon2Compression, Poseidon2Sponge},
        parameters::{
            FoldType, FoldingFactor, MultivariateParameters, SoundnessType, WhirParameters,
        },
        poly::multilinear::MultilinearPoint,
        whir::domainsep::WhirDomainSeparator,
    };

    type Perm16 = Poseidon2BabyBear<16>;
    type Perm24 = Poseidon2BabyBear<24>;

    #[test]
    fn test_basic_commitment() {
        // Define the field type and Merkle tree configuration.
        type F = BabyBear;

        // Set up Whir protocol parameters.
        let security_level = 100;
        let pow_bits = 20;
        let num_variables = 5;
        let starting_rate = 1;
        let folding_factor = 4;
        let first_round_folding_factor = 4;

        let poseidon_p16 = default_babybear_poseidon2_16();
        let poseidon_p24 = default_babybear_poseidon2_24();

        let whir_params = WhirParameters {
            initial_statement: true,
            security_level,
            pow_bits,
            folding_factor: FoldingFactor::ConstantFromSecondRound(
                first_round_folding_factor,
                folding_factor,
            ),
            merkle_hash: Poseidon2Sponge::new(poseidon_p24),
            merkle_compress: Poseidon2Compression::new(poseidon_p16),
            soundness_type: SoundnessType::ConjectureList,
            fold_optimisation: FoldType::ProverHelps,
            starting_log_inv_rate: starting_rate,
        };

        // Define multivariate parameters for the polynomial.
        let mv_params = MultivariateParameters::<F>::new(num_variables);
        let params = WhirConfig::new(mv_params, whir_params);

        // Generate a random polynomial with 32 coefficients.
        let mut rng = rand::rng();
        let polynomial = CoefficientList::<BabyBear>::new(vec![rng.random(); 32]);

        // Set up the DomainSeparator and initialize a ProverState narg_string.
        let io = DomainSeparator::new("🌪️").commit_statement(&params).add_whir_proof(&params);
        let mut prover_state = io.to_prover_state();

        // Run the Commitment Phase
        let committer = Committer::new(params.clone());
        let witness = committer.commit(&mut prover_state, polynomial.clone()).unwrap();

        // Ensure Merkle leaves are correctly generated.
        assert!(!witness.merkle_leaves.is_empty(), "Merkle leaves should not be empty");

        // Ensure OOD (out-of-domain) points are generated.
        assert!(!witness.ood_points.is_empty(), "OOD points should be generated");

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

    // #[test]
    // fn test_large_polynomial() {
    //     type F = BabyBear;

    //     let security_level = 100;
    //     let pow_bits = 20;
    //     let num_variables = 10;
    //     let starting_rate = 1;
    //     let folding_factor = 4;
    //     let first_round_folding_factor = 4;

    //     let poseidon_p16 = default_babybear_poseidon2_16();
    //     let poseidon_p24 = default_babybear_poseidon2_24();

    //     let whir_params = WhirParameters {
    //         initial_statement: true,
    //         security_level,
    //         pow_bits,
    //         folding_factor: FoldingFactor::ConstantFromSecondRound(
    //             first_round_folding_factor,
    //             folding_factor,
    //         ),
    //         merkle_hash: Poseidon2Sponge::new(poseidon_p24),
    //         merkle_compress: Poseidon2Compression::new(poseidon_p16),
    //         soundness_type: SoundnessType::ConjectureList,
    //         fold_optimisation: FoldType::ProverHelps,
    //         starting_log_inv_rate: starting_rate,
    //     };

    //     let mv_params = MultivariateParameters::<F>::new(num_variables);
    //     let params = WhirConfig::new(mv_params, whir_params);

    //     let mut rng = rand::rng();
    //     let polynomial = CoefficientList::<BabyBear>::new(vec![rng.random(); 1024]);

    //     let io = DomainSeparator::new("🌪️").commit_statement(&params);
    //     let mut prover_state = io.to_prover_state();

    //     let committer = Committer::new(params);
    //     let witness = committer.commit(&mut prover_state, polynomial).unwrap();

    //     // Expansion factor is 2
    //     assert_eq!(
    //         witness.merkle_leaves.len(),
    //         1024 * 2,
    //         "Merkle tree should have expected number of leaves"
    //     );
    // }

    // #[test]
    // fn test_commitment_without_ood_samples() {
    //     type F = BabyBear;

    //     let security_level = 100;
    //     let pow_bits = 20;
    //     let num_variables = 5;
    //     let starting_rate = 1;
    //     let folding_factor = 4;
    //     let first_round_folding_factor = 4;

    //     let poseidon_p16 = default_babybear_poseidon2_16();
    //     let poseidon_p24 = default_babybear_poseidon2_24();

    //     let whir_params = WhirParameters {
    //         initial_statement: true,
    //         security_level,
    //         pow_bits,
    //         folding_factor: FoldingFactor::ConstantFromSecondRound(
    //             first_round_folding_factor,
    //             folding_factor,
    //         ),
    //         merkle_hash: Poseidon2Sponge::new(poseidon_p24),
    //         merkle_compress: Poseidon2Compression::new(poseidon_p16),
    //         soundness_type: SoundnessType::ConjectureList,
    //         fold_optimisation: FoldType::ProverHelps,
    //         starting_log_inv_rate: starting_rate,
    //     };

    //     let mv_params = MultivariateParameters::<F>::new(num_variables);
    //     let mut params = WhirConfig::new(mv_params, whir_params);

    //     // Explicitly set OOD samples to 0
    //     params.committment_ood_samples = 0;

    //     let mut rng = rand::rng();
    //     let polynomial = CoefficientList::<BabyBear>::new(vec![rng.random(); 32]);

    //     let io = DomainSeparator::new("🌪️").commit_statement(&params);
    //     let mut prover_state = io.to_prover_state();

    //     let committer = Committer::new(params);
    //     let witness = committer.commit(&mut prover_state, polynomial).unwrap();

    //     assert!(
    //         witness.ood_points.is_empty(),
    //         "There should be no OOD points when committment_ood_samples is 0"
    //     );
    // }
}
