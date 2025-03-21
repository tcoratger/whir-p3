use super::{parameters::WhirConfig, utils::sample_ood_points};
use crate::{
    merkle_tree::WhirChallenger,
    ntt::expand_from_coeff,
    poly::{coeffs::CoefficientList, fold::transform_evaluations},
};
use p3_challenger::CanObserve;
use p3_commit::Mmcs;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField32, TwoAdicField};
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix};
use p3_merkle_tree::{MerkleTree, MerkleTreeMmcs};
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use serde::{Deserialize, Serialize};

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
pub struct Committer<F, PowStrategy, H, C>(WhirConfig<F, PowStrategy, H, C>)
where
    F: Field + TwoAdicField,
    <F as PrimeCharacteristicRing>::PrimeSubfield: TwoAdicField;

impl<F, PowStrategy, H, C> Committer<F, PowStrategy, H, C>
where
    F: Field + TwoAdicField + PrimeField32 + Eq,
    <F as PrimeCharacteristicRing>::PrimeSubfield: TwoAdicField,
{
    pub const fn new(config: WhirConfig<F, PowStrategy, H, C>) -> Self {
        Self(config)
    }

    pub fn commit<const DIGEST_ELEMS: usize>(
        &self,
        challenger: &mut WhirChallenger<F>,
        polynomial: CoefficientList<<F as PrimeCharacteristicRing>::PrimeSubfield>,
    ) -> Witness<F, H, C, DIGEST_ELEMS>
    where
        H: CryptographicHasher<F, [F; DIGEST_ELEMS]>
            + CryptographicHasher<<F as Field>::Packing, [<F as Field>::Packing; DIGEST_ELEMS]>
            + Sync,
        C: PseudoCompressionFunction<[F; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[<F as Field>::Packing; DIGEST_ELEMS], 2>
            + Sync,
        [F; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
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

        // Convert folded evaluations into a RowMajorMatrix to satisfy the `Matrix<F>` trait
        let folded_matrix = RowMajorMatrix::new(folded_evals.clone(), 1); // 1 row

        // Commit to the Merkle tree
        let merkle_tree =
            MerkleTreeMmcs::new(self.0.merkle_hash.clone(), self.0.merkle_compress.clone());
        let (root, prover_data) = merkle_tree.commit(vec![folded_matrix]);

        // Observe Merkle root in challenger
        challenger.observe_slice(root.as_ref());

        // Handle OOD (Out-Of-Domain) samples
        let (ood_points, ood_answers) = sample_ood_points(
            challenger,
            self.0.committment_ood_samples,
            self.0.mv_parameters.num_variables,
            |point| polynomial.evaluate_at_extension(point),
        );

        Witness {
            polynomial: polynomial.to_extension(),
            merkle_tree,
            prover_data,
            merkle_leaves: folded_evals,
            ood_points,
            ood_answers,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        merkle_tree::{Poseidon2Compression, Poseidon2Sponge},
        parameters::{
            FoldType, FoldingFactor, MultivariateParameters, SoundnessType, WhirParameters,
        },
        poly::multilinear::MultilinearPoint,
    };
    use p3_baby_bear::{
        BabyBear, Poseidon2BabyBear, default_babybear_poseidon2_16, default_babybear_poseidon2_24,
    };
    use p3_challenger::{HashChallenger, SerializingChallenger32};
    use p3_keccak::Keccak256Hash;
    use rand::Rng;

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
            _pow_parameters: std::marker::PhantomData::<u8>,
            starting_log_inv_rate: starting_rate,
        };

        // Define multivariate parameters for the polynomial.
        let mv_params = MultivariateParameters::<F>::new(num_variables);
        let params = WhirConfig::new(mv_params, whir_params);

        // Generate a random polynomial with 32 coefficients.
        let mut rng = rand::rng();
        let polynomial = CoefficientList::<BabyBear>::new(vec![rng.random(); 32]);

        // Setup challenger
        let hasher = Keccak256Hash {};
        let mut challenger = SerializingChallenger32::<
            BabyBear,
            HashChallenger<u8, Keccak256Hash, 32>,
        >::from_hasher(vec![], hasher);

        // Run the Commitment Phase
        let committer = Committer::new(params.clone());
        let witness = committer.commit(&mut challenger, polynomial.clone());

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

    #[test]
    fn test_large_polynomial() {
        type F = BabyBear;

        let security_level = 100;
        let pow_bits = 20;
        let num_variables = 10;
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
            _pow_parameters: std::marker::PhantomData::<u8>,
            starting_log_inv_rate: starting_rate,
        };

        let mv_params = MultivariateParameters::<F>::new(num_variables);
        let params = WhirConfig::new(mv_params, whir_params);

        let mut rng = rand::rng();
        let polynomial = CoefficientList::<BabyBear>::new(vec![rng.random(); 1024]);

        // Setup challenger
        let hasher = Keccak256Hash {};
        let mut challenger = SerializingChallenger32::<
            BabyBear,
            HashChallenger<u8, Keccak256Hash, 32>,
        >::from_hasher(vec![], hasher);

        let committer = Committer::new(params);
        let witness = committer.commit(&mut challenger, polynomial);

        // Expansion factor is 2
        assert_eq!(
            witness.merkle_leaves.len(),
            1024 * 2,
            "Merkle tree should have expected number of leaves"
        );
    }

    #[test]
    fn test_commitment_without_ood_samples() {
        type F = BabyBear;

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
            _pow_parameters: std::marker::PhantomData::<u8>,
            starting_log_inv_rate: starting_rate,
        };

        let mv_params = MultivariateParameters::<F>::new(num_variables);
        let mut params = WhirConfig::new(mv_params, whir_params);

        // Explicitly set OOD samples to 0
        params.committment_ood_samples = 0;

        let mut rng = rand::rng();
        let polynomial = CoefficientList::<BabyBear>::new(vec![rng.random(); 32]);

        // Setup challenger
        let hasher = Keccak256Hash {};
        let mut challenger = SerializingChallenger32::<
            BabyBear,
            HashChallenger<u8, Keccak256Hash, 32>,
        >::from_hasher(vec![], hasher);

        let committer = Committer::new(params);
        let witness = committer.commit(&mut challenger, polynomial);

        // Ensure there are no OOD points
        assert!(
            witness.ood_points.is_empty(),
            "There should be no OOD points when committment_ood_samples is 0"
        );
    }
}
