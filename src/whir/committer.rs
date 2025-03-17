use super::parameters::WhirConfig;
use crate::{
    merkle_tree::{Poseidon2MerkleMmcs, WhirChallenger},
    ntt::expand_from_coeff,
    poly::{coeffs::CoefficientList, fold::transform_evaluations, multilinear::MultilinearPoint},
};
use p3_baby_bear::BabyBear;
use p3_challenger::{CanObserve, CanSample};
use p3_commit::Mmcs;
use p3_field::{Field, PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_symmetric::CryptographicPermutation;

#[derive(Debug)]
pub struct Witness<F: Field, Perm16, Perm24> {
    pub(crate) polynomial: CoefficientList<F>,
    pub(crate) merkle_tree: Poseidon2MerkleMmcs<F, Perm16, Perm24>,
    pub(crate) merkle_leaves: Vec<F>,
    pub(crate) ood_points: Vec<F>,
    pub(crate) ood_answers: Vec<F>,
}

#[derive(Debug)]
pub struct Committer<F, PowStrategy, Perm16, Perm24>(WhirConfig<F, PowStrategy, Perm16, Perm24>)
where
    F: Field + TwoAdicField,
    <F as PrimeCharacteristicRing>::PrimeSubfield: TwoAdicField,
    Perm16: CryptographicPermutation<[F; 16]> + CryptographicPermutation<[F::Packing; 16]>,
    Perm24: CryptographicPermutation<[F; 24]> + CryptographicPermutation<[F::Packing; 24]>;

impl<PowStrategy, Perm16, Perm24> Committer<BabyBear, PowStrategy, Perm16, Perm24>
where
    // F: Field +
    // TwoAdicField +
    // Matrix<MontyField31<p3_baby_bear::BabyBearParameters>>,
    //
    // <F as PrimeCharacteristicRing>::PrimeSubfield: TwoAdicField,
    Perm16: CryptographicPermutation<[BabyBear; 16]>
        + CryptographicPermutation<[<BabyBear as Field>::Packing; 16]>,
    Perm24: CryptographicPermutation<[BabyBear; 24]>
        + CryptographicPermutation<[<BabyBear as Field>::Packing; 24]>,
{
    pub const fn new(config: WhirConfig<BabyBear, PowStrategy, Perm16, Perm24>) -> Self {
        Self(config)
    }

    pub fn commit(
        &self,
        challenger: &mut WhirChallenger<BabyBear>,
        polynomial: CoefficientList<<BabyBear as PrimeCharacteristicRing>::PrimeSubfield>,
    ) -> Witness<BabyBear, Perm16, Perm24> {
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
        let folded_evals: Vec<_> = evals.into_iter().map(BabyBear::from_prime_subfield).collect();

        // Convert folded evaluations into a RowMajorMatrix to satisfy the `Matrix<F>` trait
        let folded_matrix = RowMajorMatrix::new(folded_evals.clone(), 1); // 1 row

        // Commit to the Merkle tree
        let merkle_tree = Poseidon2MerkleMmcs::<BabyBear, _, _>::new(
            self.0.merkle_hash.clone(),
            self.0.merkle_compress.clone(),
        );
        let (root, _) = merkle_tree.commit(vec![folded_matrix]);

        // Observe Merkle root in challenger
        challenger.observe_slice(root.as_ref());

        // Sample OOD points
        let ood_points: Vec<_> =
            (0..self.0.committment_ood_samples).map(|_| challenger.sample()).collect();

        // Compute OOD evaluations
        let ood_answers: Vec<_> = ood_points
            .iter()
            .map(|&ood_point| {
                polynomial.evaluate_at_extension(&MultilinearPoint::expand_from_univariate(
                    ood_point,
                    self.0.mv_parameters.num_variables,
                ))
            })
            .collect();

        //  Observe OOD evaluations in challenger
        challenger.observe_slice(&ood_answers);

        Witness {
            polynomial: polynomial.to_extension(),
            merkle_tree,
            merkle_leaves: folded_evals,
            ood_points,
            ood_answers,
        }
    }
}
