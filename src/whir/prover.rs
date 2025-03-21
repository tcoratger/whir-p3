use super::{WhirProof, committer::Witness, parameters::WhirConfig, statement::Statement};
use crate::{
    domain::Domain,
    merkle_tree::WhirChallenger,
    ntt::expand_from_coeff,
    parameters::FoldType,
    poly::{
        coeffs::CoefficientList,
        fold::{compute_fold, transform_evaluations},
        multilinear::MultilinearPoint,
    },
    sumcheck::sumcheck_single::SumcheckSingle,
    utils::expand_randomness,
    whir::{fs_utils::get_challenge_stir_queries, statement::Weights, utils::sample_ood_points},
};
use p3_challenger::{CanObserve, CanSample, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField32, TwoAdicField};
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix};
use p3_merkle_tree::{MerkleTree, MerkleTreeMmcs};
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use serde::{Deserialize, Serialize};

pub(crate) struct RoundState<F, H, C, const DIGEST_ELEMS: usize>
where
    F: Field + TwoAdicField,
    <F as PrimeCharacteristicRing>::PrimeSubfield: TwoAdicField,
{
    pub(crate) round: usize,
    pub(crate) domain: Domain<F>,
    pub(crate) sumcheck_prover: Option<SumcheckSingle<F>>,
    pub(crate) folding_randomness: MultilinearPoint<F>,
    pub(crate) coefficients: CoefficientList<F>,
    pub(crate) prev_merkle:
        MerkleTreeMmcs<<F as Field>::Packing, <F as Field>::Packing, H, C, DIGEST_ELEMS>,
    pub(crate) prev_merkle_prover_data: MerkleTree<F, F, DenseMatrix<F>, DIGEST_ELEMS>,
    pub(crate) prev_merkle_answers: Vec<F>,
    /// - The first element is the Merkle proof (siblings)
    /// - The second is the opened leaf values
    pub(crate) merkle_proofs: Vec<(Vec<[F; DIGEST_ELEMS]>, Vec<Vec<F>>)>,
    pub(crate) randomness_vec: Vec<F>,
    pub(crate) statement: Statement<F>,
}

#[derive(Debug)]
pub struct Prover<F, PowStrategy, H, C>(pub WhirConfig<F, PowStrategy, H, C>)
where
    F: Field + TwoAdicField,
    <F as PrimeCharacteristicRing>::PrimeSubfield: TwoAdicField;

impl<F, PowStrategy, H, C> Prover<F, PowStrategy, H, C>
where
    F: Field + TwoAdicField,
    <F as PrimeCharacteristicRing>::PrimeSubfield: TwoAdicField,
{
    fn validate_parameters(&self) -> bool {
        self.0.mv_parameters.num_variables ==
            self.0.folding_factor.total_number(self.0.n_rounds()) + self.0.final_sumcheck_rounds
    }

    fn validate_statement(&self, statement: &Statement<F>) -> bool {
        if !statement.num_variables() == self.0.mv_parameters.num_variables {
            return false;
        }
        if !self.0.initial_statement && !statement.constraints.is_empty() {
            return false;
        }
        true
    }

    fn validate_witness<const DIGEST_ELEMS: usize>(
        &self,
        witness: &Witness<F, H, C, DIGEST_ELEMS>,
    ) -> bool {
        assert_eq!(witness.ood_points.len(), witness.ood_answers.len());
        if !self.0.initial_statement {
            assert!(witness.ood_points.is_empty());
        }
        witness.polynomial.num_variables() == self.0.mv_parameters.num_variables
    }

    pub fn prove<const DIGEST_ELEMS: usize>(
        &self,
        challenger: &mut WhirChallenger<F>,
        mut statement: Statement<F>,
        witness: Witness<F, H, C, DIGEST_ELEMS>,
    ) -> WhirProof<F, DIGEST_ELEMS>
    where
        F: PrimeField32,
        H: CryptographicHasher<F, [F; DIGEST_ELEMS]>
            + CryptographicHasher<<F as Field>::Packing, [<F as Field>::Packing; DIGEST_ELEMS]>
            + Sync,
        C: CanSample<F>
            + PseudoCompressionFunction<[F; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[<F as Field>::Packing; DIGEST_ELEMS], 2>
            + Sync,
        [F; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        // Validate parameters
        assert!(
            self.validate_parameters() &&
                self.validate_statement(&statement) &&
                self.validate_witness(&witness)
        );

        // Convert witness ood_points into constraints
        let new_constraints = witness
            .ood_points
            .into_iter()
            .zip(witness.ood_answers)
            .map(|(point, evaluation)| {
                let weights = Weights::evaluation(MultilinearPoint::expand_from_univariate(
                    point,
                    self.0.mv_parameters.num_variables,
                ));
                (weights, evaluation)
            })
            .collect();

        statement.add_constraints_in_front(new_constraints);

        let mut sumcheck_prover = None;
        let folding_randomness = if self.0.initial_statement {
            // Sample the combination randomness if we run sumcheck for an initial statement
            let combination_randomness_gen = challenger.sample();

            // Create the sumcheck prover
            let mut sumcheck = SumcheckSingle::new(
                witness.polynomial.clone(),
                &statement,
                combination_randomness_gen,
            );

            // Compute sumcheck polynomials and return the folding randomness values
            let folding_randomness = sumcheck.compute_sumcheck_polynomials(
                challenger,
                self.0.folding_factor.at_round(0),
                self.0.starting_folding_pow_bits as usize,
            );

            sumcheck_prover = Some(sumcheck);
            folding_randomness
        } else {
            // If there is no initial statement, obtain the initial folding randomness from the
            // challenger
            let folding_randomness = challenger.sample_vec(self.0.folding_factor.at_round(0));

            // Perform proof-of-work (if required)
            if self.0.starting_folding_pow_bits > 0. {
                challenger.grind(self.0.starting_folding_pow_bits as usize);
            }

            MultilinearPoint(folding_randomness)
        };

        let mut randomness_vec = vec![F::ZERO; self.0.mv_parameters.num_variables];
        let mut arr = folding_randomness.clone().0;
        arr.reverse();

        randomness_vec[..folding_randomness.0.len()].copy_from_slice(&arr);

        let round_state = RoundState {
            domain: self.0.starting_domain.clone(),
            round: 0,
            sumcheck_prover,
            folding_randomness,
            coefficients: witness.polynomial,
            prev_merkle: witness.merkle_tree,
            prev_merkle_prover_data: witness.prover_data,
            prev_merkle_answers: witness.merkle_leaves,
            merkle_proofs: vec![],
            randomness_vec,
            statement,
        };

        self.round(challenger, round_state)
    }

    fn round<const DIGEST_ELEMS: usize>(
        &self,
        challenger: &mut WhirChallenger<F>,
        mut round_state: RoundState<F, H, C, DIGEST_ELEMS>,
    ) -> WhirProof<F, DIGEST_ELEMS>
    where
        F: PrimeField32,
        H: CryptographicHasher<F, [F; DIGEST_ELEMS]>
            + CryptographicHasher<<F as Field>::Packing, [<F as Field>::Packing; DIGEST_ELEMS]>
            + Sync,
        C: CanSample<F>
            + PseudoCompressionFunction<[F; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[<F as Field>::Packing; DIGEST_ELEMS], 2>
            + Sync,
        [F; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        // Fold the coefficients
        let folded_coefficients = round_state.coefficients.fold(&round_state.folding_randomness);

        let num_variables = self.0.mv_parameters.num_variables -
            self.0.folding_factor.total_number(round_state.round);
        // num_variables should match the folded_coefficients here.
        assert_eq!(num_variables, folded_coefficients.num_variables());

        // Base case: final round reached
        if round_state.round == self.0.n_rounds() {
            return self.final_round(challenger, round_state, &folded_coefficients);
        }

        let round_params = &self.0.round_parameters[round_state.round];

        // Compute polynomial evaluations and build Merkle tree
        let new_domain = round_state.domain.scale(2);
        let expansion = new_domain.size() / folded_coefficients.num_coeffs();
        let mut evals = expand_from_coeff(folded_coefficients.coeffs(), expansion);
        transform_evaluations(
            &mut evals,
            self.0.fold_optimisation,
            new_domain.backing_domain.group_gen(),
            new_domain.backing_domain.group_gen_inv(),
            self.0.folding_factor.at_round(round_state.round + 1),
        );

        // Convert folded evaluations into a RowMajorMatrix to satisfy the `Matrix<F>` trait
        let folded_matrix = RowMajorMatrix::new(evals.clone(), 1); // 1 row

        let merkle_tree =
            MerkleTreeMmcs::new(self.0.merkle_hash.clone(), self.0.merkle_compress.clone());
        let (root, prover_data) = merkle_tree.commit(vec![folded_matrix]);

        // Observe Merkle root in challenger
        challenger.observe_slice(root.as_ref());

        // Handle OOD (Out-Of-Domain) samples
        let (ood_points, ood_answers) =
            sample_ood_points(challenger, round_params.ood_samples, num_variables, |point| {
                folded_coefficients.evaluate(point)
            });

        // STIR queries
        let stir_challenges_indexes = get_challenge_stir_queries::<F, _>(
            round_state.domain.size(), // Current domain size *before* folding
            self.0.folding_factor.at_round(round_state.round), // Current fold factor
            round_params.num_queries,
            challenger,
        );

        // Compute the generator of the folded domain, in the extension field
        let domain_scaled_gen = round_state
            .domain
            .backing_domain
            .element(1 << self.0.folding_factor.at_round(round_state.round));
        let stir_challenges: Vec<_> = ood_points
            .into_iter()
            .chain(stir_challenges_indexes.iter().map(|i| domain_scaled_gen.exp_u64(*i as u64)))
            .map(|univariate| MultilinearPoint::expand_from_univariate(univariate, num_variables))
            .collect();

        // Collect Merkle proofs for stir queries
        let fold_size = 1 << self.0.folding_factor.at_round(round_state.round);
        let merkle_proofs: Vec<_> = stir_challenges_indexes
            .iter()
            .map(|&index| {
                let auth_path =
                    round_state.prev_merkle.open_batch(index, &round_state.prev_merkle_prover_data);
                let leaf_values = round_state.prev_merkle_answers
                    [index * fold_size..(index + 1) * fold_size]
                    .to_vec();
                (auth_path.1, vec![leaf_values])
            })
            .collect();

        // TODO: try to avoid this duplication: very bad
        let answers: Vec<_> = stir_challenges_indexes
            .iter()
            .map(|i| round_state.prev_merkle_answers[i * fold_size..(i + 1) * fold_size].to_vec())
            .collect();

        // Evaluate answers in the folding randomness.
        let mut stir_evaluations = ood_answers;
        match self.0.fold_optimisation {
            FoldType::Naive => {
                // See `Verifier::compute_folds_full`
                let domain_size = round_state.domain.backing_domain.size();
                let domain_gen = round_state.domain.backing_domain.element(1);
                let domain_gen_inv = domain_gen.inverse();
                let coset_domain_size = 1 << self.0.folding_factor.at_round(round_state.round);
                // The domain (before folding) is split into cosets of size
                // `coset_domain_size` (which is just `fold_size`). Each coset
                // is generated by powers of `coset_generator` (which is just the
                // `fold_size`-root of unity) multiplied by a different
                // `coset_offset`.
                // For example, if `fold_size = 16`, and the domain size is N, then
                // the domain is (1, w, w^2, ..., w^(N-1)), the domain generator
                // is w, and the coset generator is w^(N/16).
                // The first coset is (1, w^(N/16), w^(2N/16), ..., w^(15N/16))
                // which is also a subgroup <w^(N/16)> itself (the coset_offset is 1).
                // The second coset would be w * <w^(N/16)>, the third coset would be
                // w^2 * <w^(N/16)>, and so on. Until w^(N/16-1) * <w^(N/16)>.
                let coset_generator_inv =
                    domain_gen_inv.exp_u64((domain_size / coset_domain_size) as u64);
                stir_evaluations.extend(stir_challenges_indexes.iter().zip(&answers).map(
                    |(index, answers)| {
                        // The coset is w^index * <w_coset_generator>
                        //let _coset_offset = domain_gen.pow(&[*index as u64]);
                        let coset_offset_inv = domain_gen_inv.exp_u64(*index as u64);

                        // In the Naive mode, the oracle consists directly of the
                        // evaluations of f over the domain. We leverage an
                        // algorithm to compute the evaluations of the folded f
                        // at the corresponding point in folded domain (which is
                        // coset_offset^fold_size).
                        compute_fold(
                            answers,
                            &round_state.folding_randomness.0,
                            coset_offset_inv,
                            coset_generator_inv,
                            F::TWO.inverse(),
                            self.0.folding_factor.at_round(round_state.round),
                        )
                    },
                ));
            }
            FoldType::ProverHelps => stir_evaluations.extend(answers.iter().map(|answers| {
                // In the ProverHelps mode, the oracle values have been linearly
                // transformed such that they are exactly the coefficients of the
                // multilinear polynomial whose evaluation at the folding randomness
                // is just the folding of f evaluated at the folded point.
                CoefficientList::new(answers.clone()).evaluate(&round_state.folding_randomness)
            })),
        }
        round_state.merkle_proofs.extend(merkle_proofs);

        // Perform proof-of-work if required
        if round_params.pow_bits > 0. {
            challenger.grind(round_params.pow_bits as usize);
        }

        // Randomness for combination
        let combination_randomness_gen: F = challenger.sample();
        let combination_randomness =
            expand_randomness(combination_randomness_gen, stir_challenges.len());

        #[allow(clippy::map_unwrap_or)]
        let mut sumcheck_prover = round_state
            .sumcheck_prover
            .take()
            .map(|mut sumcheck_prover| {
                sumcheck_prover.add_new_equality(
                    &stir_challenges,
                    &stir_evaluations,
                    &combination_randomness,
                );
                sumcheck_prover
            })
            .unwrap_or_else(|| {
                let mut statement = Statement::<F>::new(folded_coefficients.num_variables());

                for (point, eval) in stir_challenges.into_iter().zip(stir_evaluations) {
                    let weights = Weights::evaluation(point.clone());
                    statement.add_constraint(weights, eval);
                }
                SumcheckSingle::new(
                    folded_coefficients.clone(),
                    &statement,
                    combination_randomness[1],
                )
            });

        let folding_randomness = sumcheck_prover.compute_sumcheck_polynomials(
            challenger,
            self.0.folding_factor.at_round(round_state.round + 1),
            round_params.folding_pow_bits as usize,
        );

        let start_idx = self.0.folding_factor.total_number(round_state.round);
        let mut arr = folding_randomness.clone().0;
        arr.reverse();

        round_state.randomness_vec[start_idx..start_idx + folding_randomness.0.len()]
            .copy_from_slice(&arr);

        let round_state = RoundState {
            round: round_state.round + 1,
            domain: new_domain,
            sumcheck_prover: Some(sumcheck_prover),
            folding_randomness,
            coefficients: folded_coefficients,
            prev_merkle: merkle_tree,
            prev_merkle_answers: evals,
            randomness_vec: round_state.randomness_vec.clone(),
            statement: round_state.statement,
            prev_merkle_prover_data: prover_data,
            merkle_proofs: round_state.merkle_proofs,
        };
        self.round(challenger, round_state)
    }

    fn final_round<const DIGEST_ELEMS: usize>(
        &self,
        challenger: &mut WhirChallenger<F>,
        mut round_state: RoundState<F, H, C, DIGEST_ELEMS>,
        folded_coefficients: &CoefficientList<F>,
    ) -> WhirProof<F, DIGEST_ELEMS>
    where
        F: PrimeField32,
        H: CryptographicHasher<F, [F; DIGEST_ELEMS]>
            + CryptographicHasher<<F as Field>::Packing, [<F as Field>::Packing; DIGEST_ELEMS]>
            + Sync,
        C: CanSample<F>
            + PseudoCompressionFunction<[F; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[<F as Field>::Packing; DIGEST_ELEMS], 2>
            + Sync,
        [F; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        challenger.observe_slice(folded_coefficients.coeffs());

        // Final verifier queries and answers. The indices are over the folded domain.
        let final_challenge_indexes = get_challenge_stir_queries::<F, _>(
            // The size of the original domain before folding
            round_state.domain.size(),
            // The folding factor we used to fold the previous polynomial
            self.0.folding_factor.at_round(round_state.round),
            self.0.final_queries,
            challenger,
        );

        // Every query requires opening these many in the previous Merkle tree
        let fold_size = 1 << self.0.folding_factor.at_round(round_state.round);
        let final_merkle_proofs_and_values: Vec<_> = final_challenge_indexes
            .iter()
            .map(|&index| {
                let auth_path =
                    round_state.prev_merkle.open_batch(index, &round_state.prev_merkle_prover_data);
                let leaf_values = round_state.prev_merkle_answers
                    [index * fold_size..(index + 1) * fold_size]
                    .to_vec();
                (auth_path.1, vec![leaf_values])
            })
            .collect();
        round_state.merkle_proofs.extend(final_merkle_proofs_and_values);

        // Perform proof-of-work if required
        if self.0.final_pow_bits > 0. {
            challenger.grind(self.0.final_pow_bits as usize);
        }

        // Run final sumcheck if required
        if self.0.final_sumcheck_rounds > 0 {
            let final_folding_randomness = round_state
                .sumcheck_prover
                .unwrap_or_else(|| {
                    SumcheckSingle::new(folded_coefficients.clone(), &round_state.statement, F::ONE)
                })
                .compute_sumcheck_polynomials(
                    challenger,
                    self.0.final_sumcheck_rounds,
                    self.0.final_folding_pow_bits as usize,
                );

            let start_idx = self.0.folding_factor.total_number(round_state.round);
            let mut arr = final_folding_randomness.clone().0;
            arr.reverse();

            round_state.randomness_vec[start_idx..start_idx + final_folding_randomness.0.len()]
                .copy_from_slice(&arr);
        }

        let mut randomness_vec_rev = round_state.randomness_vec.clone();
        randomness_vec_rev.reverse();

        let mut statement_values_at_random_point = vec![];
        for (weights, _) in &round_state.statement.constraints {
            if let Weights::Linear { weight } = weights {
                statement_values_at_random_point
                    .push(weight.eval_extension(&MultilinearPoint(randomness_vec_rev.clone())));
            }
        }

        WhirProof { merkle_paths: round_state.merkle_proofs, statement_values_at_random_point }
    }
}
