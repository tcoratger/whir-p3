use p3_commit::{ExtensionMmcs, Mmcs};
use p3_field::{ExtensionField, Field, PrimeField64, TwoAdicField};
use p3_matrix::{
    dense::{DenseMatrix, RowMajorMatrix},
    extension::FlatMatrixView,
};
use p3_merkle_tree::{MerkleTree, MerkleTreeMmcs};
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use serde::{Deserialize, Serialize};

use super::{WhirProof, committer::Witness, parameters::WhirConfig, statement::Statement};
use crate::{
    domain::Domain,
    fiat_shamir::{errors::ProofResult, pow::traits::PowStrategy, prover::ProverState},
    ntt::expand_from_coeff,
    poly::{coeffs::CoefficientList, fold::transform_evaluations, multilinear::MultilinearPoint},
    sumcheck::sumcheck_single::SumcheckSingle,
    utils::expand_randomness,
    whir::{
        parameters::RoundConfig,
        statement::Weights,
        utils::{get_challenge_stir_queries, sample_ood_points},
    },
};

pub type Proof<const DIGEST_ELEMS: usize> = Vec<Vec<[u8; DIGEST_ELEMS]>>;
pub type Leafs<F> = Vec<Vec<F>>;

#[derive(Debug)]
pub(crate) struct RoundState<EF, F, const DIGEST_ELEMS: usize>
where
    F: Field + TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
{
    pub(crate) round: usize,
    pub(crate) domain: Domain<EF, F>,
    pub(crate) sumcheck_prover: Option<SumcheckSingle<F, EF>>,
    pub(crate) folding_randomness: MultilinearPoint<EF>,
    pub(crate) coefficients: CoefficientList<EF>,
    pub(crate) prev_merkle_prover_data:
        MerkleTree<F, u8, FlatMatrixView<F, EF, DenseMatrix<EF>>, DIGEST_ELEMS>,
    /// - The first element is the opened leaf values
    /// - The second element is the Merkle proof (siblings)
    pub(crate) merkle_proofs: Vec<(Leafs<EF>, Proof<DIGEST_ELEMS>)>,
    pub(crate) randomness_vec: Vec<EF>,
    pub(crate) statement: Statement<EF>,
}

#[derive(Debug)]
pub struct Prover<EF, F, H, C, PowStrategy>(pub WhirConfig<EF, F, H, C, PowStrategy>)
where
    F: Field + TwoAdicField + PrimeField64,
    EF: ExtensionField<F> + TwoAdicField;

impl<EF, F, H, C, PS> Prover<EF, F, H, C, PS>
where
    F: Field + TwoAdicField + PrimeField64,
    EF: ExtensionField<F> + TwoAdicField,
    PS: PowStrategy,
{
    fn validate_parameters(&self) -> bool {
        self.0.mv_parameters.num_variables
            == self.0.folding_factor.total_number(self.0.n_rounds()) + self.0.final_sumcheck_rounds
    }

    fn validate_statement(&self, statement: &Statement<EF>) -> bool {
        statement.num_variables() == self.0.mv_parameters.num_variables
            && (self.0.initial_statement || statement.constraints.is_empty())
    }

    fn validate_witness<const DIGEST_ELEMS: usize>(
        &self,
        witness: &Witness<EF, F, DIGEST_ELEMS>,
    ) -> bool {
        assert_eq!(witness.ood_points.len(), witness.ood_answers.len());
        if !self.0.initial_statement {
            assert!(witness.ood_points.is_empty());
        }
        witness.polynomial.num_variables() == self.0.mv_parameters.num_variables
    }

    pub fn prove<const DIGEST_ELEMS: usize>(
        &self,
        prover_state: &mut ProverState<EF, F>,
        mut statement: Statement<EF>,
        witness: Witness<EF, F, DIGEST_ELEMS>,
    ) -> ProofResult<WhirProof<EF, DIGEST_ELEMS>>
    where
        H: CryptographicHasher<F, [u8; DIGEST_ELEMS]> + Sync,
        C: PseudoCompressionFunction<[u8; DIGEST_ELEMS], 2> + Sync,
        [u8; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        // Validate parameters
        assert!(
            self.validate_parameters()
                && self.validate_statement(&statement)
                && self.validate_witness(&witness)
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
            // If there is initial statement, then we run the sum-check for
            // this initial statement.
            let [combination_randomness_gen] = prover_state.challenge_scalars()?;

            // Create the sumcheck prover
            let mut sumcheck = SumcheckSingle::new(
                witness.polynomial.clone(),
                &statement,
                combination_randomness_gen,
            );

            // Compute sumcheck polynomials and return the folding randomness values
            let folding_randomness = sumcheck.compute_sumcheck_polynomials::<PS>(
                prover_state,
                self.0.folding_factor.at_round(0),
                self.0.starting_folding_pow_bits,
            )?;

            sumcheck_prover = Some(sumcheck);
            folding_randomness
        } else {
            // If there is no initial statement, there is no need to run the
            // initial rounds of the sum-check, and the verifier directly sends
            // the initial folding randomnesses.
            let mut folding_randomness = vec![EF::ZERO; self.0.folding_factor.at_round(0)];
            prover_state.fill_challenge_scalars(&mut folding_randomness)?;

            if self.0.starting_folding_pow_bits > 0. {
                prover_state.challenge_pow::<PS>(self.0.starting_folding_pow_bits)?;
            }
            MultilinearPoint(folding_randomness)
        };
        let mut randomness_vec = Vec::with_capacity(self.0.mv_parameters.num_variables);
        randomness_vec.extend(folding_randomness.0.iter().rev().copied());
        randomness_vec.resize(self.0.mv_parameters.num_variables, EF::ZERO);

        let mut round_state = RoundState {
            domain: self.0.starting_domain.clone(),
            round: 0,
            sumcheck_prover,
            folding_randomness,
            coefficients: witness.polynomial,
            prev_merkle_prover_data: witness.prover_data,
            merkle_proofs: vec![],
            randomness_vec,
            statement,
        };

        // Run WHIR rounds
        for _round in 0..=self.0.n_rounds() {
            self.round(prover_state, &mut round_state)?;
        }

        // Extract WhirProof
        let mut randomness_vec_rev = round_state.randomness_vec;
        randomness_vec_rev.reverse();
        let statement_values_at_random_point = round_state
            .statement
            .constraints
            .iter()
            .filter_map(|(weights, _)| {
                if let Weights::Linear { weight } = weights {
                    Some(weight.eval_extension(&MultilinearPoint(randomness_vec_rev.clone())))
                } else {
                    None
                }
            })
            .collect();

        Ok(WhirProof {
            merkle_paths: round_state.merkle_proofs,
            statement_values_at_random_point,
        })
    }

    #[allow(clippy::too_many_lines)]
    fn round<const DIGEST_ELEMS: usize>(
        &self,
        prover_state: &mut ProverState<EF, F>,
        round_state: &mut RoundState<EF, F, DIGEST_ELEMS>,
    ) -> ProofResult<()>
    where
        H: CryptographicHasher<F, [u8; DIGEST_ELEMS]> + Sync,
        C: PseudoCompressionFunction<[u8; DIGEST_ELEMS], 2> + Sync,
        [u8; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        // Fold the coefficients
        let folded_coefficients = round_state
            .coefficients
            .fold(&round_state.folding_randomness);

        let num_variables = self.0.mv_parameters.num_variables
            - self.0.folding_factor.total_number(round_state.round);
        // num_variables should match the folded_coefficients here.
        assert_eq!(num_variables, folded_coefficients.num_variables());

        // Base case: final round reached
        if round_state.round == self.0.n_rounds() {
            return self.final_round(prover_state, round_state, &folded_coefficients);
        }

        let round_params = &self.0.round_parameters[round_state.round];

        // Compute the folding factors for later use
        let folding_factor_next = self.0.folding_factor.at_round(round_state.round + 1);

        // Compute polynomial evaluations and build Merkle tree
        let new_domain = round_state.domain.scale(2);
        let expansion = new_domain.size() / folded_coefficients.num_coeffs();
        let mut evals = expand_from_coeff(folded_coefficients.coeffs(), expansion);
        transform_evaluations(
            &mut evals,
            self.0.fold_optimisation,
            new_domain.backing_domain.group_gen(),
            new_domain.backing_domain.group_gen_inv(),
            folding_factor_next,
        );

        // Convert folded evaluations into a RowMajorMatrix to satisfy the `Matrix<F>` trait
        let folded_matrix = RowMajorMatrix::new(evals, 1 << folding_factor_next);

        let merkle_tree = ExtensionMmcs::new(MerkleTreeMmcs::new(
            self.0.merkle_hash.clone(),
            self.0.merkle_compress.clone(),
        ));
        let (root, prover_data) = merkle_tree.commit(vec![folded_matrix]);

        // Observe Merkle root in challenger
        prover_state.add_digest(root)?;

        // Handle OOD (Out-Of-Domain) samples
        let (ood_points, ood_answers) = sample_ood_points(
            prover_state,
            round_params.ood_samples,
            num_variables,
            |point| folded_coefficients.evaluate(point),
        )?;

        // STIR Queries
        let (stir_challenges, stir_challenges_indexes) = self.compute_stir_queries(
            prover_state,
            round_state,
            num_variables,
            round_params,
            ood_points,
        )?;

        // Collect Merkle proofs for stir queries
        let mut answers = Vec::new();
        let mut merkle_proof = Vec::new();
        for challenge in &stir_challenges_indexes {
            let (leaf, proof) =
                merkle_tree.open_batch(*challenge, &round_state.prev_merkle_prover_data);
            answers.push(leaf[0].clone());
            merkle_proof.push(proof);
        }

        // Evaluate answers in the folding randomness.
        let mut stir_evaluations = ood_answers;
        self.0.fold_optimisation.stir_evaluations_prover(
            round_state,
            &stir_challenges_indexes,
            &answers,
            self.0.folding_factor,
            &mut stir_evaluations,
        );
        round_state.merkle_proofs.push((answers, merkle_proof));

        // PoW
        if round_params.pow_bits > 0. {
            prover_state.challenge_pow::<PS>(round_params.pow_bits)?;
        }

        // Randomness for combination
        let [combination_randomness_gen] = prover_state.challenge_scalars()?;
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
                let mut statement = Statement::new(folded_coefficients.num_variables());

                for (point, eval) in stir_challenges.into_iter().zip(stir_evaluations) {
                    let weights = Weights::evaluation(point);
                    statement.add_constraint(weights, eval);
                }
                SumcheckSingle::new(
                    folded_coefficients.clone(),
                    &statement,
                    combination_randomness[1],
                )
            });

        let folding_randomness = sumcheck_prover.compute_sumcheck_polynomials::<PS>(
            prover_state,
            folding_factor_next,
            round_params.folding_pow_bits,
        )?;

        let start_idx = self.0.folding_factor.total_number(round_state.round);
        let dst_randomness =
            &mut round_state.randomness_vec[start_idx..][..folding_randomness.0.len()];

        for (dst, src) in dst_randomness
            .iter_mut()
            .zip(folding_randomness.0.iter().rev())
        {
            *dst = *src;
        }

        // Update round state
        round_state.round += 1;
        round_state.domain = new_domain;
        round_state.sumcheck_prover = Some(sumcheck_prover);
        round_state.folding_randomness = folding_randomness;
        round_state.coefficients = folded_coefficients;
        round_state.prev_merkle_prover_data = prover_data;

        Ok(())
    }

    fn final_round<const DIGEST_ELEMS: usize>(
        &self,
        prover_state: &mut ProverState<EF, F>,
        round_state: &mut RoundState<EF, F, DIGEST_ELEMS>,
        folded_coefficients: &CoefficientList<EF>,
    ) -> ProofResult<()>
    where
        H: CryptographicHasher<F, [u8; DIGEST_ELEMS]> + Sync,
        C: PseudoCompressionFunction<[u8; DIGEST_ELEMS], 2> + Sync,
        [u8; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        // Directly send coefficients of the polynomial to the verifier.
        prover_state.add_scalars(folded_coefficients.coeffs())?;
        // Final verifier queries and answers. The indices are over the folded domain.
        let final_challenge_indexes = get_challenge_stir_queries(
            // The size of the original domain before folding
            round_state.domain.size(),
            // The folding factor we used to fold the previous polynomial
            self.0.folding_factor.at_round(round_state.round),
            self.0.final_queries,
            prover_state,
        )?;

        // Every query requires opening these many in the previous Merkle tree
        let mmcs = ExtensionMmcs::new(MerkleTreeMmcs::new(
            self.0.merkle_hash.clone(),
            self.0.merkle_compress.clone(),
        ));

        let mut answers = Vec::new();
        let mut merkle_proof = Vec::new();
        for challenge in final_challenge_indexes {
            let (leaf, proof) = mmcs.open_batch(challenge, &round_state.prev_merkle_prover_data);
            answers.push(leaf[0].clone());
            merkle_proof.push(proof);
        }

        round_state.merkle_proofs.push((answers, merkle_proof));

        // PoW
        if self.0.final_pow_bits > 0. {
            prover_state.challenge_pow::<PS>(self.0.final_pow_bits)?;
        }

        // Run final sumcheck if required
        if self.0.final_sumcheck_rounds > 0 {
            let final_folding_randomness = round_state
                .sumcheck_prover
                .clone()
                .unwrap_or_else(|| {
                    SumcheckSingle::new(
                        folded_coefficients.clone(),
                        &round_state.statement,
                        EF::ONE,
                    )
                })
                .compute_sumcheck_polynomials::<PS>(
                    prover_state,
                    self.0.final_sumcheck_rounds,
                    self.0.final_folding_pow_bits,
                )?;

            let start_idx = self.0.folding_factor.total_number(round_state.round);
            let rand_dst = &mut round_state.randomness_vec
                [start_idx..start_idx + final_folding_randomness.0.len()];

            for (dst, src) in rand_dst
                .iter_mut()
                .zip(final_folding_randomness.0.iter().rev())
            {
                *dst = *src;
            }
        }

        Ok(())
    }

    fn compute_stir_queries<const DIGEST_ELEMS: usize>(
        &self,
        prover_state: &mut ProverState<EF, F>,
        round_state: &RoundState<EF, F, DIGEST_ELEMS>,
        num_variables: usize,
        round_params: &RoundConfig,
        ood_points: Vec<EF>,
    ) -> ProofResult<(Vec<MultilinearPoint<EF>>, Vec<usize>)> {
        let stir_challenges_indexes = get_challenge_stir_queries(
            round_state.domain.size(),
            self.0.folding_factor.at_round(round_state.round),
            round_params.num_queries,
            prover_state,
        )?;

        // Compute the generator of the folded domain, in the extension field
        let domain_scaled_gen = round_state
            .domain
            .backing_domain
            .element(1 << self.0.folding_factor.at_round(round_state.round));
        let stir_challenges = ood_points
            .into_iter()
            .chain(
                stir_challenges_indexes
                    .iter()
                    .map(|i| domain_scaled_gen.exp_u64(*i as u64)),
            )
            .map(|univariate| MultilinearPoint::expand_from_univariate(univariate, num_variables))
            .collect();

        Ok((stir_challenges, stir_challenges_indexes))
    }
}
