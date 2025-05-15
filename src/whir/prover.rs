use p3_commit::{ExtensionMmcs, Mmcs};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, PrimeField64, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use serde::{Deserialize, Serialize};

use super::{
    WhirProof,
    committer::{CommitmentMerkleTree, RoundMerkleTree, Witness},
    parameters::WhirConfig,
    statement::Statement,
};
use crate::{
    domain::Domain,
    fiat_shamir::{errors::ProofResult, pow::traits::PowStrategy, prover::ProverState},
    ntt::expand_from_coeff,
    parameters::FoldType,
    poly::{
        coeffs::{CoefficientList, CoefficientStorage},
        multilinear::MultilinearPoint,
    },
    sumcheck::sumcheck_single::SumcheckSingle,
    utils::expand_randomness,
    whir::{
        parameters::RoundConfig,
        statement::Weights,
        utils::{K_SKIP_SUMCHECK, get_challenge_stir_queries, sample_ood_points},
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
    pub(crate) coefficients: CoefficientStorage<F, EF>,

    /// Prover data for the commitment is over the base field
    pub(crate) commitment_merkle_prover_data: CommitmentMerkleTree<F, DIGEST_ELEMS>,
    /// Prover data for the remaining rounds is over the extension field
    /// None in the first round, or if the number of rounds is zero
    pub(crate) merkle_prover_data: Option<RoundMerkleTree<F, EF, DIGEST_ELEMS>>,
    /// Merkle proofs
    /// - The first element is the opened leaf values
    /// - The second element is the Merkle proof (siblings)
    /// - commitment_merkle_proof is None going into a round
    pub(crate) commitment_merkle_proof: Option<(Leafs<F>, Proof<DIGEST_ELEMS>)>,
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

    pub fn prove<D, const DIGEST_ELEMS: usize>(
        &self,
        dft: &D,
        prover_state: &mut ProverState<EF, F>,
        mut statement: Statement<EF>,
        witness: Witness<EF, F, DIGEST_ELEMS>,
    ) -> ProofResult<WhirProof<F, EF, DIGEST_ELEMS>>
    where
        H: CryptographicHasher<F, [u8; DIGEST_ELEMS]> + Sync,
        C: PseudoCompressionFunction<[u8; DIGEST_ELEMS], 2> + Sync,
        [u8; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
        D: TwoAdicSubgroupDft<F>,
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
            let mut sumcheck = SumcheckSingle::from_base_coeffs(
                witness.polynomial.clone(),
                &statement,
                combination_randomness_gen,
            );

            println!("avant2");
            // Compute sumcheck polynomials and return the folding randomness values
            let folding_randomness = sumcheck.compute_sumcheck_polynomials::<PS, _>(
                prover_state,
                self.0.folding_factor.at_round(0),
                self.0.starting_folding_pow_bits,
                Some(K_SKIP_SUMCHECK),
                dft,
            )?;
            println!("apres2");

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
            coefficients: CoefficientStorage::Base(witness.polynomial),
            merkle_prover_data: None,
            commitment_merkle_prover_data: witness.prover_data,
            commitment_merkle_proof: None,
            merkle_proofs: vec![],
            randomness_vec,
            statement,
        };

        // Run WHIR rounds
        for _round in 0..=self.0.n_rounds() {
            self.round(dft, prover_state, &mut round_state)?;
        }

        // Extract WhirProof
        round_state.randomness_vec.reverse();
        let statement_values_at_random_point = round_state
            .statement
            .constraints
            .iter()
            .filter_map(|(weights, _)| {
                if let Weights::Linear { weight } = weights {
                    Some(
                        weight
                            .eval_extension(&MultilinearPoint(round_state.randomness_vec.clone())),
                    )
                } else {
                    None
                }
            })
            .collect();

        Ok(WhirProof {
            commitment_merkle_paths: round_state.commitment_merkle_proof.unwrap(),
            merkle_paths: round_state.merkle_proofs,
            statement_values_at_random_point,
        })
    }

    #[allow(clippy::too_many_lines)]
    fn round<D, const DIGEST_ELEMS: usize>(
        &self,
        dft: &D,
        prover_state: &mut ProverState<EF, F>,
        round_state: &mut RoundState<EF, F, DIGEST_ELEMS>,
    ) -> ProofResult<()>
    where
        H: CryptographicHasher<F, [u8; DIGEST_ELEMS]> + Sync,
        C: PseudoCompressionFunction<[u8; DIGEST_ELEMS], 2> + Sync,
        [u8; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
        D: TwoAdicSubgroupDft<F>,
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
            return self.final_round(prover_state, round_state, &folded_coefficients, dft);
        }

        let round_params = &self.0.round_parameters[round_state.round];

        // Compute the folding factors for later use
        let folding_factor_next = self.0.folding_factor.at_round(round_state.round + 1);

        // Compute polynomial evaluations and build Merkle tree
        let new_domain = round_state.domain.scale(2);
        let expansion = new_domain.size() / folded_coefficients.num_coeffs();
        let folded_matrix = match self.0.fold_optimisation {
            FoldType::Naive => {
                let evals = expand_from_coeff(dft, folded_coefficients.coeffs(), expansion);

                // Compute the number of sub-cosets = 2^folding_factor
                let folding_factor_exp = 1 << self.0.folding_factor.at_round(0);

                // Number of rows (one per subdomain)
                let size_of_new_domain = evals.len() / folding_factor_exp;

                RowMajorMatrix::new(evals, size_of_new_domain).transpose()
            }
            FoldType::ProverHelps => {
                let mut coeffs = folded_coefficients.coeffs().to_vec();
                coeffs.resize(coeffs.len() * expansion, EF::ZERO);
                // Do DFT on only interleaved polys to be folded.
                dft.dft_algebra_batch(RowMajorMatrix::new(coeffs, 1 << folding_factor_next))
            }
        };

        let mmcs = MerkleTreeMmcs::new(self.0.merkle_hash.clone(), self.0.merkle_compress.clone());
        let extension_mmcs = ExtensionMmcs::new(mmcs.clone());
        let (root, prover_data) = extension_mmcs.commit_matrix(folded_matrix);

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
        let stir_evaluations = match &round_state.merkle_prover_data {
            None => {
                let mut answers = Vec::with_capacity(stir_challenges_indexes.len());
                let mut merkle_proof = Vec::with_capacity(stir_challenges_indexes.len());
                for challenge in &stir_challenges_indexes {
                    let (commitment_leaf, commitment_root) =
                        mmcs.open_batch(*challenge, &round_state.commitment_merkle_prover_data);
                    answers.push(commitment_leaf[0].clone());
                    merkle_proof.push(commitment_root);
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

                round_state.commitment_merkle_proof = Some((answers, merkle_proof));
                stir_evaluations
            }
            Some(data) => {
                let mut answers = Vec::with_capacity(stir_challenges_indexes.len());
                let mut merkle_proof = Vec::with_capacity(stir_challenges_indexes.len());
                for challenge in &stir_challenges_indexes {
                    let (leaf, proof) = extension_mmcs.open_batch(*challenge, data);
                    answers.push(leaf[0].clone());
                    merkle_proof.push(proof);
                }
                // Evaluate answers in the folding randomness.
                let mut stir_evaluations = ood_answers;
                self.0
                    .fold_optimisation
                    .stir_evaluations_prover::<_, EF, _, DIGEST_ELEMS>(
                        round_state,
                        &stir_challenges_indexes,
                        &answers,
                        self.0.folding_factor,
                        &mut stir_evaluations,
                    );
                round_state.merkle_proofs.push((answers, merkle_proof));
                stir_evaluations
            }
        };

        // PoW
        if round_params.pow_bits > 0. {
            prover_state.challenge_pow::<PS>(round_params.pow_bits)?;
        }

        // Randomness for combination
        let [combination_randomness_gen] = prover_state.challenge_scalars()?;
        let combination_randomness =
            expand_randomness(combination_randomness_gen, stir_challenges.len());

        let mut sumcheck_prover =
            if let Some(mut sumcheck_prover) = round_state.sumcheck_prover.take() {
                sumcheck_prover.add_new_equality(
                    &stir_challenges,
                    &stir_evaluations,
                    &combination_randomness,
                );
                sumcheck_prover
            } else {
                let mut statement = Statement::new(folded_coefficients.num_variables());

                for (point, eval) in stir_challenges.into_iter().zip(stir_evaluations) {
                    let weights = Weights::evaluation(point);
                    statement.add_constraint(weights, eval);
                }

                SumcheckSingle::from_extension_coeffs(
                    folded_coefficients.clone(),
                    &statement,
                    combination_randomness[1],
                )
            };

        let folding_randomness = sumcheck_prover.compute_sumcheck_polynomials::<PS, _>(
            prover_state,
            folding_factor_next,
            round_params.folding_pow_bits,
            None,
            dft,
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
        round_state.coefficients = CoefficientStorage::Extension(folded_coefficients);
        round_state.merkle_prover_data = Some(prover_data);

        Ok(())
    }

    fn final_round<D, const DIGEST_ELEMS: usize>(
        &self,
        prover_state: &mut ProverState<EF, F>,
        round_state: &mut RoundState<EF, F, DIGEST_ELEMS>,
        folded_coefficients: &CoefficientList<EF>,
        dft: &D,
    ) -> ProofResult<()>
    where
        H: CryptographicHasher<F, [u8; DIGEST_ELEMS]> + Sync,
        C: PseudoCompressionFunction<[u8; DIGEST_ELEMS], 2> + Sync,
        [u8; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
        D: TwoAdicSubgroupDft<F>,
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
        let mmcs = MerkleTreeMmcs::new(self.0.merkle_hash.clone(), self.0.merkle_compress.clone());
        let extension_mmcs = ExtensionMmcs::new(mmcs.clone());

        match &round_state.merkle_prover_data {
            None => {
                let mut answers = Vec::with_capacity(final_challenge_indexes.len());
                let mut merkle_proof = Vec::with_capacity(final_challenge_indexes.len());

                for challenge in final_challenge_indexes {
                    let (commitment_leaf, commitment_root) =
                        mmcs.open_batch(challenge, &round_state.commitment_merkle_prover_data);
                    answers.push(commitment_leaf[0].clone());
                    merkle_proof.push(commitment_root);
                }

                round_state.commitment_merkle_proof = Some((answers, merkle_proof));
            }

            Some(data) => {
                let mut answers = Vec::with_capacity(final_challenge_indexes.len());
                let mut merkle_proof = Vec::with_capacity(final_challenge_indexes.len());
                for challenge in final_challenge_indexes {
                    let (leaf, proof) = extension_mmcs.open_batch(challenge, data);
                    answers.push(leaf[0].clone());
                    merkle_proof.push(proof);
                }
                round_state.merkle_proofs.push((answers, merkle_proof));
            }
        }

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
                    SumcheckSingle::from_extension_coeffs(
                        folded_coefficients.clone(),
                        &round_state.statement,
                        EF::ONE,
                    )
                })
                .compute_sumcheck_polynomials::<PS, _>(
                    prover_state,
                    self.0.final_sumcheck_rounds,
                    self.0.final_folding_pow_bits,
                    None,
                    dft,
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
