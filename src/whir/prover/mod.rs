use std::ops::Deref;

use p3_challenger::{CanObserve, CanSample};
use p3_commit::{ExtensionMmcs, Mmcs};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, Packable, PrimeField64, TwoAdicField};
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use round::RoundState;
use serde::{Deserialize, Serialize};
use tracing::{info_span, instrument};

use super::{committer::Witness, parameters::WhirConfig, statement::Statement};
use crate::{
    fiat_shamir::{
        duplex_sponge::interface::Unit, errors::ProofResult, pow::traits::PowStrategy,
        prover::ProverState,
    },
    poly::{
        coeffs::{CoefficientList, CoefficientStorage},
        evals::{EvaluationStorage, EvaluationsList},
        multilinear::MultilinearPoint,
    },
    sumcheck::sumcheck_single::SumcheckSingle,
    whir::{
        parameters::RoundConfig,
        statement::weights::Weights,
        utils::{get_challenge_stir_queries, sample_ood_points},
    },
};

pub mod round;

pub type Proof<W, const DIGEST_ELEMS: usize> = Vec<Vec<[W; DIGEST_ELEMS]>>;
pub type Leafs<F> = Vec<Vec<F>>;

#[derive(Debug)]
pub struct Prover<'a, EF, F, H, C, PowStrategy, Challenger, W, const PERM_WIDTH: usize>(
    /// Reference to the protocol configuration shared across prover components.
    pub &'a WhirConfig<EF, F, H, C, PowStrategy, Challenger, W, PERM_WIDTH>,
)
where
    F: Field,
    EF: ExtensionField<F>;

impl<EF, F, H, C, PS, Challenger, W, const PERM_WIDTH: usize> Deref
    for Prover<'_, EF, F, H, C, PS, Challenger, W, PERM_WIDTH>
where
    F: Field,
    EF: ExtensionField<F>,
{
    type Target = WhirConfig<EF, F, H, C, PS, Challenger, W, PERM_WIDTH>;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl<EF, F, H, C, PS, Challenger, W, const PERM_WIDTH: usize>
    Prover<'_, EF, F, H, C, PS, Challenger, W, PERM_WIDTH>
where
    F: Field + TwoAdicField + PrimeField64,
    EF: ExtensionField<F> + TwoAdicField,
    PS: PowStrategy,
    W: Unit + Default + Copy,
    Challenger: CanObserve<W> + CanSample<W>,
{
    /// Validates that the total number of variables expected by the prover configuration
    /// matches the number implied by the folding schedule and the final rounds.
    ///
    /// This ensures that the recursive folding in the sumcheck protocol terminates
    /// precisely at the expected number of final variables.
    ///
    /// # Returns
    /// `true` if the parameter configuration is consistent, `false` otherwise.
    fn validate_parameters(&self) -> bool {
        self.mv_parameters.num_variables
            == self.folding_factor.total_number(self.n_rounds()) + self.final_sumcheck_rounds
    }

    /// Validates that the public statement is compatible with the configured number of variables.
    ///
    /// Ensures the following:
    /// - The number of variables in the statement matches the prover's expectations
    /// - If no initial statement is used, the statement must be empty
    ///
    /// # Parameters
    /// - `statement`: The public constraints that the prover will use
    ///
    /// # Returns
    /// `true` if the statement structure is valid for this protocol instance.
    fn validate_statement(&self, statement: &Statement<EF>) -> bool {
        statement.num_variables() == self.mv_parameters.num_variables
            && (self.initial_statement || statement.constraints.is_empty())
    }

    /// Validates that the witness satisfies the structural requirements of the WHIR prover.
    ///
    /// Checks the following conditions:
    /// - The number of OOD (out-of-domain) points equals the number of OOD answers
    /// - If no initial statement is used, the OOD data must be empty
    /// - The multilinear witness polynomial must match the expected number of variables
    ///
    /// # Parameters
    /// - `witness`: The private witness to be verified for structural consistency
    ///
    /// # Returns
    /// `true` if the witness structure matches expectations.
    ///
    /// # Panics
    /// - Panics if OOD lengths are inconsistent
    /// - Panics if OOD data is non-empty despite `initial_statement = false`
    fn validate_witness<const DIGEST_ELEMS: usize>(
        &self,
        witness: &Witness<EF, F, W, DenseMatrix<F>, DIGEST_ELEMS>,
    ) -> bool {
        assert_eq!(witness.ood_points.len(), witness.ood_answers.len());
        if !self.initial_statement {
            assert!(witness.ood_points.is_empty());
        }
        witness.pol_coeffs.num_variables() == self.mv_parameters.num_variables
    }

    /// Executes the full WHIR prover protocol to produce the proof.
    ///
    /// This function takes the public statement and private witness, performs the
    /// multi-round sumcheck-based polynomial folding protocol using DFTs, and returns
    /// a proof that the witness satisfies the statement.
    ///
    /// The proof includes:
    /// - Merkle authentication paths for each round's polynomial commitments
    /// - Final evaluations of the public linear statement constraints at a random point
    ///
    /// # Parameters
    /// - `dft`: A DFT backend used for evaluations
    /// - `prover_state`: Mutable prover state used across rounds (transcript, randomness, etc.)
    /// - `statement`: The public input, consisting of linear or nonlinear constraints
    /// - `witness`: The private witness satisfying the constraints, including committed values
    ///
    /// # Returns
    /// - The final random evaluation point used to evaluate deferred constraints
    /// - The list of evaluations of all deferred constraints at that point
    ///
    /// # Errors
    /// Returns an error if the witness or statement are invalid, or if a round fails.
    #[instrument(skip_all)]
    pub fn prove<D, const DIGEST_ELEMS: usize>(
        &self,
        dft: &D,
        prover_state: &mut ProverState<EF, F, Challenger, W, PERM_WIDTH>,
        statement: Statement<EF>,
        witness: Witness<EF, F, W, DenseMatrix<F>, DIGEST_ELEMS>,
    ) -> ProofResult<(MultilinearPoint<EF>, Vec<EF>)>
    where
        H: CryptographicHasher<F, [W; DIGEST_ELEMS]> + Sync,
        C: PseudoCompressionFunction<[W; DIGEST_ELEMS], 2> + Sync,
        [W; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
        D: TwoAdicSubgroupDft<F>,
        W: Eq + Packable,
    {
        // Validate parameters
        assert!(
            self.validate_parameters()
                && self.validate_statement(&statement)
                && self.validate_witness(&witness),
            "Invalid prover parameters, statement, or witness"
        );

        // Initialize the round state with inputs and initial polynomial data
        let mut round_state =
            RoundState::initialize_first_round_state(self, prover_state, statement, witness, dft)?;

        // Run the WHIR protocol round-by-round
        for round in 0..=self.n_rounds() {
            self.round(round, dft, prover_state, &mut round_state)?;
        }

        // Reverse the vector of verifier challenges (used as evaluation point)
        //
        // These challenges were pushed in round order; we reverse them to use as a single
        // evaluation point for final statement consistency checks.
        round_state.randomness_vec.reverse();
        let constraint_eval = MultilinearPoint(round_state.randomness_vec);

        // Hints for deferred constraints
        let deferred = round_state
            .statement
            .constraints
            .iter()
            .filter(|constraint| constraint.defer_evaluation)
            .map(|constraint| constraint.weights.compute(&constraint_eval))
            .collect();
        prover_state.hint::<Vec<EF>>(&deferred)?;

        Ok((constraint_eval, deferred))
    }

    #[instrument(skip_all, fields(round_number = round_index, log_size = round_state.evaluations.num_variables()))]
    #[allow(clippy::too_many_lines)]
    fn round<D, const DIGEST_ELEMS: usize>(
        &self,
        round_index: usize,
        dft: &D,
        prover_state: &mut ProverState<EF, F, Challenger, W, PERM_WIDTH>,
        round_state: &mut RoundState<EF, F, W, DenseMatrix<F>, DIGEST_ELEMS>,
    ) -> ProofResult<()>
    where
        H: CryptographicHasher<F, [W; DIGEST_ELEMS]> + Sync,
        C: PseudoCompressionFunction<[W; DIGEST_ELEMS], 2> + Sync,
        [W; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
        D: TwoAdicSubgroupDft<F>,
        W: Eq + Packable,
    {
        // - If a sumcheck already exists, use its evaluations
        // - Otherwise, fold the evaluations from the previous round
        let folded_evaluations = if let Some(sumcheck) = &round_state.sumcheck_prover {
            match &sumcheck.evaluation_of_p {
                EvaluationStorage::Base(_) => {
                    panic!("After a first round, the evaluations must be in the extension field")
                }
                EvaluationStorage::Extension(f) => f.clone(),
            }
        } else {
            round_state
                .evaluations
                .fold(&round_state.folding_randomness)
        };

        let folded_coefficients = round_state.coeffs.fold(&round_state.folding_randomness);

        let num_variables =
            self.mv_parameters.num_variables - self.folding_factor.total_number(round_index);
        // The number of variables at the given round should match the folded number of variables.
        assert_eq!(num_variables, folded_evaluations.num_variables());

        // Base case: final round reached
        if round_index == self.n_rounds() {
            return self.final_round(
                round_index,
                prover_state,
                round_state,
                &folded_coefficients,
                &folded_evaluations,
                dft,
            );
        }

        let round_params = &self.round_parameters[round_index];

        // Compute the folding factors for later use
        let folding_factor_next = self.folding_factor.at_round(round_index + 1);

        // Compute polynomial evaluations and build Merkle tree
        let domain_reduction = 1 << self.rs_reduction_factor(round_index);
        let new_domain = round_state.domain.scale(domain_reduction);
        let folded_matrix = info_span!("fold matrix").in_scope(|| {
            let coeffs = info_span!("copy_across_coeffs").in_scope(|| {
                let mut coeffs = EF::zero_vec(new_domain.size());
                coeffs[..folded_evaluations.num_evals()]
                    .copy_from_slice(folded_coefficients.coeffs());
                coeffs
            });
            // Do DFT on only interleaved polys to be folded.
            info_span!(
                "dft",
                height = coeffs.len() >> folding_factor_next,
                width = 1 << folding_factor_next
            )
            .in_scope(|| {
                dft.dft_algebra_batch(RowMajorMatrix::new(coeffs, 1 << folding_factor_next))
            })
        });

        let mmcs = MerkleTreeMmcs::new(self.merkle_hash.clone(), self.merkle_compress.clone());
        let extension_mmcs = ExtensionMmcs::new(mmcs.clone());
        let (root, prover_data) =
            info_span!("commit matrix").in_scope(|| extension_mmcs.commit_matrix(folded_matrix));

        // Observe Merkle root in challenger
        prover_state.add_digest(root)?;

        // Handle OOD (Out-Of-Domain) samples
        let (ood_points, ood_answers) = sample_ood_points(
            prover_state,
            round_params.ood_samples,
            num_variables,
            |point| folded_evaluations.evaluate(point),
        )?;

        // STIR Queries
        let (stir_challenges, stir_challenges_indexes) = self.compute_stir_queries(
            round_index,
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
                    let commitment =
                        mmcs.open_batch(*challenge, &round_state.commitment_merkle_prover_data);
                    answers.push(commitment.opened_values[0].clone());
                    merkle_proof.push(commitment.opening_proof);
                }

                prover_state.hint(&answers)?;
                prover_state.hint(&merkle_proof)?;

                // Evaluate answers in the folding randomness.
                let mut stir_evaluations = ood_answers;
                // Exactly one growth
                stir_evaluations.reserve_exact(answers.len());

                for answer in &answers {
                    stir_evaluations.push(
                        CoefficientList::new(answer.clone())
                            .evaluate(&round_state.folding_randomness),
                    );
                }

                round_state.commitment_merkle_proof = Some((answers, merkle_proof));
                stir_evaluations
            }
            Some(data) => {
                let mut answers = Vec::with_capacity(stir_challenges_indexes.len());
                let mut merkle_proof = Vec::with_capacity(stir_challenges_indexes.len());
                for challenge in &stir_challenges_indexes {
                    let commitment = extension_mmcs.open_batch(*challenge, data);
                    answers.push(commitment.opened_values[0].clone());
                    merkle_proof.push(commitment.opening_proof);
                }

                prover_state.hint(&answers)?;
                prover_state.hint(&merkle_proof)?;

                // Evaluate answers in the folding randomness.
                let mut stir_evaluations = ood_answers;
                // Exactly one growth
                stir_evaluations.reserve_exact(answers.len());

                for answer in &answers {
                    stir_evaluations.push(
                        CoefficientList::new(answer.clone())
                            .evaluate(&round_state.folding_randomness),
                    );
                }

                round_state.merkle_proofs.push((answers, merkle_proof));
                stir_evaluations
            }
        };

        // PoW
        if round_params.pow_bits > 0. {
            info_span!("pow", bits = round_params.pow_bits)
                .in_scope(|| prover_state.challenge_pow::<PS>(round_params.pow_bits))?;
        }

        // Randomness for combination
        let [combination_randomness_gen] = prover_state.challenge_scalars()?;
        let combination_randomness: Vec<_> = combination_randomness_gen
            .powers()
            .take(stir_challenges.len())
            .collect();

        let mut sumcheck_prover =
            if let Some(mut sumcheck_prover) = round_state.sumcheck_prover.take() {
                sumcheck_prover.add_new_equality(
                    &stir_challenges,
                    &stir_evaluations,
                    &combination_randomness,
                );
                sumcheck_prover
            } else {
                let mut statement = Statement::new(folded_evaluations.num_variables());

                for (point, eval) in stir_challenges.into_iter().zip(stir_evaluations) {
                    let weights = Weights::evaluation(point);
                    statement.add_constraint(weights, eval);
                }

                SumcheckSingle::from_extension_evals(
                    folded_evaluations.clone(),
                    &statement,
                    combination_randomness[1],
                )
            };

        let folding_randomness = sumcheck_prover
            .compute_sumcheck_polynomials::<PS, _, _, _, PERM_WIDTH>(
                prover_state,
                folding_factor_next,
                round_params.folding_pow_bits,
                None,
                dft,
            )?;

        let start_idx = self.folding_factor.total_number(round_index);
        let dst_randomness =
            &mut round_state.randomness_vec[start_idx..][..folding_randomness.0.len()];

        for (dst, src) in dst_randomness
            .iter_mut()
            .zip(folding_randomness.0.iter().rev())
        {
            *dst = *src;
        }

        // Update round state
        round_state.domain = new_domain;
        round_state.sumcheck_prover = Some(sumcheck_prover);
        round_state.coeffs = CoefficientStorage::Extension(folded_coefficients);
        round_state.folding_randomness = folding_randomness;
        round_state.evaluations = EvaluationStorage::Extension(folded_evaluations);
        round_state.merkle_prover_data = Some(prover_data);

        Ok(())
    }

    #[instrument(skip_all)]
    fn final_round<D, const DIGEST_ELEMS: usize>(
        &self,
        round_index: usize,
        prover_state: &mut ProverState<EF, F, Challenger, W, PERM_WIDTH>,
        round_state: &mut RoundState<EF, F, W, DenseMatrix<F>, DIGEST_ELEMS>,
        folded_coefficients: &CoefficientList<EF>,
        folded_evaluations: &EvaluationsList<EF>,
        dft: &D,
    ) -> ProofResult<()>
    where
        H: CryptographicHasher<F, [W; DIGEST_ELEMS]> + Sync,
        C: PseudoCompressionFunction<[W; DIGEST_ELEMS], 2> + Sync,
        [W; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
        D: TwoAdicSubgroupDft<F>,
        W: Eq + Packable,
    {
        // Directly send coefficients of the polynomial to the verifier.
        prover_state.add_scalars(folded_coefficients.coeffs())?;
        // Final verifier queries and answers. The indices are over the folded domain.
        let final_challenge_indexes = get_challenge_stir_queries(
            // The size of the original domain before folding
            round_state.domain.size(),
            // The folding factor we used to fold the previous polynomial
            self.folding_factor.at_round(round_index),
            self.final_queries,
            prover_state,
        )?;

        // Every query requires opening these many in the previous Merkle tree
        let mmcs = MerkleTreeMmcs::new(self.merkle_hash.clone(), self.merkle_compress.clone());
        let extension_mmcs = ExtensionMmcs::new(mmcs.clone());

        match &round_state.merkle_prover_data {
            None => {
                let mut answers = Vec::with_capacity(final_challenge_indexes.len());
                let mut merkle_proof = Vec::with_capacity(final_challenge_indexes.len());

                for challenge in final_challenge_indexes {
                    let commitment =
                        mmcs.open_batch(challenge, &round_state.commitment_merkle_prover_data);
                    answers.push(commitment.opened_values[0].clone());
                    merkle_proof.push(commitment.opening_proof);
                }

                prover_state.hint(&answers)?;
                prover_state.hint(&merkle_proof)?;

                round_state.commitment_merkle_proof = Some((answers, merkle_proof));
            }

            Some(data) => {
                let mut answers = Vec::with_capacity(final_challenge_indexes.len());
                let mut merkle_proof = Vec::with_capacity(final_challenge_indexes.len());
                for challenge in final_challenge_indexes {
                    let commitment = extension_mmcs.open_batch(challenge, data);
                    answers.push(commitment.opened_values[0].clone());
                    merkle_proof.push(commitment.opening_proof);
                }

                prover_state.hint(&answers)?;
                prover_state.hint(&merkle_proof)?;

                round_state.merkle_proofs.push((answers, merkle_proof));
            }
        }

        // PoW
        if self.final_pow_bits > 0. {
            prover_state.challenge_pow::<PS>(self.final_pow_bits)?;
        }

        // Run final sumcheck if required
        if self.final_sumcheck_rounds > 0 {
            let final_folding_randomness = round_state
                .sumcheck_prover
                .clone()
                .unwrap_or_else(|| {
                    SumcheckSingle::from_extension_evals(
                        folded_evaluations.clone(),
                        &round_state.statement,
                        EF::ONE,
                    )
                })
                .compute_sumcheck_polynomials::<PS, _, _, _, PERM_WIDTH>(
                    prover_state,
                    self.final_sumcheck_rounds,
                    self.final_folding_pow_bits,
                    None,
                    dft,
                )?;

            let start_idx = self.folding_factor.total_number(round_index);
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

    #[instrument(skip_all, level = "debug")]
    fn compute_stir_queries<const DIGEST_ELEMS: usize>(
        &self,
        round_index: usize,
        prover_state: &mut ProverState<EF, F, Challenger, W, PERM_WIDTH>,
        round_state: &RoundState<EF, F, W, DenseMatrix<F>, DIGEST_ELEMS>,
        num_variables: usize,
        round_params: &RoundConfig<EF>,
        ood_points: Vec<EF>,
    ) -> ProofResult<(Vec<MultilinearPoint<EF>>, Vec<usize>)> {
        let stir_challenges_indexes = get_challenge_stir_queries(
            round_state.domain.size(),
            self.folding_factor.at_round(round_index),
            round_params.num_queries,
            prover_state,
        )?;

        // Compute the generator of the folded domain, in the extension field
        let domain_scaled_gen = round_state
            .domain
            .backing_domain
            .element(1 << self.folding_factor.at_round(round_index));
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
