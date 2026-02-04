use alloc::vec::Vec;
use core::ops::Deref;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::{ExtensionMmcs, Mmcs};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, PackedValue, TwoAdicField};
use p3_matrix::{
    Matrix,
    dense::{DenseMatrix, RowMajorMatrixView},
};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CryptographicHasher, Hash, PseudoCompressionFunction};
use round_state::RoundState;
use serde::{Deserialize, Serialize};
use tracing::{info_span, instrument};

use super::{committer::ProverData, constraints::statement::EqStatement, parameters::WhirConfig};
use crate::{
    fiat_shamir::errors::FiatShamirError,
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    whir::{
        constraints::{
            Constraint,
            statement::{SelectStatement, initial::InitialStatement},
        },
        parameters::SumcheckStrategy,
        proof::{QueryOpening, SumcheckData, WhirProof},
        utils::get_challenge_stir_queries,
    },
};

pub mod round_state;

pub type Proof<W, const DIGEST_ELEMS: usize> = Vec<Vec<[W; DIGEST_ELEMS]>>;
pub type Leafs<F> = Vec<Vec<F>>;

#[derive(Debug)]
pub struct Prover<'a, EF, F, H, C, Challenger>(
    /// Reference to the protocol configuration shared across prover components.
    pub &'a WhirConfig<EF, F, H, C, Challenger>,
)
where
    F: Field,
    EF: ExtensionField<F>;

impl<EF, F, H, C, Challenger> Deref for Prover<'_, EF, F, H, C, Challenger>
where
    F: Field,
    EF: ExtensionField<F>,
{
    type Target = WhirConfig<EF, F, H, C, Challenger>;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl<EF, F, H, C, Challenger> Prover<'_, EF, F, H, C, Challenger>
where
    F: TwoAdicField + Ord,
    EF: ExtensionField<F> + TwoAdicField,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    /// Validates that the total number of variables expected by the prover configuration
    /// matches the number implied by the folding schedule and the final rounds.
    ///
    /// This ensures that the recursive folding in the sumcheck protocol terminates
    /// precisely at the expected number of final variables.
    ///
    /// # Returns
    /// `true` if the parameter configuration is consistent, `false` otherwise.
    const fn validate_parameters(&self) -> bool {
        self.0.num_variables
            == self.0.folding_factor.total_number(self.0.n_rounds()) + self.0.final_sumcheck_rounds
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
    const fn validate_statement(&self, statement: &EqStatement<EF>) -> bool {
        statement.num_variables() == self.0.num_variables
    }

    const fn initial_statement(
        &self,
        polynomial: EvaluationsList<F>,
        sumcheck_strategy: SumcheckStrategy,
    ) -> InitialStatement<F, EF> {
        self.0.initial_statement(polynomial, sumcheck_strategy)
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
    /// - `proof`: Mutable proof structure to store the generated proof data
    /// - `challenger`: Mutable Fiat-Shamir challenger for transcript management
    /// - `statement`: The public input, consisting of linear or nonlinear constraints
    /// - `witness`: The private witness satisfying the constraints, including committed values
    ///
    ///
    /// # Errors
    /// Returns an error if the witness or statement are invalid, or if a round fails.
    #[instrument(skip_all)]
    pub fn prove<Dft, P, W, PW, const DIGEST_ELEMS: usize>(
        &self,
        dft: &Dft,
        proof: &mut WhirProof<F, EF, W, DIGEST_ELEMS>,
        challenger: &mut Challenger,
        statement: &InitialStatement<F, EF>,
        prover_data: ProverData<F, DenseMatrix<F>, W, DIGEST_ELEMS>,
    ) -> Result<(), FiatShamirError>
    where
        Dft: TwoAdicSubgroupDft<F>,
        P: PackedValue<Value = F> + Eq + Send + Sync,
        W: PackedValue<Value = W> + Eq + Send + Sync,
        PW: PackedValue<Value = W> + Eq + Send + Sync,
        H: CryptographicHasher<F, [W; DIGEST_ELEMS]>
            + CryptographicHasher<P, [PW; DIGEST_ELEMS]>
            + Sync,
        C: PseudoCompressionFunction<[W; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[PW; DIGEST_ELEMS], 2>
            + Sync,
        Challenger: CanObserve<Hash<F, W, DIGEST_ELEMS>>,
        [W; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        // Validate parameters
        assert!(self.validate_parameters(), "Invalid prover parameters");

        // Initialize the round state with inputs and initial polynomial data
        let mut round_state = RoundState::initialize_first_round_state(
            &mut proof.initial_sumcheck,
            challenger,
            statement,
            prover_data,
            self.folding_factor.at_round(0),
            self.starting_folding_pow_bits,
        )?;

        // Run the WHIR protocol round-by-round
        for round in 0..=self.n_rounds() {
            self.round::<P, W, PW, DIGEST_ELEMS, _>(
                dft,
                round,
                proof,
                challenger,
                &mut round_state,
            )?;
        }

        Ok(())
    }

    #[instrument(skip_all, fields(round_number = round_index, log_size = self.num_variables - self.folding_factor.total_number(round_index)))]
    #[allow(clippy::too_many_lines)]
    fn round<P, W, PW, const DIGEST_ELEMS: usize, Dft: TwoAdicSubgroupDft<F>>(
        &self,
        dft: &Dft,
        round_index: usize,
        proof: &mut WhirProof<F, EF, W, DIGEST_ELEMS>,
        challenger: &mut Challenger,
        round_state: &mut RoundState<EF, F, W, DenseMatrix<F>, DIGEST_ELEMS>,
    ) -> Result<(), FiatShamirError>
    where
        P: PackedValue<Value = F> + Eq + Send + Sync,
        W: PackedValue<Value = W> + Eq + Send + Sync,
        PW: PackedValue<Value = W> + Eq + Send + Sync,
        H: CryptographicHasher<F, [W; DIGEST_ELEMS]>
            + CryptographicHasher<P, [PW; DIGEST_ELEMS]>
            + Sync,
        C: PseudoCompressionFunction<[W; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[PW; DIGEST_ELEMS], 2>
            + Sync,
        Challenger: CanObserve<Hash<F, W, DIGEST_ELEMS>>,
        [W; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        let folded_evaluations = &round_state.sumcheck_prover.evals();
        let num_variables = self.num_variables - self.folding_factor.total_number(round_index);
        assert_eq!(num_variables, folded_evaluations.num_variables());

        // Base case: final round reached
        if round_index == self.n_rounds() {
            return self.final_round::<P, W, PW, DIGEST_ELEMS>(
                round_index,
                proof,
                challenger,
                round_state,
            );
        }

        let round_params = &self.round_parameters[round_index];

        // Compute the folding factors for later use
        let folding_factor_next = self.folding_factor.at_round(round_index + 1);
        let inv_rate = self.inv_rate(round_index);

        // Transpose for reverse variable order
        // And then pad with zeros
        let padded = info_span!("transpose & pad").in_scope(|| {
            let num_vars = folded_evaluations.num_variables();
            let mut mat = RowMajorMatrixView::new(
                folded_evaluations.as_slice(),
                1 << (num_vars - folding_factor_next),
            )
            .transpose();

            mat.pad_to_height(inv_rate * (1 << (num_vars - folding_factor_next)), EF::ZERO);
            mat
        });

        // Perform DFT on the padded evaluations matrix
        let folded_matrix = info_span!("dft", height = padded.height(), width = padded.width())
            .in_scope(|| dft.dft_algebra_batch(padded).to_row_major_matrix());

        let mmcs = MerkleTreeMmcs::<P, PW, H, C, DIGEST_ELEMS>::new(
            self.merkle_hash.clone(),
            self.merkle_compress.clone(),
        );
        let extension_mmcs = ExtensionMmcs::new(mmcs.clone());
        let (root, prover_data) =
            info_span!("commit matrix").in_scope(|| extension_mmcs.commit_matrix(folded_matrix));

        // Observe the round merkle tree commitment
        challenger.observe(root);

        // Store commitment in proof
        proof.rounds[round_index].commitment = root.into();

        // Handle OOD (Out-Of-Domain) samples
        let mut ood_statement = EqStatement::initialize(num_variables);
        let mut ood_answers = Vec::with_capacity(round_params.ood_samples);
        (0..round_params.ood_samples).for_each(|_| {
            let point = MultilinearPoint::expand_from_univariate(
                challenger.sample_algebra_element(),
                num_variables,
            );
            let eval = round_state.sumcheck_prover.eval(&point);
            challenger.observe_algebra_element(eval);

            ood_answers.push(eval);
            ood_statement.add_evaluated_constraint(point, eval);
        });

        // Store OOD answers in proof
        proof.rounds[round_index].ood_answers = ood_answers;

        // CRITICAL: Perform proof-of-work grinding to finalize the transcript before querying.
        //
        // This is a crucial security step to prevent a malicious prover from influencing the
        // verifier's challenges.
        //
        // The verifier's query locations (the `stir_challenges`) are generated based on the
        // current transcript state, which includes the prover's polynomial commitment (the Merkle
        // root) for this round. Without grinding, a prover could repeatedly try different
        // commitments until they find one that results in "easy" queries, breaking soundness.
        //
        // By forcing the prover to perform this expensive proof-of-work *after* committing but
        // *before* receiving the queries, we make it computationally infeasible to "shop" for
        // favorable challenges. The grinding effectively "locks in" the prover's commitment.
        if round_params.pow_bits > 0 {
            proof.rounds[round_index].pow_witness = challenger.grind(round_params.pow_bits);
        }

        challenger.sample();

        // STIR Queries
        let stir_challenges_indexes = get_challenge_stir_queries::<Challenger, F, EF>(
            round_params.domain_size,
            self.folding_factor.at_round(round_index),
            round_params.num_queries,
            challenger,
        )?;

        let stir_vars = stir_challenges_indexes
            .iter()
            .map(|&i| round_params.folded_domain_gen.exp_u64(i as u64))
            .collect::<Vec<_>>();

        let mut stir_statement = SelectStatement::initialize(num_variables);

        // Initialize vector of queries
        let mut queries = Vec::with_capacity(stir_challenges_indexes.len());

        // Collect Merkle proofs for stir queries
        match &round_state.merkle_prover_data {
            None => {
                let mut answers = Vec::with_capacity(stir_challenges_indexes.len());
                for challenge in &stir_challenges_indexes {
                    let commitment =
                        mmcs.open_batch(*challenge, &round_state.commitment_merkle_prover_data);
                    let answer = commitment.opened_values[0].clone();
                    answers.push(answer.clone());

                    queries.push(QueryOpening::Base {
                        values: answer.clone(),
                        proof: commitment.opening_proof,
                    });
                }

                // Process each set of evaluations retrieved from the Merkle tree openings.
                for (answer, var) in answers.iter().zip(stir_vars.into_iter()) {
                    let evals = EvaluationsList::new(answer.clone());
                    // Fold the polynomial represented by the `answer` evaluations using the verifier's challenge.
                    // The evaluation method depends on whether this is a "skip round" or a "standard round".

                    // Case 2: Standard Sumcheck Round
                    //
                    // The `answer` represents a standard multilinear polynomial.

                    // Perform a standard multilinear evaluation at the full challenge point `r`.
                    let eval = evals.evaluate_hypercube_base(&round_state.folding_randomness);
                    stir_statement.add_constraint(var, eval);
                }
            }
            Some(data) => {
                let mut answers = Vec::with_capacity(stir_challenges_indexes.len());
                for challenge in &stir_challenges_indexes {
                    let commitment = extension_mmcs.open_batch(*challenge, data);
                    let answer = commitment.opened_values[0].clone();
                    answers.push(answer.clone());
                    queries.push(QueryOpening::Extension {
                        values: answer.clone(),
                        proof: commitment.opening_proof,
                    });
                }

                // Process each set of evaluations retrieved from the Merkle tree openings.
                for (answer, var) in answers.iter().zip(stir_vars.into_iter()) {
                    // Wrap the evaluations to represent the polynomial.
                    let evals = EvaluationsList::new(answer.clone());
                    // Perform a standard multilinear evaluation at the full challenge point `r`.
                    let eval = evals.evaluate_hypercube_ext::<F>(&round_state.folding_randomness);
                    stir_statement.add_constraint(var, eval);
                }
            }
        }

        // Store queries in proof
        proof.rounds[round_index].queries = queries;

        let constraint = Constraint::new(
            challenger.sample_algebra_element(),
            ood_statement,
            stir_statement,
        );

        let mut sumcheck_data: SumcheckData<F, EF> = SumcheckData::default();
        let folding_randomness = round_state.sumcheck_prover.compute_sumcheck_polynomials(
            &mut sumcheck_data,
            challenger,
            folding_factor_next,
            round_params.folding_pow_bits,
            Some(constraint),
        );
        proof.set_sumcheck_data_at(sumcheck_data, round_index);

        // Update round state
        round_state.folding_randomness = folding_randomness;
        round_state.merkle_prover_data = Some(prover_data);

        Ok(())
    }

    #[instrument(skip_all)]
    fn final_round<P, W, PW, const DIGEST_ELEMS: usize>(
        &self,
        round_index: usize,
        proof: &mut WhirProof<F, EF, W, DIGEST_ELEMS>,
        challenger: &mut Challenger,
        round_state: &mut RoundState<EF, F, W, DenseMatrix<F>, DIGEST_ELEMS>,
    ) -> Result<(), FiatShamirError>
    where
        P: PackedValue<Value = F> + Eq + Send + Sync,
        W: PackedValue<Value = W> + Eq + Send + Sync,
        PW: PackedValue<Value = W> + Eq + Send + Sync,
        H: CryptographicHasher<F, [W; DIGEST_ELEMS]>
            + CryptographicHasher<P, [PW; DIGEST_ELEMS]>
            + Sync,
        C: PseudoCompressionFunction<[W; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[PW; DIGEST_ELEMS], 2>
            + Sync,
        [W; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        // Directly send coefficients of the polynomial to the verifier.
        challenger.observe_algebra_slice(round_state.sumcheck_prover.evals().as_slice());

        // Store the final polynomial in the proof
        proof.final_poly = Some(round_state.sumcheck_prover.evals());

        // CRITICAL: Perform proof-of-work grinding to finalize the transcript before querying.
        //
        // This is a crucial security step to prevent a malicious prover from influencing the
        // verifier's challenges.
        //
        // The verifier's query locations (the `stir_challenges`) are generated based on the
        // current transcript state, which includes the prover's polynomial commitment (the Merkle
        // root) for this round. Without grinding, a prover could repeatedly try different
        // commitments until they find one that results in "easy" queries, breaking soundness.
        //
        // By forcing the prover to perform this expensive proof-of-work *after* committing but
        // *before* receiving the queries, we make it computationally infeasible to "shop" for
        // favorable challenges. The grinding effectively "locks in" the prover's commitment.
        if self.final_pow_bits > 0 {
            proof.final_pow_witness = challenger.grind(self.final_pow_bits);
        }

        // Final verifier queries and answers. The indices are over the folded domain.
        let final_challenge_indexes = get_challenge_stir_queries::<Challenger, F, EF>(
            // The size of the original domain before folding
            self.final_round_config().domain_size,
            // The folding factor we used to fold the previous polynomial
            self.folding_factor.at_round(round_index),
            // Number of final verification queries
            self.final_queries,
            challenger,
        )?;

        // Every query requires opening these many in the previous Merkle tree
        let mmcs = MerkleTreeMmcs::<P, PW, H, C, DIGEST_ELEMS>::new(
            self.merkle_hash.clone(),
            self.merkle_compress.clone(),
        );
        let extension_mmcs = ExtensionMmcs::new(mmcs.clone());

        match &round_state.merkle_prover_data {
            None => {
                for challenge in final_challenge_indexes {
                    let commitment =
                        mmcs.open_batch(challenge, &round_state.commitment_merkle_prover_data);

                    proof.final_queries.push(QueryOpening::Base {
                        values: commitment.opened_values[0].clone(),
                        proof: commitment.opening_proof,
                    });
                }
            }

            Some(data) => {
                for challenge in final_challenge_indexes {
                    let commitment = extension_mmcs.open_batch(challenge, data);
                    proof.final_queries.push(QueryOpening::Extension {
                        values: commitment.opened_values[0].clone(),
                        proof: commitment.opening_proof,
                    });
                }
            }
        }

        // Run final sumcheck if required
        if self.final_sumcheck_rounds > 0 {
            let mut sumcheck_data: SumcheckData<F, EF> = SumcheckData::default();
            round_state.sumcheck_prover.compute_sumcheck_polynomials(
                &mut sumcheck_data,
                challenger,
                self.final_sumcheck_rounds,
                self.final_folding_pow_bits,
                None,
            );
            proof.set_final_sumcheck_data(sumcheck_data);
        }

        Ok(())
    }
}
