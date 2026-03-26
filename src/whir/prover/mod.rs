use alloc::vec::Vec;
use core::ops::Deref;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::{ExtensionMmcs, Mmcs};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::{
    Matrix,
    dense::{DenseMatrix, RowMajorMatrixView},
    extension::FlatMatrixView,
};
use p3_multilinear_util::{evals::EvaluationsList, multilinear::MultilinearPoint};
use round_state::{BatchRoundData, RoundState};
use tracing::{info_span, instrument};

use crate::{
    constraints::{
        Constraint,
        statement::{EqStatement, SelectStatement, initial::InitialStatement},
    },
    fiat_shamir::errors::FiatShamirError,
    parameters::WhirConfig,
    sumcheck::{extrapolate_012, product_polynomial::ProductPolynomial, prover::SumcheckProver},
    whir::{
        proof::{
            BatchWhirProof, QueryOpening, SumcheckData, WhirProof, fold_ood_constraints,
            single_constraint,
        },
        utils::get_challenge_stir_queries,
    },
};

pub mod round_state;

pub type Proof<W, const DIGEST_ELEMS: usize> = Vec<Vec<[W; DIGEST_ELEMS]>>;
pub type Leafs<F> = Vec<Vec<F>>;

#[derive(Debug)]
pub struct Prover<'a, EF, F, MT, Challenger>(
    /// Reference to the protocol configuration shared across prover components.
    pub &'a WhirConfig<EF, F, MT, Challenger>,
)
where
    F: Field,
    EF: ExtensionField<F>;

impl<EF, F, MT, Challenger> Deref for Prover<'_, EF, F, MT, Challenger>
where
    F: Field,
    EF: ExtensionField<F>,
{
    type Target = WhirConfig<EF, F, MT, Challenger>;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl<EF, F, MT, Challenger> Prover<'_, EF, F, MT, Challenger>
where
    F: TwoAdicField + Ord,
    EF: ExtensionField<F> + TwoAdicField,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    MT: Mmcs<F>,
{
    /// Validates that the total number of variables expected by the prover configuration
    /// matches the number implied by the folding schedule and the final rounds.
    ///
    /// This ensures that the recursive folding in the sumcheck protocol terminates
    /// precisely at the expected number of final variables.
    ///
    /// # Returns
    /// `true` if the parameter configuration is consistent, `false` otherwise.
    pub(crate) const fn validate_parameters(&self) -> bool {
        self.0.num_variables
            == self.0.folding_factor.total_number(self.0.n_rounds()) + self.0.final_sumcheck_rounds
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
    pub fn prove<Dft>(
        &self,
        dft: &Dft,
        proof: &mut WhirProof<F, EF, MT>,
        challenger: &mut Challenger,
        statement: &InitialStatement<F, EF>,
        prover_data: MT::ProverData<DenseMatrix<F>>,
    ) -> Result<(), FiatShamirError>
    where
        Dft: TwoAdicSubgroupDft<F>,
        Challenger: CanObserve<MT::Commitment>,
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
            self.round(dft, round, proof, challenger, &mut round_state)?;
        }

        Ok(())
    }

    #[instrument(skip_all, fields(round_number = round_index, log_size = self.num_variables - self.folding_factor.total_number(round_index)))]
    #[allow(clippy::too_many_lines)]
    #[allow(clippy::type_complexity)]
    pub(crate) fn round<Dft: TwoAdicSubgroupDft<F>>(
        &self,
        dft: &Dft,
        round_index: usize,
        proof: &mut WhirProof<F, EF, MT>,
        challenger: &mut Challenger,
        round_state: &mut RoundState<
            EF,
            F,
            MT::ProverData<DenseMatrix<F>>,
            MT::ProverData<FlatMatrixView<F, EF, DenseMatrix<EF>>>,
        >,
    ) -> Result<(), FiatShamirError>
    where
        Challenger: CanObserve<MT::Commitment>,
    {
        let folded_evaluations = &round_state.sumcheck_prover.evals();
        let num_variables = self.num_variables - self.folding_factor.total_number(round_index);
        assert_eq!(num_variables, folded_evaluations.num_variables());

        // Base case: final round reached
        if round_index == self.n_rounds() {
            return self.final_round(round_index, proof, challenger, round_state);
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

        let extension_mmcs = ExtensionMmcs::new(self.mmcs.clone());
        let (root, prover_data) =
            info_span!("commit matrix").in_scope(|| extension_mmcs.commit_matrix(folded_matrix));

        // Observe the round merkle tree commitment
        challenger.observe(root.clone());

        // Store commitment in proof
        proof.rounds[round_index].commitment = Some(root);

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
            None if round_state.batch_data.is_some() => {
                // Batch mode: open BOTH commitment trees and fold values
                let batch = round_state.batch_data.as_ref().unwrap();
                let one_minus_r0 = EF::ONE - batch.r_0;

                let mut folded_answers: Vec<Vec<EF>> =
                    Vec::with_capacity(stir_challenges_indexes.len());

                for challenge in &stir_challenges_indexes {
                    let commit_a = self
                        .mmcs
                        .open_batch(*challenge, &round_state.commitment_merkle_prover_data);
                    let commit_b = self.mmcs.open_batch(*challenge, &batch.base_data);

                    let values_a = commit_a.opened_values[0].clone();
                    let values_b = commit_b.opened_values[0].clone();

                    // Fold opened values: g(b) = r_0·f_a(b) + (1-r_0)·f_b(b)
                    let folded: Vec<EF> = values_a
                        .iter()
                        .zip(values_b.iter())
                        .map(|(&a, &b)| batch.r_0 * EF::from(a) + one_minus_r0 * EF::from(b))
                        .collect();
                    folded_answers.push(folded);

                    queries.push(QueryOpening::Batch {
                        values_a,
                        proof_a: commit_a.opening_proof,
                        values_b,
                        proof_b: commit_b.opening_proof,
                    });
                }

                // Process folded evaluations for STIR constraints
                for (answer, var) in folded_answers.iter().zip(stir_vars.into_iter()) {
                    let evals = EvaluationsList::new(answer.clone());
                    let eval = evals.evaluate_hypercube_ext::<F>(&round_state.folding_randomness);
                    stir_statement.add_constraint(var, eval);
                }
            }
            None => {
                // Single-polynomial mode: open base commitment
                let mut answers = Vec::with_capacity(stir_challenges_indexes.len());
                for challenge in &stir_challenges_indexes {
                    let commitment = self
                        .mmcs
                        .open_batch(*challenge, &round_state.commitment_merkle_prover_data);
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
    #[allow(clippy::type_complexity)]
    fn final_round(
        &self,
        round_index: usize,
        proof: &mut WhirProof<F, EF, MT>,
        challenger: &mut Challenger,
        round_state: &mut RoundState<
            EF,
            F,
            MT::ProverData<DenseMatrix<F>>,
            MT::ProverData<FlatMatrixView<F, EF, DenseMatrix<EF>>>,
        >,
    ) -> Result<(), FiatShamirError>
where {
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

        let extension_mmcs = ExtensionMmcs::new(self.mmcs.clone());
        match &round_state.merkle_prover_data {
            None if round_state.batch_data.is_some() => {
                // Batch mode: open both trees and store batch queries
                let batch = round_state.batch_data.as_ref().unwrap();
                for challenge in final_challenge_indexes {
                    let commit_a = self
                        .mmcs
                        .open_batch(challenge, &round_state.commitment_merkle_prover_data);
                    let commit_b = self.mmcs.open_batch(challenge, &batch.base_data);

                    proof.final_queries.push(QueryOpening::Batch {
                        values_a: commit_a.opened_values[0].clone(),
                        proof_a: commit_a.opening_proof,
                        values_b: commit_b.opened_values[0].clone(),
                        proof_b: commit_b.opening_proof,
                    });
                }
            }
            None => {
                for challenge in final_challenge_indexes {
                    let commitment = self
                        .mmcs
                        .open_batch(challenge, &round_state.commitment_merkle_prover_data);

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

    /// Performs the selector sumcheck round for batch opening.
    ///
    /// This is a single round of sumcheck over the selector variable X, which
    /// folds two polynomials f_a and f_b into a single polynomial g.
    ///
    /// Returns `(sumcheck_prover, r_0)` where:
    /// - `sumcheck_prover` contains the folded polynomial g and weight w'
    /// - `r_0` is the selector challenge from Fiat-Shamir
    #[instrument(skip_all)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn selector_round(
        &self,
        selector_data: &mut SumcheckData<F, EF>,
        challenger: &mut Challenger,
        f_a: &EvaluationsList<F>,
        f_b: &EvaluationsList<F>,
        z_a: &MultilinearPoint<EF>,
        z_b: &MultilinearPoint<EF>,
        v_a: EF,
        v_b: EF,
        alpha: EF,
    ) -> (SumcheckProver<F, EF>, EF) {
        let n = f_a.num_variables();
        debug_assert_eq!(n, f_b.num_variables());
        debug_assert_eq!(n, z_a.num_variables());
        debug_assert_eq!(n, z_b.num_variables());

        // Build virtual combined polynomial: f_c = [f_b | f_a]
        // X=0 half is f_b, X=1 half is f_a
        let combined_evals: Vec<F> = f_b
            .as_slice()
            .iter()
            .chain(f_a.as_slice().iter())
            .copied()
            .collect();
        let combined_poly = EvaluationsList::new(combined_evals);

        // Build combined weights: w = [α·eq(·, z_b) | eq(·, z_a)]
        let eq_z_b = EvaluationsList::new_from_point(z_b.as_slice(), alpha);
        let eq_z_a = EvaluationsList::new_from_point(z_a.as_slice(), EF::ONE);
        let combined_weights: Vec<EF> = eq_z_b
            .as_slice()
            .iter()
            .chain(eq_z_a.as_slice().iter())
            .copied()
            .collect();
        let combined_weights = EvaluationsList::new(combined_weights);

        // Compute sumcheck coefficients: h(X) = c0 + c1·X + c2·X²
        let (c0, c2) = combined_poly.sumcheck_coefficients(&combined_weights);

        // Sanity check: h(0) = sum_{x} f_b(x)·α·eq(x,z_b) = α·v_b
        debug_assert_eq!(c0, alpha * v_b);

        // Fiat-Shamir: commit (c0, c2) and receive challenge r_0
        let r_0 = selector_data.observe_and_sample::<_, F>(
            challenger,
            c0,
            c2,
            self.starting_folding_pow_bits,
        );

        // Materialize g(x) = r_0·f_a(x) + (1-r_0)·f_b(x)
        let one_minus_r0 = EF::ONE - r_0;
        let g: Vec<EF> = f_a
            .as_slice()
            .iter()
            .zip(f_b.as_slice().iter())
            .map(|(&a, &b)| r_0 * EF::from(a) + one_minus_r0 * EF::from(b))
            .collect();
        let g = EvaluationsList::new(g);

        // Materialize w'(x) = r_0·eq(x,z_a) + α·(1-r_0)·eq(x,z_b)
        let w_prime: Vec<EF> = eq_z_a
            .as_slice()
            .iter()
            .zip(eq_z_b.as_slice().iter())
            .map(|(&a, &b)| r_0 * a + one_minus_r0 * b)
            .collect();
        let w_prime = EvaluationsList::new(w_prime);

        // Compute folded sum: σ' = h(r_0) via quadratic extrapolation
        let sigma = v_a + alpha * v_b;
        let sigma_prime = extrapolate_012(c0, sigma - c0, c2, r_0);

        // Create sumcheck prover for continuation
        let poly = ProductPolynomial::new_small(g, w_prime);
        debug_assert_eq!(poly.dot_product(), sigma_prime);

        (
            SumcheckProver {
                poly,
                sum: sigma_prime,
            },
            r_0,
        )
    }

    /// Executes the batch opening proof protocol for two polynomials.
    #[instrument(skip_all)]
    #[allow(clippy::too_many_arguments)]
    pub fn batch_prove<Dft>(
        &self,
        dft: &Dft,
        proof: &mut BatchWhirProof<F, EF, MT>,
        challenger: &mut Challenger,
        prover_data_a: MT::ProverData<DenseMatrix<F>>,
        prover_data_b: MT::ProverData<DenseMatrix<F>>,
        f_a: &EvaluationsList<F>,
        f_b: &EvaluationsList<F>,
        statement_a: &EqStatement<EF>,
        statement_b: &EqStatement<EF>,
        ood_statement_a: &EqStatement<EF>,
        ood_statement_b: &EqStatement<EF>,
    ) -> Result<(), FiatShamirError>
    where
        Dft: TwoAdicSubgroupDft<F>,
        Challenger: CanObserve<MT::Commitment>,
    {
        let num_variables = f_a.num_variables();
        assert_eq!(num_variables, f_b.num_variables());
        assert!(self.validate_parameters());

        // Extract single evaluation claims: f_a(z_a) = v_a, f_b(z_b) = v_b
        let (z_a, v_a) = single_constraint(statement_a);
        let (z_b, v_b) = single_constraint(statement_b);

        // Sample batching randomness α
        let alpha: EF = challenger.sample_algebra_element();

        // Run selector sumcheck round
        let (mut sumcheck_prover, r_0) = self.selector_round(
            &mut proof.selector_sumcheck,
            challenger,
            f_a,
            f_b,
            &z_a,
            &z_b,
            v_a,
            v_b,
            alpha,
        );

        // Fold OOD constraints: ood_folded = r_0·ood_a + (1-r_0)·ood_b
        let folded_ood = fold_ood_constraints(ood_statement_a, ood_statement_b, r_0);

        // Combine folded OOD constraints into the sumcheck (same as initial round in single-poly)
        let ood_constraint =
            Constraint::new_eq_only(challenger.sample_algebra_element(), folded_ood);

        // Run folding_factor rounds of sumcheck incorporating OOD constraints
        let folding_factor = self.folding_factor.at_round(0);
        let folding_randomness = sumcheck_prover.compute_sumcheck_polynomials(
            &mut proof.inner_proof.initial_sumcheck,
            challenger,
            folding_factor,
            self.starting_folding_pow_bits,
            Some(ood_constraint),
        );

        // Initialize RoundState with BOTH commitment trees
        let mut round_state = RoundState {
            sumcheck_prover,
            folding_randomness,
            commitment_merkle_prover_data: prover_data_a,
            merkle_prover_data: None,
            batch_data: Some(BatchRoundData {
                base_data: prover_data_b,
                r_0,
            }),
        };

        // Run standard WHIR rounds
        for round in 0..=self.n_rounds() {
            self.round(
                dft,
                round,
                &mut proof.inner_proof,
                challenger,
                &mut round_state,
            )?;
        }

        Ok(())
    }
}
