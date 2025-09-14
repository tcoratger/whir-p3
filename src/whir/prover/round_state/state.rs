//! Round state implementation for WHIR protocol.

use std::sync::Arc;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::dense::DenseMatrix;
use p3_merkle_tree::MerkleTree;
use tracing::instrument;

use crate::{
    constant::K_SKIP_SUMCHECK,
    fiat_shamir::{errors::ProofResult, prover::ProverState},
    poly::multilinear::MultilinearPoint,
    sumcheck::sumcheck_single::SumcheckSingle,
    whir::{
        committer::{RoundMerkleTree, Witness},
        prover::Prover,
        statement::Statement,
    },
};

/// Holds all per-round prover state required during the execution of the WHIR protocol.
///
/// Each round involves:
/// - A domain extension and folding step,
/// - Merkle commitments and openings,
/// - A sumcheck polynomial generation and folding randomness sampling,
/// - Bookkeeping of constraints and evaluation points.
///
/// The `RoundState` evolves with each round and captures all intermediate data required
/// to continue proving or to verify challenges from the verifier.
#[derive(Debug)]
pub struct RoundState<EF, F, W, M, const DIGEST_ELEMS: usize>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
{
    /// The size of the current evaluation domain.
    ///
    /// This represents the number of evaluation points in the current round's domain,
    /// which decreases by a factor of 2^k in each round where k is the folding factor.
    pub domain_size: usize,

    /// Generator for the next (smaller) evaluation domain.
    ///
    /// This is the primitive root of unity for the domain that will be used in the
    /// next round after folding. It's computed as the current generator raised to
    /// the power of the reduction factor (2^folding_factor).
    pub next_domain_gen: F,

    /// The sumcheck prover responsible for managing constraint accumulation and sumcheck rounds.
    /// Initialized in the first round (if applicable), and reused/updated in each subsequent round.
    pub sumcheck_prover: SumcheckSingle<F, EF>,

    /// The sampled folding randomness for this round, used to collapse a subset of variables.
    /// Length equals the folding factor at this round.
    pub folding_randomness: MultilinearPoint<EF>,

    /// Merkle commitment prover data for the **base field** polynomial from the first round.
    /// This is used to open values at queried locations.
    pub commitment_merkle_prover_data: Arc<MerkleTree<F, W, M, DIGEST_ELEMS>>,

    /// Merkle commitment prover data for the **extension field** polynomials (folded rounds).
    /// Present only after the first round.
    pub merkle_prover_data: Option<RoundMerkleTree<F, EF, W, DIGEST_ELEMS>>,

    /// Flat vector of challenge values used across all rounds.
    /// Populated progressively as folding randomness is sampled.
    /// The `i`-th index corresponds to variable `X_{n - 1 - i}`.
    pub randomness_vec: Vec<EF>,

    /// The accumulated set of linear equality constraints for this round.
    /// Used in computing the weighted sum for the sumcheck polynomial.
    pub statement: Statement<EF>,
}

#[allow(clippy::mismatching_type_param_order)]
impl<EF, F, const DIGEST_ELEMS: usize> RoundState<EF, F, F, DenseMatrix<F>, DIGEST_ELEMS>
where
    F: TwoAdicField + Ord,
    EF: ExtensionField<F> + TwoAdicField,
{
    /// Initializes the prover's state for the first round of the WHIR protocol.
    ///
    /// This function prepares all round-local state needed to begin the interactive proof:
    /// - If the WHIR protocol has an initial statement, it runs the first sumcheck round and
    ///   samples folding randomness using Fiat-Shamir.
    /// - Otherwise, it directly absorbs verifier-supplied randomness for folding.
    /// - It incorporates any out-of-domain (OOD) constraints derived from the witness,
    ///   and prepares the polynomial coefficients accordingly.
    ///
    /// Returns a fully-formed `RoundState` containing:
    /// - The active domain,
    /// - The initial polynomial (as coefficients),
    /// - The first sumcheck prover (if applicable),
    /// - The sampled folding randomness,
    /// - Constraint tracking data,
    /// - Merkle tree commitment data.
    ///
    /// This function should be called once at the beginning of the proof, before entering the
    /// main WHIR folding loop.
    #[instrument(skip_all)]
    pub fn initialize_first_round_state<MyChallenger, C, Challenger>(
        prover: &Prover<'_, EF, F, MyChallenger, C, Challenger>,
        prover_state: &mut ProverState<F, EF, Challenger>,
        mut statement: Statement<EF>,
        witness: Witness<EF, F, DenseMatrix<F>, DIGEST_ELEMS>,
    ) -> ProofResult<Self>
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
        MyChallenger: Clone,
        C: Clone,
    {
        // Add constraints from OOD evaluations of the witness
        let new_constraints = witness
            .ood_points
            .into_iter()
            .zip(witness.ood_answers)
            .map(|(point, evaluation)| {
                let constraint_point =
                    MultilinearPoint::expand_from_univariate(point, prover.num_variables);
                (constraint_point, evaluation)
            })
            .collect();

        statement.add_constraints_in_front(new_constraints);

        // Generate sumcheck and folding randomness based on whether we have an initial statement
        let (sumcheck_prover, folding_randomness) = if prover.initial_statement {
            // Run sumcheck on the initial statement
            let combination_randomness_gen: EF = prover_state.sample();

            let (sumcheck, folding_randomness) =
                if prover.univariate_skip && K_SKIP_SUMCHECK <= prover.folding_factor.at_round(0) {
                    SumcheckSingle::with_skip(
                        &witness.polynomial,
                        &statement,
                        combination_randomness_gen,
                        prover_state,
                        prover.folding_factor.at_round(0),
                        prover.starting_folding_pow_bits,
                        K_SKIP_SUMCHECK,
                    )
                } else {
                    SumcheckSingle::from_base_evals(
                        &witness.polynomial,
                        &statement,
                        combination_randomness_gen,
                        prover_state,
                        prover.folding_factor.at_round(0),
                        prover.starting_folding_pow_bits,
                    )
                };

            (sumcheck, folding_randomness)
        } else {
            // If there is no initial statement, sample folding randomness directly
            let folding_randomness = MultilinearPoint::new(
                (0..prover.folding_factor.at_round(0))
                    .map(|_| prover_state.sample())
                    .collect::<Vec<_>>(),
            );

            let poly = witness.polynomial.fold(&folding_randomness);
            let num_variables = poly.num_variables();

            // Create the sumcheck prover without running any rounds
            let sumcheck =
                SumcheckSingle::from_extension_evals(poly, &Statement::new(num_variables), EF::ONE);

            prover_state.pow_grinding(prover.starting_folding_pow_bits);

            (sumcheck, folding_randomness)
        };

        // Build the randomness vector
        let mut randomness_vec = Vec::with_capacity(prover.num_variables);
        randomness_vec.extend(folding_randomness.iter().rev().copied());
        randomness_vec.resize(prover.num_variables, EF::ZERO);

        Ok(Self {
            domain_size: prover.starting_domain_size(),
            next_domain_gen: F::two_adic_generator(
                prover.starting_domain_size().ilog2() as usize
                    - prover.folding_factor.at_round(0),
            ),
            sumcheck_prover,
            folding_randomness,
            commitment_merkle_prover_data: witness.prover_data,
            merkle_prover_data: None,
            randomness_vec,
            statement,
        })
    }
}