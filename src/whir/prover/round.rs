use p3_field::{ExtensionField, Field, PrimeField64, TwoAdicField};

use super::{Leafs, Proof, Prover};
use crate::{
    domain::Domain,
    fiat_shamir::{errors::ProofResult, pow::traits::PowStrategy, prover::ProverState},
    poly::{coeffs::CoefficientStorage, multilinear::MultilinearPoint},
    sumcheck::sumcheck_single::SumcheckSingle,
    whir::{
        committer::{CommitmentMerkleTree, RoundMerkleTree, Witness},
        statement::{Statement, Weights},
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
pub(crate) struct RoundState<EF, F, const DIGEST_ELEMS: usize>
where
    F: Field + TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
{
    /// Current round index (starts from 0).
    pub(crate) round: usize,

    /// The domain used in this round, including the size and generator.
    /// This is typically a scaled version of the previous round’s domain.
    pub(crate) domain: Domain<EF, F>,

    /// The sumcheck prover responsible for managing constraint accumulation and sumcheck rounds.
    /// Initialized in the first round (if applicable), and reused/updated in each subsequent round.
    pub(crate) sumcheck_prover: Option<SumcheckSingle<F, EF>>,

    /// The sampled folding randomness for this round, used to collapse a subset of variables.
    /// Length equals the folding factor at this round.
    pub(crate) folding_randomness: MultilinearPoint<EF>,

    /// The multilinear polynomial coefficients at the start of this round.
    /// These are updated by folding the previous round’s coefficients using `folding_randomness`.
    pub(crate) coefficients: CoefficientStorage<F, EF>,

    /// Merkle commitment prover data for the **base field** polynomial from the first round.
    /// This is used to open values at queried locations.
    pub(crate) commitment_merkle_prover_data: CommitmentMerkleTree<F, DIGEST_ELEMS>,

    /// Merkle commitment prover data for the **extension field** polynomials (folded rounds).
    /// Present only after the first round.
    pub(crate) merkle_prover_data: Option<RoundMerkleTree<F, EF, DIGEST_ELEMS>>,

    /// Merkle proof from the initial commitment round.
    /// - First: list of opened leaf values in the base field.
    /// - Second: corresponding Merkle authentication paths.
    /// - Empty during setup; populated during final query phase.
    pub(crate) commitment_merkle_proof: Option<(Leafs<F>, Proof<DIGEST_ELEMS>)>,

    /// Merkle proofs for intermediate folded rounds.
    /// Each entry contains:
    /// - The opened values at verifier-chosen locations,
    /// - The corresponding authentication paths.
    pub(crate) merkle_proofs: Vec<(Leafs<EF>, Proof<DIGEST_ELEMS>)>,

    /// Flat vector of challenge values used across all rounds.
    /// Populated progressively as folding randomness is sampled.
    /// The `i`-th index corresponds to variable `X_{n - 1 - i}`.
    pub(crate) randomness_vec: Vec<EF>,

    /// The accumulated set of linear equality constraints for this round.
    /// Used in computing the weighted sum for the sumcheck polynomial.
    pub(crate) statement: Statement<EF>,
}

impl<EF, F, const DIGEST_ELEMS: usize> RoundState<EF, F, DIGEST_ELEMS>
where
    F: Field + TwoAdicField + PrimeField64,
    EF: ExtensionField<F> + TwoAdicField,
{
    /// Initializes the prover’s state for the first round of the WHIR protocol.
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
    pub(crate) fn initialize_first_round_state<H, C, PS>(
        prover: &Prover<EF, F, H, C, PS>,
        prover_state: &mut ProverState<EF, F>,
        mut statement: Statement<EF>,
        witness: Witness<EF, F, DIGEST_ELEMS>,
    ) -> ProofResult<Self>
    where
        PS: PowStrategy,
    {
        // Convert witness ood_points into constraints
        let new_constraints = witness
            .ood_points
            .into_iter()
            .zip(witness.ood_answers)
            .map(|(point, evaluation)| {
                let weights = Weights::evaluation(MultilinearPoint::expand_from_univariate(
                    point,
                    prover.mv_parameters.num_variables,
                ));
                (weights, evaluation)
            })
            .collect();

        statement.add_constraints_in_front(new_constraints);

        let mut sumcheck_prover = None;
        let folding_randomness = if prover.initial_statement {
            // If there is initial statement, then we run the sum-check for
            // this initial statement.
            let [combination_randomness_gen] = prover_state.challenge_scalars()?;

            // Create the sumcheck prover
            let mut sumcheck = SumcheckSingle::from_base_coeffs(
                witness.polynomial.clone(),
                &statement,
                combination_randomness_gen,
            );

            // Compute sumcheck polynomials and return the folding randomness values
            let folding_randomness = sumcheck.compute_sumcheck_polynomials::<PS>(
                prover_state,
                prover.folding_factor.at_round(0),
                prover.starting_folding_pow_bits,
            )?;

            sumcheck_prover = Some(sumcheck);
            folding_randomness
        } else {
            // If there is no initial statement, there is no need to run the
            // initial rounds of the sum-check, and the verifier directly sends
            // the initial folding randomnesses.
            let mut folding_randomness = vec![EF::ZERO; prover.folding_factor.at_round(0)];
            prover_state.fill_challenge_scalars(&mut folding_randomness)?;

            if prover.starting_folding_pow_bits > 0. {
                prover_state.challenge_pow::<PS>(prover.starting_folding_pow_bits)?;
            }
            MultilinearPoint(folding_randomness)
        };
        let mut randomness_vec = Vec::with_capacity(prover.mv_parameters.num_variables);
        randomness_vec.extend(folding_randomness.0.iter().rev().copied());
        randomness_vec.resize(prover.mv_parameters.num_variables, EF::ZERO);

        Ok(Self {
            domain: prover.starting_domain.clone(),
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
        })
    }
}
