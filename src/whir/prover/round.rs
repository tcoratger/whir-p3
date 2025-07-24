use std::sync::Arc;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::dense::DenseMatrix;
use p3_merkle_tree::MerkleTree;
use tracing::{info_span, instrument};

use super::Prover;
use crate::{
    fiat_shamir::{errors::ProofResult, prover::ProverState},
    poly::multilinear::MultilinearPoint,
    sumcheck::{K_SKIP_SUMCHECK, sumcheck_single::SumcheckSingle},
    whir::{
        committer::{RoundMerkleTree, Witness},
        statement::{Statement, weights::Weights},
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
pub(crate) struct RoundState<EF, F, W, M, const DIGEST_ELEMS: usize>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
{
    pub(crate) domain_size: usize,

    pub(crate) next_domain_gen: F,

    /// The sumcheck prover responsible for managing constraint accumulation and sumcheck rounds.
    /// Initialized in the first round (if applicable), and reused/updated in each subsequent round.
    pub(crate) sumcheck_prover: SumcheckSingle<F, EF>,

    /// The sampled folding randomness for this round, used to collapse a subset of variables.
    /// Length equals the folding factor at this round.
    pub(crate) folding_randomness: MultilinearPoint<EF>,

    /// Merkle commitment prover data for the **base field** polynomial from the first round.
    /// This is used to open values at queried locations.
    pub(crate) commitment_merkle_prover_data: Arc<MerkleTree<F, W, M, DIGEST_ELEMS>>,

    /// Merkle commitment prover data for the **extension field** polynomials (folded rounds).
    /// Present only after the first round.
    pub(crate) merkle_prover_data: Option<RoundMerkleTree<F, EF, W, DIGEST_ELEMS>>,

    /// Flat vector of challenge values used across all rounds.
    /// Populated progressively as folding randomness is sampled.
    /// The `i`-th index corresponds to variable `X_{n - 1 - i}`.
    pub(crate) randomness_vec: Vec<EF>,

    /// The accumulated set of linear equality constraints for this round.
    /// Used in computing the weighted sum for the sumcheck polynomial.
    pub(crate) statement: Statement<EF>,
}

#[allow(clippy::mismatching_type_param_order)]
impl<EF, F, const DIGEST_ELEMS: usize> RoundState<EF, F, F, DenseMatrix<F>, DIGEST_ELEMS>
where
    F: TwoAdicField,
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
    #[instrument(skip_all)]
    pub(crate) fn initialize_first_round_state<MyChallenger, C, Challenger>(
        prover: &Prover<'_, EF, F, MyChallenger, C, Challenger>,
        prover_state: &mut ProverState<F, EF, Challenger>,
        mut statement: Statement<EF>,
        witness: Witness<EF, F, DenseMatrix<F>, DIGEST_ELEMS>,
    ) -> ProofResult<Self>
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
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

        let (sumcheck_prover, folding_randomness) = if prover.initial_statement {
            // If there is initial statement, then we run the sum-check for
            // this initial statement.
            let combination_randomness_gen: EF = prover_state.sample();

            // Create the sumcheck prover
            let (sumcheck, folding_randomness) = if prover.univariate_skip {
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
            // If there is no initial statement, there is no need to run the
            // initial rounds of the sum-check, and the verifier directly sends
            // the initial folding randomnesses.

            let folding_randomness = MultilinearPoint(
                (0..prover.folding_factor.at_round(0))
                    .map(|_| prover_state.sample())
                    .collect::<Vec<_>>(),
            );

            let poly = witness.polynomial.fold(&folding_randomness);
            let num_variables = poly.num_variables();

            // Create the sumcheck prover w/o running any rounds.
            let sumcheck =
                SumcheckSingle::from_extension_evals(poly, &Statement::new(num_variables), EF::ONE);

            prover_state.pow_grinding(prover.starting_folding_pow_bits);

            (sumcheck, folding_randomness)
        };

        let randomness_vec = info_span!("copy_across_random_vec").in_scope(|| {
            let mut randomness_vec = Vec::with_capacity(prover.mv_parameters.num_variables);
            randomness_vec.extend(folding_randomness.iter().rev().copied());
            randomness_vec.resize(prover.mv_parameters.num_variables, EF::ZERO);
            randomness_vec
        });

        Ok(Self {
            domain_size: prover.starting_domain_size(),
            next_domain_gen: F::two_adic_generator(
                prover.starting_domain_size().ilog2() as usize - prover.folding_factor.at_round(0),
            ),
            sumcheck_prover,
            folding_randomness,
            merkle_prover_data: None,
            commitment_merkle_prover_data: witness.prover_data,
            randomness_vec,
            statement,
        })
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_field::extension::BinomialExtensionField;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::{SeedableRng, rngs::SmallRng};

    use super::*;
    use crate::{
        dft::EvalsDft,
        fiat_shamir::domain_separator::DomainSeparator,
        parameters::{
            FoldingFactor, MultivariateParameters, ProtocolParameters, errors::SecurityAssumption,
        },
        poly::evals::EvaluationsList,
        whir::{WhirConfig, committer::writer::CommitmentWriter},
    };

    type F = BabyBear;
    type EF4 = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;

    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

    const DIGEST_ELEMS: usize = 8;

    /// Create a WHIR protocol configuration for test scenarios.
    ///
    /// This utility function builds a `WhirConfig` using the provided parameters:
    /// - `num_variables`: Number of variables in the multilinear polynomial.
    /// - `initial_statement`: Whether to start with an initial sumcheck statement.
    /// - `folding_factor`: Number of variables to fold per round.
    /// - `pow_bits`: Difficulty of the proof-of-work challenge used in Fiat-Shamir.
    ///
    /// The returned config can be used to initialize a prover and set up domain commitments
    /// for round state construction in WHIR tests.
    fn make_test_config(
        num_variables: usize,
        initial_statement: bool,
        folding_factor: usize,
        pow_bits: usize,
    ) -> WhirConfig<EF4, F, MyHash, MyCompress, MyChallenger> {
        // Construct the multivariate parameter set with `num_variables` variables,
        // determining the size of the evaluation domain.
        let mv_params = MultivariateParameters::<EF4>::new(num_variables);

        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);

        let merkle_hash = MyHash::new(perm.clone());
        let merkle_compress = MyCompress::new(perm);

        // Define the core protocol parameters for WHIR, customizing behavior based
        // on whether to start with an initial sumcheck and how to fold the polynomial.
        let protocol_params = ProtocolParameters {
            initial_statement,
            security_level: 80,
            pow_bits,
            rs_domain_initial_reduction_factor: 1,
            folding_factor: FoldingFactor::Constant(folding_factor),
            merkle_hash,
            merkle_compress,
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: 1,
            univariate_skip: false,
        };

        // Combine the multivariate and protocol parameters into a full WHIR config
        WhirConfig::new(mv_params, protocol_params)
    }

    /// Prepare the Fiat-Shamir domain, prover state, and Merkle commitment for a test polynomial.
    ///
    /// This helper sets up the necessary transcript (`DomainSeparator`) and
    /// commits to a polynomial using a `CommitmentWriter`. It returns:
    /// - the initialized domain separator,
    /// - the `ProverState` transcript context for Fiat-Shamir interaction,
    /// - and a `Witness` object containing the committed polynomial and Merkle data.
    ///
    /// This is used as a boilerplate step before running the first WHIR round.
    #[allow(clippy::type_complexity)]
    fn setup_domain_and_commitment(
        params: &WhirConfig<EF4, F, MyHash, MyCompress, MyChallenger>,
        poly: EvaluationsList<F>,
    ) -> (
        DomainSeparator<EF4, F>,
        ProverState<F, EF4, MyChallenger>,
        Witness<EF4, F, DenseMatrix<F>, DIGEST_ELEMS>,
    ) {
        // Create a new Fiat-Shamir domain separator.
        let mut domsep = DomainSeparator::new(vec![]);

        // Observe the public statement into the transcript for binding.
        domsep.commit_statement::<_, _, _, 8>(params);

        // Reserve transcript space for WHIR proof messages.
        domsep.add_whir_proof::<_, _, _, 8>(params);

        let mut rng = SmallRng::seed_from_u64(1);
        let challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));

        // Convert the domain separator into a mutable prover-side transcript.
        let mut prover_state = domsep.to_prover_state::<_>(challenger);

        // Create a committer using the protocol configuration (Merkle parameters, hashers, etc.).
        let committer = CommitmentWriter::new(params);

        // Perform DFT-based commitment to the polynomial, producing a witness
        // which includes the Merkle tree and polynomial values.
        let witness = committer
            .commit(&EvalsDft::<F>::default(), &mut prover_state, poly)
            .unwrap();

        // Return all initialized components needed for round state setup.
        (domsep, prover_state, witness)
    }
}
