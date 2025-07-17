use std::sync::Arc;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::dense::DenseMatrix;
use p3_merkle_tree::MerkleTree;
use tracing::{info_span, instrument};

use super::Prover;
use crate::{
    domain::Domain,
    fiat_shamir::{errors::ProofResult, prover::ProverState},
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
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
    /// The domain used in this round, including the size and generator.
    /// This is typically a scaled version of the previous round’s domain.
    pub(crate) domain: Domain<EF>,

    /// The sumcheck prover responsible for managing constraint accumulation and sumcheck rounds.
    /// Initialized in the first round (if applicable), and reused/updated in each subsequent round.
    pub(crate) sumcheck_prover: Option<SumcheckSingle<F, EF>>,

    /// The sampled folding randomness for this round, used to collapse a subset of variables.
    /// Length equals the folding factor at this round.
    pub(crate) folding_randomness: MultilinearPoint<EF>,

    /// The multilinear polynomial evaluations at the start of this round.
    /// These are updated by folding the previous round’s coefficients using `folding_randomness`.
    pub(crate) initial_evaluations: Option<EvaluationsList<F>>,

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

        let mut sumcheck_prover = None;
        let folding_randomness = if prover.initial_statement {
            // If there is initial statement, then we run the sum-check for
            // this initial statement.
            let combination_randomness_gen: EF = prover_state.sample();

            // Create the sumcheck prover
            let (sumcheck, folding_randomness) = if !prover.univariate_skip {
                SumcheckSingle::from_base_evals(
                    &witness.polynomial,
                    &statement,
                    combination_randomness_gen,
                    prover_state,
                    prover.folding_factor.at_round(0),
                    prover.starting_folding_pow_bits,
                )
            } else {
                SumcheckSingle::with_skip(
                    &witness.polynomial,
                    &statement,
                    combination_randomness_gen,
                    prover_state,
                    prover.folding_factor.at_round(0),
                    prover.starting_folding_pow_bits,
                    K_SKIP_SUMCHECK,
                )
            };

            sumcheck_prover = Some(sumcheck);
            folding_randomness
        } else {
            // If there is no initial statement, there is no need to run the
            // initial rounds of the sum-check, and the verifier directly sends
            // the initial folding randomnesses.
            let folding_randomness = std::iter::repeat(prover_state.sample())
                .take(prover.folding_factor.at_round(0))
                .collect();
            prover_state.pow_grinding(prover.starting_folding_pow_bits);
            MultilinearPoint(folding_randomness)
        };
        let randomness_vec = info_span!("copy_across_random_vec").in_scope(|| {
            let mut randomness_vec = Vec::with_capacity(prover.mv_parameters.num_variables);
            randomness_vec.extend(folding_randomness.0.iter().rev().copied());
            randomness_vec.resize(prover.mv_parameters.num_variables, EF::ZERO);
            randomness_vec
        });

        let initial_evaluations = sumcheck_prover
            .as_ref()
            .map_or(Some(witness.polynomial), |_| None);
        Ok(Self {
            domain: prover.starting_domain.clone(),
            sumcheck_prover,
            folding_randomness,
            initial_evaluations,
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
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::{SeedableRng, rngs::SmallRng};

    use super::*;
    use crate::{
        dft::EvalsDft,
        fiat_shamir::domain_separator::DomainSeparator,
        parameters::{
            FoldingFactor, MultivariateParameters, ProtocolParameters, errors::SecurityAssumption,
        },
        poly::{coeffs::CoefficientList, evals::EvaluationsList},
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

    #[test]
    fn test_no_initial_statement_no_sumcheck() {
        // Number of variables in the multilinear polynomial
        let num_variables = 2;

        // Create a WHIR protocol config with:
        // - no initial sumcheck,
        // - folding factor 2,
        // - no PoW grinding.
        let config = make_test_config(num_variables, false, 2, 0);

        // Define a polynomial
        let poly = EvaluationsList::new(vec![F::from_u64(3); 1 << num_variables]);

        // Initialize:
        // - domain separator for Fiat-Shamir transcript,
        // - prover state,
        // - witness containing Merkle tree for `poly`.
        let (_, mut prover_state, witness) = setup_domain_and_commitment(&config, poly);

        // Create an empty public statement (no constraints)
        let statement = Statement::<EF4>::new(num_variables);

        // Initialize the round state using the setup configuration and witness
        let state = RoundState::initialize_first_round_state(
            &Prover(&config),
            &mut prover_state,
            statement,
            witness,
        )
        .unwrap();

        // Since there's no initial statement, the sumcheck should not be created
        assert!(state.sumcheck_prover.is_none());

        // Folding factor was 2, so we expect 2 sampled folding randomness values
        assert_eq!(state.folding_randomness.0.len(), 2);

        // Full randomness vector should be padded up to `num_variables`
        assert_eq!(state.randomness_vec.len(), num_variables);

        // Domain should match the starting parameters
        assert_eq!(
            state.domain,
            Domain::new(1 << num_variables, config.starting_log_inv_rate).unwrap()
        );

        // Since this is the first round, no Merkle data for folded rounds should exist
        assert!(state.merkle_prover_data.is_none());
    }

    #[test]
    fn test_initial_statement_with_folding_factor_3() {
        // Set the number of variables in the multilinear polynomial
        let num_variables = 3;

        // Create a WHIR configuration with:
        // - initial statement enabled (sumcheck will run),
        // - folding factor = 3 (fold all variables in the first round),
        // - PoW disabled.
        let config = make_test_config(num_variables, true, 3, 0);

        // Define the multilinear polynomial:
        // f(X0, X1, X2) = 1 + 2*X2 + 3*X1 + 4*X1*X2
        //              + 5*X0 + 6*X0*X2 + 7*X0*X1 + 8*X0*X1*X2
        let c1 = F::from_u64(1);
        let c2 = F::from_u64(2);
        let c3 = F::from_u64(3);
        let c4 = F::from_u64(4);
        let c5 = F::from_u64(5);
        let c6 = F::from_u64(6);
        let c7 = F::from_u64(7);
        let c8 = F::from_u64(8);

        let poly = CoefficientList::new(vec![c1, c2, c3, c4, c5, c6, c7, c8]).to_evaluations();

        // Manual redefinition of the same polynomial as a function for evaluation
        let f = |x0: EF4, x1: EF4, x2: EF4| {
            x2 * c2
                + x1 * c3
                + x1 * x2 * c4
                + x0 * c5
                + x0 * x2 * c6
                + x0 * x1 * c7
                + x0 * x1 * x2 * c8
                + c1
        };

        // Add a single equality constraint to the statement: f(1,1,1) = expected value
        let mut statement = Statement::<EF4>::new(num_variables);
        statement.add_constraint(
            Weights::evaluation(MultilinearPoint(vec![EF4::ONE, EF4::ONE, EF4::ONE])),
            f(EF4::ONE, EF4::ONE, EF4::ONE),
        );

        // Set up the domain separator, prover state, and witness for this configuration
        let (_, mut prover_state, witness) = setup_domain_and_commitment(&config, poly);

        // Run the first round state initialization (this will trigger sumcheck)
        let state = RoundState::initialize_first_round_state(
            &Prover(&config),
            &mut prover_state,
            statement,
            witness,
        )
        .unwrap();

        // Extract the constructed sumcheck prover and folding randomness
        let sumcheck = state.sumcheck_prover.as_ref().unwrap();
        let sumcheck_randomness = state.folding_randomness.clone();

        // With a folding factor of 3, all variables are collapsed in 1 round, so we expect only 1 evaluation left
        assert_eq!(sumcheck.evals.len(), 1);

        // The value of f at the folding point should match the evaluation
        let eval_at_point = sumcheck.evals[0];
        let expected = f(
            sumcheck_randomness.0[0],
            sumcheck_randomness.0[1],
            sumcheck_randomness.0[2],
        );
        assert_eq!(eval_at_point, expected);

        // Check that dot product of evaluations and weights matches the final sum
        let dot_product: EF4 = sumcheck
            .evals
            .iter()
            .zip(sumcheck.weights.evals())
            .map(|(f, w)| *f * *w)
            .sum();
        assert_eq!(dot_product, sumcheck.sum);

        // Verify that the `randomness_vec` (which is in reverse variable order) matches the expected layout
        assert_eq!(
            state.randomness_vec,
            vec![
                sumcheck_randomness.0[2],
                sumcheck_randomness.0[1],
                sumcheck_randomness.0[0]
            ]
        );

        // The `folding_randomness` should store values in forward order (X0, X1, X2)
        assert_eq!(
            state.folding_randomness.0,
            vec![
                sumcheck_randomness.0[0],
                sumcheck_randomness.0[1],
                sumcheck_randomness.0[2]
            ]
        );

        // Domain should match expected domain: 2^3 = 8 elements with inverse rate = 1
        assert_eq!(
            state.domain,
            Domain::new(1 << num_variables, config.starting_log_inv_rate).unwrap()
        );

        // No folded Merkle tree data should exist at this point
        assert!(state.merkle_prover_data.is_none());
    }

    #[test]
    fn test_zero_poly_multiple_constraints() {
        // Use a polynomial with 3 variables
        let num_variables = 3;

        // Build a WHIR config with an initial statement, folding factor 1, and no PoW
        let config = make_test_config(num_variables, true, 1, 0);

        // Define a zero polynomial: f(X) = 0 for all X
        let poly = EvaluationsList::new(vec![F::ZERO; 1 << num_variables]);

        // Generate domain separator, prover state, and Merkle commitment witness for the poly
        let (_, mut prover_state, witness) = setup_domain_and_commitment(&config, poly);

        // Create a new statement with multiple constraints
        let mut statement = Statement::<EF4>::new(num_variables);

        // Add one equality constraint per Boolean input: f(x) = 0 for all x ∈ {0,1}³
        for i in 0..1 << num_variables {
            let point = (0..num_variables)
                .map(|b| EF4::from_u64(((i >> b) & 1) as u64))
                .collect();
            statement.add_constraint(Weights::evaluation(MultilinearPoint(point)), EF4::ZERO);
        }

        // Initialize the first round of the WHIR protocol with the zero polynomial and constraints
        let state = RoundState::initialize_first_round_state(
            &Prover(&config),
            &mut prover_state,
            statement,
            witness,
        )
        .unwrap();

        // Extract the sumcheck prover and folding randomness
        let sumcheck = state.sumcheck_prover.as_ref().unwrap();
        let sumcheck_randomness = state.folding_randomness.0.clone();

        for (f, w) in sumcheck.evals.iter().zip(sumcheck.weights.evals()) {
            // Each evaluation should be 0
            assert_eq!(*f, EF4::ZERO);
            // Their contribution to the weighted sum should also be 0
            assert_eq!(*f * *w, EF4::ZERO);
        }
        // Final claimed sum is 0
        assert_eq!(sumcheck.sum, EF4::ZERO);

        // Folding randomness should have length equal to the folding factor (1)
        assert_eq!(sumcheck_randomness.len(), 1);

        // The `randomness_vec` is populated in reverse variable order, padded with 0s
        assert_eq!(
            state.randomness_vec,
            vec![sumcheck_randomness[0], EF4::ZERO, EF4::ZERO]
        );

        // Confirm that folding randomness matches exactly
        assert_eq!(
            state.folding_randomness,
            MultilinearPoint(vec![sumcheck_randomness[0]])
        );

        // Coefficients should match the original zero polynomial
        assert!(state.initial_evaluations.is_none());

        // Domain must match the WHIR config's expected size
        assert_eq!(
            state.domain,
            Domain::new(1 << num_variables, config.starting_log_inv_rate).unwrap()
        );

        // No Merkle commitment data for folded rounds yet
        assert!(state.merkle_prover_data.is_none());
    }

    #[test]
    fn test_initialize_round_state_with_initial_statement() {
        // Use a polynomial in 3 variables
        let num_variables = 3;

        // Set PoW grinding difficulty (used in Fiat-Shamir)
        let pow_bits = 4;

        // Build a WHIR configuration with:
        // - initial statement enabled,
        // - folding factor of 1 (fold one variable in the first round),
        // - PoW bits enabled.
        let config = make_test_config(num_variables, true, 1, pow_bits);

        // Define a multilinear polynomial:
        // f(X0, X1, X2) = 1 + 2*X2 + 3*X1 + 4*X1*X2 + 5*X0 + 6*X0*X2 + 7*X0*X1 + 8*X0*X1*X2
        let c1 = F::from_u64(1);
        let c2 = F::from_u64(2);
        let c3 = F::from_u64(3);
        let c4 = F::from_u64(4);
        let c5 = F::from_u64(5);
        let c6 = F::from_u64(6);
        let c7 = F::from_u64(7);
        let c8 = F::from_u64(8);
        let poly = CoefficientList::new(vec![c1, c2, c3, c4, c5, c6, c7, c8]).to_evaluations();

        // Equivalent function for evaluating the polynomial manually
        let f = |x0: EF4, x1: EF4, x2: EF4| {
            x2 * c2
                + x1 * c3
                + x1 * x2 * c4
                + x0 * c5
                + x0 * x2 * c6
                + x0 * x1 * c7
                + x0 * x1 * x2 * c8
                + c1
        };

        // Construct a statement with one evaluation constraint at the point (1, 0, 1)
        let mut statement = Statement::<EF4>::new(num_variables);
        statement.add_constraint(
            Weights::evaluation(MultilinearPoint(vec![EF4::ONE, EF4::ZERO, EF4::ONE])),
            f(EF4::ONE, EF4::ZERO, EF4::ONE),
        );

        // Set up Fiat-Shamir domain and produce commitment + witness
        let (_, mut prover_state, witness) = setup_domain_and_commitment(&config, poly);

        // Run the first round initialization
        let state = RoundState::initialize_first_round_state(
            &Prover(&config),
            &mut prover_state,
            statement,
            witness,
        )
        .expect("RoundState initialization failed");

        // Unwrap the sumcheck prover and get the sampled folding randomness
        let sumcheck = state.sumcheck_prover.unwrap();
        let sumcheck_randomness = &state.folding_randomness;

        // Evaluate f at (32636, 9876, r0) and match it with the sumcheck's recovered evaluation
        let evals_f = &sumcheck.evals;
        assert_eq!(
            evals_f.evaluate(&MultilinearPoint(vec![
                EF4::from_u64(32636),
                EF4::from_u64(9876)
            ])),
            f(
                EF4::from_u64(32636),
                EF4::from_u64(9876),
                sumcheck_randomness.0[0]
            )
        );

        // Manually verify that ⟨f, w⟩ = claimed sum
        let dot_product = evals_f.evals()[0] * sumcheck.weights.evals()[0]
            + evals_f.evals()[1] * sumcheck.weights.evals()[1]
            + evals_f.evals()[2] * sumcheck.weights.evals()[2]
            + evals_f.evals()[3] * sumcheck.weights.evals()[3];
        assert_eq!(dot_product, sumcheck.sum);

        // Evaluation storage must match original polynomial
        assert!(state.initial_evaluations.is_none());

        // Domain should match expected size and rate
        assert_eq!(
            state.domain,
            Domain::new(1 << num_variables, config.starting_log_inv_rate).unwrap()
        );

        // No Merkle tree data has been created for folded rounds yet
        assert!(state.merkle_prover_data.is_none());

        // The randomness_vec must contain the sampled folding randomness, reversed and zero-padded
        assert_eq!(
            state.randomness_vec,
            vec![sumcheck_randomness.0[0], EF4::ZERO, EF4::ZERO]
        );

        // The folding randomness must match what was sampled by the sumcheck
        assert_eq!(
            state.folding_randomness,
            MultilinearPoint(vec![sumcheck_randomness.0[0]])
        );
    }
}
