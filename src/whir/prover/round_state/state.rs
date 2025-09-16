//! Round state implementation for WHIR protocol.
//!
//! This module implements the core round state management for the WHIR protocol.

use std::sync::Arc;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::dense::DenseMatrix;
use p3_merkle_tree::MerkleTree;
use tracing::instrument;

use crate::{
    constant::K_SKIP_SUMCHECK,
    fiat_shamir::{errors::FiatShamirError, prover::ProverState},
    poly::multilinear::MultilinearPoint,
    sumcheck::sumcheck_single::SumcheckSingle,
    whir::{
        committer::{RoundMerkleTree, Witness},
        constraints::statement::Statement,
        prover::Prover,
    },
};

/// Holds all per-round prover state required during the execution of the WHIR protocol.
///
/// This structure encapsulates the complete state needed for each round of the WHIR
/// interactive proof system.
#[derive(Debug)]
pub struct RoundState<EF, F, W, M, const DIGEST_ELEMS: usize>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
{
    /// The size of the current evaluation domain D.
    ///
    /// In WHIR, each round operates on a multiplicative subgroup H ⊆ F of size |H| = 2^m.
    ///
    /// This field tracks |H| for the current round. The domain shrinks by a factor of
    /// 2^k in each folding step, where k is the round's folding factor, ensuring
    /// exponential convergence: |H_0| → |H_1| → ... → |H_final| where |H_{i+1}| = |H_i|/2^k.
    ///
    /// This size determines the degree bound for polynomials in the current round and
    /// directly impacts both prover complexity O(|H| log |H|) and verifier query complexity.
    pub domain_size: usize,

    /// Generator ω for the next evaluation domain H_{i+1}.
    ///
    /// When folding from domain H_i with generator ω_i to smaller domain H_{i+1},
    /// the new generator is ω_{i+1} = ω_i^{2^k} where k is the folding factor.
    ///
    /// This ensures H_{i+1} = {1, ω_{i+1}, ω_{i+1}^2, ..., ω_{i+1}^{|H_{i+1}|-1}}
    /// remains a multiplicative subgroup with the correct size and structure for
    /// Reed-Solomon encoding in the subsequent round.
    pub next_domain_gen: F,

    /// Sumcheck prover managing constraint batching and polynomial evaluation.
    ///
    /// In WHIR, the sumcheck protocol enables efficient verification of constraint
    /// satisfaction over large domains. The sumcheck prover maintains the constraint
    /// polynomial S(X_0, ..., X_{n-1}) = Σ_j λ_j · C_j(X_0, ..., X_{n-1}) where
    /// {C_j} are individual constraints and {λ_j} are batching coefficients.
    ///
    /// Each round, this prover either:
    /// - Runs sumcheck rounds to prove S evaluates correctly over H^n
    /// - Updates constraint sets with new evaluation points from polynomial folding
    /// - Manages the transition from base field to extension field operations
    pub sumcheck_prover: SumcheckSingle<F, EF>,

    /// Folding randomness (α_1, α_2, ..., α_k) sampled for the current round.
    ///
    /// This vector contains the challenge values used in polynomial folding:
    /// f(X_0, ..., X_{n-1}) ↦ f'(X_k, ..., X_{n-1}) where
    /// f'(x_k, ..., x_{n-1}) := f(α_1, α_2, ..., α_k, x_k, ..., x_{n-1}).
    ///
    /// The randomness is sampled via Fiat-Shamir from the current transcript state,
    /// ensuring both parties derive identical challenge values while maintaining
    /// cryptographic soundness. The length equals the folding factor k for this round.
    pub folding_randomness: MultilinearPoint<EF>,

    /// Merkle tree commitment for the base field polynomial f: F^n → F.
    ///
    /// This commitment covers the initial polynomial evaluation table over the starting
    /// domain H_0. The Merkle tree enables selective opening of polynomial values at
    /// verifier-chosen query points while maintaining cryptographic integrity.
    ///
    /// In WHIR's proximity testing, this commitment proves the prover knows some
    /// polynomial that is purportedly close to a Reed-Solomon codeword. The verifier
    /// can later query specific positions to verify proximity claims.
    pub commitment_merkle_prover_data: Arc<MerkleTree<F, W, M, DIGEST_ELEMS>>,

    /// Merkle tree commitment for extension field polynomials f': (EF)^{n-k} → EF.
    ///
    /// After the first round, polynomial folding moves operations into the extension
    /// field EF to accommodate the accumulated folding randomness α_1, ..., α_k ∈ EF.
    /// This optional commitment handles the folded polynomial evaluations.
    ///
    /// The extension field structure enables efficient constraint batching while
    /// preserving the Reed-Solomon proximity properties necessary for soundness.
    pub merkle_prover_data: Option<RoundMerkleTree<F, EF, W, DIGEST_ELEMS>>,

    /// Global randomness vector tracking all folding challenges across rounds.
    ///
    /// This vector accumulates folding randomness in reverse variable order:
    /// position i stores the challenge for variable X_{n-1-i}. As rounds progress,
    /// new challenges are prepended, building the complete evaluation point needed
    /// for final constraint verification.
    ///
    /// The vector enables efficient polynomial evaluation at the accumulated
    /// challenge point while maintaining the proper variable ordering for
    /// multilinear extensions and sumcheck interactions.
    pub randomness_vec: Vec<EF>,

    /// Current constraint set defining the Reed-Solomon proximity testing problem.
    ///
    /// The statement contains linear equality constraints of the form
    /// f(r_j) = v_j for evaluation points r_j ∈ (EF)^n and target values v_j ∈ EF.
    /// These constraints evolve through rounds as:
    ///
    /// 1. **Initial**: Out-of-domain evaluations from the witness
    /// 2. **Per Round**: New constraints from sumcheck protocol interactions
    /// 3. **Evolution**: Constraint points updated with folding randomness
    ///
    /// The WHIR protocol maintains that satisfying these constraints implies
    /// the committed polynomial is close to some Reed-Solomon codeword.
    pub statement: Statement<EF>,
}

#[allow(clippy::mismatching_type_param_order)]
impl<EF, F, const DIGEST_ELEMS: usize> RoundState<EF, F, F, DenseMatrix<F>, DIGEST_ELEMS>
where
    F: TwoAdicField + Ord,
    EF: ExtensionField<F> + TwoAdicField,
{
    /// Initializes the first round state for the WHIR protocol.
    ///
    /// Prepares all state needed for the interactive proof by:
    /// 1. Adding out-of-domain constraints from the witness to the statement
    /// 2. Running sumcheck (if initial statement exists) or sampling folding randomness directly
    /// 3. Building the global randomness tracking vector
    /// 4. Setting up domain parameters for the first round
    ///
    /// The function handles two protocol variants:
    /// - **With initial statement**: Runs sumcheck protocol to batch constraints, then samples folding randomness
    /// - **Without initial statement**: Directly samples folding randomness and creates trivial sumcheck state
    ///
    /// Returns the complete `RoundState` ready for the first WHIR folding round.
    #[instrument(skip_all)]
    pub fn initialize_first_round_state<MyChallenger, C, Challenger>(
        prover: &Prover<'_, EF, F, MyChallenger, C, Challenger>,
        prover_state: &mut ProverState<F, EF, Challenger>,
        mut statement: Statement<EF>,
        witness: Witness<EF, F, DenseMatrix<F>, DIGEST_ELEMS>,
    ) -> Result<Self, FiatShamirError>
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
        MyChallenger: Clone,
        C: Clone,
    {
        // Convert witness OOD evaluations f(r_j) = v_j into linear constraints
        let new_constraint_points: Vec<_> = witness
            .ood_points
            .into_iter()
            .map(|point| {
                // Expand univariate OOD point to multilinear constraint in (EF)^n
                MultilinearPoint::expand_from_univariate(point, prover.num_variables)
            })
            .collect();

        // Prepend OOD constraints to statement for Reed-Solomon proximity testing
        statement.add_constraints_in_front(&new_constraint_points, &witness.ood_answers);

        // Protocol branching based on initial statement configuration
        let (sumcheck_prover, folding_randomness) = if prover.initial_statement {
            // Branch A: Initial statement exists - run sumcheck for constraint batching
            let combination_randomness_gen: EF = prover_state.sample();

            // Choose sumcheck strategy: with or without univariate skip optimization
            let (sumcheck, folding_randomness) =
                if prover.univariate_skip && K_SKIP_SUMCHECK <= prover.folding_factor.at_round(0) {
                    // Use univariate skip by skipping k variables
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
                    // Standard sumcheck protocol without optimization
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
            // Branch B: No initial statement - direct polynomial folding path
            let folding_randomness = MultilinearPoint::new(
                (0..prover.folding_factor.at_round(0))
                    .map(|_| prover_state.sample()) // Sample folding challenges α_1, ..., α_k
                    .collect::<Vec<_>>(),
            );

            // Apply folding transformation: f(X_0, ..., X_{n-1}) → f'(X_k, ..., X_{n-1})
            let poly = witness.polynomial.fold(&folding_randomness);
            let num_variables = poly.num_variables();

            // Create trivial sumcheck prover (no constraints to batch)
            let sumcheck = SumcheckSingle::from_extension_evals(
                poly,
                &Statement::initialize(num_variables),
                EF::ONE,
            );

            // Apply proof-of-work grinding for transcript security
            prover_state.pow_grinding(prover.starting_folding_pow_bits);

            (sumcheck, folding_randomness)
        };

        // Build global randomness accumulator for multi-round evaluation
        let mut randomness_vec = Vec::with_capacity(prover.num_variables);
        // Store challenges in reverse order: α_k, α_{k-1}, ..., α_1 (for variable X_{n-1-i})
        randomness_vec.extend(folding_randomness.iter().rev().copied());
        // Pad with zeros for variables not yet folded: X_{n-1}, X_{n-2}, ..., X_k
        randomness_vec.resize(prover.num_variables, EF::ZERO);

        // Initialize complete round state for first WHIR protocol round
        Ok(Self {
            // Starting domain H_0 with |H_0| = 2^m evaluation points
            domain_size: prover.starting_domain_size(),
            // Compute next domain generator: ω_1 = ω_0^{2^k} for H_1 after folding
            next_domain_gen: F::two_adic_generator(
                prover.starting_domain_size().ilog2() as usize - prover.folding_factor.at_round(0),
            ),
            // Sumcheck prover configured for constraint verification
            sumcheck_prover,
            // Current round's folding challenges (α_1, ..., α_k)
            folding_randomness,
            // Merkle commitment from witness for base field polynomial
            commitment_merkle_prover_data: witness.prover_data,
            // No extension field commitment yet (first round operates in base field)
            merkle_prover_data: None,
            // Global challenge vector for cross-round polynomial evaluation
            randomness_vec,
            // Constraint set augmented with OOD evaluations
            statement,
        })
    }
}
