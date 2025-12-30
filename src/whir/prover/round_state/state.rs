//! Round state implementation for WHIR protocol.
//!
//! This module implements the core round state management for the WHIR protocol.

use alloc::{sync::Arc, vec, vec::Vec};

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::dense::DenseMatrix;
use p3_merkle_tree::MerkleTree;
use tracing::instrument;

use crate::{
    constant::K_SKIP_SUMCHECK,
    fiat_shamir::errors::FiatShamirError,
    poly::multilinear::MultilinearPoint,
    sumcheck::sumcheck_single::SumcheckSingle,
    whir::{
        committer::{RoundMerkleTree, Witness},
        constraints::{Constraint, statement::EqStatement},
        proof::{InitialPhase, SumcheckSkipData, WhirProof},
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

    /// Merkle tree commitments for base field polynomials f: F^n → F.
    ///
    /// This vector contains commitments for one or more polynomials. For single-proof
    /// scenarios, this contains exactly one tree. For batch opening, it contains
    /// multiple trees (one per polynomial being batch-opened).
    ///
    /// The trees enable selective opening of polynomial values at
    /// verifier-chosen query points while maintaining cryptographic integrity.
    ///
    /// In WHIR's proximity testing, these commitments prove the prover knows
    /// polynomials that are purportedly close to Reed-Solomon codewords. The verifier
    /// can later query specific positions to verify proximity claims.
    ///
    /// Using `Vec<Arc<...>>` allows sharing individual trees independently,
    /// which is useful when passing single trees to functions without cloning
    /// the entire collection.
    pub commitment_merkle_prover_data: Vec<Arc<MerkleTree<F, W, M, DIGEST_ELEMS>>>,

    /// Merkle tree commitment for extension field polynomials f': (EF)^{n-k} → EF.
    ///
    /// After the first round, polynomial folding moves operations into the extension
    /// field EF to accommodate the accumulated folding randomness α_1, ..., α_k ∈ EF.
    /// This optional commitment handles the folded polynomial evaluations.
    ///
    /// The extension field structure enables efficient constraint batching while
    /// preserving the Reed-Solomon proximity properties necessary for soundness.
    pub merkle_prover_data: Option<RoundMerkleTree<F, EF, W, DIGEST_ELEMS>>,

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
    pub statement: EqStatement<EF>,
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
        proof: &mut WhirProof<F, EF, DIGEST_ELEMS>,
        challenger: &mut Challenger,
        mut statement: EqStatement<EF>,
        witness: Witness<EF, F, DenseMatrix<F>, DIGEST_ELEMS>,
    ) -> Result<Self, FiatShamirError>
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
        MyChallenger: Clone,
        C: Clone,
    {
        // Append OOD constraints to statement for Reed-Solomon proximity testing
        statement.concatenate(&witness.ood_statement);

        // Protocol branching based on initial phase variant in proof
        let (sumcheck_prover, folding_randomness) = match &mut proof.initial_phase {
            // Branch: WithStatementSkip - use univariate skip optimization
            InitialPhase::WithStatementSkip(skip_data)
                if K_SKIP_SUMCHECK <= prover.folding_factor.at_round(0) =>
            {
                // Build constraint with random linear combination
                let constraint =
                    Constraint::new_eq_only(challenger.sample_algebra_element(), statement.clone());

                // Use univariate skip by skipping k variables
                SumcheckSingle::with_skip(
                    &witness.polynomial,
                    skip_data,
                    challenger,
                    prover.folding_factor.at_round(0),
                    prover.starting_folding_pow_bits,
                    K_SKIP_SUMCHECK,
                    &constraint,
                )
            }

            // Branch: WithStatementSvo - SVO optimization
            InitialPhase::WithStatementSvo { sumcheck } => {
                // SVO optimization requirements (see Procedure 9 in https://eprint.iacr.org/2025/1117):
                // 1. At least 2 * NUM_SVO_ROUNDS variables - The SVO algorithm partitions
                //    variables into Prefix (k), Inner (l/2), and Outer segments. For these
                //    segments not to overlap, we need k + l/2 <= l, which gives l >= 2k.
                // 2. Exactly one equality constraint (SVO algorithm assumes single point)
                //
                // TODO: The single constraint requirement is a current limitation.
                // This approach should be generalized to handle multiple constraints.
                // See: https://hackmd.io/@tcoratger/H1SNENAeZg for details.
                const MIN_SVO_FOLDING_FACTOR: usize = 6;

                // Build constraint with random linear combination
                let constraint =
                    Constraint::new_eq_only(challenger.sample_algebra_element(), statement.clone());

                let folding_factor = prover.folding_factor.at_round(0);
                let has_single_constraint = constraint.eq_statement.len() == 1;

                if folding_factor >= MIN_SVO_FOLDING_FACTOR && has_single_constraint {
                    // Use SVO optimization: first 3 rounds use specialized algorithm,
                    // remaining rounds use standard Algorithm 5
                    SumcheckSingle::from_base_evals_svo(
                        &witness.polynomial,
                        sumcheck,
                        challenger,
                        folding_factor,
                        prover.starting_folding_pow_bits,
                        &constraint,
                    )
                } else {
                    // Fall back to classic sumcheck when:
                    // - Input is too small (folding_factor < MIN_SVO_FOLDING_FACTOR)
                    // - Multiple constraints exist (SVO only handles single constraint, see TODO above)
                    SumcheckSingle::from_base_evals(
                        &witness.polynomial,
                        sumcheck,
                        challenger,
                        folding_factor,
                        prover.starting_folding_pow_bits,
                        &constraint,
                    )
                }
            }

            // Branch: WithStatement or WithStatementSkip (fallback when folding_factor < K_SKIP)
            InitialPhase::WithStatement { sumcheck }
            | InitialPhase::WithStatementSkip(SumcheckSkipData { sumcheck, .. }) => {
                // Build constraint with random linear combination
                let constraint =
                    Constraint::new_eq_only(challenger.sample_algebra_element(), statement.clone());

                // Standard sumcheck protocol without optimization
                SumcheckSingle::from_base_evals(
                    &witness.polynomial,
                    sumcheck,
                    challenger,
                    prover.folding_factor.at_round(0),
                    prover.starting_folding_pow_bits,
                    &constraint,
                )
            }

            // Branch: WithoutStatement - direct polynomial folding path
            InitialPhase::WithoutStatement { pow_witness } => {
                // Sample folding challenges α_1, ..., α_k
                let folding_randomness = MultilinearPoint::new(
                    (0..prover.folding_factor.at_round(0))
                        .map(|_| challenger.sample_algebra_element())
                        .collect::<Vec<_>>(),
                );

                // Apply folding transformation: f(X_0, ..., X_{n-1}) → f'(X_k, ..., X_{n-1})
                let poly = witness.polynomial.fold(&folding_randomness);
                let num_variables = poly.num_variables();

                // Create trivial sumcheck prover (no constraints to batch)
                let sumcheck = SumcheckSingle::from_extension_evals(
                    poly,
                    EqStatement::initialize(num_variables),
                    EF::ONE,
                );

                // Apply proof-of-work grinding and store witness (only if pow_bits > 0)
                if prover.starting_folding_pow_bits > 0 {
                    *pow_witness = challenger.grind(prover.starting_folding_pow_bits);
                }

                (sumcheck, folding_randomness)
            }
        };

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
            // Merkle commitment from witness for base field polynomial (single-element vector)
            commitment_merkle_prover_data: vec![witness.prover_data],
            // No extension field commitment yet (first round operates in base field)
            merkle_prover_data: None,
            // Constraint set augmented with OOD evaluations
            statement,
        })
    }
}
