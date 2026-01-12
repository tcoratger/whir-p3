//! Round state implementation for WHIR protocol.
//!
//! This module implements the core round state management for the WHIR protocol.

use alloc::{sync::Arc, vec::Vec};

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_dft::TwoAdicSubgroupDft;
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
    pub fn initialize_first_round_state<MyChallenger, C, Challenger, Dft>(
        dft: &Dft,
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
        Dft: TwoAdicSubgroupDft<F>,
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
                    dft,
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
            // Sumcheck prover configured for constraint verification
            sumcheck_prover,
            // Current round's folding challenges (α_1, ..., α_k)
            folding_randomness,
            // Merkle commitment from witness for base field polynomial
            commitment_merkle_prover_data: witness.prover_data,
            // No extension field commitment yet (first round operates in base field)
            merkle_prover_data: None,
            // Constraint set augmented with OOD evaluations
            // statement,
        })
    }
}
