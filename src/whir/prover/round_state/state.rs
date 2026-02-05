//! Round state implementation for WHIR protocol.
//!
//! This module implements the core round state management for the WHIR protocol.

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::dense::DenseMatrix;
use tracing::instrument;

use crate::{
    fiat_shamir::errors::FiatShamirError,
    poly::multilinear::MultilinearPoint,
    sumcheck::sumcheck_prover::Sumcheck,
    whir::{
        committer::{ProverData, ProverDataView},
        constraints::statement::initial::InitialStatement,
        proof::SumcheckData,
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
    pub sumcheck_prover: Sumcheck<F, EF>,

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
    pub commitment_merkle_prover_data: ProverData<F, M, W, DIGEST_ELEMS>,

    /// Merkle tree commitment for extension field polynomials f': (EF)^{n-k} → EF.
    ///
    /// After the first round, polynomial folding moves operations into the extension
    /// field EF to accommodate the accumulated folding randomness α_1, ..., α_k ∈ EF.
    /// This optional commitment handles the folded polynomial evaluations.
    ///
    /// The extension field structure enables efficient constraint batching while
    /// preserving the Reed-Solomon proximity properties necessary for soundness.
    pub merkle_prover_data: Option<ProverDataView<F, EF, W, DIGEST_ELEMS>>,
}

#[allow(clippy::mismatching_type_param_order)]
impl<EF, F, W, const DIGEST_ELEMS: usize> RoundState<EF, F, W, DenseMatrix<F>, DIGEST_ELEMS>
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
    pub fn initialize_first_round_state<Challenger>(
        sumcheck_data: &mut SumcheckData<F, EF>,
        challenger: &mut Challenger,
        statement: &InitialStatement<F, EF>,
        prover_data: ProverData<F, DenseMatrix<F>, W, DIGEST_ELEMS>,
        folding_factor: usize,
        pow_bits: usize,
    ) -> Result<Self, FiatShamirError>
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        let (sumcheck_prover, folding_randomness) = Sumcheck::from_base_evals(
            sumcheck_data,
            challenger,
            folding_factor,
            pow_bits,
            statement,
        );

        // Initialize complete round state for first WHIR protocol round
        Ok(Self {
            // Sumcheck prover configured for constraint verification
            sumcheck_prover,
            // Current round's folding challenges (α_1, ..., α_k)
            folding_randomness,
            // Merkle commitment from witness for base field polynomial
            commitment_merkle_prover_data: prover_data,
            // No extension field commitment yet (first round operates in base field)
            merkle_prover_data: None,
        })
    }
}
