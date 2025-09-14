use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_interpolation::interpolate_subgroup;
use p3_matrix::dense::DenseMatrix;

use super::{
    StirConfig,
    openings::{BaseFieldOpenings, ExtensionFieldOpenings},
};
use crate::{
    constant::K_SKIP_SUMCHECK, fiat_shamir::prover::ProverState, poly::evals::EvaluationsList,
    whir::prover::round::RoundState,
};

/// Evaluates base field polynomial answers using configuration-dependent methods.
///
/// Chooses between standard multilinear evaluation and univariate skip evaluation
/// based on STIR protocol configuration and round conditions.
///
/// # Arguments
///
/// * `config` - STIR protocol configuration parameters
/// * `round_index` - Current round index for optimization decisions
/// * `answers` - Base field polynomial evaluation answers to process
/// * `round_state` - Current round state containing folding randomness
///
/// # Returns
///
/// Vector of evaluated results in the extension field
pub(crate) fn evaluate_base_field_answers<F, EF, const DIGEST_ELEMS: usize>(
    config: &StirConfig,
    round_index: usize,
    answers: &[Vec<F>],
    round_state: &RoundState<EF, F, F, DenseMatrix<F>, DIGEST_ELEMS>,
) -> Vec<EF>
where
    F: Field + TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
{
    // Determine if univariate skip optimization applies for this round
    let should_skip = config.should_apply_univariate_skip(round_index);

    // Process each answer using the appropriate evaluation method
    answers
        .iter()
        .map(|answer| {
            if should_skip {
                // Use evaluation for univariate skip
                evaluate_with_univariate_skip(answer, round_state)
            } else {
                // Use standard multilinear evaluation
                evaluate_standard_multilinear(answer, round_state)
            }
        })
        .collect()
}

/// Evaluates extension field polynomial answers using standard multilinear evaluation.
///
/// Processes polynomial answers that are already in the extension field.
/// Always uses standard multilinear evaluation since extension field operations
/// don't require univariate skip optimization handling.
///
/// # Arguments
///
/// * `answers` - Extension field polynomial evaluation answers to process
/// * `round_state` - Current round state containing folding randomness
///
/// # Returns
///
/// Vector of evaluated results in the extension field
pub(crate) fn evaluate_extension_field_answers<F, EF, const DIGEST_ELEMS: usize>(
    answers: &[Vec<EF>],
    round_state: &RoundState<EF, F, F, DenseMatrix<F>, DIGEST_ELEMS>,
) -> Vec<EF>
where
    F: Field + TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
{
    // Process each extension field answer using standard multilinear evaluation
    answers
        .iter()
        .map(|answer| {
            // Create evaluation list and evaluate with folding randomness
            EvaluationsList::new(answer.clone()).evaluate(&round_state.folding_randomness)
        })
        .collect()
}

/// Evaluates a polynomial using univariate skip optimization.
///
/// Implements the sophisticated two-stage evaluation process that enables
/// skipping sumcheck variables for improved verifier efficiency.
///
/// # Arguments
///
/// * `answer` - Base field polynomial evaluation values to process
/// * `round_state` - Current round state containing folding randomness
///
/// # Returns
///
/// Evaluated result in the extension field after univariate skip processing
pub(crate) fn evaluate_with_univariate_skip<F, EF, const DIGEST_ELEMS: usize>(
    answer: &[F],
    round_state: &RoundState<EF, F, F, DenseMatrix<F>, DIGEST_ELEMS>,
) -> EF
where
    F: Field + TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
{
    // Create evaluation list from polynomial answers
    let evals = EvaluationsList::new(answer.to_vec());
    // Calculate remaining variables after skipping
    let num_remaining_vars = evals.num_variables() - K_SKIP_SUMCHECK;
    // Determine matrix width for restructuring
    let width = 1 << num_remaining_vars;
    // Convert to matrix form for interpolation
    let matrix = evals.into_mat(width);
    let challenges = &round_state.folding_randomness;

    // Extract the skip challenge from the last variable
    let skip_challenge = *challenges
        .last_variable()
        .expect("skip challenge must be present");
    // Get the remaining challenges for final evaluation
    let remaining_challenges = challenges.get_subpoint_over_range(0..num_remaining_vars);

    // Stage 1: Interpolate over the subgroup using skip challenge
    let folded_row = interpolate_subgroup(&matrix, skip_challenge);
    // Stage 2: Evaluate the folded polynomial with remaining challenges
    EvaluationsList::new(folded_row).evaluate(&remaining_challenges)
}

/// Evaluates a polynomial using standard multilinear evaluation.
///
/// Standard evaluation method used when univariate skip optimization
/// is not applicable or enabled. Performs direct multilinear evaluation
/// with all folding randomness values.
///
/// # Arguments
///
/// * `answer` - Base field polynomial evaluation values to process
/// * `round_state` - Current round state containing folding randomness
///
/// # Returns
///
/// Evaluated result in the extension field using standard method
pub(crate) fn evaluate_standard_multilinear<F, EF, const DIGEST_ELEMS: usize>(
    answer: &[F],
    round_state: &RoundState<EF, F, F, DenseMatrix<F>, DIGEST_ELEMS>,
) -> EF
where
    F: Field + TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
{
    // Direct multilinear evaluation with all folding randomness
    EvaluationsList::new(answer.to_vec()).evaluate(&round_state.folding_randomness)
}

/// Hints base field opening results to the prover state for transcript inclusion.
///
/// Provides polynomial evaluation values and authentication paths to the
/// prover state, enabling the verifier to reconstruct and verify the
/// STIR proof transcript during verification.
///
/// # Arguments
///
/// * `openings` - Base field opening results containing answers and proofs
/// * `prover_state` - Prover state to receive the hinted data
pub(crate) fn hint_base_field_openings<F, EF, Challenger, const DIGEST_ELEMS: usize>(
    openings: &BaseFieldOpenings<F, DIGEST_ELEMS>,
    prover_state: &mut ProverState<F, EF, Challenger>,
) where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    // Hint all polynomial evaluation values to the prover state
    for answer in openings.answers() {
        prover_state.hint_base_scalars(answer);
    }

    // Hint all Merkle authentication paths to the prover state
    hint_merkle_proofs(openings.proofs(), prover_state);
}

/// Hints extension field opening results to the prover state for transcript inclusion.
///
/// Provides extension field polynomial evaluation values and base field
/// authentication paths to the prover state, enabling verification
/// of STIR operations requiring extension field computations.
///
/// # Arguments
///
/// * `openings` - Extension field opening results containing answers and proofs
/// * `prover_state` - Prover state to receive the hinted data
pub(crate) fn hint_extension_field_openings<F, EF, Challenger, const DIGEST_ELEMS: usize>(
    openings: &ExtensionFieldOpenings<F, EF, DIGEST_ELEMS>,
    prover_state: &mut ProverState<F, EF, Challenger>,
) where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    // Hint all extension field polynomial evaluation values
    for answer in openings.answers() {
        prover_state.hint_extension_scalars(answer);
    }

    // Hint all base field Merkle authentication paths
    hint_merkle_proofs(openings.proofs(), prover_state);
}

/// Hints Merkle proof data to the prover state for authentication verification.
///
/// Utility function handling the common pattern of hinting Merkle tree
/// authentication path data to the prover state. Processes digest arrays
/// from proof vectors for transcript inclusion.
///
/// # Arguments
///
/// * `proofs` - Iterator over proof vectors containing authentication paths
/// * `prover_state` - Prover state to receive the authentication data
pub(crate) fn hint_merkle_proofs<F, EF, Challenger, const DIGEST_ELEMS: usize>(
    proofs: impl IntoIterator<Item = impl AsRef<Vec<[F; DIGEST_ELEMS]>>>,
    prover_state: &mut ProverState<F, EF, Challenger>,
) where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    // Process each proof vector in the collection
    for proof in proofs {
        // Hint each digest in the authentication path
        for digest in proof.as_ref() {
            prover_state.hint_base_scalars(digest);
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    #[test]
    fn test_config_skip_logic() {
        let config = StirConfig {
            initial_statement: true,
            univariate_skip: true,
            folding_factor_at_round: K_SKIP_SUMCHECK,
        };

        assert!(config.should_apply_univariate_skip(0));
        assert!(!config.should_apply_univariate_skip(1));

        let config_no_skip = StirConfig {
            initial_statement: false,
            univariate_skip: true,
            folding_factor_at_round: K_SKIP_SUMCHECK,
        };

        assert!(!config_no_skip.should_apply_univariate_skip(0));
    }

    #[test]
    fn test_hint_functions() {
        let mut base_openings: BaseFieldOpenings<F, 4> = BaseFieldOpenings::with_capacity(1);
        base_openings.push(vec![F::ONE], vec![[F::ZERO; 4]]);

        assert_eq!(base_openings.len(), 1);
        assert_eq!(base_openings.answers().count(), 1);
        assert_eq!(base_openings.proofs().count(), 1);

        let mut ext_openings: ExtensionFieldOpenings<F, EF, 4> =
            ExtensionFieldOpenings::with_capacity(1);
        ext_openings.push(vec![EF::ONE], vec![[F::ZERO; 4]]);

        assert_eq!(ext_openings.len(), 1);
        assert_eq!(ext_openings.answers().count(), 1);
        assert_eq!(ext_openings.proofs().count(), 1);
    }
}
