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

/// Evaluates polynomial answers using the appropriate method based on configuration.
///
/// This function encapsulates the logic for choosing between standard multilinear
/// evaluation and univariate skip evaluation based on the round configuration.
///
/// # Arguments
///
/// * `config` - The STIR configuration
/// * `round_index` - The current round index
/// * `answers` - The polynomial evaluation answers
/// * `round_state` - The current round state
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
    let should_skip = config.should_apply_univariate_skip(round_index);

    answers
        .iter()
        .map(|answer| {
            if should_skip {
                evaluate_with_univariate_skip(answer, round_state)
            } else {
                evaluate_standard_multilinear(answer, round_state)
            }
        })
        .collect()
}

/// Evaluates extension field polynomial answers.
///
/// This function handles the evaluation of polynomial answers that are already
/// in the extension field, using standard multilinear evaluation.
///
/// # Arguments
///
/// * `answers` - The polynomial evaluation answers in extension field
/// * `round_state` - The current round state
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
    answers
        .iter()
        .map(|answer| {
            EvaluationsList::new(answer.clone()).evaluate(&round_state.folding_randomness)
        })
        .collect()
}

/// Evaluates a polynomial using univariate skip optimization.
///
/// This function implements the complex two-stage evaluation process used
/// when the univariate skip optimization is applicable.
///
/// # Arguments
///
/// * `answer` - The polynomial evaluation values
/// * `round_state` - The current round state
///
/// # Returns
///
/// The evaluated result in the extension field
pub(crate) fn evaluate_with_univariate_skip<F, EF, const DIGEST_ELEMS: usize>(
    answer: &[F],
    round_state: &RoundState<EF, F, F, DenseMatrix<F>, DIGEST_ELEMS>,
) -> EF
where
    F: Field + TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
{
    let evals = EvaluationsList::new(answer.to_vec());
    let num_remaining_vars = evals.num_variables() - K_SKIP_SUMCHECK;
    let width = 1 << num_remaining_vars;
    let matrix = evals.into_mat(width);
    let challenges = &round_state.folding_randomness;

    // Extract skip challenge and remaining challenges
    let skip_challenge = *challenges
        .last_variable()
        .expect("skip challenge must be present");
    let remaining_challenges = challenges.get_subpoint_over_range(0..num_remaining_vars);

    // Two-stage evaluation: interpolate then evaluate
    let folded_row = interpolate_subgroup(&matrix, skip_challenge);
    EvaluationsList::new(folded_row).evaluate(&remaining_challenges)
}

/// Evaluates a polynomial using standard multilinear evaluation.
///
/// This is the standard evaluation method used when no optimizations apply.
///
/// # Arguments
///
/// * `answer` - The polynomial evaluation values
/// * `round_state` - The current round state
///
/// # Returns
///
/// The evaluated result in the extension field
pub(crate) fn evaluate_standard_multilinear<F, EF, const DIGEST_ELEMS: usize>(
    answer: &[F],
    round_state: &RoundState<EF, F, F, DenseMatrix<F>, DIGEST_ELEMS>,
) -> EF
where
    F: Field + TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
{
    EvaluationsList::new(answer.to_vec()).evaluate(&round_state.folding_randomness)
}

/// Hints base field opening results to the prover state.
///
/// This function provides the opened values and authentication paths to the
/// prover state for inclusion in the proof transcript.
///
/// # Arguments
///
/// * `openings` - The base field opening results
/// * `prover_state` - The prover state to hint to
pub(crate) fn hint_base_field_openings<F, EF, Challenger, const DIGEST_ELEMS: usize>(
    openings: &BaseFieldOpenings<F, DIGEST_ELEMS>,
    prover_state: &mut ProverState<F, EF, Challenger>,
) where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    // Hint the leaf values
    for answer in openings.answers() {
        prover_state.hint_base_scalars(answer);
    }

    // Hint the authentication paths
    hint_merkle_proofs(openings.proofs(), prover_state);
}

/// Hints extension field opening results to the prover state.
///
/// This function provides the opened values and authentication paths to the
/// prover state for inclusion in the proof transcript.
///
/// # Arguments
///
/// * `openings` - The extension field opening results
/// * `prover_state` - The prover state to hint to
pub(crate) fn hint_extension_field_openings<F, EF, Challenger, const DIGEST_ELEMS: usize>(
    openings: &ExtensionFieldOpenings<F, EF, DIGEST_ELEMS>,
    prover_state: &mut ProverState<F, EF, Challenger>,
) where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    // Hint the leaf values in extension field
    for answer in openings.answers() {
        prover_state.hint_extension_scalars(answer);
    }

    // Hint the authentication paths in base field
    hint_merkle_proofs(openings.proofs(), prover_state);
}

/// Hints Merkle proof data to the prover state.
///
/// This is a utility function that handles the common pattern of hinting
/// authentication path data to the prover state.
///
/// # Arguments
///
/// * `proofs` - Iterator over proof vectors
/// * `prover_state` - The prover state to hint to
pub(crate) fn hint_merkle_proofs<F, EF, Challenger, const DIGEST_ELEMS: usize>(
    proofs: impl IntoIterator<Item = impl AsRef<Vec<[F; DIGEST_ELEMS]>>>,
    prover_state: &mut ProverState<F, EF, Challenger>,
) where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    for proof in proofs {
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
        // These are integration points with the prover state
        // In a real implementation, we'd test the hint functions with mock prover states
        // For now, we test that the openings structures are created correctly

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
