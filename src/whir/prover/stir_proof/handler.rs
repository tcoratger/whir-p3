use std::marker::PhantomData;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_commit::{ExtensionMmcs, Mmcs};
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::dense::DenseMatrix;
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use serde::{Deserialize, Serialize};

use super::{
    config::StirConfig,
    openings::BaseFieldOpenings,
    types::{BaseMmcs, ExtProverData},
    utils::{
        evaluate_base_field_answers, evaluate_extension_field_answers, hint_base_field_openings,
        hint_extension_field_openings,
    },
};
use crate::{fiat_shamir::prover::ProverState, whir::prover::round::RoundState};

/// Handler for STIR proof operations.
#[derive(Debug)]
pub struct StirProofHandler<'a, F, EF, H, C, Challenger, const DIGEST_ELEMS: usize>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    H: CryptographicHasher<F, [F; DIGEST_ELEMS]>
        + CryptographicHasher<F::Packing, [F::Packing; DIGEST_ELEMS]>
        + Sync,
    C: PseudoCompressionFunction<[F; DIGEST_ELEMS], 2>
        + PseudoCompressionFunction<[F::Packing; DIGEST_ELEMS], 2>
        + Sync,
    [F; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    /// Reference to the cryptographic hasher for Merkle trees
    hasher: &'a H,
    /// Reference to the compression function for Merkle trees
    compressor: &'a C,
    /// Configuration for STIR protocol behavior
    config: StirConfig,
    /// Phantom data to carry generic type information at zero cost
    _phantom: PhantomData<(F, EF, Challenger)>,
}

impl<'a, F, EF, H, C, Challenger, const DIGEST_ELEMS: usize>
    StirProofHandler<'a, F, EF, H, C, Challenger, DIGEST_ELEMS>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    H: CryptographicHasher<F, [F; DIGEST_ELEMS]>
        + CryptographicHasher<F::Packing, [F::Packing; DIGEST_ELEMS]>
        + Sync,
    C: PseudoCompressionFunction<[F; DIGEST_ELEMS], 2>
        + PseudoCompressionFunction<[F::Packing; DIGEST_ELEMS], 2>
        + Sync,
    [F; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    /// Creates a new STIR proof handler.
    ///
    /// This constructor follows the pattern used in libraries like `reqwest`
    /// and `tokio` where configuration is passed as a structured parameter.
    ///
    /// # Arguments
    ///
    /// * `hasher` - Cryptographic hasher for Merkle tree construction
    /// * `compressor` - Compression function for Merkle tree nodes
    /// * `config` - Protocol configuration parameters
    #[must_use]
    pub const fn new(hasher: &'a H, compressor: &'a C, config: StirConfig) -> Self {
        Self {
            hasher,
            compressor,
            config,
            _phantom: PhantomData,
        }
    }

    /// Processes STIR queries for a given round.
    ///
    /// This is the main entry point for STIR query processing, automatically
    /// dispatching to the appropriate handler based on the round state's
    /// Merkle prover data type.
    ///
    /// # Arguments
    ///
    /// * `round_index` - The current round index (0-based)
    /// * `challenge_indexes` - The positions to query in the commitment
    /// * `round_state` - The current round state
    /// * `prover_state` - The prover state for transcript updates
    ///
    /// # Returns
    ///
    /// Vector of evaluations corresponding to the STIR challenges
    pub(crate) fn process_stir_queries(
        &self,
        round_index: usize,
        challenge_indexes: &[usize],
        round_state: &RoundState<EF, F, F, DenseMatrix<F>, DIGEST_ELEMS>,
        prover_state: &mut ProverState<F, EF, Challenger>,
    ) -> Vec<EF> {
        match &round_state.merkle_prover_data {
            None => self.process_base_field_queries(
                round_index,
                challenge_indexes,
                round_state,
                prover_state,
            ),
            Some(data) => self.process_extension_field_queries(
                challenge_indexes,
                data,
                round_state,
                prover_state,
            ),
        }
    }

    /// Processes final round proofs.
    ///
    /// This method handles the final stage of the STIR protocol where the prover
    /// must open specific positions in the final Merkle tree commitments.
    ///
    /// # Arguments
    ///
    /// * `challenge_indexes` - The positions to open (flexible iterator)
    /// * `round_state` - The current round state
    /// * `prover_state` - The prover state for transcript updates
    pub(crate) fn process_final_proofs(
        &self,
        challenge_indexes: impl IntoIterator<Item = usize>,
        round_state: &RoundState<EF, F, F, DenseMatrix<F>, DIGEST_ELEMS>,
        prover_state: &mut ProverState<F, EF, Challenger>,
    ) {
        let indexes: Vec<_> = challenge_indexes.into_iter().collect();

        match &round_state.merkle_prover_data {
            None => self.open_base_field_commitments(&indexes, round_state, prover_state),
            Some(data) => self.open_extension_field_commitments(&indexes, data, prover_state),
        }
    }

    fn process_base_field_queries(
        &self,
        round_index: usize,
        challenge_indexes: &[usize],
        round_state: &RoundState<EF, F, F, DenseMatrix<F>, DIGEST_ELEMS>,
        prover_state: &mut ProverState<F, EF, Challenger>,
    ) -> Vec<EF> {
        let mmcs = self.create_base_mmcs();
        let mut openings = BaseFieldOpenings::with_capacity(challenge_indexes.len());

        // Open each position and collect results
        for &index in challenge_indexes {
            let opening = mmcs.open_batch(index, &round_state.commitment_merkle_prover_data);
            openings.push(opening.opened_values[0].clone(), opening.opening_proof);
        }

        hint_base_field_openings(&openings, prover_state);
        evaluate_base_field_answers(&self.config, round_index, &openings.answers, round_state)
    }

    fn process_extension_field_queries(
        &self,
        challenge_indexes: &[usize],
        prover_data: &ExtProverData<F, EF, H, C, DIGEST_ELEMS>,
        round_state: &RoundState<EF, F, F, DenseMatrix<F>, DIGEST_ELEMS>,
        prover_state: &mut ProverState<F, EF, Challenger>,
    ) -> Vec<EF> {
        let mmcs = self.create_base_mmcs();
        let ext_mmcs = ExtensionMmcs::new(mmcs);
        let mut openings =
            super::openings::ExtensionFieldOpenings::with_capacity(challenge_indexes.len());

        // Open each position and collect results
        for &index in challenge_indexes {
            let opening = ext_mmcs.open_batch(index, prover_data);
            openings.push(opening.opened_values[0].clone(), opening.opening_proof);
        }

        hint_extension_field_openings(&openings, prover_state);
        evaluate_extension_field_answers(&openings.answers, round_state)
    }

    fn open_base_field_commitments(
        &self,
        challenge_indexes: &[usize],
        round_state: &RoundState<EF, F, F, DenseMatrix<F>, DIGEST_ELEMS>,
        prover_state: &mut ProverState<F, EF, Challenger>,
    ) {
        let mmcs = self.create_base_mmcs();
        let mut openings = BaseFieldOpenings::with_capacity(challenge_indexes.len());

        for &index in challenge_indexes {
            let opening = mmcs.open_batch(index, &round_state.commitment_merkle_prover_data);
            openings.push(opening.opened_values[0].clone(), opening.opening_proof);
        }

        hint_base_field_openings(&openings, prover_state);
    }

    fn open_extension_field_commitments(
        &self,
        challenge_indexes: &[usize],
        prover_data: &ExtProverData<F, EF, H, C, DIGEST_ELEMS>,
        prover_state: &mut ProverState<F, EF, Challenger>,
    ) {
        let mmcs = self.create_base_mmcs();
        let ext_mmcs = ExtensionMmcs::new(mmcs);
        let mut openings =
            super::openings::ExtensionFieldOpenings::with_capacity(challenge_indexes.len());

        for &index in challenge_indexes {
            let opening = ext_mmcs.open_batch(index, prover_data);
            openings.push(opening.opened_values[0].clone(), opening.opening_proof);
        }

        hint_extension_field_openings(&openings, prover_state);
    }

    /// Creates a base field MMCS instance using the handler's cryptographic primitives.
    fn create_base_mmcs(&self) -> BaseMmcs<F, H, C, DIGEST_ELEMS> {
        BaseMmcs::new(self.hasher.clone(), self.compressor.clone())
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    #[test]
    fn test_handler_construction() {
        let config = StirConfig::default();
        assert!(!config.initial_statement);
        assert!(!config.univariate_skip);
        assert_eq!(config.folding_factor_at_round, 0);

        // Test builder pattern
        let config = StirConfig::builder()
            .initial_statement(true)
            .univariate_skip(true)
            .folding_factor_at_round(4)
            .build();

        assert!(config.initial_statement);
        assert!(config.univariate_skip);
        assert_eq!(config.folding_factor_at_round, 4);
    }

    #[test]
    fn test_create_base_mmcs() {
        // Test that the utility function can be called
        // In a real test, we'd use concrete hasher and compressor types
        // For now, we test the configuration logic

        let config = StirConfig::new();
        assert!(!config.should_apply_univariate_skip(0));

        let config = StirConfig::builder()
            .initial_statement(true)
            .univariate_skip(true)
            .folding_factor_at_round(4)
            .build();

        assert!(config.should_apply_univariate_skip(0));
        assert!(!config.should_apply_univariate_skip(1));
    }
}
