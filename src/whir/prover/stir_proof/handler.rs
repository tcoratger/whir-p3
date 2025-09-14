use std::marker::PhantomData;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_commit::{ExtensionMmcs, Mmcs};
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::dense::DenseMatrix;
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use serde::{Deserialize, Serialize};

use super::{
    ExtensionFieldOpenings,
    config::StirConfig,
    openings::BaseFieldOpenings,
    types::{BaseMmcs, ExtProverData},
    utils::{
        evaluate_base_field_answers, evaluate_extension_field_answers, hint_base_field_openings,
        hint_extension_field_openings,
    },
};
use crate::{fiat_shamir::prover::ProverState, whir::prover::round_state::RoundState};

/// STIR proof operations handler for Reed-Solomon proximity testing.
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
    /// Cryptographic hasher for Merkle tree construction and verification
    hasher: &'a H,
    /// Compression function for Merkle tree internal node computation
    compressor: &'a C,
    /// STIR protocol configuration controlling optimization behavior
    config: StirConfig,
    /// Phantom data carrying generic type information at zero runtime cost
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
    /// Creates a new STIR proof handler with cryptographic primitives and configuration.
    ///
    /// Initializes the handler with the necessary cryptographic components
    /// for Merkle tree operations and STIR protocol configuration parameters.
    ///
    /// # Arguments
    ///
    /// * `hasher` - Cryptographic hasher for Merkle tree construction
    /// * `compressor` - Compression function for Merkle tree nodes
    /// * `config` - STIR protocol configuration parameters
    #[must_use]
    pub const fn new(hasher: &'a H, compressor: &'a C, config: StirConfig) -> Self {
        // Initialize handler with provided cryptographic primitives and config
        Self {
            hasher,
            compressor,
            config,
            _phantom: PhantomData,
        }
    }

    /// Processes STIR queries for a given round with automatic field type dispatch.
    ///
    /// Main entry point for STIR query processing that automatically selects
    /// between base field and extension field operations based on the round
    /// state's Merkle prover data configuration.
    ///
    /// # Arguments
    ///
    /// * `round_index` - Current round index for optimization decisions
    /// * `challenge_indexes` - Positions to query in the commitment
    /// * `round_state` - Current round state containing commitment data
    /// * `prover_state` - Prover state for transcript updates
    ///
    /// # Returns
    ///
    /// Vector of evaluations corresponding to the STIR challenge positions
    pub(crate) fn process_stir_queries(
        &self,
        round_index: usize,
        challenge_indexes: &[usize],
        round_state: &RoundState<EF, F, F, DenseMatrix<F>, DIGEST_ELEMS>,
        prover_state: &mut ProverState<F, EF, Challenger>,
    ) -> Vec<EF> {
        // Dispatch based on the presence of extension field Merkle prover data
        match &round_state.merkle_prover_data {
            // Base field operations when no extension field data present
            None => self.process_base_field_queries(
                round_index,
                challenge_indexes,
                round_state,
                prover_state,
            ),
            // Extension field operations when extension data is available
            Some(data) => self.process_extension_field_queries(
                challenge_indexes,
                data,
                round_state,
                prover_state,
            ),
        }
    }

    /// Processes final round proofs for STIR queries.
    ///
    /// Handles the final stage of the STIR protocol where the prover must
    /// open specific positions in Merkle tree commitments. Automatically
    /// dispatches between base field and extension field commitment opening.
    ///
    /// # Arguments
    ///
    /// * `challenge_indexes` - Positions to open in the final commitment
    /// * `round_state` - Current round state containing commitment data
    /// * `prover_state` - Prover state for transcript updates
    pub(crate) fn process_final_proofs(
        &self,
        challenge_indexes: impl IntoIterator<Item = usize>,
        round_state: &RoundState<EF, F, F, DenseMatrix<F>, DIGEST_ELEMS>,
        prover_state: &mut ProverState<F, EF, Challenger>,
    ) {
        // Collect challenge indexes into a vector for processing
        let indexes: Vec<_> = challenge_indexes.into_iter().collect();

        // Dispatch final proof opening based on commitment type
        match &round_state.merkle_prover_data {
            // Open base field commitments when no extension data present
            None => self.open_base_field_commitments(&indexes, round_state, prover_state),
            // Open extension field commitments when extension data available
            Some(data) => self.open_extension_field_commitments(&indexes, data, prover_state),
        }
    }

    /// Processes STIR queries using base field operations.
    fn process_base_field_queries(
        &self,
        round_index: usize,
        challenge_indexes: &[usize],
        round_state: &RoundState<EF, F, F, DenseMatrix<F>, DIGEST_ELEMS>,
        prover_state: &mut ProverState<F, EF, Challenger>,
    ) -> Vec<EF> {
        // Create base field MMCS for Merkle tree operations
        let mmcs = self.create_base_mmcs();
        // Pre-allocate openings collection for efficiency
        let mut openings = BaseFieldOpenings::with_capacity(challenge_indexes.len());

        // Process each challenge position by opening the commitment
        for &index in challenge_indexes {
            let opening = mmcs.open_batch(index, &round_state.commitment_merkle_prover_data);
            // Store the opened values and authentication proof
            openings.push(opening.opened_values[0].clone(), opening.opening_proof);
        }

        // Hint opening data to prover state for transcript inclusion
        hint_base_field_openings(&openings, prover_state);
        // Evaluate answers using appropriate method based on configuration
        evaluate_base_field_answers(&self.config, round_index, &openings.answers, round_state)
    }

    /// Processes STIR queries using extension field operations.
    fn process_extension_field_queries(
        &self,
        challenge_indexes: &[usize],
        prover_data: &ExtProverData<F, EF, H, C, DIGEST_ELEMS>,
        round_state: &RoundState<EF, F, F, DenseMatrix<F>, DIGEST_ELEMS>,
        prover_state: &mut ProverState<F, EF, Challenger>,
    ) -> Vec<EF> {
        // Create base MMCS and wrap with extension field functionality
        let mmcs = self.create_base_mmcs();
        let ext_mmcs = ExtensionMmcs::new(mmcs);
        // Pre-allocate extension field openings collection
        let mut openings = ExtensionFieldOpenings::with_capacity(challenge_indexes.len());

        // Process each challenge position using extension field operations
        for &index in challenge_indexes {
            let opening = ext_mmcs.open_batch(index, prover_data);
            // Store extension field values with base field authentication proof
            openings.push(opening.opened_values[0].clone(), opening.opening_proof);
        }

        // Hint extension field opening data to prover state
        hint_extension_field_openings(&openings, prover_state);
        // Evaluate extension field answers using standard multilinear method
        evaluate_extension_field_answers(&openings.answers, round_state)
    }

    /// Opens base field commitments for final proof generation.
    ///
    /// Performs final commitment openings for base field data.
    ///
    /// Only hints the opening data to the prover state without performing evaluations.
    fn open_base_field_commitments(
        &self,
        challenge_indexes: &[usize],
        round_state: &RoundState<EF, F, F, DenseMatrix<F>, DIGEST_ELEMS>,
        prover_state: &mut ProverState<F, EF, Challenger>,
    ) {
        // Create base field MMCS for final openings
        let mmcs = self.create_base_mmcs();
        // Pre-allocate openings collection for final proof data
        let mut openings = BaseFieldOpenings::with_capacity(challenge_indexes.len());

        // Open each final challenge position
        for &index in challenge_indexes {
            let opening = mmcs.open_batch(index, &round_state.commitment_merkle_prover_data);
            // Store final opened values and proofs
            openings.push(opening.opened_values[0].clone(), opening.opening_proof);
        }

        // Hint final base field opening data to prover state
        hint_base_field_openings(&openings, prover_state);
    }

    /// Opens extension field commitments for final proof generation.
    ///
    /// Performs final commitment openings for extension field data.
    ///
    /// Only hints the opening data to the prover state without performing evaluations.
    fn open_extension_field_commitments(
        &self,
        challenge_indexes: &[usize],
        prover_data: &ExtProverData<F, EF, H, C, DIGEST_ELEMS>,
        prover_state: &mut ProverState<F, EF, Challenger>,
    ) {
        // Create extension field MMCS for final openings
        let mmcs = self.create_base_mmcs();
        let ext_mmcs = ExtensionMmcs::new(mmcs);
        // Pre-allocate extension field openings collection for final proof
        let mut openings = ExtensionFieldOpenings::with_capacity(challenge_indexes.len());

        // Open each final challenge position using extension field operations
        for &index in challenge_indexes {
            let opening = ext_mmcs.open_batch(index, prover_data);
            // Store final extension field values with base field proofs
            openings.push(opening.opened_values[0].clone(), opening.opening_proof);
        }

        // Hint final extension field opening data to prover state
        hint_extension_field_openings(&openings, prover_state);
    }

    /// Creates a base field MMCS instance using the handler's cryptographic primitives.
    fn create_base_mmcs(&self) -> BaseMmcs<F::Packing, H, C, DIGEST_ELEMS> {
        // Create MMCS with cloned references to cryptographic primitives
        BaseMmcs::new(self.hasher.clone(), self.compressor.clone())
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;

    use super::*;
    use crate::constant::K_SKIP_SUMCHECK;

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
        let config = StirConfig::new();
        assert!(!config.should_apply_univariate_skip(0));

        let config = StirConfig::builder()
            .initial_statement(true)
            .univariate_skip(true)
            .folding_factor_at_round(K_SKIP_SUMCHECK + 1)
            .build();

        assert!(config.should_apply_univariate_skip(0));
        assert!(!config.should_apply_univariate_skip(1));
    }
}
