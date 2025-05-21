use p3_commit::{ExtensionMmcs, Mmcs};
use p3_field::{ExtensionField, Field, PrimeField64, TwoAdicField};
use p3_matrix::Dimensions;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CryptographicHasher, Hash, PseudoCompressionFunction};
use serde::{Deserialize, Serialize};

use super::{Verifier, utils::read_sumcheck_rounds};
use crate::{
    fiat_shamir::{
        errors::{ProofError, ProofResult},
        pow::traits::PowStrategy,
        verifier::VerifierState,
    },
    poly::multilinear::MultilinearPoint,
    sumcheck::sumcheck_polynomial::SumcheckPolynomial,
    whir::{Leafs, Proof, prover::proof::WhirProof, utils::get_challenge_stir_queries},
};

/// Tracks the verifier's internal state across folding rounds in the WHIR protocol.
///
/// This structure is used to manage the verifier's evolving context as each folding
/// round is processed. After each round, the verifier updates this state
/// to reflect the next recursive domain and commitment.
///
/// The fields are updated in-place across rounds to avoid redundant allocations and
/// allow seamless round-to-round transitions.
#[derive(Debug)]
pub(crate) struct VerifierRoundState<F, SF, H, C, const DIGEST_ELEMS: usize>
where
    H: CryptographicHasher<SF, [u8; DIGEST_ELEMS]> + Sync,
    C: PseudoCompressionFunction<[u8; DIGEST_ELEMS], 2> + Sync,
    [u8; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    SF: Field + TwoAdicField + PrimeField64,
    F: Field + ExtensionField<SF> + TwoAdicField,
{
    /// Merkle root of the current round's commitment.
    ///
    /// This hash is used to verify Merkle openings of queried evaluations in the
    /// current domain. It gets updated after each round to reflect the root of the
    /// newly committed (folded) evaluation table.
    pub(crate) prev_root: Hash<SF, u8, DIGEST_ELEMS>,

    /// Folding randomness vector for this round.
    ///
    /// This is the challenge point used to evaluate multilinear polynomials
    /// during the folding step. It is derived via Fiat-Shamir and updated each round.
    pub(crate) folding_randomness: MultilinearPoint<F>,

    /// Generator of the current multiplicative evaluation domain.
    ///
    /// Each round reduces the domain size, and this generator corresponds to the
    /// updated subgroup used for querying STIR constraints.
    pub(crate) domain_gen: F,

    /// Inverse of the current domain generator.
    ///
    /// This is needed for interpolation or evaluation logic in the verifier.
    pub(crate) domain_gen_inv: F,

    /// Exponential generator of the coset used for STIR evaluations.
    ///
    /// This is equal to `domain_gen^{2^fold_r}` and is used to evaluate
    /// points in the affine coset during Merkle checks.
    pub(crate) exp_domain_gen: F,

    /// Size of the current evaluation domain.
    ///
    /// This gets halved in each folding round. It determines the height
    /// of the 2D matrix over which evaluations are structured.
    pub(crate) domain_size: usize,

    /// Merkle commitment scheme for base field tables.
    ///
    /// Used during the first round to verify Merkle openings when evaluations
    /// are committed in the base field.
    pub(crate) mmcs: MerkleTreeMmcs<SF, u8, H, C, DIGEST_ELEMS>,

    /// Merkle commitment scheme for extension field tables.
    ///
    /// Used in all subsequent rounds when the committed evaluations
    /// lie in the extension field.
    pub(crate) extension_mmcs: ExtensionMmcs<SF, F, MerkleTreeMmcs<SF, u8, H, C, DIGEST_ELEMS>>,
}

impl<F, SF, H, C, const DIGEST_ELEMS: usize> VerifierRoundState<F, SF, H, C, DIGEST_ELEMS>
where
    H: CryptographicHasher<SF, [u8; DIGEST_ELEMS]> + Sync,
    C: PseudoCompressionFunction<[u8; DIGEST_ELEMS], 2> + Sync,
    [u8; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    SF: Field + TwoAdicField + PrimeField64,
    F: ExtensionField<SF> + TwoAdicField,
{
    pub(crate) fn build_parsed_round<PS: PowStrategy>(
        &mut self,
        verifier: &Verifier<'_, F, SF, H, C, PS>,
        verifier_state: &mut VerifierState<'_, F, SF>,
        whir_proof: &WhirProof<SF, F, DIGEST_ELEMS>,
        r: usize,
    ) -> ProofResult<ParsedRound<F>> {
        let round_params = &verifier.round_parameters[r];
        let fold_r = verifier.folding_factor.at_round(r);

        let new_root = verifier_state.read_digest()?;

        let mut ood_points = vec![F::ZERO; round_params.ood_samples];
        let mut ood_answers = vec![F::ZERO; round_params.ood_samples];
        if round_params.ood_samples > 0 {
            verifier_state.fill_challenge_scalars(&mut ood_points)?;
            verifier_state.fill_next_scalars(&mut ood_answers)?;
        }

        let stir_challenges_indexes = get_challenge_stir_queries(
            self.domain_size,
            fold_r,
            round_params.num_queries,
            verifier_state,
        )?;

        let stir_challenges_points: Vec<_> = stir_challenges_indexes
            .iter()
            .map(|&i| self.exp_domain_gen.exp_u64(i as u64))
            .collect();

        // Verify Merkle openings using `verify_batch`
        let dimensions = vec![Dimensions {
            height: self.domain_size >> fold_r,
            width: 1 << fold_r,
        }];

        let stir_challenges_answers: Vec<Vec<F>> = if r == 0 {
            // Case: r == 0, use base field

            // Deserialize base field answers and Merkle proofs from verifier transcript
            let _answers = verifier_state.hint::<Leafs<SF>>()?;
            let _merkle_proof = verifier_state.hint::<Proof<DIGEST_ELEMS>>()?;

            // Get reference to prover-committed data (must match deserialized hints)
            let (answers, merkle_proof) = &whir_proof.commitment_merkle_paths;

            // Verify each queried leaf
            for (i, &stir_challenges_index) in stir_challenges_indexes.iter().enumerate() {
                self.mmcs
                    .verify_batch(
                        &self.prev_root,
                        &dimensions,
                        stir_challenges_index,
                        &[answers[i].iter().map(|v| v.as_base().unwrap()).collect()],
                        &merkle_proof[i],
                    )
                    .map_err(|_| ProofError::InvalidProof)?;
            }

            // Convert answers from SF to F for downstream usage
            answers
                .iter()
                .map(|inner| inner.iter().map(|&f_el| f_el.into()).collect())
                .collect()
        } else {
            // Case: r > 0, use extension field

            // Deserialize extension field answers and Merkle proofs from verifier transcript
            let _answers = verifier_state.hint::<Leafs<F>>()?;
            let _merkle_proof = verifier_state.hint::<Proof<DIGEST_ELEMS>>()?;

            // Get reference to prover-committed data
            let (answers, merkle_proof) = &whir_proof.merkle_paths[r - 1];

            // Verify each queried leaf
            for (i, &stir_challenges_index) in stir_challenges_indexes.iter().enumerate() {
                self.extension_mmcs
                    .verify_batch(
                        &self.prev_root,
                        &dimensions,
                        stir_challenges_index,
                        &[answers[i].clone()],
                        &merkle_proof[i],
                    )
                    .map_err(|_| ProofError::InvalidProof)?;
            }

            // Simply return the deserialized extension field leaf vectors
            answers.clone()
        };

        if round_params.pow_bits > 0. {
            verifier_state.challenge_pow::<PS>(round_params.pow_bits)?;
        }

        let [combination_randomness_gen] = verifier_state.challenge_scalars()?;
        let combination_randomness = combination_randomness_gen
            .powers()
            .take(stir_challenges_indexes.len() + round_params.ood_samples)
            .collect();

        // We read:
        // - The sumcheck polynomials produced by the prover,
        // - The folding randomness used in each corresponding round
        let (sumcheck_rounds, new_folding_randomness) = read_sumcheck_rounds::<_, _, PS>(
            verifier_state,
            verifier.folding_factor.at_round(r + 1),
            round_params.folding_pow_bits,
            false,
        )?;

        let parsed = ParsedRound {
            folding_randomness: self.folding_randomness.clone(),
            ood_points,
            ood_answers,
            stir_challenges_indexes,
            stir_challenges_points,
            stir_challenges_answers,
            combination_randomness,
            sumcheck_rounds,
            domain_gen_inv: self.domain_gen_inv,
        };

        self.folding_randomness = new_folding_randomness;
        self.prev_root = new_root;
        self.domain_gen = self.domain_gen.square();
        self.exp_domain_gen = self
            .domain_gen
            .exp_power_of_2(verifier.folding_factor.at_round(r + 1));
        self.domain_gen_inv = self.domain_gen_inv.square();
        self.domain_size /= 2;

        Ok(parsed)
    }
}

/// Represents a single folding round in the WHIR protocol.
///
/// This structure enables recursive compression and verification of a Reed–Solomon
/// proximity test under algebraic constraints.
#[derive(Default, Debug, Clone)]
pub(crate) struct ParsedRound<F> {
    /// Folding randomness vector used in this round.
    pub(crate) folding_randomness: MultilinearPoint<F>,
    /// Out-of-domain query points.
    pub(crate) ood_points: Vec<F>,
    /// OOD answers at each query point.
    pub(crate) ood_answers: Vec<F>,
    /// Indexes of STIR constraint polynomials used in this round.
    pub(crate) stir_challenges_indexes: Vec<usize>,
    /// STIR constraint evaluation points.
    pub(crate) stir_challenges_points: Vec<F>,
    /// Answers to the STIR constraints at each evaluation point.
    pub(crate) stir_challenges_answers: Vec<Vec<F>>,
    /// Randomness used to linearly combine constraints.
    pub(crate) combination_randomness: Vec<F>,
    /// Sumcheck messages and challenge values for verifying correctness.
    pub(crate) sumcheck_rounds: Vec<(SumcheckPolynomial<F>, F)>,
    /// Inverse of the domain generator used in this round.
    pub(crate) domain_gen_inv: F,
}
