use std::{fmt::Write, marker::PhantomData};

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field, TwoAdicField};

use crate::{
    fiat_shamir::{proof_data::ProofData, prover::ProverState, verifier::VerifierState},
    sumcheck::K_SKIP_SUMCHECK,
    whir::parameters::WhirConfig,
};

/// This is the separator between operations in the IO Pattern
/// and as such is the only forbidden character in labels.
const SEP_BYTE: &str = "\0";

/// Configuration parameters for a sumcheck phase in the protocol.
#[derive(Debug)]
pub struct SumcheckParams {
    /// Total number of sumcheck rounds to perform.
    ///
    /// Each round corresponds to a polynomial sent by the prover and a challenge
    /// sampled by the verifier.
    pub rounds: usize,

    /// Number of bits required for the proof-of-work challenge.
    ///
    /// - If `pow_bits > 0`, a PoW challenge is inserted after each round.
    /// - If `pow_bits == 0`, PoW is disabled.
    pub pow_bits: usize,

    /// Optional number of variables to skip using the univariate skip optimization.
    ///
    /// - If `None`, all rounds are performed normally.
    /// - If `Some(k)`, the first `k` variables are skipped by replacing them with a
    ///   single low-degree extension (LDE) step, provided `k > 1`.
    pub univariate_skip: Option<usize>,
}

/// The IO Pattern of an interactive protocol.
///
/// An IO pattern is a string that specifies the protocol in a simple,
/// non-ambiguous, human-readable format. A typical example is the following:
///
/// ```text
///     domain-separator A32generator A32public-key R A32commitment S32challenge A32response
/// ```
/// The domain-separator is a user-specified string uniquely identifying the end-user application
/// (to avoid cross-protocol attacks). The letter `A` indicates the absorption of a public input (an
/// `ABSORB`), while the letter `S` indicates the squeezing (a `SQUEEZE`) of a challenge. The letter
/// `R` indicates a ratcheting operation: ratcheting means invoking the hash function even on an
/// incomplete block. It provides forward secrecy and allows it to start from a clean rate.
/// After the operation type, is the number of elements in base 10 that are being absorbed/squeezed.
/// Then, follows the label associated with the element being absorbed/squeezed. This often comes
/// from the underlying description of the protocol. The label cannot start with a digit or contain
/// the NULL byte.
///
/// ## Guarantees
///
/// The struct [`DomainSeparator`] guarantees the creation of a valid IO Pattern string, whose
/// lengths are coherent with the types described in the protocol. No information about the types
/// themselves is stored in an IO Pattern. This means that [`ProverState`][`crate::ProverState`] or [`VerifierState`][`crate::VerifierState`] instances can generate successfully a protocol transcript respecting the length constraint but not the types. See [issue #6](https://github.com/arkworks-rs/spongefish/issues/6) for a discussion on the topic.
#[derive(Clone, Debug)]
pub struct DomainSeparator<EF, F> {
    /// The internal IOPattern string representation.
    ///
    /// This string encodes a sequence of transcript actions such as absorptions, squeezes,
    /// and ratchets, in the format: `domain\0A32label\0S16challenge\0R...`.
    ///
    /// It is constructed incrementally by calling methods like `obverse`, `sample`, etc.,
    /// and is later parsed into a queue of [`Op`] instructions by `finalize()`.
    io: String,

    /// Phantom marker for the base field type `F`.
    ///
    /// Ensures that field operations (e.g., `as_basis_coefficients_slice`) are
    /// computed correctly for the given field implementation.
    _field: PhantomData<F>,

    /// Phantom marker for the extension field type `EF`.
    ///
    /// Provides type-level tracking of the extension degree and element structure used in
    /// challenge generation and scalar absorption.
    _extension_field: PhantomData<EF>,
}

impl<EF, F> DomainSeparator<EF, F>
where
    EF: ExtensionField<F> + TwoAdicField,
    F: Field + TwoAdicField,
{
    #[must_use]
    pub const fn from_string(io: String) -> Self {
        Self {
            io,
            _field: PhantomData,
            _extension_field: PhantomData,
        }
    }

    /// Create a new DomainSeparator with the domain separator.
    #[must_use]
    pub fn new(session_identifier: &str) -> Self {
        assert!(
            !session_identifier.contains(SEP_BYTE),
            "Domain separator cannot contain the separator BYTE."
        );
        Self::from_string(session_identifier.to_string())
    }

    /// Observe `count` native elements.
    pub fn observe(&mut self, count: usize, label: &str) {
        assert!(count > 0, "Count must be positive.");
        assert!(
            !label.contains(SEP_BYTE),
            "Label cannot contain the separator BYTE."
        );
        assert!(
            label
                .chars()
                .next()
                .is_none_or(|char| !char.is_ascii_digit()),
            "Label cannot start with a digit."
        );

        self.io += SEP_BYTE;
        write!(self.io, "O{count}{label}").expect("writing to String cannot fail");
    }

    /// Sample `count` native elements.
    pub fn sample(&mut self, count: usize, label: &str) {
        assert!(count > 0, "Count must be positive.");
        assert!(
            !label.contains(SEP_BYTE),
            "Label cannot contain the separator BYTE."
        );
        assert!(
            label
                .chars()
                .next()
                .is_none_or(|char| !char.is_ascii_digit()),
            "Label cannot start with a digit."
        );

        self.io += SEP_BYTE;
        write!(self.io, "S{count}{label}").expect("writing to String cannot fail");
    }

    /// Hint `count` native elements.
    pub fn hint(&mut self, label: &str) {
        assert!(
            !label.contains(SEP_BYTE),
            "Label cannot contain the separator BYTE."
        );

        self.io += SEP_BYTE;
        write!(self.io, "H{label}").expect("writing to String cannot fail");
    }

    // /// Return the IO Pattern as a Vec of Units.
    // #[must_use]
    // pub fn as_units(&self) -> Vec<U> {
    //     U::slice_from_u8_slice(self.io.as_bytes())
    // }

    #[must_use]
    pub fn as_field_elements(&self) -> Vec<F> {
        // let (value, _) =
        //     bincode::serde::decode_from_slice(self.io.as_bytes(), bincode::config::standard())
        //         .unwrap();

        // value

        (0..self.io.len()).map(|i| F::from_u64(i as u64)).collect()
    }

    /// Create an [`crate::ProverState`] instance from the IO Pattern.
    #[must_use]
    pub fn to_prover_state<Challenger, const DIGEST_ELEMS: usize>(
        &self,
        challenger: Challenger,
    ) -> ProverState<EF, F, Challenger, DIGEST_ELEMS>
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        ProverState::new(self, challenger)
    }

    /// Create a [`crate::VerifierState`] instance from the IO Pattern and the protocol transcript
    /// (bytes).
    #[must_use]
    pub fn to_verifier_state<Challenger, const DIGEST_ELEMS: usize>(
        &self,
        proof_data: ProofData<EF, F, F, DIGEST_ELEMS>,
        challenger: Challenger,
    ) -> VerifierState<EF, F, Challenger, DIGEST_ELEMS>
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        VerifierState::new(self, proof_data, challenger)
    }

    pub fn add_ood(&mut self, num_samples: usize) {
        if num_samples > 0 {
            self.sample(num_samples, "ood_query");
            self.observe(num_samples, "ood_ans");
        }
    }

    pub fn commit_statement<HC, C, Challenger, const DIGEST_ELEMS: usize>(
        &mut self,
        params: &WhirConfig<EF, F, HC, C, Challenger>,
    ) where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // TODO: Add params
        self.observe(DIGEST_ELEMS, "merkle_digest");
        if params.committment_ood_samples > 0 {
            assert!(params.initial_statement);
            self.add_ood(params.committment_ood_samples);
        }
    }

    pub fn add_whir_proof<HC, C, Challenger, const DIGEST_ELEMS: usize>(
        &mut self,
        params: &WhirConfig<EF, F, HC, C, Challenger>,
    ) where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // TODO: Add statement
        if params.initial_statement {
            self.sample(1, "initial_combination_randomness");
            self.add_sumcheck(&SumcheckParams {
                rounds: params.folding_factor.at_round(0),
                pow_bits: params.starting_folding_pow_bits,
                univariate_skip: if params.univariate_skip {
                    Some(K_SKIP_SUMCHECK)
                } else {
                    None
                },
            });
        } else {
            self.sample(params.folding_factor.at_round(0), "folding_randomness");
            self.pow(params.starting_folding_pow_bits);
        }

        let mut domain_size = params.starting_domain.size();
        for (round, r) in params.round_parameters.iter().enumerate() {
            let folded_domain_size = domain_size >> params.folding_factor.at_round(round);
            let domain_size_bytes = ((folded_domain_size * 2 - 1).ilog2() as usize).div_ceil(8);
            self.observe(DIGEST_ELEMS, "merkle_digest");
            self.add_ood(r.ood_samples);
            self.sample(r.num_queries * domain_size_bytes, "stir_queries");
            self.hint("stir_queries");
            self.hint("merkle_proof");
            self.pow(r.pow_bits);
            self.sample(1, "combination_randomness");

            self.add_sumcheck(&SumcheckParams {
                rounds: params.folding_factor.at_round(round + 1),
                pow_bits: r.folding_pow_bits,
                univariate_skip: None,
            });
            domain_size >>= params.rs_reduction_factor(round);
        }

        let folded_domain_size = domain_size
            >> params
                .folding_factor
                .at_round(params.round_parameters.len());
        let domain_size_bytes = ((folded_domain_size * 2 - 1).ilog2() as usize).div_ceil(8);

        self.observe(1 << params.final_sumcheck_rounds, "final_coeffs");

        self.sample(domain_size_bytes * params.final_queries, "final_queries");
        self.hint("stir_answers");
        self.hint("merkle_proof");
        self.pow(params.final_pow_bits);
        self.add_sumcheck(&SumcheckParams {
            rounds: params.final_sumcheck_rounds,
            pow_bits: params.final_folding_pow_bits,
            univariate_skip: None,
        });
        self.hint("deferred_weight_evaluations");
    }

    /// Append the sumcheck protocol transcript steps to the domain separator.
    ///
    /// This method encodes one or more rounds of the sumcheck protocol, including:
    /// - Absorbing polynomial coefficients sent by the prover.
    /// - Sampling verifier challenges for folding randomness.
    /// - Optionally performing a proof-of-work challenge for each round.
    ///
    /// If `univariate_skip` is enabled with `k > 1`, the first `k` variables are skipped
    /// using a low-degree extension (LDE) over a coset. In this case, the transcript
    /// includes a single LDE observation step and challenge, followed by the remaining rounds.
    ///
    /// # Parameters
    /// - `rounds`: Total number of variables folded by the sumcheck protocol.
    /// - `pow_bits`: If greater than 0.0, a proof-of-work challenge is appended after each round.
    /// - `univariate_skip`: If `Some(k)`, applies the univariate skip optimization by skipping
    ///   the first `k` rounds and replacing them with a single LDE + challenge step.
    pub fn add_sumcheck(&mut self, params: &SumcheckParams) {
        let SumcheckParams {
            rounds,
            pow_bits,
            univariate_skip,
        } = *params;

        // Determine the number of rounds to skip using univariate skip optimization.
        let k = univariate_skip.unwrap_or(0);

        // If univariate skip is active and skipping more than 1 round,
        // perform a low-degree extension (LDE) step over a coset:
        // - Absorb 2^{k+1} scalars (the LDE evaluations of the skipped polynomial).
        // - Sample 1 challenge for the folding randomness.
        // - Optionally perform PoW after the LDE step.
        if k > 1 {
            let lde_size = 1 << (k + 1);
            self.observe(lde_size, "sumcheck_poly_skip");
            self.sample(1, "folding_randomness_skip");
            self.pow(pow_bits);
        }

        // Proceed with the remaining (unskipped) sumcheck rounds.
        // Each round:
        // - Absorbs 3 scalars (coefficients of a degree-2 polynomial).
        // - Samples 1 folding randomness challenge.
        // - Optionally performs a PoW challenge.
        for _ in k..rounds {
            self.observe(3, "sumcheck_poly");
            self.sample(1, "folding_randomness");
            self.pow(pow_bits);
        }
    }

    /// Optionally append a proof-of-work challenge to the domain separator.
    ///
    /// This function adds a transcript step that enforces a [proof-of-work (PoW)](https://en.wikipedia.org/wiki/Proof_of_work) requirement
    /// during Fiat-Shamir transformation. If `bits` is positive, it adds:
    ///
    /// 1. A 32-byte challenge sampled from the transcript, labeled by `"pow-queries"`.
    /// 2. An 8-byte observed nonce, labeled `"pow-nonce"`.
    ///
    /// The verifier will later check that the nonce satisfies the PoW condition relative to the challenge
    /// and the `bits` difficulty.
    ///
    /// # Parameters
    ///
    /// - `bits`: Number of bits of PoW difficulty.
    ///     - If `bits == 0.0`, nothing is added.
    ///     - If `bits > 0.0`, a PoW round is added.
    pub fn pow(&mut self, bits: usize) {
        if bits > 0 {
            // Step 1: Sample a 32-byte challenge (typically used as PoW preimage)
            self.sample(32, "pow-queries");

            // Step 2: Observe an 8-byte nonce in response
            self.observe(8, "pow-nonce");
        }
    }
}
