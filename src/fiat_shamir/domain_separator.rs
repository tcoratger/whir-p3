use std::marker::PhantomData;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field, TwoAdicField};

use crate::{
    constant::K_SKIP_SUMCHECK,
    fiat_shamir::{
        pattern::{Hint, Observe, Pattern, Sample},
        prover::ProverState,
        verifier::VerifierState,
    },
    whir::parameters::WhirConfig,
};

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

/// The pattern of an interactive protocol.
#[derive(Clone, Debug)]
pub struct DomainSeparator<EF, F> {
    /// The internal pattern finite field representation.
    pattern: Vec<F>,

    /// Phantom marker for the extension field type `EF`.
    ///
    /// Provides type-level tracking of the extension degree and element structure used in
    /// challenge generation and scalar absorption.
    _extension_field: PhantomData<EF>,
}

impl<EF, F> DomainSeparator<EF, F>
where
    EF: ExtensionField<F>,
    F: Field,
{
    #[must_use]
    pub const fn from_pattern(pattern: Vec<F>) -> Self {
        Self {
            pattern,
            _extension_field: PhantomData,
        }
    }

    /// Create a new DomainSeparator with the domain separator.
    #[must_use]
    pub const fn new(pattern: Vec<F>) -> Self {
        Self::from_pattern(pattern)
    }

    /// Observe `count` native elements.
    pub fn observe(&mut self, count: usize, pattern: Observe) {
        self.pattern.push(
            pattern.as_field_element::<F>()
                + F::from_usize(count)
                + Pattern::Observe.as_field_element::<F>(),
        );
    }

    /// Sample `count` native elements.
    pub fn sample(&mut self, count: usize, pattern: Sample) {
        self.pattern.push(
            pattern.as_field_element::<F>()
                + F::from_usize(count)
                + Pattern::Sample.as_field_element::<F>(),
        );
    }

    /// Hint `count` native elements.
    pub fn hint(&mut self, pattern: Hint) {
        self.pattern
            .push(pattern.as_field_element::<F>() + Pattern::Hint.as_field_element::<F>());
    }

    #[must_use]
    pub fn as_field_elements(&self) -> Vec<F> {
        self.pattern.clone()
    }

    /// Create a prover state from the domain separator
    #[must_use]
    pub fn to_prover_state<Challenger>(
        &self,
        challenger: Challenger,
    ) -> ProverState<F, EF, Challenger>
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        ProverState::new(self, challenger)
    }

    /// Create a verifier state from the domain separator
    #[must_use]
    pub fn to_verifier_state<Challenger>(
        &self,
        proof_data: Vec<F>,
        challenger: Challenger,
    ) -> VerifierState<F, EF, Challenger>
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        VerifierState::new(self, proof_data, challenger)
    }

    pub fn add_ood(&mut self, num_samples: usize) {
        if num_samples > 0 {
            self.sample(num_samples, Sample::OodQuery);
            self.observe(num_samples, Observe::OodAnswers);
        }
    }

    pub fn commit_statement<HC, C, Challenger, const DIGEST_ELEMS: usize>(
        &mut self,
        params: &WhirConfig<EF, F, HC, C, Challenger>,
    ) where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // TODO: Add params
        self.observe(DIGEST_ELEMS, Observe::MerkleDigest);
        if params.commitment_ood_samples > 0 {
            assert!(params.initial_statement);
            self.add_ood(params.commitment_ood_samples);
        }
    }

    pub fn add_whir_proof<HC, C, Challenger, const DIGEST_ELEMS: usize>(
        &mut self,
        params: &WhirConfig<EF, F, HC, C, Challenger>,
    ) where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
        EF: TwoAdicField,
        F: TwoAdicField,
    {
        // TODO: Add statement
        if params.initial_statement {
            self.sample(1, Sample::InitialCombinationRandomness);
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
            self.sample(params.folding_factor.at_round(0), Sample::FoldingRandomness);
            self.pow(params.starting_folding_pow_bits);
        }

        let mut domain_size = params.starting_domain_size();
        for (round, r) in params.round_parameters.iter().enumerate() {
            let folded_domain_size = domain_size >> params.folding_factor.at_round(round);
            let domain_size_bytes = ((folded_domain_size * 2 - 1).ilog2() as usize).div_ceil(8);
            self.observe(DIGEST_ELEMS, Observe::MerkleDigest);
            self.add_ood(r.ood_samples);
            self.sample(r.num_queries * domain_size_bytes, Sample::StirQueries);
            self.hint(Hint::StirQueries);
            self.hint(Hint::MerkleProof);
            self.pow(r.pow_bits);
            self.sample(1, Sample::CombinationRandomness);

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

        self.observe(1 << params.final_sumcheck_rounds, Observe::FinalCoeffs);

        self.sample(
            domain_size_bytes * params.final_queries,
            Sample::FinalQueries,
        );
        self.hint(Hint::StirAnswers);
        self.hint(Hint::MerkleProof);
        self.pow(params.final_pow_bits);
        self.add_sumcheck(&SumcheckParams {
            rounds: params.final_sumcheck_rounds,
            pow_bits: params.final_folding_pow_bits,
            univariate_skip: None,
        });
        self.hint(Hint::DeferredWeightEvaluations);
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
            self.observe(lde_size, Observe::SumcheckPolySkip);
            self.sample(1, Sample::FoldingRandomnessSkip);
            self.pow(pow_bits);
        }

        // Proceed with the remaining (unskipped) sumcheck rounds.
        // Each round:
        // - Absorbs 3 scalars (coefficients of a degree-2 polynomial).
        // - Samples 1 folding randomness challenge.
        // - Optionally performs a PoW challenge.
        for _ in k..rounds {
            self.observe(3, Observe::SumcheckPoly);
            self.sample(1, Sample::FoldingRandomness);
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
            self.sample(32, Sample::PowQueries);

            // Step 2: Observe an 8-byte nonce in response
            self.observe(8, Observe::PowNonce);
        }
    }
}
