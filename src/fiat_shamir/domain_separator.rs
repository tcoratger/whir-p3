use std::{fmt::Write, marker::PhantomData};

use p3_challenger::{CanObserve, CanSample};
use p3_field::{ExtensionField, Field, PrimeField64, TwoAdicField};

use crate::{
    fiat_shamir::{prover::ProverState, unit::Unit, verifier::VerifierState},
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
    /// - If `pow_bits > 0.0`, a PoW challenge is inserted after each round.
    /// - If `pow_bits == 0.0`, PoW is disabled.
    pub pow_bits: f64,

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
pub struct DomainSeparator<EF, F, U>
where
    U: Unit,
{
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

    /// Phantom marker for the unit type `U`.
    _unit: PhantomData<U>,
}

impl<EF, F, U> DomainSeparator<EF, F, U>
where
    U: Unit + Default + Copy,
    EF: ExtensionField<F> + TwoAdicField,
    F: Field + TwoAdicField + PrimeField64,
{
    #[must_use]
    pub const fn from_string(io: String) -> Self {
        Self {
            io,
            _field: PhantomData,
            _extension_field: PhantomData,
            _unit: PhantomData,
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

    /// Return the IO Pattern as a Vec of Units.
    #[must_use]
    pub fn as_units(&self) -> Vec<U> {
        U::slice_from_u8_slice(self.io.as_bytes())
    }

    /// Create an [`crate::ProverState`] instance from the IO Pattern.
    #[must_use]
    pub fn to_prover_state<H>(&self, challenger: H) -> ProverState<EF, F, H, U>
    where
        H: CanObserve<U> + CanSample<U> + Clone,
    {
        ProverState::new(self, challenger)
    }

    /// Create a [`crate::VerifierState`] instance from the IO Pattern and the protocol transcript
    /// (bytes).
    #[must_use]
    pub fn to_verifier_state<'a, H>(
        &self,
        transcript: &'a [u8],
        challenger: H,
    ) -> VerifierState<'a, EF, F, H, U>
    where
        H: CanObserve<U> + CanSample<U> + Clone,
    {
        VerifierState::new(self, transcript, challenger)
    }

    pub fn add_ood(&mut self, num_samples: usize) {
        if num_samples > 0 {
            self.sample_scalars(num_samples, "ood_query");
            self.observe_scalars(num_samples, "ood_ans");
        }
    }

    pub fn commit_statement<PowStrategy, HC, C, Challenger, const DIGEST_ELEMS: usize>(
        &mut self,
        params: &WhirConfig<EF, F, HC, C, PowStrategy, Challenger, U>,
    ) where
        Challenger: CanObserve<U> + CanSample<U>,
    {
        // TODO: Add params
        self.observe(DIGEST_ELEMS, "merkle_digest");
        if params.committment_ood_samples > 0 {
            assert!(params.initial_statement);
            self.add_ood(params.committment_ood_samples);
        }
    }

    pub fn add_whir_proof<PowStrategy, HC, C, Challenger, const DIGEST_ELEMS: usize>(
        &mut self,
        params: &WhirConfig<EF, F, HC, C, PowStrategy, Challenger, U>,
    ) where
        Challenger: CanObserve<U> + CanSample<U>,
    {
        // TODO: Add statement
        if params.initial_statement {
            self.sample_scalars(1, "initial_combination_randomness");
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
            self.sample_scalars(params.folding_factor.at_round(0), "folding_randomness");
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
            self.sample_scalars(1, "combination_randomness");

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

        self.observe_scalars(1 << params.final_sumcheck_rounds, "final_coeffs");

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
            self.observe_scalars(lde_size, "sumcheck_poly_skip");
            self.sample_scalars(1, "folding_randomness_skip");
            self.pow(pow_bits);
        }

        // Proceed with the remaining (unskipped) sumcheck rounds.
        // Each round:
        // - Absorbs 3 scalars (coefficients of a degree-2 polynomial).
        // - Samples 1 folding randomness challenge.
        // - Optionally performs a PoW challenge.
        for _ in k..rounds {
            self.observe_scalars(3, "sumcheck_poly");
            self.sample_scalars(1, "folding_randomness");
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
    pub fn pow(&mut self, bits: f64) {
        if bits > 0.0 {
            // Step 1: Sample a 32-byte challenge (typically used as PoW preimage)
            self.sample(32, "pow-queries");

            // Step 2: Observe an 8-byte nonce in response
            self.observe(8, "pow-nonce");
        }
    }

    pub fn observe_scalars(&mut self, count: usize, label: &str) {
        // Observe `count` scalars into the transcript.
        self.observe(U::scalar_observe_count::<F, EF>(count), label);
    }

    pub fn sample_scalars(&mut self, count: usize, label: &str) {
        // Sample `count` scalars from the transcript using a "challenge" tag.
        self.sample(U::scalar_sample_count::<F, EF>(count), label);
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;

    use super::*;

    type F = BabyBear;
    type EF4 = BinomialExtensionField<F, 4>;

    #[test]
    fn test_domain_separator_new_and_bytes() {
        let ds = DomainSeparator::<EF4, F, u8>::new("session");
        assert_eq!(ds.as_units(), b"session");
    }

    #[test]
    #[should_panic]
    fn test_new_with_separator_byte_panics() {
        // This should panic because "\0" is forbidden in the session identifier.
        let _ = DomainSeparator::<EF4, F, u8>::new("invalid\0session");
    }

    #[test]
    fn test_observe_return_value_format() {
        let mut ds = DomainSeparator::<EF4, F, u8>::new("proto");
        ds.observe(3, "input");
        let expected_str = "proto\0O3input"; // initial + SEP + observe op + label
        assert_eq!(ds.as_units(), expected_str.as_bytes());
    }

    #[test]
    #[should_panic]
    fn test_observe_zero_panics() {
        DomainSeparator::<EF4, F, u8>::new("x").observe(0, "label");
    }

    #[test]
    #[should_panic]
    fn test_label_with_separator_byte_panics() {
        DomainSeparator::<EF4, F, u8>::new("x").observe(1, "bad\0label");
    }

    #[test]
    #[should_panic]
    fn test_label_starts_with_digit_panics() {
        DomainSeparator::<EF4, F, u8>::new("x").observe(1, "1label");
    }

    #[test]
    fn test_round_trip_operations() {
        let mut ds1 = DomainSeparator::<EF4, F, u8>::new("foo");
        ds1.observe(2, "a");
        ds1.sample(3, "b");
        let ops1 = ds1.clone().io;

        let tag = String::from_utf8(ds1.as_units()).unwrap();
        let ds2 = DomainSeparator::<EF4, F, u8>::from_string(tag);
        let ops2 = ds2.io;

        assert_eq!(ops1, ops2);
    }

    #[test]
    fn test_squeeze_returns_correct_string() {
        let mut ds = DomainSeparator::<EF4, F, u8>::new("proto");
        ds.sample(4, "challenge");
        let expected_str = "proto\0S4challenge";
        assert_eq!(ds.as_units(), expected_str.as_bytes());
    }

    #[test]
    #[should_panic]
    fn test_squeeze_zero_count_panics() {
        DomainSeparator::<EF4, F, u8>::new("proto").sample(0, "label");
    }

    #[test]
    #[should_panic]
    fn test_squeeze_label_with_null_byte_panics() {
        DomainSeparator::<EF4, F, u8>::new("proto").sample(2, "bad\0label");
    }

    #[test]
    #[should_panic]
    fn test_squeeze_label_starts_with_digit_panics() {
        DomainSeparator::<EF4, F, u8>::new("proto").sample(2, "1invalid");
    }

    #[test]
    fn test_multiple_squeeze_chaining() {
        let mut ds = DomainSeparator::<EF4, F, u8>::new("proto");
        ds.sample(1, "first");
        ds.sample(2, "second");
        let expected_str = "proto\0S1first\0S2second";
        assert_eq!(ds.as_units(), expected_str.as_bytes());
    }

    #[test]
    fn test_add_scalars_babybear() {
        // Test observation of scalar field elements (BabyBear).
        // - BabyBear is a base field with extension degree = 1
        // - bits = 31 → NUM_BYTES = 4
        // - 2 scalars * 1 * 4 = 8 bytes absorbed
        // - "O" indicates observation in the domain separator
        let mut domsep: DomainSeparator<F, F, u8> = DomainSeparator::new("babybear");
        domsep.observe_scalars(2, "foo");
        let expected = b"babybear\0O8foo";
        assert_eq!(domsep.as_units(), expected);
    }

    #[test]
    fn test_challenge_scalars_babybear() {
        // Test sampling scalar field elements (BabyBear).
        // - BabyBear has extension degree = 1
        // - bits = 31 → bytes_uniform_modp(31) = 5
        // - 3 scalars * 1 * 5 = 15 bytes squeezed
        // - "S" indicates sampling in the domain separator
        let mut domsep: DomainSeparator<F, F, u8> = DomainSeparator::new("bb");
        domsep.sample_scalars(3, "bar");
        let expected = b"bb\0S57bar";
        assert_eq!(domsep.as_units(), expected);
    }

    #[test]
    fn test_add_scalars_quartic_ext_field() {
        // Test observation of scalars from a quartic extension field EF4.
        // - EF4 has extension degree = 4
        // - Base field bits = 31 → NUM_BYTES = 4
        // - 2 scalars * 4 * 4 = 32 bytes observed
        let mut domsep: DomainSeparator<EF4, F, u8> = DomainSeparator::new("ext");
        domsep.observe_scalars(2, "a");
        let expected = b"ext\0O32a";
        assert_eq!(domsep.as_units(), expected);
    }

    #[test]
    fn test_challenge_scalars_quartic_ext_field() {
        // Test squeezing of scalars from a quartic extension field EF4.
        // - EF4 has extension degree = 4
        // - Base field bits = 31 → bytes_uniform_modp(31) = 19
        // - 1 scalar * 4 * 19 = 76 bytes squeezed
        // - "S" indicates squeezing in the domain separator
        let mut domsep: DomainSeparator<EF4, F, u8> = DomainSeparator::new("ext2");
        domsep.sample_scalars(1, "b");

        let expected = b"ext2\0S76b";
        assert_eq!(domsep.as_units(), expected);
    }

    #[test]
    fn test_add_ood() {
        let iop: DomainSeparator<F, F, u8> = DomainSeparator::new("test_protocol");
        let mut updated_iop = iop.clone();
        let mut unchanged_iop = iop;

        // Apply OOD query addition
        updated_iop.add_ood(3);

        // Convert to a string for inspection
        let pattern_str = String::from_utf8(updated_iop.as_units()).unwrap();

        // Check if "ood_query" and "ood_ans" were correctly appended
        assert!(pattern_str.contains("ood_query"));
        assert!(pattern_str.contains("ood_ans"));

        // Test case where num_samples = 0 (should not modify anything)
        unchanged_iop.add_ood(0);
        let unchanged_str = String::from_utf8(unchanged_iop.as_units()).unwrap();
        assert_eq!(unchanged_str, "test_protocol"); // Should remain the same
    }

    #[test]
    fn test_pow() {
        let iop: DomainSeparator<F, F, u8> = DomainSeparator::new("test_protocol");
        let mut updated_iop = iop.clone();
        let mut unchanged_iop = iop;

        // Apply PoW challenge
        updated_iop.pow(10.0);

        // Convert to a string for inspection
        let pattern_str = String::from_utf8(updated_iop.as_units()).unwrap();

        // Check if "pow_queries" was correctly added
        assert!(pattern_str.contains("pow-queries"));

        // Test case where bits = 0 (should not modify anything)
        unchanged_iop.pow(0.0);
        let unchanged_str = String::from_utf8(unchanged_iop.as_units()).unwrap();
        assert_eq!(unchanged_str, "test_protocol"); // Should remain the same
    }

    #[test]
    fn test_hint_format_is_correct_in_bytes() {
        let mut ds = DomainSeparator::<EF4, F, u8>::new("proto");
        ds.hint("my_hint");
        let expected = b"proto\0Hmy_hint";
        assert_eq!(ds.as_units(), expected);
    }

    #[test]
    #[should_panic]
    fn test_hint_label_with_null_byte_panics() {
        let mut ds = DomainSeparator::<EF4, F, u8>::new("x");
        ds.hint("bad\0hint");
    }

    #[test]
    fn test_add_sumcheck_regular_two_rounds_no_pow() {
        // Set up a new domain separator
        let mut ds = DomainSeparator::<EF4, F, u8>::new("regular");

        // Add a sumcheck with 2 folding rounds, no PoW, no skipping
        ds.add_sumcheck(&SumcheckParams {
            rounds: 2,
            pow_bits: 0.,
            univariate_skip: None,
        });

        // Finalize the domain separator into a sequence of transcript operations
        let ops = ds.io;

        // Manually construct the expected transcript: 2 rounds of
        // - Observe 3 scalars (sumcheck_poly)
        // - Challenge 1 scalar (folding_randomness)
        let mut ds_expected = DomainSeparator::<EF4, F, u8>::new("regular");
        ds_expected.observe_scalars(3, "sumcheck_poly");
        ds_expected.sample_scalars(1, "folding_randomness");

        ds_expected.observe_scalars(3, "sumcheck_poly");
        ds_expected.sample_scalars(1, "folding_randomness");

        let ops_expected = ds_expected.io;
        assert_eq!(ops, ops_expected);
    }

    #[test]
    fn test_add_sumcheck_one_round_with_pow() {
        // One round with PoW enabled
        let mut ds = DomainSeparator::<EF4, F, u8>::new("pow");
        ds.add_sumcheck(&SumcheckParams {
            rounds: 1,
            pow_bits: 7.0,
            univariate_skip: None,
        });
        let ops = ds.io;

        // Expected operations:
        // - Observe 3 scalars (sumcheck_poly)
        // - Sample 1 scalar (folding_randomness)
        // - PoW: Sample 32 bytes, then Observe 8 bytes (pow-nonce)
        let mut expected = DomainSeparator::<EF4, F, u8>::new("pow");
        expected.observe_scalars(3, "sumcheck_poly");
        expected.sample_scalars(1, "folding_randomness");
        expected.pow(7.);

        assert_eq!(ops, expected.io);
    }

    #[test]
    fn test_add_sumcheck_skip_two_rounds_no_pow() {
        // With univariate skip enabled and skipped_rounds = 2
        let mut ds = DomainSeparator::<EF4, F, u8>::new("skip2");
        ds.add_sumcheck(&SumcheckParams {
            rounds: 3,
            pow_bits: 0.,
            univariate_skip: Some(2),
        });
        let ops = ds.io;

        // Expected:
        // - One LDE observation of 2^(2+1) = 8 scalars
        // - One challenge for folding randomness
        // - Then one regular round of sumcheck: (3 + 1 scalars)
        let mut expected = DomainSeparator::<EF4, F, u8>::new("skip2");
        expected.observe_scalars(8, "sumcheck_poly_skip");
        expected.sample_scalars(1, "folding_randomness_skip");

        expected.observe_scalars(3, "sumcheck_poly");
        expected.sample_scalars(1, "folding_randomness");

        assert_eq!(ops, expected.io);
    }

    #[test]
    fn test_add_sumcheck_skip_two_rounds_with_pow() {
        // With univariate skip and PoW active
        let mut ds = DomainSeparator::<EF4, F, u8>::new("skip2pow");
        ds.add_sumcheck(&SumcheckParams {
            rounds: 3,
            pow_bits: 10.,
            univariate_skip: Some(2),
        });
        let ops = ds.io;

        let mut expected = DomainSeparator::<EF4, F, u8>::new("skip2pow");
        expected.observe_scalars(8, "sumcheck_poly_skip");
        expected.sample_scalars(1, "folding_randomness_skip");
        expected.pow(10.);

        expected.observe_scalars(3, "sumcheck_poly");
        expected.sample_scalars(1, "folding_randomness");
        expected.pow(10.);

        assert_eq!(ops, expected.io);
    }

    #[test]
    fn test_add_sumcheck_skip_one_round_behaves_regular() {
        // Skip = 1 is not enough to trigger shortcut logic
        let mut ds = DomainSeparator::<EF4, F, u8>::new("skip1");
        ds.add_sumcheck(&SumcheckParams {
            rounds: 3,
            pow_bits: 0.,
            univariate_skip: None,
        });
        let ops = ds.io;

        let mut expected = DomainSeparator::<EF4, F, u8>::new("skip1");

        // Round 1
        expected.observe_scalars(3, "sumcheck_poly");
        expected.sample_scalars(1, "folding_randomness");

        // Round 2
        expected.observe_scalars(3, "sumcheck_poly");
        expected.sample_scalars(1, "folding_randomness");

        // Round 3
        expected.observe_scalars(3, "sumcheck_poly");
        expected.sample_scalars(1, "folding_randomness");

        assert_eq!(ops, expected.io);
    }
}
