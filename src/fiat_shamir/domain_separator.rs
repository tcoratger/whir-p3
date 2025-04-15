use std::{collections::VecDeque, fmt::Write, marker::PhantomData};

use p3_field::{BasedVectorSpace, ExtensionField, Field, PrimeCharacteristicRing, TwoAdicField};

use super::{
    DefaultHash,
    duplex_sponge::interface::DuplexSpongeInterface,
    errors::DomainSeparatorMismatch,
    utils::{bytes_modp, bytes_uniform_modp},
};
use crate::{
    fiat_shamir::{prover::ProverState, verifier::VerifierState},
    whir::parameters::WhirConfig,
};

/// This is the separator between operations in the IO Pattern
/// and as such is the only forbidden character in labels.
const SEP_BYTE: &str = "\0";

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
pub struct DomainSeparator<H = DefaultHash>
where
    H: DuplexSpongeInterface<u8>,
{
    io: String,
    _hash: PhantomData<H>,
}

impl<H: DuplexSpongeInterface<u8>> DomainSeparator<H> {
    pub const fn from_string(io: String) -> Self {
        Self {
            io,
            _hash: PhantomData,
        }
    }

    /// Create a new DomainSeparator with the domain separator.
    pub fn new(session_identifier: &str) -> Self {
        assert!(
            !session_identifier.contains(SEP_BYTE),
            "Domain separator cannot contain the separator BYTE."
        );
        Self::from_string(session_identifier.to_string())
    }

    /// Absorb `count` native elements.
    pub fn absorb(&mut self, count: usize, label: &str) {
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
        write!(self.io, "A{count}{label}").expect("writing to String cannot fail");
    }

    /// Squeeze `count` native elements.
    pub fn squeeze(&mut self, count: usize, label: &str) {
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

    /// Return the IO Pattern as bytes.
    pub fn as_bytes(&self) -> &[u8] {
        self.io.as_bytes()
    }

    /// Parse the givern IO Pattern into a sequence of [`Op`]'s.
    pub(crate) fn finalize(&self) -> VecDeque<Op> {
        // Guaranteed to succeed as instances are all valid domain_separators
        Self::parse_domsep(self.io.as_bytes())
            .expect("Internal error. Please submit issue to m@orru.net")
    }

    fn parse_domsep(domain_separator: &[u8]) -> Result<VecDeque<Op>, DomainSeparatorMismatch> {
        let mut stack = VecDeque::new();

        // skip the domain separator
        for part in domain_separator
            .split(|&b| b == SEP_BYTE.as_bytes()[0])
            .skip(1)
        {
            let next_id = part[0] as char;
            let next_length = part[1..]
                .iter()
                .take_while(|x| x.is_ascii_digit())
                .fold(0, |acc, x| acc * 10 + (x - b'0') as usize);

            // check that next_length != 0 is performed internally on Op::new
            let next_op = Op::new(next_id, Some(next_length))?;
            stack.push_back(next_op);
        }

        // consecutive calls are merged into one
        match stack.pop_front() {
            None => Ok(stack),
            Some(x) => Self::simplify_stack(VecDeque::from([x]), stack),
        }
    }

    fn simplify_stack(
        mut dst: VecDeque<Op>,
        mut stack: VecDeque<Op>,
    ) -> Result<VecDeque<Op>, DomainSeparatorMismatch> {
        if stack.is_empty() {
            Ok(dst)
        } else {
            // guaranteed never to fail, since:
            assert!(!dst.is_empty() && !stack.is_empty());
            let previous = dst.pop_back().unwrap();
            let next = stack.pop_front().unwrap();

            match (previous, next) {
                (Op::Squeeze(a), Op::Squeeze(b)) => {
                    dst.push_back(Op::Squeeze(a + b));
                    Self::simplify_stack(dst, stack)
                }
                (Op::Absorb(a), Op::Absorb(b)) => {
                    dst.push_back(Op::Absorb(a + b));
                    Self::simplify_stack(dst, stack)
                }
                // (Op::Divide, Op::Divide)
                // is useless but unharmful
                (a, b) => {
                    dst.push_back(a);
                    dst.push_back(b);
                    Self::simplify_stack(dst, stack)
                }
            }
        }
    }

    /// Create an [`crate::ProverState`] instance from the IO Pattern.
    pub fn to_prover_state(&self) -> ProverState<H> {
        self.into()
    }

    /// Create a [`crate::VerifierState`] instance from the IO Pattern and the protocol transcript
    /// (bytes).
    pub fn to_verifier_state<'a>(&self, transcript: &'a [u8]) -> VerifierState<'a, H> {
        VerifierState::new(self, transcript)
    }

    pub fn add_ood<F>(&mut self, num_samples: usize)
    where
        F: Field + ExtensionField<<F as PrimeCharacteristicRing>::PrimeSubfield>,
    {
        if num_samples > 0 {
            self.challenge_scalars::<F>(num_samples, "ood_query");
            self.add_scalars::<F>(num_samples, "ood_ans");
        }
    }

    pub fn commit_statement<PowStrategy, EF, F, HC, C>(
        &mut self,
        params: &WhirConfig<EF, F, HC, C, PowStrategy>,
    ) where
        F: Field + TwoAdicField,
        EF: ExtensionField<F> + TwoAdicField<PrimeSubfield = F>,
    {
        // TODO: Add params
        self.add_digest("merkle_digest");
        if params.committment_ood_samples > 0 {
            assert!(params.initial_statement);
            self.add_ood::<EF>(params.committment_ood_samples);
        }
    }

    pub fn add_whir_proof<PowStrategy, EF, F, HC, C>(
        &mut self,
        params: &WhirConfig<EF, F, HC, C, PowStrategy>,
    ) where
        F: Field + TwoAdicField,
        EF: ExtensionField<F> + TwoAdicField<PrimeSubfield = F>,
    {
        // TODO: Add statement
        if params.initial_statement {
            self.challenge_scalars::<EF>(1, "initial_combination_randomness");
            self.add_sumcheck::<EF>(
                params.folding_factor.at_round(0),
                params.starting_folding_pow_bits,
            );
        } else {
            self.challenge_scalars::<EF>(params.folding_factor.at_round(0), "folding_randomness");
            self.pow(params.starting_folding_pow_bits);
        }

        let mut domain_size = params.starting_domain.size();
        for (round, r) in params.round_parameters.iter().enumerate() {
            let folded_domain_size = domain_size >> params.folding_factor.at_round(round);
            let domain_size_bytes = ((folded_domain_size * 2 - 1).ilog2() as usize).div_ceil(8);
            self.add_digest("merkle_digest");
            self.add_ood::<EF>(r.ood_samples);
            self.squeeze(r.num_queries * domain_size_bytes, "stir_queries");
            self.pow(r.pow_bits);
            self.challenge_scalars::<EF>(1, "combination_randomness");

            self.add_sumcheck::<EF>(
                params.folding_factor.at_round(round + 1),
                r.folding_pow_bits,
            );
            domain_size >>= 1;
        }

        let folded_domain_size = domain_size
            >> params
                .folding_factor
                .at_round(params.round_parameters.len());
        let domain_size_bytes = ((folded_domain_size * 2 - 1).ilog2() as usize).div_ceil(8);

        self.add_scalars::<EF>(1 << params.final_sumcheck_rounds, "final_coeffs");

        self.squeeze(domain_size_bytes * params.final_queries, "final_queries");
        self.pow(params.final_pow_bits);
        self.add_sumcheck::<EF>(params.final_sumcheck_rounds, params.final_folding_pow_bits);
    }

    pub fn add_digest(&mut self, label: &str) {
        self.absorb(32, label);
    }

    /// Performs `folding_factor` rounds of sumcheck interaction with the transcript.
    ///
    /// In each round:
    /// - Samples 3 scalars for the sumcheck polynomial.
    /// - Samples 1 scalar for folding randomness.
    /// - Optionally performs a PoW challenge if `pow_bits > 0`.
    pub fn add_sumcheck<F>(&mut self, folding_factor: usize, pow_bits: f64)
    where
        F: Field + ExtensionField<<F as PrimeCharacteristicRing>::PrimeSubfield>,
    {
        for _ in 0..folding_factor {
            self.add_scalars::<F>(3, "sumcheck_poly");
            self.challenge_scalars::<F>(1, "folding_randomness");
            self.pow(pow_bits);
        }
    }

    pub fn pow(&mut self, bits: f64) {
        if bits > 0. {
            self.challenge_pow("pow_queries");
        }
    }

    /// Adds a [`PoWChallenge`] to the [`spongefish::DomainSeparator`].
    ///
    /// In order to squeeze a proof-of-work challenge, we extract a 32-byte challenge using
    /// the byte interface, and then we find a 16-byte nonce that satisfies the proof-of-work.
    /// The nonce a 64-bit integer encoded as an unsigned integer and written in big-endian and
    /// added to the protocol transcript as the nonce for the proof-of-work.
    ///
    /// The number of bits used for the proof of work are **not** encoded within the
    /// [`spongefish::DomainSeparator`]. It is up to the implementor to change the domain
    /// separator or the label in order to reflect changes in the proof in order to preserve
    /// simulation extractability.
    pub fn challenge_pow(&mut self, label: &str) {
        // 16 bytes challenge and 16 bytes nonce (that will be written)
        self.squeeze(32, label);
        self.absorb(8, "pow-nonce");
    }

    pub fn add_scalars<F>(&mut self, count: usize, label: &str)
    where
        F: Field + ExtensionField<<F as PrimeCharacteristicRing>::PrimeSubfield>,
    {
        // Absorb `count` scalars into the transcript using an "absorb" tag.
        //
        // The total number of bytes to absorb is calculated as:
        //
        //     count × extension_degree × bytes_modp(bits)
        //
        // where:
        // - `count` is the number of scalar values
        // - `extension_degree` is the number of limbs (e.g., 4 for quartic extensions)
        // - `bytes_modp(bits)` gives the compressed byte length for the prime subfield
        self.absorb(
            count
                * F::DIMENSION
                * F::PrimeSubfield::DIMENSION
                * bytes_modp(F::PrimeSubfield::bits() as u32),
            label,
        );
    }

    pub fn challenge_scalars<F>(&mut self, count: usize, label: &str)
    where
        F: Field + ExtensionField<<F as PrimeCharacteristicRing>::PrimeSubfield>,
    {
        // Squeeze `count` scalars from the transcript using a "challenge" tag.
        //
        // The total number of bytes to squeeze is calculated as:
        //
        //     count × extension_degree × bytes_uniform_modp(bits)
        //
        // where `bytes_uniform_modp` gives the number of bytes needed to sample uniformly
        // over the base field.
        self.squeeze(
            count
                * F::DIMENSION
                * F::PrimeSubfield::DIMENSION
                * bytes_uniform_modp(F::PrimeSubfield::bits() as u32),
            label,
        );
    }
}

/// Sponge operations.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Op {
    /// Indicates absorption of `usize` lanes.
    ///
    /// In a tag, absorb is indicated with 'A'.
    Absorb(usize),
    /// Indicates squeezing of `usize` lanes.
    ///
    /// In a tag, squeeze is indicated with 'S'.
    Squeeze(usize),
    /// Indicates a ratchet operation.
    ///
    /// For sponge functions, we squeeze sizeof(capacity) lanes
    /// and initialize a new state filling the capacity.
    /// This allows for a more efficient preprocessing, and for removal of
    /// private information stored in the rate.
    Ratchet,
}

impl Op {
    /// Create a new OP from the portion of a tag.
    fn new(id: char, count: Option<usize>) -> Result<Self, DomainSeparatorMismatch> {
        match (id, count) {
            ('A', Some(c)) if c > 0 => Ok(Self::Absorb(c)),
            ('R', None | Some(0)) => Ok(Self::Ratchet),
            ('S', Some(c)) if c > 0 => Ok(Self::Squeeze(c)),
            _ => Err("Invalid tag".into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;

    use super::*;

    type H = DefaultHash;
    type F = BabyBear;
    type EF4 = BinomialExtensionField<F, 4>;

    #[test]
    fn test_op_new_invalid_cases() {
        assert!(Op::new('A', Some(0)).is_err()); // absorb with zero
        assert!(Op::new('S', Some(0)).is_err()); // squeeze with zero
        assert!(Op::new('X', Some(1)).is_err()); // invalid op char
        assert!(Op::new('R', Some(5)).is_err()); // R doesn't support > 0
        assert!(Op::new('R', Some(0)).is_ok()); // ratchet with 0
        assert!(Op::new('R', None).is_ok()); // ratchet with None
    }

    #[test]
    fn test_domain_separator_new_and_bytes() {
        let ds = DomainSeparator::<H>::new("session");
        assert_eq!(ds.as_bytes(), b"session");
    }

    #[test]
    #[should_panic]
    fn test_new_with_separator_byte_panics() {
        // This should panic because "\0" is forbidden in the session identifier.
        let _ = DomainSeparator::<H>::new("invalid\0session");
    }

    #[test]
    fn test_domain_separator_absorb_and_squeeze() {
        let mut ds = DomainSeparator::<H>::new("proto");
        ds.absorb(2, "input");
        ds.squeeze(1, "challenge");
        let ops = ds.finalize();
        assert_eq!(ops, vec![Op::Absorb(2), Op::Squeeze(1)]);
    }

    #[test]
    fn test_absorb_return_value_format() {
        let mut ds = DomainSeparator::<H>::new("proto");
        ds.absorb(3, "input");
        let expected_str = "proto\0A3input"; // initial + SEP + absorb op + label
        assert_eq!(ds.as_bytes(), expected_str.as_bytes());
    }

    #[test]
    #[should_panic]
    fn test_absorb_zero_panics() {
        DomainSeparator::<H>::new("x").absorb(0, "label");
    }

    #[test]
    #[should_panic]
    fn test_label_with_separator_byte_panics() {
        DomainSeparator::<H>::new("x").absorb(1, "bad\0label");
    }

    #[test]
    #[should_panic]
    fn test_label_starts_with_digit_panics() {
        DomainSeparator::<H>::new("x").absorb(1, "1label");
    }

    #[test]
    fn test_merge_consecutive_absorbs_and_squeezes() {
        let mut ds = DomainSeparator::<H>::new("merge");
        ds.absorb(1, "a");
        ds.absorb(2, "b");
        ds.squeeze(3, "c");
        ds.squeeze(1, "d");
        let ops = ds.finalize();
        assert_eq!(ops, vec![Op::Absorb(3), Op::Squeeze(4)]);
    }

    #[test]
    fn test_parse_domsep_multiple_ops() {
        let tag = "main\0A1x\0A2y\0S3z\0R\0S2w";
        let ds = DomainSeparator::<H>::from_string(tag.to_string());
        let ops = ds.finalize();
        assert_eq!(
            ops,
            vec![Op::Absorb(3), Op::Squeeze(3), Op::Ratchet, Op::Squeeze(2)]
        );
    }

    #[test]
    fn test_byte_domain_separator_trait_impl() {
        let mut ds = DomainSeparator::<H>::new("x");
        ds.absorb(1, "a");
        ds.squeeze(2, "b");
        let ops = ds.finalize();
        assert_eq!(ops, vec![Op::Absorb(1), Op::Squeeze(2)]);
    }

    #[test]
    fn test_empty_operations() {
        let ds = DomainSeparator::<H>::new("tag");
        let ops = ds.finalize();
        assert!(ops.is_empty());
    }

    #[test]
    fn test_unicode_labels() {
        let mut ds = DomainSeparator::<H>::new("emoji");
        ds.absorb(1, "🦀");
        ds.squeeze(1, "🎯");
        let ops = ds.finalize();
        assert_eq!(ops, vec![Op::Absorb(1), Op::Squeeze(1)]);
    }

    #[test]
    fn test_large_counts_and_labels() {
        let label = "x".repeat(100);
        let mut ds = DomainSeparator::<H>::new("big");
        ds.absorb(12345, &label);
        ds.squeeze(54321, &label);
        let ops = ds.finalize();
        assert_eq!(ops, vec![Op::Absorb(12345), Op::Squeeze(54321)]);
    }

    #[test]
    fn test_malformed_tag_parsing_fails() {
        // Missing count
        let broken = "proto\0Ax";
        let ds = DomainSeparator::<H>::from_string(broken.to_string());
        let res = DomainSeparator::<H>::parse_domsep(ds.as_bytes());
        assert!(res.is_err());
    }

    #[test]
    fn test_simplify_stack_keeps_unlike_ops() {
        let tag = "test\0A2x\0S3y\0A1z";
        let ds = DomainSeparator::<H>::from_string(tag.to_string());
        let ops = ds.finalize();
        assert_eq!(ops, vec![Op::Absorb(2), Op::Squeeze(3), Op::Absorb(1)]);
    }

    #[test]
    fn test_round_trip_operations() {
        let mut ds1 = DomainSeparator::<H>::new("foo");
        ds1.absorb(2, "a");
        ds1.squeeze(3, "b");
        let ops1 = ds1.finalize();

        let tag = std::str::from_utf8(ds1.as_bytes()).unwrap();
        let ds2 = DomainSeparator::<H>::from_string(tag.to_string());
        let ops2 = ds2.finalize();

        assert_eq!(ops1, ops2);
    }

    #[test]
    fn test_squeeze_returns_correct_string() {
        let mut ds = DomainSeparator::<H>::new("proto");
        ds.squeeze(4, "challenge");
        let expected_str = "proto\0S4challenge";
        assert_eq!(ds.as_bytes(), expected_str.as_bytes());
    }

    #[test]
    #[should_panic]
    fn test_squeeze_zero_count_panics() {
        DomainSeparator::<H>::new("proto").squeeze(0, "label");
    }

    #[test]
    #[should_panic]
    fn test_squeeze_label_with_null_byte_panics() {
        DomainSeparator::<H>::new("proto").squeeze(2, "bad\0label");
    }

    #[test]
    #[should_panic]
    fn test_squeeze_label_starts_with_digit_panics() {
        DomainSeparator::<H>::new("proto").squeeze(2, "1invalid");
    }

    #[test]
    fn test_multiple_squeeze_chaining() {
        let mut ds = DomainSeparator::<H>::new("proto");
        ds.squeeze(1, "first");
        ds.squeeze(2, "second");
        let expected_str = "proto\0S1first\0S2second";
        assert_eq!(ds.as_bytes(), expected_str.as_bytes());
    }

    #[test]
    fn test_finalize_mixed_ops_order_preserved() {
        let tag = "zkp\0A1a\0S1b\0A2c\0S3d\0R\0A4e\0S1f";
        let ds = DomainSeparator::<H>::from_string(tag.to_string());
        let ops = ds.finalize();
        assert_eq!(
            ops,
            vec![
                Op::Absorb(1),
                Op::Squeeze(1),
                Op::Absorb(2),
                Op::Squeeze(3),
                Op::Ratchet,
                Op::Absorb(4),
                Op::Squeeze(1),
            ]
        );
    }

    #[test]
    fn test_finalize_large_values_and_merge() {
        let tag = "main\0A5a\0A10b\0S8c\0S2d";
        let ds = DomainSeparator::<H>::from_string(tag.to_string());
        let ops = ds.finalize();
        assert_eq!(ops, vec![Op::Absorb(15), Op::Squeeze(10)]);
    }

    #[test]
    fn test_finalize_merge_and_breaks() {
        let tag = "example\0A2x\0A1y\0R\0A3z\0S4u\0S1v";
        let ds = DomainSeparator::<H>::from_string(tag.to_string());
        let ops = ds.finalize();
        assert_eq!(
            ops,
            vec![Op::Absorb(3), Op::Ratchet, Op::Absorb(3), Op::Squeeze(5),]
        );
    }

    #[test]
    fn test_finalize_only_ratchets() {
        let tag = "onlyratchets\0R\0R\0R";
        let ds = DomainSeparator::<H>::from_string(tag.to_string());
        let ops = ds.finalize();
        assert_eq!(ops, vec![Op::Ratchet, Op::Ratchet, Op::Ratchet]);
    }

    #[test]
    fn test_finalize_complex_merge_boundaries() {
        let tag = "demo\0A1a\0A1b\0S2c\0S2d\0A3e\0S1f";
        let ds = DomainSeparator::<H>::from_string(tag.to_string());
        let ops = ds.finalize();
        assert_eq!(
            ops,
            vec![
                Op::Absorb(2),  // A1a + A1b
                Op::Squeeze(4), // S2c + S2d
                Op::Absorb(3),  // A3e
                Op::Squeeze(1), // S1f
            ]
        );
    }

    #[test]
    fn test_add_scalars_babybear() {
        // Test absorption of scalar field elements (BabyBear).
        // - BabyBear is a base field with extension degree = 1
        // - bits = 31 → bytes_modp(31) = 4
        // - 2 scalars * 1 * 4 = 8 bytes absorbed
        // - "A" indicates absorption in the domain separator
        let mut domsep: DomainSeparator<H> = DomainSeparator::new("babybear");
        domsep.add_scalars::<F>(2, "foo");
        let expected = b"babybear\0A8foo";
        assert_eq!(domsep.as_bytes(), expected);
    }

    #[test]
    fn test_challenge_scalars_babybear() {
        // Test squeezing scalar field elements (BabyBear).
        // - BabyBear has extension degree = 1
        // - bits = 31 → bytes_uniform_modp(31) = 5
        // - 3 scalars * 1 * 5 = 15 bytes squeezed
        // - "S" indicates squeezing in the domain separator
        let mut domsep: DomainSeparator<H> = DomainSeparator::new("bb");
        domsep.challenge_scalars::<F>(3, "bar");
        let expected = b"bb\0S57bar";
        assert_eq!(domsep.as_bytes(), expected);
    }

    #[test]
    fn test_add_scalars_quartic_ext_field() {
        // Test absorption of scalars from a quartic extension field EF4.
        // - EF4 has extension degree = 4
        // - Base field bits = 31 → bytes_modp(31) = 4
        // - 2 scalars * 4 * 4 = 32 bytes absorbed
        let mut domsep: DomainSeparator<H> = DomainSeparator::new("ext");
        domsep.add_scalars::<EF4>(2, "a");
        let expected = b"ext\0A32a";
        assert_eq!(domsep.as_bytes(), expected);
    }

    #[test]
    fn test_challenge_scalars_quartic_ext_field() {
        // Test squeezing of scalars from a quartic extension field EF4.
        // - EF4 has extension degree = 4
        // - Base field bits = 31 → bytes_uniform_modp(31) = 19
        // - 1 scalar * 4 * 19 = 76 bytes squeezed
        // - "S" indicates squeezing in the domain separator
        let mut domsep: DomainSeparator<H> = DomainSeparator::new("ext2");
        domsep.challenge_scalars::<EF4>(1, "b");

        let expected = b"ext2\0S76b";
        assert_eq!(domsep.as_bytes(), expected);
    }

    #[test]
    fn test_add_ood() {
        let iop: DomainSeparator = DomainSeparator::new("test_protocol");
        let mut updated_iop = iop.clone();
        let mut unchanged_iop = iop;

        // Apply OOD query addition
        updated_iop.add_ood::<BabyBear>(3);

        // Convert to a string for inspection
        let pattern_str = String::from_utf8(updated_iop.as_bytes().to_vec()).unwrap();

        // Check if "ood_query" and "ood_ans" were correctly appended
        assert!(pattern_str.contains("ood_query"));
        assert!(pattern_str.contains("ood_ans"));

        // Test case where num_samples = 0 (should not modify anything)
        unchanged_iop.add_ood::<BabyBear>(0);
        let unchanged_str = String::from_utf8(unchanged_iop.as_bytes().to_vec()).unwrap();
        assert_eq!(unchanged_str, "test_protocol"); // Should remain the same
    }

    #[test]
    fn test_pow() {
        let iop: DomainSeparator = DomainSeparator::new("test_protocol");
        let mut updated_iop = iop.clone();
        let mut unchanged_iop = iop;

        // Apply PoW challenge
        updated_iop.pow(10.0);

        // Convert to a string for inspection
        let pattern_str = String::from_utf8(updated_iop.as_bytes().to_vec()).unwrap();

        // Check if "pow_queries" was correctly added
        assert!(pattern_str.contains("pow_queries"));

        // Test case where bits = 0 (should not modify anything)
        unchanged_iop.pow(0.0);
        let unchanged_str = String::from_utf8(unchanged_iop.as_bytes().to_vec()).unwrap();
        assert_eq!(unchanged_str, "test_protocol"); // Should remain the same
    }
}
