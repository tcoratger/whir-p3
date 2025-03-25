use std::{collections::VecDeque, marker::PhantomData};

use super::{
    DefaultHash, DefaultRng,
    duplex_sponge::{Unit, interface::DuplexSpongeInterface},
    errors::DomainSeparatorMismatch,
    traits::ByteDomainSeparator,
};
use crate::fiat_shamir::{prover::ProverState, verifier::VerifierState};

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
#[derive(Clone)]
pub struct DomainSeparator<H = DefaultHash, U = u8>
where
    U: Unit,
    H: DuplexSpongeInterface<U>,
{
    io: String,
    _hash: PhantomData<(H, U)>,
}

impl<H: DuplexSpongeInterface<U>, U: Unit> DomainSeparator<H, U> {
    pub fn from_string(io: String) -> Self {
        Self { io, _hash: PhantomData }
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
    pub fn absorb(self, count: usize, label: &str) -> Self {
        assert!(count > 0, "Count must be positive.");
        assert!(!label.contains(SEP_BYTE), "Label cannot contain the separator BYTE.");
        assert!(
            match label.chars().next() {
                Some(char) => !char.is_ascii_digit(),
                None => true,
            },
            "Label cannot start with a digit."
        );

        Self::from_string(self.io + SEP_BYTE + &format!("A{}", count) + label)
    }

    /// Squeeze `count` native elements.
    pub fn squeeze(self, count: usize, label: &str) -> Self {
        assert!(count > 0, "Count must be positive.");
        assert!(!label.contains(SEP_BYTE), "Label cannot contain the separator BYTE.");
        assert!(
            match label.chars().next() {
                Some(char) => !char.is_ascii_digit(),
                None => true,
            },
            "Label cannot start with a digit."
        );

        Self::from_string(self.io + SEP_BYTE + &format!("S{}", count) + label)
    }

    /// Ratchet the state.
    pub fn ratchet(self) -> Self {
        Self::from_string(self.io + SEP_BYTE + "R")
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
        for part in domain_separator.split(|&b| b == SEP_BYTE.as_bytes()[0]).skip(1) {
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
    pub fn to_prover_state(&self) -> ProverState<H, U, DefaultRng> {
        self.into()
    }

    /// Create a [`crate::VerifierState`] instance from the IO Pattern and the protocol transcript
    /// (bytes).
    pub fn to_verifier_state<'a>(&self, transcript: &'a [u8]) -> VerifierState<'a, H, U> {
        VerifierState::new(self, transcript)
    }
}

impl<U: Unit, H: DuplexSpongeInterface<U>> core::fmt::Debug for DomainSeparator<H, U> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Ensure that the state isn't accidentally logged
        write!(f, "DomainSeparator({:?})", self.io)
    }
}

impl<H: DuplexSpongeInterface> ByteDomainSeparator for DomainSeparator<H> {
    #[inline]
    fn add_bytes(self, count: usize, label: &str) -> Self {
        self.absorb(count, label)
    }

    #[inline]
    fn challenge_bytes(self, count: usize, label: &str) -> Self {
        self.squeeze(count, label)
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
            ('A', Some(c)) if c > 0 => Ok(Op::Absorb(c)),
            ('R', None) | ('R', Some(0)) => Ok(Op::Ratchet),
            ('S', Some(c)) if c > 0 => Ok(Op::Squeeze(c)),
            _ => Err("Invalid tag".into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use rand::TryRngCore;

    use super::*;
    use crate::fiat_shamir::{keccak::Keccak, traits::ByteWriter};

    #[test]
    fn test_domain_separator() {
        // test that the byte separator is always added
        let domain_separator = DomainSeparator::<Keccak>::new("example.com");
        assert!(domain_separator.as_bytes().starts_with(b"example.com"));
    }

    #[test]
    fn test_prover_rng_basic() {
        let domain_separator = DomainSeparator::<Keccak>::new("example.com");
        let mut prover_state = domain_separator.to_prover_state();
        let rng = prover_state.rng();

        let mut random_bytes = [0u8; 32];
        rng.try_fill_bytes(&mut random_bytes).unwrap();
        let random_u32 = rng.try_next_u32().unwrap();
        let random_u64 = rng.try_next_u64().unwrap();
        assert_ne!(random_bytes, [0u8; 32]);
        assert_ne!(random_u32, 0);
        assert_ne!(random_u64, 0);
        assert!(random_bytes.iter().any(|&x| x != random_bytes[0]));
    }

    #[test]
    fn test_prover_state_bytewriter() {
        let domain_separator = DomainSeparator::<Keccak>::new("example.com").absorb(1, "🥕");
        let mut prover_state = domain_separator.to_prover_state();
        assert!(prover_state.add_bytes(&[0u8]).is_ok());
        assert!(prover_state.add_bytes(&[1u8]).is_err());
        assert_eq!(prover_state.narg_string(), b"\0", "Protocol Transcript survives errors");

        let mut prover_state = domain_separator.to_prover_state();
        // TODO: reactivate this test if needed
        // assert!(prover_state.public_bytes(&[0u8]).is_ok());
        assert_eq!(prover_state.narg_string(), b"");
    }
}
