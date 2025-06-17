use std::{collections::VecDeque, marker::PhantomData};

use p3_challenger::{CanObserve, CanSample};
use p3_field::{ExtensionField, Field, PrimeField64, TwoAdicField};

use super::{
    domain_separator::{DomainSeparator, Op},
    errors::DomainSeparatorMismatch,
};
use crate::fiat_shamir::unit::Unit;

/// A stateful hash object that interfaces with duplex interfaces.
#[derive(Clone, Debug)]
pub struct ChallengerWithInstructions<Challenger, U>
where
    U: Unit,
    Challenger: CanObserve<U> + CanSample<U>,
{
    /// The internal challenger.
    pub(crate) challenger: Challenger,
    /// A stack of expected sponge operations.
    stack: VecDeque<Op>,
    /// At the beginning of the protocol, the list of Fiat-Shamir operations
    /// (absorb, squeeze, hint) must be declared, in order to build the Domain Separator.
    /// This ensures that even a slightly different protocol will start with a different
    /// random state (to avoid replay attacks).
    /// When `verify_operations` is set to true, these operations are verified at runtime.
    /// This is, in theory, redondant, as long as the domain separator is built correctly.
    /// To make the development of a new PIOP easier, we can disable this check.
    verify_operations: bool,
    /// Marker for the unit type `U`.
    _unit: PhantomData<U>,
}

impl<Challenger, U> ChallengerWithInstructions<Challenger, U>
where
    U: Unit + Default + Copy,
    Challenger: CanObserve<U> + CanSample<U>,
{
    /// Create a new Fiat-Shamir transcript state from a domain separator and challenger.
    ///
    /// This initializes the internal transcript with the serialized domain separator,
    /// sets up the expected operation sequence (`stack`), and enforces optional runtime
    /// validation of operation order via `verify_operations`.
    pub fn new<EF, F>(
        domain_separator: &DomainSeparator<EF, F, U>,
        mut challenger: Challenger,
        verify_operations: bool,
    ) -> Self
    where
        EF: ExtensionField<F> + TwoAdicField,
        F: Field + TwoAdicField + PrimeField64,
    {
        let stack = domain_separator.finalize();
        let iop_units = domain_separator.as_units();
        challenger.observe_slice(&iop_units);

        Self {
            challenger,
            stack,
            verify_operations,
            _unit: PhantomData,
        }
    }

    /// Observe input elements into the transcript, advancing the expected operation stack.
    ///
    /// This method must be called exactly when the next expected operation is `Absorb`.
    /// If `verify_operations` is enabled, the input length must match the declared absorb length.
    ///
    /// # Errors
    /// Returns an error if:
    /// - the next expected operation is not `Absorb`,
    /// - the input length exceeds the expected absorb length,
    /// - the stack is empty.
    pub fn observe(&mut self, input: &[U]) -> Result<(), DomainSeparatorMismatch> {
        if !self.verify_operations {
            self.challenger.observe_slice(input);
            return Ok(());
        }
        match self.stack.pop_front() {
            Some(Op::Absorb(length)) if length >= input.len() => {
                if length > input.len() {
                    self.stack.push_front(Op::Absorb(length - input.len()));
                }
                self.challenger.observe_slice(input);
                Ok(())
            }
            None => {
                self.stack.clear();
                Err(format!(
                    "Invalid tag. Stack empty, got {:?}",
                    Op::Absorb(input.len())
                )
                .into())
            }
            Some(op) => {
                self.stack.clear();
                Err(format!(
                    "Invalid tag. Got {:?}, expected {:?}",
                    Op::Absorb(input.len()),
                    op
                )
                .into())
            }
        }
    }

    /// Sample output elements from the transcript, advancing the expected operation stack.
    ///
    /// This method must be called exactly when the next expected operation is `Squeeze`.
    /// It fills the `output` slice with challenge elements derived from the current transcript state.
    ///
    /// # Errors
    /// Returns an error if:
    /// - the next expected operation is not `Squeeze`,
    /// - the requested output length exceeds what remains,
    /// - the stack is empty.
    pub fn sample(&mut self, output: &mut [U]) -> Result<(), DomainSeparatorMismatch> {
        if !self.verify_operations {
            for out in output.iter_mut() {
                *out = self.challenger.sample();
            }
            return Ok(());
        }
        match self.stack.pop_front() {
            Some(Op::Squeeze(length)) if output.len() <= length => {
                for out in output.iter_mut() {
                    *out = self.challenger.sample();
                }
                if length != output.len() {
                    self.stack.push_front(Op::Squeeze(length - output.len()));
                }
                Ok(())
            }
            None => {
                self.stack.clear();
                Err(format!(
                    "Invalid tag. Stack empty, got {:?}",
                    Op::Squeeze(output.len())
                )
                .into())
            }
            Some(op) => {
                self.stack.clear();
                Err(format!(
                    "Invalid tag. Got {:?}, expected {:?}. The stack remaining is: {:?}",
                    Op::Squeeze(output.len()),
                    op,
                    self.stack
                )
                .into())
            }
        }
    }

    /// Send or receive a hint from the proof stream.
    pub fn hint(&mut self) -> Result<(), DomainSeparatorMismatch> {
        if !self.verify_operations {
            return Ok(());
        }
        match self.stack.pop_front() {
            Some(Op::Hint) => Ok(()),
            Some(op) => Err(format!("Invalid tag. Got Op::Hint, expected {op:?}",).into()),
            None => Err(format!("Invalid tag. Stack empty, got {:?}", Op::Hint).into()),
        }
    }
}

#[cfg(test)]
#[allow(clippy::bool_assert_comparison)]
mod tests {
    use std::cell::RefCell;

    use p3_baby_bear::BabyBear;

    use super::*;

    type F = BabyBear;

    /// Minimal challenger that mimics a sponge for Fiat-Shamir tests.
    #[derive(Default, Clone)]
    struct DummyChallenger {
        pub observed: RefCell<Vec<u8>>,
        pub counter: RefCell<u8>,
    }

    impl CanObserve<u8> for DummyChallenger {
        fn observe(&mut self, value: u8) {
            self.observed.borrow_mut().push(value);
        }
    }

    impl CanSample<u8> for DummyChallenger {
        fn sample(&mut self) -> u8 {
            let mut counter = self.counter.borrow_mut();
            let out = *counter;
            *counter = counter.wrapping_add(1);
            out
        }
    }

    #[test]
    fn test_absorb_works_and_modifies_stack() {
        let mut domsep = DomainSeparator::<F, F, u8>::new("test", true);
        domsep.observe(2, "x");
        let challenger = DummyChallenger::default();
        let mut state =
            ChallengerWithInstructions::<DummyChallenger, _>::new(&domsep, challenger, true);

        assert_eq!(state.stack.len(), 1);

        let result = state.observe(&[1, 2]);
        assert!(result.is_ok());

        assert_eq!(state.stack.len(), 0);
    }

    #[test]
    fn test_absorb_too_much_returns_error() {
        let mut domsep = DomainSeparator::<F, F, u8>::new("test", true);
        domsep.observe(2, "x");
        let challenger = DummyChallenger::default();
        let mut state =
            ChallengerWithInstructions::<DummyChallenger, _>::new(&domsep, challenger, true);

        let result = state.observe(&[1, 2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_squeeze_works() {
        let mut domsep = DomainSeparator::<F, F, u8>::new("test", true);
        domsep.sample(3, "y");
        let challenger = DummyChallenger::default();
        let mut state =
            ChallengerWithInstructions::<DummyChallenger, _>::new(&domsep, challenger, true);

        let mut out = [0u8; 3];
        let result = state.sample(&mut out);
        assert!(result.is_ok());
        assert_eq!(out, [0, 1, 2]);
    }

    #[test]
    fn test_squeeze_with_leftover_updates_stack() {
        let mut domsep = DomainSeparator::<F, F, u8>::new("test", true);
        domsep.sample(4, "z");
        let challenger = DummyChallenger::default();
        let mut state =
            ChallengerWithInstructions::<DummyChallenger, _>::new(&domsep, challenger, true);

        let mut out = [0u8; 2];
        let result = state.sample(&mut out);
        assert!(result.is_ok());

        assert_eq!(state.stack.front(), Some(&Op::Squeeze(2)));
    }

    #[test]
    fn test_multiple_absorbs_deplete_stack_properly() {
        let mut domsep = DomainSeparator::<F, F, u8>::new("test", true);
        domsep.observe(5, "a");
        let challenger = DummyChallenger::default();
        let mut state =
            ChallengerWithInstructions::<DummyChallenger, _>::new(&domsep, challenger, true);

        let res1 = state.observe(&[1, 2]);
        assert!(res1.is_ok());
        assert_eq!(state.stack.front(), Some(&Op::Absorb(3)));

        let res2 = state.observe(&[3, 4, 5]);
        assert!(res2.is_ok());
        assert!(state.stack.is_empty());
    }

    #[test]
    fn test_multiple_squeeze_deplete_stack_properly() {
        let mut domsep = DomainSeparator::<F, F, u8>::new("test", true);
        domsep.sample(5, "z");
        let challenger = DummyChallenger::default();
        let mut state =
            ChallengerWithInstructions::<DummyChallenger, _>::new(&domsep, challenger, true);

        let mut out1 = [0u8; 2];
        assert!(state.sample(&mut out1).is_ok());
        assert_eq!(state.stack.front(), Some(&Op::Squeeze(3)));

        let mut out2 = [0u8; 3];
        assert!(state.sample(&mut out2).is_ok());
        assert!(state.stack.is_empty());
    }

    #[test]
    fn test_absorb_then_wrong_squeeze_clears_stack() {
        let mut domsep = DomainSeparator::<F, F, u8>::new("test", true);
        domsep.observe(3, "in");
        let challenger = DummyChallenger::default();
        let mut state =
            ChallengerWithInstructions::<DummyChallenger, _>::new(&domsep, challenger, true);

        let mut out = [0u8; 1];
        let result = state.sample(&mut out);
        assert!(result.is_err());
        assert!(state.stack.is_empty());
    }

    #[test]
    fn test_absorb_exact_then_too_much() {
        let mut domsep = DomainSeparator::<F, F, u8>::new("test", true);
        domsep.observe(2, "x");
        let challenger = DummyChallenger::default();
        let mut state =
            ChallengerWithInstructions::<DummyChallenger, _>::new(&domsep, challenger, true);

        assert!(state.observe(&[10, 20]).is_ok());
        assert!(state.observe(&[30]).is_err()); // no ops left
        assert!(state.stack.is_empty());
    }

    #[test]
    fn test_from_impl_constructs_hash_state() {
        let mut domsep = DomainSeparator::<F, F, u8>::new("from", true);
        domsep.observe(1, "in");
        let challenger = DummyChallenger::default();
        let state =
            ChallengerWithInstructions::<DummyChallenger, _>::new(&domsep, challenger, true);

        assert_eq!(state.stack.len(), 1);
        assert_eq!(state.stack.front(), Some(&Op::Absorb(1)));
    }

    #[test]
    fn test_generate_tag_is_deterministic() {
        let mut ds1 = DomainSeparator::<F, F, u8>::new("session1", true);
        ds1.observe(1, "x");
        let mut ds2 = DomainSeparator::<F, F, u8>::new("session1", true);
        ds2.observe(1, "x");

        let challenger1 = DummyChallenger::default();
        let tag1 = ChallengerWithInstructions::<DummyChallenger, _>::new(&ds1, challenger1, true);
        let challenger2 = DummyChallenger::default();
        let tag2 = ChallengerWithInstructions::<DummyChallenger, _>::new(&ds2, challenger2, true);

        assert_eq!(
            &*tag1.challenger.observed.borrow(),
            &*tag2.challenger.observed.borrow()
        );
    }

    #[test]
    fn test_hint_works_and_removes_stack_entry() {
        let mut domsep = DomainSeparator::<F, F, u8>::new("test", true);
        domsep.hint("hint");
        let challenger = DummyChallenger::default();
        let mut state =
            ChallengerWithInstructions::<DummyChallenger, _>::new(&domsep, challenger, true);

        assert_eq!(state.stack.len(), 1);
        let result = state.hint();
        assert!(result.is_ok());
        assert!(state.stack.is_empty());
    }

    #[test]
    fn test_hint_wrong_op_errors_and_clears_stack() {
        let mut domsep = DomainSeparator::<F, F, u8>::new("test", true);
        domsep.observe(1, "x");
        let challenger = DummyChallenger::default();
        let mut state =
            ChallengerWithInstructions::<DummyChallenger, _>::new(&domsep, challenger, true);

        let result = state.hint(); // Should expect Op::Hint, but see Op::Absorb
        assert!(result.is_err());
        assert!(state.stack.is_empty());
    }

    #[test]
    fn test_hint_on_empty_stack_errors() {
        let domsep = DomainSeparator::<F, F, u8>::new("test", true);
        let challenger = DummyChallenger::default();
        let mut state =
            ChallengerWithInstructions::<DummyChallenger, _>::new(&domsep, challenger, true);

        let result = state.hint(); // Stack is empty
        assert!(result.is_err());
    }
}
