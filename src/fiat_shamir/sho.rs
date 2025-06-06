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
pub struct HashStateWithInstructions<H, U>
where
    U: Unit,
    H: CanObserve<U> + CanSample<U>,
{
    /// The internal duplex sponge used for absorbing and squeezing data.
    pub(crate) ds: H,
    /// A stack of expected sponge operations.
    stack: VecDeque<Op>,
    /// Marker for the unit type `U`.
    _unit: PhantomData<U>,
}

impl<H, U> HashStateWithInstructions<H, U>
where
    U: Unit + Default + Copy,
    H: CanObserve<U> + CanSample<U>,
{
    /// Initialise a stateful hash object,
    /// setting up the state of the sponge function and parsing the tag string.
    #[must_use]
    pub fn new<EF, F>(domain_separator: &DomainSeparator<EF, F, U>, mut challenger: H) -> Self
    where
        EF: ExtensionField<F> + TwoAdicField,
        F: Field + TwoAdicField + PrimeField64,
    {
        let stack = domain_separator.finalize();
        let iop_units = domain_separator.as_units();
        challenger.observe_slice(&iop_units);

        Self {
            ds: challenger,
            stack,
            _unit: PhantomData,
        }
    }

    /// Perform secure absorption of the elements in `input`.
    ///
    /// Absorb calls can be batched together, or provided separately for streaming-friendly
    /// protocols.
    pub fn absorb(&mut self, input: &[U]) -> Result<(), DomainSeparatorMismatch> {
        match self.stack.pop_front() {
            Some(Op::Absorb(length)) if length >= input.len() => {
                if length > input.len() {
                    self.stack.push_front(Op::Absorb(length - input.len()));
                }
                self.ds.observe_slice(input);
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

    /// Perform a secure squeeze operation, filling the output buffer with uniformly random bytes.
    ///
    /// For byte-oriented sponges, this operation is equivalent to the squeeze operation.
    /// However, for algebraic hashes, this operation is non-trivial.
    /// This function provides no guarantee of streaming-friendliness.
    pub fn squeeze(&mut self, output: &mut [U]) -> Result<(), DomainSeparatorMismatch> {
        match self.stack.pop_front() {
            Some(Op::Squeeze(length)) if output.len() <= length => {
                for out in output.iter_mut() {
                    *out = self.ds.sample();
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
        let mut domsep = DomainSeparator::<F, F, u8>::new("test");
        domsep.absorb(2, "x");
        let challenger = DummyChallenger::default();
        let mut state = HashStateWithInstructions::<DummyChallenger, _>::new(&domsep, challenger);

        assert_eq!(state.stack.len(), 1);

        let result = state.absorb(&[1, 2]);
        assert!(result.is_ok());

        assert_eq!(state.stack.len(), 0);
    }

    #[test]
    fn test_absorb_too_much_returns_error() {
        let mut domsep = DomainSeparator::<F, F, u8>::new("test");
        domsep.absorb(2, "x");
        let challenger = DummyChallenger::default();
        let mut state = HashStateWithInstructions::<DummyChallenger, _>::new(&domsep, challenger);

        let result = state.absorb(&[1, 2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_squeeze_works() {
        let mut domsep = DomainSeparator::<F, F, u8>::new("test");
        domsep.squeeze(3, "y");
        let challenger = DummyChallenger::default();
        let mut state = HashStateWithInstructions::<DummyChallenger, _>::new(&domsep, challenger);

        let mut out = [0u8; 3];
        let result = state.squeeze(&mut out);
        assert!(result.is_ok());
        assert_eq!(out, [0, 1, 2]);
    }

    #[test]
    fn test_squeeze_with_leftover_updates_stack() {
        let mut domsep = DomainSeparator::<F, F, u8>::new("test");
        domsep.squeeze(4, "z");
        let challenger = DummyChallenger::default();
        let mut state = HashStateWithInstructions::<DummyChallenger, _>::new(&domsep, challenger);

        let mut out = [0u8; 2];
        let result = state.squeeze(&mut out);
        assert!(result.is_ok());

        assert_eq!(state.stack.front(), Some(&Op::Squeeze(2)));
    }

    #[test]
    fn test_multiple_absorbs_deplete_stack_properly() {
        let mut domsep = DomainSeparator::<F, F, u8>::new("test");
        domsep.absorb(5, "a");
        let challenger = DummyChallenger::default();
        let mut state = HashStateWithInstructions::<DummyChallenger, _>::new(&domsep, challenger);

        let res1 = state.absorb(&[1, 2]);
        assert!(res1.is_ok());
        assert_eq!(state.stack.front(), Some(&Op::Absorb(3)));

        let res2 = state.absorb(&[3, 4, 5]);
        assert!(res2.is_ok());
        assert!(state.stack.is_empty());
    }

    #[test]
    fn test_multiple_squeeze_deplete_stack_properly() {
        let mut domsep = DomainSeparator::<F, F, u8>::new("test");
        domsep.squeeze(5, "z");
        let challenger = DummyChallenger::default();
        let mut state = HashStateWithInstructions::<DummyChallenger, _>::new(&domsep, challenger);

        let mut out1 = [0u8; 2];
        assert!(state.squeeze(&mut out1).is_ok());
        assert_eq!(state.stack.front(), Some(&Op::Squeeze(3)));

        let mut out2 = [0u8; 3];
        assert!(state.squeeze(&mut out2).is_ok());
        assert!(state.stack.is_empty());
    }

    #[test]
    fn test_absorb_then_wrong_squeeze_clears_stack() {
        let mut domsep = DomainSeparator::<F, F, u8>::new("test");
        domsep.absorb(3, "in");
        let challenger = DummyChallenger::default();
        let mut state = HashStateWithInstructions::<DummyChallenger, _>::new(&domsep, challenger);

        let mut out = [0u8; 1];
        let result = state.squeeze(&mut out);
        assert!(result.is_err());
        assert!(state.stack.is_empty());
    }

    #[test]
    fn test_absorb_exact_then_too_much() {
        let mut domsep = DomainSeparator::<F, F, u8>::new("test");
        domsep.absorb(2, "x");
        let challenger = DummyChallenger::default();
        let mut state = HashStateWithInstructions::<DummyChallenger, _>::new(&domsep, challenger);

        assert!(state.absorb(&[10, 20]).is_ok());
        assert!(state.absorb(&[30]).is_err()); // no ops left
        assert!(state.stack.is_empty());
    }

    #[test]
    fn test_from_impl_constructs_hash_state() {
        let mut domsep = DomainSeparator::<F, F, u8>::new("from");
        domsep.absorb(1, "in");
        let challenger = DummyChallenger::default();
        let state = HashStateWithInstructions::<DummyChallenger, _>::new(&domsep, challenger);

        assert_eq!(state.stack.len(), 1);
        assert_eq!(state.stack.front(), Some(&Op::Absorb(1)));
    }

    #[test]
    fn test_generate_tag_is_deterministic() {
        let mut ds1 = DomainSeparator::<F, F, u8>::new("session1");
        ds1.absorb(1, "x");
        let mut ds2 = DomainSeparator::<F, F, u8>::new("session1");
        ds2.absorb(1, "x");

        let challenger1 = DummyChallenger::default();
        let tag1 = HashStateWithInstructions::<DummyChallenger, _>::new(&ds1, challenger1);
        let challenger2 = DummyChallenger::default();
        let tag2 = HashStateWithInstructions::<DummyChallenger, _>::new(&ds2, challenger2);

        assert_eq!(&*tag1.ds.observed.borrow(), &*tag2.ds.observed.borrow());
    }

    #[test]
    fn test_hint_works_and_removes_stack_entry() {
        let mut domsep = DomainSeparator::<F, F, u8>::new("test");
        domsep.hint("hint");
        let challenger = DummyChallenger::default();
        let mut state = HashStateWithInstructions::<DummyChallenger, _>::new(&domsep, challenger);

        assert_eq!(state.stack.len(), 1);
        let result = state.hint();
        assert!(result.is_ok());
        assert!(state.stack.is_empty());
    }

    #[test]
    fn test_hint_wrong_op_errors_and_clears_stack() {
        let mut domsep = DomainSeparator::<F, F, u8>::new("test");
        domsep.absorb(1, "x");
        let challenger = DummyChallenger::default();
        let mut state = HashStateWithInstructions::<DummyChallenger, _>::new(&domsep, challenger);

        let result = state.hint(); // Should expect Op::Hint, but see Op::Absorb
        assert!(result.is_err());
        assert!(state.stack.is_empty());
    }

    #[test]
    fn test_hint_on_empty_stack_errors() {
        let domsep = DomainSeparator::<F, F, u8>::new("test");
        let challenger = DummyChallenger::default();
        let mut state = HashStateWithInstructions::<DummyChallenger, _>::new(&domsep, challenger);

        let result = state.hint(); // Stack is empty
        assert!(result.is_err());
    }
}
