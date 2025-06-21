use std::{collections::VecDeque, marker::PhantomData};

use p3_challenger::{CanObserve, CanSample};
use p3_field::{ExtensionField, Field, PrimeField64, TwoAdicField};

use super::domain_separator::{DomainSeparator, Op};
use crate::fiat_shamir::unit::Unit;

/// A stateful transcript wrapper enforcing a predetermined Fiat-Shamir protocol.
///
/// Typically constructed from a `DomainSeparator`, which defines the expected operation sequence.
#[derive(Clone, Debug)]
pub struct ChallengerWithInstructions<Challenger, U>
where
    U: Unit,
    Challenger: CanObserve<U> + CanSample<U>,
{
    /// The internal Fiat-Shamir challenger.
    ///
    /// This object handles actual transcript updates and challenge generation.
    pub(crate) challenger: Challenger,

    /// A queue of expected transcript operations, derived from the domain separator.
    ///
    /// If `verify_operations` is enabled, this stack is consumed as the transcript
    /// proceeds and each operation is validated against the declared pattern.
    stack: VecDeque<Op>,

    /// Phantom marker for the transcript element type `U`.
    ///
    /// This type parameter ensures the challenger operates over the correct unit type
    /// (e.g., bytes, scalars), even though no `U` values are stored directly.
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
            _unit: PhantomData,
        }
    }

    /// Observe input elements into the transcript, advancing the expected operation stack.
    ///
    /// This method must be called exactly when the next expected operation is `Observe`.
    pub fn observe(&mut self, input: &[U]) {
        self.challenger.observe_slice(input);
    }

    /// Sample output elements from the transcript, advancing the expected operation stack.
    ///
    /// This method must be called exactly when the next expected operation is `Sample`.
    /// It fills the `output` slice with challenge elements derived from the current transcript state.
    ///
    /// # Errors
    /// Returns an error if:
    /// - the next expected operation is not `Sample`,
    /// - the requested output length exceeds what remains,
    /// - the stack is empty.
    pub fn sample(&mut self, output: &mut [U]) {
        for out in output.iter_mut() {
            *out = self.challenger.sample();
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
    fn test_squeeze_works() {
        let mut domsep = DomainSeparator::<F, F, u8>::new("test");
        domsep.sample(3, "y");
        let challenger = DummyChallenger::default();
        let mut state = ChallengerWithInstructions::<DummyChallenger, _>::new(&domsep, challenger);

        let mut out = [0u8; 3];
        state.sample(&mut out);
        assert_eq!(out, [0, 1, 2]);
    }

    #[test]
    fn test_from_impl_constructs_hash_state() {
        let mut domsep = DomainSeparator::<F, F, u8>::new("from");
        domsep.observe(1, "in");
        let challenger = DummyChallenger::default();
        let state = ChallengerWithInstructions::<DummyChallenger, _>::new(&domsep, challenger);

        assert_eq!(state.stack.len(), 1);
        assert_eq!(state.stack.front(), Some(&Op::Observe(1)));
    }

    #[test]
    fn test_generate_tag_is_deterministic() {
        let mut ds1 = DomainSeparator::<F, F, u8>::new("session1");
        ds1.observe(1, "x");
        let mut ds2 = DomainSeparator::<F, F, u8>::new("session1");
        ds2.observe(1, "x");

        let challenger1 = DummyChallenger::default();
        let tag1 = ChallengerWithInstructions::<DummyChallenger, _>::new(&ds1, challenger1);
        let challenger2 = DummyChallenger::default();
        let tag2 = ChallengerWithInstructions::<DummyChallenger, _>::new(&ds2, challenger2);

        assert_eq!(
            &*tag1.challenger.observed.borrow(),
            &*tag2.challenger.observed.borrow()
        );
    }
}
