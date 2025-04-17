use std::{collections::VecDeque, marker::PhantomData};

use p3_field::{ExtensionField, Field, PrimeField64, TwoAdicField};

use super::{
    domain_separator::{DomainSeparator, Op},
    duplex_sponge::interface::DuplexSpongeInterface,
    errors::DomainSeparatorMismatch,
    keccak::Keccak,
};

/// A stateful hash object that interfaces with duplex interfaces.
#[derive(Clone, Debug)]
pub struct HashStateWithInstructions<H>
where
    H: DuplexSpongeInterface<u8>,
{
    /// The internal duplex sponge used for absorbing and squeezing data.
    pub(crate) ds: H,
    /// A stack of expected sponge operations.
    stack: VecDeque<Op>,
    /// Marker to associate the unit type `U` without storing a value.
    _unit: PhantomData<u8>,
}

impl<H: DuplexSpongeInterface<u8>> HashStateWithInstructions<H> {
    /// Initialise a stateful hash object,
    /// setting up the state of the sponge function and parsing the tag string.
    pub fn new<EF, F>(domain_separator: &DomainSeparator<EF, F, H>) -> Self
    where
        EF: ExtensionField<F> + TwoAdicField,
        F: Field + TwoAdicField + PrimeField64,
    {
        let stack = domain_separator.finalize();
        let tag = Self::generate_tag(domain_separator.as_bytes());
        Self::unchecked_load_with_stack(tag, stack)
    }

    /// Perform secure absorption of the elements in `input`.
    ///
    /// Absorb calls can be batched together, or provided separately for streaming-friendly
    /// protocols.
    pub fn absorb(&mut self, input: &[u8]) -> Result<(), DomainSeparatorMismatch> {
        match self.stack.pop_front() {
            Some(Op::Absorb(length)) if length >= input.len() => {
                if length > input.len() {
                    self.stack.push_front(Op::Absorb(length - input.len()));
                }
                self.ds.absorb_unchecked(input);
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
    pub fn squeeze(&mut self, output: &mut [u8]) -> Result<(), DomainSeparatorMismatch> {
        match self.stack.pop_front() {
            Some(Op::Squeeze(length)) if output.len() <= length => {
                self.ds.squeeze_unchecked(output);
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

    fn generate_tag(iop_bytes: &[u8]) -> [u8; 32] {
        let mut keccak = Keccak::default();
        keccak.absorb_unchecked(iop_bytes);
        let mut tag = [0u8; 32];
        keccak.squeeze_unchecked(&mut tag);
        tag
    }

    fn unchecked_load_with_stack(tag: [u8; 32], stack: VecDeque<Op>) -> Self {
        Self {
            ds: H::new(tag),
            stack,
            _unit: PhantomData,
        }
    }
}

impl<H: DuplexSpongeInterface<u8>> Drop for HashStateWithInstructions<H> {
    /// Destroy the sponge state.
    fn drop(&mut self) {
        // it's a bit violent to panic here,
        // because any other issue in the protocol transcript causing `Safe` to get out of scope
        // (like another panic) will pollute the traceback.
        // debug_assert!(self.stack.is_empty());
        if !self.stack.is_empty() {
            eprintln!(
                "HashStateWithInstructions dropped with unfinished operations:\n{:?}",
                self.stack
            );
        }
        // XXX. is the compiler going to optimize this out?
        self.ds.zeroize();
    }
}

#[cfg(test)]
#[allow(clippy::bool_assert_comparison)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use p3_baby_bear::BabyBear;

    use super::*;

    type F = BabyBear;

    #[derive(Default, Clone)]
    struct DummySponge {
        pub absorbed: Rc<RefCell<Vec<u8>>>,
        pub squeezed: Rc<RefCell<Vec<u8>>>,
        pub ratcheted: Rc<RefCell<bool>>,
    }

    impl zeroize::Zeroize for DummySponge {
        fn zeroize(&mut self) {
            self.absorbed.borrow_mut().clear();
            self.squeezed.borrow_mut().clear();
            *self.ratcheted.borrow_mut() = false;
        }
    }

    impl DummySponge {
        fn new_inner() -> Self {
            Self {
                absorbed: Rc::new(RefCell::new(Vec::new())),
                squeezed: Rc::new(RefCell::new(Vec::new())),
                ratcheted: Rc::new(RefCell::new(false)),
            }
        }
    }

    impl DuplexSpongeInterface<u8> for DummySponge {
        fn new(_iv: [u8; 32]) -> Self {
            Self::new_inner()
        }

        fn absorb_unchecked(&mut self, input: &[u8]) -> &mut Self {
            self.absorbed.borrow_mut().extend_from_slice(input);
            self
        }

        fn squeeze_unchecked(&mut self, output: &mut [u8]) -> &mut Self {
            for (i, byte) in output.iter_mut().enumerate() {
                *byte = i as u8; // Dummy output
            }
            self.squeezed.borrow_mut().extend_from_slice(output);
            self
        }
    }

    #[test]
    fn test_absorb_works_and_modifies_stack() {
        let mut domsep = DomainSeparator::<F, F, DummySponge>::new("test");
        domsep.absorb(2, "x");
        let mut state = HashStateWithInstructions::<DummySponge>::new(&domsep);

        assert_eq!(state.stack.len(), 1);

        let result = state.absorb(&[1, 2]);
        assert!(result.is_ok());

        assert_eq!(state.stack.len(), 0);
        let inner = state.ds.absorbed.borrow();
        assert_eq!(&*inner, &[1, 2]);
    }

    #[test]
    fn test_absorb_too_much_returns_error() {
        let mut domsep = DomainSeparator::<F, F, DummySponge>::new("test");
        domsep.absorb(2, "x");
        let mut state = HashStateWithInstructions::<DummySponge>::new(&domsep);

        let result = state.absorb(&[1, 2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_squeeze_works() {
        let mut domsep = DomainSeparator::<F, F, DummySponge>::new("test");
        domsep.squeeze(3, "y");
        let mut state = HashStateWithInstructions::<DummySponge>::new(&domsep);

        let mut out = [0u8; 3];
        let result = state.squeeze(&mut out);
        assert!(result.is_ok());
        assert_eq!(out, [0, 1, 2]);
    }

    #[test]
    fn test_squeeze_with_leftover_updates_stack() {
        let mut domsep = DomainSeparator::<F, F, DummySponge>::new("test");
        domsep.squeeze(4, "z");
        let mut state = HashStateWithInstructions::<DummySponge>::new(&domsep);

        let mut out = [0u8; 2];
        let result = state.squeeze(&mut out);
        assert!(result.is_ok());

        assert_eq!(state.stack.front(), Some(&Op::Squeeze(2)));
    }

    #[test]
    fn test_multiple_absorbs_deplete_stack_properly() {
        let mut domsep = DomainSeparator::<F, F, DummySponge>::new("test");
        domsep.absorb(5, "a");
        let mut state = HashStateWithInstructions::<DummySponge>::new(&domsep);

        let res1 = state.absorb(&[1, 2]);
        assert!(res1.is_ok());
        assert_eq!(state.stack.front(), Some(&Op::Absorb(3)));

        let res2 = state.absorb(&[3, 4, 5]);
        assert!(res2.is_ok());
        assert!(state.stack.is_empty());

        assert_eq!(&*state.ds.absorbed.borrow(), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_multiple_squeeze_deplete_stack_properly() {
        let mut domsep = DomainSeparator::<F, F, DummySponge>::new("test");
        domsep.squeeze(5, "z");
        let mut state = HashStateWithInstructions::<DummySponge>::new(&domsep);

        let mut out1 = [0u8; 2];
        assert!(state.squeeze(&mut out1).is_ok());
        assert_eq!(state.stack.front(), Some(&Op::Squeeze(3)));

        let mut out2 = [0u8; 3];
        assert!(state.squeeze(&mut out2).is_ok());
        assert!(state.stack.is_empty());
        assert_eq!(&*state.ds.squeezed.borrow(), &[0, 1, 0, 1, 2]);
    }

    #[test]
    fn test_absorb_then_wrong_squeeze_clears_stack() {
        let mut domsep = DomainSeparator::<F, F, DummySponge>::new("test");
        domsep.absorb(3, "in");
        let mut state = HashStateWithInstructions::<DummySponge>::new(&domsep);

        let mut out = [0u8; 1];
        let result = state.squeeze(&mut out);
        assert!(result.is_err());
        assert!(state.stack.is_empty());
    }

    #[test]
    fn test_absorb_exact_then_too_much() {
        let mut domsep = DomainSeparator::<F, F, DummySponge>::new("test");
        domsep.absorb(2, "x");
        let mut state = HashStateWithInstructions::<DummySponge>::new(&domsep);

        assert!(state.absorb(&[10, 20]).is_ok());
        assert!(state.absorb(&[30]).is_err()); // no ops left
        assert!(state.stack.is_empty());
    }

    #[test]
    fn test_from_impl_constructs_hash_state() {
        let mut domsep = DomainSeparator::<F, F, DummySponge>::new("from");
        domsep.absorb(1, "in");
        let state = HashStateWithInstructions::<DummySponge>::new(&domsep);

        assert_eq!(state.stack.len(), 1);
        assert_eq!(state.stack.front(), Some(&Op::Absorb(1)));
    }

    #[test]
    fn test_generate_tag_is_deterministic() {
        let mut ds1 = DomainSeparator::<F, F, DummySponge>::new("session1");
        ds1.absorb(1, "x");
        let mut ds2 = DomainSeparator::<F, F, DummySponge>::new("session1");
        ds2.absorb(1, "x");

        let tag1 = HashStateWithInstructions::<DummySponge>::new(&ds1);
        let tag2 = HashStateWithInstructions::<DummySponge>::new(&ds2);

        assert_eq!(&*tag1.ds.absorbed.borrow(), &*tag2.ds.absorbed.borrow());
    }
}
