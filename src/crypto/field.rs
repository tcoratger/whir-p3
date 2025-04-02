use p3_field::extension::BinomialExtensionField;
use p3_goldilocks::Goldilocks;
use p3_monty_31::MontyField31;

pub trait ExtensionDegree {
    fn extension_degree() -> usize;
}

impl<Base: ExtensionDegree, const D: usize> ExtensionDegree for BinomialExtensionField<Base, D> {
    fn extension_degree() -> usize {
        D * Base::extension_degree()
    }
}

impl<MP: p3_monty_31::MontyParameters> ExtensionDegree for MontyField31<MP> {
    fn extension_degree() -> usize {
        1
    }
}

impl ExtensionDegree for Goldilocks {
    fn extension_degree() -> usize {
        1
    }
}
