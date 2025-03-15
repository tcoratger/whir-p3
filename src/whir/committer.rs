use p3_field::Field;

use super::parameters::WhirConfig;
use crate::{merkle_tree::WhirMerkleTree, poly::coeffs::CoefficientList};

#[derive(Debug)]
pub struct Witness<F> {
    pub(crate) polynomial: CoefficientList<F>,
    pub(crate) merkle_tree: WhirMerkleTree,
    pub(crate) merkle_leaves: Vec<F>,
    pub(crate) ood_points: Vec<F>,
    pub(crate) ood_answers: Vec<F>,
}

#[derive(Debug)]
pub struct Committer<F, PowStrategy>(WhirConfig<F, PowStrategy>);

impl<F, PowStrategy> Committer<F, PowStrategy>
where
    F: Field,
{
    pub const fn new(config: WhirConfig<F, PowStrategy>) -> Self {
        Self(config)
    }
}
