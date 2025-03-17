use super::parameters::WhirConfig;
use crate::{merkle_tree::WhirMerkleTree, poly::coeffs::CoefficientList};
use p3_challenger::FieldChallenger;
use p3_field::Field;

/// Signals an invalid IO pattern.
///
/// This error indicates a wrong IO Pattern declared
/// upon instantiation of the SAFE sponge.
#[derive(Debug, Clone)]
pub struct IOPatternError(String);

/// An error happened when creating or verifying a proof.
#[derive(Debug, Clone)]
pub enum ProofError {
    /// Signals the verification equation has failed.
    InvalidProof,
    /// The IO pattern specified mismatches the IO pattern used during the protocol execution.
    InvalidIO(IOPatternError),
    /// Serialization/Deserialization led to errors.
    SerializationError,
}

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

    // pub fn commit<Merlin>(&self, merlin: &mut Merlin, polynomial:
    // CoefficientList<F::PrimeSubfield>) // -> Result<Witness<F>, ProofError>
    // where
    //     Merlin: FieldChallenger<F>,
    // {
    //     // let base_domain = self.0.starting_domain.base_domain.unwrap();
    // }
}
