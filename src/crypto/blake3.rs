use std::marker::PhantomData;

use p3_field::Field;

use crate::{
    fiat_shamir::{
        domain_separator::DomainSeparator,
        errors::{ProofError, ProofResult},
        prover::ProverState,
        traits::{ByteDomainSeparator, BytesToUnitSerialize},
    },
    whir::{domainsep::DigestDomainSeparator, utils::DigestToUnitSerialize},
};

#[derive(Debug, Default, Clone, Copy, Eq, PartialEq, Hash)]
pub struct Blake3Digest([u8; 32]);

// impl DigestToUnitSerialize<Blake3Digest> for ProverState {
//     fn add_digest(&mut self, digest: Blake3Digest) -> ProofResult<()> {
//         self.add_bytes(&digest.0).map_err(ProofError::InvalidDomainSeparator)
//     }
// }
