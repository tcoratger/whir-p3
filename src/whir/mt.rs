use alloc::vec::Vec;
use core::ops::Deref;

use p3_challenger::CanObserve;
use p3_commit::Mmcs;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::dense::DenseMatrix;
use p3_merkle_tree::MerkleTreeMmcs;

struct Proof<F, Comm> {
    pub initial_commitment: Comm,
    _marker: core::marker::PhantomData<F>,
}

struct Committer<F: Send + Sync + Clone, MT: Mmcs<F>>
where
    MT::Commitment: Send + Sync + Clone,
{
    mt: MT,
    _marker: core::marker::PhantomData<F>,
}

impl<F: Send + Sync + Clone, MT: Mmcs<F>> Committer<F, MT>
where
    MT::Commitment: Send + Sync + Clone,
{
    fn commit<Challenger>(
        &self,
        proof: &mut Proof<F, MT::Commitment>,
        challenger: &mut Challenger,
        data: DenseMatrix<F>,
    ) where
        Challenger: CanObserve<MT::Commitment>,
    {
        let (comm, data) = self.mt.commit_matrix(data);
        challenger.observe(comm.clone());
        proof.initial_commitment = comm;
        // let commitment = self.mt.commit(data);
        // Proof {
        //     initial_commitment: commitment,
        // }
    }
}

#[derive(Debug)]
struct Config<F: Field, MT: Mmcs<F>> {
    merkle: MT,
    _marker: core::marker::PhantomData<F>,
}

#[derive(Debug)]
struct Prover<'a, F, MT: Mmcs<F>>(&'a Config<F, MT>)
where
    F: Field;

impl<F, MT> Deref for Prover<'_, F, MT>
where
    F: Field,
    MT: Mmcs<F>,
{
    type Target = Config<F, MT>;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

#[derive(Debug)]
pub struct RoundState<F: Send + Sync + Clone, MT>
where
    MT: Mmcs<F>,
{
    data: MT::ProverData<DenseMatrix<F>>,
}

pub struct QueryOpening<F, Proof> {
    values: Vec<F>,
    proof: Proof,
}

impl<F, MT> Prover<'_, F, MT>
where
    F: TwoAdicField,
    MT: Mmcs<F>,
{
    fn prove<Dft, Challenger>(
        &self,
        dft: &Dft,
        proof: &mut Proof<F, MT::Commitment>,
        challenger: &mut Challenger,
        prover_data: MT::ProverData<DenseMatrix<F>>,
    ) where
        Dft: TwoAdicSubgroupDft<F>,
        Challenger: CanObserve<MT::Commitment>,
    {
        // mmcs.open_batch(*challenge, &round_state.commitment_merkle_prover_data);

        let comm = self.merkle.open_batch(0, &prover_data);
        let answer = comm.opened_values[0].clone();
        let proof = comm.opening_proof;
    }
}

// pub fn commit<Dft, P, W, PW, const DIGEST_ELEMS: usize>(
//     &self,
//     dft: &Dft,
//     proof: &mut WhirProof<F, EF, W, DIGEST_ELEMS>,
//     challenger: &mut Challenger,
//     statement: &mut InitialStatement<F, EF>,
// ) -> Result<MerkleTree<F, W, DenseMatrix<F>, DIGEST_ELEMS>, FiatShamirError>
// where
