use std::fmt::Debug;

use p3_field::{ExtensionField, Field};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use serde::{Serialize, de::DeserializeOwned};

use crate::whir::pcs::query::MlQuery;

pub mod proof;
pub mod prover_data;
pub mod query;
pub mod whir;

/// A multilinear polynomial commitment scheme.
///
/// This trait enables committing to one or more multilinear polynomials and later
/// proving and verifying their evaluation at specified points.
pub trait MlPcs<Challenge, Challenger>
where
    Challenge: ExtensionField<Self::Val>,
{
    /// The base field used for polynomial coefficients and matrix entries.
    type Val: Field;

    /// Commitment type sent to the verifier to represent a batch of polynomials.
    type Commitment: Clone + Debug + Serialize + DeserializeOwned;

    /// Data retained by the prover to assist with generating openings.
    type ProverData;

    /// A view into the evaluations of a polynomial matrix.
    ///
    /// Typically a reference to a `RowMajorMatrix` slice or similar.
    type Evaluations<'a>: Matrix<Self::Val> + 'a;

    /// A proof that openings to committed polynomials are correct.
    type Proof: Clone + Debug + Serialize + DeserializeOwned;

    /// An error that may occur during verification.
    type Error: Debug;

    /// Commit to a list of multilinear polynomial evaluation matrices.
    ///
    /// # Arguments
    /// - `evaluations`: A vector of `RowMajorMatrix`s, where each matrix encodes
    ///   the evaluation table of a multilinear polynomial over the Boolean hypercube.
    ///
    /// # Returns
    /// - A commitment to the batch and auxiliary data (`ProverData`) to support openings.
    fn commit(
        &self,
        evaluations: Vec<RowMajorMatrix<Self::Val>>,
    ) -> (Self::Commitment, Self::ProverData);

    /// Access the evaluations associated with a committed matrix.
    ///
    /// # Arguments
    /// - `prover_data`: Auxiliary data returned during commitment.
    /// - `idx`: Index of the matrix (in the batch) to retrieve evaluations for.
    fn get_evaluations<'a>(
        &self,
        prover_data: &'a Self::ProverData,
        idx: usize,
    ) -> Self::Evaluations<'a>;

    /// Open committed matrices at one or more query points.
    ///
    /// The opening is done over a series of "rounds", where each round may involve
    /// multiple matrices, and each matrix may be queried at several points.
    ///
    /// The caller is responsible for recording any revealed evaluations into the transcript.
    ///
    /// # Arguments
    /// - `rounds`: For each round, a list of matrices and their respective queries and values.
    /// - `challenger`: A stateful cryptographic oracle used for sampling challenges.
    ///
    /// # Returns
    /// - A proof that the evaluations are consistent with the commitments.
    #[allow(clippy::type_complexity)]
    fn open(
        &self,
        // For each round,
        rounds: Vec<(
            &Self::ProverData,
            // for each matrix,
            Vec<
                // for each query:
                Vec<(
                    // the query,
                    MlQuery<Challenge>,
                    // values at the query
                    Vec<Challenge>,
                )>,
            >,
        )>,
        challenger: &mut Challenger,
    ) -> Self::Proof;

    /// Verify that committed polynomials evaluate to the claimed values at queried points.
    ///
    /// The verifier uses the same challenge-generation interface as the prover, and must
    /// observe the same sequence of evaluations.
    ///
    /// # Arguments
    /// - `rounds`: A vector of (commitment, queries) per round and matrix.
    /// - `proof`: The proof returned by the prover during `open`.
    /// - `challenger`: The verifier’s cryptographic oracle, used to sample challenges.
    ///
    /// # Returns
    /// - `Ok(())` if the proof is valid, or an error otherwise.
    #[allow(clippy::type_complexity)]
    fn verify(
        &self,
        // For each round:
        rounds: Vec<(
            Self::Commitment,
            // for each matrix:
            Vec<
                // for each query:
                Vec<(
                    // the query,
                    MlQuery<Challenge>,
                    // values at the query
                    Vec<Challenge>,
                )>,
            >,
        )>,
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error>;
}
