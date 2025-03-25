use prover::Leafs;

use crate::whir::prover::Proof;

pub mod committer;
pub mod fiat_shamir;
pub mod parameters;
pub mod parsed_proof;
pub mod prover;
pub mod statement;
pub mod stir_evaluations;
pub mod utils;
pub mod verifier;

// Only includes the authentication paths
#[derive(Debug, Default, Clone)]
pub struct WhirProof<F, const DIGEST_ELEMS: usize> {
    pub merkle_paths: Vec<(Leafs<F>, Proof<F, DIGEST_ELEMS>)>,
    pub statement_values_at_random_point: Vec<F>,
}

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

/// The result type when trying to prove or verify a proof using Fiat-Shamir.
pub type ProofResult<T> = Result<T, ProofError>;

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_keccak::Keccak256Hash;
    use p3_symmetric::Hash;
    use rand::rng;

    use super::{
        committer::Committer,
        prover::Prover,
        statement::{StatementVerifier, Weights},
        verifier::Verifier,
    };
    use crate::{
        merkle_tree::{MyCompress, MyHash, Perm, WhirChallenger},
        parameters::{
            FoldType, FoldingFactor, MultivariateParameters, SoundnessType, WhirParameters,
        },
        poly::{coeffs::CoefficientList, evals::EvaluationsList, multilinear::MultilinearPoint},
        whir::{
            fiat_shamir::WhirChallengerTranscript, parameters::WhirConfig, statement::Statement,
        },
    };

    type F = BabyBear;

    fn make_whir_things(
        num_variables: usize,
        folding_factor: FoldingFactor,
        num_points: usize,
        soundness_type: SoundnessType,
        pow_bits: usize,
        fold_type: FoldType,
    ) {
        let num_coeffs = 1 << num_variables;

        let perm = Perm::new_from_rng_128(&mut rng());

        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);

        let mv_params = MultivariateParameters::<F>::new(num_variables);

        let whir_params = WhirParameters::<(), _, _> {
            initial_statement: true,
            security_level: 32,
            pow_bits,
            folding_factor,
            merkle_hash: hash,
            merkle_compress: compress,
            soundness_type,
            _pow_parameters: Default::default(),
            starting_log_inv_rate: 1,
            fold_optimisation: fold_type,
        };

        let params = WhirConfig::new(mv_params, whir_params);

        let polynomial = CoefficientList::new(vec![F::from_u64(1); num_coeffs]);

        let points: Vec<_> = (0..num_points)
            .map(|_| MultilinearPoint::<F>::rand(&mut rng(), num_variables))
            .collect();

        let mut statement = Statement::<F>::new(num_variables);

        for point in &points {
            let eval = polynomial.evaluate(point);
            let weights = Weights::evaluation(point.clone());
            statement.add_constraint(weights, eval);
        }

        let input = CoefficientList::new((0..1 << num_variables).map(F::from_u64).collect());

        let linear_claim_weight = Weights::linear(input.into());

        let poly = EvaluationsList::from(polynomial.clone().to_extension());

        let sum = linear_claim_weight.weighted_sum(&poly);
        statement.add_constraint(linear_claim_weight, sum);

        // Create an empty digest of zeros to simulate the Merkle root.
        let dummy_digest: Hash<F, u8, 32> = Hash::from([0; 32]);

        // Create the challenger with an empty Keccak state.
        let mut proof_challenger = WhirChallenger::<F>::from_hasher(vec![], Keccak256Hash {});

        // Run the transcript logic.
        proof_challenger.commit_statement(&params, dummy_digest);
        proof_challenger.add_whir_proof(&params, dummy_digest);

        let committer = Committer::new(params.clone());
        let witness = committer.commit(&mut proof_challenger, polynomial);

        let prover = Prover(params.clone());
        let statement_verifier = StatementVerifier::from_statement(&statement);

        let proof = prover.prove(&mut proof_challenger, statement, witness);

        let verifier = Verifier::new(params);

        assert!(verifier.verify(&mut proof_challenger, &statement_verifier, &proof).is_ok());
    }
}
