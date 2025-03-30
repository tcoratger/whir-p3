use committer::{reader::CommitmentReader, writer::CommitmentWriter};
use p3_blake3::Blake3;
use p3_matrix::Dimensions;
use parameters::WhirConfig;
use prover::{Leafs, Prover};
use statement::{Statement, StatementVerifier, Weights};
use verifier::Verifier;

use crate::{
    fiat_shamir::{domain_separator::DomainSeparator, pow::blake3::Blake3PoW},
    parameters::{FoldType, FoldingFactor, MultivariateParameters, SoundnessType, WhirParameters},
    poly::{coeffs::CoefficientList, evals::EvaluationsList, multilinear::MultilinearPoint},
    whir::{domainsep::WhirDomainSeparator, prover::Proof},
};

pub mod committer;
pub mod domainsep;
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
    pub merkle_paths: Vec<(Leafs<F>, Proof<DIGEST_ELEMS>)>,
    pub statement_values_at_random_point: Vec<F>,
    pub mmcs_dimensions: Vec<Dimensions>,
}

use p3_baby_bear::BabyBear;
use p3_field::PrimeCharacteristicRing;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher32};

type F = BabyBear;
type ByteHash = Blake3;
type FieldHash = SerializingHasher32<ByteHash>;
type MyCompress = CompressionFunctionFromHasher<ByteHash, 2, 32>;

/// Run a complete WHIR STARK proof lifecycle.
///
/// This function performs the full pipeline:
/// - Defines a multilinear polynomial with `num_variables`
/// - Generates constraints for this polynomial
/// - Initializes a Fiat-Shamir transcript (challenger)
/// - Commits to the polynomial and produces a witness
/// - Generates a STARK proof with the prover
/// - Verifies the proof with the verifier
///
/// The protocol is configured using folding, soundness, and PoW parameters.
pub fn make_whir_things(
    num_variables: usize,
    folding_factor: FoldingFactor,
    num_points: usize,
    soundness_type: SoundnessType,
    pow_bits: usize,
    fold_type: FoldType,
) {
    // Number of coefficients = 2^num_variables
    let num_coeffs = 1 << num_variables;

    // Create hash and compression functions for the Merkle tree
    let byte_hash = ByteHash {};
    let merkle_hash = FieldHash::new(byte_hash);
    let merkle_compress = MyCompress::new(byte_hash);

    // Set the multivariate polynomial parameters
    let mv_params = MultivariateParameters::<F>::new(num_variables);

    // Construct WHIR protocol parameters
    let whir_params = WhirParameters::<_, _> {
        initial_statement: true,
        security_level: 16,
        pow_bits,
        folding_factor,
        merkle_hash,
        merkle_compress,
        soundness_type,
        starting_log_inv_rate: 1,
        fold_optimisation: fold_type,
    };

    // Combine protocol and polynomial parameters into a single config
    let params = WhirConfig::<F, FieldHash, MyCompress, Blake3PoW>::new(mv_params, whir_params);

    // Define a polynomial with all coefficients set to 1 (i.e., constant 1 polynomial)
    let polynomial = CoefficientList::new(vec![F::ONE; num_coeffs]);

    // Sample `num_points` random multilinear points in the Boolean hypercube
    let points: Vec<_> = (0..num_points)
        .map(|_| MultilinearPoint((0..num_variables).map(|i| F::from_u64(i as u64)).collect()))
        .collect();

    // Construct a new statement with the correct number of variables
    let mut statement = Statement::<F>::new(num_variables);

    // Add constraints for each sampled point (equality constraints)
    for point in &points {
        let eval = polynomial.evaluate(point);
        let weights = Weights::evaluation(point.clone());
        statement.add_constraint(weights, eval);
    }

    // Construct a linear constraint to test sumcheck
    let input = CoefficientList::new((0..1 << num_variables).map(F::from_u64).collect());
    let linear_claim_weight = Weights::linear(input.into());

    // Convert the polynomial to extension form for weighted evaluation
    let poly = EvaluationsList::from(polynomial.clone().to_extension());

    // Evaluate the weighted sum and add it as a linear constraint
    let sum = linear_claim_weight.weighted_sum(&poly);
    statement.add_constraint(linear_claim_weight, sum);

    // Define the Fiat-Shamir domain separator pattern for committing and proving
    let domainsep = DomainSeparator::new("🌪️")
        .commit_statement(&params)
        .add_whir_proof(&params);

    // Initialize the Merlin transcript from the IOPattern
    let mut prover_state = domainsep.to_prover_state();

    // Commit to the polynomial and produce a witness
    let committer = CommitmentWriter::new(params.clone());

    let witness = committer.commit(&mut prover_state, polynomial).unwrap();

    // Generate a proof using the prover
    let prover = Prover(params.clone());

    // Extract verifier-side version of the statement (only public data)
    let statement_verifier = StatementVerifier::from_statement(&statement);

    // Generate a STARK proof for the given statement and witness
    let proof = prover.prove(&mut prover_state, statement, witness).unwrap();

    // Create a commitment reader
    let commitment_reader = CommitmentReader::new(&params);

    // Create a verifier with matching parameters
    let verifier = Verifier::new(&params);

    // Reconstruct verifier's view of the transcript using the DomainSeparator and prover's data
    let mut verifier_state = domainsep.to_verifier_state(prover_state.narg_string());

    // Parse the commitment
    let parsed_commitment = commitment_reader
        .parse_commitment::<_, 32>(&mut verifier_state)
        .unwrap();

    // Verify that the generated proof satisfies the statement
    assert!(
        verifier
            .verify(
                &mut verifier_state,
                &parsed_commitment,
                &statement_verifier,
                &proof
            )
            .is_ok()
    );
}

#[cfg(test)]
mod tests {
    use crate::{
        parameters::SoundnessType,
        whir::{FoldType, FoldingFactor, make_whir_things},
    };

    #[test]
    fn test_whir_end_to_end() {
        let folding_factors = [1, 2];
        let soundness_type = [
            SoundnessType::ConjectureList,
            SoundnessType::ProvableList,
            SoundnessType::UniqueDecoding,
        ];
        let fold_types = [FoldType::Naive, FoldType::ProverHelps];
        let num_points = [0, 1, 2];
        let pow_bits = [0, 5];

        for folding_factor in folding_factors {
            let num_variables = folding_factor..=3 * folding_factor;
            for num_variable in num_variables {
                for fold_type in fold_types {
                    for num_points in num_points {
                        for soundness_type in soundness_type {
                            for pow_bits in pow_bits {
                                make_whir_things(
                                    num_variable,
                                    FoldingFactor::Constant(folding_factor),
                                    num_points,
                                    soundness_type,
                                    pow_bits,
                                    fold_type,
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
