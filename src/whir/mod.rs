use committer::{reader::CommitmentReader, writer::CommitmentWriter};
use p3_baby_bear::BabyBear;
use p3_blake3::Blake3;
use p3_challenger::HashChallenger;
use p3_dft::Radix2DitSmallBatch;
use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};
use p3_keccak::Keccak256Hash;
use p3_monty_31::dft::RecursiveDft;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use parameters::WhirConfig;
use prover::Prover;
use statement::{Statement, weights::Weights};
use verifier::Verifier;

use crate::{
    fiat_shamir::{domain_separator::DomainSeparator, pow::blake3::Blake3PoW},
    parameters::{
        FoldingFactor, MultivariateParameters, ProtocolParameters, errors::SecurityAssumption,
    },
    poly::{coeffs::CoefficientList, multilinear::MultilinearPoint},
};

pub mod committer;
pub mod parameters;
pub mod prover;
pub mod statement;
pub mod utils;
pub mod verifier;

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type ByteHash = Blake3;
type FieldHash = SerializingHasher<ByteHash>;
type MyCompress = CompressionFunctionFromHasher<ByteHash, 2, 32>;
type MyChallenger = HashChallenger<u8, Keccak256Hash, 32>;
type W = u8;
const PERM_WIDTH: usize = 200;

/// Run a complete WHIR proof lifecycle.
///
/// This function performs the full pipeline:
/// - Defines a multilinear polynomial with `num_variables`
/// - Generates constraints for this polynomial
/// - Initializes a Fiat-Shamir transcript (challenger)
/// - Commits to the polynomial and produces a witness
/// - Generates a WHIR proof
/// - Verifies the proof
///
/// The protocol is configured using folding, soundness, and PoW parameters.
pub fn make_whir_things(
    num_variables: usize,
    folding_factor: FoldingFactor,
    num_points: usize,
    soundness_type: SecurityAssumption,
    pow_bits: usize,
    rs_domain_initial_reduction_factor: usize,
) {
    // Number of coefficients = 2^num_variables
    let num_coeffs = 1 << num_variables;

    // Create hash and compression functions for the Merkle tree
    let byte_hash = ByteHash {};
    let merkle_hash = FieldHash::new(byte_hash);
    let merkle_compress = MyCompress::new(byte_hash);

    // Set the multivariate polynomial parameters
    let mv_params = MultivariateParameters::<EF>::new(num_variables);

    // Construct WHIR protocol parameters
    let whir_params = ProtocolParameters::<_, _> {
        initial_statement: true,
        security_level: 32,
        pow_bits,
        rs_domain_initial_reduction_factor,
        folding_factor,
        merkle_hash,
        merkle_compress,
        soundness_type,
        starting_log_inv_rate: 1,
    };

    // Combine protocol and polynomial parameters into a single config
    let params =
        WhirConfig::<EF, F, FieldHash, MyCompress, Blake3PoW, MyChallenger, W, PERM_WIDTH>::new(
            mv_params,
            whir_params,
        );

    // Define a polynomial with all coefficients set to 1
    let polynomial = CoefficientList::new(vec![F::ONE; num_coeffs]).to_evaluations();

    // Sample `num_points` random multilinear points in the Boolean hypercube
    let points: Vec<_> = (0..num_points)
        .map(|_| MultilinearPoint((0..num_variables).map(|i| EF::from_u64(i as u64)).collect()))
        .collect();

    // Construct a new statement with the correct number of variables
    let mut statement = Statement::<EF>::new(num_variables);

    // Add constraints for each sampled point (equality constraints)
    for point in &points {
        let eval = polynomial.evaluate(point);
        let weights = Weights::evaluation(point.clone());
        statement.add_constraint(weights, eval);
    }

    // Construct a linear constraint to test sumcheck
    let input = CoefficientList::new((0..1 << num_variables).map(EF::from_u64).collect());
    let linear_claim_weight = Weights::linear(input.to_evaluations::<F>());

    // Evaluate the weighted sum and add it as a linear constraint
    let sum = linear_claim_weight.evaluate_evals(&polynomial);
    statement.add_constraint(linear_claim_weight, sum);

    // Define the Fiat-Shamir domain separator pattern for committing and proving
    let mut domainsep = DomainSeparator::new("🌪️");
    domainsep.commit_statement(&params);
    domainsep.add_whir_proof(&params);

    let challenger = MyChallenger::new(vec![], Keccak256Hash);

    // Initialize the Merlin transcript from the IOPattern
    let mut prover_state = domainsep.to_prover_state::<_, 32>(challenger.clone());

    // Commit to the polynomial and produce a witness
    let committer = CommitmentWriter::new(&params);

    let dft_committer = RecursiveDft::<F>::default();

    let witness = committer
        .commit(&dft_committer, &mut prover_state, polynomial)
        .unwrap();

    let prover = Prover(&params);

    let dft_prover = Radix2DitSmallBatch::<F>::default();

    // Generate a proof for the given statement and witness
    prover
        .prove(&dft_prover, &mut prover_state, statement.clone(), witness)
        .unwrap();

    // Create a commitment reader
    let commitment_reader = CommitmentReader::new(&params);

    // Create a verifier with matching parameters
    let verifier = Verifier::new(&params);

    // Reconstruct verifier's view of the transcript using the DomainSeparator and prover's data
    let mut verifier_state =
        domainsep.to_verifier_state::<_, 32>(prover_state.narg_string(), challenger);

    // Parse the commitment
    let parsed_commitment = commitment_reader
        .parse_commitment::<32>(&mut verifier_state)
        .unwrap();

    // Verify that the generated proof satisfies the statement
    verifier
        .verify(&mut verifier_state, &parsed_commitment, &statement)
        .unwrap();
}

#[cfg(test)]
mod tests {
    use crate::{
        parameters::errors::SecurityAssumption,
        whir::{FoldingFactor, make_whir_things},
    };

    #[test]
    fn test_whir_end_to_end() {
        let folding_factors = [
            FoldingFactor::Constant(1),
            FoldingFactor::Constant(2),
            FoldingFactor::Constant(3),
            FoldingFactor::Constant(4),
            FoldingFactor::ConstantFromSecondRound(2, 1),
            FoldingFactor::ConstantFromSecondRound(3, 1),
            FoldingFactor::ConstantFromSecondRound(3, 2),
            FoldingFactor::ConstantFromSecondRound(5, 2),
        ];
        let soundness_type = [
            SecurityAssumption::JohnsonBound,
            SecurityAssumption::CapacityBound,
            SecurityAssumption::UniqueDecoding,
        ];
        let num_points = [0, 1, 2];
        let pow_bits = [0, 5, 10];
        let rs_domain_initial_reduction_factors = 1..=3;

        for rs_domain_initial_reduction_factor in rs_domain_initial_reduction_factors {
            for folding_factor in folding_factors {
                if folding_factor.at_round(0) < rs_domain_initial_reduction_factor {
                    continue;
                }
                let num_variables = folding_factor.at_round(0)..=3 * folding_factor.at_round(0);
                for num_variable in num_variables {
                    for num_points in num_points {
                        for soundness_type in soundness_type {
                            for pow_bits in pow_bits {
                                make_whir_things(
                                    num_variable,
                                    folding_factor,
                                    num_points,
                                    soundness_type,
                                    pow_bits,
                                    rs_domain_initial_reduction_factor,
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
