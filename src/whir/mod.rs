use prover::Leafs;

use crate::whir::prover::Proof;

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
}

// #[cfg(test)]
// mod tests {
//     use p3_baby_bear::BabyBear;
//     use p3_field::PrimeCharacteristicRing;
//     use p3_keccak::Keccak256Hash;
//     use p3_symmetric::Hash;
//     use rand::rng;

//     use super::{
//         committer::Committer,
//         prover::Prover,
//         statement::{StatementVerifier, Weights},
//         verifier::Verifier,
//     };
//     use crate::{
//         merkle_tree::{MyCompress, MyHash, Perm, WhirChallenger},
//         parameters::{
//             FoldType, FoldingFactor, MultivariateParameters, SoundnessType, WhirParameters,
//         },
//         poly::{coeffs::CoefficientList, evals::EvaluationsList, multilinear::MultilinearPoint},
//         whir::{
//             fiat_shamir::WhirChallengerTranscript, parameters::WhirConfig, statement::Statement,
//         },
//     };

//     /// Field type used in the tests.
//     type F = BabyBear;

//     /// Run a complete WHIR STARK proof lifecycle.
//     ///
//     /// This function performs the full pipeline:
//     /// - Defines a multilinear polynomial with `num_variables`
//     /// - Generates constraints for this polynomial
//     /// - Initializes a Fiat-Shamir transcript (challenger)
//     /// - Commits to the polynomial and produces a witness
//     /// - Generates a STARK proof with the prover
//     /// - Verifies the proof with the verifier
//     ///
//     /// The protocol is configured using folding, soundness, and PoW parameters.
//     fn make_whir_things(
//         num_variables: usize,
//         folding_factor: FoldingFactor,
//         num_points: usize,
//         soundness_type: SoundnessType,
//         pow_bits: usize,
//         fold_type: FoldType,
//     ) {
//         // Number of coefficients = 2^num_variables
//         let num_coeffs = 1 << num_variables;

//         // Initialize a random permutation for the Merkle tree.
//         let perm = Perm::new_from_rng_128(&mut rng());

//         // Create hash and compression functions for the Merkle tree
//         let hash = MyHash::new(perm.clone());
//         let compress = MyCompress::new(perm);

//         // Set the multivariate polynomial parameters
//         let mv_params = MultivariateParameters::<F>::new(num_variables);

//         // Construct WHIR protocol parameters
//         let whir_params = WhirParameters::<_, _> {
//             initial_statement: true,
//             security_level: 32,
//             pow_bits,
//             folding_factor,
//             merkle_hash: hash,
//             merkle_compress: compress,
//             soundness_type,
//             starting_log_inv_rate: 1,
//             fold_optimisation: fold_type,
//         };

//         // Combine protocol and polynomial parameters into a single config
//         let params = WhirConfig::new(mv_params, whir_params);

//         // Define a polynomial with all coefficients set to 1 (i.e., constant 1 polynomial)
//         let polynomial = CoefficientList::new(vec![F::ONE; num_coeffs]);

//         // Sample `num_points` random multilinear points in the Boolean hypercube
//         let points: Vec<_> = (0..num_points)
//             .map(|_| MultilinearPoint::<F>::rand(&mut rng(), num_variables))
//             .collect();

//         // Construct a new statement with the correct number of variables
//         let mut statement = Statement::<F>::new(num_variables);

//         // Add constraints for each sampled point (equality constraints)
//         for point in &points {
//             let eval = polynomial.evaluate(point);
//             let weights = Weights::evaluation(point.clone());
//             statement.add_constraint(weights, eval);
//         }

//         // Construct a linear constraint to test sumcheck
//         let input = CoefficientList::new((0..1 << num_variables).map(F::from_u64).collect());
//         let linear_claim_weight = Weights::linear(input.into());

//         // Convert the polynomial to extension form for weighted evaluation
//         let poly = EvaluationsList::from(polynomial.clone().to_extension());

//         // Evaluate the weighted sum and add it as a linear constraint
//         let sum = linear_claim_weight.weighted_sum(&poly);
//         statement.add_constraint(linear_claim_weight, sum);

//         // Create a dummy digest representing the initial Merkle root
//         let dummy_digest: Hash<F, u8, 32> = Hash::from([0; 32]);

//         // Instantiate the challenger with an empty state
//         let mut proof_challenger = WhirChallenger::<F>::from_hasher(vec![], Keccak256Hash {});

//         // Run the IOPattern logic: commit to the statement and run the proof phase
//         proof_challenger.commit_statement(&params, dummy_digest);
//         proof_challenger.add_whir_proof(&params, dummy_digest);

//         // Commit to the polynomial and produce a witness
//         let committer = Committer::new(params.clone());
//         let witness = committer.commit(&mut proof_challenger, polynomial);

//         // Generate a proof using the prover
//         let prover = Prover(params.clone());
//         let statement_verifier = StatementVerifier::from_statement(&statement);
//         let proof = prover.prove(&mut proof_challenger, statement, witness);

//         // Verify the proof using the verifier and the same transcript
//         let verifier = Verifier::new(params);
//         assert!(verifier.verify(&mut proof_challenger, &statement_verifier, &proof).is_ok());
//     }

//     #[test]
//     #[ignore]
//     fn test_whir() {
//         // let folding_factors = [1, 2, 3, 4];
//         // let soundness_type = [
//         //     SoundnessType::ConjectureList,
//         //     SoundnessType::ProvableList,
//         //     SoundnessType::UniqueDecoding,
//         // ];
//         // let fold_types = [FoldType::Naive, FoldType::ProverHelps];
//         // let num_points = [0, 1, 2];
//         // let pow_bits = [0, 5, 10];

//         let folding_factors = [1];
//         let soundness_type = [SoundnessType::ConjectureList];
//         let fold_types = [FoldType::Naive];
//         let num_points = [0];
//         let pow_bits = [0];

//         for folding_factor in folding_factors {
//             let num_variables = folding_factor..=3 * folding_factor;
//             for num_variable in num_variables {
//                 for fold_type in fold_types {
//                     for num_points in num_points {
//                         for soundness_type in soundness_type {
//                             for pow_bits in pow_bits {
//                                 make_whir_things(
//                                     num_variable,
//                                     FoldingFactor::Constant(folding_factor),
//                                     num_points,
//                                     soundness_type,
//                                     pow_bits,
//                                     fold_type,
//                                 );
//                             }
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }
