use committer::{reader::CommitmentReader, writer::CommitmentWriter};
use constraints::statement::Statement;
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::{CanObserve, CanSample, DuplexChallenger};
use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use parameters::WhirConfig;
use prover::Prover;
use rand::{SeedableRng, rngs::SmallRng};
use verifier::Verifier;

use crate::{
    dft::EvalsDft,
    parameters::{FoldingFactor, ProtocolParameters, errors::SecurityAssumption},
    poly::{coeffs::CoefficientList, multilinear::MultilinearPoint},
};
use crate::whir::proof::{InitialPhase, WhirProof};

pub mod committer;
pub mod constraints;
pub mod parameters;
pub mod prover;
pub mod utils;
pub mod verifier;
pub mod proof;

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type Perm = Poseidon2BabyBear<16>;

type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

/// Run a complete WHIR proof lifecycle with configurable parameters.
#[allow(clippy::too_many_arguments)]
pub fn make_whir_things(
    num_variables: usize,
    folding_factor: FoldingFactor,
    num_points: usize,
    soundness_type: SecurityAssumption,
    pow_bits: usize,
    rs_domain_initial_reduction_factor: usize,
    use_univariate_skip: bool,
    initial_statement: bool,
) {
    // Calculate polynomial size: 2^num_variables coefficients for multilinear polynomial
    let num_coeffs = 1 << num_variables;

    // Initialize deterministic RNG for reproducible test results
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);

    // Create cryptographic primitives for Merkle tree operations
    //
    // Hash function for internal nodes
    let merkle_hash = MyHash::new(perm.clone());
    // Compression for leaf-to-parent hashing
    let merkle_compress = MyCompress::new(perm);

    // Configure WHIR protocol with all security and performance parameters
    let whir_params = ProtocolParameters {
        initial_statement,
        security_level: 32,
        pow_bits,
        rs_domain_initial_reduction_factor,
        folding_factor,
        merkle_hash,
        merkle_compress,
        soundness_type,
        starting_log_inv_rate: 1,
        univariate_skip: use_univariate_skip,
    };

    // Create unified configuration combining protocol and polynomial parameters
    let params =
        WhirConfig::<EF, F, MyHash, MyCompress, MyChallenger>::new(num_variables, whir_params.clone());

    // Initialize the Whir Proof structure
    let mut proof = WhirProof::from_protocol_parameters(&whir_params, num_variables);

    // Define test polynomial: all coefficients = 1 for simple verification
    //
    // TODO: replace with a random polynomial
    let polynomial = CoefficientList::new(vec![F::ONE; num_coeffs]).to_evaluations();

    // Generate evaluation points
    let points: Vec<_> = (0..num_points)
        .map(|_| {
            MultilinearPoint::new((0..num_variables).map(|i| EF::from_u64(i as u64)).collect())
        })
        .collect();

    // Initialize constraint system for the given number of variables
    let mut statement = Statement::<EF>::initialize(num_variables);

    // Add equality constraints: polynomial(point) = expected_value for each point
    for point in &points {
        statement.add_unevaluated_constraint(point.clone(), &polynomial);
    }


    // Set up the Domain separation by observing the current Whir Proof structure
    // Create fresh RNG and challenger for transcript randomness
    let mut rng = SmallRng::seed_from_u64(1);
    let mut challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));

    // Observe protocol parameters for domain separation (based on static params, not runtime proof)
    challenger.observe(F::from_u64(params.n_rounds() as u64));
    challenger.observe(F::from_u64(params.final_queries as u64));

    // Observe initial phase variant (0=WithoutStatement, 1=WithStatement, 2=WithStatementSkip)
    let phase_tag = match &proof.initial_phase {
        InitialPhase::WithoutStatement { .. } => 0,
        InitialPhase::WithStatement { .. } => 1,
        InitialPhase::WithStatementSkip { .. } => 2,
    };
    challenger.observe(F::from_u64(phase_tag as u64));

    // Create polynomial commitment using Merkle tree over evaluation domain
    let committer = CommitmentWriter::new(&params);
    // DFT evaluator for polynomial
    let dft_committer = EvalsDft::<F>::default();

    // Commit to polynomial evaluations and generate cryptographic witness
    let witness = committer
        .commit(&dft_committer, &mut proof, &mut challenger, polynomial.clone());
    proof.initial_ood_answers = witness.ood_answers.clone();

    // Initialize WHIR prover with the configured parameters
    let prover = Prover(&params);
    // DFT evaluator for proving
    let dft_prover = EvalsDft::<F>::default();

    // Generate WHIR proof
    prover.prove(&dft_prover, &mut proof, &mut challenger, statement.clone(), witness);

    // Sample final challenge to ensure transcript consistency between prover/verifier
    let checkpoint_prover: EF = challenger.sample();

    // Initialize commitment parser for verifier-side operations
    let commitment_reader = CommitmentReader::new(&params);

    // Create WHIR verifier with identical parameters to prover
    let verifier = Verifier::new(&params);

    // Create a fresh challenger for the verifier (simulating independent verification)
    let mut rng_verifier = SmallRng::seed_from_u64(1);
    let mut verifier_challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng_verifier));

    // Verifier observes the same protocol parameters as prover for domain separation
    verifier_challenger.observe(F::from_u64(params.n_rounds() as u64));
    verifier_challenger.observe(F::from_u64(params.final_queries as u64));

    // Observe initial phase variant
    let phase_tag = match &proof.initial_phase {
        InitialPhase::WithoutStatement { .. } => 0,
        InitialPhase::WithStatement { .. } => 1,
        InitialPhase::WithStatementSkip { .. } => 2,
    };
    verifier_challenger.observe(F::from_u64(phase_tag as u64));

    // Observe the statement (number of constraints and num_variables)
    verifier_challenger.observe(F::from_u64(statement.len() as u64));
    verifier_challenger.observe(F::from_u64(statement.num_variables() as u64));

    // Observe the commitment root (must match prover's observation)
    verifier_challenger.observe_slice(&proof.initial_commitment);

    // Parse and validate the polynomial commitment from proof data
    let parsed_commitment = commitment_reader
        .parse_commitment::<8>(&mut proof, &mut verifier_challenger);

    // Execute WHIR verification
    verifier
        .verify(&mut proof, &mut verifier_challenger, &parsed_commitment, &statement)
        .unwrap();

    let checkpoint_verifier: EF = verifier_challenger.sample();
    assert_eq!(checkpoint_prover, checkpoint_verifier);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whir_end_to_end_without_univariate_skip() {
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
                                    false,
                                    true,
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_whir_end_to_end_with_univariate_skip() {
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
                                    true,
                                    true,
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_whir_with_initial_statement_false() {
        let folding_factors = [
            FoldingFactor::Constant(2),
            FoldingFactor::ConstantFromSecondRound(2, 1),
        ];
        let soundness_types = [
            SecurityAssumption::JohnsonBound,
            SecurityAssumption::CapacityBound,
        ];
        let pow_bits_options = [0, 5];
        let rs_reduction_factors = [2, 3];

        for folding_factor in folding_factors {
            for rs_reduction_factor in rs_reduction_factors {
                // Skip invalid parameter combinations
                if folding_factor.at_round(0) < rs_reduction_factor {
                    continue;
                }

                for soundness_type in soundness_types {
                    for pow_bits in pow_bits_options {
                        make_whir_things(
                            2,
                            folding_factor,
                            0,
                            soundness_type,
                            pow_bits,
                            rs_reduction_factor,
                            false,
                            false, // initial_statement: FALSE
                        );
                    }
                }
            }
        }
    }
}
