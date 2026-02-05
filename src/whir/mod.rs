pub mod committer;
pub mod constraints;
pub mod parameters;
pub mod proof;
pub mod prover;
pub mod utils;
pub mod verifier;

#[cfg(test)]
mod test {

    use alloc::vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::{DuplexChallenger, FieldChallenger};
    use p3_dft::Radix2DFTSmallBatch;
    use p3_field::{Field, extension::BinomialExtensionField};
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::{Rng, SeedableRng, rngs::SmallRng};

    use crate::{
        fiat_shamir::domain_separator::DomainSeparator,
        parameters::{FoldingFactor, ProtocolParameters, errors::SecurityAssumption},
        poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
        whir::{
            committer::{reader::CommitmentReader, writer::CommitmentWriter},
            parameters::{SumcheckStrategy, WhirConfig},
            proof::WhirProof,
            prover::Prover,
            verifier::Verifier,
        },
    };

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;

    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

    /// Run a complete WHIR proof lifecycle with configurable parameters.
    #[allow(clippy::too_many_arguments)]
    fn make_whir_things(
        num_variables: usize,
        folding_factor: FoldingFactor,
        num_points: usize,
        soundness_type: SecurityAssumption,
        pow_bits: usize,
        rs_domain_initial_reduction_factor: usize,
        sumcheck_strategy: SumcheckStrategy,
    ) {
        // Calculate polynomial size: 2^num_variables coefficients for multilinear polynomial
        let num_evaluations = 1 << num_variables;

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
            security_level: 32,
            pow_bits,
            rs_domain_initial_reduction_factor,
            folding_factor,
            merkle_hash,
            merkle_compress,
            soundness_type,
            starting_log_inv_rate: 1,
        };

        // Create unified configuration combining protocol and polynomial parameters
        let params = WhirConfig::<EF, F, MyHash, MyCompress, MyChallenger>::new(
            num_variables,
            whir_params.clone(),
        );

        // Define test polynomial with random evaluations
        let polynomial = EvaluationsList::new((0..num_evaluations).map(|_| rng.random()).collect());
        // New initial statement
        let mut statement = params.initial_statement(polynomial, sumcheck_strategy);

        // And equality constraints: polynomial(point) = expected_value for each point
        for _ in 0..num_points {
            let point = MultilinearPoint::expand_from_univariate(rng.random(), num_variables);
            let _ = statement.evaluate(&point);
        }
        // Normalize to classic eq statement for verifier
        let verifier_statement = statement.normalize();

        // Setup Fiat-Shamir transcript structure for non-interactive proof generation
        let mut domainsep = DomainSeparator::new(vec![]);
        // Add statement commitment to transcript
        domainsep.commit_statement::<_, _, _, 8>(&params);
        // Add proof structure to transcript
        domainsep.add_whir_proof::<_, _, _, 8>(&params);

        // Create fresh RNG and challenger for transcript randomness
        // Initialize prover's view of the Fiat-Shamir transcript
        let mut rng = SmallRng::seed_from_u64(1);
        let mut prover_challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));
        domainsep.observe_domain_separator(&mut prover_challenger);

        // Create polynomial commitment using Merkle tree over evaluation domain
        let committer = CommitmentWriter::new(&params);
        // DFT evaluator for polynomial
        let dft = Radix2DFTSmallBatch::<F>::default();

        let mut proof =
            WhirProof::<F, EF, F, 8>::from_protocol_parameters(&whir_params, num_variables);

        // Commit to polynomial evaluations and generate cryptographic witness
        let prover_data = committer
            .commit::<_, <F as Field>::Packing, F, <F as Field>::Packing, 8>(
                &dft,
                &mut proof,
                &mut prover_challenger,
                &mut statement,
            )
            .unwrap();

        // Initialize WHIR prover with the configured parameters
        let prover = Prover(&params);

        // Generate WHIR proof
        prover
            .prove::<_, <F as Field>::Packing, F, <F as Field>::Packing, 8>(
                &dft,
                &mut proof,
                &mut prover_challenger,
                &statement,
                prover_data,
            )
            .unwrap();

        // Sample final challenge to ensure transcript consistency between prover/verifier
        let checkpoint_prover: EF = prover_challenger.sample_algebra_element();

        // Initialize commitment parser for verifier-side operations
        let commitment_reader = CommitmentReader::new(&params);

        // Create WHIR verifier with identical parameters to prover
        let verifier = Verifier::new(&params);

        // Reconstruct verifier's transcript from proof data and domain separator
        let mut rng = SmallRng::seed_from_u64(1);
        let mut verifier_challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));
        domainsep.observe_domain_separator(&mut verifier_challenger);

        // Parse and validate the polynomial commitment from proof data
        let parsed_commitment =
            commitment_reader.parse_commitment::<F, 8>(&proof, &mut verifier_challenger);

        // Execute WHIR verification
        verifier
            .verify::<<F as Field>::Packing, F, <F as Field>::Packing, 8>(
                &proof,
                &mut verifier_challenger,
                &parsed_commitment,
                verifier_statement,
            )
            .unwrap();

        let checkpoint_verifier: EF = verifier_challenger.sample_algebra_element();
        assert_eq!(checkpoint_prover, checkpoint_verifier);
    }

    #[cfg(test)]
    mod tests {

        use super::*;

        #[test]
        fn test_whir_end_to_end1() {
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
                                        SumcheckStrategy::Svo,
                                    );
                                    make_whir_things(
                                        num_variable,
                                        folding_factor,
                                        num_points,
                                        soundness_type,
                                        pow_bits,
                                        rs_domain_initial_reduction_factor,
                                        SumcheckStrategy::Classic,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[cfg(test)]
    mod keccak_tests {
        use alloc::vec;

        use p3_challenger::{HashChallenger, SerializingChallenger32};
        use p3_dft::Radix2DFTSmallBatch;
        use p3_field::extension::BinomialExtensionField;
        use p3_keccak::{Keccak256Hash, KeccakF};
        use p3_koala_bear::KoalaBear;
        use p3_symmetric::{CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher};
        use rand::{Rng, SeedableRng, rngs::SmallRng};

        use super::*;
        use crate::whir::parameters::WhirConfig;

        // Field types for Keccak tests
        type F = KoalaBear;
        type EF = BinomialExtensionField<F, 4>;

        // Keccak hash types producing [u64; 4] digests
        type U64Hash = PaddingFreeSponge<KeccakF, 25, 17, 4>;
        type KeccakFieldHash = SerializingHasher<U64Hash>;
        type KeccakCompress = CompressionFunctionFromHasher<U64Hash, 2, 4>;

        // Keccak challenger using byte-based HashChallenger
        type KeccakChallenger = SerializingChallenger32<F, HashChallenger<u8, Keccak256Hash, 32>>;

        /// Run a complete WHIR proof lifecycle with Keccak-based Merkle trees.
        #[allow(clippy::too_many_arguments)]
        fn make_whir_things_keccak(
            num_variables: usize,
            folding_factor: FoldingFactor,
            num_points: usize,
            soundness_type: SecurityAssumption,
            pow_bits: usize,
            rs_domain_initial_reduction_factor: usize,
            sumcheck_strategy: SumcheckStrategy,
        ) {
            let num_evaluations = 1 << num_variables;

            // Create Keccak primitives
            let u64_hash = U64Hash::new(KeccakF {});
            let merkle_hash = KeccakFieldHash::new(u64_hash);
            let merkle_compress = KeccakCompress::new(u64_hash);

            // Configure WHIR protocol with Keccak hashing
            let whir_params = ProtocolParameters {
                security_level: 32,
                pow_bits,
                rs_domain_initial_reduction_factor,
                folding_factor,
                merkle_hash,
                merkle_compress,
                soundness_type,
                starting_log_inv_rate: 1,
            };

            let params =
                WhirConfig::<EF, F, KeccakFieldHash, KeccakCompress, KeccakChallenger>::new(
                    num_variables,
                    whir_params.clone(),
                );

            // Create random polynomial
            let mut rng = SmallRng::seed_from_u64(1);
            let polynomial =
                EvaluationsList::new((0..num_evaluations).map(|_| rng.random()).collect());

            // New initial statement
            let mut statement = params.initial_statement(polynomial, sumcheck_strategy);
            // And equality constraints: polynomial(point) = expected_value for each point
            for _ in 0..num_points {
                let point = MultilinearPoint::expand_from_univariate(rng.random(), num_variables);
                let _ = statement.evaluate(&point);
            }
            // Normalize to classic eq statement for verifier
            let verifier_statement = statement.normalize();

            // Setup Fiat-Shamir transcript
            let mut domainsep = DomainSeparator::new(vec![]);
            domainsep.commit_statement::<_, _, _, 4>(&params);
            domainsep.add_whir_proof::<_, _, _, 4>(&params);

            // Create prover challenger
            let inner = HashChallenger::<u8, Keccak256Hash, 32>::new(vec![], Keccak256Hash {});
            let mut prover_challenger = KeccakChallenger::new(inner);
            domainsep.observe_domain_separator(&mut prover_challenger);

            // Commit and prove
            let committer = CommitmentWriter::new(&params);
            let dft = Radix2DFTSmallBatch::<F>::default();

            let mut proof =
                WhirProof::<F, EF, u64, 4>::from_protocol_parameters(&whir_params, num_variables);

            let prover_data = committer
                .commit::<_, F, u64, u64, 4>(
                    &dft,
                    &mut proof,
                    &mut prover_challenger,
                    &mut statement,
                )
                .unwrap();

            let prover = Prover(&params);
            prover
                .prove::<_, F, u64, u64, 4>(
                    &dft,
                    &mut proof,
                    &mut prover_challenger,
                    &statement,
                    prover_data,
                )
                .unwrap();

            let checkpoint_prover: EF = prover_challenger.sample_algebra_element();

            // Verify
            let commitment_reader = CommitmentReader::new(&params);
            let verifier = Verifier::new(&params);

            let inner = HashChallenger::<u8, Keccak256Hash, 32>::new(vec![], Keccak256Hash {});
            let mut verifier_challenger = KeccakChallenger::new(inner);
            domainsep.observe_domain_separator(&mut verifier_challenger);

            let parsed_commitment =
                commitment_reader.parse_commitment::<u64, 4>(&proof, &mut verifier_challenger);

            verifier
                .verify::<F, u64, u64, 4>(
                    &proof,
                    &mut verifier_challenger,
                    &parsed_commitment,
                    verifier_statement,
                )
                .unwrap();

            let checkpoint_verifier: EF = verifier_challenger.sample_algebra_element();
            assert_eq!(checkpoint_prover, checkpoint_verifier);
        }

        #[test]
        fn test_whir_keccak_end_to_end() {
            make_whir_things_keccak(
                10,
                FoldingFactor::Constant(4),
                2,
                SecurityAssumption::CapacityBound,
                0,
                1,
                SumcheckStrategy::default(),
            );
        }
    }
}
