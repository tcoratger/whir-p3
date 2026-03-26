pub mod committer;
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
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_multilinear_util::{evals::EvaluationsList, multilinear::MultilinearPoint};
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::{RngExt, SeedableRng, rngs::SmallRng};

    use crate::{
        fiat_shamir::domain_separator::DomainSeparator,
        parameters::{
            FoldingFactor, ProtocolParameters, SecurityAssumption, SumcheckStrategy, WhirConfig,
        },
        whir::{
            committer::{reader::CommitmentReader, writer::CommitmentWriter},
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

    type PackedF = <F as Field>::Packing;
    type MyMmcs = MerkleTreeMmcs<PackedF, PackedF, MyHash, MyCompress, 2, 8>;

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
        let mmcs = MyMmcs::new(merkle_hash, merkle_compress, 0);

        // Configure WHIR protocol with all security and performance parameters
        let whir_params = ProtocolParameters {
            security_level: 32,
            pow_bits,
            rs_domain_initial_reduction_factor,
            folding_factor,
            mmcs,
            soundness_type,
            starting_log_inv_rate: 1,
        };

        // Create unified configuration combining protocol and polynomial parameters
        let params =
            WhirConfig::<EF, F, MyMmcs, MyChallenger>::new(num_variables, whir_params.clone());

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
        domainsep.commit_statement::<_, _, 8>(&params);
        // Add proof structure to transcript
        domainsep.add_whir_proof::<_, _, 8>(&params);

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
            WhirProof::<F, EF, MyMmcs>::from_protocol_parameters(&whir_params, num_variables);

        // Commit to polynomial evaluations and generate cryptographic witness
        let prover_data = committer
            .commit(&dft, &mut proof, &mut prover_challenger, &mut statement)
            .unwrap();

        // Initialize WHIR prover with the configured parameters
        let prover = Prover(&params);

        // Generate WHIR proof
        prover
            .prove(
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
            .verify(
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
        use rand::{RngExt, SeedableRng, rngs::SmallRng};

        use super::*;
        use crate::parameters::WhirConfig;

        // Field types for Keccak tests
        type F = KoalaBear;
        type EF = BinomialExtensionField<F, 4>;

        // Keccak hash types producing [u64; 4] digests
        type U64Hash = PaddingFreeSponge<KeccakF, 25, 17, 4>;
        type KeccakFieldHash = SerializingHasher<U64Hash>;
        type KeccakCompress = CompressionFunctionFromHasher<U64Hash, 2, 4>;

        // Keccak challenger using byte-based HashChallenger
        type KeccakChallenger = SerializingChallenger32<F, HashChallenger<u8, Keccak256Hash, 32>>;
        type MyMmcs = MerkleTreeMmcs<F, u64, KeccakFieldHash, KeccakCompress, 2, 4>;

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
            let mmcs = MyMmcs::new(merkle_hash, merkle_compress, 0);

            // Configure WHIR protocol with Keccak hashing
            let whir_params = ProtocolParameters {
                security_level: 32,
                pow_bits,
                rs_domain_initial_reduction_factor,
                folding_factor,
                mmcs,
                soundness_type,
                starting_log_inv_rate: 1,
            };

            let params = WhirConfig::<EF, F, MyMmcs, KeccakChallenger>::new(
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
            domainsep.commit_statement::<_, _, 4>(&params);
            domainsep.add_whir_proof::<_, _, 4>(&params);

            // Create prover challenger
            let inner = HashChallenger::<u8, Keccak256Hash, 32>::new(vec![], Keccak256Hash {});
            let mut prover_challenger = KeccakChallenger::new(inner);
            domainsep.observe_domain_separator(&mut prover_challenger);

            // Commit and prove
            let committer = CommitmentWriter::new(&params);
            let dft = Radix2DFTSmallBatch::<F>::default();

            let mut proof =
                WhirProof::<F, EF, MyMmcs>::from_protocol_parameters(&whir_params, num_variables);

            let prover_data = committer
                .commit(&dft, &mut proof, &mut prover_challenger, &mut statement)
                .unwrap();

            let prover = Prover(&params);
            prover
                .prove(
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
                .verify(
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

#[cfg(test)]
mod batch_tests {
    use alloc::vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::{DuplexChallenger, FieldChallenger};
    use p3_field::{Field, PrimeCharacteristicRing, extension::BinomialExtensionField};
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_multilinear_util::{evals::EvaluationsList, multilinear::MultilinearPoint};
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::{RngExt, SeedableRng, rngs::SmallRng};

    use crate::{
        constraints::statement::EqStatement,
        parameters::{FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig},
        sumcheck::SumcheckData,
        whir::{
            committer::{reader::BatchCommitmentReader, writer::BatchCommitmentWriter},
            proof::{BatchWhirProof, WhirProof},
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
    type PackedF = <F as Field>::Packing;
    type MyMmcs = MerkleTreeMmcs<PackedF, PackedF, MyHash, MyCompress, 2, 8>;

    fn make_batch_test_config(
        num_variables: usize,
        folding_factor: FoldingFactor,
    ) -> (
        WhirConfig<EF, F, MyMmcs, MyChallenger>,
        ProtocolParameters<MyMmcs>,
        SmallRng,
    ) {
        let mut rng = SmallRng::seed_from_u64(42);
        let perm = Perm::new_from_rng_128(&mut rng);
        let mmcs = MyMmcs::new(MyHash::new(perm.clone()), MyCompress::new(perm), 0);

        let whir_params = ProtocolParameters {
            security_level: 32,
            pow_bits: 0,
            rs_domain_initial_reduction_factor: 1,
            folding_factor,
            mmcs,
            soundness_type: SecurityAssumption::UniqueDecoding,
            starting_log_inv_rate: 1,
        };

        let params =
            WhirConfig::<EF, F, MyMmcs, MyChallenger>::new(num_variables, whir_params.clone());

        (params, whir_params, rng)
    }

    #[test]
    fn test_selector_round_correctness() {
        let (params, _whir_params, mut rng) = make_batch_test_config(6, FoldingFactor::Constant(3));

        let num_variables = 6;
        let num_evals = 1 << num_variables;
        let poly_a = EvaluationsList::new((0..num_evals).map(|_| rng.random()).collect());
        let poly_b = EvaluationsList::new((0..num_evals).map(|_| rng.random()).collect());

        let z_a = MultilinearPoint::expand_from_univariate(rng.random(), num_variables);
        let z_b = MultilinearPoint::expand_from_univariate(rng.random(), num_variables);
        let v_a: EF = poly_a.evaluate_hypercube_base(&z_a);
        let v_b: EF = poly_b.evaluate_hypercube_base(&z_b);

        let perm = Perm::new_from_rng_128(&mut rng);
        let mut challenger = MyChallenger::new(perm);
        let alpha: EF = challenger.sample_algebra_element();

        let mut selector_data: SumcheckData<F, EF> = SumcheckData::default();

        let prover = Prover(&params);
        let (sumcheck_prover, r_0) = prover.selector_round(
            &mut selector_data,
            &mut challenger,
            &poly_a,
            &poly_b,
            &z_a,
            &z_b,
            v_a,
            v_b,
            alpha,
        );

        // Verify: g should equal r_0·f_a + (1-r_0)·f_b at every point
        let g = sumcheck_prover.evals();
        let one_minus_r0 = EF::ONE - r_0;
        for i in 0..num_evals {
            let expected = r_0 * EF::from(poly_a.as_slice()[i])
                + one_minus_r0 * EF::from(poly_b.as_slice()[i]);
            assert_eq!(g.as_slice()[i], expected, "g mismatch at index {i}");
        }

        // Verify: selector sumcheck stored exactly 1 round of data
        assert_eq!(selector_data.polynomial_evaluations().len(), 1);

        // Verify: c0 == α·v_b (the h(0) value)
        let [c0, _c2] = selector_data.polynomial_evaluations()[0];
        assert_eq!(c0, alpha * v_b);
    }

    /// Run the full batch prove + verify flow and check transcript consistency.
    fn run_batch_prove_verify(num_variables: usize, folding_factor: FoldingFactor) {
        let (params, whir_params, mut rng) = make_batch_test_config(num_variables, folding_factor);
        let dft = p3_dft::Radix2DFTSmallBatch::<F>::default();

        let num_evals = 1 << num_variables;
        let poly_a = EvaluationsList::new((0..num_evals).map(|_| rng.random()).collect());
        let poly_b = EvaluationsList::new((0..num_evals).map(|_| rng.random()).collect());

        let z_a = MultilinearPoint::expand_from_univariate(rng.random(), num_variables);
        let z_b = MultilinearPoint::expand_from_univariate(rng.random(), num_variables);
        let v_a: EF = poly_a.evaluate_hypercube_base(&z_a);
        let v_b: EF = poly_b.evaluate_hypercube_base(&z_b);

        let mut statement_a = EqStatement::initialize(num_variables);
        statement_a.add_evaluated_constraint(z_a, v_a);
        let mut statement_b = EqStatement::initialize(num_variables);
        statement_b.add_evaluated_constraint(z_b, v_b);

        // Both challengers start from the same state
        let perm = Perm::new_from_rng_128(&mut rng);
        let base_challenger = MyChallenger::new(perm);
        let mut prover_challenger = base_challenger.clone();
        let mut verifier_challenger = base_challenger;

        // --- Prover side ---
        let batch_committer = BatchCommitmentWriter::new(&params);
        let mut proof = BatchWhirProof {
            commitment_a: None,
            commitment_b: None,
            initial_ood_answers: [vec![], vec![]],
            selector_sumcheck: SumcheckData::default(),
            inner_proof: WhirProof::<F, EF, MyMmcs>::from_protocol_parameters(
                &whir_params,
                num_variables,
            ),
        };

        let batch_data =
            batch_committer.commit(&dft, &mut proof, &mut prover_challenger, &poly_a, &poly_b);

        let prover = Prover(&params);
        prover
            .batch_prove(
                &dft,
                &mut proof,
                &mut prover_challenger,
                batch_data.prover_data_a,
                batch_data.prover_data_b,
                &poly_a,
                &poly_b,
                &statement_a,
                &statement_b,
                &batch_data.ood_statement_a,
                &batch_data.ood_statement_b,
            )
            .unwrap();

        let checkpoint_prover: EF = prover_challenger.sample_algebra_element();

        // --- Verifier side ---
        let batch_reader = BatchCommitmentReader::new(&params);
        let (parsed_a, parsed_b) =
            batch_reader.parse_batch_commitment(&proof, &mut verifier_challenger);

        let verifier = Verifier::new(&params);
        let (_folding_randomness, _r_0) = verifier
            .batch_verify(
                &proof,
                &mut verifier_challenger,
                &parsed_a,
                &parsed_b,
                &statement_a,
                &statement_b,
            )
            .unwrap();

        let checkpoint_verifier: EF = verifier_challenger.sample_algebra_element();
        assert_eq!(checkpoint_prover, checkpoint_verifier);
    }

    #[test]
    fn test_batch_prove_verify_folding_4() {
        run_batch_prove_verify(10, FoldingFactor::Constant(4));
    }

    #[test]
    fn test_batch_prove_verify_folding_2() {
        run_batch_prove_verify(8, FoldingFactor::Constant(2));
    }

    #[test]
    fn test_batch_prove_verify_folding_3() {
        run_batch_prove_verify(9, FoldingFactor::Constant(3));
    }

    #[test]
    fn test_batch_prove_verify_mixed_folding() {
        run_batch_prove_verify(10, FoldingFactor::ConstantFromSecondRound(4, 2));
    }

    #[cfg(not(debug_assertions))]
    mod performance {
        extern crate std;

        use std::time::Instant;

        use super::*;
        use crate::{parameters::SumcheckStrategy, whir::committer::writer::CommitmentWriter};

        /// Run a single-polynomial commit + prove cycle. Returns elapsed time.
        #[allow(clippy::too_many_arguments)]
        fn single_poly_prove(
            params: &WhirConfig<EF, F, MyMmcs, MyChallenger>,
            whir_params: &ProtocolParameters<MyMmcs>,
            dft: &p3_dft::Radix2DFTSmallBatch<F>,
            poly: EvaluationsList<F>,
            num_variables: usize,
            num_points: usize,
            rng: &mut SmallRng,
        ) -> std::time::Duration {
            let mut statement = params.initial_statement(poly, SumcheckStrategy::default());
            for _ in 0..num_points {
                let point = MultilinearPoint::expand_from_univariate(rng.random(), num_variables);
                let _ = statement.evaluate(&point);
            }

            let perm = Perm::new_from_rng_128(rng);
            let mut challenger = MyChallenger::new(perm);

            let committer = CommitmentWriter::new(params);
            let mut proof =
                WhirProof::<F, EF, MyMmcs>::from_protocol_parameters(whir_params, num_variables);

            let start = Instant::now();

            let prover_data = committer
                .commit(dft, &mut proof, &mut challenger, &mut statement)
                .unwrap();

            let prover = Prover(params);
            prover
                .prove(dft, &mut proof, &mut challenger, &statement, prover_data)
                .unwrap();

            start.elapsed()
        }

        /// Run a batch prove cycle for two polynomials. Returns elapsed time.
        #[allow(clippy::too_many_arguments)]
        fn batch_poly_prove(
            params: &WhirConfig<EF, F, MyMmcs, MyChallenger>,
            whir_params: &ProtocolParameters<MyMmcs>,
            dft: &p3_dft::Radix2DFTSmallBatch<F>,
            poly_a: &EvaluationsList<F>,
            poly_b: &EvaluationsList<F>,
            num_variables: usize,
            rng: &mut SmallRng,
        ) -> std::time::Duration {
            let z_a = MultilinearPoint::expand_from_univariate(rng.random(), num_variables);
            let z_b = MultilinearPoint::expand_from_univariate(rng.random(), num_variables);
            let v_a: EF = poly_a.evaluate_hypercube_base(&z_a);
            let v_b: EF = poly_b.evaluate_hypercube_base(&z_b);

            let mut statement_a = EqStatement::initialize(num_variables);
            statement_a.add_evaluated_constraint(z_a, v_a);
            let mut statement_b = EqStatement::initialize(num_variables);
            statement_b.add_evaluated_constraint(z_b, v_b);

            let perm = Perm::new_from_rng_128(rng);
            let mut challenger = MyChallenger::new(perm);

            let batch_committer = BatchCommitmentWriter::new(params);
            let mut proof = BatchWhirProof {
                commitment_a: None,
                commitment_b: None,
                initial_ood_answers: [vec![], vec![]],
                selector_sumcheck: SumcheckData::default(),
                inner_proof: WhirProof::<F, EF, MyMmcs>::from_protocol_parameters(
                    whir_params,
                    num_variables,
                ),
            };

            let start = Instant::now();

            let batch_data =
                batch_committer.commit(dft, &mut proof, &mut challenger, poly_a, poly_b);

            let prover = Prover(params);
            prover
                .batch_prove(
                    dft,
                    &mut proof,
                    &mut challenger,
                    batch_data.prover_data_a,
                    batch_data.prover_data_b,
                    poly_a,
                    poly_b,
                    &statement_a,
                    &statement_b,
                    &batch_data.ood_statement_a,
                    &batch_data.ood_statement_b,
                )
                .unwrap();

            start.elapsed()
        }

        #[test]
        fn test_batch_vs_separate_performance() {
            const ITERS: usize = 3;

            let num_variables = 20;
            let folding_factor = FoldingFactor::Constant(4);
            let (params, whir_params, mut rng) =
                make_batch_test_config(num_variables, folding_factor);
            let dft = p3_dft::Radix2DFTSmallBatch::<F>::default();

            let num_evals = 1 << num_variables;
            let poly_a = EvaluationsList::new((0..num_evals).map(|_| rng.random()).collect());
            let poly_b = EvaluationsList::new((0..num_evals).map(|_| rng.random()).collect());

            let mut separate_total = std::time::Duration::ZERO;
            for _ in 0..ITERS {
                let t1 = single_poly_prove(
                    &params,
                    &whir_params,
                    &dft,
                    poly_a.clone(),
                    num_variables,
                    1,
                    &mut rng,
                );
                let t2 = single_poly_prove(
                    &params,
                    &whir_params,
                    &dft,
                    poly_b.clone(),
                    num_variables,
                    1,
                    &mut rng,
                );
                separate_total += t1 + t2;
            }

            let mut batch_total = std::time::Duration::ZERO;
            for _ in 0..ITERS {
                let t = batch_poly_prove(
                    &params,
                    &whir_params,
                    &dft,
                    &poly_a,
                    &poly_b,
                    num_variables,
                    &mut rng,
                );
                batch_total += t;
            }

            let separate_avg = separate_total / ITERS as u32;
            let batch_avg = batch_total / ITERS as u32;
            let speedup = separate_avg.as_secs_f64() / batch_avg.as_secs_f64();

            std::eprintln!();
            std::eprintln!(
                "=== Batch vs Separate Performance ({num_variables} variables, {ITERS} iterations) ==="
            );
            std::eprintln!("  Two separate proofs: {separate_avg:?} avg");
            std::eprintln!("  One batch proof:     {batch_avg:?} avg");
            std::eprintln!("  Speedup:             {speedup:.2}x");
            std::eprintln!();

            assert!(
                batch_avg < separate_avg,
                "Batch proof ({batch_avg:?}) should be faster than two separate proofs ({separate_avg:?})"
            );
        }
    }
}
