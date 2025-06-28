use itertools::{Itertools, izip};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::{DuplexChallenger, FieldChallenger};
use p3_commit::Mmcs;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, PrimeField64, TwoAdicField, extension::BinomialExtensionField};
use p3_matrix::{
    Dimensions, Matrix,
    dense::{RowMajorMatrix, RowMajorMatrixView},
    horizontally_truncated::HorizontallyTruncated,
};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{
    CryptographicHasher, PaddingFreeSponge, PseudoCompressionFunction, TruncatedPermutation,
};
use p3_field::integers::QuotientMap;

use rand::{SeedableRng, rngs::SmallRng};
use serde::{Serialize, de::DeserializeOwned};
use std::{iter::repeat_with, marker::PhantomData, sync::Arc};
use tracing::info_span;

use crate::{
    dft::EvalsDft,
    fiat_shamir::{domain_separator::DomainSeparator, errors::ProofError},
    parameters::{MultivariateParameters, ProtocolParameters},
    poly::{
        coeffs::CoefficientList, evals::EvaluationsList, multilinear::MultilinearPoint,
        wavelet::Radix2WaveletKernel,
    },
    whir::{
        committer::{Witness, reader::ParsedCommitment},
        parameters::WhirConfig,
        pcs::{
            MlPcs,
            proof::WhirProof,
            prover_data::{ConcatMats, ConcatMatsMeta},
            query::MlQuery,
        },
        prover::Prover,
        statement::Statement,
        verifier::Verifier,
    },
};

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type Perm = Poseidon2BabyBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

#[derive(Debug)]
pub struct WhirPcs<Val, Dft, Hash, Compression, const DIGEST_ELEMS: usize> {
    dft: Dft,
    whir: ProtocolParameters<Hash, Compression>,
    _phantom: PhantomData<Val>,
}

impl<Val, Dft, Hash, Compression, const DIGEST_ELEMS: usize>
    WhirPcs<Val, Dft, Hash, Compression, DIGEST_ELEMS>
{
    pub const fn new(dft: Dft, whir: ProtocolParameters<Hash, Compression>) -> Self {
        Self {
            dft,
            whir,
            _phantom: PhantomData,
        }
    }
}

impl<Val, Dft, Hash, Compression, Challenge, Challenger, const DIGEST_ELEMS: usize>
    MlPcs<Challenge, Challenger> for WhirPcs<Val, Dft, Hash, Compression, DIGEST_ELEMS>
where
    Val: TwoAdicField + PrimeField64,
    Dft: TwoAdicSubgroupDft<Val>,
    Hash: Sync + CryptographicHasher<Val, [Val; DIGEST_ELEMS]>,
    Compression: Sync + PseudoCompressionFunction<[Val; DIGEST_ELEMS], 2>,
    Challenge: TwoAdicField + ExtensionField<Val>,
    Challenger: FieldChallenger<Val>,
    [Val; DIGEST_ELEMS]: Serialize + DeserializeOwned,
{
    type Val = Val;

    type Commitment =
        <MerkleTreeMmcs<Val, Val, Hash, Compression, DIGEST_ELEMS> as Mmcs<Val>>::Commitment;

    type ProverData = (
        ConcatMats<Val>,
        Arc<
            <MerkleTreeMmcs<Val, Val, Hash, Compression, DIGEST_ELEMS> as Mmcs<Val>>::ProverData<
                RowMajorMatrix<Val>,
            >,
        >,
    );

    type Evaluations<'a> = HorizontallyTruncated<Val, RowMajorMatrixView<'a, Val>>;

    type Proof = Vec<WhirProof<Challenge>>;

    type Error = ProofError;

    fn commit(
        &self,
        evaluations: Vec<RowMajorMatrix<Self::Val>>,
    ) -> (Self::Commitment, Self::ProverData) {
        let concat_mats = info_span!("concat matrices").in_scope(|| ConcatMats::new(evaluations));

        let (commitment, merkle_tree) = {
            let coeffs = info_span!("evals to coeffs").in_scope(|| {
                let size = 1 << (concat_mats.meta.log_b + self.whir.starting_log_inv_rate);
                let mut evals = Vec::with_capacity(size);
                evals.extend(&concat_mats.values);
                let mut coeffs =
                    Radix2WaveletKernel::default().inverse_wavelet_transform_algebra(evals);
                coeffs.resize(size, Val::ZERO);
                coeffs
            });

            let folded_codeword = info_span!("compute folded codeword").in_scope(|| {
                let width = 1 << self.whir.folding_factor.at_round(0);
                let folded_coeffs = RowMajorMatrix::new(coeffs, width);
                self.dft.dft_batch(folded_coeffs).to_row_major_matrix()
            });

            let mmcs = MerkleTreeMmcs::new(
                self.whir.merkle_hash.clone(),
                self.whir.merkle_compress.clone(),
            );

            mmcs.commit_matrix(folded_codeword)
        };

        (commitment, (concat_mats, Arc::new(merkle_tree)))
    }

    fn get_evaluations<'a>(
        &self,
        (concat_mats, _): &'a Self::ProverData,
        idx: usize,
    ) -> Self::Evaluations<'a> {
        concat_mats.mat(idx)
    }

    fn open(
        &self,
        rounds: Vec<(
            &Self::ProverData,
            Vec<Vec<(MlQuery<Challenge>, Vec<Challenge>)>>,
        )>,
        challenger: &mut Challenger,
    ) -> Self::Proof {
        rounds
            .iter()
            .map(|((concat_mats, merkle_tree), queries_and_evals)| {
      
                let config = {
                    let mut rng = SmallRng::seed_from_u64(1);
                    let perm = Perm::new_from_rng_128(&mut rng);
                    let merkle_hash = MyHash::new(perm.clone());
                    let merkle_compress = MyCompress::new(perm);

                    let whir_params = ProtocolParameters {
                        initial_statement: self.whir.initial_statement,
                        security_level: self.whir.security_level,
                        pow_bits: self.whir.pow_bits,
                        folding_factor: self.whir.folding_factor,
                        merkle_hash,
                        merkle_compress,
                        soundness_type: self.whir.soundness_type,
                        starting_log_inv_rate: self.whir.starting_log_inv_rate,
                        rs_domain_initial_reduction_factor: self
                            .whir
                            .rs_domain_initial_reduction_factor,
                        univariate_skip: self.whir.univariate_skip,
                    };

                    WhirConfig::<EF, F, MyHash, MyCompress, MyChallenger>::new(
                        MultivariateParameters::new(concat_mats.meta.log_b),
                        whir_params,
                    )
                };

                let pol_evals = EvaluationsList::new(concat_mats.values.clone());
                let pol_coeffs = info_span!("evals to coeffs").in_scope(|| {
                    CoefficientList::new(
                        Radix2WaveletKernel::default()
                            .inverse_wavelet_transform_algebra(pol_evals.evals().to_vec()),
                    )
                });

                let (ood_points, ood_answers) = info_span!("compute ood answers").in_scope(|| {
                    repeat_with(|| {
                        let ood_point: Challenge = challenger.sample_algebra_element();
                        let ood_answer =
                            pol_coeffs.evaluate(&MultilinearPoint::expand_from_univariate(
                                ood_point,
                                concat_mats.meta.log_b,
                            ));
                        (ood_point, ood_answer)
                    })
                    .take(config.committment_ood_samples)
                    .collect::<(Vec<_>, Vec<_>)>()
                });

                let r = repeat_with(|| challenger.sample_algebra_element::<Challenge>())
                    .take(concat_mats.meta.max_log_width())
                    .collect_vec();

                let statement = info_span!("compute weights").in_scope(|| {
                    let mut statement = Statement::new(concat_mats.meta.log_b);
                    queries_and_evals
                        .iter()
                        .enumerate()
                        .for_each(|(idx, queries_and_evals)| {
                            for (query, evals) in queries_and_evals {
                                let (weights, sum) =
                                    concat_mats.meta.constraint(idx, query, evals, &r);
                                statement.add_constraint(weights, sum);
                            }
                        });
                    statement
                });

                let mut prover_state = {
                    let mut domainsep = DomainSeparator::new(vec![]);
                    domainsep.add_whir_proof::<_, _, _, DIGEST_ELEMS>(&config);

                    let mut rng = SmallRng::seed_from_u64(1);
                    let challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));
                    domainsep.to_prover_state(challenger)
                };

                info_span!("prove").in_scope(|| {
                    let witness = Witness {
                        polynomial: pol_evals,
                        prover_data: Arc::clone(merkle_tree),
                        ood_points,
                        ood_answers: ood_answers.clone(),
                    };
                    
               
                    let dft = EvalsDft::<F>::new(1 << config.max_fft_size());
                    
                    let prover = Prover(&config);
                    prover
                        .prove(&dft, &mut prover_state, statement, witness)
                        .unwrap();

                    WhirProof {
                        ood_answers,
                        narg_string: prover_state.narg_string(),
                    }
                })
            })
            .collect()
    }

    fn verify(
        &self,
        rounds: Vec<(
            Self::Commitment,
            Vec<Vec<(MlQuery<Challenge>, Vec<Challenge>)>>,
        )>,
        proofs: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        izip!(rounds, proofs).try_for_each(|((commitment, round), proof)| {
            let concat_mats_meta = ConcatMatsMeta::new(
                &round
                    .iter()
                    .map(|mat| Dimensions {
                        width: mat[0].1.len(),
                        height: 1 << mat[0].0.log_b(),
                    })
                    .collect_vec(),
            );
            let config = {
                let mut rng = SmallRng::seed_from_u64(1);
                let perm = Perm::new_from_rng_128(&mut rng);
                let merkle_hash = MyHash::new(perm.clone());
                let merkle_compress = MyCompress::new(perm);

                let whir_params = ProtocolParameters {
                    initial_statement: self.whir.initial_statement,
                    security_level: self.whir.security_level,
                    pow_bits: self.whir.pow_bits,
                    folding_factor: self.whir.folding_factor,
                    merkle_hash,
                    merkle_compress,
                    soundness_type: self.whir.soundness_type,
                    starting_log_inv_rate: self.whir.starting_log_inv_rate,
                    rs_domain_initial_reduction_factor: self
                        .whir
                        .rs_domain_initial_reduction_factor,
                    univariate_skip: self.whir.univariate_skip,
                };

                WhirConfig::<EF, F, MyHash, MyCompress, MyChallenger>::new(
                    MultivariateParameters::new(concat_mats_meta.log_b),
                    whir_params,
                )
            };

            let ood_points = repeat_with(|| challenger.sample_algebra_element::<Challenge>())
                .take(config.committment_ood_samples)
                .collect_vec();

            let r = repeat_with(|| challenger.sample_algebra_element::<Challenge>())
                .take(concat_mats_meta.max_log_width())
                .collect_vec();

            let mut statement = Statement::new(concat_mats_meta.log_b);
            round.iter().enumerate().for_each(|(idx, evals)| {
                for (query, evals) in evals {
                    let (weights, sum) = concat_mats_meta.constraint(idx, query, evals, &r);
                    statement.add_constraint(weights, sum);
                }
            });

            let mut verifier_state = {
                let tornado_pattern = "🌪️"  
                        .bytes()  
                        .map(|b| F::from_canonical_checked(b).unwrap())  
                        .collect::<Vec<_>>();
                let mut domainsep = DomainSeparator::new(tornado_pattern);
                domainsep.add_whir_proof::<_, _, _, DIGEST_ELEMS>(&config);

                let mut rng = SmallRng::seed_from_u64(1);
                let challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));
                domainsep.to_verifier_state(proof.narg_string.clone(), challenger)
            };

            Verifier::new(&config).verify(
                &mut verifier_state,
                &ParsedCommitment {
                    num_variables: concat_mats_meta.log_b,
                    root: commitment,
                    ood_points,
                    ood_answers: proof.ood_answers.clone(),
                },
                &statement,
            )?;
            Ok(())
        })
    }
}
