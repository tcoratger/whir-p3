// use std::{cell::RefCell, marker::PhantomData};

// use p3_challenger::FieldChallenger;
// use p3_commit::Mmcs;
// use p3_dft::TwoAdicSubgroupDft;
// use p3_field::{ExtensionField, PrimeField64, TwoAdicField};
// use p3_matrix::{
//     dense::{RowMajorMatrix, RowMajorMatrixView},
//     horizontally_truncated::HorizontallyTruncated,
// };
// use p3_merkle_tree::MerkleTreeMmcs;
// use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
// use serde::{Serialize, de::DeserializeOwned};
// use tracing::info_span;

// use crate::{fiat_shamir::errors::ProofError, parameters::ProtocolParameters, whir::pcs::MlPcs};

// #[derive(Debug)]
// pub struct WhirPcs<Val, Dft, Hash, Compression, const DIGEST_ELEMS: usize> {
//     dft: Dft,
//     whir: ProtocolParameters<Hash, Compression>,
//     _phantom: PhantomData<Val>,
// }

// impl<Val, Dft, Hash, Compression, const DIGEST_ELEMS: usize>
//     WhirPcs<Val, Dft, Hash, Compression, DIGEST_ELEMS>
// {
//     pub const fn new(dft: Dft, whir: ProtocolParameters<Hash, Compression>) -> Self {
//         Self {
//             dft,
//             whir,
//             _phantom: PhantomData,
//         }
//     }
// }

// impl<Val, Dft, Hash, Compression, Challenge, Challenger, const DIGEST_ELEMS: usize>
//     MlPcs<Challenge, Challenger> for WhirPcs<Val, Dft, Hash, Compression, DIGEST_ELEMS>
// where
//     Val: TwoAdicField + PrimeField64,
//     Dft: TwoAdicSubgroupDft<Val>,
//     Hash: Sync + CryptographicHasher<Val, [u8; DIGEST_ELEMS]>,
//     Compression: Sync + PseudoCompressionFunction<[u8; DIGEST_ELEMS], 2>,
//     Challenge: TwoAdicField + ExtensionField<Val>,
//     Challenger: FieldChallenger<Val>,
//     [u8; DIGEST_ELEMS]: Serialize + DeserializeOwned,
// {
//     type Val = Val;
//     type Commitment =
//         <MerkleTreeMmcs<Val, u8, Hash, Compression, DIGEST_ELEMS> as Mmcs<Val>>::Commitment;
//     type ProverData = (
//         ConcatMats<Val>,
//         // TODO(whir-p3): Use reference to merkle tree in `Witness` to avoid cloning or ownership taking.
//         RefCell<
//             Option<
//                 <MerkleTreeMmcs<Val, u8, Hash, Compression, DIGEST_ELEMS> as Mmcs<Val>>::ProverData<
//                     RowMajorMatrix<Val>,
//                 >,
//             >,
//         >,
//     );
//     type Evaluations<'a> = HorizontallyTruncated<Val, RowMajorMatrixView<'a, Val>>;
//     type Proof = Vec<WhirProof<Challenge>>;
//     type Error = ProofError;

//     fn commit(
//         &self,
//         evaluations: Vec<RowMajorMatrix<Self::Val>>,
//     ) -> (Self::Commitment, Self::ProverData) {
//         // Concat matrices into single polynomial.
//         let concat_mats = info_span!("concat matrices").in_scope(|| ConcatMats::new(evaluations));

//         // This should generate the same codeword and commitment as in `whir_p3`.
//         let (commitment, merkle_tree) = {
//             let coeffs = info_span!("evals to coeffs").in_scope(|| {
//                 let size = 1 << (concat_mats.meta.log_b + self.whir.starting_log_inv_rate);
//                 let mut evals = Vec::with_capacity(size);
//                 evals.extend(&concat_mats.values);
//                 let mut coeffs =
//                     Radix2WaveletKernel::default().inverse_wavelet_transform_algebra(evals);
//                 coeffs.resize(size, Val::ZERO);
//                 coeffs
//             });
//             let folded_codeword = info_span!("compute folded codeword").in_scope(|| {
//                 let width = 1 << self.whir.folding_factor.at_round(0);
//                 let folded_coeffs = RowMajorMatrix::new(coeffs, width);
//                 self.dft.dft_batch(folded_coeffs).to_row_major_matrix()
//             });
//             let mmcs = MerkleTreeMmcs::new(
//                 self.whir.merkle_hash.clone(),
//                 self.whir.merkle_compress.clone(),
//             );
//             mmcs.commit(vec![folded_codeword])
//         };

//         (commitment, (concat_mats, RefCell::new(Some(merkle_tree))))
//     }

//     fn get_evaluations<'a>(
//         &self,
//         (concat_mats, _): &'a Self::ProverData,
//         idx: usize,
//     ) -> Self::Evaluations<'a> {
//         concat_mats.mat(idx)
//     }

//     fn open(
//         &self,
//         // For each round,
//         rounds: Vec<(
//             &Self::ProverData,
//             // for each matrix,
//             Vec<
//                 // for each query:
//                 Vec<(
//                     // the query,
//                     MlQuery<Challenge>,
//                     // values at the query
//                     Vec<Challenge>,
//                 )>,
//             >,
//         )>,
//         challenger: &mut Challenger,
//     ) -> Self::Proof {
//         rounds
//             .iter()
//             .map(|((concat_mats, merkle_tree), queries_and_evals)| {
//                 let config = WhirConfig::<
//                     Challenge,
//                     Val,
//                     Hash,
//                     Compression,
//                     Blake3PoW,
//                     KeccakF,
//                     Keccak,
//                     u8,
//                     KECCAK_WIDTH_BYTES,
//                 >::new(
//                     MultivariateParameters::new(concat_mats.meta.log_b),
//                     self.whir.clone(),
//                 );

//                 let pol_evals = EvaluationsList::new(concat_mats.values.clone());
//                 let pol_coeffs = info_span!("evals to coeffs").in_scope(|| {
//                     CoefficientList::new(
//                         Radix2WaveletKernel::default()
//                             .inverse_wavelet_transform_algebra(pol_evals.evals().to_vec()),
//                     )
//                 });
//                 let (ood_points, ood_answers) = info_span!("compute ood answers").in_scope(|| {
//                     repeat_with(|| {
//                         let ood_point: Challenge = challenger.sample_algebra_element();
//                         let ood_answer =
//                             pol_coeffs.evaluate(&MultilinearPoint::expand_from_univariate(
//                                 ood_point,
//                                 concat_mats.meta.log_b,
//                             ));
//                         (ood_point, ood_answer)
//                     })
//                     .take(config.committment_ood_samples)
//                     .collect::<(Vec<_>, Vec<_>)>()
//                 });

//                 // Challenge for random linear combining columns.
//                 let r = repeat_with(|| challenger.sample_algebra_element::<Challenge>())
//                     .take(concat_mats.meta.max_log_width())
//                     .collect_vec();

//                 let statement = info_span!("compute weights").in_scope(|| {
//                     let mut statement = Statement::new(concat_mats.meta.log_b);
//                     queries_and_evals
//                         .iter()
//                         .enumerate()
//                         .for_each(|(idx, queries_and_evals)| {
//                             queries_and_evals.iter().for_each(|(query, evals)| {
//                                 let (weights, sum) =
//                                     concat_mats.meta.constraint(idx, query, evals, &r);
//                                 statement.add_constraint(weights, sum);
//                             })
//                         });
//                     statement
//                 });

//                 let mut prover_state = {
//                     let mut domainsep = DomainSeparator::new("🌪️", KeccakF);
//                     domainsep.add_whir_proof(&config);
//                     domainsep.to_prover_state::<_, 32>()
//                 };
//                 info_span!("prove").in_scope(|| {
//                     let witness = Witness {
//                         pol_coeffs,
//                         pol_evals,
//                         prover_data: merkle_tree.take().unwrap(),
//                         ood_points,
//                         ood_answers: ood_answers.clone(),
//                     };
//                     Prover(&config)
//                         .prove(&self.dft, &mut prover_state, statement, witness)
//                         .unwrap()
//                 });
//                 WhirProof {
//                     ood_answers,
//                     narg_string: prover_state.narg_string().to_vec(),
//                 }
//             })
//             .collect()
//     }

//     fn verify(
//         &self,
//         // For each round:
//         rounds: Vec<(
//             Self::Commitment,
//             // for each matrix:
//             Vec<
//                 // for each query:
//                 Vec<(
//                     // the query,
//                     MlQuery<Challenge>,
//                     // values at the query
//                     Vec<Challenge>,
//                 )>,
//             >,
//         )>,
//         proofs: &Self::Proof,
//         challenger: &mut Challenger,
//     ) -> Result<(), Self::Error> {
//         izip!(rounds, proofs).try_for_each(|((commitment, round), proof)| {
//             let concat_mats_meta = ConcatMatsMeta::new(
//                 round
//                     .iter()
//                     .map(|mat| Dimensions {
//                         width: mat[0].1.len(),
//                         height: 1 << mat[0].0.log_b(),
//                     })
//                     .collect(),
//             );

//             let config = WhirConfig::<
//                 Challenge,
//                 Val,
//                 Hash,
//                 Compression,
//                 Blake3PoW,
//                 KeccakF,
//                 Keccak,
//                 u8,
//                 KECCAK_WIDTH_BYTES,
//             >::new(
//                 MultivariateParameters::new(concat_mats_meta.log_b),
//                 self.whir.clone(),
//             );

//             let ood_points = repeat_with(|| challenger.sample_algebra_element::<Challenge>())
//                 .take(config.committment_ood_samples)
//                 .collect_vec();

//             let r = repeat_with(|| challenger.sample_algebra_element::<Challenge>())
//                 .take(concat_mats_meta.max_log_width())
//                 .collect_vec();

//             let mut statement = Statement::new(concat_mats_meta.log_b);
//             round.iter().enumerate().for_each(|(idx, evals)| {
//                 evals.iter().for_each(|(query, evals)| {
//                     let (weights, sum) = concat_mats_meta.constraint(idx, query, evals, &r);
//                     statement.add_constraint(weights, sum);
//                 })
//             });

//             let mut verifier_state = {
//                 let mut domainsep = DomainSeparator::new("🌪️", KeccakF);
//                 domainsep.add_whir_proof(&config);
//                 domainsep.to_verifier_state::<_, 32>(&proof.narg_string)
//             };
//             Verifier::new(&config).verify(
//                 &mut verifier_state,
//                 &ParsedCommitment {
//                     num_variables: concat_mats_meta.log_b,
//                     root: commitment,
//                     ood_points,
//                     ood_answers: proof.ood_answers.clone(),
//                 },
//                 &statement,
//             )?;
//             Ok(())
//         })
//     }
// }
