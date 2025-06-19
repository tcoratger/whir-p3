// use std::{cell::RefCell, iter::repeat_with, marker::PhantomData};

// use itertools::{Itertools, izip};
// use p3_challenger::{FieldChallenger, HashChallenger};
// use p3_commit::Mmcs;
// use p3_dft::TwoAdicSubgroupDft;
// use p3_field::{ExtensionField, PrimeField64, TwoAdicField};
// use p3_keccak::Keccak256Hash;
// use p3_matrix::{
//     Dimensions, Matrix,
//     dense::{RowMajorMatrix, RowMajorMatrixView},
//     horizontally_truncated::HorizontallyTruncated,
// };
// use p3_merkle_tree::MerkleTreeMmcs;
// use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
// use serde::{Serialize, de::DeserializeOwned};
// use tracing::info_span;

// use crate::{
//     dft::EvalsDft,
//     fiat_shamir::{domain_separator::DomainSeparator, errors::ProofError, pow::blake3::Blake3PoW},
//     parameters::{MultivariateParameters, ProtocolParameters},
//     poly::{
//         coeffs::CoefficientList, evals::EvaluationsList, multilinear::MultilinearPoint,
//         wavelet::Radix2WaveletKernel,
//     },
//     whir::{
//         committer::{Witness, reader::ParsedCommitment},
//         parameters::WhirConfig,
//         pcs::{
//             MlPcs,
//             proof::WhirProof,
//             prover_data::{ConcatMats, ConcatMatsMeta},
//             query::MlQuery,
//         },
//         prover::Prover,
//         statement::Statement,
//         verifier::Verifier,
//     },
// };

// type MyChallenger = HashChallenger<u8, Keccak256Hash, 32>;

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
//         // Flatten the input list of evaluation matrices into a single dense matrix.
//         //
//         // We concatenate matrices into a single polynomial.
//         let concat_mats = info_span!("concat matrices").in_scope(|| ConcatMats::new(evaluations));

//         // Build the Merkle tree commitment and recover the internal tree for later proving.
//         let (commitment, merkle_tree) = {
//             // Recover the full evaluation vector and perform an inverse wavelet transform.
//             let coeffs = info_span!("evals to coeffs").in_scope(|| {
//                 // Determine the padded size of the evaluation vector.
//                 let size = 1 << (concat_mats.meta.log_b + self.whir.starting_log_inv_rate);

//                 // Copy the flattened evaluation values.
//                 let mut evals = Vec::with_capacity(size);
//                 evals.extend(&concat_mats.values);

//                 // Apply the inverse wavelet transform to obtain polynomial coefficients.
//                 let mut coeffs =
//                     Radix2WaveletKernel::default().inverse_wavelet_transform_algebra(evals);

//                 // Pad with zeros to the required power-of-two size (if needed).
//                 coeffs.resize(size, Val::ZERO);
//                 coeffs
//             });

//             // Fold the coefficient vector into a matrix by grouping columns for FFT.
//             let folded_codeword = info_span!("compute folded codeword").in_scope(|| {
//                 // The width of the folded matrix is determined by the folding factor at round 0.
//                 let width = 1 << self.whir.folding_factor.at_round(0);

//                 // Reshape the coefficient vector into a row-major matrix.
//                 let folded_coeffs = RowMajorMatrix::new(coeffs, width);

//                 // Apply the DFT to each row of the folded matrix and convert back to row-major layout.
//                 self.dft.dft_batch(folded_coeffs).to_row_major_matrix()
//             });

//             // Commit to the folded codeword using a Merkle tree commitment scheme.
//             let mmcs = MerkleTreeMmcs::new(
//                 self.whir.merkle_hash.clone(),
//                 self.whir.merkle_compress.clone(),
//             );

//             // Returns (commitment root, full Merkle tree).
//             mmcs.commit_matrix(folded_codeword)
//         };

//         //  Return the Merkle commitment root along with the prover data.
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
//                     MyChallenger,
//                     u8,
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
//                             for (query, evals) in queries_and_evals {
//                                 let (weights, sum) =
//                                     concat_mats.meta.constraint(idx, query, evals, &r);
//                                 statement.add_constraint(weights, sum);
//                             }
//                         });
//                     statement
//                 });

//                 let mut prover_state = {
//                     let mut domainsep = DomainSeparator::new("🌪️", true);
//                     domainsep.add_whir_proof(&config);
//                     domainsep.to_prover_state(MyChallenger::new(vec![], Keccak256Hash))
//                 };
//                 info_span!("prove").in_scope(|| {
//                     let witness = Witness {
//                         polynomial: pol_evals,
//                         prover_data: merkle_tree.take().unwrap(),
//                         ood_points,
//                         ood_answers: ood_answers.clone(),
//                     };

//                     // Construct a Radix-2 FFT backend that supports small batch DFTs over `F`.
//                     let dft = EvalsDft::<Val>::new(1 << config.max_fft_size());

//                     Prover(&config)
//                         .prove(&dft, &mut prover_state, statement, witness)
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
//                 &round
//                     .iter()
//                     .map(|mat| Dimensions {
//                         width: mat[0].1.len(),
//                         height: 1 << mat[0].0.log_b(),
//                     })
//                     .collect_vec(),
//             );

//             let config =
//                 WhirConfig::<Challenge, Val, Hash, Compression, Blake3PoW, MyChallenger, u8>::new(
//                     MultivariateParameters::new(concat_mats_meta.log_b),
//                     self.whir.clone(),
//                 );

//             let ood_points = repeat_with(|| challenger.sample_algebra_element::<Challenge>())
//                 .take(config.committment_ood_samples)
//                 .collect_vec();

//             let r = repeat_with(|| challenger.sample_algebra_element::<Challenge>())
//                 .take(concat_mats_meta.max_log_width())
//                 .collect_vec();

//             println!("avant constraint");

//             let mut statement = Statement::new(concat_mats_meta.log_b);
//             round.iter().enumerate().for_each(|(idx, evals)| {
//                 for (query, evals) in evals {
//                     let (weights, sum) = concat_mats_meta.constraint(idx, query, evals, &r);
//                     statement.add_constraint(weights, sum);
//                 }
//             });

//             println!("apres constraint");

//             let mut verifier_state = {
//                 let mut domainsep = DomainSeparator::new("🌪️", true);
//                 domainsep.add_whir_proof(&config);
//                 domainsep
//                     .to_verifier_state(&proof.narg_string, MyChallenger::new(vec![], Keccak256Hash))
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
