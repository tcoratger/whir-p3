// use std::iter::repeat_with;

// use itertools::{Itertools, izip};
// use p3_challenger::{CanObserve, FieldChallenger, HashChallenger, SerializingChallenger32};
// use p3_dft::Radix2DitParallel;
// use p3_field::{ExtensionField, Field, extension::BinomialExtensionField};
// use p3_keccak::Keccak256Hash;
// use p3_koala_bear::KoalaBear;
// use p3_matrix::{Matrix, dense::RowMajorMatrix};
// use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
// use rand::{
//     Rng, SeedableRng,
//     distr::{Distribution, StandardUniform},
//     rngs::StdRng,
// };

// use crate::{
//     parameters::{FoldingFactor, ProtocolParameters, errors::SecurityAssumption},
//     poly::multilinear::MultilinearPoint,
//     whir::pcs::{MlPcs, query::MlQuery, whir::WhirPcs},
// };

// fn seeded_rng() -> impl Rng {
//     StdRng::seed_from_u64(0)
// }

// fn do_test_whir_pcs<Val, Challenge, Challenger, P>(
//     (pcs, challenger): &(P, Challenger),
//     log_bs_by_round: &[&[usize]],
// ) where
//     P: MlPcs<Challenge, Challenger, Val = Val>,
//     Val: Field,
//     StandardUniform: Distribution<Val>,
//     Challenge: ExtensionField<Val>,
//     Challenger: Clone + CanObserve<P::Commitment> + FieldChallenger<Val>,
// {
//     let num_rounds = log_bs_by_round.len();
//     let mut rng = seeded_rng();

//     let mut p_challenger = challenger.clone();

//     let log_bs_and_polys_by_round = log_bs_by_round
//         .iter()
//         .map(|log_bs| {
//             log_bs
//                 .iter()
//                 .map(|&log_b| {
//                     let height = 1 << log_b;
//                     // random width 5-15
//                     let width = 5 + rng.random_range(0..=10);
//                     (log_b, RowMajorMatrix::<Val>::rand(&mut rng, height, width))
//                 })
//                 .collect_vec()
//         })
//         .collect_vec();

//     let (commits_by_round, data_by_round): (Vec<_>, Vec<_>) = log_bs_and_polys_by_round
//         .iter()
//         .map(|log_bs_and_polys| {
//             pcs.commit(
//                 log_bs_and_polys
//                     .iter()
//                     .map(|(_, poly)| poly.clone())
//                     .collect(),
//             )
//         })
//         .unzip();
//     assert_eq!(commits_by_round.len(), num_rounds);
//     assert_eq!(data_by_round.len(), num_rounds);
//     p_challenger.observe_slice(&commits_by_round);

//     let zeta: Vec<Challenge> = repeat_with(|| p_challenger.sample_algebra_element())
//         .take(itertools::max(log_bs_by_round.iter().copied().flatten().copied()).unwrap())
//         .collect_vec();

//     let queries_by_round = log_bs_by_round
//         .iter()
//         .map(|log_bs| {
//             log_bs
//                 .iter()
//                 .map(|log_b| vec![MlQuery::Eq(MultilinearPoint(zeta[..*log_b].to_vec()))])
//                 .collect_vec()
//         })
//         .collect_vec();
//     let opening_by_round = data_by_round
//         .iter()
//         .zip(&queries_by_round)
//         .map(|(data, queries)| {
//             queries
//                 .iter()
//                 .enumerate()
//                 .map(|(idx, queries)| {
//                     let mat = pcs.get_evaluations(data, idx);
//                     queries
//                         .iter()
//                         .map(|query| mat.columnwise_dot_product(&query.to_mle(Challenge::ONE)))
//                         .collect_vec()
//                 })
//                 .collect_vec()
//         })
//         .collect_vec();

//     let data_and_queries_and_evals = data_by_round
//         .iter()
//         .zip(
//             queries_by_round
//                 .iter()
//                 .zip(&opening_by_round)
//                 .map(|(queries, evals)| {
//                     queries
//                         .iter()
//                         .zip(evals)
//                         .map(|(queries, evals)| {
//                             queries
//                                 .iter()
//                                 .zip(evals)
//                                 .map(|(query, evals)| (query.clone(), evals.to_vec()))
//                                 .collect_vec()
//                         })
//                         .collect_vec()
//                 })
//                 .collect_vec(),
//         )
//         .collect_vec();
//     let proof = pcs.open(data_and_queries_and_evals, &mut p_challenger);

//     // Verify the proof.
//     let mut v_challenger = challenger.clone();
//     v_challenger.observe_slice(&commits_by_round);
//     let verifier_zeta: Vec<Challenge> = repeat_with(|| v_challenger.sample_algebra_element())
//         .take(itertools::max(log_bs_by_round.iter().copied().flatten().copied()).unwrap())
//         .collect_vec();
//     assert_eq!(verifier_zeta, zeta);

//     let commits_and_claims_by_round = izip!(
//         commits_by_round,
//         log_bs_and_polys_by_round,
//         opening_by_round
//     )
//     .map(|(commit, log_bs_and_polys, openings)| {
//         let claims = log_bs_and_polys
//             .iter()
//             .zip(openings)
//             .map(|((log_b, _), mat_openings)| {
//                 vec![(
//                     MlQuery::Eq(MultilinearPoint(zeta[..*log_b].to_vec())),
//                     mat_openings[0].clone(),
//                 )]
//             })
//             .collect_vec();
//         (commit, claims)
//     })
//     .collect_vec();
//     assert_eq!(commits_and_claims_by_round.len(), num_rounds);

//     pcs.verify(commits_and_claims_by_round, &proof, &mut v_challenger)
//         .unwrap()
// }

// type Val = KoalaBear;
// type Challenge = BinomialExtensionField<Val, 4>;
// type ByteHash = Keccak256Hash;
// type FieldHash = SerializingHasher<ByteHash>;
// type Compress = CompressionFunctionFromHasher<ByteHash, 2, 32>;
// type Dft = Radix2DitParallel<Val>;
// type Challenger = SerializingChallenger32<Val, HashChallenger<u8, ByteHash, 32>>;
// type MyPcs = WhirPcs<Val, Dft, FieldHash, Compress, 32>;

// fn get_pcs(
//     log_blowup: usize,
//     folding_factor: usize,
//     first_round_folding_factor: usize,
// ) -> (MyPcs, Challenger) {
//     let dft = Dft::default();
//     let security_level = 100;
//     let pow_bits = 20;
//     let byte_hash = ByteHash {};
//     let field_hash = FieldHash::new(byte_hash);
//     let compress = Compress::new(byte_hash);
//     let whir_params = ProtocolParameters {
//         initial_statement: true,
//         security_level,
//         pow_bits,
//         folding_factor: FoldingFactor::ConstantFromSecondRound(
//             first_round_folding_factor,
//             folding_factor,
//         ),
//         merkle_hash: field_hash,
//         merkle_compress: compress,
//         soundness_type: SecurityAssumption::CapacityBound,
//         starting_log_inv_rate: log_blowup,
//         rs_domain_initial_reduction_factor: 3,
//     };
//     (
//         MyPcs::new(dft, whir_params),
//         Challenger::from_hasher(Vec::new(), byte_hash),
//     )
// }

// mod blowup_1 {
//     use super::*;

//     #[test]
//     fn single() {
//         let p = get_pcs(1, 4, 4);
//         for i in 3..6 {
//             do_test_whir_pcs::<_, Challenge, _, _>(&p, &[&[i]]);
//         }
//     }

// #[test]
// fn many_equal() {
//     let p = get_pcs(1, 4, 4);
//     for i in 5..8 {
//         do_test_whir_pcs::<_, Challenge, _, _>(&p, &[&[i; 5]]);
//     }
// }

// #[test]
// fn many_different() {
//     let p = get_pcs(1, 4, 4);
//     for i in 3..8 {
//         let log_bs = (3..3 + i).collect::<Vec<_>>();
//         do_test_whir_pcs::<_, Challenge, _, _>(&p, &[&log_bs]);
//     }
// }

// #[test]
// fn many_different_rev() {
//     let p = get_pcs(1, 4, 4);
//     for i in 3..8 {
//         let log_bs = (3..3 + i).rev().collect::<Vec<_>>();
//         do_test_whir_pcs::<_, Challenge, _, _>(&p, &[&log_bs]);
//     }
// }

// #[test]
// fn multiple_rounds() {
//     let p = get_pcs(1, 4, 4);
//     do_test_whir_pcs::<_, Challenge, _, _>(&p, &[&[3]]);
//     do_test_whir_pcs::<_, Challenge, _, _>(&p, &[&[3], &[3]]);
//     do_test_whir_pcs::<_, Challenge, _, _>(&p, &[&[3], &[2]]);
//     do_test_whir_pcs::<_, Challenge, _, _>(&p, &[&[2], &[3]]);
//     do_test_whir_pcs::<_, Challenge, _, _>(&p, &[&[3, 4], &[3, 4]]);
//     do_test_whir_pcs::<_, Challenge, _, _>(&p, &[&[4, 2], &[4, 2]]);
//     do_test_whir_pcs::<_, Challenge, _, _>(&p, &[&[2, 2], &[3, 3]]);
//     do_test_whir_pcs::<_, Challenge, _, _>(&p, &[&[3, 3], &[2, 2]]);
//     // Skip or fix the following if types don't align (e.g., &[2] vs &[3, 3])
//     // do_test_whir_pcs::<_, Challenge, _, _>(&p, &[&[2], &[3, 3]]);
// }
// }

// mod blowup_2 {
//     use super::*;

//     #[test]
//     fn single() {
//         let p = get_pcs(2, 4, 4);
//         for i in 3..6 {
//             do_test_whir_pcs::<_, Challenge, _, _>(&p, &[&[i]]);
//         }
//     }

//     #[test]
//     fn many_equal() {
//         let p = get_pcs(2, 4, 4);
//         for i in 5..8 {
//             do_test_whir_pcs::<_, Challenge, _, _>(&p, &[&[i; 5]]);
//         }
//     }

//     #[test]
//     fn many_different() {
//         let p = get_pcs(2, 4, 4);
//         for i in 3..8 {
//             let log_bs = (3..3 + i).collect::<Vec<_>>();
//             do_test_whir_pcs::<_, Challenge, _, _>(&p, &[&log_bs]);
//         }
//     }

//     #[test]
//     fn many_different_rev() {
//         let p = get_pcs(2, 4, 4);
//         for i in 3..8 {
//             let log_bs = (3..3 + i).rev().collect::<Vec<_>>();
//             do_test_whir_pcs::<_, Challenge, _, _>(&p, &[&log_bs]);
//         }
//     }

//     #[test]
//     fn multiple_rounds() {
//         let p = get_pcs(2, 4, 4);
//         do_test_whir_pcs::<_, Challenge, _, _>(&p, &[&[3]]);
//         do_test_whir_pcs::<_, Challenge, _, _>(&p, &[&[3], &[3]]);
//         do_test_whir_pcs::<_, Challenge, _, _>(&p, &[&[3], &[2]]);
//         do_test_whir_pcs::<_, Challenge, _, _>(&p, &[&[2], &[3]]);
//         do_test_whir_pcs::<_, Challenge, _, _>(&p, &[&[3, 4], &[3, 4]]);
//         do_test_whir_pcs::<_, Challenge, _, _>(&p, &[&[4, 2], &[4, 2]]);
//         do_test_whir_pcs::<_, Challenge, _, _>(&p, &[&[2, 2], &[3, 3]]);
//         do_test_whir_pcs::<_, Challenge, _, _>(&p, &[&[3, 3], &[2, 2]]);
//         // Skip or fix the following if types don't align
//         // do_test_whir_pcs::<_, Challenge, _, _>(&p, &[&[2], &[3, 3]]);
//     }
// }
