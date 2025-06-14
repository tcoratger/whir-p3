// use core::{
//     cell::RefCell, cmp::Reverse, iter::repeat_with, marker::PhantomData, mem::replace, ops::Range,
// };

// use itertools::{Itertools, chain, cloned, izip, rev};
// use p3_challenger::FieldChallenger;
// use p3_commit::Mmcs;
// use p3_dft::TwoAdicSubgroupDft;
// use p3_field::{ExtensionField, Field, PrimeField64, TwoAdicField, dot_product};
// use p3_keccak::KeccakF;
// use p3_matrix::{
//     Dimensions, Matrix,
//     dense::{RowMajorMatrix, RowMajorMatrixView},
//     horizontally_truncated::HorizontallyTruncated,
// };
// use p3_maybe_rayon::prelude::*;
// use p3_merkle_tree::MerkleTreeMmcs;
// use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
// use p3_util::{log2_ceil_usize, log2_strict_usize};
// use rayon::{
//     iter::{IndexedParallelIterator, ParallelIterator},
//     slice::ParallelSliceMut,
// };
// use serde::{Deserialize, Serialize, de::DeserializeOwned};
// use tracing::info_span;

// use crate::{
//     poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
//     utils::eval_eq,
//     whir::{pcs::query::MlQuery, statement::weights::Weights},
// };

// pub struct ConcatMatsMeta {
//     log_b: usize,
//     dimensions: Vec<Dimensions>,
//     ranges: Vec<Range<usize>>,
// }

// impl ConcatMatsMeta {
//     fn new(dims: Vec<Dimensions>) -> Self {
//         let (dimensions, ranges) = dims
//             .iter()
//             .enumerate()
//             // Sorted by matrix size in descending order.
//             .sorted_by_key(|(_, dim)| Reverse(dim.width * dim.height))
//             // Calculate sub-cube range for each matrix (power-of-2 aligned).
//             .scan(0, |offset, (idx, dim)| {
//                 let size = dim.width.next_power_of_two() * dim.height;
//                 let offset = replace(offset, *offset + size);
//                 Some((idx, dim, offset..offset + size))
//             })
//             // Store the dimension and range in original order.
//             .sorted_by_key(|(idx, _, _)| *idx)
//             .map(|(_, dim, range)| (dim, range))
//             .collect::<(Vec<_>, Vec<_>)>();
//         // Calculate number of variable of concated polynomial.
//         let log_b = log2_ceil_usize(
//             ranges
//                 .iter()
//                 .map(|range| range.end)
//                 .max()
//                 .unwrap_or_default(),
//         );
//         Self {
//             log_b,
//             dimensions,
//             ranges,
//         }
//     }

//     fn max_log_width(&self) -> usize {
//         self.dimensions
//             .iter()
//             .map(|dim| log2_ceil_usize(dim.width))
//             .max()
//             .unwrap_or_default()
//     }

//     fn constraint<Challenge: Field>(
//         &self,
//         idx: usize,
//         query: &MlQuery<Challenge>,
//         ys: &[Challenge],
//         r: &[Challenge],
//     ) -> (Weights<Challenge>, Challenge) {
//         let log_width = log2_ceil_usize(self.dimensions[idx].width);

//         let r = &r[..log_width];
//         let mut eq_r = vec![Challenge::ZERO; 1 << r.len()];
//         eval_eq::<_, _, false>(r, &mut eq_r, Challenge::ONE);

//         let sum = dot_product(cloned(ys), cloned(&eq_r[..ys.len()]));

//         let weights = match query {
//             MlQuery::Eq(z) => {
//                 let point = rev(chain![
//                     cloned(r),
//                     cloned(z),
//                     (log2_strict_usize(self.ranges[idx].len())..self.log_b)
//                         .map(|i| Challenge::from_bool((self.ranges[idx].start >> i) & 1 == 1))
//                 ])
//                 .collect();
//                 Weights::evaluation(MultilinearPoint(point))
//             }
//             // TODO(whir-p3): Introduce a new weights variant to generate such evaluations.
//             MlQuery::EqRotateRight(_, _) => {
//                 let mut weight = Challenge::zero_vec(1 << self.log_b);
//                 weight[self.ranges[idx].clone()]
//                     .par_chunks_mut(eq_r.len())
//                     .zip(query.to_mle(Challenge::ONE))
//                     .for_each(|(weight, query)| {
//                         izip!(weight, &eq_r).for_each(|(weight, eq_r)| *weight = *eq_r * query)
//                     });
//                 Weights::linear(EvaluationsList::new(weight))
//             }
//         };

//         (weights, sum)
//     }
// }

// pub struct ConcatMats<Val> {
//     values: Vec<Val>,
//     meta: ConcatMatsMeta,
// }

// impl<Val: Field> ConcatMats<Val> {
//     fn new(mats: Vec<RowMajorMatrix<Val>>) -> Self {
//         let meta = ConcatMatsMeta::new(mats.iter().map(Matrix::dimensions).collect());
//         let mut values = Val::zero_vec(1 << meta.log_b);
//         izip!(&meta.ranges, mats).for_each(|(range, mat)| {
//             // Copy and pad each row into power-of-2 length into concated polynomial.
//             values[range.clone()]
//                 .par_chunks_mut(mat.width().next_power_of_two())
//                 .zip(mat.par_row_slices())
//                 .for_each(|(dst, src)| dst[..src.len()].copy_from_slice(src));
//         });
//         Self { values, meta }
//     }

//     fn mat(&self, idx: usize) -> HorizontallyTruncated<Val, RowMajorMatrixView<'_, Val>> {
//         HorizontallyTruncated::new(
//             RowMajorMatrixView::new(
//                 &self.values[self.meta.ranges[idx].clone()],
//                 self.meta.dimensions[idx].width.next_power_of_two(),
//             ),
//             self.meta.dimensions[idx].width,
//         )
//         .unwrap()
//     }
// }
