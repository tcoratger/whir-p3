use itertools::Itertools;
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field, PrimeField64, TwoAdicField};
use tracing::instrument;

use crate::{
    fiat_shamir::{errors::ProofResult, prover::ProverState},
    poly::multilinear::MultilinearPoint,
};

/// Computes the optimal workload size for `T` to fit in L1 cache (32 KB).
///
/// Ensures efficient memory access by dividing the cache size by `T`'s size.
/// The result represents how many elements of `T` can be processed per thread.
///
/// Helps minimize cache misses and improve performance in parallel workloads.
#[must_use]
pub const fn workload_size<T: Sized>() -> usize {
    const L1_CACHE_SIZE: usize = 1 << 15; // 32 KB
    L1_CACHE_SIZE / size_of::<T>()
}

/// Generates a list of unique challenge queries within a folded domain.
///
/// Given a `domain_size` and `folding_factor`, this function:
/// - Computes the folded domain size: `folded_domain_size = domain_size / 2^folding_factor`.
/// - Derives query indices from random transcript bytes.
/// - Deduplicates indices while preserving order.
pub fn get_challenge_stir_queries<Challenger, F>(
    domain_size: usize,
    folding_factor: usize,
    num_queries: usize,
    challenger: &mut Challenger,
) -> ProofResult<Vec<usize>>
where
    F: Field,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    let folded_domain_size = domain_size >> folding_factor;
    // Compute required bytes per index: `domain_size_bytes = ceil(log2(folded_domain_size) / 8)`
    let domain_size_bytes = ((folded_domain_size * 2 - 1).ilog2() as usize).div_ceil(8);

    // Allocate space for query bytes
    let queries = challenger.sample_vec(num_queries * domain_size_bytes);

    // Convert bytes into indices in **one efficient pass**
    Ok(queries
        .chunks_exact(domain_size_bytes)
        .map(|chunk| {
            chunk.iter().fold(0usize, |acc, &b| {
                let mut raw_bytes = b.into_bytes().into_iter().collect_vec();

                // Pad with zeros at the end if needed
                raw_bytes.resize(8, 0);

                (acc << 8) | u64::from_be_bytes(raw_bytes.try_into().unwrap()) as usize
            }) % folded_domain_size
        })
        .sorted_unstable()
        .dedup()
        .collect())
}

/// A utility function to sample Out-of-Domain (OOD) points and evaluate them.
///
/// This should be used on the prover side.
#[instrument(skip_all)]
pub fn sample_ood_points<F, EF, E, Challenger, const DIGEST_ELEMS: usize>(
    prover_state: &mut ProverState<EF, F, Challenger, DIGEST_ELEMS>,
    num_samples: usize,
    num_variables: usize,
    evaluate_fn: E,
) -> (Vec<EF>, Vec<EF>)
where
    F: PrimeField64 + TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    E: Fn(&MultilinearPoint<EF>) -> EF,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    let mut ood_points = EF::zero_vec(num_samples);
    let mut ood_answers = Vec::with_capacity(num_samples);

    if num_samples > 0 {
        // Generate OOD points from ProverState randomness
        for ood_point in &mut ood_points {
            *ood_point = prover_state.challenger.sample_algebra_element();
        }

        // Evaluate the function at each OOD point
        ood_answers.extend(ood_points.iter().map(|ood_point| {
            evaluate_fn(&MultilinearPoint::expand_from_univariate(
                *ood_point,
                num_variables,
            ))
        }));
    }

    (ood_points, ood_answers)
}
