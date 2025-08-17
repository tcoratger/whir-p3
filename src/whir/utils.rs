use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field};
use tracing::instrument;

use crate::{
    fiat_shamir::{ChallengSampler, errors::ProofResult, prover::ProverState},
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

/// Samples a list of unique query indices from a folded evaluation domain.
///
/// This optimized implementation reduces challenger invocations by sampling all required
/// bits in a single call and then parsing the indices from the resulting bitstream.
///
/// ## Parameters
/// - `domain_size`: Size of the original evaluation domain
/// - `folding_factor`: Number of folding rounds applied
/// - `num_queries`: Number of query indices to generate
/// - `state`: State containing the Fiat-Shamir transcript (ProverState or VerifierState)
///
/// ## Returns
/// Sorted and deduplicated list of query indices in the folded domain
#[inline]
pub fn get_challenge_stir_queries<Chal: ChallengSampler<EF>, F, EF>(
    domain_size: usize,
    folding_factor: usize,
    num_queries: usize,
    prover_state: &mut Chal,
) -> ProofResult<Vec<usize>>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let folded_domain_size = domain_size >> folding_factor;
    // Calculate the number of bytes needed for each index
    let domain_size_bytes = ((folded_domain_size * 2 - 1).ilog2() as usize).div_ceil(8);

    // Simple approach: sample bytes one by one
    let total_bytes = num_queries * domain_size_bytes;
    let mut query_bytes = Vec::with_capacity(total_bytes);

    for _ in 0..total_bytes {
        let byte = prover_state.sample_bits(8) as u8;
        query_bytes.push(byte);
    }

    // Batch convert bytes to indices
    let mut indices: Vec<usize> = query_bytes
        .chunks_exact(domain_size_bytes)
        .map(|chunk| {
            chunk.iter().fold(0usize, |acc, &b| (acc << 8) | b as usize) % folded_domain_size
        })
        .collect();

    // Sort and deduplicate
    indices.sort_unstable();
    indices.dedup();

    Ok(indices)
}

/// A utility function to sample Out-of-Domain (OOD) points and evaluate them.
///
/// This should be used on the prover side.
#[instrument(skip_all)]
pub fn sample_ood_points<F: Field, EF: ExtensionField<F>, E, Challenger>(
    prover_state: &mut ProverState<F, EF, Challenger>,
    num_samples: usize,
    num_variables: usize,
    evaluate_fn: E,
) -> (Vec<EF>, Vec<EF>)
where
    E: Fn(&MultilinearPoint<EF>) -> EF,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    let mut ood_points = EF::zero_vec(num_samples);
    let mut ood_answers = Vec::with_capacity(num_samples);

    if num_samples > 0 {
        // Generate OOD points from ProverState randomness
        for ood_point in &mut ood_points {
            *ood_point = prover_state.sample();
        }

        // Evaluate the function at each OOD point
        ood_answers.extend(ood_points.iter().map(|ood_point| {
            evaluate_fn(&MultilinearPoint::expand_from_univariate(
                *ood_point,
                num_variables,
            ))
        }));

        prover_state.add_extension_scalars(&ood_answers);
    }

    (ood_points, ood_answers)
}
