use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field};
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::{
    fiat_shamir::{ChallengeSampler, errors::FiatShamirError, prover::ProverState},
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
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    const CACHE_SIZE: usize = 1 << 17; // 128KB for Apple Silicon

    #[cfg(all(target_arch = "aarch64", any(target_os = "ios", target_os = "android")))]
    const CACHE_SIZE: usize = 1 << 16; // 64KB for mobile ARM

    #[cfg(target_arch = "x86_64")]
    const CACHE_SIZE: usize = 1 << 15; // 32KB for x86-64

    #[cfg(not(any(
        all(target_arch = "aarch64", target_os = "macos"),
        all(target_arch = "aarch64", any(target_os = "ios", target_os = "android")),
        target_arch = "x86_64"
    )))]
    const CACHE_SIZE: usize = 1 << 15; // 32KB default

    CACHE_SIZE / size_of::<T>()
}

/// WHIR STIR Query Sampler: Generates cryptographically secure query indices for Reed–Solomon proximity testing.
///
/// This is an optimized version that batches randomness sampling when beneficial,
/// reducing the number of expensive transcript operations.
///
/// ## Parameters
/// - `domain_size`: Original evaluation domain size before folding
/// - `folding_factor`: Number of folding rounds (domain reduction = 2^folding_factor)
/// - `num_queries`: Target number of query indices to sample
/// - `prover_state`: Fiat-Shamir transcript state for deterministic randomness
///
/// **WARNING:** The domain size must be a power of two.
///
/// ## Returns
/// Sorted, deduplicated vector of query indices in [0, folded_domain_size)
pub fn get_challenge_stir_queries<Challenger, F, EF>(
    domain_size: usize,
    folding_factor: usize,
    num_queries: usize,
    prover_state: &mut Challenger,
) -> Result<Vec<usize>, FiatShamirError>
where
    Challenger: ChallengeSampler<EF>,
    F: Field,
    EF: ExtensionField<F>,
{
    // COMPUTE DOMAIN AND BATCHING PARAMETERS

    // Apply folding to get the final, smaller domain size.
    //
    // Example: 2^20 domain, 4 folds -> 2^20 >> 4 -> 2^16 domain size.
    let folded_domain_size = domain_size >> folding_factor;
    // Calculate the number of bits required to represent an index in the folded domain.
    //
    // Example: 2^16 domain -> log2(2^16) = 16 bits per query.
    let domain_size_bits = log2_strict_usize(folded_domain_size);

    // Determine the maximum number of bits we can safely sample in a single transcript call.
    //
    // This is a conservative limit to avoid potential statistical bias from the underlying field.
    let max_bits_per_call = (F::bits() - 1).min(20);

    // Calculate the total amount of random bits needed for all queries.
    //
    // Example: 80 queries * 16 bits/query = 1280 bits total.
    let total_bits_needed = num_queries * domain_size_bits;

    // Pre-allocate the vector for the query indices to avoid reallocations.
    let mut queries = Vec::with_capacity(num_queries);

    // EXECUTE SAMPLING BASED ON OPTIMAL STRATEGY
    if total_bits_needed <= max_bits_per_call {
        // STRATEGY 1: SINGLE BATCH (HIGHEST EFFICIENCY)
        //
        // This path is taken when the total entropy for all queries fits within a single,
        // safe call to the transcript, reducing N transcript operations to just 1.

        // Sample all the random bits needed for all queries in one go.
        let mut all_bits = prover_state.sample_bits(total_bits_needed);
        // Create a bitmask to extract `domain_size_bits` chunks from the sampled randomness.
        //
        // Example: 16 bits -> (1 << 16) - 1 -> 0b1111_1111_1111_1111
        let mask = (1 << domain_size_bits) - 1;

        // Unpack the single large integer into individual query indices.
        for _ in 0..num_queries {
            // Use the mask to extract the lowest `domain_size_bits`.
            let query_bits = all_bits & mask;

            // Map the extracted bits to a valid index in the folded domain.
            queries.push(query_bits % folded_domain_size);

            // Right-shift the packed bits to consume the bits we just used,
            // exposing the next query's bits at the LSB position.
            all_bits >>= domain_size_bits;
        }
    } else {
        // STRATEGY 2 or 3: MULTI-BATCH OR SIMPLE FALLBACK

        // Calculate how many full queries can fit into a single transcript call.
        //
        // This determines if batching is beneficial.
        let queries_per_batch = max_bits_per_call / domain_size_bits;

        if queries_per_batch >= 2 {
            // STRATEGY 2: MULTI-BATCH (MEDIUM EFFICIENCY)
            //
            // This path is taken if we can't get all bits at once, but we can still
            // fit at least two queries' worth of bits into each transcript call.

            let mut remaining = num_queries;
            let mask = (1 << domain_size_bits) - 1;

            // Loop, processing queries in batches until all are generated.
            while remaining > 0 {
                // Determine the size of the current batch (the last one might be smaller).
                let batch_size = remaining.min(queries_per_batch);
                let batch_bits = batch_size * domain_size_bits;

                // Sample just enough bits for the current batch.
                //
                // This is the expensive operation.
                let mut all_bits = prover_state.sample_bits(batch_bits);

                // Unpack the batch of bits into query indices, same as the single-batch path.
                for _ in 0..batch_size {
                    let query_index = (all_bits & mask) % folded_domain_size;
                    queries.push(query_index);
                    all_bits >>= domain_size_bits;
                }

                // Decrement the counter for the next iteration.
                remaining -= batch_size;
            }
        } else {
            // STRATEGY 3: SIMPLE FALLBACK (LEAST EFFICIENT)
            //
            // If batching is not possible or offers no benefit (i.e., we can fit less than
            // 2 queries per call), we fall back to the naive approach of one call per query.

            queries.extend(
                (0..num_queries)
                    .map(|_| prover_state.sample_bits(domain_size_bits) % folded_domain_size),
            );
        }
    }

    // FINALIZE QUERY LIST

    // Sort the collected indices to have a canonical order.
    queries.sort_unstable();

    // Remove any duplicate indices that may have been generated by chance.
    queries.dedup();

    Ok(queries)
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
