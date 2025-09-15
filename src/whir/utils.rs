use itertools::Itertools;
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field};
use p3_util::{log2_ceil_usize, log2_strict_usize};
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
/// After folding reduces the Reed–Solomon codeword domain, this function selects random
/// query positions for the proximity test. Uses Fiat-Shamir randomness to ensure unpredictable,
/// verifier-determined challenge locations.
///
/// ## Parameters
/// - `domain_size`: Original evaluation domain size before folding
/// - `folding_factor`: Number of folding rounds (domain reduction = 2^folding_factor)
/// - `num_queries`: Target number of query indices to sample
/// - `prover_state`: Fiat-Shamir transcript state for deterministic randomness
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
    // PHASE 1: Compute folded domain parameters

    // Apply WHIR folding: domain_size / 2^folding_factor
    //
    // Example: 512 domain → 3 folds → 512 >> 3 = 64 domain
    let folded_domain_size = domain_size >> folding_factor;

    // Bits required to index into the folded domain
    //
    // Example: 64 domain needs log_2(64) ≈ 6 bits per query
    let domain_size_bits = log2_strict_usize(folded_domain_size);

    // PHASE 2: Determine batching strategy

    // The maximum number of bits that can be safely sampled in a single call from the challenger.
    //
    // To avoid statistical bias when sampling randomness, the number of bits requested should be
    // less than the bit length of the challenger's underlying field modulus.
    let max_bits_per_call = F::bits() - 1;

    // Total entropy needed: num_queries × bits_per_query
    //
    // Example: 100 queries × 4 bits = 400 bits total
    let total_bits_needed = num_queries * domain_size_bits;

    // PHASE 3: Execute sampling
    let queries = if total_bits_needed > 0 && domain_size_bits > 0 {
        // Pre-allocate result vector
        let mut all_queries = Vec::with_capacity(num_queries);

        if total_bits_needed <= max_bits_per_call {
            // SINGLE BATCH PATH
            //
            // All entropy fits in one sponge squeeze - maximum efficiency
            let all_bits = prover_state.sample_bits(total_bits_needed);

            // Bit mask for extracting domain_size_bits chunks
            //
            // Example: 4 bits → mask = 0b1111 = 15
            let mask = (1 << domain_size_bits) - 1;

            // Extract each query index from the packed bit stream
            for i in 0..num_queries {
                // Bit position for query i: i × bits_per_query
                let start_bit = i * domain_size_bits;

                // Extract bits [start_bit, start_bit + domain_size_bits)
                let query_bits = (all_bits >> start_bit) & mask;

                // Map raw bits to valid domain index via modular reduction
                let query_index = query_bits % folded_domain_size;
                all_queries.push(query_index);
            }
        } else {
            // MULTI BATCH PATH
            //
            // Too many bits for one call - use chunked sampling

            // Queries that fit in one `max_bits_per_call` sample
            let queries_per_batch = max_bits_per_call / domain_size_bits;
            let mut remaining_queries = num_queries;

            // Process queries in batches until all are sampled
            while remaining_queries > 0 {
                // Current batch size (last batch may be smaller)
                let batch_size = remaining_queries.min(queries_per_batch);
                let batch_bits = batch_size * domain_size_bits;

                // Sample entropy for this batch
                let all_bits = prover_state.sample_bits(batch_bits);
                let mask = (1 << domain_size_bits) - 1;

                // Extract all queries from this batch
                for i in 0..batch_size {
                    let start_bit = i * domain_size_bits;
                    let query_bits = (all_bits >> start_bit) & mask;
                    let query_index = query_bits % folded_domain_size;
                    all_queries.push(query_index);
                }

                remaining_queries -= batch_size;
            }
        }

        // PHASE 4: Finalize query list
        //
        // Sort indices and remove duplicates (WHIR protocol requirement)
        all_queries.into_iter().sorted_unstable().dedup().collect()
    } else {
        // Edge case: no queries requested or invalid parameters
        Vec::new()
    };

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
