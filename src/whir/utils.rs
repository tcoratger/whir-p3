use itertools::Itertools;
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field};
use p3_util::log2_ceil_usize;
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

#[derive(Debug)]
pub struct BitstreamReader<'a> {
    /// Data source storing random bits
    source: &'a [usize],
    /// Current word index in `source`
    word_idx: usize,
    /// Current bit index in `source[word_idx]` (from LSB to MSB)
    bit_idx: usize,
}

impl<'a> BitstreamReader<'a> {
    const WORD_BITS: usize = usize::BITS as usize;

    pub fn new(source: &'a [usize]) -> Self {
        Self {
            source,
            word_idx: 0,
            bit_idx: 0,
        }
    }

    /// Reads `n` bits from the stream and returns them as a `usize`.
    /// `n` must be less than or equal to `WORD_BITS`.
    #[inline]
    pub fn read_bits(&mut self, n: usize) -> usize {
        debug_assert!(n <= Self::WORD_BITS, "Cannot read more bits than a word");
        debug_assert!(n > 0, "Must read at least 1 bit");

        // Fast path: all bits are within the current word
        if self.bit_idx + n <= Self::WORD_BITS {
            // Use bit shifting and masking to extract bits
            let mask = (1usize << n).wrapping_sub(1); // Avoid overflow when n=64
            let result =
                (unsafe { *self.source.get_unchecked(self.word_idx) } >> self.bit_idx) & mask;

            self.bit_idx += n;
            // Use conditional move to avoid branching
            let (new_word_idx, new_bit_idx) = if self.bit_idx == Self::WORD_BITS {
                (self.word_idx + 1, 0)
            } else {
                (self.word_idx, self.bit_idx)
            };
            self.word_idx = new_word_idx;
            self.bit_idx = new_bit_idx;

            result
        } else {
            // Slow path: need to read across word boundaries
            let bits_in_first_word = Self::WORD_BITS - self.bit_idx;
            let bits_in_second_word = n - bits_in_first_word;

            // Read from the high bits of the first word
            let first_part = unsafe { *self.source.get_unchecked(self.word_idx) } >> self.bit_idx;

            // Move to the next word
            self.word_idx += 1;

            // Read from the low bits of the second word
            let second_part_mask = (1usize << bits_in_second_word).wrapping_sub(1);
            let second_part =
                unsafe { *self.source.get_unchecked(self.word_idx) } & second_part_mask;

            // Update bit index
            self.bit_idx = bits_in_second_word;

            // Combine the two parts
            (second_part << bits_in_first_word) | first_part
        }
    }

    /// Checks if there are enough bits available to read
    pub fn has_bits(&self, n: usize) -> bool {
        let remaining_bits_in_current_word = Self::WORD_BITS - self.bit_idx;
        let remaining_words = self.source.len().saturating_sub(self.word_idx + 1);
        let total_remaining_bits =
            remaining_bits_in_current_word + remaining_words * Self::WORD_BITS;
        total_remaining_bits >= n
    }

    /// Gets the number of remaining readable bits
    pub fn remaining_bits(&self) -> usize {
        if self.word_idx >= self.source.len() {
            return 0;
        }
        let remaining_bits_in_current_word = Self::WORD_BITS - self.bit_idx;
        let remaining_words = self.source.len().saturating_sub(self.word_idx + 1);
        remaining_bits_in_current_word + remaining_words * Self::WORD_BITS
    }
}

/// Samples a list of unique query indices from a folded evaluation domain, using transcript randomness.
///
/// This function is used to select random query locations for verifying proximity to a folded codeword.
/// The folding reduces the domain size exponentially (e.g. by 2^folding_factor), so we sample indices
/// in the reduced "folded" domain.
///
/// ## Parameters
/// - `domain_size`: The size of the original evaluation domain (e.g., 2^22).
/// - `folding_factor`: The number of folding rounds applied (e.g., k = 1 means domain halves).
/// - `num_queries`: The number of query *indices* we want to obtain.
/// - `challenger`: A Fiat–Shamir transcript used to sample randomness deterministically.
///
/// ## Returns
/// A sorted and deduplicated list of random query indices in the folded domain.
// pub fn get_challenge_stir_queries<Chal: ChallengSampler<EF>, F, EF>(
//     domain_size: usize,
//     folding_factor: usize,
//     num_queries: usize,
//     prover_state: &mut Chal,
// ) -> ProofResult<Vec<usize>>
// where
//     F: Field,
//     EF: ExtensionField<F>,
// {
//     // Folded domain size = domain_size / 2^folding_factor.
//     let folded_domain_size = domain_size >> folding_factor;

//     // Number of bits needed to represent an index in the folded domain.
//     let domain_size_bits = log2_ceil_usize(folded_domain_size);

//     // Sample one integer per query, each with domain_size_bits of entropy.
//     let queries = (0..num_queries)
//         .map(|_| prover_state.sample_bits(domain_size_bits) % folded_domain_size)
//         .sorted_unstable()
//         .dedup()
//         .collect();

//     Ok(queries)
// }

pub fn get_challenge_stir_queries<Challenger, F>(
    domain_size: usize,
    folding_factor: usize,
    num_queries: usize,
    challenger: &mut Challenger,
) -> ProofResult<Vec<usize>>
where
    F: Field,
    Challenger: ChallengSampler<F>,
{
    // Folded domain size = domain_size / 2^folding_factor.
    let folded_domain_size = domain_size >> folding_factor;

    // Number of bits needed to represent an index in the folded domain.
    let domain_size_bits = log2_ceil_usize(folded_domain_size);

    // Calculate total bits needed
    let total_bits = num_queries * domain_size_bits;

    // Ultra-fast path: for very small requests, the original method may be faster
    // Avoids overhead from additional allocations and complex logic
    if total_bits <= 64 {
        let queries = (0..num_queries)
            .map(|_| challenger.sample_bits(domain_size_bits) % folded_domain_size)
            .sorted_unstable()
            .dedup()
            .collect();
        return Ok(queries);
    }

    // Batch sampling optimization: suitable for medium to large requests
    // Only fall back to original method for extremely large requests (>4096 bits)
    if total_bits <= 30 * 136 {
        // 30 bits per word * 136 words = 4080 bits threshold
        // Calculate how many usize words we need
        let words_needed = (total_bits + 29) / 30; // Round up division

        // Pre-allocate with exact capacity to avoid reallocations
        let mut random_words = Vec::with_capacity(words_needed);

        // Batch sample all required words at once
        // Note: We sample 30 bits to stay within BabyBear field order (2^31 - 2^27 + 1)
        random_words.extend((0..words_needed).map(|_| challenger.sample_bits(30)));

        // Create bitstream reader
        let mut reader = BitstreamReader::new(&random_words);

        // Pre-allocate result vector with exact capacity
        let mut queries = Vec::with_capacity(num_queries);

        // Extract queries from the bitstream using unrolled loop for better performance
        let chunks = num_queries / 4;
        let remainder = num_queries % 4;

        // Process in chunks of 4 for better instruction-level parallelism
        for _ in 0..chunks {
            queries.push(reader.read_bits(domain_size_bits) % folded_domain_size);
            queries.push(reader.read_bits(domain_size_bits) % folded_domain_size);
            queries.push(reader.read_bits(domain_size_bits) % folded_domain_size);
            queries.push(reader.read_bits(domain_size_bits) % folded_domain_size);
        }

        // Handle remaining queries
        for _ in 0..remainder {
            queries.push(reader.read_bits(domain_size_bits) % folded_domain_size);
        }

        // Sort and deduplicate
        queries.sort_unstable();
        queries.dedup();

        Ok(queries)
    } else {
        // Fall back to original method for extremely large requests
        let queries = (0..num_queries)
            .map(|_| challenger.sample_bits(domain_size_bits) % folded_domain_size)
            .sorted_unstable()
            .dedup()
            .collect();

        Ok(queries)
    }
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
