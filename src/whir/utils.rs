use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field};
use p3_util::log2_ceil_usize;
use tracing::instrument;

use crate::fiat_shamir::errors::ProofError;
use crate::{
    fiat_shamir::{ChallengSampler, errors::ProofResult, prover::ProverState},
    parameters::ChallengeQueryConfig,
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

    #[must_use]
    const fn new(source: &'a [usize]) -> Self {
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
    #[must_use]
    pub const fn has_bits(&self, n: usize) -> bool {
        let remaining_bits_in_current_word = Self::WORD_BITS - self.bit_idx;
        let remaining_words = self.source.len().saturating_sub(self.word_idx + 1);
        let total_remaining_bits =
            remaining_bits_in_current_word + remaining_words * Self::WORD_BITS;
        total_remaining_bits >= n
    }

    /// Gets the number of remaining readable bits
    #[must_use]
    pub const fn remaining_bits(&self) -> usize {
        if self.word_idx >= self.source.len() {
            return 0;
        }
        let remaining_bits_in_current_word = Self::WORD_BITS - self.bit_idx;
        let remaining_words = self.source.len().saturating_sub(self.word_idx + 1);
        remaining_bits_in_current_word + remaining_words * Self::WORD_BITS
    }
}

/// Convert a large bit value into a byte array using little-endian encoding
fn bits_to_bytes(bits: usize, num_bytes: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(num_bytes);
    let mut remaining_bits = bits;

    for _ in 0..num_bytes {
        bytes.push((remaining_bits & 0xFF) as u8);
        remaining_bits >>= 8;
    }

    bytes
}

/// Extract bits from a byte array starting at a specific bit offset
fn extract_bits(bytes: &[u8], bit_offset: usize, num_bits: usize) -> usize {
    let mut result = 0usize;
    let mut bits_read = 0;

    while bits_read < num_bits {
        let byte_idx = (bit_offset + bits_read) / 8;
        let bit_idx = (bit_offset + bits_read) % 8;
        let bits_in_byte = (8 - bit_idx).min(num_bits - bits_read);

        if byte_idx < bytes.len() {
            let byte_val = bytes[byte_idx];
            // Handle the case where bits_in_byte == 8 to avoid overflow
            let mask = if bits_in_byte == 8 {
                0xFF
            } else {
                (1u8 << bits_in_byte) - 1
            } << bit_idx;
            let extracted = ((byte_val & mask) >> bit_idx) as usize;
            result |= extracted << bits_read;
        }

        bits_read += bits_in_byte;
    }

    result
}

/// Samples a list of unique query indices from a folded evaluation domain using adaptive optimization.
///
/// This implementation uses a smart hybrid approach: for small requests where the overhead
/// of batch processing exceeds benefits, it uses the direct method. For larger requests,
/// it employs optimized batch processing to minimize challenger calls while respecting
/// platform and field constraints.
///
/// ## Parameters
/// - `domain_size`: Size of the original evaluation domain
/// - `folding_factor`: Number of folding rounds applied
/// - `num_queries`: Number of query indices to generate
/// - `challenger`: Fiat-Shamir transcript for deterministic randomness
/// - `config`: Configuration parameters for query generation (optional, uses default if None)
///
/// ## Returns
/// Sorted and deduplicated list of query indices in the folded domain
pub fn get_challenge_stir_queries<Challenger, F>(
    domain_size: usize,
    folding_factor: usize,
    num_queries: usize,
    challenger: &mut Challenger,
    config: Option<&ChallengeQueryConfig>,
) -> ProofResult<Vec<usize>>
where
    F: Field,
    Challenger: ChallengSampler<F>,
{
    // Use provided config or default
    let default_config = ChallengeQueryConfig::default();
    let config = config.unwrap_or(&default_config);

    // Validate configuration at runtime
    config.validate().map_err(|_| ProofError::InvalidProof)?;

    let folded_domain_size = domain_size >> folding_factor;
    let domain_size_bits = log2_ceil_usize(folded_domain_size);

    // Calculate total bit requirements for all queries
    let total_bits = num_queries * domain_size_bits;

    if total_bits < config.batch_threshold {
        // Fast path: direct sampling for small requests
        let mut queries = Vec::with_capacity(num_queries);
        for _ in 0..num_queries {
            let query = challenger.sample_bits(domain_size_bits) % folded_domain_size;
            queries.push(query);
        }
        queries.sort_unstable();
        queries.dedup();
        Ok(queries)
    } else {
        // Batch processing path: optimized for larger requests
        let bytes_needed = total_bits.div_ceil(8);
        let mut random_bytes = Vec::with_capacity(bytes_needed);

        // Use maximum safe chunk size for field constraints
        let mut remaining_bits = total_bits;

        while remaining_bits > 0 {
            let chunk_bits = remaining_bits.min(config.max_bits_per_call);
            let chunk_bytes = chunk_bits.div_ceil(8);

            // Get random bits for this chunk
            let chunk_random_bits = challenger.sample_bits(chunk_bits);
            let chunk_bytes_vec = bits_to_bytes(chunk_random_bits, chunk_bytes);

            // Append to our byte array
            random_bytes.extend_from_slice(&chunk_bytes_vec);
            remaining_bits -= chunk_bits;
        }

        // Extract queries from the continuous bitstream
        let mut queries = Vec::with_capacity(num_queries);
        let mut bit_offset = 0;

        for _ in 0..num_queries {
            let query =
                extract_bits(&random_bytes, bit_offset, domain_size_bits) % folded_domain_size;
            queries.push(query);
            bit_offset += domain_size_bits;
        }

        // Sort and remove duplicates
        queries.sort_unstable();
        queries.dedup();
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
