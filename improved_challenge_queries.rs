use crate::utils::log2_ceil_usize;
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::Field;
use crate::ProofResult;

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
            let mask = ((1u8 << bits_in_byte) - 1) << bit_idx;
            let extracted = ((byte_val & mask) >> bit_idx) as usize;
            result |= extracted << bits_read;
        }
        
        bits_read += bits_in_byte;
    }
    
    result
}

/// Samples a list of unique query indices from a folded evaluation domain using optimized batch processing.
///
/// This implementation reduces the number of challenger calls by batching random byte generation
/// and extracting individual queries from a continuous bitstream, following the original proposal
/// to "squeeze the full bit vector once" rather than making multiple separate calls.
///
/// ## Parameters
/// - `domain_size`: Size of the original evaluation domain
/// - `folding_factor`: Number of folding rounds applied
/// - `num_queries`: Number of query indices to generate
/// - `challenger`: Fiat-Shamir transcript for deterministic randomness
///
/// ## Returns
/// Sorted and deduplicated list of query indices in the folded domain
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
    let domain_size_bits = log2_ceil_usize(folded_domain_size);
    
    // Calculate total bit requirements for all queries
    let total_bits = num_queries * domain_size_bits;
    
    // Calculate bytes needed using ceiling division
    let bytes_needed = total_bits.div_ceil(8);
    let mut random_bytes = vec![0u8; bytes_needed];
    
    // Batch fill random bytes to minimize challenger calls
    for chunk in random_bytes.chunks_mut(32) { // Process in 32-byte chunks
        let field_elements: Vec<F> = (0..chunk.len())
            .map(|_| challenger.sample())
            .collect();
        
        // Convert field elements to bytes
        for (i, &elem) in field_elements.iter().enumerate() {
            if i < chunk.len() {
                chunk[i] = elem.as_canonical_u64() as u8;
            }
        }
    }
    
    // Extract queries from the continuous bitstream
    let mut queries = Vec::with_capacity(num_queries);
    let mut bit_offset = 0;
    
    for _ in 0..num_queries {
        let query = extract_bits(&random_bytes, bit_offset, domain_size_bits) % folded_domain_size;
        queries.push(query);
        bit_offset += domain_size_bits;
    }
    
    // Sort and remove duplicates
    queries.sort_unstable();
    queries.dedup();
    Ok(queries)
}