use itertools::Itertools;
use nimue::ProofResult;
use p3_challenger::CanSample;
use p3_field::{Field, PrimeField32};

/// Generates a list of unique challenge queries within a folded domain.
///
/// Given a `domain_size` and `folding_factor`, this function:
/// - Computes the folded domain size: `folded_domain_size = domain_size / 2^folding_factor`.
/// - Derives query indices from random transcript bytes.
/// - Deduplicates indices while preserving order.
pub fn get_challenge_stir_queries<F, C>(
    domain_size: usize,
    folding_factor: usize,
    num_queries: usize,
    challenger: &mut C,
) -> Vec<usize>
where
    F: Field + PrimeField32,
    C: CanSample<F>,
{
    let folded_domain_size = domain_size >> folding_factor;
    // Compute required bytes per index: `domain_size_bytes = ceil(log2(folded_domain_size) / 8)`
    let domain_size_bytes = ((folded_domain_size * 2 - 1).ilog2() as usize).div_ceil(8);

    // Sample bytes from the challenger
    let queries: Vec<_> = (0..num_queries * domain_size_bytes)
        .map(|_| challenger.sample().as_canonical_u32() as usize)
        .collect();

    // Convert bytes into indices in **one efficient pass**
    queries
        .chunks_exact(domain_size_bytes)
        .map(|chunk| chunk.iter().fold(0usize, |acc, &b| (acc << 8) | b) % folded_domain_size)
        .sorted_unstable()
        .dedup()
        .collect()
}

pub trait DigestWriter<MerkleInnerDigest> {
    fn add_digest(&mut self, digest: MerkleInnerDigest);
}

pub trait DigestReader<MerkleInnerDigest> {
    fn read_digest(&mut self) -> MerkleInnerDigest;
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_baby_bear::BabyBear;
    use p3_challenger::{HashChallenger, SerializingChallenger32};
    use p3_keccak::Keccak256Hash;

    #[test]
    fn test_challenge_stir_queries_single_byte_indices() {
        let domain_size = 256;
        let folding_factor = 1;
        let num_queries = 5;
        let folded_domain_size = domain_size >> folding_factor; // 128

        // Challenger used for sampling randomness
        let hasher = Keccak256Hash {};
        let mut challenger = SerializingChallenger32::<
            BabyBear,
            HashChallenger<u8, Keccak256Hash, 32>,
        >::from_hasher(vec![], hasher);

        let result = get_challenge_stir_queries::<BabyBear, _>(
            domain_size,
            folding_factor,
            num_queries,
            &mut challenger,
        );

        // Check number of queries returned
        assert_eq!(result.len(), num_queries, "Incorrect number of queries generated");

        // Ensure all indices are within bounds
        assert!(result.iter().all(|&index| index < folded_domain_size));
    }

    #[test]
    fn test_challenge_stir_queries_two_byte_indices() {
        let domain_size = 65536; // 2^16
        let folding_factor = 3; // 2^3 = 8
        let num_queries = 5;
        let folded_domain_size = domain_size >> folding_factor; // 8192

        let hasher = Keccak256Hash {};
        let mut challenger = SerializingChallenger32::<
            BabyBear,
            HashChallenger<u8, Keccak256Hash, 32>,
        >::from_hasher(vec![], hasher);

        let result = get_challenge_stir_queries::<BabyBear, _>(
            domain_size,
            folding_factor,
            num_queries,
            &mut challenger,
        );

        assert_eq!(result.len(), num_queries, "Incorrect number of queries generated");
        assert!(result.iter().all(|&index| index < folded_domain_size));
    }

    #[test]
    fn test_challenge_stir_queries_three_byte_indices() {
        let domain_size = 2usize.pow(24); // 16,777,216
        let folding_factor = 4; // 2^4 = 16
        let num_queries = 4;
        let folded_domain_size = domain_size >> folding_factor; // 1,048,576

        let hasher = Keccak256Hash {};
        let mut challenger = SerializingChallenger32::<
            BabyBear,
            HashChallenger<u8, Keccak256Hash, 32>,
        >::from_hasher(vec![], hasher);

        let result = get_challenge_stir_queries::<BabyBear, _>(
            domain_size,
            folding_factor,
            num_queries,
            &mut challenger,
        );

        assert_eq!(result.len(), num_queries, "Incorrect number of queries generated");
        assert!(result.iter().all(|&index| index < folded_domain_size));
    }

    #[test]
    fn test_challenge_stir_queries_duplicate_indices() {
        let domain_size = 128;
        let folding_factor = 0;
        let num_queries = 5;
        let folded_domain_size = domain_size >> folding_factor; // 128

        let hasher = Keccak256Hash {};
        let mut challenger = SerializingChallenger32::<
            BabyBear,
            HashChallenger<u8, Keccak256Hash, 32>,
        >::from_hasher(vec![], hasher);

        let result = get_challenge_stir_queries::<BabyBear, _>(
            domain_size,
            folding_factor,
            num_queries,
            &mut challenger,
        );

        assert!(result.len() <= num_queries, "Duplicates should be removed");
        assert!(result.iter().all(|&index| index < folded_domain_size));
    }
}
