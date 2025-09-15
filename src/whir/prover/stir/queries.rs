//! STIR query generation and challenge computation for Reed-Solomon proximity testing.

use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::dense::DenseMatrix;

use crate::{
    fiat_shamir::{ChallengeSampler, errors::FiatShamirError},
    poly::multilinear::MultilinearPoint,
    whir::{prover::round_state::RoundState, utils::get_challenge_stir_queries},
};

/// STIR query challenges containing evaluation points and their corresponding domain indices.
#[derive(Debug, Clone)]
pub struct StirChallenges<F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    /// Out-of-domain challenges f(r_j) for Reed-Solomon proximity verification.
    ///
    /// These points lie outside the evaluation domain to test polynomial structure.
    pub ood_challenges: Vec<MultilinearPoint<EF>>,

    /// STIR challenge points sampled from the folded domain H_{i+1}.
    ///
    /// Used for querying polynomial values at verifier-chosen positions.
    pub stir_challenges: Vec<MultilinearPoint<F>>,

    /// Domain indices corresponding to STIR challenges in the folded commitment.
    ///
    /// Enable efficient Merkle tree opening and proof generation.
    pub challenge_indices: Vec<usize>,
}

impl<F, EF> StirChallenges<F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    /// Creates new STIR challenges from constituent parts.
    ///
    /// # Arguments
    /// * `ood_challenges` - Out-of-domain evaluation points for proximity testing
    /// * `stir_challenges` - Challenge points from the folded evaluation domain
    /// * `challenge_indices` - Domain indices for Merkle tree queries
    #[must_use]
    pub const fn new(
        ood_challenges: Vec<MultilinearPoint<EF>>,
        stir_challenges: Vec<MultilinearPoint<F>>,
        challenge_indices: Vec<usize>,
    ) -> Self {
        Self {
            ood_challenges,
            stir_challenges,
            challenge_indices,
        }
    }

    /// Returns the number of STIR challenge queries.
    #[must_use]
    pub const fn num_queries(&self) -> usize {
        self.stir_challenges.len()
    }

    /// Returns the number of out-of-domain evaluation points.
    #[must_use]
    pub const fn num_ood_points(&self) -> usize {
        self.ood_challenges.len()
    }
}

/// STIR query generator for computing verification challenges in WHIR protocol rounds.
///
/// This generator handles the complete STIR query pipeline:
/// 1. **Index Sampling**: Generate random query indices in the folded domain
/// 2. **Point Conversion**: Transform indices to multilinear evaluation points
/// 3. **Challenge Assembly**: Package results for verification and proof generation
///
/// The generator ensures that challenges are unpredictable and uniformly distributed,
/// providing strong soundness guarantees for Reed-Solomon proximity testing.
#[derive(Debug)]
pub struct StirQueryGenerator {
    /// Number of queries to generate per round for soundness.
    num_queries: usize,
}

impl StirQueryGenerator {
    /// Creates a new STIR query generator with specified query count.
    ///
    /// # Arguments
    /// * `num_queries` - Number of challenge queries to generate per round
    ///
    /// # Returns
    /// A new generator ready for STIR challenge computation
    #[must_use]
    pub const fn new(num_queries: usize) -> Self {
        Self { num_queries }
    }

    /// Computes complete STIR challenges for a protocol round.
    ///
    /// This is the main entry point for STIR challenge generation, combining:
    /// - Out-of-domain point expansion to multilinear coordinates
    /// - Random index sampling in the folded evaluation domain
    /// - Challenge point computation using the domain generator
    ///
    /// # Protocol Context
    /// In WHIR, each round operates on a domain H_i that gets folded to H_{i+1} where
    /// |H_{i+1}| = |H_i|/2^k. The STIR challenges query specific positions in H_{i+1}
    /// to verify that the folded polynomial maintains Reed-Solomon structure.
    ///
    /// # Arguments
    /// * `round_state` - Current round state containing domain parameters and commitments
    /// * `prover_state` - Fiat-Shamir challenger for deterministic randomness
    /// * `ood_points` - Out-of-domain evaluation points from the witness
    /// * `folding_factor` - Number of variables folded in this round (k)
    /// * `num_variables` - Total number of polynomial variables (n)
    ///
    /// # Returns
    /// Complete STIR challenges ready for verification and proof generation
    pub(crate) fn compute_challenges<F, EF, const DIGEST_ELEMS: usize>(
        &self,
        round_state: &RoundState<EF, F, F, DenseMatrix<F>, DIGEST_ELEMS>,
        prover_state: &mut impl ChallengeSampler<EF>,
        ood_points: &[EF],
        folding_factor: usize,
        num_variables: usize,
    ) -> Result<StirChallenges<F, EF>, FiatShamirError>
    where
        F: TwoAdicField,
        EF: ExtensionField<F> + TwoAdicField,
    {
        // Sample random indices in the folded domain H_{i+1}
        let challenge_indices = get_challenge_stir_queries(
            round_state.domain_size,
            folding_factor,
            self.num_queries,
            prover_state,
        )?;

        // Convert out-of-domain points to multilinear evaluation coordinates
        let ood_challenges = ood_points
            .iter()
            .map(|&univariate| {
                // Expand univariate OOD point r_j to multilinear point in (EF)^n
                MultilinearPoint::expand_from_univariate(univariate, num_variables)
            })
            .collect();

        // Transform domain indices to challenge evaluation points
        let domain_generator = round_state.next_domain_gen;
        let stir_challenges = challenge_indices
            .iter()
            .map(|&index| {
                // Compute ω^index where ω is the generator of the folded domain H_{i+1}
                let domain_point = domain_generator.exp_u64(index as u64);
                // Expand to multilinear coordinates for polynomial evaluation
                MultilinearPoint::expand_from_univariate(domain_point, num_variables)
            })
            .collect();

        Ok(StirChallenges::new(
            ood_challenges,
            stir_challenges,
            challenge_indices,
        ))
    }

    /// Generates final round challenge indices for terminal verification queries.
    ///
    /// In the final WHIR round, the protocol needs to query the ultimate folded polynomial
    /// to complete the Reed-Solomon proximity test. This function generates the indices
    /// for these final queries using the same deterministic sampling approach.
    ///
    /// # Arguments
    /// * `domain_size` - Size of the domain before final folding
    /// * `folding_factor` - Folding factor for the final round
    /// * `num_final_queries` - Number of final verification queries
    /// * `prover_state` - Fiat-Shamir challenger for randomness
    ///
    /// # Returns
    /// Indices for final round queries in the terminal folded domain
    pub fn generate_final_queries<F, EF>(
        domain_size: usize,
        folding_factor: usize,
        num_final_queries: usize,
        prover_state: &mut impl ChallengeSampler<EF>,
    ) -> Result<Vec<usize>, FiatShamirError>
    where
        F: Field,
        EF: ExtensionField<F>,
    {
        // Sample indices for the final verification round using the shared utility
        get_challenge_stir_queries(domain_size, folding_factor, num_final_queries, prover_state)
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    #[test]
    fn test_stir_challenges_creation() {
        let ood_challenges = vec![MultilinearPoint::new(vec![EF::ONE, EF::ZERO])];
        let stir_challenges = vec![MultilinearPoint::new(vec![F::ONE, F::TWO])];
        let challenge_indices = vec![0, 1, 5];

        let challenges = StirChallenges::new(
            ood_challenges.clone(),
            stir_challenges.clone(),
            challenge_indices.clone(),
        );

        assert_eq!(challenges.num_queries(), 1);
        assert_eq!(challenges.num_ood_points(), 1);
        assert_eq!(challenges.ood_challenges, ood_challenges);
        assert_eq!(challenges.stir_challenges, stir_challenges);
        assert_eq!(challenges.challenge_indices, challenge_indices);
    }

    #[test]
    fn test_generator_integrates_with_utils() {
        // Test that our generator creates objects with the expected query count
        let generator = StirQueryGenerator::new(5);
        assert_eq!(generator.num_queries, 5);

        let generator = StirQueryGenerator::new(10);
        assert_eq!(generator.num_queries, 10);
    }
}
