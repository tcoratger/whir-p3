use itertools::Itertools;
use p3_field::{BasedVectorSpace, Field, PrimeField64};

use crate::{
    fiat_shamir::{errors::ProofResult, prover::ProverState, verifier::VerifierState},
    poly::multilinear::MultilinearPoint,
};

/// Generates a list of unique challenge queries within a folded domain.
///
/// Given a `domain_size` and `folding_factor`, this function:
/// - Computes the folded domain size: `folded_domain_size = domain_size / 2^folding_factor`.
/// - Derives query indices from random transcript bytes.
/// - Deduplicates indices while preserving order.
pub fn get_challenge_stir_queries_verifier(
    domain_size: usize,
    folding_factor: usize,
    num_queries: usize,
    narg_string: &mut VerifierState<'_>,
) -> ProofResult<Vec<usize>> {
    let folded_domain_size = domain_size >> folding_factor;
    // Compute required bytes per index: `domain_size_bytes = ceil(log2(folded_domain_size) / 8)`
    let domain_size_bytes = ((folded_domain_size * 2 - 1).ilog2() as usize).div_ceil(8);

    // Allocate space for query bytes
    let mut queries = vec![0u8; num_queries * domain_size_bytes];
    narg_string.fill_challenge_bytes(&mut queries)?;

    // Convert bytes into indices in **one efficient pass**
    Ok(queries
        .chunks_exact(domain_size_bytes)
        .map(|chunk| {
            chunk.iter().fold(0usize, |acc, &b| (acc << 8) | b as usize) % folded_domain_size
        })
        .sorted_unstable()
        .dedup()
        .collect())
}

/// Generates a list of unique challenge queries within a folded domain.
///
/// Given a `domain_size` and `folding_factor`, this function:
/// - Computes the folded domain size: `folded_domain_size = domain_size / 2^folding_factor`.
/// - Derives query indices from random transcript bytes.
/// - Deduplicates indices while preserving order.
pub fn get_challenge_stir_queries_prover(
    domain_size: usize,
    folding_factor: usize,
    num_queries: usize,
    narg_string: &mut ProverState,
) -> ProofResult<Vec<usize>> {
    let folded_domain_size = domain_size >> folding_factor;
    // Compute required bytes per index: `domain_size_bytes = ceil(log2(folded_domain_size) / 8)`
    let domain_size_bytes = ((folded_domain_size * 2 - 1).ilog2() as usize).div_ceil(8);

    // Allocate space for query bytes
    let mut queries = vec![0u8; num_queries * domain_size_bytes];
    narg_string.fill_challenge_bytes(&mut queries)?;

    // Convert bytes into indices in **one efficient pass**
    Ok(queries
        .chunks_exact(domain_size_bytes)
        .map(|chunk| {
            chunk.iter().fold(0usize, |acc, &b| (acc << 8) | b as usize) % folded_domain_size
        })
        .sorted_unstable()
        .dedup()
        .collect())
}

/// A utility function to sample Out-of-Domain (OOD) points and evaluate them.
///
/// This should be used on the prover side.
pub fn sample_ood_points<F, E>(
    prover_state: &mut ProverState,
    num_samples: usize,
    num_variables: usize,
    evaluate_fn: E,
) -> ProofResult<(Vec<F>, Vec<F>)>
where
    F: Field + BasedVectorSpace<F::PrimeSubfield>,
    F::PrimeSubfield: PrimeField64,
    E: Fn(&MultilinearPoint<F>) -> F,
{
    let mut ood_points = vec![F::ZERO; num_samples];
    let mut ood_answers = Vec::with_capacity(num_samples);

    if num_samples > 0 {
        // Generate OOD points from ProverState randomness
        prover_state.fill_challenge_scalars::<F>(&mut ood_points)?;

        // Evaluate the function at each OOD point
        ood_answers.extend(ood_points.iter().map(|ood_point| {
            evaluate_fn(&MultilinearPoint::expand_from_univariate(
                *ood_point,
                num_variables,
            ))
        }));

        // Commit the answers to the narg_string
        prover_state.add_scalars(&ood_answers)?;
    }

    Ok((ood_points, ood_answers))
}
