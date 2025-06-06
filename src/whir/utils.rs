use itertools::Itertools;
use p3_challenger::{CanObserve, CanSample};
use p3_field::{ExtensionField, PrimeField64, TwoAdicField};
use tracing::instrument;

use crate::{
    fiat_shamir::{
        UnitToBytes, duplex_sponge::interface::Unit, errors::ProofResult, prover::ProverState,
    },
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
pub fn get_challenge_stir_queries<T, W>(
    domain_size: usize,
    folding_factor: usize,
    num_queries: usize,
    narg_string: &mut T,
) -> ProofResult<Vec<usize>>
where
    T: UnitToBytes<W>,
    W: Unit + Default + Copy,
{
    let folded_domain_size = domain_size >> folding_factor;
    // Compute required bytes per index: `domain_size_bytes = ceil(log2(folded_domain_size) / 8)`
    let domain_size_bytes = ((folded_domain_size * 2 - 1).ilog2() as usize).div_ceil(8);

    // Allocate space for query bytes
    let mut queries = vec![W::default(); num_queries * domain_size_bytes];
    narg_string.fill_challenge_units(&mut queries)?;

    // Convert bytes into indices in **one efficient pass**
    Ok(queries
        .chunks_exact(domain_size_bytes)
        .map(|chunk| {
            chunk
                .iter()
                .fold(0usize, |acc, &b| (acc << 8) | W::to_u8(b) as usize)
                % folded_domain_size
        })
        .sorted_unstable()
        .dedup()
        .collect())
}

/// A utility function to sample Out-of-Domain (OOD) points and evaluate them.
///
/// This should be used on the prover side.
#[instrument(skip_all)]
pub fn sample_ood_points<F, EF, E, Challenger, W>(
    prover_state: &mut ProverState<EF, F, Challenger, W>,
    num_samples: usize,
    num_variables: usize,
    evaluate_fn: E,
) -> ProofResult<(Vec<EF>, Vec<EF>)>
where
    F: PrimeField64 + TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    E: Fn(&MultilinearPoint<EF>) -> EF,
    W: Unit + Default + Copy,
    Challenger: CanObserve<W> + CanSample<W>,
{
    let mut ood_points = EF::zero_vec(num_samples);
    let mut ood_answers = Vec::with_capacity(num_samples);

    if num_samples > 0 {
        // Generate OOD points from ProverState randomness
        prover_state.fill_challenge_scalars(&mut ood_points)?;

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
