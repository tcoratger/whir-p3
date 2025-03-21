use p3_challenger::{CanObserve, CanSample};
use p3_field::{Field, PrimeCharacteristicRing, TwoAdicField};

use crate::poly::multilinear::MultilinearPoint;

/// A utility function to sample Out-of-Domain (OOD) points and evaluate them.
///
/// This should be used on the prover side.
pub fn sample_ood_points<F, C, E>(
    challenger: &mut C,
    num_samples: usize,
    num_variables: usize,
    evaluate_fn: E,
) -> (Vec<F>, Vec<F>)
where
    F: Field + TwoAdicField,
    <F as PrimeCharacteristicRing>::PrimeSubfield: TwoAdicField,
    C: CanSample<F> + CanObserve<F>,
    E: Fn(&MultilinearPoint<F>) -> F,
{
    // Sample OOD points
    let ood_points = challenger.sample_vec(num_samples);

    // Compute OOD evaluations
    let ood_answers: Vec<_> = ood_points
        .iter()
        .map(|&ood_point| {
            evaluate_fn(&MultilinearPoint::expand_from_univariate(ood_point, num_variables))
        })
        .collect();

    //  Observe OOD evaluations in challenger
    challenger.observe_slice(&ood_answers);

    (ood_points, ood_answers)
}
