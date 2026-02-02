use alloc::{vec, vec::Vec};

use itertools::Itertools;
use p3_field::{ExtensionField, Field, PackedFieldExtension, PackedValue, dot_product};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;

use crate::{
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    sumcheck::lagrange_weights_012,
};

/// Grid points for SVO accumulators: returns [pts_0, pts_2] where
/// pts_0 has 3^(l-1) points in {0,1,2}^(l-1) × {0} (for h(0))
/// pts_2 has 3^(l-1) points in {0,1,2}^(l-1) × {2} (for h(2))
pub(super) fn points_012<F: Field>(l: usize) -> [Vec<MultilinearPoint<F>>; 2] {
    fn expand<F: Field>(pts: &[MultilinearPoint<F>], values: &[usize]) -> Vec<MultilinearPoint<F>> {
        values
            .iter()
            .flat_map(|&v| {
                pts.iter().cloned().map(move |mut p| {
                    p.0.push(F::from_u32(v as u32));
                    p
                })
            })
            .collect()
    }

    assert!(l > 0);
    let mut pts = vec![MultilinearPoint::new(vec![])];
    (0..l - 1).for_each(|_| pts = expand(pts.as_slice(), &[0, 1, 2]));
    [expand(pts.as_slice(), &[0]), expand(pts.as_slice(), &[2])]
}

/// Lagrange weights for domain {0,1,2}^k where k = rs.len().
/// Returns 3^k weights for interpolating to point (rs[0], rs[1], ...).
pub(super) fn lagrange_weights_012_multi<F: Field>(rs: &[F]) -> Vec<F> {
    let mut weights = vec![F::ONE];
    for &r in rs {
        weights = lagrange_weights_012(r)
            .iter()
            .flat_map(|li| weights.iter().map(|&w| w * *li))
            .collect();
    }
    weights
}

/// Compute f(u) * eq(u, point) for each grid point u in `us`.
/// These values are later interpolated using `lagrange_weights_012_multi`.
pub(super) fn calculate_accumulators<F: Field, EF: ExtensionField<F>>(
    us: &[MultilinearPoint<F>],
    partial_evals: &[EF],
    point: &MultilinearPoint<EF>,
) -> Vec<EF> {
    fn log3_strict(n: usize) -> usize {
        assert_ne!(n, 0);
        let mut res = 0usize;
        let mut t = n;
        loop {
            t /= 3;
            if t == 0 {
                break;
            }
            res += 1;
        }
        assert_eq!(n, 3usize.pow(res as u32));
        res
    }

    let l0 = log2_strict_usize(partial_evals.len());
    let off = l0 - log3_strict(us.len()) - 1;

    let (z0, z1) = point.split_at(point.num_variables() - off);
    let eq0 = EvaluationsList::new_from_point(z0.as_slice(), EF::ONE);
    let eq1 = EvaluationsList::new_from_point(z1.as_slice(), EF::ONE);

    let partial_evals = partial_evals
        .chunks(eq1.num_evals())
        .map(|chunk| dot_product::<EF, _, _>(eq1.iter().copied(), chunk.iter().copied()))
        .collect::<Vec<_>>();

    us.iter()
        .map(|u| {
            let coeffs = EvaluationsList::new_from_point(u.as_slice(), F::ONE);
            dot_product::<EF, _, _>(eq0.iter().copied(), coeffs.iter().copied())
                * dot_product::<EF, _, _>(partial_evals.iter().copied(), coeffs.iter().copied())
        })
        .collect::<Vec<_>>()
}

/// SplitEq implements algorithm 5 in https://eprint.iacr.org/2025/1117
#[derive(Debug, Clone)]
pub(super) struct SplitEq<F: Field, EF: ExtensionField<F>> {
    left: EvaluationsList<EF::ExtensionPacking>,
    right: EvaluationsList<EF>,
}

impl<F: Field, EF: ExtensionField<F>> SplitEq<F, EF> {
    #[tracing::instrument(skip_all)]
    pub(super) fn new(point: &MultilinearPoint<EF>, alpha: EF) -> Self {
        let k = point.num_variables();
        assert!(k >= 2 * log2_strict_usize(F::Packing::WIDTH));
        let (right, left) = point.split_at(k / 2);
        let left = EvaluationsList::new_packed_from_point(left.as_slice(), alpha);
        let right = EvaluationsList::new_from_point(right.as_slice(), EF::ONE);
        Self { left, right }
    }

    const fn k(&self) -> usize {
        self.left.num_variables()
            + self.right.num_variables()
            + log2_strict_usize(F::Packing::WIDTH)
    }

    #[tracing::instrument(skip_all)]
    pub(super) fn partial_evals(&self, poly: &EvaluationsList<F>) -> Vec<EF> {
        let chunk_size = 1 << self.k();
        poly.0
            .chunks(chunk_size)
            .map(|poly| {
                let poly = F::Packing::pack_slice(poly);
                let e_part = poly
                    .par_chunks(self.left.0.len())
                    .zip_eq(self.right.0.par_iter())
                    .map(|(poly, &right)| {
                        poly.iter()
                            .zip_eq(self.left.0.iter())
                            .map(|(&f, &left)| left * f)
                            .sum::<EF::ExtensionPacking>()
                            * right
                    })
                    .sum::<EF::ExtensionPacking>();
                EF::ExtensionPacking::to_ext_iter([e_part]).sum()
            })
            .collect::<Vec<_>>()
    }

    #[tracing::instrument(skip_all, fields(k = log2_strict_usize(out.len()), eqs = eqs.len()))]
    pub(super) fn into_packed(out: &mut [EF::ExtensionPacking], eqs: &[Self], scale: &[EF]) {
        if eqs.is_empty() {
            return;
        }
        let k = eqs.iter().map(Self::k).all_equal_value().unwrap();
        assert_eq!(out.len(), 1 << (k - log2_strict_usize(F::Packing::WIDTH)));
        assert_eq!(scale.len(), eqs.len());
        for (eq, &scale) in eqs.iter().zip(scale.iter()) {
            out.par_chunks_mut(eq.left.0.len())
                .zip(eq.right.0.par_iter())
                .for_each(|(chunk, &right)| {
                    chunk
                        .iter_mut()
                        .zip(eq.left.iter())
                        .for_each(|(out, &left)| *out += left * right * scale);
                });
        }
    }
}

#[cfg(test)]
mod test {
    use alloc::vec::Vec;

    use p3_field::{PrimeCharacteristicRing, dot_product, extension::BinomialExtensionField};
    use p3_koala_bear::KoalaBear;
    use rand::{Rng, SeedableRng, rngs::SmallRng};

    use crate::poly::{evals::EvaluationsList, multilinear::MultilinearPoint};

    #[test]
    fn test_accumulators() {
        type F = KoalaBear;
        type EF = BinomialExtensionField<F, 4>;
        let k = 10;
        let mut rng = SmallRng::seed_from_u64(1);

        let f = EvaluationsList::new((0..1 << k).map(|_| rng.random()).collect());
        let z = MultilinearPoint::<EF>::rand(&mut rng, f.num_variables());
        let eq = EvaluationsList::new_from_point(z.as_slice(), EF::ONE);

        for l in 1..k / 2 {
            let (z_svo, z_split) = z.split_at(l);
            let split_eq = super::SplitEq::<F, EF>::new(&z_split, EF::ONE);
            let partial_evals = split_eq.partial_evals(&f);
            let us = super::points_012::<F>(l);

            let u_evals0 = super::calculate_accumulators(&us[0], &partial_evals, &z_svo);
            us[0].iter().zip(u_evals0.iter()).for_each(|(u, &e0)| {
                let u = u.iter().copied().map(EF::from).collect::<Vec<_>>();
                let f = f.compress_multi(&u);
                let eq = eq.compress_multi(&u);
                let e1: EF = dot_product(eq.iter().copied(), f.iter().copied());
                assert_eq!(e0, e1);
            });

            let u_evals2 = super::calculate_accumulators(&us[1], &partial_evals, &z_svo);
            us[1].iter().zip(u_evals2.iter()).for_each(|(u, &e0)| {
                let u = u.iter().copied().map(EF::from).collect::<Vec<_>>();
                let f = f.compress_multi(&u);
                let eq = eq.compress_multi(&u);
                let e1: EF = dot_product(eq.iter().copied(), f.iter().copied());
                assert_eq!(e0, e1);
            });
        }
    }
}
