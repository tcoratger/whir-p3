use p3_field::Field;

pub mod product_polynomial;
pub mod sumcheck_prover;
pub(super) mod svo;

#[cfg(test)]
mod tests;

/// Given r, finds L0(r), L1(r), L2(r) where Li are the Lagrange basis polynomials for points 0,1,2.
pub(super) fn lagrange_weights_012<F: Field>(r: F) -> [F; 3] {
    // L0(r) = (r-1)(r-2)/((0-1)(0-2)) = (r-1)(r-2)/2
    // L1(r) = (r-0)(r-2)/((1-0)(1-2)) = -r(r-2)
    // L2(r) = (r-0)(r-1)/((2-0)(2-1)) = r(r-1)/2
    let inv_two = F::TWO.inverse();
    let l0 = (r - F::ONE) * (r - F::TWO) * inv_two;
    let l1 = r * (F::TWO - r);
    let l2 = (r * (r - F::ONE)) * inv_two;
    [l0, l1, l2]
}

/// Given h(0), h(1), h(2) for a quadratic polynomial h, evaluate h(r).
pub(super) fn extrapolate_012<F: Field>(e0: F, e1: F, e2: F, r: F) -> F {
    let w = lagrange_weights_012(r);
    e0 * w[0] + e1 * w[1] + e2 * w[2]
}

#[cfg(test)]
mod test {
    use p3_field::PrimeCharacteristicRing;
    use p3_koala_bear::KoalaBear;
    use rand::{Rng, SeedableRng, rngs::SmallRng};

    use crate::sumcheck::extrapolate_012;

    #[test]
    fn test_extrapolate() {
        use p3_field::Field;

        type F = KoalaBear;
        let mut rng = SmallRng::seed_from_u64(1);

        let e: [F; 3] = rng.random();
        assert_eq!(extrapolate_012(e[0], e[1], e[2], F::ZERO), e[0]);
        assert_eq!(extrapolate_012(e[0], e[1], e[2], F::ONE), e[1]);
        assert_eq!(extrapolate_012(e[0], e[1], e[2], F::TWO), e[2]);

        let e: [F; 3] = rng.random();
        let r: F = rng.random();

        let c0 = e[0];
        let c2 = (e[0] - e[1].double() + e[2]) * F::TWO.inverse();
        let c1 = e[1] - e[0] - c2;

        let v0 = c0 + c1 * r + c2 * r * r;
        let v1 = extrapolate_012(e[0], e[1], e[2], r);
        assert_eq!(v0, v1);
    }
}
