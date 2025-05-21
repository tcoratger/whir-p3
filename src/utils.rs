use p3_field::Field;

/// Computes the equality polynomial evaluations efficiently.
///
/// Given an evaluation point vector `eval`, the function computes
/// the equality polynomial recursively using the formula:
///
/// ```text
/// eq(X) = ∏ (1 - X_i + 2X_i z_i)
/// ```
///
/// where `z_i` are the constraint points.
pub(crate) fn eval_eq<F: Field>(eval: &[F], out: &mut [F], scalar: F) {
    // Ensure that the output buffer size is correct:
    // It should be of size `2^n`, where `n` is the number of variables.
    debug_assert_eq!(out.len(), 1 << eval.len());

    match eval.len() {
        0 => {
            out[0] += scalar;
        }
        1 => {
            // Manually unroll for single variable case
            let x = eval[0];
            let s1 = scalar * x;
            let s0 = scalar - s1;

            out[0] += s0;
            out[1] += s1;
        }
        2 => {
            // Manually unroll for two variables case
            let x0 = eval[0];
            let x1 = eval[1];

            let s1 = scalar * x0;
            let s0 = scalar - s1;

            let s01 = s0 * x1;
            let s00 = s0 - s01;
            let s11 = s1 * x1;
            let s10 = s1 - s11;

            out[0] += s00;
            out[1] += s01;
            out[2] += s10;
            out[3] += s11;
        }
        3 => {
            // Three variables case (manually unrolled)
            let x0 = eval[0];
            let x1 = eval[1];
            let x2 = eval[2];

            let s1 = scalar * x0;
            let s0 = scalar - s1;

            let s01 = s0 * x1;
            let s00 = s0 - s01;
            let s11 = s1 * x1;
            let s10 = s1 - s11;

            let s001 = s00 * x2;
            let s000 = s00 - s001;
            let s011 = s01 * x2;
            let s010 = s01 - s011;
            let s101 = s10 * x2;
            let s100 = s10 - s101;
            let s111 = s11 * x2;
            let s110 = s11 - s111;

            out[0] += s000;
            out[1] += s001;
            out[2] += s010;
            out[3] += s011;
            out[4] += s100;
            out[5] += s101;
            out[6] += s110;
            out[7] += s111;
        }
        _ => {
            let (&x, tail) = eval.split_first().unwrap();

            // Divide the output buffer into two halves: one for `X_i = 0` and one for `X_i = 1`
            let (low, high) = out.split_at_mut(out.len() / 2);

            // Compute weight updates for the two branches:
            // - `s0` corresponds to the case when `X_i = 0`
            // - `s1` corresponds to the case when `X_i = 1`
            //
            // Mathematically, this follows the recurrence:
            // ```text
            // eq_{X1, ..., Xn}(X) = (1 - X_1) * eq_{X2, ..., Xn}(X) + X_1 * eq_{X2, ..., Xn}(X)
            // ```
            let s1 = scalar * x; // Contribution when `X_i = 1`
            let s0 = scalar - s1; // Contribution when `X_i = 0`

            // Use parallel execution if the number of remaining variables is large.
            #[cfg(feature = "parallel")]
            {
                const PARALLEL_THRESHOLD: usize = 10;
                if tail.len() > PARALLEL_THRESHOLD {
                    rayon::join(|| eval_eq(tail, low, s0), || eval_eq(tail, high, s1));
                    return;
                }
            }
            // Default sequential execution
            eval_eq(tail, low, s0);
            eval_eq(tail, high, s1);
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    use super::*;

    type F = BabyBear;

    #[test]
    fn test_eval_eq_functionality() {
        let mut output = vec![F::ZERO; 4]; // n=2 → 2^2 = 4 elements
        let eval = vec![F::from_u64(1), F::from_u64(0)]; // (X1, X2) = (1,0)
        let scalar = F::from_u64(2);

        eval_eq(&eval, &mut output, scalar);

        // Expected results for (X1, X2) = (1,0)
        let expected_output = vec![F::ZERO, F::ZERO, F::from_u64(2), F::ZERO];

        assert_eq!(output, expected_output);
    }
}
