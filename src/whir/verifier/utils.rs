use p3_field::{ExtensionField, Field, PrimeField64, TwoAdicField};

use crate::{
    fiat_shamir::{errors::ProofResult, pow::traits::PowStrategy, verifier::VerifierState},
    sumcheck::sumcheck_polynomial::SumcheckPolynomial,
};

/// Extracts a sequence of `(SumcheckPolynomial, folding_randomness)` pairs from the verifier state.
///
/// - `num_rounds`: Number of sumcheck rounds to extract.
/// - `pow_bits`: Optional PoW challenge after each round (0 disables).
/// - `verifier_state`: Verifier interaction state.
///
/// Returns a vector of `(SumcheckPolynomial, folding_randomness)` tuples.
pub(crate) fn read_sumcheck_rounds<EF, F, PS>(
    verifier_state: &mut VerifierState<'_, EF, F>,
    num_rounds: usize,
    pow_bits: f64,
) -> ProofResult<Vec<(SumcheckPolynomial<EF>, EF)>>
where
    F: Field + TwoAdicField + PrimeField64,
    EF: ExtensionField<F> + TwoAdicField,
    PS: PowStrategy,
{
    let mut result = Vec::with_capacity(num_rounds);
    for _ in 0..num_rounds {
        let evals: [_; 3] = verifier_state.next_scalars()?;
        let poly = SumcheckPolynomial::new(evals.to_vec(), 1);
        let [rand] = verifier_state.challenge_scalars()?;
        result.push((poly, rand));

        if pow_bits > 0.0 {
            verifier_state.challenge_pow::<PS>(pow_bits)?;
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_dft::NaiveDft;
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};

    use super::*;
    use crate::{
        fiat_shamir::domain_separator::DomainSeparator,
        poly::{coeffs::CoefficientList, evals::EvaluationStorage, multilinear::MultilinearPoint},
        sumcheck::sumcheck_single::SumcheckSingle,
        whir::{
            Blake3PoW,
            statement::{Statement, Weights},
        },
    };

    type F = BabyBear;
    type EF4 = BinomialExtensionField<F, 4>;

    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_read_sumcheck_rounds_variants() {
        // Define a multilinear polynomial in 3 variables:
        // f(X0, X1, X2) = 1 + 2*X2 + 3*X1 + 4*X1*X2
        //              + 5*X0 + 6*X0*X2 + 7*X0*X1 + 8*X0*X1*X2
        let c1 = F::from_u64(1);
        let c2 = F::from_u64(2);
        let c3 = F::from_u64(3);
        let c4 = F::from_u64(4);
        let c5 = F::from_u64(5);
        let c6 = F::from_u64(6);
        let c7 = F::from_u64(7);
        let c8 = F::from_u64(8);

        let coeffs = CoefficientList::new(vec![c1, c2, c3, c4, c5, c6, c7, c8]);

        // Define the actual polynomial function over EF4
        let f = |x0: EF4, x1: EF4, x2: EF4| {
            x2 * c2
                + x1 * c3
                + x1 * x2 * c4
                + x0 * c5
                + x0 * x2 * c6
                + x0 * x1 * c7
                + x0 * x1 * x2 * c8
                + c1
        };

        let n_vars = coeffs.num_variables();
        assert_eq!(n_vars, 3);

        // Create a constraint system with evaluations of f at various points
        let mut statement = Statement::new(n_vars);

        let x_000 = MultilinearPoint(vec![EF4::ZERO, EF4::ZERO, EF4::ZERO]);
        let x_100 = MultilinearPoint(vec![EF4::ONE, EF4::ZERO, EF4::ZERO]);
        let x_110 = MultilinearPoint(vec![EF4::ONE, EF4::ONE, EF4::ZERO]);
        let x_111 = MultilinearPoint(vec![EF4::ONE, EF4::ONE, EF4::ONE]);
        let x_011 = MultilinearPoint(vec![EF4::ZERO, EF4::ONE, EF4::ONE]);

        let f_000 = f(EF4::ZERO, EF4::ZERO, EF4::ZERO);
        let f_100 = f(EF4::ONE, EF4::ZERO, EF4::ZERO);
        let f_110 = f(EF4::ONE, EF4::ONE, EF4::ZERO);
        let f_111 = f(EF4::ONE, EF4::ONE, EF4::ONE);
        let f_011 = f(EF4::ZERO, EF4::ONE, EF4::ONE);

        statement.add_constraint(Weights::evaluation(x_000), f_000);
        statement.add_constraint(Weights::evaluation(x_100), f_100);
        statement.add_constraint(Weights::evaluation(x_110), f_110);
        statement.add_constraint(Weights::evaluation(x_111), f_111);
        statement.add_constraint(Weights::evaluation(x_011), f_011);

        // Instantiate the prover with base field coefficients
        let mut prover = SumcheckSingle::<F, EF4>::from_base_coeffs(coeffs, &statement, EF4::ONE);

        // Get the f evaluations
        let evals_f = match prover.evaluation_of_p {
            EvaluationStorage::Base(ref evals_f) => evals_f.evals(),
            EvaluationStorage::Extension(_) => panic!("We should be in the base field"),
        };
        // Get the w evaluations
        let evals_w = prover.weights.evals();

        // Compute expected sum of evaluations via dot product
        let expected_initial_sum = evals_w[0] * evals_f[0]
            + evals_w[1] * evals_f[1]
            + evals_w[2] * evals_f[2]
            + evals_w[3] * evals_f[3]
            + evals_w[4] * evals_f[4]
            + evals_w[5] * evals_f[5]
            + evals_w[6] * evals_f[6]
            + evals_w[7] * evals_f[7];
        assert_eq!(prover.sum, expected_initial_sum);

        // Set up domain separator
        let mut domsep: DomainSeparator<EF4, F> = DomainSeparator::new("tag");

        let folding_factor = 3;
        let pow_bits = 1.;

        // Reserve the number of interactions required
        for _ in 0..folding_factor {
            domsep.add_scalars(3, "tag");
            domsep.challenge_scalars(1, "tag");
            domsep.challenge_pow("pow_queries");
        }

        // Convert domain separator into prover state object
        let mut prover_state = domsep.to_prover_state();

        // Perform sumcheck folding using Fiat-Shamir-derived randomness and PoW
        let result_sumcheck = prover
            .compute_sumcheck_polynomials::<Blake3PoW, _>(
                &mut prover_state,
                folding_factor,
                pow_bits,
                None,
                &NaiveDft,
            )
            .unwrap();

        // Reconstruct verifier state to simulate the rounds
        let mut verifier_state = domsep.to_verifier_state(prover_state.narg_string());

        // Start with the claimed sum before folding
        let mut current_sum = expected_initial_sum;

        let mut expected = Vec::with_capacity(folding_factor);

        for i in 0..folding_factor {
            // Get the 3 evaluations of sumcheck polynomial h_i(X) at X = 0, 1, 2
            let sumcheck_evals: [_; 3] = verifier_state.next_scalars().unwrap();
            let poly = SumcheckPolynomial::new(sumcheck_evals.to_vec(), 1);

            // Verify sum over Boolean points {0,1} matches current sum
            let sum = poly.evaluations()[0] + poly.evaluations()[1];
            assert_eq!(
                sum, current_sum,
                "Sumcheck round {i}: sum rule failed (h(0) + h(1) != current_sum)"
            );

            // Sample random challenge r_i ∈ F and evaluate h_i(r_i)
            let [r] = verifier_state.challenge_scalars().unwrap();
            current_sum = poly.evaluate_at_point(&r.into());

            if pow_bits > 0.0 {
                verifier_state.challenge_pow::<Blake3PoW>(pow_bits).unwrap();
            }

            expected.push((poly, r));
        }

        // Reconstruct verifier's view of the transcript using the DomainSeparator and prover's data
        let mut verifier_state = domsep.to_verifier_state(prover_state.narg_string());

        let result = read_sumcheck_rounds::<EF4, F, Blake3PoW>(
            &mut verifier_state,
            folding_factor,
            pow_bits,
        )
        .unwrap();

        // Check that number of parsed rounds is correct
        assert_eq!(result.len(), folding_factor);

        // Verify that parsed (poly, randomness) tuples match those we simulated
        for (i, ((expected_poly, expected_rand), (actual_poly, actual_rand))) in
            expected.iter().zip(result.iter()).enumerate()
        {
            assert_eq!(
                actual_poly, expected_poly,
                "Mismatch in round {i}: polynomial"
            );
            assert_eq!(
                actual_rand, expected_rand,
                "Mismatch in round {i}: folding randomness"
            );
        }

        // Check that sumcheck result's final points (reverse order) match the parsed randomness
        assert_eq!(result_sumcheck.0.len(), result.len());
        for (i, r) in result.iter().rev().enumerate() {
            assert_eq!(
                result_sumcheck.0[i], r.1,
                "Mismatch in reverse order randomness at index {i}"
            );
        }
    }
}
