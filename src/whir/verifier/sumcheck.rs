use p3_challenger::{CanObserve, CanSample};
use p3_field::{ExtensionField, Field, PrimeField64, TwoAdicField};
use p3_interpolation::interpolate_subgroup;
use p3_matrix::dense::RowMajorMatrix;

use crate::{
    fiat_shamir::{
        duplex_sponge::interface::Unit,
        errors::{ProofError, ProofResult},
        pow::traits::PowStrategy,
        verifier::VerifierState,
    },
    poly::multilinear::MultilinearPoint,
    sumcheck::{K_SKIP_SUMCHECK, sumcheck_polynomial::SumcheckPolynomial},
    whir::Verifier,
};

/// The full vector of folding randomness values, in reverse round order.
type SumcheckRandomness<F> = MultilinearPoint<F>;

impl<EF, F, H, C, PS, Challenger, W> Verifier<'_, EF, F, H, C, PS, Challenger, W>
where
    F: Field + TwoAdicField + PrimeField64,
    EF: ExtensionField<F> + TwoAdicField,
    PS: PowStrategy,
    W: Unit + Default + Copy,
    Challenger: CanObserve<W> + CanSample<W>,
{
    /// Extracts a sequence of `(SumcheckPolynomial, folding_randomness)` pairs from the verifier transcript,
    /// and computes the corresponding `MultilinearPoint` folding randomness in reverse order.
    ///
    /// This function reads from the Fiat–Shamir transcript to simulate verifier interaction
    /// in the sumcheck protocol. For each round, it recovers:
    /// - One univariate polynomial (usually degree ≤ 2) sent by the prover.
    /// - One challenge scalar chosen by the verifier (folding randomness).
    ///
    /// ## Modes
    ///
    /// - **Standard mode** (`is_univariate_skip = false`):
    ///   Each round represents a single variable being folded.
    ///   The polynomial is evaluated at 3 points, typically `{0, 1, r}` for quadratic reduction.
    ///
    /// - **Univariate skip mode** (`is_univariate_skip = true`):
    ///   The first `K_SKIP_SUMCHECK` variables are folded simultaneously by evaluating a single univariate polynomial
    ///   over a coset of size `2^{k+1}`. This yields a larger polynomial but avoids several later rounds.
    ///
    /// # Arguments
    ///
    /// - `verifier_state`: The verifier's Fiat–Shamir transcript state.
    /// - `rounds`: Total number of variables being folded.
    /// - `pow_bits`: Optional proof-of-work difficulty (0 disables PoW).
    /// - `is_univariate_skip`: If true, apply the univariate skip optimization on the first `K_SKIP_SUMCHECK` variables.
    ///
    /// # Returns
    ///
    /// - A `MultilinearPoint` of folding randomness values in reverse order.
    pub(crate) fn verify_sumcheck_rounds(
        &self,
        verifier_state: &mut VerifierState<'_, EF, F, Challenger, W>,
        claimed_sum: &mut EF,
        rounds: usize,
        pow_bits: f64,
        is_univariate_skip: bool,
    ) -> ProofResult<SumcheckRandomness<EF>> {
        // Calculate how many `(poly, rand)` pairs to expect based on skip mode
        //
        // If skipping: we do 1 large round for the skip, and the remaining normally
        let effective_rounds = if is_univariate_skip {
            1 + (rounds - K_SKIP_SUMCHECK)
        } else {
            rounds
        };

        // Preallocate vector to hold the randomness values
        let mut randomness = Vec::with_capacity(effective_rounds);

        // Handle the univariate skip case
        if is_univariate_skip {
            // Read `2^{k+1}` evaluations (size of coset domain) for the skipping polynomial
            let evals = verifier_state.next_scalars::<{ 1 << (K_SKIP_SUMCHECK + 1) }>()?;

            // Interpolate into a univariate polynomial (over the coset domain)
            let poly = SumcheckPolynomial::new(evals.to_vec(), 1);

            // Sample the challenge scalar r₀ ∈ 𝔽 for this round
            let [rand] = verifier_state.challenge_scalars()?;

            // Update the claimed sum using the univariate polynomial and randomness.
            //
            // We interpolate the univariate polynomial at the randomness point.
            *claimed_sum =
                interpolate_subgroup(&RowMajorMatrix::new_col(poly.evaluations().to_vec()), rand)
                    [0];

            // Record this round’s randomness
            randomness.push(rand);

            // Optional: apply proof-of-work query
            self.verify_proof_of_work(verifier_state, pow_bits)?;
        }

        // Continue with the remaining sumcheck rounds (each using 3 evaluations)
        let start_round = if is_univariate_skip {
            K_SKIP_SUMCHECK // skip the first k rounds
        } else {
            0
        };

        for _ in start_round..rounds {
            // Extract the 3 evaluations of the quadratic sumcheck polynomial h(X)
            let evals = verifier_state.next_scalars::<3>()?;
            let poly = SumcheckPolynomial::new(evals.to_vec(), 1);

            // Verify claimed sum is consistent with polynomial
            if poly.sum_over_boolean_hypercube() != *claimed_sum {
                return Err(ProofError::InvalidProof);
            }

            // Sample the next verifier folding randomness rᵢ
            let [rand] = verifier_state.challenge_scalars()?;

            // Update claimed sum using folding randomness
            *claimed_sum = poly.evaluate_at_point(&rand.into());

            // Store this round’s randomness
            randomness.push(rand);

            // Optional PoW interaction (grinding resistance)
            self.verify_proof_of_work(verifier_state, pow_bits)?;
        }

        // We should reverse the order of the randomness points:
        // This is because the randomness points are originally reverted at the end of the sumcheck rounds.
        randomness.reverse();

        Ok(MultilinearPoint(randomness))
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_challenger::HashChallenger;
    use p3_dft::NaiveDft;
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};
    use p3_interpolation::interpolate_subgroup;
    use p3_keccak::Keccak256Hash;
    use p3_matrix::dense::RowMajorMatrix;

    use super::*;
    use crate::{
        fiat_shamir::domain_separator::DomainSeparator,
        parameters::{
            FoldingFactor, MultivariateParameters, ProtocolParameters, errors::SecurityAssumption,
        },
        poly::{coeffs::CoefficientList, evals::EvaluationStorage, multilinear::MultilinearPoint},
        sumcheck::sumcheck_single::SumcheckSingle,
        whir::{
            Blake3PoW, ByteHash, FieldHash, MyCompress, W,
            parameters::WhirConfig,
            statement::{Statement, weights::Weights},
        },
    };

    type F = BabyBear;
    type EF4 = BinomialExtensionField<F, 4>;
    type H = HashChallenger<u8, Keccak256Hash, 32>;

    /// Constructs a default WHIR configuration for testing
    fn default_whir_config(
        num_variables: usize,
    ) -> WhirConfig<EF4, F, FieldHash, MyCompress, Blake3PoW, H, W> {
        // Create hash and compression functions for the Merkle tree
        let byte_hash = ByteHash {};
        let merkle_hash = FieldHash::new(byte_hash);
        let merkle_compress = MyCompress::new(byte_hash);

        // Set the multivariate polynomial parameters
        let mv_params = MultivariateParameters::<EF4>::new(num_variables);

        // Construct WHIR protocol parameters
        let whir_params = ProtocolParameters::<_, _> {
            initial_statement: true,
            security_level: 32,
            pow_bits: 0,
            rs_domain_initial_reduction_factor: 1,
            folding_factor: FoldingFactor::Constant(2),
            merkle_hash,
            merkle_compress,
            soundness_type: SecurityAssumption::UniqueDecoding,
            starting_log_inv_rate: 1,
        };

        // Combine protocol and polynomial parameters into a single config
        WhirConfig::<EF4, F, FieldHash, MyCompress, Blake3PoW, H, W>::new(mv_params, whir_params)
    }

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
        let mut expected_initial_sum = evals_w[0] * evals_f[0]
            + evals_w[1] * evals_f[1]
            + evals_w[2] * evals_f[2]
            + evals_w[3] * evals_f[3]
            + evals_w[4] * evals_f[4]
            + evals_w[5] * evals_f[5]
            + evals_w[6] * evals_f[6]
            + evals_w[7] * evals_f[7];
        assert_eq!(prover.sum, expected_initial_sum);

        // Set up domain separator
        let mut domsep: DomainSeparator<EF4, F, u8> = DomainSeparator::new("tag");

        let folding_factor = 3;
        let pow_bits = 1.;

        // Reserve the number of interactions required
        for _ in 0..folding_factor {
            domsep.add_scalars(3, "tag");
            domsep.challenge_scalars(1, "tag");
            domsep.challenge_pow("pow_queries");
        }

        let challenger = H::new(vec![], Keccak256Hash);

        // Convert domain separator into prover state object
        let mut prover_state = domsep.to_prover_state::<_, 32>(challenger.clone());

        // Perform sumcheck folding using Fiat-Shamir-derived randomness and PoW
        let _ = prover
            .compute_sumcheck_polynomials::<Blake3PoW, _, _, _>(
                &mut prover_state,
                folding_factor,
                pow_bits,
                None,
                &NaiveDft,
            )
            .unwrap();

        // Reconstruct verifier state to simulate the rounds
        let mut verifier_state =
            domsep.to_verifier_state::<H, 32>(prover_state.narg_string(), challenger.clone());

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
        let mut verifier_state =
            domsep.to_verifier_state::<_, 32>(prover_state.narg_string(), challenger);

        // Setup the WHIR verifier
        let whir_config = default_whir_config(n_vars);
        let verifier =
            Verifier::<EF4, F, FieldHash, MyCompress, Blake3PoW, H, W>::new(&whir_config);

        let randomness = verifier
            .verify_sumcheck_rounds(
                &mut verifier_state,
                &mut expected_initial_sum,
                folding_factor,
                pow_bits,
                false,
            )
            .unwrap();

        // Check that number of parsed rounds is correct
        assert_eq!(randomness.0.len(), folding_factor);

        // Reconstruct the expected MultilinearPoint from reversed order of expected randomness
        let expected_randomness =
            MultilinearPoint(expected.iter().map(|&(_, r)| r).rev().collect());
        assert_eq!(
            randomness, expected_randomness,
            "Mismatch in full MultilinearPoint folding randomness"
        );
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_read_sumcheck_rounds_with_univariate_skip() {
        // -------------------------------------------------------------
        // Define a multilinear polynomial in K = K_SKIP_SUMCHECK + 3 variables.
        // The number of coefficients must be 2^K.
        // -------------------------------------------------------------
        const K_SKIP: usize = K_SKIP_SUMCHECK;
        const NUM_VARS: usize = K_SKIP + 3;
        let num_points = 1 << NUM_VARS;

        // Construct simple deterministic coefficients f = [1, 2, ..., 2^K]
        let coeffs: Vec<F> = (1..=num_points).map(F::from_u64).collect();
        let coeffs = CoefficientList::new(coeffs);

        assert_eq!(coeffs.num_variables(), NUM_VARS);

        // -------------------------------------------------------------
        // Construct a Statement by evaluating f at several Boolean points
        // These evaluations will serve as equality constraints
        // -------------------------------------------------------------
        let mut statement = Statement::new(NUM_VARS);
        for i in 0..5 {
            let bool_point: Vec<_> = (0..NUM_VARS)
                .map(|j| {
                    if (i >> j) & 1 == 1 {
                        EF4::ONE
                    } else {
                        EF4::ZERO
                    }
                })
                .collect();
            let ml_point = MultilinearPoint(bool_point.clone());
            let expected_val = coeffs.evaluate(&ml_point);
            statement.add_constraint(Weights::evaluation(ml_point), expected_val);
        }

        // -------------------------------------------------------------
        // Construct prover with base coefficients
        // -------------------------------------------------------------
        let mut prover = SumcheckSingle::<F, EF4>::from_base_coeffs(coeffs, &statement, EF4::ONE);

        // Compute expected weighted sum: dot product of f(b) * w(b)
        let evals_f = match prover.evaluation_of_p {
            EvaluationStorage::Base(ref evals) => evals.evals(),
            EvaluationStorage::Extension(_) => panic!("Expected base evaluation"),
        };
        let evals_w = prover.weights.evals();

        let mut expected_sum = evals_f
            .iter()
            .zip(evals_w)
            .map(|(a, b)| *b * *a)
            .sum::<EF4>();
        assert_eq!(prover.sum, expected_sum);

        // -------------------------------------------------------------
        // Simulate Fiat-Shamir transcript
        // Reserve interactions for:
        // - 1 skipped round: 2^k_skip + 1 values
        // - remaining rounds: 3 values each
        // -------------------------------------------------------------
        let mut domsep: DomainSeparator<EF4, F, u8> = DomainSeparator::new("test");
        domsep.add_scalars(1 << (K_SKIP + 1), "skip");
        domsep.challenge_scalars(1, "skip");

        for _ in 0..(NUM_VARS - K_SKIP) {
            domsep.add_scalars(3, "round");
            domsep.challenge_scalars(1, "round");
        }

        let challenger = H::new(vec![], Keccak256Hash);

        // Convert to prover state
        let mut prover_state = domsep.to_prover_state::<_, 32>(challenger.clone());

        // -------------------------------------------------------------
        // Run prover-side folding
        // -------------------------------------------------------------
        let _ = prover
            .compute_sumcheck_polynomials::<Blake3PoW, _, _, _>(
                &mut prover_state,
                NUM_VARS,
                0.0,
                Some(K_SKIP),
                &NaiveDft,
            )
            .unwrap();

        // -------------------------------------------------------------
        // Manually extract expected sumcheck rounds by replaying transcript
        // -------------------------------------------------------------
        let mut verifier_state =
            domsep.to_verifier_state::<H, 32>(prover_state.narg_string(), challenger.clone());
        let mut expected = Vec::new();

        // First skipped round (wide DFT LDE)
        let evals: [_; 1 << (K_SKIP + 1)] = verifier_state.next_scalars().unwrap();
        let poly = SumcheckPolynomial::new(evals.to_vec(), 1);
        let [r0] = verifier_state.challenge_scalars().unwrap();
        expected.push((poly, r0));

        let mat = RowMajorMatrix::new(evals.to_vec(), 1);
        let mut current_sum = interpolate_subgroup(&mat, r0)[0];

        // Remaining quadratic rounds
        for _ in 0..(NUM_VARS - K_SKIP) {
            let evals: [_; 3] = verifier_state.next_scalars().unwrap();
            let poly = SumcheckPolynomial::new(evals.to_vec(), 1);
            let [r] = verifier_state.challenge_scalars().unwrap();
            assert_eq!(poly.evaluations()[0] + poly.evaluations()[1], current_sum);
            current_sum = poly.evaluate_at_point(&r.into());
            expected.push((poly, r));
        }

        // Setup the WHIR verifier
        let whir_config = default_whir_config(NUM_VARS);
        let verifier =
            Verifier::<EF4, F, FieldHash, MyCompress, Blake3PoW, H, W>::new(&whir_config);

        // -------------------------------------------------------------
        // Use verify_sumcheck_rounds with skip enabled
        // -------------------------------------------------------------
        let mut verifier_state =
            domsep.to_verifier_state::<_, 32>(prover_state.narg_string(), challenger);
        let randomness = verifier
            .verify_sumcheck_rounds(&mut verifier_state, &mut expected_sum, NUM_VARS, 0.0, true)
            .unwrap();

        // Check length:
        // - 1 randomness for the first K skipped rounds
        // - 1 randomness for each regular round
        assert_eq!(randomness.0.len(), NUM_VARS - K_SKIP + 1);

        // Reconstruct the expected MultilinearPoint from reversed order of expected randomness
        let expected_randomness =
            MultilinearPoint(expected.iter().map(|&(_, r)| r).rev().collect());
        assert_eq!(
            randomness, expected_randomness,
            "Mismatch in full MultilinearPoint folding randomness"
        );
    }
}
