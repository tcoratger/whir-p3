use alloc::{format, string::ToString, vec::Vec};

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, TwoAdicField};

use crate::{
    poly::multilinear::MultilinearPoint,
    sumcheck::extrapolate_012,
    whir::{proof::SumcheckData, verifier::VerifierError},
};

/// Verifies standard sumcheck rounds and extracts folding randomness from the transcript.
///
/// This function reads from the Fiat–Shamir transcript to simulate verifier interaction
/// in the sumcheck protocol. For each round, it recovers:
/// - One univariate polynomial (usually degree ≤ 2) sent by the prover.
/// - One challenge scalar chosen by the verifier (folding randomness).
///
/// # Arguments
///
/// - `polynomial_evaluations`: The polynomial evaluations (c0, c2) for each round.
/// - `pow_witnesses`: PoW witnesses from the prover.
/// - `challenger`: The Fiat-Shamir challenger for transcript management.
/// - `claimed_sum`: Mutable reference to the claimed sum, updated each round.
/// - `pow_bits`: Proof-of-work difficulty (0 disables PoW).
///
/// # Returns
///
/// - A `MultilinearPoint` of folding randomness values in reverse order.
///   Common helper function to verify standard sumcheck rounds
pub(crate) fn verify_sumcheck_rounds<F, EF, Challenger>(
    sumcheck: &SumcheckData<F, EF>,
    challenger: &mut Challenger,
    claimed_sum: &mut EF,
    pow_bits: usize,
) -> Result<MultilinearPoint<EF>, VerifierError>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    let mut randomness = Vec::with_capacity(sumcheck.polynomial_evaluations.len());

    for (i, &[c0, c2]) in sumcheck.polynomial_evaluations.iter().enumerate() {
        // Derive h(1) from the sumcheck equation: h(0) + h(1) = claimed_sum

        // Observe only the sent polynomial evaluations (c0 and c2)
        challenger.observe_algebra_slice(&[c0, c2]);

        // Verify PoW (only if pow_bits > 0)
        if pow_bits > 0 && !challenger.check_witness(pow_bits, sumcheck.pow_witnesses[i]) {
            return Err(VerifierError::InvalidPowWitness);
        }

        // Sample challenge
        let r: EF = challenger.sample_algebra_element();
        // Evaluate sumcheck polynomial at r
        *claimed_sum = extrapolate_012(c0, *claimed_sum - c0, c2, r);
        randomness.push(r);
    }

    Ok(MultilinearPoint::new(randomness))
}

pub(crate) fn verify_initial_sumcheck_rounds_without_statement<F, EF, Challenger>(
    pow_witness: F,
    challenger: &mut Challenger,
    rounds: usize,
    pow_bits: usize,
) -> Result<MultilinearPoint<EF>, VerifierError>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    // No sumcheck - just sample folding randomness directly
    let randomness: Vec<EF> = (0..rounds)
        .map(|_| challenger.sample_algebra_element())
        .collect();

    // Check PoW
    if pow_bits > 0 && !challenger.check_witness(pow_bits, pow_witness) {
        return Err(VerifierError::InvalidPowWitness);
    }

    Ok(MultilinearPoint::new(randomness))
}

/// Verify the final sumcheck rounds.
///
/// # Returns
///
/// - A `MultilinearPoint` of folding randomness values in reverse order.
pub(crate) fn verify_final_sumcheck_rounds<F, EF, Challenger>(
    final_sumcheck: Option<&SumcheckData<F, EF>>,
    challenger: &mut Challenger,
    claimed_sum: &mut EF,
    rounds: usize,
    pow_bits: usize,
) -> Result<MultilinearPoint<EF>, VerifierError>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    if rounds == 0 {
        // No final sumcheck expected
        return Ok(MultilinearPoint::new(Vec::new()));
    }

    let sumcheck = final_sumcheck.ok_or_else(|| VerifierError::SumcheckFailed {
        round: 0,
        expected: format!("{rounds} final sumcheck rounds"),
        actual: "None".to_string(),
    })?;

    if sumcheck.polynomial_evaluations.len() != rounds {
        return Err(VerifierError::SumcheckFailed {
            round: 0,
            expected: format!("{rounds} rounds"),
            actual: format!("{} rounds in proof", sumcheck.polynomial_evaluations.len()),
        });
    }
    verify_sumcheck_rounds(sumcheck, challenger, claimed_sum, pow_bits)
}
#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::{SeedableRng, rngs::SmallRng};

    use super::*;
    use crate::{
        fiat_shamir::domain_separator::{DomainSeparator, SumcheckParams},
        parameters::{FoldingFactor, ProtocolParameters, errors::SecurityAssumption},
        poly::evals::EvaluationsList,
        sumcheck::sumcheck_prover::Sumcheck,
        whir::{
            constraints::statement::EqStatement,
            parameters::SumcheckStrategy,
            proof::{InitialPhase, WhirProof},
        },
    };

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;

    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

    // Digest size matches MyCompress output size (the 3rd parameter of TruncatedPermutation)
    const DIGEST_ELEMS: usize = 8;

    /// Constructs a default WHIR configuration for testing
    fn create_proof_from_test_protocol_params(
        num_variables: usize,
        folding_factor: FoldingFactor,
    ) -> WhirProof<F, EF, F, DIGEST_ELEMS> {
        // Create hash and compression functions for the Merkle tree
        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);

        let merkle_hash = MyHash::new(perm.clone());
        let merkle_compress = MyCompress::new(perm);

        // Construct WHIR protocol parameters
        let whir_params = ProtocolParameters {
            initial_statement: true,
            security_level: 32,
            pow_bits: 0,
            rs_domain_initial_reduction_factor: 1,
            folding_factor,
            merkle_hash,
            merkle_compress,
            soundness_type: SecurityAssumption::UniqueDecoding,
            starting_log_inv_rate: 1,
        };

        // Combine protocol and polynomial parameters into a single config
        WhirProof::from_protocol_parameters(&whir_params, num_variables)
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_read_sumcheck_rounds_variants() {
        // Define a multilinear polynomial in 3 variables:
        // f(X0, X1, X2) = 1 + 2*X2 + 3*X1 + 4*X1*X2
        //              + 5*X0 + 6*X0*X2 + 7*X0*X1 + 8*X0*X1*X2
        let e1 = F::from_u64(1);
        let e2 = F::from_u64(2);
        let e3 = F::from_u64(3);
        let e4 = F::from_u64(4);
        let e5 = F::from_u64(5);
        let e6 = F::from_u64(6);
        let e7 = F::from_u64(7);
        let e8 = F::from_u64(8);

        let evals = EvaluationsList::new(vec![
            e1,
            e1 + e2,
            e1 + e3,
            e1 + e2 + e3 + e4,
            e1 + e5,
            e1 + e2 + e5 + e6,
            e1 + e3 + e5 + e7,
            e1 + e2 + e3 + e4 + e5 + e6 + e7 + e8,
        ]);

        // Define the actual polynomial function over EF4
        let f = |x0: EF, x1: EF, x2: EF| {
            x2 * e2
                + x1 * e3
                + x1 * x2 * e4
                + x0 * e5
                + x0 * x2 * e6
                + x0 * x1 * e7
                + x0 * x1 * x2 * e8
                + e1
        };

        let n_vars = evals.num_variables();
        assert_eq!(n_vars, 3);

        // Create a constraint system with evaluations of f at various points
        let mut statement = EqStatement::initialize(n_vars);

        let x_000 = MultilinearPoint::new(vec![EF::ZERO, EF::ZERO, EF::ZERO]);
        let x_100 = MultilinearPoint::new(vec![EF::ONE, EF::ZERO, EF::ZERO]);
        let x_110 = MultilinearPoint::new(vec![EF::ONE, EF::ONE, EF::ZERO]);
        let x_111 = MultilinearPoint::new(vec![EF::ONE, EF::ONE, EF::ONE]);
        let x_011 = MultilinearPoint::new(vec![EF::ZERO, EF::ONE, EF::ONE]);

        let f_000 = f(EF::ZERO, EF::ZERO, EF::ZERO);
        let f_100 = f(EF::ONE, EF::ZERO, EF::ZERO);
        let f_110 = f(EF::ONE, EF::ONE, EF::ZERO);
        let f_111 = f(EF::ONE, EF::ONE, EF::ONE);
        let f_011 = f(EF::ZERO, EF::ONE, EF::ONE);

        statement.add_evaluated_constraint(x_000, f_000);
        statement.add_evaluated_constraint(x_100, f_100);
        statement.add_evaluated_constraint(x_110, f_110);
        statement.add_evaluated_constraint(x_111, f_111);
        statement.add_evaluated_constraint(x_011, f_011);

        let folding_factor = 3;
        let pow_bits = 0;

        // Set up domain separator
        // - Add sumcheck
        let mut domsep: DomainSeparator<EF, F> = DomainSeparator::new(vec![]);
        domsep.add_sumcheck(&SumcheckParams {
            rounds: folding_factor,
            pow_bits,
        });

        let mut rng = SmallRng::seed_from_u64(1);
        let challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));
        let mut prover_challenger = challenger.clone();

        // Initialize proof and challenger
        let mut proof =
            create_proof_from_test_protocol_params(n_vars, FoldingFactor::Constant(folding_factor));
        domsep.observe_domain_separator(&mut prover_challenger);

        // Extract sumcheck data from the initial phase
        let InitialPhase::WithStatement { ref mut data, .. } = proof.initial_phase else {
            panic!("Expected WithStatement variant");
        };

        // Instantiate the prover with base field coefficients
        let (_, _) = Sumcheck::<F, EF>::from_base_evals(
            SumcheckStrategy::default(),
            &evals,
            data,
            &mut prover_challenger,
            folding_factor,
            pow_bits,
            &statement,
        );

        // Reconstruct verifier state to simulate the rounds
        let mut verifier_challenger = challenger;
        domsep.observe_domain_separator(&mut verifier_challenger);

        // Save a fresh copy for verify_initial_sumcheck_rounds
        let mut verifier_challenger_for_verify = verifier_challenger.clone();

        let mut t = EvaluationsList::zero(statement.num_variables());
        let mut expected_initial_sum = EF::ZERO;
        statement.combine_hypercube::<F, false>(&mut t, &mut expected_initial_sum, EF::ONE);
        // Start with the claimed sum before folding
        let mut current_sum = expected_initial_sum;

        let mut expected = Vec::with_capacity(folding_factor);

        // Extract and verify each sumcheck round
        let InitialPhase::WithStatement {
            data: initial_sumcheck_data,
            ..
        } = &proof.initial_phase
        else {
            panic!("Expected WithStatement variant")
        };

        // First round: read c_0 = h(0) and c_2 (quadratic coefficient)
        let [c_0, c_2] = initial_sumcheck_data.polynomial_evaluations[0];
        let h_1 = current_sum - c_0;

        // Observe polynomial evaluations (must match what verify_initial_sumcheck_rounds does)
        verifier_challenger.observe_algebra_slice(&[c_0, c_2]);

        // Sample random challenge r_i ∈ EF4 and evaluate h_i(r_i)
        let r: EF = verifier_challenger.sample_algebra_element();
        // h(r) = c_2 * r^2 + (h(1) - c_0 - c_2) * r + c_0
        current_sum = c_2 * r.square() + (h_1 - c_0 - c_2) * r + c_0;
        expected.push(r);

        for i in 0..folding_factor - 1 {
            // Read c_0 = h(0) and c_2 (quadratic coefficient), derive h(1) = claimed_sum - c_0
            let [c_0, c_2] = initial_sumcheck_data.polynomial_evaluations[i + 1];
            let h_1 = current_sum - c_0;

            // Observe polynomial evaluations
            verifier_challenger.observe_algebra_slice(&[c_0, c_2]);

            // Sample random challenge r
            let r: EF = verifier_challenger.sample_algebra_element();
            // h(r) = c_2 * r^2 + (h(1) - c_0 - c_2) * r + c_0
            current_sum = c_2 * r.square() + (h_1 - c_0 - c_2) * r + c_0;

            if pow_bits > 0 {
                // verifier_state.challenge_pow::<Blake3PoW>(pow_bits).unwrap();
            }

            expected.push(r);
        }

        let randomness = verify_sumcheck_rounds(
            proof.initial_phase.sumcheck_data().unwrap(),
            &mut verifier_challenger_for_verify,
            &mut expected_initial_sum,
            pow_bits,
        )
        .unwrap();

        // Check that number of parsed rounds is correct
        assert_eq!(randomness.num_variables(), folding_factor);

        // Reconstruct the expected MultilinearPoint from expected randomness
        let expected_randomness = MultilinearPoint::new(expected);
        assert_eq!(
            randomness, expected_randomness,
            "Mismatch in full MultilinearPoint folding randomness"
        );
    }
}
