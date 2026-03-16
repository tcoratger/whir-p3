use alloc::vec::Vec;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_multilinear_util::multilinear::MultilinearPoint;
use serde::{Deserialize, Serialize};

use super::{SumcheckError, extrapolate_012};

/// Sumcheck polynomial data
///
/// Stores the polynomial evaluations for sumcheck rounds in a compact format.
/// Each round stores `[h(0), h(2)]` where `h(1)` is derived as `claimed_sum - h(0)`.
#[derive(Default, Serialize, Deserialize, Clone, Debug)]
pub struct SumcheckData<F, EF> {
    /// Polynomial evaluations for each sumcheck round
    ///
    /// Each entry is `[h(0), h(2)]` - the evaluations at 0 and 2
    ///
    /// `h(1)` is derived as `claimed_sum - h(0)` by the verifier
    ///
    /// Length: folding_factor
    pub(crate) polynomial_evaluations: Vec<[EF; 2]>,

    /// PoW witnesses for each sumcheck round
    /// Length: folding_factor
    pub(crate) pow_witnesses: Vec<F>,
}

impl<F, EF> SumcheckData<F, EF> {
    /// Returns the polynomial evaluations `[h(0), h(2)]` for each round.
    #[must_use]
    pub fn polynomial_evaluations(&self) -> &[[EF; 2]] {
        &self.polynomial_evaluations
    }

    /// Returns the number of rounds stored in this proof data.
    #[must_use]
    pub const fn num_rounds(&self) -> usize {
        self.polynomial_evaluations.len()
    }

    /// Commits polynomial coefficients to the transcript and returns a challenge.
    ///
    /// This helper function handles the Fiat-Shamir interaction for a sumcheck round.
    ///
    /// # Arguments
    ///
    /// * `challenger` - Fiat-Shamir transcript.
    /// * `c0` - Constant coefficient `h(0)`.
    /// * `c2` - Quadratic coefficient.
    /// * `pow_bits` - PoW difficulty (0 to skip grinding).
    ///
    /// # Returns
    ///
    /// The sampled challenge `r \in EF`.
    pub fn observe_and_sample<Challenger, BF>(
        &mut self,
        challenger: &mut Challenger,
        c0: EF,
        c2: EF,
        pow_bits: usize,
    ) -> EF
    where
        BF: Field,
        EF: ExtensionField<BF>,
        F: Clone,
        Challenger: FieldChallenger<BF> + GrindingChallenger<Witness = F>,
    {
        // Record the polynomial coefficients in the proof.
        self.polynomial_evaluations.push([c0, c2]);

        // Absorb coefficients into the transcript.
        //
        // Note: We only send (c_0, c_2). The verifier derives c_1 from the sum constraint.
        challenger.observe_algebra_slice(&[c0, c2]);

        // Optional proof-of-work to increase prover cost.
        //
        // This makes it expensive for a malicious prover to "mine" favorable challenges.
        if pow_bits > 0 {
            self.pow_witnesses.push(challenger.grind(pow_bits));
        }

        // Sample the verifier's challenge for this round.
        challenger.sample_algebra_element()
    }

    /// Verifies standard sumcheck rounds and extracts folding randomness from the transcript.
    ///
    /// This method reads from the Fiat–Shamir transcript to simulate verifier interaction
    /// in the sumcheck protocol. For each round, it recovers:
    /// - One univariate polynomial (usually degree ≤ 2) sent by the prover.
    /// - One challenge scalar chosen by the verifier (folding randomness).
    ///
    /// # Returns
    ///
    /// A `MultilinearPoint` of folding randomness values.
    pub fn verify_rounds<Challenger>(
        &self,
        challenger: &mut Challenger,
        claimed_sum: &mut EF,
        pow_bits: usize,
    ) -> Result<MultilinearPoint<EF>, SumcheckError>
    where
        F: TwoAdicField,
        EF: ExtensionField<F> + TwoAdicField,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        let mut randomness = Vec::with_capacity(self.polynomial_evaluations.len());

        for (i, &[c0, c2]) in self.polynomial_evaluations.iter().enumerate() {
            // Observe only the sent polynomial evaluations (c0 and c2)
            challenger.observe_algebra_slice(&[c0, c2]);

            // Verify PoW (only if pow_bits > 0)
            if pow_bits > 0 && !challenger.check_witness(pow_bits, self.pow_witnesses[i]) {
                return Err(SumcheckError::InvalidPowWitness);
            }

            // Sample challenge
            let r: EF = challenger.sample_algebra_element();
            // Evaluate sumcheck polynomial at r
            *claimed_sum = extrapolate_012(c0, *claimed_sum - c0, c2, r);
            randomness.push(r);
        }

        Ok(MultilinearPoint::new(randomness))
    }
}

/// Verify the final sumcheck rounds.
///
/// This is a free function because the caller may not have a `SumcheckData` at all when `rounds == 0`.
///
/// # Returns
///
/// A `MultilinearPoint` of folding randomness values.
pub fn verify_final_sumcheck_rounds<F, EF, Challenger>(
    final_sumcheck: Option<&SumcheckData<F, EF>>,
    challenger: &mut Challenger,
    claimed_sum: &mut EF,
    rounds: usize,
    pow_bits: usize,
) -> Result<MultilinearPoint<EF>, SumcheckError>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    if rounds == 0 {
        return Ok(MultilinearPoint::new(Vec::new()));
    }

    let sumcheck = final_sumcheck.ok_or(SumcheckError::MissingSumcheckData {
        expected_rounds: rounds,
    })?;

    if sumcheck.polynomial_evaluations.len() != rounds {
        return Err(SumcheckError::RoundCountMismatch {
            expected: rounds,
            actual: sumcheck.polynomial_evaluations.len(),
        });
    }
    sumcheck.verify_rounds(challenger, claimed_sum, pow_bits)
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_field::{Field, PrimeCharacteristicRing, extension::BinomialExtensionField};
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_multilinear_util::evals::EvaluationsList;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::{SeedableRng, rngs::SmallRng};

    use super::*;
    use crate::{
        fiat_shamir::domain_separator::{DomainSeparator, SumcheckParams},
        parameters::{FoldingFactor, ProtocolParameters, errors::SecurityAssumption},
        sumcheck::prover::SumcheckProver,
        whir::{
            constraints::statement::initial::InitialStatement, parameters::SumcheckStrategy,
            proof::WhirProof,
        },
    };

    /// Type alias for the base field used in tests
    type F = BabyBear;

    /// Type alias for the extension field used in tests
    type EF = BinomialExtensionField<F, 4>;

    /// Type alias for the permutation used in tests
    type Perm = Poseidon2BabyBear<16>;

    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;

    /// Type alias for the challenger used in observe_and_sample tests.
    type TestChallenger = DuplexChallenger<F, Perm, 16, 8>;

    type PackedF = <F as Field>::Packing;
    type MyMmcs = MerkleTreeMmcs<PackedF, PackedF, MyHash, MyCompress, 2, DIGEST_ELEMS>;

    const DIGEST_ELEMS: usize = 8;

    /// Creates a fresh challenger for testing.
    ///
    /// The challenger is seeded deterministically so tests are reproducible.
    fn create_test_challenger() -> TestChallenger {
        let perm = Perm::new_from_rng_128(&mut rand::rngs::SmallRng::seed_from_u64(42));
        DuplexChallenger::new(perm)
    }

    #[test]
    fn test_sumcheck_data_default() {
        // Create a default SumcheckData
        let sumcheck: SumcheckData<F, EF> = SumcheckData::default();

        // Verify polynomial_evaluations is empty
        assert_eq!(sumcheck.polynomial_evaluations.len(), 0);

        // Verify pow_witnesses is empty
        assert!(sumcheck.pow_witnesses.is_empty());
    }

    #[test]
    fn test_push_pow_witness() {
        let mut sumcheck: SumcheckData<F, EF> = SumcheckData::default();

        // First push
        let witness1 = F::from_u64(42);
        sumcheck.pow_witnesses.push(witness1);

        assert_eq!(sumcheck.pow_witnesses.len(), 1);
        assert_eq!(sumcheck.pow_witnesses[0], witness1);

        // Second push should append to existing vector
        let witness2 = F::from_u64(123);
        sumcheck.pow_witnesses.push(witness2);

        assert_eq!(sumcheck.pow_witnesses.len(), 2);
        assert_eq!(sumcheck.pow_witnesses[1], witness2);
    }

    #[test]
    fn test_observe_and_sample_records_coefficients() {
        // The method should push [c0, c2] to polynomial_evaluations.
        //
        // polynomial_evaluations stores the sumcheck polynomial coefficients
        // for each round: [h(0), h(2)] where h(1) is derived by the verifier.
        let c0 = EF::from_u64(5);
        let c2 = EF::from_u64(7);

        let mut sumcheck: SumcheckData<F, EF> = SumcheckData::default();
        let mut challenger = create_test_challenger();

        // polynomial_evaluations should be empty initially
        assert!(sumcheck.polynomial_evaluations.is_empty());

        // Call observe_and_sample with pow_bits = 0 (no grinding)
        let pow_bits = 0;
        let _r = sumcheck.observe_and_sample::<_, F>(&mut challenger, c0, c2, pow_bits);

        // polynomial_evaluations should now have one entry: [c0, c2]
        assert_eq!(sumcheck.polynomial_evaluations.len(), 1);
        assert_eq!(sumcheck.polynomial_evaluations[0][0], c0);
        assert_eq!(sumcheck.polynomial_evaluations[0][1], c2);
    }

    #[test]
    fn test_observe_and_sample_multiple_rounds() {
        // Multiple calls should accumulate coefficients in order.
        //
        // Round 0: push [c0_0, c2_0]
        // Round 1: push [c0_1, c2_1]
        // Round 2: push [c0_2, c2_2]
        let c0_0 = EF::from_u64(1);
        let c2_0 = EF::from_u64(2);
        let c0_1 = EF::from_u64(3);
        let c2_1 = EF::from_u64(4);
        let c0_2 = EF::from_u64(5);
        let c2_2 = EF::from_u64(6);

        let mut sumcheck: SumcheckData<F, EF> = SumcheckData::default();
        let mut challenger = create_test_challenger();
        let pow_bits = 0;

        // Round 0
        let _r0 = sumcheck.observe_and_sample::<_, F>(&mut challenger, c0_0, c2_0, pow_bits);
        assert_eq!(sumcheck.polynomial_evaluations.len(), 1);

        // Round 1
        let _r1 = sumcheck.observe_and_sample::<_, F>(&mut challenger, c0_1, c2_1, pow_bits);
        assert_eq!(sumcheck.polynomial_evaluations.len(), 2);

        // Round 2
        let _r2 = sumcheck.observe_and_sample::<_, F>(&mut challenger, c0_2, c2_2, pow_bits);
        assert_eq!(sumcheck.polynomial_evaluations.len(), 3);

        // Verify all stored coefficients match input order
        assert_eq!(sumcheck.polynomial_evaluations[0], [c0_0, c2_0]);
        assert_eq!(sumcheck.polynomial_evaluations[1], [c0_1, c2_1]);
        assert_eq!(sumcheck.polynomial_evaluations[2], [c0_2, c2_2]);
    }

    #[test]
    fn test_observe_and_sample_without_pow() {
        // When pow_bits = 0, no PoW witness should be recorded.
        //
        // The method skips the grinding step when pow_bits is zero,
        // so pow_witnesses should remain empty.
        let c0 = EF::from_u64(10);
        let c2 = EF::from_u64(20);

        let mut sumcheck: SumcheckData<F, EF> = SumcheckData::default();
        let mut challenger = create_test_challenger();

        // pow_witnesses should be empty initially
        assert!(sumcheck.pow_witnesses.is_empty());

        // Call with pow_bits = 0
        let pow_bits = 0;
        let _r = sumcheck.observe_and_sample::<_, F>(&mut challenger, c0, c2, pow_bits);

        // pow_witnesses should still be empty (no grinding performed)
        assert!(sumcheck.pow_witnesses.is_empty());
    }

    #[test]
    fn test_observe_and_sample_with_pow() {
        // When pow_bits > 0, a PoW witness should be recorded.
        //
        // The method calls challenger.grind(pow_bits) and pushes
        // the resulting witness to pow_witnesses.
        let c0 = EF::from_u64(10);
        let c2 = EF::from_u64(20);

        let mut sumcheck: SumcheckData<F, EF> = SumcheckData::default();
        let mut challenger = create_test_challenger();

        // pow_witnesses should be empty initially
        assert!(sumcheck.pow_witnesses.is_empty());

        // Call with pow_bits = 1 (minimal PoW)
        let pow_bits = 1;
        let _r = sumcheck.observe_and_sample::<_, F>(&mut challenger, c0, c2, pow_bits);

        // pow_witnesses should now have one entry
        assert_eq!(sumcheck.pow_witnesses.len(), 1);
    }

    #[test]
    fn test_observe_and_sample_pow_accumulates() {
        // Multiple rounds with PoW should accumulate witnesses.
        //
        // Each call with pow_bits > 0 should add one witness.
        let c0 = EF::from_u64(1);
        let c2 = EF::from_u64(2);

        let mut sumcheck: SumcheckData<F, EF> = SumcheckData::default();
        let mut challenger = create_test_challenger();
        let pow_bits = 1;

        // Three rounds with PoW
        let _r0 = sumcheck.observe_and_sample::<_, F>(&mut challenger, c0, c2, pow_bits);
        let _r1 = sumcheck.observe_and_sample::<_, F>(&mut challenger, c0, c2, pow_bits);
        let _r2 = sumcheck.observe_and_sample::<_, F>(&mut challenger, c0, c2, pow_bits);

        // Should have 3 witnesses
        assert_eq!(sumcheck.pow_witnesses.len(), 3);
        // And 3 polynomial evaluations
        assert_eq!(sumcheck.polynomial_evaluations.len(), 3);
    }

    #[test]
    fn test_observe_and_sample_deterministic_challenge() {
        // Fiat-Shamir property: same inputs produce same challenge.
        //
        // Two challengers with the same initial state, observing the same
        // coefficients, should sample the same challenge.
        let c0 = EF::from_u64(42);
        let c2 = EF::from_u64(99);
        let pow_bits = 0;

        // First run
        let mut sumcheck1: SumcheckData<F, EF> = SumcheckData::default();
        let mut challenger1 = create_test_challenger();
        let r1 = sumcheck1.observe_and_sample::<_, F>(&mut challenger1, c0, c2, pow_bits);

        // Second run with fresh but identically-seeded challenger
        let mut sumcheck2: SumcheckData<F, EF> = SumcheckData::default();
        let mut challenger2 = create_test_challenger();
        let r2 = sumcheck2.observe_and_sample::<_, F>(&mut challenger2, c0, c2, pow_bits);

        // Challenges should be identical
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_observe_and_sample_challenge_depends_on_history() {
        // The challenge at round i depends on all previous observations.
        //
        // Two sequences with different history should produce different
        // challenges even if the final round has the same coefficients.
        let c0 = EF::from_u64(100);
        let c2 = EF::from_u64(200);
        let pow_bits = 0;

        // Sequence A: observe once, then observe (c0, c2)
        let mut sumcheck_a: SumcheckData<F, EF> = SumcheckData::default();
        let mut challenger_a = create_test_challenger();
        let _r0_a =
            sumcheck_a.observe_and_sample::<_, F>(&mut challenger_a, EF::ONE, EF::TWO, pow_bits);
        let r1_a = sumcheck_a.observe_and_sample::<_, F>(&mut challenger_a, c0, c2, pow_bits);

        // Sequence B: directly observe (c0, c2) without prior round
        let mut sumcheck_b: SumcheckData<F, EF> = SumcheckData::default();
        let mut challenger_b = create_test_challenger();
        let r_b = sumcheck_b.observe_and_sample::<_, F>(&mut challenger_b, c0, c2, pow_bits);

        // Challenges should differ due to different transcript history
        assert_ne!(r1_a, r_b);
    }

    #[test]
    fn test_observe_and_sample_returns_extension_field_element() {
        // The returned challenge should be a valid extension field element.
        //
        // This is verified implicitly by the type system, but we can also
        // check that it's not trivially zero (with high probability).
        let c0 = EF::from_u64(7);
        let c2 = EF::from_u64(11);
        let pow_bits = 0;

        let mut sumcheck: SumcheckData<F, EF> = SumcheckData::default();
        let mut challenger = create_test_challenger();

        let r: EF = sumcheck.observe_and_sample::<_, F>(&mut challenger, c0, c2, pow_bits);

        // The challenge should (with overwhelming probability) be non-zero
        assert_ne!(r, EF::ZERO);
    }

    /// Constructs a default WHIR configuration for testing
    fn create_proof_from_test_protocol_params(
        num_variables: usize,
        folding_factor: FoldingFactor,
    ) -> WhirProof<F, EF, MyMmcs> {
        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);

        let merkle_hash = MyHash::new(perm.clone());
        let merkle_compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(merkle_hash, merkle_compress, 0);

        let whir_params = ProtocolParameters {
            security_level: 32,
            pow_bits: 0,
            rs_domain_initial_reduction_factor: 1,
            folding_factor,
            mmcs,
            soundness_type: SecurityAssumption::UniqueDecoding,
            starting_log_inv_rate: 1,
        };

        WhirProof::from_protocol_parameters(&whir_params, num_variables)
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_verify_rounds() {
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
        let folding_factor = 3;
        let pow_bits = 0;

        // Create a constraint system with evaluations of f at various points
        let mut statement =
            InitialStatement::new(evals, folding_factor, SumcheckStrategy::default());

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

        assert_eq!(f_000, statement.evaluate(&x_000));
        assert_eq!(f_100, statement.evaluate(&x_100));
        assert_eq!(f_110, statement.evaluate(&x_110));
        assert_eq!(f_111, statement.evaluate(&x_111));
        assert_eq!(f_011, statement.evaluate(&x_011));

        // Set up domain separator
        let mut domsep: DomainSeparator<EF, F> = DomainSeparator::new(vec![]);
        domsep.add_sumcheck(&SumcheckParams {
            rounds: folding_factor,
            pow_bits,
        });

        let mut rng = SmallRng::seed_from_u64(1);
        let challenger = TestChallenger::new(Perm::new_from_rng_128(&mut rng));
        let mut prover_challenger = challenger.clone();

        // Initialize proof and challenger
        let mut proof =
            create_proof_from_test_protocol_params(n_vars, FoldingFactor::Constant(folding_factor));
        domsep.observe_domain_separator(&mut prover_challenger);

        // Instantiate the prover with base field coefficients
        let (_, _) = SumcheckProver::<F, EF>::from_base_evals(
            &mut proof.initial_sumcheck,
            &mut prover_challenger,
            folding_factor,
            pow_bits,
            &statement,
        );

        // Reconstruct verifier state to simulate the rounds
        let mut verifier_challenger = challenger;
        domsep.observe_domain_separator(&mut verifier_challenger);

        // Save a fresh copy for verify_rounds
        let mut verifier_challenger_for_verify = verifier_challenger.clone();

        let mut t = EvaluationsList::zero(statement.num_variables());
        let mut expected_initial_sum = EF::ZERO;
        statement.normalize().combine_hypercube::<F, false>(
            &mut t,
            &mut expected_initial_sum,
            EF::ONE,
        );

        // Start with the claimed sum before folding
        let mut current_sum = expected_initial_sum;

        let mut expected = Vec::with_capacity(folding_factor);

        // First round: read c_0 = h(0) and c_2 (quadratic coefficient)
        let [c_0, c_2] = proof.initial_sumcheck.polynomial_evaluations[0];
        let h_1 = current_sum - c_0;

        // Observe polynomial evaluations (must match what verify_rounds does)
        verifier_challenger.observe_algebra_slice(&[c_0, c_2]);

        // Sample random challenge r_i ∈ EF4 and evaluate h_i(r_i)
        let r: EF = verifier_challenger.sample_algebra_element();
        // h(r) = c_2 * r^2 + (h(1) - c_0 - c_2) * r + c_0
        current_sum = c_2 * r.square() + (h_1 - c_0 - c_2) * r + c_0;
        expected.push(r);

        for i in 0..folding_factor - 1 {
            // Read c_0 = h(0) and c_2 (quadratic coefficient), derive h(1) = claimed_sum - c_0
            let [c_0, c_2] = proof.initial_sumcheck.polynomial_evaluations[i + 1];
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

        let randomness = proof
            .initial_sumcheck
            .verify_rounds(
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
