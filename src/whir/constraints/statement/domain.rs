use alloc::vec::Vec;

use itertools::Itertools;
use p3_dft::{Radix2DFTSmallBatch, TwoAdicSubgroupDft};
use p3_field::{
    ExtensionField, Field, PackedFieldExtension, PackedValue, TwoAdicField, dot_product,
};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::poly::evals::EvaluationsList;

/// A batched system of `select`-based evaluation constraints for multilinear polynomials.
///
/// This struct represents a collection of evaluation constraints of the form `p(z_i) = s_i`
/// for a multilinear polynomial `p` over the Boolean hypercube `{0,1}^k`.
///
/// # The Select Function
///
/// For vectors `X, Y ∈ F^k`, the select function is defined as:
///
/// ```text
/// select(X, Y) = ∏_i (X_i · Y_i + (1 - Y_i))
/// ```
///
/// **Key Property:** When `Y ∈ {0,1}^k` is a Boolean vector and `X = pow(z)`:
///
/// ```text
/// select(pow(z), b) = z^{int(b)}
/// ```
///
/// where `pow(z) = (z, z^2, z^4, ..., z^{2^{k-1}})` and `int(b)` interprets the Boolean
/// vector `b` as an integer in binary.
///
/// **Derivation:**
/// ```text
/// select(pow(z), b) = ∏_i (z^{2^i} · b_i + (1 - b_i))
///                   = ∏_{i: b_i=1} (z^{2^i})     [since b_i ∈ {0,1}]
///                   = z^{Σ_{i: b_i=1} 2^i}
///                   = z^{int(b)}
/// ```
///
/// # Verification Claims
///
/// Each constraint `(z_i, s_i)` in this statement asserts:
///
/// ```text
/// Σ_{b ∈ {0,1}^k} P(b) · select(pow(z_i), b) = s_i
/// ```
///
/// where `P(b)` are the evaluations of the polynomial over the Boolean hypercube.
///
/// # Batching
///
/// Multiple constraints are batched using random challenge `γ` to produce:
///
/// - **Weight polynomial**: `W(b) = Σ_i γ^i · select(pow(z_i), b)`
/// - **Target sum**: `S = Σ_i γ^i · s_i`
///
/// This reduces `n` separate verification claims to a single sumcheck:
///
/// ```text
/// Σ_{b ∈ {0,1}^k} P(b) · W(b) = S
/// ```
#[derive(Clone, Debug)]
pub struct DomainStatement<EF> {
    /// Number of variables `k` defining the Boolean hypercube `{0,1}^k`.
    ///
    /// This determines the dimension of the multilinear polynomial space and the size
    /// of the evaluation domain (2^k points).
    num_variables: usize,

    /// Domain size is the range of stir indicies.
    k_domain: usize,

    /// Evaluation points `[z_1, z_2, ..., z_n]` where each constraint checks `p(z_i) = s_i`.
    ///
    /// Each `z_i ∈ F` is a base field element. The `pow` map will expand it to
    /// `pow(z_i) = (z_i, z_i^2, z_i^4, ..., z_i^{2^{k-1}})` for the select function.
    indicies: Vec<usize>,

    /// Expected evaluation values `[s_1, s_2, ..., s_n]` corresponding to each constraint.
    ///
    /// Each `s_i ∈ EF` is an extension field element representing the claimed evaluation
    /// of the polynomial at point `z_i`.
    evaluations: Vec<EF>,
}

impl<EF: Field> DomainStatement<EF> {
    /// Creates an empty select statement for polynomials over `{0,1}^k`.
    ///
    /// # Parameters
    ///
    /// - `num_variables`: The dimension `k` of the Boolean hypercube
    ///
    /// # Returns
    ///
    /// An initialized statement with no constraints, ready to accept constraints.
    #[must_use]
    pub fn initialize(num_variables: usize, k_domain: usize) -> Self {
        assert!(num_variables <= k_domain);
        Self {
            k_domain,
            num_variables,
            indicies: Vec::new(),
            evaluations: Vec::new(),
        }
    }

    /// Creates a select statement pre-populated with constraints.
    ///
    /// # Parameters
    ///
    /// - `num_variables`: The dimension `k` of the Boolean hypercube
    /// - `vars`: Evaluation points `[z_1, ..., z_n]`
    /// - `evaluations`: Expected values `[s_1, ..., s_n]`
    ///
    /// # Panics
    ///
    /// Panics if the number of variables and evaluations do not match.
    /// Panics if an index is larger than domain.
    #[must_use]
    pub fn new(
        num_variables: usize,
        k_domain: usize,
        indicies: &[usize],
        evaluations: &[EF],
    ) -> Self {
        assert!(num_variables <= k_domain);
        assert!(indicies.len() == evaluations.len());
        assert!(indicies.iter().all(|&index| index < (1 << k_domain)));
        // Remove duplicates
        let (indicies, evaluations): (Vec<_>, _) = indicies
            .iter()
            .copied()
            .zip(evaluations.iter().copied())
            .sorted_by(|(i0, _), (i1, _)| i0.cmp(i1))
            .dedup_by(|(i0, _), (i1, _)| i0 == i1)
            .unzip();

        Self {
            num_variables,
            k_domain,
            indicies,
            evaluations,
        }
    }

    /// Returns the number of variables `k` defining the polynomial space dimension.
    ///
    /// This is the dimension of the Boolean hypercube `{0,1}^k` over which polynomials
    /// are defined, containing `2^k` evaluation points.
    #[must_use]
    pub const fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// Returns `true` if no constraints have been added to this statement.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        debug_assert!(self.indicies.is_empty() == self.evaluations.is_empty());
        self.indicies.is_empty()
    }

    /// Returns an iterator over constraint pairs `(z_i, s_i)`.
    ///
    /// Each pair represents one evaluation constraint: `p(z_i) = s_i`.
    pub fn iter<F: TwoAdicField>(&self) -> impl Iterator<Item = (F, &EF)>
    where
        EF: ExtensionField<F>,
    {
        if self.is_empty() {
            itertools::Either::Left(core::iter::empty())
        } else {
            let omega = F::two_adic_generator(self.k_domain);
            itertools::Either::Right(
                self.indicies
                    .iter()
                    .map(move |&index| omega.exp_u64(index as u64))
                    .zip(self.evaluations.iter()),
            )
        }
    }

    /// Returns the number of evaluation constraints `n` in this statement.
    #[must_use]
    pub const fn len(&self) -> usize {
        debug_assert!(self.indicies.len() == self.evaluations.len());
        self.indicies.len()
    }

    /// Verifies that a given polynomial satisfies all constraints in the statement.
    ///
    /// For each constraint `(z_i, s_i)`, this method interprets the evaluation table as
    /// coefficients of a univariate polynomial, evaluates it at `z_i` using Horner's method,
    /// and checks if the result equals the expected value `s_i`.
    ///
    /// For a polynomial represented by evaluations `[c_0, c_1, ..., c_{2^k-1}]`:
    ///
    /// ```text
    /// p(z) = c_0 + z(c_1 + z(c_2 + z(...)))
    /// ```
    ///
    /// This is computed right-to-left as:
    /// ```text
    /// acc = 0
    /// for i = 2^k-1 down to 0:
    ///     acc = acc * z + c_i
    /// ```
    ///
    /// # Parameters
    ///
    /// - `poly`: Evaluation table treated as univariate polynomial coefficients
    ///
    /// # Returns
    ///
    /// `true` if all constraints are satisfied, `false` otherwise.
    #[must_use]
    pub fn verify<F: TwoAdicField>(&self, poly: &EvaluationsList<EF>) -> bool
    where
        EF: ExtensionField<F>,
    {
        self.iter::<F>().all(|(var, &expected_eval)| {
            // Evaluate the polynomial at `var` using Horner's method.
            // This computes: p(var) = c_0 + var(c_1 + var(c_2 + ...))
            poly.iter()
                .rfold(EF::ZERO, |result, coeff| result * var + *coeff)
                == expected_eval
        })
    }

    /// Adds a single evaluation constraint `p(z) = s` to the statement.
    ///
    /// # Parameters
    ///
    /// - `var`: Evaluation point `z ∈ F`
    /// - `eval`: Expected evaluation value `s ∈ EF`
    pub fn add_constraint(&mut self, index: usize, eval: EF) {
        assert!(index < (1 << self.k_domain));
        if !self.indicies.contains(&index) {
            self.evaluations.push(eval);
            self.indicies.push(index);
        }
    }

    /// Batches all constraints into a single weighted polynomial and target sum for sumcheck.
    ///
    /// Given constraints `p(z_1) = s_1, ..., p(z_n) = s_n`, this method transforms them into
    /// a single sumcheck claim using random challenge `γ`:
    ///
    /// ```text
    /// Σ_{b ∈ {0,1}^k} P(b) · W(b) = S
    /// ```
    ///
    /// where:
    /// - **Weight polynomial**: `W(b) = Σ_i γ^{i+shift} · select(pow(z_i), b)`
    /// - **Target sum**: `S = Σ_i γ^{i+shift} · s_i`
    ///
    /// The method computes `W(b)` for all `b ∈ {0,1}^k` and `S`, adding them to the
    /// provided accumulators.
    ///
    /// # Parameters
    ///
    /// - `acc_weights`: Accumulator for the weight polynomial `W(b)`. Must have `2^k` entries.
    ///   This method **adds** the batched weights to existing values.
    ///
    /// - `acc_sum`: Accumulator for the target sum `S`. This method **adds** the batched
    ///   evaluations to the existing value.
    ///
    /// - `challenge`: Random challenge `γ ∈ EF` used for batching.
    ///
    /// - `shift`: Power offset for challenge. Constraint `i` uses weight `γ^{i+shift}`.
    ///   Allows multiple statement types to use non-overlapping challenge powers.
    #[instrument(skip_all, fields(num_constraints = self.len(), num_variables = self.num_variables()))]
    pub fn combine<F: TwoAdicField>(
        &self,
        weights: &mut EvaluationsList<EF>,
        eval: &mut EF,
        alpha: EF,
        shift: usize,
    ) where
        EF: ExtensionField<F>,
    {
        // Early return for empty statement:
        //
        // No constraints means no contribution to the batched claim.
        if self.indicies.is_empty() {
            return;
        }

        // Combine expected evaluations: S = ∑_i γ^i * s_i
        self.combine_evals(eval, alpha, shift);

        let dft = Radix2DFTSmallBatch::<F>::default();
        let mut pows_combined = EF::zero_vec(1 << self.k_domain);
        self.indicies
            .iter()
            .zip(alpha.powers().skip(shift))
            .for_each(|(&index, challenge)| pows_combined[index] = challenge);
        let pows_combined = dft.dft_algebra(pows_combined);

        weights
            .0
            .par_iter_mut()
            .zip(pows_combined.par_iter())
            .for_each(|(out, &val)| *out += val);
    }

    #[instrument(skip_all, fields(num_constraints = self.len(), num_variables = self.num_variables()))]
    pub fn combine_packed<F: TwoAdicField>(
        &self,
        weights: &mut EvaluationsList<EF::ExtensionPacking>,
        eval: &mut EF,
        challenge: EF,
        shift: usize,
    ) where
        EF: ExtensionField<F>,
    {
        // Early return for empty statement:
        //
        // No constraints means no contribution to the batched claim.
        if self.indicies.is_empty() {
            return;
        }

        let k = self.num_variables();
        let k_pack = log2_strict_usize(F::Packing::WIDTH);
        assert!(k >= k_pack);
        assert_eq!(weights.num_variables() + k_pack, k);

        // Combine expected evaluations: S = ∑_i γ^i * s_i
        self.combine_evals(eval, challenge, shift);

        let dft = Radix2DFTSmallBatch::<F>::default();
        let mut pows_combined = EF::zero_vec(1 << self.k_domain);
        self.indicies
            .iter()
            .zip(challenge.powers().skip(shift))
            .for_each(|(&index, challenge)| pows_combined[index] = challenge);
        let pows_combined = dft.dft_algebra(pows_combined);

        weights
            .0
            .par_iter_mut()
            .zip(pows_combined.par_chunks(F::Packing::WIDTH))
            .for_each(|(out, chunk)| *out += EF::ExtensionPacking::from_ext_slice(chunk));
    }

    /// Batches expected evaluation values into a single target sum using challenge powers.
    ///
    /// Computes and adds to `claimed_eval`:
    ///
    /// ```text
    /// S = Σ_i γ^{i+shift} · s_i
    /// ```
    ///
    /// where `s_i` are the expected evaluation values in `self.evaluations`.
    ///
    /// # Parameters
    ///
    /// - `claimed_eval`: Accumulator for the target sum. This method **adds** the batched
    ///   evaluations to the existing value.
    ///
    /// - `challenge`: Random challenge `γ ∈ EF` used for batching.
    ///
    /// - `shift`: Power offset. Constraint `i` uses weight `γ^{i+shift}`.
    pub fn combine_evals(&self, claimed_eval: &mut EF, challenge: EF, shift: usize) {
        // Compute: Σ_i γ^{i+shift} · s_i
        // This is equivalent to dot_product(evaluations, [γ^shift, γ^{shift+1}, ...])
        *claimed_eval += dot_product::<EF, _, _>(
            self.evaluations.iter().copied(),
            challenge.powers().skip(shift).take(self.len()),
        );
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::{
        PackedFieldExtension, PrimeCharacteristicRing, extension::BinomialExtensionField,
    };
    use proptest::prelude::*;
    use rand::{SeedableRng, rngs::SmallRng};

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    #[test]
    fn test_domain_statement_initialize() {
        // Test that initialize creates an empty statement with correct num_variables.
        let statement = DomainStatement::<F>::initialize(3, 3);

        // The statement should have 3 variables.
        assert_eq!(statement.num_variables(), 3);
        // The statement should be empty (no constraints).
        assert!(statement.is_empty());
        // The length should be 0.
        assert_eq!(statement.len(), 0);
    }

    #[test]
    fn test_domain_statement_new() {
        // Test that new creates a statement with pre-populated constraints.
        let indicies = vec![0, 1];
        let evaluations = vec![F::from_u64(10), F::from_u64(20)];

        let statement = DomainStatement::new(2, 2, &indicies, &evaluations);

        // The statement should have 2 variables.
        assert_eq!(statement.num_variables(), 2);
        // The statement should not be empty.
        assert!(!statement.is_empty());
        // The statement should have 2 constraints.
        assert_eq!(statement.len(), 2);
        // The vars and evaluations should match.
        assert_eq!(statement.indicies, indicies);
        assert_eq!(statement.evaluations, evaluations);
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn test_domain_statement_new_mismatched_lengths() {
        // Test that new panics when vars.len() != evaluations.len().
        let indicies = vec![5];
        let evaluations = vec![F::from_u64(10), F::from_u64(20)];

        // This should panic due to length mismatch.
        let _ = DomainStatement::new(2, 2, &indicies, &evaluations);
    }

    #[test]
    fn test_domain_statement_add_constraint() {
        // Test adding constraints one at a time.
        let num_variables = 10;
        let omega = F::two_adic_generator(num_variables);
        let mut statement = DomainStatement::<F>::initialize(num_variables, num_variables);

        // Initially empty.
        assert!(statement.is_empty());
        assert_eq!(statement.len(), 0);

        // Add first constraint: p(5) = 10.
        statement.add_constraint(5, F::from_u64(10));
        assert!(!statement.is_empty());
        assert_eq!(statement.len(), 1);

        // Add second constraint: p(7) = 20.
        statement.add_constraint(7, F::from_u64(20));
        assert_eq!(statement.len(), 2);

        // Verify the constraints were added correctly.
        let constraints: Vec<_> = statement.iter().collect();
        assert_eq!(constraints.len(), 2);
        assert_eq!(constraints[0].0, omega.exp_u64(5));
        assert_eq!(*constraints[0].1, F::from_u64(10));
        assert_eq!(constraints[1].0, omega.exp_u64(7));
        assert_eq!(*constraints[1].1, F::from_u64(20));
    }

    #[test]
    fn test_domain_statement_verify_basic() {
        // Test the verify method with a simple polynomial.
        //
        // Create a polynomial with evaluations [c0, c1, c2, c3] over {0,1}^2.
        let c0 = F::from_u64(1);
        let c1 = F::from_u64(2);
        let c2 = F::from_u64(3);
        let c3 = F::from_u64(4);
        let poly = EvaluationsList::new(vec![c0, c1, c2, c3]);

        // Create a statement with k=2 variables.
        let k = 2;
        let omega = F::two_adic_generator(k);
        let mut statement = DomainStatement::<F>::initialize(k, k);

        // The polynomial evaluations [c0, c1, c2, c3] can be interpreted as a univariate polynomial:
        // p(z) = c0 + c1*z + c2*z^2 + c3*z^3
        //
        // Test p(0) = c0 = 1.
        let eval0 = c0 + c1 + c2 + c3;
        statement.add_constraint(0, eval0);
        assert!(statement.verify(&poly));

        // Test p(1) = c0 + c1 + c2 + c3
        let mut statement2 = DomainStatement::<F>::initialize(k, k);
        let z1 = omega;
        let eval1 = c0 + c1 * z1 + c2 * z1 * z1 + c3 * z1 * z1 * z1;
        statement2.add_constraint(1, eval1);
        assert!(statement2.verify(&poly));

        // Test p(2) = c0 + c1*2 + c2*4 + c3*8
        let mut statement3 = DomainStatement::<F>::initialize(k, k);
        let z2 = omega * omega;
        let eval2 = c0 + c1 * z2 + c2 * z2 * z2 + c3 * z2 * z2 * z2;
        statement3.add_constraint(2, eval2);
        assert!(statement3.verify(&poly));

        // Test a failing verification: p(1) = wrong_eval
        let mut statement4 = DomainStatement::<F>::initialize(k, k);
        let wrong_eval = F::from_u64(56765);
        statement4.add_constraint(3, wrong_eval);
        assert!(!statement4.verify(&poly));
    }

    #[test]
    fn test_domain_statement_combine_single_constraint() {
        // Test combining a single constraint.
        //
        // For k=2 variables, we have a 2^2 = 4-point domain.
        let k = 2;
        let domain_size = 1 << k;
        let omega = F::two_adic_generator(k);

        // Create a statement with one constraint: p(z) = s.
        let mut statement = DomainStatement::<F>::initialize(k, k);
        let i = 2;
        let z = omega.exp_u64(i as u64);
        let s = F::from_u64(100);
        statement.add_constraint(i, s);

        // The challenge γ is unused for a single constraint (it would multiply by γ^0 = 1).
        let gamma = F::from_u64(2);
        let shift = 0;

        // Initialize accumulators.
        let mut acc_weights = EvaluationsList::zero(k);
        let mut acc_sum = F::ZERO;

        // Combine the constraints.
        statement.combine(&mut acc_weights, &mut acc_sum, gamma, shift);

        // The target sum should be S = γ^0 · s = 1 · s = s.
        let expected_sum = s;
        assert_eq!(acc_sum, expected_sum);

        // The weight polynomial should be W(b) = select(pow(z), b) for all b ∈ {0,1}^k.
        //
        // Verify each entry manually using the property: select(pow(z), b) = z^b.
        for b in 0..domain_size {
            let expected_weight = z.exp_u64(b as u64);
            assert_eq!(
                acc_weights.as_slice()[b],
                expected_weight,
                "Weight mismatch at index {b}"
            );
        }
    }

    #[test]
    fn test_domain_statement_combine_multiple_constraints() {
        // Test combining multiple constraints with batching.
        //
        // For k=2 variables, we have a 2^2 = 4-point domain.
        let k = 2;
        let domain_size = 1 << k;
        let omega = F::two_adic_generator(k);

        // Create a statement with two constraints:
        // - Constraint 0: p(z0) = s0
        // - Constraint 1: p(z1) = s1
        let mut statement = DomainStatement::<F>::initialize(k, k);
        let i0 = 1;
        let z0 = omega.exp_u64(i0 as u64);
        let s0 = F::from_u64(10);
        let i1 = 2;
        let z1 = omega.exp_u64(i1 as u64);
        let s1 = F::from_u64(20);
        statement.add_constraint(i0, s0);
        statement.add_constraint(i1, s1);

        // Use challenge γ for batching.
        let gamma = F::from_u64(2);
        let shift = 0;

        // Initialize accumulators.
        let mut acc_weights = EvaluationsList::zero(k);
        let mut acc_sum = F::ZERO;

        // Combine the constraints.
        statement.combine(&mut acc_weights, &mut acc_sum, gamma, shift);

        // The target sum should be:
        // S = γ^0 · s0 + γ^1 · s1 = 1·s0 + γ·s1 = s0 + gamma*s1.
        let expected_sum = s0 + gamma * s1;
        assert_eq!(acc_sum, expected_sum);

        // The weight polynomial should be:
        // W(b) = γ^0 · select(pow(z0), b) + γ^1 · select(pow(z1), b)
        //      = select(pow(z0), b) + gamma · select(pow(z1), b)
        // Using the property: select(pow(z), b) = z^b.
        for b in 0..domain_size {
            let weight0 = z0.exp_u64(b as u64);
            let weight1 = z1.exp_u64(b as u64);
            let expected_weight = weight0 + gamma * weight1;
            assert_eq!(
                acc_weights.as_slice()[b],
                expected_weight,
                "Weight mismatch at index {b}"
            );
        }
    }

    #[test]
    fn test_domain_statement_combine_with_shift() {
        // Test combining constraints with a non-zero shift parameter.
        //
        // The shift parameter allows multiple statement types to use non-overlapping
        // challenge powers for batching.
        let k = 1;
        let domain_size = 1 << k;
        let omega = F::two_adic_generator(k);

        // Create a statement with one constraint: p(z) = s.
        let mut statement = DomainStatement::<F>::initialize(k, k);
        let i = 1;
        let z = omega.exp_u64(i as u64);
        let s = F::from_u64(100);
        statement.add_constraint(i, s);

        // Use challenge γ with shift.
        // This means the constraint will be weighted by γ^{0+shift} = γ^shift.
        let gamma = F::from_u64(2);
        let shift = 3;

        // Initialize accumulators.
        let mut acc_weights = EvaluationsList::zero(k);
        let mut acc_sum = F::ZERO;

        // Combine the constraints.
        statement.combine(&mut acc_weights, &mut acc_sum, gamma, shift);

        // The target sum should be S = γ^shift · s.
        let gamma_to_shift = gamma.exp_u64(shift as u64);
        let expected_sum = gamma_to_shift * s;
        assert_eq!(acc_sum, expected_sum);

        // The weight polynomial should be W(b) = γ^shift · select(pow(z), b).
        // Using the property: select(pow(z), b) = z^b.
        for b in 0..domain_size {
            let select_val = z.exp_u64(b as u64);
            let expected_weight = gamma_to_shift * select_val;
            assert_eq!(
                acc_weights.as_slice()[b],
                expected_weight,
                "Weight mismatch at index {b}"
            );
        }
    }

    #[test]
    fn test_domain_statement_combine_empty() {
        // Test that combining an empty statement does nothing.
        let k = 2;
        let statement = DomainStatement::<F>::initialize(k, k);

        // Initialize accumulators with non-zero values.
        let w0 = F::from_u64(1);
        let w1 = F::from_u64(2);
        let w2 = F::from_u64(3);
        let w3 = F::from_u64(4);
        let mut acc_weights = EvaluationsList::new(vec![w0, w1, w2, w3]);
        let initial_sum = F::from_u64(99);
        let mut acc_sum = initial_sum;

        // Store original values.
        let original_weights = acc_weights.clone();
        let original_sum = acc_sum;

        // Combine the empty statement.
        let gamma = F::from_u64(2);
        let shift = 0;
        statement.combine(&mut acc_weights, &mut acc_sum, gamma, shift);

        // The accumulators should remain unchanged.
        assert_eq!(acc_weights, original_weights);
        assert_eq!(acc_sum, original_sum);
    }

    #[test]
    fn test_domain_statement_combine_accumulation() {
        // Test that combine properly accumulates (adds to) existing values.
        //
        // This is important for batching multiple statements together.
        let k = 1;
        let domain_size = 1 << k;
        let omega = F::two_adic_generator(k);

        // Create first statement with constraint p(z1) = s1.
        let mut statement1 = DomainStatement::<F>::initialize(k, k);
        let i1 = 0;
        let s1 = F::from_u64(5);
        statement1.add_constraint(i1, s1);

        // Create second statement with constraint p(z2) = s2.
        let mut statement2 = DomainStatement::<F>::initialize(k, k);
        let i2 = 1;
        let z2 = omega.exp_u64(i2 as u64);
        let s2 = F::from_u64(7);
        statement2.add_constraint(i2, s2);

        let gamma = F::from_u64(2);
        let shift = 0;

        // Initialize accumulators.
        let mut acc_weights = EvaluationsList::zero(k);
        let mut acc_sum = F::ZERO;

        // Combine first statement.
        statement1.combine(&mut acc_weights, &mut acc_sum, gamma, shift);

        // Store intermediate values.
        let intermediate_weights = acc_weights.clone();
        let intermediate_sum = acc_sum;

        // Combine second statement (should add to existing values).
        statement2.combine(&mut acc_weights, &mut acc_sum, gamma, shift);

        // The accumulated sum should be intermediate_sum + s2.
        let expected_sum = intermediate_sum + s2;
        assert_eq!(acc_sum, expected_sum);

        // The accumulated weights should be the sum of both select functions.
        // Using the property: select(pow(z), b) = z^b.
        for b in 0..domain_size {
            let weight2 = z2.exp_u64(b as u64);
            let expected_weight = intermediate_weights.as_slice()[b] + weight2;
            assert_eq!(
                acc_weights.as_slice()[b],
                expected_weight,
                "Accumulated weight mismatch at index {b}"
            );
        }
    }

    #[test]
    fn test_domain_statement_combine_evals() {
        // Test the combine_evals method.
        let k = 2;

        // Create a statement with two constraints.
        let mut statement = DomainStatement::<F>::initialize(k, k);
        let s0 = F::from_u64(10);
        let s1 = F::from_u64(20);
        statement.add_constraint(0, s0);
        statement.add_constraint(1, s1);

        let gamma = F::from_u64(2);
        let shift = 1;

        // Test combine_evals.
        let mut claimed_eval = F::ZERO;
        statement.combine_evals(&mut claimed_eval, gamma, shift);

        // Expected: S = γ^{shift} · s0 + γ^{shift+1} · s1 = γ^1·s0 + γ^2·s1.
        let gamma_1 = gamma.exp_u64(shift as u64);
        let gamma_2 = gamma.exp_u64((shift + 1) as u64);
        let expected = gamma_1 * s0 + gamma_2 * s1;
        assert_eq!(claimed_eval, expected);
    }

    #[test]
    fn test_domain_statement_combine_evals_accumulation() {
        // Test that combine_evals properly accumulates.
        let k = 1;

        let mut statement = DomainStatement::<F>::initialize(k, k);
        let s = F::from_u64(10);
        statement.add_constraint(0, s);

        let gamma = F::from_u64(3);
        let shift = 0;

        // Start with a non-zero claimed_eval.
        let initial_eval = F::from_u64(42);
        let mut claimed_eval = initial_eval;

        // Combine evals should add to the existing value.
        statement.combine_evals(&mut claimed_eval, gamma, shift);

        // Expected: initial_eval + γ^0 · s = initial_eval + 1·s = initial_eval + s.
        let expected = initial_eval + s;
        assert_eq!(claimed_eval, expected);
    }

    #[test]
    fn test_domain_combine_consistency_with_verify() {
        // Test that combine and verify are consistent.
        //
        // If we create a polynomial that satisfies the constraints, then:
        // 1. verify() should return true
        // 2. The combined weights should correctly compute the polynomial evaluations
        let k = 2;
        let domain_size = 1 << k;
        let omega = F::two_adic_generator(k);

        // Create a simple polynomial: evaluations [c0, c1, c2, c3].
        let c0 = F::from_u64(1);
        let c1 = F::from_u64(2);
        let c2 = F::from_u64(3);
        let c3 = F::from_u64(4);
        let poly = EvaluationsList::new(vec![c0, c1, c2, c3]);

        // Create constraints that match the polynomial.
        // Using Horner evaluation: p(z) = c0 + c1*z + c2*z^2 + c3*z^3.
        let mut statement = DomainStatement::<F>::initialize(k, k);

        // Evaluate p(z) at z using Horner's method.
        // let z = F::from_u64(2);
        let i = 1;
        let z = omega.exp_u64(i as u64);
        let expected_eval = poly
            .iter()
            .rfold(F::ZERO, |result, &coeff| result * z + coeff);
        statement.add_constraint(i, expected_eval);

        // Verify should pass.
        assert!(statement.verify(&poly));

        // Now combine and check that the weight polynomial correctly represents
        // the select function.
        let gamma = F::from_u64(3);
        let shift = 0;
        let mut acc_weights = EvaluationsList::zero(k);
        let mut acc_sum = F::ZERO;
        statement.combine(&mut acc_weights, &mut acc_sum, gamma, shift);

        // The sum should match the expected evaluation.
        assert_eq!(acc_sum, expected_eval);

        // The weight polynomial should satisfy:
        // Σ_{b ∈ {0,1}^k} poly(b) · W(b) = expected_eval
        let mut computed_sum = F::ZERO;
        for b in 0..domain_size {
            computed_sum += poly.as_slice()[b] * acc_weights.as_slice()[b];
        }
        assert_eq!(computed_sum, expected_eval);
    }

    proptest! {
        #[test]
        fn prop_select_statement_combine_sum(
            // Number of variables (1 to 4 for reasonable test size).
            k in 1usize..=4,
            // Number of constraints (1 to 5).
            num_constraints in 1usize..=5,
            // Random evaluation points (avoiding 0 for better coverage).
            // Generate exactly num_constraints values.
            z_indicies in prop::collection::vec(1u32..100, 1..=5),
            // Random expected evaluations.
            s_values in prop::collection::vec(0u32..100, 1..=5),
            // Random challenge.
            challenge in 1u32..50,
        ) {
            // Ensure we have enough values for the test.
            let actual_num_constraints = num_constraints.min(z_indicies.len()).min(s_values.len());
            if actual_num_constraints == 0 {
                return Ok(());
            }

            let z_indicies = &z_indicies[..actual_num_constraints];
            let z_indicies = z_indicies.iter().map(|&z| z  as usize % (1 << k)).sorted().dedup().collect::<Vec<_>>();
            let s_values = &s_values[..z_indicies.len()];

            // Create statement with random constraints.
            let mut statement = DomainStatement::<F, >::initialize(k, k);
            for (&z, &s) in z_indicies.iter().zip(s_values.iter()) {
                statement.add_constraint(z, F::from_u32(s));
            }

            let gamma = F::from_u32(challenge);

            // Combine with shift=0.
            let mut acc_weights = EvaluationsList::zero(k);
            let mut acc_sum = F::ZERO;
            statement.combine(&mut acc_weights, &mut acc_sum, gamma, 0);

            // Compute expected sum manually: S = Σ_i γ^i · s_i.
            let mut expected_sum = F::ZERO;
            for (i, &s) in s_values.iter().enumerate() {
                expected_sum += gamma.exp_u64(i as u64) * F::from_u32(s);
            }

            prop_assert_eq!(acc_sum, expected_sum);
        }
    }

    proptest! {
        #[test]
        fn prop_select_statement_verify(
            // Polynomial evaluations (2^k values for k=3).
            poly_evals in prop::collection::vec(0u32..100, 8),
            // Evaluation point (avoiding 0 for better coverage).
            i in 1u32..50,
        ) {
            let k = 3; // Fixed k=3 gives 2^3 = 8 evaluations.
            let poly = EvaluationsList::new(poly_evals.into_iter().map(F::from_u32).collect());

            // Compute expected evaluation using Horner's method.
            let i = i as usize % (1 << k);
            let omega = F::two_adic_generator(k);
            let z = omega.exp_u64(i as u64);
            let expected_eval = poly
                .iter()
                .rfold(F::ZERO, |result, &coeff| result * z + coeff);

            // Create statement with correct constraint.
            let mut statement = DomainStatement::<F, >::initialize(k, k);
            statement.add_constraint(i, expected_eval);

            // Verify should pass.
            prop_assert!(statement.verify(&poly));

            // Add a wrong constraint (off by 1, unless it wraps to same value).
            let wrong_eval = expected_eval + F::ONE;
            if wrong_eval != expected_eval {
                statement.add_constraint((i+1) % (1 << k), wrong_eval);
                // Verify should fail now.
                prop_assert!(!statement.verify(&poly));
            }
        }
    }

    proptest! {
        #[test]
        fn prop_combine_evals_consistency(
            // Number of constraints.
            num_constraints in 1usize..=5,
            // Random evaluations.
            s_values in prop::collection::vec(0u32..100, 1..=5),
            // Random challenge.
            challenge in 1u32..50,
            // Random shift.
            shift in 0usize..3,
        ) {
            let num_variables = 2;
            let s_values = &s_values[..num_constraints.min(s_values.len()).min(1<<num_variables)];

            // Create statement with arbitrary z values (they don't matter for this test).
            let mut statement = DomainStatement::<F, >::initialize(num_variables, num_variables);
            for (i, &s) in s_values.iter().enumerate() {
                statement.add_constraint(i, F::from_u32(s));
            }

            let gamma = F::from_u32(challenge);

            // Method 1: Use combine_evals.
            let mut claimed_eval1 = F::ZERO;
            statement.combine_evals(&mut claimed_eval1, gamma, shift);

            // Method 2: Compute manually.
            let mut claimed_eval2 = F::ZERO;
            for (i, &s) in s_values.iter().enumerate() {
                claimed_eval2 += gamma.exp_u64((i + shift) as u64) * F::from_u32(s);
            }

            prop_assert_eq!(claimed_eval1, claimed_eval2);
        }
    }

    fn combine_ref<F: TwoAdicField, EF: ExtensionField<F>>(
        out: &mut EvaluationsList<EF>,
        statement: &DomainStatement<EF>,
        alpha: EF,
        shift: usize,
    ) {
        let k = statement.num_variables();
        statement
            .iter::<F>()
            .zip(alpha.powers().skip(shift))
            .for_each(|((var, _), alpha)| {
                EF::from(var)
                    .shifted_powers(alpha)
                    .take(1 << k)
                    .zip(out.0.iter_mut())
                    .for_each(|(el, out)| *out += el);
            });
    }

    #[test]
    fn test_packed_combine() {
        type PackedExt = <EF as ExtensionField<F>>::ExtensionPacking;

        let mut rng = SmallRng::seed_from_u64(1);
        let alpha: EF = rng.random();
        let k_pack = log2_strict_usize(<F as Field>::Packing::WIDTH);

        let mut shift = 0;
        for k in k_pack..10 {
            for rate in 0..3 {
                let k_domain = k + rate;
                let mut out0 = EvaluationsList::zero(k);
                let mut out1 = EvaluationsList::zero(k);
                let mut out_packed = EvaluationsList::<PackedExt>::zero(k - k_pack);
                let mut sum0 = EF::ZERO;
                let mut sum1 = EF::ZERO;
                for n in [1, 2, 10, 11] {
                    let indicies = (0..n)
                        .map(|_| rng.random_range::<usize, _>(0..1 << (k_domain)))
                        .sorted()
                        .dedup()
                        .collect::<Vec<usize>>();

                    let evals = (0..indicies.len())
                        .map(|_| rng.random())
                        .collect::<Vec<_>>();

                    let statement = DomainStatement::<EF>::new(k, k_domain, &indicies, &evals);
                    combine_ref::<F, EF>(&mut out0, &statement, alpha, shift);
                    statement.combine::<F>(&mut out1, &mut sum0, alpha, shift);
                    assert_eq!(out0, out1);
                    statement.combine_packed::<F>(&mut out_packed, &mut sum1, alpha, shift);

                    assert_eq!(sum0, sum1);
                    assert_eq!(
                        out0.0,
                        <<EF as ExtensionField<F>>::ExtensionPacking as PackedFieldExtension<
                            F,
                            EF,
                        >>::to_ext_iter(
                            out_packed.as_slice().iter().copied(),
                        )
                        .collect::<Vec<_>>()
                    );

                    shift += statement.len();
                }
            }
        }
    }
}
