use alloc::{vec, vec::Vec};

use p3_field::{ExtensionField, Field, dot_product};
use p3_multilinear_util::eq_batch::eval_pow_batch_base;
use tracing::instrument;

use crate::poly::evals::EvaluationsList;

/// A batched system of `select` constraints of the form `p(z_i) = s_i` on `{0,1}^k`.
///
/// Each entry ties a variable `z_i ∈ F` to an expected polynomial evaluation `s_i ∈ EF`.
/// This check is performed using the "select trick", which verifies the claim:
///
/// $$ \sum_{b \in \{0,1\}^k} P(b) \cdot select(pow(z_i), b) = s_i $$
///
/// Batching multiple constraints with a random challenge `γ` produces a single
/// combined weight polynomial `W(b)` and a single scalar `S`.
///
/// Invariants
/// ----------
/// - `vars.len() == evaluations.len()`.
#[derive(Clone, Debug)]
pub struct SelectStatement<F, EF> {
    /// Number of variables in the multilinear polynomial space.
    num_variables: usize,
    /// List of evaluation points.
    pub(crate) vars: Vec<F>,
    /// List of target evaluations.
    evaluations: Vec<EF>,
}

impl<F: Field, EF: ExtensionField<F>> SelectStatement<F, EF> {
    /// Creates an empty `Statement<F>` for polynomials with `num_variables` variables.
    #[must_use]
    pub const fn initialize(num_variables: usize) -> Self {
        Self {
            num_variables,
            vars: Vec::new(),
            evaluations: Vec::new(),
        }
    }

    /// Creates a filled `Statement<F>` for polynomials with `num_variables` variables.
    #[must_use]
    pub fn new(num_variables: usize, vars: Vec<F>, evaluations: Vec<EF>) -> Self {
        assert_eq!(vars.len(), evaluations.len());
        Self {
            num_variables,
            vars,
            evaluations,
        }
    }

    /// Returns the number of variables defining the polynomial space.
    #[must_use]
    pub const fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// Returns true if the statement contains no constraints.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        debug_assert!(self.vars.is_empty() == self.evaluations.is_empty());
        self.vars.is_empty()
    }

    /// Returns an iterator over the evaluation constraints in the statement.
    pub fn iter(&self) -> impl Iterator<Item = (&F, &EF)> {
        self.vars.iter().zip(self.evaluations.iter())
    }

    /// Returns the number of constraints in the statement.
    #[must_use]
    pub const fn len(&self) -> usize {
        debug_assert!(self.vars.len() == self.evaluations.len());
        self.vars.len()
    }

    /// Verifies that a given polynomial satisfies all constraints in the statement.
    #[must_use]
    pub fn verify(&self, poly: &EvaluationsList<EF>) -> bool {
        self.iter().all(|(&var, &expected_eval)| {
            // Horner evaluation of p at var
            poly.iter()
                .rfold(EF::ZERO, |result, coeff| result * var + *coeff)
                == expected_eval
        })
    }

    /// Adds an evaluation constraint `p(z) = s` to the system.
    ///
    /// Assumes the evaluation `s` is already known.
    pub fn add_constraint(&mut self, var: F, eval: EF) {
        self.vars.push(var);
        self.evaluations.push(eval);
    }

    /// Combines all constraints into a single aggregated polynomial and expected sum using a challenge.
    #[instrument(skip_all, fields(num_constraints = self.len(), num_variables = self.num_variables()))]
    pub fn combine<Base>(
        &self,
        acc_weights: &mut EvaluationsList<EF>,
        acc_sum: &mut EF,
        challenge: EF,
        shift: usize,
    ) where
        Base: Field,
        F: ExtensionField<Base>,
    {
        // If there are no constraints, the combination is:
        // - The combined polynomial W(X) is identically zero (all evaluations = 0).
        // - The combined expected sum S is zero.
        if self.vars.is_empty() {
            return;
        }

        // Number of constraints (columns).
        let n = self.len();

        // Precompute γ^i for i = 0..n-1 (random linear-combination weights).
        let challenges = challenge.powers().skip(shift).take(n).collect::<Vec<_>>();

        eval_pow_batch_base::<F, EF, true>(&self.vars, &mut acc_weights.0, challenges.as_slice());

        // Combine expected evaluations: S = ∑_i γ^i * s_i
        *acc_sum +=
            dot_product::<EF, _, _>(challenges.into_iter(), self.evaluations.iter().copied());
    }

    /// Combines a list of evals into a single linear combination using powers of `gamma`,
    /// and updates the running claimed_eval in place.
    ///
    /// # Arguments
    /// - `claimed_eval`: Mutable reference to the total accumulated claimed eval so far. Updated in place.
    /// - `challenge`: A random extension field element used to weight the evals.
    pub fn combine_evals(&self, claimed_eval: &mut EF, challenge: EF, shift: usize) {
        *claimed_eval += dot_product::<EF, _, _>(
            self.evaluations.iter().copied(),
            challenge.powers().skip(shift).take(self.len()),
        );
    }
}
