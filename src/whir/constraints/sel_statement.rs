use p3_field::{ExtensionField, Field, dot_product};
use p3_maybe_rayon::prelude::*;
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
        // Number of variables (bits).
        let k = self.num_variables();

        // v0^(2^0), v0^(2^1), ..., v0^(2^{k-1})
        // v1^(2^0), v1^(2^1), ..., v1^(2^{k-1})
        let bin_powers = self
            .vars
            .par_iter()
            .copied()
            .map(|mut var| {
                (0..k)
                    .map(|_| {
                        let ret = var;
                        var = var.square();
                        ret
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        // Equality matrix.
        let mut acc: Vec<F> = F::zero_vec((1 << k) * n);

        // Initialize row 0 with 1s. We will apply randomness later
        acc[..n].copy_from_slice(&vec![F::ONE; n]);

        // Expand row 0 into 2^k rows with a simple two-branch split per bit.
        //   `high  = low * z`
        // Here z ∈ {0,1} is the i-th coordinate of the constraint point.
        for i in 0..k {
            // Split the first 2^i rows into low and high halves in place.
            let (lo, hi) = acc.split_at_mut((1 << i) * n);

            // Fetch binary powers for the current round
            let bin_powers = bin_powers.iter().map(|c| c[i]).collect::<Vec<_>>();

            // Work in parallel over row pairs. Each pair has n columns.
            lo.par_chunks_mut(n)
                .zip(hi.par_chunks_mut(n))
                .for_each(|(lo, hi)| {
                    // For each column j: read its constraint point, update the pair.
                    bin_powers
                        .iter()
                        .zip(lo.iter_mut())
                        .zip(hi.iter_mut())
                        .for_each(|((&z, lo), hi)| *hi = *lo * z);
                });
        }

        // Precompute γ^i for i = 0..n-1 (random linear-combination weights).
        let challenges = challenge.powers().skip(shift).take(n).collect::<Vec<_>>();

        acc.par_chunks(n)
            .zip(acc_weights.0.par_iter_mut())
            .for_each(|(row, out)| {
                *out += row
                    .iter()
                    .zip(challenges.iter())
                    .fold(EF::ZERO, |acc, (&v, &alpha)| acc + alpha * v);
            });

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
