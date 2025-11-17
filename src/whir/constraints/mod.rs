use p3_field::{ExtensionField, Field};

use crate::{
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    whir::constraints::statement::{EqStatement, SelectStatement},
};

/// Constraint evaluation utilities.
pub mod evaluator;

/// Statement types for polynomial evaluation constraints.
pub mod statement;

#[derive(Clone, Debug)]
pub struct Constraint<F: Field, EF: ExtensionField<F>> {
    pub eq_statement: EqStatement<EF>,
    pub sel_statement: SelectStatement<F, EF>,
    pub challenge: EF,
}

impl<F: Field, EF: ExtensionField<F>> Constraint<F, EF> {
    pub const fn new(
        challenge: EF,
        eq_statement: EqStatement<EF>,
        sel_statement: SelectStatement<F, EF>,
    ) -> Self {
        assert!(eq_statement.num_variables() == sel_statement.num_variables());
        Self {
            eq_statement,
            sel_statement,
            challenge,
        }
    }

    pub const fn new_eq_only(challenge: EF, eq_statement: EqStatement<EF>) -> Self {
        let num_variables = eq_statement.num_variables();
        Self::new(
            challenge,
            eq_statement,
            SelectStatement::initialize(num_variables),
        )
    }

    pub const fn num_variables(&self) -> usize {
        self.eq_statement.num_variables()
    }

    pub fn combine_evals(&self, eval: &mut EF) {
        self.eq_statement.combine_evals(eval, self.challenge);
        self.sel_statement
            .combine_evals(eval, self.challenge, self.eq_statement.len());
    }

    pub fn combine(&self, combined: &mut EvaluationsList<EF>, eval: &mut EF) {
        self.eq_statement
            .combine_hypercube::<F, true>(combined, eval, self.challenge);
        self.sel_statement
            .combine(combined, eval, self.challenge, self.eq_statement.len());
    }

    pub fn combine_new(&self) -> (EvaluationsList<EF>, EF) {
        let mut combined = EvaluationsList::zero(self.num_variables());
        let mut eval = EF::ZERO;
        self.eq_statement
            .combine_hypercube::<F, false>(&mut combined, &mut eval, self.challenge);
        self.sel_statement.combine(
            &mut combined,
            &mut eval,
            self.challenge,
            self.eq_statement.len(),
        );
        (combined, eval)
    }

    pub const fn validate_for_skip_case(&self) {
        assert!(
            self.sel_statement.is_empty(),
            "select constraints not supported in skip case"
        );
    }

    pub fn iter_eqs(&self) -> impl Iterator<Item = (&MultilinearPoint<EF>, EF)> {
        self.eq_statement.points.iter().zip(self.challenge.powers())
    }

    pub fn iter_sels(&self) -> impl Iterator<Item = (&F, EF)> {
        self.sel_statement
            .vars
            .iter()
            .zip(self.challenge.powers().skip(self.eq_statement.len()))
    }
}
