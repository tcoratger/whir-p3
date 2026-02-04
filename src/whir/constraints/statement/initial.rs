use alloc::{vec, vec::Vec};

use p3_field::{ExtensionField, Field, PackedValue};
use p3_util::log2_strict_usize;

use crate::{
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    sumcheck::svo::SplitEq,
    whir::{constraints::statement::EqStatement, parameters::SumcheckStrategy},
};

#[derive(Clone, Debug)]
pub struct InitialStatement<F: Field, EF: ExtensionField<F>> {
    pub(crate) poly: EvaluationsList<F>,
    pub(crate) inner: InitialStatementInner<F, EF>,
}

#[derive(Clone, Debug)]
pub(crate) enum InitialStatementInner<F: Field, EF: ExtensionField<F>> {
    Classic(EqStatement<EF>),
    Svo {
        l0: usize,
        split_eqs: Vec<SplitEq<F, EF>>,
    },
}

impl<F: Field, EF: ExtensionField<F>> InitialStatement<F, EF> {
    const fn new_classic(poly: EvaluationsList<F>) -> Self {
        let num_variables = poly.num_variables();
        Self {
            poly,
            inner: InitialStatementInner::new_classic(num_variables),
        }
    }

    const fn new_svo(poly: EvaluationsList<F>, l0: usize) -> Self {
        Self {
            poly,
            inner: InitialStatementInner::new_svo(l0),
        }
    }

    #[must_use]
    pub const fn new(poly: EvaluationsList<F>, l0: usize, strategy: SumcheckStrategy) -> Self {
        let k = poly.num_variables();
        match strategy {
            SumcheckStrategy::Classic => Self::new_classic(poly),
            SumcheckStrategy::Svo => {
                if k > 2 * log2_strict_usize(F::Packing::WIDTH) + l0 {
                    Self::new_svo(poly, l0)
                } else {
                    // Fallback to classic sumcheck if size is not large enough to apply SVO
                    Self::new_classic(poly)
                }
            }
        }
    }

    #[must_use]
    pub fn evaluate(&mut self, point: &MultilinearPoint<EF>) -> EF {
        assert_eq!(point.num_variables(), self.num_variables());
        self.inner.evaluate(point, &self.poly)
    }

    pub(crate) const fn num_variables(&self) -> usize {
        self.poly.num_variables()
    }

    pub(crate) const fn is_empty(&self) -> bool {
        match &self.inner {
            InitialStatementInner::Classic(statement) => statement.is_empty(),
            InitialStatementInner::Svo { split_eqs, .. } => split_eqs.is_empty(),
        }
    }

    pub(crate) const fn len(&self) -> usize {
        match &self.inner {
            InitialStatementInner::Classic(statement) => statement.len(),
            InitialStatementInner::Svo { split_eqs, .. } => split_eqs.len(),
        }
    }

    #[must_use]
    pub fn normalize(&self) -> EqStatement<EF> {
        match &self.inner {
            InitialStatementInner::Classic(statement) => statement.clone(),
            InitialStatementInner::Svo { split_eqs, .. } => {
                let points: Vec<MultilinearPoint<EF>> =
                    split_eqs.iter().map(|se| se.point.clone()).collect();
                let evals: Vec<EF> = split_eqs.iter().map(|se| se.eval).collect();
                let mut statement = EqStatement::initialize(self.num_variables());
                points
                    .iter()
                    .zip(evals.iter())
                    .for_each(|(pt, &ev)| statement.add_evaluated_constraint(pt.clone(), ev));
                statement
            }
        }
    }
}

impl<F: Field, EF: ExtensionField<F>> InitialStatementInner<F, EF> {
    const fn new_classic(num_variables: usize) -> Self {
        Self::Classic(EqStatement::initialize(num_variables))
    }

    const fn new_svo(l0: usize) -> Self {
        Self::Svo {
            split_eqs: vec![],
            l0,
        }
    }

    pub(crate) fn evaluate(
        &mut self,
        point: &MultilinearPoint<EF>,
        poly: &EvaluationsList<F>,
    ) -> EF {
        match self {
            Self::Classic(statement) => {
                let eval = poly.evaluate_hypercube_base(point);
                statement.add_evaluated_constraint(point.clone(), eval);
                eval
            }
            Self::Svo { split_eqs, l0 } => {
                let split_eq = SplitEq::new(point, *l0, poly);
                let eval = split_eq.eval;
                split_eqs.push(split_eq);
                eval
            }
        }
    }
}
