use super::{committer::Witness, parameters::WhirConfig, statement::Statement};
use p3_field::{Field, PrimeCharacteristicRing, TwoAdicField};

#[derive(Debug)]
pub struct Prover<F, PowStrategy, Perm16, Perm24>(pub WhirConfig<F, PowStrategy, Perm16, Perm24>)
where
    F: Field + TwoAdicField,
    <F as PrimeCharacteristicRing>::PrimeSubfield: TwoAdicField;

impl<F, PowStrategy, Perm16, Perm24> Prover<F, PowStrategy, Perm16, Perm24>
where
    F: Field + TwoAdicField,
    <F as PrimeCharacteristicRing>::PrimeSubfield: TwoAdicField,
{
    fn validate_parameters(&self) -> bool {
        self.0.mv_parameters.num_variables ==
            self.0.folding_factor.total_number(self.0.n_rounds()) + self.0.final_sumcheck_rounds
    }

    fn validate_statement(&self, statement: &Statement<F>) -> bool {
        if !statement.num_variables() == self.0.mv_parameters.num_variables {
            return false;
        }
        if !self.0.initial_statement && !statement.constraints.is_empty() {
            return false;
        }
        true
    }

    fn validate_witness(&self, witness: &Witness<F, Perm16, Perm24>) -> bool {
        assert_eq!(witness.ood_points.len(), witness.ood_answers.len());
        if !self.0.initial_statement {
            assert!(witness.ood_points.is_empty());
        }
        witness.polynomial.num_variables() == self.0.mv_parameters.num_variables
    }
}
