use p3_challenger::{CanSample, GrindingChallenger};
use p3_field::{Field, PrimeCharacteristicRing, PrimeField32, TwoAdicField};

use super::parameters::WhirConfig;
use crate::{
    fs_utils::{OODIOPattern, WhirPoWIOPattern},
    sumcheck::sumcheck_single_io_pattern::SumcheckSingleChallenger,
};

/// Trait for adding Merkle digests to a Fiat-Shamir transcript.
pub trait DigestChallenger<F: Field, const DIGEST_ELEMS: usize> {
    fn observe_digest(&mut self, label: &str);
}

/// Trait that defines how a Whir proof's transcript interaction is constructed.
pub trait WhirChallengerTranscript<F, const DIGEST_ELEMS: usize>
where
    F: Field + PrimeField32 + TwoAdicField,
    <F as PrimeCharacteristicRing>::PrimeSubfield: TwoAdicField,
{
    fn commit_statement(&mut self, params: &WhirConfig<F, (), (), ()>);

    fn add_whir_proof(&mut self, params: &WhirConfig<F, (), (), ()>);
}

impl<F, Challenger, const DIGEST_ELEMS: usize> WhirChallengerTranscript<F, DIGEST_ELEMS>
    for Challenger
where
    F: Field + PrimeField32 + TwoAdicField,
    <F as PrimeCharacteristicRing>::PrimeSubfield: TwoAdicField,
    Challenger: CanSample<F>
        + GrindingChallenger
        + WhirPoWIOPattern
        + SumcheckSingleChallenger<F>
        + DigestChallenger<F, DIGEST_ELEMS>
        + OODIOPattern<F>,
{
    fn commit_statement(&mut self, params: &WhirConfig<F, (), (), ()>) {
        self.observe_digest("merkle_digest");

        if params.committment_ood_samples > 0 {
            assert!(params.initial_statement);
            self.add_ood(params.committment_ood_samples);
        }
    }

    fn add_whir_proof(&mut self, params: &WhirConfig<F, (), (), ()>) {
        if params.initial_statement {
            // Simulate initial sumcheck round
            let _initial_combination = self.sample();
            self.add_sumcheck(
                params.folding_factor.at_round(0),
                params.starting_folding_pow_bits as usize,
            );
        } else {
            let _folding_randomness = self.sample_vec(params.folding_factor.at_round(0));
            self.pow(params.starting_folding_pow_bits as usize);
        }

        let mut domain_size = params.starting_domain.size();
        for (round, r) in params.round_parameters.iter().enumerate() {
            let folded_domain_size = domain_size >> params.folding_factor.at_round(round);
            let domain_size_bits =
                ((folded_domain_size * 2 - 1).ilog2() as usize).next_power_of_two();

            self.observe_digest("merkle_digest");
            self.add_ood(r.ood_samples);
            for _ in 0..r.num_queries {
                let _index: usize = self.sample_bits(domain_size_bits);
            }

            self.pow(r.pow_bits as usize);
            let _comb_rand = self.sample();
            self.add_sumcheck(
                params.folding_factor.at_round(round + 1),
                r.folding_pow_bits as usize,
            );

            domain_size >>= 1;
        }

        let folded_domain_size =
            domain_size >> params.folding_factor.at_round(params.round_parameters.len());
        let domain_size_bits = ((folded_domain_size * 2 - 1).ilog2() as usize).next_power_of_two();

        let _final_coeffs = self.sample_vec(1 << params.final_sumcheck_rounds);
        for _ in 0..params.final_queries {
            let _query: usize = self.sample_bits(domain_size_bits);
        }
        self.pow(params.final_pow_bits as usize);
        self.add_sumcheck(params.final_sumcheck_rounds, params.final_folding_pow_bits as usize);
    }
}
