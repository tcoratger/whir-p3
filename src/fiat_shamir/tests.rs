use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, extension::BinomialExtensionField};
use proptest::prelude::*;
use rand::SeedableRng;

use crate::fiat_shamir::{
    domain_separator::DomainSeparator, prover::ProverState, verifier::VerifierState,
};

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type Perm = Poseidon2BabyBear<16>;
type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

fn make_challenger() -> MyChallenger {
    let perm = Perm::new_from_rng_128(&mut rand::rngs::SmallRng::seed_from_u64(42));
    DuplexChallenger::new(perm)
}

fn make_domain_separator() -> DomainSeparator<EF, F> {
    DomainSeparator::new(vec![])
}

proptest! {
    #[test]
    fn test_base_scalar_roundtrip(vals in prop::collection::vec(1u64..10, 1..10)) {
        let vals: Vec<F> = vals.into_iter().map(F::from_u64).collect();
        let domsep = make_domain_separator();
        let challenger = make_challenger();

        let mut prover = ProverState::<F, EF, MyChallenger>::new(&domsep, challenger.clone());
        prover.add_base_scalars(&vals);
        let proof_data = prover.proof_data().to_vec();

        let mut verifier = VerifierState::<F, EF, MyChallenger>::new(&domsep, proof_data, challenger);
        let recovered = verifier.next_base_scalars_vec(vals.len()).unwrap();
        prop_assert_eq!(vals, recovered);
    }

    #[test]
    fn test_extension_scalar_roundtrip(vals in prop::collection::vec(prop::array::uniform4(1u64..10), 1..5)) {
        let ext_vals: Vec<EF> = vals.into_iter()
            .map(|arr| EF::from_basis_coefficients_slice(&arr.map(F::from_u64)).unwrap())
            .collect();

        let domsep = make_domain_separator();
        let challenger = make_challenger();

        let mut prover = ProverState::<F, EF, MyChallenger>::new(&domsep, challenger.clone());
        prover.add_extension_scalars(&ext_vals);
        let proof_data = prover.proof_data().to_vec();

        let mut verifier = VerifierState::<F, EF, MyChallenger>::new(&domsep, proof_data, challenger);
        let recovered = verifier.next_extension_scalars_vec(ext_vals.len()).unwrap();
        prop_assert_eq!(ext_vals, recovered);
    }

    #[test]
    fn test_hint_base_scalar_roundtrip(vals in prop::collection::vec(1u64..10, 1..10)) {
        let vals: Vec<F> = vals.into_iter().map(F::from_u64).collect();
        let domsep = make_domain_separator();
        let challenger = make_challenger();

        let mut prover = ProverState::<F, EF, MyChallenger>::new(&domsep, challenger.clone());
        prover.hint_base_scalars(&vals);
        let proof_data = prover.proof_data().to_vec();

        let mut verifier = VerifierState::<F, EF, MyChallenger>::new(&domsep, proof_data, challenger);
        let recovered = verifier.receive_hint_base_scalars(vals.len()).unwrap();
        prop_assert_eq!(vals, recovered);
    }

    #[test]
    fn test_hint_extension_scalar_roundtrip(vals in prop::collection::vec(prop::array::uniform4(1u64..10), 1..5)) {
        let ext_vals: Vec<EF> = vals.into_iter()
            .map(|arr| EF::from_basis_coefficients_slice(&arr.map(F::from_u64)).unwrap())
            .collect();

        let domsep = make_domain_separator();
        let challenger = make_challenger();

        let mut prover = ProverState::<F, EF, MyChallenger>::new(&domsep, challenger.clone());
        prover.hint_extension_scalars(&ext_vals);
        let proof_data = prover.proof_data().to_vec();

        let mut verifier = VerifierState::<F, EF, MyChallenger>::new(&domsep, proof_data, challenger);
        let recovered = verifier.receive_hint_extension_scalars(ext_vals.len()).unwrap();
        prop_assert_eq!(ext_vals, recovered);
    }

    #[test]
    fn test_pow_grinding_roundtrip(bits in 1usize..8) {
        let domsep = make_domain_separator();
        let challenger = make_challenger();

        let mut prover = ProverState::<F, EF, MyChallenger>::new(&domsep, challenger.clone());
        prover.pow_grinding(bits);
        let proof_data = prover.proof_data().to_vec();

        let mut verifier = VerifierState::<F, EF, MyChallenger>::new(&domsep, proof_data, challenger);
        verifier.check_pow_grinding(bits).unwrap();
    }
}
