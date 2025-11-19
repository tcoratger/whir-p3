use p3_baby_bear::BabyBear;
use p3_field::{extension::BinomialExtensionField, AbstractField};

type F = BabyBear;
type EF4 = BinomialExtensionField<F, 4>;

use whir_p3::poly::evals::EvaluationsList;

fn main() {
    const N: usize = 2;
    
    let poly = EvaluationsList::new(
        (1..=16).map(|i| EF4::from_canonical_u64(i)).collect::<Vec<_>>(),
    );
    
    let r1 = EF4::from_canonical_u64(2);
    let r0 = EF4::from_canonical_u64(3);
    
    let e_in = EvaluationsList::new(vec![
        (EF4::ONE - r1) * (EF4::ONE - r0),
        (EF4::ONE - r1) * r0,
        r1 * (EF4::ONE - r0),
        r1 * r0,
    ]);
    
    let s0 = EF4::from_canonical_u64(5);
    let s1 = EF4::from_canonical_u64(7);
    
    let e_out = [
        EvaluationsList::new(vec![EF4::ONE - s0, s0]),
        EvaluationsList::new(vec![EF4::ONE - s1, s1]),
    ];
    
    let result = poly.compute_svo_accumulators::<EF4, N>(&e_in, &e_out);
    
    println!("E_in values:");
    for (i, val) in e_in.as_slice().iter().enumerate() {
        println!("  e_in[{}] = {:?}", i, val);
    }
    
    println!("\nE_out[0] values:");
    for (i, val) in e_out[0].as_slice().iter().enumerate() {
        println!("  e_out[0][{}] = {:?}", i, val);
    }
    
    println!("\nE_out[1] values:");
    for (i, val) in e_out[1].as_slice().iter().enumerate() {
        println!("  e_out[1][{}] = {:?}", i, val);
    }
    
    println!("\nRound 0 accumulators:");
    for (i, val) in result.at_round(0).iter().enumerate() {
        println!("  round0[{}] = {:?}", i, val);
    }
    
    println!("\nRound 1 accumulators:");
    for (i, val) in result.at_round(1).iter().enumerate() {
        println!("  round1[{}] = {:?}", i, val);
    }
}
