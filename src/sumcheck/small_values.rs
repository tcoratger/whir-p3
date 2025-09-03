#[derive(Debug, Clone, Eq, PartialEq)]
enum EvaluationPoint {
    Infinity,
    Zero,
    One 
}
impl EvaluationPoint {
    fn to_usize(&self) -> usize {
        match self {
            EvaluationPoint::Infinity => 2,
            EvaluationPoint::Zero => 0,
            EvaluationPoint::One => 1
        }
    }
}

// Esta fuinción mapea una tupla del tipo (i,v,u,y) que resulta de aplicar indx4(\beta) a un indice que correpende a un acumulador especifico.
// Recall: v in {0, 1, inf}, u in {0, 1, inf}, y in {0, 1}.
// Vamos a tomar u y v como coeficientes en base 3 (donde inf lo tomamos como 2), los representamos en binario y los concatenamos con y en su forma binaria.
// Observar que en cada ronda cambian los tamaños de v, u and y.
fn from_beta_to_index(round: u8, beta_1: &EvaluationPoint, beta_2: &EvaluationPoint, beta_3: &EvaluationPoint) -> usize {
    let beta_1 = beta_1.to_usize();
    let beta_2 = beta_2.to_usize();
    let beta_3 = beta_3.to_usize();

    match round {
        1 => {
            // (beta_1, beta_2, beta_3) = (u, y1, y2)
            // The index is the concatenation u || y
            let u = beta_1;
            let y = (beta_2 << 1) | beta_3;
            let index =  (u << 2) | y; 
            return index
        }
        2 => {
            // (beta_1, beta_2, beta_3) = (v, u, y)
            // The index is the concatenation (v * 3 + u) || y.
            let v = beta_1;
            let u = beta_2; 
            let y = beta_3; 
            let index = ((v * 3 + u) << 1) | y; 
            return index
        }
        3 => {
            // (beta_1, beta_2, beta_3) = (v1, v2, u)
            //  The index is v1 * 3^2 + v2 * 3 + u.
            let v1 = beta_1;
            let v2 = beta_2;
            let u = beta_3;
            let index = v1 * 9 + v2 * 3 + u;
            return index
        }
        _ => unreachable!()
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_from_beta_to_index() {
        let inf = EvaluationPoint::Infinity;
        let zero = EvaluationPoint::Zero;
        let one = EvaluationPoint::One;

        // Round 1
        assert_eq!(from_beta_to_index(1, &zero, &zero, &zero), 0);
        assert_eq!(from_beta_to_index(1, &zero, &zero, &one), 1);
        assert_eq!(from_beta_to_index(1, &zero, &one, &zero), 2);
        assert_eq!(from_beta_to_index(1, &zero, &one, &one), 3);
        assert_eq!(from_beta_to_index(1, &one, &zero, &zero), 4);
        assert_eq!(from_beta_to_index(1, &one, &zero,&one), 5);
        assert_eq!(from_beta_to_index(1, &one, &one, &zero), 6);
        assert_eq!(from_beta_to_index(1, &one, &one, &one), 7);
        assert_eq!(from_beta_to_index(1, &inf, &zero, &zero), 8);
        assert_eq!(from_beta_to_index(1, &inf, &zero, &one), 9);
        assert_eq!(from_beta_to_index(1, &inf, &one, &zero), 10);
        assert_eq!(from_beta_to_index(1, &inf, &one, &one), 11);

        // Round 2
        assert_eq!(from_beta_to_index(2, &zero, &zero, &zero), 0);
        assert_eq!(from_beta_to_index(2, &zero, &zero, &one), 1);
        assert_eq!(from_beta_to_index(2, &zero, &one, &zero), 2);
        assert_eq!(from_beta_to_index(2, &zero, &one, &one), 3);
        assert_eq!(from_beta_to_index(2, &zero, &inf, &zero), 4);
        assert_eq!(from_beta_to_index(2, &zero, &inf, &one), 5);

        assert_eq!(from_beta_to_index(2, &one, &zero, &zero), 6);
        assert_eq!(from_beta_to_index(2, &one, &zero, &one), 7);
        assert_eq!(from_beta_to_index(2, &one, &one, &zero), 8);
        assert_eq!(from_beta_to_index(2, &one, &one, &one), 9);
        assert_eq!(from_beta_to_index(2, &one, &inf, &zero), 10);
        assert_eq!(from_beta_to_index(2, &one, &inf, &one), 11);

        assert_eq!(from_beta_to_index(2, &inf, &zero, &zero), 12);
        assert_eq!(from_beta_to_index(2, &inf, &zero, &one), 13);
        assert_eq!(from_beta_to_index(2, &inf, &one, &zero), 14);
        assert_eq!(from_beta_to_index(2, &inf, &one, &one), 15);
        assert_eq!(from_beta_to_index(2, &inf, &inf, &zero), 16);
        assert_eq!(from_beta_to_index(2, &inf, &inf, &one), 17);

        // Round 3
        assert_eq!(from_beta_to_index(3, &zero, &zero, &zero), 0);
        assert_eq!(from_beta_to_index(3, &zero, &zero, &one), 1);
        assert_eq!(from_beta_to_index(3, &zero, &zero, &inf), 2);
        assert_eq!(from_beta_to_index(3, &zero, &one, &zero), 3);
        assert_eq!(from_beta_to_index(3, &zero, &one, &one), 4);
        assert_eq!(from_beta_to_index(3, &zero, &one, &inf), 5);
        assert_eq!(from_beta_to_index(3, &zero, &inf, &zero), 6);
        assert_eq!(from_beta_to_index(3, &zero, &inf, &one), 7);
        assert_eq!(from_beta_to_index(3, &zero, &inf, &inf), 8);
        
        assert_eq!(from_beta_to_index(3, &one, &zero, &zero), 9);
        assert_eq!(from_beta_to_index(3, &one, &zero, &one), 10);
        assert_eq!(from_beta_to_index(3, &one, &zero, &inf), 11);
        assert_eq!(from_beta_to_index(3, &one, &one, &zero), 12);
        assert_eq!(from_beta_to_index(3, &one, &one, &one), 13);
        assert_eq!(from_beta_to_index(3, &one, &one, &inf), 14);
        assert_eq!(from_beta_to_index(3, &one, &inf, &zero), 15);
        assert_eq!(from_beta_to_index(3, &one, &inf, &one), 16);
        assert_eq!(from_beta_to_index(3, &one, &inf, &inf), 17);
        
        assert_eq!(from_beta_to_index(3, &inf, &zero, &zero), 18);
        assert_eq!(from_beta_to_index(3, &inf, &zero, &one), 19);
        assert_eq!(from_beta_to_index(3, &inf, &zero, &inf), 20);
        assert_eq!(from_beta_to_index(3, &inf, &one, &zero), 21);
        assert_eq!(from_beta_to_index(3, &inf, &one, &one), 22);
        assert_eq!(from_beta_to_index(3, &inf, &one, &inf), 23);
        assert_eq!(from_beta_to_index(3, &inf, &inf, &zero), 24);
        assert_eq!(from_beta_to_index(3, &inf, &inf, &one), 25);
        assert_eq!(from_beta_to_index(3, &inf, &inf, &inf), 26);
    }
}