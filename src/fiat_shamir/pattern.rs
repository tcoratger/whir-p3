use p3_field::Field;

#[derive(Debug, Clone, Copy)]
pub enum Pattern {
    Sample,
    Observe,
    Hint,
}

impl Pattern {
    #[must_use]
    pub fn as_field_element<F: Field>(self) -> F {
        F::from_u8(self as u8)
    }
}

/// Labels for items that are sampled.
#[derive(Debug, Clone, Copy)]
pub enum Sample {
    InitialCombinationRandomness,
    FoldingRandomness,
    CombinationRandomness,
    StirQueries,
    FinalQueries,
    PowQueries,
    OodQuery,
}

impl Sample {
    #[must_use]
    pub fn as_field_element<F: Field>(self) -> F {
        F::from_u8(self as u8)
    }
}

/// Labels for items that are observed.
#[derive(Debug, Clone, Copy)]
pub enum Observe {
    MerkleDigest,
    OodAnswers,
    SumcheckPoly,
    StirAnswers,
    FinalCoeffs,
    PowNonce,
}

impl Observe {
    #[must_use]
    pub fn as_field_element<F: Field>(self) -> F {
        F::from_u8(self as u8)
    }
}

/// Labels for items that are hints.
#[derive(Debug, Clone, Copy)]
pub enum Hint {
    StirQueries,
    StirAnswers,
    MerkleProof,
    DeferredWeightEvaluations,
}

impl Hint {
    #[must_use]
    pub fn as_field_element<F: Field>(self) -> F {
        F::from_u8(self as u8)
    }
}
