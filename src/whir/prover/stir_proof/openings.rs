use p3_field::{ExtensionField, Field};

/// Opening results from base field Merkle commitments in STIR protocol.
#[derive(Debug, Clone)]
pub struct BaseFieldOpenings<F, const DIGEST_ELEMS: usize>
where
    F: Field,
{
    /// Polynomial evaluation values at queried positions
    pub answers: Vec<Vec<F>>,
    /// Merkle authentication paths proving commitment integrity
    pub proofs: Vec<Vec<[F; DIGEST_ELEMS]>>,
}

impl<F, const DIGEST_ELEMS: usize> BaseFieldOpenings<F, DIGEST_ELEMS>
where
    F: Field,
{
    /// Creates a new instance with pre-allocated capacity.
    ///
    /// Optimizes memory allocation for known number of openings.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Expected number of openings
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        // Pre-allocate vectors to avoid reallocation during construction
        Self {
            answers: Vec::with_capacity(capacity),
            proofs: Vec::with_capacity(capacity),
        }
    }

    /// Returns the number of openings contained.
    ///
    /// Counts total opening results stored in this collection.
    #[must_use]
    pub const fn len(&self) -> usize {
        // Ensure consistency between answers and proofs vectors
        debug_assert!(self.answers.len() == self.proofs.len());
        self.answers.len()
    }

    /// Returns true if no openings are stored.
    ///
    /// Checks if the collection is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Adds a new opening result to the collection.
    ///
    /// Stores both the polynomial evaluation and its authentication path.
    ///
    /// # Arguments
    ///
    /// * `answer` - Polynomial values at the opened position
    /// * `proof` - Merkle authentication path for verification
    pub fn push(&mut self, answer: Vec<F>, proof: Vec<[F; DIGEST_ELEMS]>) {
        // Store the evaluation values
        self.answers.push(answer);
        // Store the corresponding authentication path
        self.proofs.push(proof);
    }

    /// Returns an iterator over polynomial evaluation vectors.
    ///
    /// Provides access to all stored answer vectors.
    pub fn answers(&self) -> impl Iterator<Item = &Vec<F>> {
        self.answers.iter()
    }

    /// Returns an iterator over authentication path vectors.
    ///
    /// Provides access to all stored proof vectors.
    pub fn proofs(&self) -> impl Iterator<Item = &Vec<[F; DIGEST_ELEMS]>> {
        self.proofs.iter()
    }
}

/// Opening results from extension field Merkle commitments in STIR protocol.
#[derive(Debug, Clone)]
pub struct ExtensionFieldOpenings<F, EF, const DIGEST_ELEMS: usize>
where
    F: Field,
    EF: ExtensionField<F>,
{
    /// Polynomial evaluation values in the extension field
    pub answers: Vec<Vec<EF>>,
    /// Merkle authentication paths in base field
    pub proofs: Vec<Vec<[F; DIGEST_ELEMS]>>,
}

impl<F, EF, const DIGEST_ELEMS: usize> ExtensionFieldOpenings<F, EF, DIGEST_ELEMS>
where
    F: Field,
    EF: ExtensionField<F>,
{
    /// Creates a new instance with pre-allocated capacity.
    ///
    /// Optimizes memory allocation for known number of extension field openings.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Expected number of openings
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        // Pre-allocate vectors to avoid reallocation during construction
        Self {
            answers: Vec::with_capacity(capacity),
            proofs: Vec::with_capacity(capacity),
        }
    }

    /// Returns the number of openings contained.
    ///
    /// Counts total opening results stored in this collection.
    #[must_use]
    pub const fn len(&self) -> usize {
        // Ensure consistency between answers and proofs vectors
        debug_assert!(self.answers.len() == self.proofs.len());
        self.answers.len()
    }

    /// Returns true if no openings are stored.
    ///
    /// Checks if the collection is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Adds a new opening result to the collection.
    ///
    /// Stores extension field polynomial evaluation and base field authentication path.
    ///
    /// # Arguments
    ///
    /// * `answer` - Polynomial values in extension field at the opened position
    /// * `proof` - Merkle authentication path in base field for verification
    pub fn push(&mut self, answer: Vec<EF>, proof: Vec<[F; DIGEST_ELEMS]>) {
        // Store the extension field evaluation values
        self.answers.push(answer);
        // Store the corresponding base field authentication path
        self.proofs.push(proof);
    }

    /// Returns an iterator over extension field polynomial evaluation vectors.
    ///
    /// Provides access to all stored extension field answer vectors.
    pub fn answers(&self) -> impl Iterator<Item = &Vec<EF>> {
        self.answers.iter()
    }

    /// Returns an iterator over base field authentication path vectors.
    ///
    /// Provides access to all stored proof vectors in base field.
    pub fn proofs(&self) -> impl Iterator<Item = &Vec<[F; DIGEST_ELEMS]>> {
        self.proofs.iter()
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    #[test]
    fn test_base_field_openings() {
        let mut openings: BaseFieldOpenings<F, 4> = BaseFieldOpenings::with_capacity(2);

        assert!(openings.is_empty());
        assert_eq!(openings.len(), 0);

        openings.push(vec![F::ONE], vec![[F::ZERO; 4]]);
        assert!(!openings.is_empty());
        assert_eq!(openings.len(), 1);

        openings.push(vec![F::TWO], vec![[F::ONE; 4]]);
        assert_eq!(openings.len(), 2);

        // Test iterators
        assert_eq!(openings.answers().count(), 2);
        assert_eq!(openings.proofs().count(), 2);
    }

    #[test]
    fn test_extension_field_openings() {
        let mut openings: ExtensionFieldOpenings<F, EF, 4> =
            ExtensionFieldOpenings::with_capacity(1);

        assert!(openings.is_empty());
        assert_eq!(openings.len(), 0);

        openings.push(vec![EF::ONE], vec![[F::ZERO; 4]]);
        assert!(!openings.is_empty());
        assert_eq!(openings.len(), 1);

        // Test iterators
        assert_eq!(openings.answers().count(), 1);
        assert_eq!(openings.proofs().count(), 1);
    }
}
