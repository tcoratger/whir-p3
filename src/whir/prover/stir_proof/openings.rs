use p3_field::{ExtensionField, Field};

/// Results from opening positions in base field Merkle commitments.
///
/// This struct encapsulates the opened values and their corresponding
/// authentication paths for base field operations.
#[derive(Debug, Clone)]
pub struct BaseFieldOpenings<F, const DIGEST_ELEMS: usize>
where
    F: Field,
{
    /// The opened polynomial evaluation values
    pub answers: Vec<Vec<F>>,
    /// The Merkle authentication paths for each opening
    pub proofs: Vec<Vec<[F; DIGEST_ELEMS]>>,
}

impl<F, const DIGEST_ELEMS: usize> BaseFieldOpenings<F, DIGEST_ELEMS>
where
    F: Field,
{
    /// Creates a new instance with pre-allocated capacity.
    ///
    /// # Arguments
    ///
    /// * `capacity` - The expected number of openings
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            answers: Vec::with_capacity(capacity),
            proofs: Vec::with_capacity(capacity),
        }
    }

    /// Returns the number of openings contained.
    #[must_use]
    pub fn len(&self) -> usize {
        debug_assert_eq!(self.answers.len(), self.proofs.len());
        self.answers.len()
    }

    /// Returns `true` if there are no openings.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Adds a new opening result to the collection.
    ///
    /// # Arguments
    ///
    /// * `answer` - The opened polynomial values
    /// * `proof` - The authentication path
    pub fn push(&mut self, answer: Vec<F>, proof: Vec<[F; DIGEST_ELEMS]>) {
        self.answers.push(answer);
        self.proofs.push(proof);
    }

    /// Returns an iterator over the answer vectors.
    pub fn answers(&self) -> impl Iterator<Item = &Vec<F>> {
        self.answers.iter()
    }

    /// Returns an iterator over the proof vectors.
    pub fn proofs(&self) -> impl Iterator<Item = &Vec<[F; DIGEST_ELEMS]>> {
        self.proofs.iter()
    }
}

/// Results from opening positions in extension field Merkle commitments.
///
/// This struct encapsulates the opened values and their corresponding
/// authentication paths for extension field operations.
#[derive(Debug, Clone)]
pub struct ExtensionFieldOpenings<F, EF, const DIGEST_ELEMS: usize>
where
    F: Field,
    EF: ExtensionField<F>,
{
    /// The opened polynomial evaluation values in the extension field
    pub answers: Vec<Vec<EF>>,
    /// The Merkle authentication paths (always in base field)
    pub proofs: Vec<Vec<[F; DIGEST_ELEMS]>>,
}

impl<F, EF, const DIGEST_ELEMS: usize> ExtensionFieldOpenings<F, EF, DIGEST_ELEMS>
where
    F: Field,
    EF: ExtensionField<F>,
{
    /// Creates a new instance with pre-allocated capacity.
    ///
    /// # Arguments
    ///
    /// * `capacity` - The expected number of openings
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            answers: Vec::with_capacity(capacity),
            proofs: Vec::with_capacity(capacity),
        }
    }

    /// Returns the number of openings contained.
    #[must_use]
    pub fn len(&self) -> usize {
        debug_assert_eq!(self.answers.len(), self.proofs.len());
        self.answers.len()
    }

    /// Returns `true` if there are no openings.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Adds a new opening result to the collection.
    ///
    /// # Arguments
    ///
    /// * `answer` - The opened polynomial values in extension field
    /// * `proof` - The authentication path in base field
    pub fn push(&mut self, answer: Vec<EF>, proof: Vec<[F; DIGEST_ELEMS]>) {
        self.answers.push(answer);
        self.proofs.push(proof);
    }

    /// Returns an iterator over the answer vectors.
    pub fn answers(&self) -> impl Iterator<Item = &Vec<EF>> {
        self.answers.iter()
    }

    /// Returns an iterator over the proof vectors.
    pub fn proofs(&self) -> impl Iterator<Item = &Vec<[F; DIGEST_ELEMS]>> {
        self.proofs.iter()
    }
}

/// Utility functions for batch opening operations.
pub mod batch_ops {
    use p3_field::{ExtensionField, Field};

    use super::{BaseFieldOpenings, ExtensionFieldOpenings};

    /// Creates a new BaseFieldOpenings with the given capacity.
    #[must_use]
    pub fn create_base_field_openings<F, const DIGEST_ELEMS: usize>(
        capacity: usize,
    ) -> BaseFieldOpenings<F, DIGEST_ELEMS>
    where
        F: Field,
    {
        BaseFieldOpenings::with_capacity(capacity)
    }

    /// Creates a new ExtensionFieldOpenings with the given capacity.
    #[must_use]
    pub fn create_extension_field_openings<F, EF, const DIGEST_ELEMS: usize>(
        capacity: usize,
    ) -> ExtensionFieldOpenings<F, EF, DIGEST_ELEMS>
    where
        F: Field,
        EF: ExtensionField<F>,
    {
        ExtensionFieldOpenings::with_capacity(capacity)
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
