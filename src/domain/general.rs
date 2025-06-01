use p3_field::{Field, TwoAdicField};

use super::radix2::Radix2EvaluationDomain;

/// Defines a domain over which finite field (I)FFTs can be performed.
///
/// Generally tries to build a radix-2 domain and falls back to a mixed-radix
/// domain if the radix-2 multiplicative subgroup is too small.
#[derive(Copy, Clone, Hash, Eq, PartialEq, Debug)]
pub enum GeneralEvaluationDomain<F> {
    /// Radix-2 domain
    Radix2(Radix2EvaluationDomain<F>),
}

impl<F: Field + TwoAdicField> GeneralEvaluationDomain<F> {
    /// Construct a domain that is large enough for evaluations of a polynomial
    /// having `num_coeffs` coefficients.
    ///
    /// If the field specifies a small subgroup for a mixed-radix FFT and
    /// the radix-2 FFT cannot be constructed, this method tries
    /// constructing a mixed-radix FFT instead.
    pub fn new(num_coeffs: usize) -> Option<Self> {
        Radix2EvaluationDomain::new(num_coeffs).map(Self::Radix2)
    }

    #[inline]
    pub const fn size(&self) -> usize {
        match self {
            Self::Radix2(domain) => domain.size(),
        }
    }

    #[inline]
    pub const fn group_gen(&self) -> F {
        match self {
            Self::Radix2(domain) => domain.group_gen(),
        }
    }

    #[inline]
    pub const fn group_gen_inv(&self) -> F {
        match self {
            Self::Radix2(domain) => domain.group_gen_inv(),
        }
    }

    #[inline]
    pub const fn log_size_of_group(&self) -> u32 {
        match self {
            Self::Radix2(domain) => domain.log_size_of_group(),
        }
    }

    #[inline]
    pub fn size_as_field_element(&self) -> F {
        F::from_u64(self.size() as u64)
    }

    #[inline]
    pub const fn size_inv(&self) -> F {
        match self {
            Self::Radix2(domain) => domain.size_inv(),
        }
    }

    #[inline]
    pub const fn coset_offset(&self) -> F {
        match self {
            Self::Radix2(domain) => domain.coset_offset(),
        }
    }

    #[inline]
    pub const fn coset_offset_inv(&self) -> F {
        match self {
            Self::Radix2(domain) => domain.coset_offset_inv(),
        }
    }

    #[inline]
    pub const fn coset_offset_pow_size(&self) -> F {
        match self {
            Self::Radix2(domain) => domain.coset_offset_pow_size(),
        }
    }

    #[inline]
    /// Returns the `i`-th element of the domain.
    pub fn element(&self, i: usize) -> F {
        let mut result = self.group_gen().exp_u64(i as u64);
        if !self.coset_offset().is_one() {
            result *= self.coset_offset();
        }
        result
    }
}
