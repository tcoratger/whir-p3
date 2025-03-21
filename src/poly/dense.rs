use p3_field::Field;
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSlice,
};
use std::cmp::max;

// Set some minimum number of field elements to be worked on per thread
// to avoid per-thread costs dominating parallel execution time.
const MIN_ELEMENTS_PER_THREAD: usize = 16;

/// Stores a polynomial in coefficient form.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct DensePolynomial<F: Field> {
    /// The coefficient of `x^i` is stored at location `i` in `self.coeffs`.
    pub coeffs: Vec<F>,
}

impl<F: Field> DensePolynomial<F> {
    /// Constructs a new polynomial from a list of coefficients.
    pub fn from_coefficients_slice(coeffs: &[F]) -> Self {
        Self::from_coefficients_vec(coeffs.to_vec())
    }

    /// Constructs a new polynomial from a list of coefficients.
    fn from_coefficients_vec(coeffs: Vec<F>) -> Self {
        let mut result = Self { coeffs };
        // While there are zeros at the end of the coefficient vector, pop them off.
        result.truncate_leading_zeros();
        // Check that either the coefficients vec is empty or that the last coeff is
        // non-zero.
        assert!(result.coeffs.last().map_or(true, |coeff| !coeff.is_zero()));
        result
    }

    fn truncate_leading_zeros(&mut self) {
        while self.coeffs.last().is_some_and(|c| c.is_zero()) {
            self.coeffs.pop();
        }
    }

    /// Checks if the given polynomial is zero.
    fn is_zero(&self) -> bool {
        self.coeffs.is_empty() || self.coeffs.iter().all(|coeff| coeff.is_zero())
    }

    /// Evaluates `self` at the given `point` in `Self::Point`.
    pub fn evaluate(&self, point: &F) -> F {
        if self.is_zero() {
            return F::ZERO;
        } else if point.is_zero() {
            return self.coeffs[0];
        }
        self.internal_evaluate(point)
    }

    fn internal_evaluate(&self, point: &F) -> F {
        // Horners method - parallel method
        // compute the number of threads we will be using.
        let num_cpus_available = rayon::current_num_threads();
        let num_coeffs = self.coeffs.len();
        let num_elem_per_thread = max(num_coeffs / num_cpus_available, MIN_ELEMENTS_PER_THREAD);

        // run Horners method on each thread as follows:
        // 1) Split up the coefficients across each thread evenly.
        // 2) Do polynomial evaluation via horner's method for the thread's coefficients
        // 3) Scale the result point^{thread coefficient start index}
        // Then obtain the final polynomial evaluation by summing each threads result.
        self.coeffs
            .par_chunks(num_elem_per_thread)
            .enumerate()
            .map(|(i, chunk)| {
                Self::horner_evaluate(chunk, point) *
                    point.exp_u64((i * num_elem_per_thread) as u64)
            })
            .sum()
    }

    #[inline]
    // Horner's method for polynomial evaluation
    fn horner_evaluate(poly_coeffs: &[F], point: &F) -> F {
        poly_coeffs.iter().rfold(F::ZERO, move |result, &coeff| result * *point + coeff)
    }
}
