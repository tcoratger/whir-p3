/// A query to a multilinear polynomial over an extension field.
///
/// This enum represents the type of query made to a multilinear polynomial.
/// It supports:
/// - Exact evaluation at a point (`Eq`)
/// - Evaluation at a rotated version of a point (`EqRotateRight`)
///
/// These queries are typically used in polynomial commitment schemes for verifying
/// proximity to multilinear codes or enforcing algebraic constraints.
///
/// # Type Parameters
/// - `Challenge`: The field element type used in queries (typically an extension field).
#[derive(Clone, Debug)]
pub enum MlQuery<Challenge> {
    /// A standard query asking for evaluation of the polynomial at the given point.
    ///
    /// The point is represented as a vector of field elements corresponding to the
    /// $m$-dimensional input of a multilinear function $\hat{f} : \{0,1\}^m \to F$.
    ///
    /// # Example
    /// ```text
    /// Eq([α_0, α_1, α_2]) → query at (α_0, α_1, α_2)
    /// ```
    Eq(Vec<Challenge>),

    /// A rotated query: evaluate the polynomial at a rotated version of the point.
    ///
    /// This is useful when enforcing symmetries or constraints that involve bit rotations.
    /// The second argument `r` indicates a right rotation by `r` bits of the input vector.
    ///
    /// The rotation is **circular** (wraps around), and is applied *before* evaluation.
    ///
    /// # Example
    /// ```text
    /// EqRotateRight([α_0, α_1, α_2], 1) → query at (α_2, α_0, α_1)
    /// ```
    EqRotateRight(Vec<Challenge>, usize),
}
