/// Equality statement for polynomial evaluation constraints.
pub mod eq;
pub mod initial;

/// Selection statement for conditional constraints.
pub mod select;

// Re-export main types for convenient access.
pub use eq::EqStatement;
pub use select::SelectStatement;
