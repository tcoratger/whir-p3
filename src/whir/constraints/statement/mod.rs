/// Equality statement for polynomial evaluation constraints.
pub mod eq;

// pub mod select;

/// Selection statement for conditional constraints.
pub mod domain;

// Re-export main types for convenient access.
pub use domain::DomainStatement;
pub use eq::EqStatement;
