//! STIR (Short Transparent Interactive Reed-Solomon) proof system for WHIR protocol.
//!
//! This module provides the complete STIR proof implementation, including query generation,
//! challenge computation, and verification support for Reed-Solomon proximity testing.

pub mod config;
pub mod handler;
pub mod openings;
pub mod queries;
pub mod types;
pub mod utils;

pub use config::StirConfig;
pub use handler::StirProofHandler;
pub use openings::{BaseFieldOpenings, ExtensionFieldOpenings};
pub use queries::{StirChallenges, StirQueryGenerator};
