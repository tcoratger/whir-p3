//! STIR proof handling module.

pub mod config;
pub mod handler;
pub mod openings;
pub mod types;
pub mod utils;

pub use config::StirConfig;
pub use handler::StirProofHandler;
pub use openings::{BaseFieldOpenings, ExtensionFieldOpenings};
