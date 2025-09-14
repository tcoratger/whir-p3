//! # Round State Management for WHIR Protocol
//!
//! This module provides round state management for the WHIR protocol.

pub mod state;

#[cfg(test)]
mod tests;

pub use state::RoundState;
