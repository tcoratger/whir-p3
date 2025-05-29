use p3_keccak::KeccakF;

use super::duplex_sponge::DuplexSponge;

/// A duplex sponge based on Keccak
pub type Keccak = DuplexSponge<u8, KeccakF>;
