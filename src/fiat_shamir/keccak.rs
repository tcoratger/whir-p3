use p3_keccak::KeccakF;

use super::duplex_sponge::DuplexSponge;

/// Width of the Keccak-f1600 sponge (in bytes)
pub const KECCAK_WIDTH_BYTES: usize = 200;
/// Rate of the sponge (bytes): 136
pub const KECCAK_RATE_BYTES: usize = 136;

/// A duplex sponge based on Keccak
pub type Keccak = DuplexSponge<u8, KeccakF, KECCAK_WIDTH_BYTES, KECCAK_RATE_BYTES>;
