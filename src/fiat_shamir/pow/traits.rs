pub trait PowStrategy: Clone + Sync {
    /// Creates a new proof-of-work challenge.
    /// The `challenge` is a 32-byte array that represents the challenge.
    /// The `bits` is the binary logarithm of the expected amount of work.
    /// When `bits` is large (i.e. close to 64), a valid solution may not be found.
    fn new(challenge: [u8; 32], bits: f64) -> Self;

    /// Check if the `nonce` satisfies the challenge.
    fn check(&mut self, nonce: u64) -> bool;

    /// Finds the minimal `nonce` that satisfies the challenge.
    fn solve(&mut self) -> Option<u64>;
}
