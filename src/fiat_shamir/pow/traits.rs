pub trait PowStrategy: Clone + Sync {
    /// Creates a new proof-of-work challenge.
    /// The `challenge` is a 32-byte array that represents the challenge.
    /// The `bits` is the binary logarithm of the expected amount of work.
    /// When `bits` is large (i.e. close to 64), a valid solution may not be found.
    fn new(challenge: [u8; 32], bits: f64) -> Self;

    /// Check if the `nonce` satisfies the challenge.
    fn check(&mut self, nonce: u64) -> bool;

    /// Finds the minimal `nonce` that satisfies the challenge.
    #[cfg(not(feature = "parallel"))]
    fn solve(&mut self) -> Option<u64> {
        // TODO: Parallel default impl
        (0..=u64::MAX).find(|&nonce| self.check(nonce))
    }

    #[cfg(feature = "parallel")]
    fn solve(&mut self) -> Option<u64> {
        // Split the work across all available threads.
        // Use atomics to find the unique deterministic lowest satisfying nonce.

        use std::sync::atomic::{AtomicU64, Ordering};

        use rayon::broadcast;
        let global_min = AtomicU64::new(u64::MAX);
        let _ = broadcast(|ctx| {
            let mut worker = self.clone();
            let nonces = (ctx.index() as u64..).step_by(ctx.num_threads());
            for nonce in nonces {
                // Use relaxed ordering to eventually get notified of another thread's solution.
                // (Propagation delay should be in the order of tens of nanoseconds.)
                if nonce >= global_min.load(Ordering::Relaxed) {
                    break;
                }
                if worker.check(nonce) {
                    // We found a solution, store it in the global_min.
                    // Use fetch_min to solve race condition with simultaneous solutions.
                    global_min.fetch_min(nonce, Ordering::SeqCst);
                    break;
                }
            }
        });
        match global_min.load(Ordering::SeqCst) {
            u64::MAX => self.check(u64::MAX).then_some(u64::MAX),
            nonce => Some(nonce),
        }
    }
}
