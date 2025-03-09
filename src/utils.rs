/// Returns true if `n`:
/// - is a power of two
/// - is greater than zero
pub const fn is_power_of_two(n: usize) -> bool {
    n != 0 && n.is_power_of_two()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_power_of_two() {
        assert!(!is_power_of_two(0));
        assert!(is_power_of_two(1));
        assert!(is_power_of_two(2));
        assert!(!is_power_of_two(3));
        assert!(!is_power_of_two(usize::MAX));
    }
}
