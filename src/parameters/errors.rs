use alloc::{format, string::String};
use core::{f64::consts::LOG2_10, fmt::Display, str::FromStr};

use serde::Serialize;

/// Security assumptions determines which proximity parameters and conjectures are assumed by the error computation.
///
/// # References
///
/// The proximity gaps analysis is based on:
/// - **[BCI+20]**: Ben-Sasson, Carmon, Ishai, Kopparty, Saraf. "Proximity Gaps for Reed–Solomon Codes".
///   FOCS 2020. <https://eprint.iacr.org/2020/654>
/// - **[BCSS25]**: Ben-Sasson, Carmon, Haböck, Kopparty, Saraf. "On Proximity Gaps for Reed–Solomon Codes".
///   <https://eprint.iacr.org/2025/2055>
///
/// The [BCSS25] paper significantly improves the Johnson bound proximity gaps from `O(n^2/η^7)` to `O(n/η^5)`,
/// enabling provable 128-bit security with smaller extension fields (e.g., degree-5 extension of KoalaBear).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum SecurityAssumption {
    /// Unique decoding assumes that the distance of each oracle is within the UDR of the code.
    /// We refer to this configuration as UD for short.
    /// This requires no conjectures.
    UniqueDecoding,

    /// Johnson bound assumes that the distance of each oracle is within the Johnson bound (1 - √ρ).
    /// We refer to this configuration as JB for short.
    /// This assumes that RS have mutual correlated agreement for proximity parameter up to (1 - √ρ).
    ///
    /// # Proximity Gaps Improvement
    ///
    /// The error bound uses Theorem 1.5 from [BCSS25]:
    /// - **Old [BCI+20]**: `a = O(n^2/η^7)` exceptional z's
    /// - **New [BCSS25]**: `a = O(n/η^5)` exceptional z's
    ///
    /// This improvement of factor `n·η^2` translates to approximately `log_2(n)` additional bits of security,
    /// making provable 128-bit security achievable with degree-5 extensions of small prime fields.
    JohnsonBound,

    /// Capacity bound assumes that the distance of each oracle is within the capacity bound 1 - ρ.
    /// We refer to this configuration as CB for short.
    /// This requires conjecturing that RS codes are decodable up to capacity and have correlated agreement (mutual in WHIR) up to capacity.
    CapacityBound,
}

impl SecurityAssumption {
    /// In both JB and CB theorems such as list-size only hold for proximity parameters slighly below the bound.
    /// E.g. in JB proximity gaps holds for every δ ∈ (0, 1 - √ρ).
    /// η is the distance between the chosen proximity parameter and the bound.
    /// I.e. in JB δ = 1 - √ρ - η and in CB δ = 1 - ρ - η.
    // TODO: Maybe it makes more sense to be multiplicative. I think this can be set in a better way.
    #[must_use]
    pub const fn log_eta(&self, log_inv_rate: usize) -> f64 {
        match self {
            // We don't use η in UD
            Self::UniqueDecoding => 0., // TODO: Maybe just panic and avoid calling it in UD?
            // Set as √ρ/20
            Self::JohnsonBound => -(0.5 * log_inv_rate as f64 + LOG2_10 + 1.),
            // Set as ρ/20
            Self::CapacityBound => -(log_inv_rate as f64 + LOG2_10 + 1.),
        }
    }

    /// Given a RS code (specified by the log of the degree and log inv of the rate), compute the list size at the specified distance δ.
    #[must_use]
    pub const fn list_size_bits(&self, log_degree: usize, log_inv_rate: usize) -> f64 {
        let log_eta = self.log_eta(log_inv_rate);
        match self {
            // In UD the list size is 1
            Self::UniqueDecoding => 0.,

            // By the JB, RS codes are (1 - √ρ - η, (2*η*√ρ)^-1)-list decodable.
            Self::JohnsonBound => {
                let log_inv_sqrt_rate: f64 = log_inv_rate as f64 / 2.;
                log_inv_sqrt_rate - (1. + log_eta)
            }

            // In CB we assume that RS codes are (1 - ρ - η, d/ρ*η)-list decodable (see Conjecture 5.6 in STIR).
            Self::CapacityBound => (log_degree + log_inv_rate) as f64 - log_eta,
        }
    }

    /// Given a RS code (specified by the log of the degree and log inv of the rate) a field_size
    /// and an arity, compute the proximity gaps error (in bits) at the specified distance.
    ///
    /// # Johnson Bound Improvement
    ///
    /// For the Johnson bound case, this uses the improved Theorem 1.5 from [BCSS25]:
    ///
    /// > "On Proximity Gaps for Reed–Solomon Codes" (eprint 2025/2055)
    /// > Ben-Sasson, Carmon, Haböck, Kopparty, Saraf
    ///
    /// The theorem states that for proximity parameter `γ < J(δ) - η` (below Johnson bound by η),
    /// the number of exceptional z's satisfies:
    ///
    /// ```text
    /// a > (2(m + 1/2)^5 + 3(m + 1/2)·γ·ρ) / (3·ρ^(3/2)) · n + (m + 1/2) / √ρ
    /// ```
    ///
    /// where `m = max(⌈√ρ / (2η)⌉, 3)` and the asymptotic behavior is `O(n/η^5)`.
    ///
    /// ## Comparison with prior work
    ///
    /// | Reference | Bound on exceptional z's | Proximity loss |
    /// |-----------|--------------------------|----------------|
    /// | [BCI+20]  | `O(n^2/η^7)`             | 0              |
    /// | [BCSS25]  | `O(n/η^5)`               | 0              |
    ///
    /// The improvement factor of `n·η^2` translates to approximately `log_2(n)` additional bits
    /// of provable security, enabling 128-bit security with degree-5 extensions of KoalaBear.
    #[must_use]
    pub fn prox_gaps_error(
        &self,
        log_degree: usize,
        log_inv_rate: usize,
        field_size_bits: usize,
        num_functions: usize,
    ) -> f64 {
        assert!(
            num_functions >= 2,
            "num_functions must be >= 2 to compute proximity gaps error",
        );

        let log_eta = self.log_eta(log_inv_rate);

        // Note that this does not include the field_size
        let error = match self {
            // In UD the error is |L|/|F| = d/(ρ·|F|)
            Self::UniqueDecoding => (log_degree + log_inv_rate) as f64,

            // From Theorem 1.5 in [BCSS25] "On Proximity Gaps for Reed–Solomon Codes":
            //
            // For γ < J(δ) - η, the number of exceptional z's is bounded by:
            //   a > (2(m + 1/2)^5 + 3(m + 1/2)·γ·ρ) / (3·ρ^(3/2)) · n + (m + 1/2) / √ρ
            //
            // With η = √ρ/20 (safe gap), m = max(⌈√ρ/(2η)⌉, 3) = max(10, 3) = 10.
            //
            // The dominant term for large n:
            //   a ≈ (2 · 10.5^5) / (3 · ρ^(3/2)) · n
            //
            // In log form:
            //   log_2(a) = log_2(n) + log_2(2 · 10.5^5 / 3) + 1.5 · log_2(1/ρ)
            //            = (log_degree + log_inv_rate) + 16.38 + 1.5 · log_inv_rate
            //            = log_degree + 2.5 · log_inv_rate + 16.38
            //
            // This improves over [BCI+20] which had:
            //   log_2(a) = 2·log_degree + 3.5·log_inv_rate + 23.24
            Self::JohnsonBound => {
                // n = 2^(log_degree + log_inv_rate)
                let log_n = (log_degree + log_inv_rate) as f64;

                // Constant from (2 · 10.5^5 / 3)
                let constant = libm::log2(2. * libm::pow(10.5, 5.) / 3.);

                // ρ^(-3/2) contributes 1.5 · log_inv_rate
                let log_rho_neg_3_2 = 1.5 * log_inv_rate as f64;

                log_n + constant + log_rho_neg_3_2
            }

            // In CB we assume the error is degree/(η·ρ^2)
            Self::CapacityBound => (log_degree + 2 * log_inv_rate) as f64 - log_eta,
        };

        // Error is (num_functions - 1) * error/|F|;
        let num_functions_1_log = libm::log2(num_functions as f64 - 1.);
        field_size_bits as f64 - (error + num_functions_1_log)
    }

    /// The query error is (1 - δ)^t where t is the number of queries.
    /// This computes log(1 - δ).
    /// - In UD, δ is (1 - ρ)/2
    /// - In JB, δ is (1 - √ρ - η)
    /// - In CB, δ is (1 - ρ - η)
    #[must_use]
    pub fn log_1_delta(&self, log_inv_rate: usize) -> f64 {
        let log_eta = self.log_eta(log_inv_rate);
        let eta = libm::pow(2., log_eta);
        let rate = 1. / f64::from(1 << log_inv_rate);

        let delta = match self {
            Self::UniqueDecoding => 0.5 * (1. - rate),
            Self::JohnsonBound => 1. - libm::sqrt(rate) - eta,
            Self::CapacityBound => 1. - rate - eta,
        };

        libm::log2(1. - delta)
    }

    /// Compute the number of queries to match the security level
    /// The error to drive down is (1-δ)^t < 2^-λ.
    /// Where δ is set as in the `log_1_delta` function.
    #[must_use]
    pub fn queries(&self, protocol_security_level: usize, log_inv_rate: usize) -> usize {
        let num_queries_f = -(protocol_security_level as f64) / self.log_1_delta(log_inv_rate);

        libm::ceil(num_queries_f) as usize
    }

    /// Compute the error for the given number of queries
    /// The error to drive down is (1-δ)^t < 2^-λ.
    /// Where δ is set as in the `log_1_delta` function.
    #[must_use]
    pub fn queries_error(&self, log_inv_rate: usize, num_queries: usize) -> f64 {
        let num_queries = num_queries as f64;

        -num_queries * self.log_1_delta(log_inv_rate)
    }

    /// Compute the error for the OOD samples of the protocol
    /// See Lemma 4.5 in STIR.
    /// The error is list_size^2 * (degree/field_size_bits)^reps
    /// NOTE: Here we are discounting the domain size as we assume it is negligible compared to the size of the field.
    #[must_use]
    pub const fn ood_error(
        &self,
        log_degree: usize,
        log_inv_rate: usize,
        field_size_bits: usize,
        ood_samples: usize,
    ) -> f64 {
        if matches!(self, Self::UniqueDecoding) {
            return 0.;
        }

        let list_size_bits = self.list_size_bits(log_degree, log_inv_rate);

        let error = 2. * list_size_bits + (log_degree * ood_samples) as f64;
        (ood_samples * field_size_bits) as f64 + 1. - error
    }

    /// Computes the number of OOD samples required to achieve security_level bits of security
    /// We note that in both STIR and WHIR there are various strategies to set OOD samples.
    /// In this case, we are just sampling one element from the extension field
    #[must_use]
    pub fn determine_ood_samples(
        &self,
        security_level: usize,
        log_degree: usize,
        log_inv_rate: usize,
        field_size_bits: usize,
    ) -> usize {
        if matches!(self, Self::UniqueDecoding) {
            return 0;
        }

        for ood_samples in 1..64 {
            if self.ood_error(log_degree, log_inv_rate, field_size_bits, ood_samples)
                >= security_level as f64
            {
                return ood_samples;
            }
        }

        panic!("Could not find an appropriate number of OOD samples");
    }
}

impl Display for SecurityAssumption {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::JohnsonBound => "JohnsonBound",
            Self::CapacityBound => "CapacityBound",
            Self::UniqueDecoding => "UniqueDecoding",
        })
    }
}

impl FromStr for SecurityAssumption {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "JohnsonBound" => Ok(Self::JohnsonBound),
            "CapacityBound" => Ok(Self::CapacityBound),
            "UniqueDecoding" => Ok(Self::UniqueDecoding),
            _ => Err(format!("Invalid soundness specification: {s}")),
        }
    }
}

#[cfg(test)]
#[allow(clippy::cast_lossless)]
mod tests {
    use alloc::string::ToString;

    use super::*;

    #[test]
    fn test_soundness_type_display() {
        assert_eq!(SecurityAssumption::JohnsonBound.to_string(), "JohnsonBound");
        assert_eq!(
            SecurityAssumption::CapacityBound.to_string(),
            "CapacityBound"
        );
        assert_eq!(
            SecurityAssumption::UniqueDecoding.to_string(),
            "UniqueDecoding"
        );
    }

    #[test]
    fn test_soundness_type_from_str() {
        assert_eq!(
            SecurityAssumption::from_str("JohnsonBound"),
            Ok(SecurityAssumption::JohnsonBound)
        );
        assert_eq!(
            SecurityAssumption::from_str("CapacityBound"),
            Ok(SecurityAssumption::CapacityBound)
        );
        assert_eq!(
            SecurityAssumption::from_str("UniqueDecoding"),
            Ok(SecurityAssumption::UniqueDecoding)
        );

        // Invalid cases
        assert!(SecurityAssumption::from_str("InvalidType").is_err());
        assert!(SecurityAssumption::from_str("").is_err()); // Empty string
    }

    #[test]
    #[should_panic(expected = "num_functions must be >= 2")]
    fn prox_gaps_error_panics_when_num_functions_is_one() {
        let assumption = SecurityAssumption::UniqueDecoding;
        let _ = assumption.prox_gaps_error(1, 1, 64, 1);
    }

    #[test]
    #[should_panic(expected = "num_functions must be >= 2")]
    fn prox_gaps_error_panics_when_num_functions_is_zero() {
        let assumption = SecurityAssumption::UniqueDecoding;
        let _ = assumption.prox_gaps_error(1, 1, 64, 0);
    }

    #[test]
    fn test_ud_errors() {
        let assumption = SecurityAssumption::UniqueDecoding;

        // Setting
        let log_degree = 20;
        let degree = (1 << log_degree) as f64;
        let log_inv_rate = 2;
        let rate = 1. / (1 << log_inv_rate) as f64;

        let field_size_bits = 128;

        // List size
        assert!(assumption.list_size_bits(log_degree, log_inv_rate) - 0. < 0.01);

        // Prox gaps
        let computed_error =
            assumption.prox_gaps_error(log_degree, log_inv_rate, field_size_bits, 2);
        let real_error_non_log = degree / rate;
        let real_error = field_size_bits as f64 - real_error_non_log.log2();

        assert!((computed_error - real_error).abs() < 0.01);
    }

    #[test]
    fn test_jb_errors() {
        let assumption = SecurityAssumption::JohnsonBound;

        // Setting
        let log_degree = 20;
        let log_inv_rate = 2;
        let rate = 1. / (1 << log_inv_rate) as f64;

        let eta = rate.sqrt() / 20.;

        let field_size_bits = 128;

        // List size (unchanged from before)
        let real_list_size = 1. / (2. * eta * rate.sqrt());
        let computed_list_size = assumption.list_size_bits(log_degree, log_inv_rate);
        assert!((real_list_size.log2() - computed_list_size).abs() < 0.01);

        // Prox gaps - Updated to use Theorem 1.5 from [BCSS25]
        //
        // From "On Proximity Gaps for Reed–Solomon Codes" (eprint 2025/2055):
        // With η = √ρ/20, m = 10, the error bound is:
        //   a ≈ (2 · 10.5^5) / (3 · ρ^(3/2)) · n
        //
        // where n = 2^(log_degree + log_inv_rate)
        let computed_error =
            assumption.prox_gaps_error(log_degree, log_inv_rate, field_size_bits, 2);

        // n = 2^(log_degree + log_inv_rate) = 2^22
        let n = (1_u64 << (log_degree + log_inv_rate)) as f64;
        // ρ = rate = 2^(-log_inv_rate) = 0.25
        let rho = rate;
        // Constant from Theorem 1.5: (2 · 10.5^5) / 3 ≈ 85085.44
        let constant = 2. * 10.5_f64.powi(5) / 3.;
        // a ≈ constant · n / ρ^(3/2)
        let real_error_non_log = constant * n / rho.powf(1.5);
        let real_error = field_size_bits as f64 - real_error_non_log.log2();

        assert!(
            (computed_error - real_error).abs() < 0.01,
            "computed: {computed_error}, expected: {real_error}"
        );
    }

    #[test]
    fn test_cb_errors() {
        let assumption = SecurityAssumption::CapacityBound;

        // Setting
        let log_degree = 20;
        let degree = (1 << log_degree) as f64;
        let log_inv_rate = 2;
        let rate = 1. / (1 << log_inv_rate) as f64;

        let eta = rate / 20.;

        let field_size_bits = 128;

        // List size
        let real_list_size = degree / (rate * eta);
        let computed_list_size = assumption.list_size_bits(log_degree, log_inv_rate);
        assert!((real_list_size.log2() - computed_list_size).abs() < 0.01);

        // Prox gaps
        let computed_error =
            assumption.prox_gaps_error(log_degree, log_inv_rate, field_size_bits, 2);
        let real_error_non_log = degree / (eta * rate.powi(2));
        let real_error = field_size_bits as f64 - real_error_non_log.log2();

        assert!((computed_error - real_error).abs() < 0.01);
    }
}
