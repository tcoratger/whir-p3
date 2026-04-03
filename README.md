# whir-p3

A version of https://github.com/WizardOfMenlo/whir/ which uses the Plonky3 library.

## Sumcheck benchmark comparison

The repository includes a unified benchmark for comparing the classical sumcheck prover
against the SVO implementation ported from the Jolt approach:

```bash
cargo bench --bench sumcheck_compare -- --noplot
```

This comparison uses the same setup for both implementations:

- `BabyBear` field
- the same random multilinear polynomial
- the same 3 sampled evaluation constraints
- the same challenger initialization
- the same first-round folding parameter `l0 = 4`

Criterion midpoint timings from the unified benchmark:

| num_vars | Classic | Svo | Speedup |
| --- | ---: | ---: | ---: |
| 16 | 1.3647 ms | 1.1459 ms | 1.19x |
| 18 | 3.1021 ms | 1.9203 ms | 1.62x |
| 20 | 8.0524 ms | 3.6310 ms | 2.22x |
| 22 | 23.991 ms | 8.3096 ms | 2.89x |
| 24 | 86.095 ms | 21.958 ms | 3.92x |

The main takeaway is that the SVO/Jolt-style path becomes increasingly faster than the
classical prover as the number of variables grows.
