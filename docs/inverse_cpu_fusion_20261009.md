# CPU inverse: two-phase roots fusion

The default roots adapter now computes the matrix and its finite flag together,
checks that flag on the host **before eig**, then computes eigenvalues and their
finite/spectrum metadata together. The original matrix arithmetic, optimization
barriers and eig helper bodies are unchanged. Output order, dtype restoration,
error handling and owned host copies are preserved.

## Matched observation

Baseline `9b4679ad` and the isolated candidate ran in separate fresh processes on
CPU 8–15, x64, with the same frozen flower input and synchronized profiling
harness. One process per arm limits conclusions about variability.

| Instrumented interval | Baseline | Candidate |
| --- | ---: | ---: |
| Setup | 19.3211 s | 19.5138 s |
| First inverse after setup | 24.4378 s | 17.5747 s |
| Median of five warm inverses | 0.614889 s | 0.608158 s |
| Warm range | 0.613452–0.617816 s | 0.606770–0.609122 s |

The observed first-after-setup ratio is **1.3905×**. Setup itself invokes roots
and compilation, so this is not pristine process-cold timing. The roughly 1.1%
warm-median difference does not establish a broad warm-speed improvement.
First-inverse backend compilation calls fell from 544 to 370, exactly 174 fewer;
all five warm calls in both arms had no compiler events. Profiling intervals
overlap and must not be summed as independent costs.

All 80 corresponding binary captures per arm were identical in shape, dtype
and raw bytes, including forward/derivative coefficients, normalized root
inputs, targets, Brent results, inverse coefficients and domains. Forward and
inverse lengths remained 2090 and 8880. Actual Brent target sizes were
`2, 4097, 4096, 8192, 2`; both roundtrip errors were
`1.0547118733938987e-15` at the unchanged `1e-10` check.

## Qualification and limits

- 45 helper/IR/adapter controls and 30 unchanged native/regression cases passed.
- Nonempty real/complex roots, unsorted provider order, exact represented-input
  matrix arithmetic, and failures before eig/filtering were tested.
- Each timing arm passed its runtime audit with 2964 observed file hashes and
  no surviving owned processes. Peak summed RSS was 2428724 and 2174620 KiB.
- Root independently reviewed runtime bindings, raw captures and optimized IR.
- The agent quiet window did not reserve an exclusive host. Profiling and
  synchronization overhead were identical between arms.
- No current matched MATLAB timing is available. This result does not establish
  full native parity or revive the historical 800× comparison.

[Machine-readable timings and immutable qualification links](inverse_cpu_fusion_20261009.json)
include the raw warm timings and the prior 45+30 qualification packets.
The earlier [CPU profile](inverse_cpu_profile_20261009.md) remains a historical
measurement of the unfused implementation.
