# Open numerical parity investigations

These source-bound observations narrow the remaining work; none closes full
function or example parity. Native reference is Chebfun 7574c77.

| Path | Evidence | Next decision |
|---|---|---|
| Fixed48 PDE | Native-built initial coefficients succeed in both runtimes; Python-built coefficients fail in both. Sample nodes/values match bits;10of22coefficients differ by at most1.11e-16 after transform. | Resolve transform fidelity or establish another source fault. Preserve the authored stability/decay test. |
| Complex rational fit | Same302nodes/samples. Matched nativeD/f product and mirrored602×151buffer are bit-identical to JAX; sameinput normalized inverse FFT differs1.11e-16. Public native output neutrality confirmed. Assembly amplification leaves poles about1.04e-4 apart. | Establish general transform execution/normalization arithmetic with independent-input validation. Product rewrite, conjugacy removal and tolerance workarounds are unsupported. |
| Small Hermite ASY | Source hyperbolic arithmetic did not close retained trig differences. A bounded observer recorded no helper entries inside its stage markers. | Establish the executed primitive route; zero stage records are not a provider oracle. |

[Detailed bindings and limitations](transform_sensitive_diagnostics_cpu_20261017.json)
and [rational stage report](ratinterp_complex_stages_cpu_20261017.json) retain
archive hashes, source versions and diagnostic limits. Native intermediate
crossfeeds are diagnostics, never production substitutions. No tolerances were
changed and no final page, figure or performance parity follows.

The [sphere extrema check](sphere_extrema_tie_cpu_20261017.json) is a separate
passing scope: all10original optimization predicates and two authored even
polynomial controls pass. Both maximum antipodes agree. The handoff did not
specify the historical even-function operand, so that broader claim remains
unqualified. Different minima on a continuum are not evidence of an error.

[Pre-IFFT causal follow-up](ratinterp_preifft_cpu_20261017.json) excludes
product arithmetic and mirroring for the retained complex fit. The gap remains
open; diagnostic native crossfeeds are never production substitutions.
