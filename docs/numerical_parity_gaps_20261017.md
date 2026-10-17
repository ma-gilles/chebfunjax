# Open numerical parity investigations

These source-bound observations narrow the remaining work; none closes full
function or example parity. Native reference is Chebfun 7574c77.

| Path | Evidence | Next decision |
|---|---|---|
| Fixed48 PDE | Native-built initial coefficients succeed in both runtimes; Python-built coefficients fail in both. Sample nodes/values match bits;10of22coefficients differ by at most1.11e-16 after transform. | Resolve transform fidelity or establish another source fault. Preserve the authored stability/decay test. |
| Complex rational fit | Same302nodes and complex samples; assembly differs1.69e-16 and poles about1.04e-4. Native matrix fed to JAX sharply reduces denominator discrepancy. | A pre-IFFT native product capture would separate multiplication from transform arithmetic. No fitting/roots rewrite is justified yet. |
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
