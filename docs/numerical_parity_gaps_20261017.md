# Open numerical parity investigations

These source-bound observations narrow the remaining work; none closes full
function or example parity. Native reference is Chebfun 7574c77.

| Path | Evidence | Next decision |
|---|---|---|
| Fixed48 PDE | Published scalar33 real CPU transform repair makes the unchanged authored fixed48 test pass;21parent controls qualify that scope. [Evidence](fft64_pde_cpu_20261017.json). | Broader FFT/PDE plans and latest-main CI remain required. |
| Complex rational fit | Production poles still differ about1.04e-4. New complete JAX602inverse diagnostic matches367220native words; production transform is unchanged. [Evidence](ratinterp_full602_cpu_20261017.json). | Real602diagnostic now reproduces D302exactly; integrate both source transforms and verify public coefficients/poles plus independent controls. |
| Small Hermite ASY | Later executed-route diagnostics qualify initializer/power/atan and Airy product/sqrt arithmetic. Free n2 improves to x8.98e-9/v4.18e-8 but still fails. [Sealed diagnostic review](/scratch/gpfs/CRYOEM/gilleslab/mg6942/_agent_scratch/hermite_airy_step8_review_20261017/REVIEW.json). | Resolve same-theta step9 geometry and remaining free trajectory; no guard removal or production Airy patch yet. |

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

## Complete complex602 inverse diagnostic — later 2026-10-17

The complete JAX7x2x43inverse now matches all367220native outputwords across
positive, actual602x151 and independently generated602x151 matrices. Outer
radix2/7arithmetic and independent static constants are qualified; normalization
must divide by602 rather than multiply by its rounded reciprocal. This is a
diagnostic component, with no production patch or full pole acceptance. The production
real2x7x43route constructing D302still differs up to4.44e-16; later diagnostic
qualification below reproduces it exactly, with integration still required.
[Full evidence and preserved failed attempts](ratinterp_full602_cpu_20261017.json).

## Real602 and D302 diagnostic — later 2026-10-17

The source-derived real602FFT matches275424nonredundant native outputwords
across positive, actual and independent arrays. All690captured stagewords
match, and sourceD302construction matches all91204words. Both real and complex
transform diagnostics are now qualified on these inputs. Production integration,
exact final assembly/public poles and independent API/AD/performance controls
remain required. [Evidence](ratinterp_real602_cpu_20261017.json).
