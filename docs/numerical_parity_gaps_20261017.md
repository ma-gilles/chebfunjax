# Open numerical parity investigations

These source-bound observations narrow the remaining work; none closes full
function or example parity. Native reference is Chebfun 7574c77.

| Path | Evidence | Next decision |
|---|---|---|
| Fixed48 PDE | Published scalar33 real CPU transform repair makes the unchanged authored fixed48 test pass;21parent controls qualify that scope. [Evidence](fft64_pde_cpu_20261017.json). | Broader FFT/PDE plans and latest-main CI remain required. |
| Complex rational fit | Published FFT602/C2R integration qualifies129final-source controls and exactD/V/Z assembly. Pole gap is2.076e-11; denominator gap5.739e-12. [Evidence](fft602_rational_integration_cpu_20261017.json). | Fresh unchanged ThreeBody415coefficients equal retained input; full run104.57s,1380imports preguarded. Same-input JAX default/QR SVD both retain5.739e-12vector discrepancy. Full native ODE/second-fit/figure/performance parity remains open. |
| Dense Chebfun3T constructor7 | Fresh47a current-source original assertion fails2.864e-14 > unchanged100eps2.220e-14, shape249/259/263;34.48s, stable/no survivors. [Evidence](chebfun3t_current_transform_cpu_20261017.json). | Shared inverse-normalization change did not fix FFT/envelope/chop sensitivity. Strict xfail retained; matched-source diagnosis needed, no fitted cutoff or wider bound. |
| Small Hermite ASY | Later diagnostics pass n2/5/12/16/20 scoped bounds; n3 preserves native source error. n4/6 fail and n8/10 exceed retained phase support. [Review](/scratch/gpfs/CRYOEM/gilleslab/mg6942/_agent_scratch/hermite_small_scope_review_20261017/REVIEW.json). | General gamma/Miller diagnostics improve n4 but leave node failure; derivative/Newton ordering remains under investigation. Production small-order guard retained. |

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

## Historical complex602 inverse diagnostic — superseded by integration below

The complete JAX7x2x43inverse now matches all367220native outputwords across
positive, actual602x151 and independently generated602x151 matrices. Outer
radix2/7arithmetic and independent static constants are qualified; normalization
must divide by602 rather than multiply by its rounded reciprocal. This is a
diagnostic component, with no production patch or full pole acceptance. The production
real2x7x43route constructing D302still differs up to4.44e-16; later diagnostic
qualification below reproduces it exactly, with integration still required.
[Full evidence and preserved failed attempts](ratinterp_full602_cpu_20261017.json).

## Historical real602 and D302 diagnostic — superseded by integration below

The source-derived real602FFT matches275424nonredundant native outputwords
across positive, actual and independent arrays. All690captured stagewords
match, and sourceD302construction matches all91204words. Both real and complex
transform diagnostics are now qualified on these inputs. Production integration,
exact final assembly/public poles and independent API/AD/performance controls
remain required. [Evidence](ratinterp_real602_cpu_20261017.json).

## Production integration and fresh ThreeBody execution

[Production integration](fft602_rational_integration_cpu_20261017.json) supersedes
the diagnostic-only production-status statements above. The unchanged complete
[ThreeBody rerun](threebody_fft602_current_cpu_20261017.json) reproduces both
rational fits and all four figures. All reference dimensions match; all four
reference pixel comparisons differ. Five printed output blocks use actual
captured stdout. The two changed PNGs and current numerical outputs are
published. This does not close full native ODE, second-fit residue, renderer
or matched native performance parity.

## Real-even C2R diagnostics — still open

[Source-bound comparison](transform_c2r_diagnostics_cpu_20261017.json) rejects
a production patch from these observations. Dense363cubed C2R matches fewer
native coefficient words; both crops fail100eps on the normalized diagnostic
grid. This does not replace the literal original constructor failure receipt.
Inverse C2R matches native8630coefficient count while raw coefficient agreement
is not improved. Native724/32768plans and fresh packed profiling are next.
