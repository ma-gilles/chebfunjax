# SIN output underflow and JAX numeric disk PhaseOne — 2026-10-18

SIN now preserves the separately rounded cos(real)*sinh(imag) product through
minimum normal, using bounded source cosine for |real|<=pi and primitive
fallback elsewhere. All904selected native imaginary words match eager/JIT;
all prior real/outside-mask words remain. Exact analytic JVP AST unchanged.
Two authored preservation oracles supersede exactly92expected words per mode
inside the approved extension mask; original324native assertions, original
fixtures and bounds unchanged. Child1,914PASS26XFAIL is a separate full cohort.
192outside imaginary word gaps remain, including72finite-native/JAX-infinite
cases near y=±710; other component/provider gaps are under separate diagnosis.

Numeric disk PhaseOne arithmetic now uses JAX IEEE helpers, literal reciprocal
GE staging and sum/divide mean. Tiny exact sums use a proved representability
bound, including signed-zero source controls. Full source pole indices now
reach public grid conversion, correcting tiny-origin pole location from
(-pi,0) to native(-2pi/3,0). Legacy15records and factor words in28revision
replays remain exact except4intended location corrections. Child319PASS3XFAIL
includes a final unchanged original k7PASS78.42s across separate scoped gates.
Three strict word gaps remain:2post-GE transform cases and1inactive outer
signed zero. Generic reductions/NaNpayloads/FFT/projection, getTol host
arithmetic, complete constructor/API/JIT/AD and matched speed remain open.

Actual combined parent387PASS3strictXFAIL across two disjoint processes:
69SIN/roots/CIprotocol and318diskPASS3XFAIL. All8477guarded inputs stable;
1542/1491observed imports preguarded/stable, no censor or survivors. Initial
69PASS wrapper run failed because cached JAX files were omitted from guard;
raw receipt/inventory retained, fresh corrected run passes. Import/semicolon
lint corrections preserve all non-import ASTs. Archive3593payloads streamed
and verified.1116old CI lane prefixes retained→1117owners/13lanes; dedicated
Hermite lane12 and aggregate79%gate unchanged. Earlier a9d5461aCI completed
all46jobs; this does not qualify newer source. Latest-mainCI remains required.

322final-sourcepages/1305scientificfigures, broader original function/assertion
coverage, correctness/API/JAX and matched CPU speed remain open. Expected
failures are unfinished work. Evidence: docs/sin_disk_jax_parent_cpu_20261018.json.

# JAX real-root filtering — 2026-10-18

The eigenvalue leaf now applies strict imaginary acceptance, expanded-domain
filtering, stable sorting and endpoint-only clipping in JAX. Integer keys retain
subnormal order/comparisons and stable signed-zero words; the host only slices
the retained count and copies storage. Native8/64element sort controls pass.
All other module ASTs and the existing eigenvalue callable remain unchanged.

Actual parent37PASS=14filter+23CIprotocol. Initial84PASS; expanded102node run
has100PASS2ownershipFAIL, including all79science checks and actual flower
inverse. Missing new core owner corrected; all23protocol checks thenPASS.
Failed run retained; no original assertion/bound change or single102PASSclaim.
Source/test bridge includes import-order-only formatting;150distinct eventual
passing nodes across overlapping gates, not one150node run. Observed parent
imports preguarded/stable, no censor/survivors. Archive3598payloads verified.
1115old lane prefixes retained→1116owners/13lanes/Hermite12/79%unchanged.

One sameCPU public-root pair retains all3outputs but observes3–9%warm overhead;
firstcall costs also recorded. No native/statistical speed or regression-free
claim. A proposed fusion was rejected before execution to retain the existing
callable/instrumentation contract. Host trim/norm/recursive mapping/pruning,
full roots/API/JIT/AD/provider words,322pages/1305figures, known numerical/speed
gaps and latest-mainCI remain open. Evidence: docs/roots_filter_jax_cpu_20261018.json.

# SIN subnormal component and numeric Diskfun branches — 2026-10-18

SIN now retains the gradual cos(real)*imaginary product for nonzero binary64
subnormal imaginary input and finite real input with normal finite cosine.
324selected native imaginary words match eager/JIT; analytic JVPs unchanged.
One original metadata record removed14→13.26unaryxfail cases include repeated
inputs;92fresh minimum-normal-input losses remain additional explicit gaps.
Three narrowly approved authored primitive-preservation oracles corrected;
original native fixtures/assertions/bounds unchanged, failed trials retained.
Broader near-cosine-zero/output-underflow diagnosis continues separately.

Numeric Diskfun preserve_lengths=True now follows source rankcapmin(m,n),
NaN-omitting first-column-major extrema, exhausted-factor errors, coupled pivot
location duplication, whole-column-matrix extrapolation and zero factor storage.
15legacy defaultFalse records exact. Child81PASS3strictXFAIL includes original
k7with documented one-zero-return-shape bridge; not a finalfilek7rerun claim.
Three factor/subnormal word gaps, tolerance power and host GE remain open.
New extrema use JAX; inherited host constructor arithmetic is not fully ported.

Actual combined parent2,052PASS29XFAIL: unary1,906/26 plus integration146/3,
including disk83cases, roots43 andCIprotocol23. Observed Python-import files
1543/1596 preguarded/stable; no censor/survivors. Archive3541payloads stream
verified.1114old CI lane prefixes retained→1115owners/13lanes/Hermite12/79%.
README now links the running checklist and states full parity is in progress.
FullAPI/JAX/providerwords,322pages/1305figures, known correctness/performance
gaps and latest-mainCI remain required. Evidence: docs/sin_disk_parent_cpu_20261018.json.

# TAN imaginary-NaN real component and original predicate maps — 2026-10-18

TAN now returns canonical positive real NaN for exact-zero real input with
imaginary NaN, following four captured native binary64 signed-quiet-NaN points.
Primitive imaginary words and prior real words outside that mask are retained.
A barrier around the complete prior result fixes an introduced complex64 sign
regression; all analytic JVPs stay exact. The first failed full run is retained.
Three new author-oracle failures were already differences between the prior
helper and standalone primitive; only those new oracle targets were corrected
using immutable prior-helper words. Original native assertions/fixtures/bounds
stay unchanged. A postgate docstring-only whole-module AST bridge is explicit.
Actual parent1,956PASS27XFAIL: unary1,881PASS27XFAIL plus75disk/roots/CIprotocol
checks. Observed Python-import files1541/1550 preguarded/stable; no censor or
survivors. One supplemental expected failure removed;14original records remain.
27cases=14original+13fresh-fixture cases, including7repeated original inputs.
Native imaginary NaN sign/payload and signaling-NaN behavior remain unqualified.

Fresh unmodified original native test_inv/test_roots/test_minandmax return
48/24/16true slots,88total. Every slot has a source/formula/Python-case mapping.
Inverse48slots include30algorithm predicates plus3broadcast predicates repeated
six times (33evaluations); Python counterpart files have61mapped cases plus
3supplementals, with17provider checks making the existing81casegate. Four raw
complex-root order controls pass at original bounds without sorting, resolving
that adapter concern for these cases. Real-root sorting and stronger absolute/
position predicates are explicit. Alternate preferences, full APIs/provider
words, warning semantics and performance are not qualified. Inventory attaches
nested default-predicate qualifications; full-item statuses remain unverified.

All1113owners/13lanes/HermiteCore12/79%gate unchanged. Other6subnormal unary
families contradict a shared analytic-product rule; SIN whole-subnormal family
is under qualification. Disk rank-cap/NaN/error and source arithmetic/JAX debt,
322pages/1305scientificfigures, known slow paths and latest-mainCI remain open.
Evidence: docs/tan_imaginary_nan_parent_cpu_20261018.json and
 docs/original_inv_roots_minmax_cpu_20261018.json.

# Disk constructor storage and inverse finishing — 2026-10-18

Coefficient-to-Diskfun conversion now follows native numeric construction:
retain factor lengths, project directly, and handle origin-only inputs before
zero/tolerance logic. Pivot locations use the literal source getPoints grids.
Nine native cases qualify ranks/lengths/pivots/locations/flags; remaining factor
provider differences are recorded explicitly. Existing defaultFalse behavior
is retained within nine captured controls. The original conversion prefix and
all unrelated method ASTs stay exact. New factor allowance64*n*eps*scale is
an authored bound; original tests/fixtures/tolerances remain unchanged.
Child106distinct checks pass, including13original Helmholtz/rank checks. The
first caller batch timed out after26completed passes; only32unfinished cases
were resumed. Both receipts are retained, not called a completed initial gate.
Public515cold4.274→3.296s includes first factor-value materialization; warm
ranges overlap. Storage13/1→515/256 deliberately changes reconstructed words,
max2.24e-17; no exact old-output/native-performance claim.

Inverse sampled-root final Clenshaw arithmetic now shares capacity-shaped
compilation before unchanged original-nFFT. Existing17strict provider controls
and full selected inverse/roots/minmax cohort81PASS; all20actual inverse output
artifacts remain word-identical. One sameCPU40–43pair observes cold17.018→16.017s,
warmmedians1.654→1.620s. No candidatebackendcompile-time or native/statistical
speed claim. Separate instrumented baseline attributes179compiles26.70s,
including11finish calls8.37s; diagnostic observer overhead is explicit.

Actual combined parent85PASS, observed Python-import hashes preguarded/stable,
no censor/survivors. Parent count and child106/81qualifications overlap and are
not summed as disjoint tests. All1112oldCIowner lane prefixes retained plus
35node file→1113owners/13lanes/HermiteCore12/79%unchanged. Native numeric rank-cap
and reciprocal semantics, inherited host GE, broader API/JAX/provider words,
322pages/1305scientific figures, CPU performance and latest-mainCI remain open.
Evidence: docs/constructor_inverse_parent_cpu_20261018.json.

# Numeric nonzero system PDE trajectories — 2026-10-18

Four actual unmodified native pde15s/installedode15s runs cover stationary and
decaying three-species polynomials on[-1,1]and[2,6], numeric endpoint targets
2/-3,N9,AdjustBCsfalse,times0/.005/.01,rtol1e-9/atol1e-11. All108public samples
compare within6.22e-15ofnative; both implementations meet the predeclared
established1e-7analytic bound (largest analytic error2.72e-9). No production
edit. Exact native capture/collector archived; four regression cases appended
without changing any prior test AST, assertion or bound. Actual parent79PASS
includes full existing native/manufactured, boundary-contract, empty-boundary
andCIprotocol modules. ObservedPythonimport files preguarded/stable; native
source/capture/executable guards stable, not exhaustive dynamic-library guard.

First collector wrongly asserted fixedNreturnedtimes=requestedtimes after the
native solve. Source nonAdaptiveEvent updates solution columns but never tOut;
literal native tOut=0andthreecolumns now recorded. Initial failed collector/
receipt retained; no numeric bound weakened. Broader output grammar remains
open. Per-case wall observations have different compilation/warm contexts,
not a matched native performance benchmark. Existing1112owners/13lanes/
HermiteCore12/79%gate unchanged. Full PDE/API/JAX/performance,322pages/
1,305scientific figures and latest-mainCI remain required.
Evidence: docs/pde_numeric_evolution_cpu_20261018.json.

# TAN/TANH exact-axis zero conventions — 2026-10-18

TAN exact-real-axis imaginary output is positive zero; at both-component-zero
inputs it retains the input real-zero sign. TANH real output is positive zero
only for exact-zero real input with finite nonzero imaginary input. TANH origin
and all other primitive component words remain unchanged. Word-aware masks
exclude nonzero subnormals. Custom JVPs retain original primitive analytic
tangents; independent real-coordinate first/second/transverse/RHS checks pass.
Nine original records repaired,23→14. Original fixtures/assertions/bounds retained;
one authored primitive-preservation oracle superseded only on the stated masks.

Actual parent1,919PASS28XFAIL: unary1,852PASS28XFAIL plus67PDE/protocol checks.
28expected-failure cases comprise14original+14fresh-fixture cases, including
7repeated original inputs and7supplementary gap rows; not28distinct defects.
Observed Python-import files preguarded/stable; no exhaustive native-library
guard claim. Native34call capture has9late unpreguarded libraries, explicitly
retained. Root streamverified3,558child archive payloads. Failed root launcher
stopped before tests; corrected launcher and successful receipt retained.

Follow-up corrects a diagnostic attribution: pure-imaginary Inf already agrees
with native; the differing class occurs at imaginary NaN. Neither shape nor
array-context dependence is established. No Inf repair is justified. All14
original gaps, supplementary cases, broader API/JAX/provider semantics,
322pages/1,305scientific figures, CPU performance and latest-mainCI remain open.
Evidence: docs/tan_tanh_parent_cpu_20261018.json.

## Grouped disk construction and SIN/COS boundaries — 2026-10-18

Actual parent1,951PASS23XFAIL, all runtime imports preguarded/stable. Source-static
construction grouping reduces observed public515cold7.50→4.40s; warm95.8→101.8ms
shows no warm improvement. Source public AST and1,324,024assembled/public words
retained; separate13original child tests pass through exact source bridge.
Generic traced19/64and11/64word differences remain explicit. No matched native
speed closure or universal backend/API/JAX claim.

SIN/COS exact-axis imag+0 and complexSIN both-word-zero real sign now match
captured native controls, with analytic first/second transverse derivatives.
Two metadata records removed25→23; three separate primitive real-sign defects
now pass, no supplemental expected failures. Author-preservation oracle masks
corrected only on these source-supported boundaries; native bounds unchanged.
Remaining23unary records, complex multiplication/scalar grammar, constructor/
provider words, pages/figures/performance and finalCI remain open.
[Parent evidence](disk_unary_parent_cpu_20261018.json).

## Numeric scalar system PDE endpoints — 2026-10-18

The real scalar numeric system-target rejection is repaired: repeat one
Dirichlet target per component, preserving native offset arithmetic.28baseline
failures become passing within67parent controls;12actual native boundary
assemblies match. Native observer stops before ODE evolution, so trajectories,
nonfinite solutions, general numeric grammar/API/JAX/performance and final
pages/figures/CI remain open. Original bounds/fixtures retained.
[Evidence](pde_numeric_system_target_cpu_20261018.json).

## TAN boundary and complex scalar diagnosis — 2026-10-18

Two TAN nonfinite real-axis records repaired; actual parent1,673PASS25XFAIL,
all loaded runtime inputs preguarded/stable. Original bounds/fixtures retained.
Remaining25unary records and whole API/IEEE/JAX/pages/speed/CI remain open.
[Parent evidence](unary_tan_axis_parent_cpu_20261018.json).

Five native source traces localize complex multiplication first divergence to
a1*M1.440native primitive calls show actual-five sparse/full equivalence; both
component-scaling and four-product interpretations fail. Extra finite negative
signed-zero exceptions retained. No scalar repair justified; five final-matrix
cases and original-numel/trailingzero shortcut grammar remain open.
[Diagnosis](ultras_complex_scalar_diagnosis_cpu_20261018.json).

## CONJ/UMINUS and nonfinite sparse conversion — 2026-10-18

Four CONJ/UMINUS imaginary-zero records and two real sparse-conversion expected
failures repaired. Actual combined parent1,774PASS27XFAIL, all loaded runtime
files preguarded. Prior unary/finite conversion arithmetic and original bounds
preserved. Five complex nonfinite final recurrence cases remain unresolved;
conversion-only class mismatches90→86 across8captures, no individual case worse.
Remaining27unary records and full API/JAX/IEEE/performance/pages/CI remain open.
[Parent evidence](nonfinite_pair_parent_cpu_20261018.json).

## SIN/COS boundary and matched disk CPU follow-up — 2026-10-18

Four native SIN/COS nonfinite real-axis storage gaps repaired; parent1587PASS/
31XFAIL and all1639runtime files guarded. Other unary records remain open.
[Evidence](unary_sin_cos_axis_parent_cpu_20261018.json).

Matched public disk515x256case: Python firstcall7.441s/native.6704s; warmed95/125ms
medians with native63–155msrange. Sevenprobe error3.33e-16, coefficientmax1.67e-16,
29,073real coefficient words differ. Source real projection is deliberate for
this complexK. Cold overhead remains; no uniform warm speed or full API claim.
Separate profiled call records208backend compilations and~5.94scumulative compiler
time within8.84sprofiledwall; it is diagnostic, not natural speed qualification.
[Evidence](disk_matched_performance_cpu_20261018.json).

## Shared ultras tail and nonfinite conversion — 2026-10-18

Source tail stopping now passes122parent checks; two strictxfails retain native
NaN/Inf conversion expectations. Native sparse conversion localizes nonfinite
terms while dense0*nonfinite spreadsNaN. This is separate from the repaired tail
condition. A matched515assembly pair improves14.99→3.79s with exact outputs at
threeKvalues; full native/end-to-end speed and JAX/API coverage remain open.
[Evidence](ultras_tail_parent_cpu_20261018.json).

Hermite n11 is an external dot accumulation discrepancy, independent of the
observed cosine differences. Explicit product sums match; correctly rounded
exact dot does not recover the native printout. No production repair inferred.
[Diagnosis](hermitequad_n11_attribution_cpu_20261018.json).

## Hermite GW barycentric follow-up — 2026-10-18

Source third-output normalization repairs54of82small-order controls; parent181
checks pass. Four explicitGW32/42cases retain eigenprovider differences. Native
row replay through42 tests normalization only. Nodes/weights and the page's
unrequested-third-output path are unchanged. Full page/figure/speed/API/JAX
qualification remains open. [Evidence](hermite_gw_bary_parent_cpu_20261018.json).

## Disk Helmholtz banded solve follow-up — 2026-10-18

Exact five-band coefficient solves pass81 parent checks and36 separate worker
complex derivative equation controls. Production-size warmed observations improve
about2.5x; small cases do not improve, and dense source assembly remains expensive.
Full native sparse ordering, status/nonfinite/IEEE/API/JAX and matched native
performance remain open. [Evidence](disk_banded_parent_cpu_20261018.json).

# Open numerical parity investigations

These source-bound observations narrow the remaining work; none closes full
function or example parity. Native reference is Chebfun 7574c77.

| Path | Evidence | Next decision |
|---|---|---|
| Hermite GW and HermiteQuad | Source normalization/grid corrected;70parent checks,78nodes/78weights and108coordinates exact; fresh full page. [Evidence](hermitequad_gw_parent_cpu_20261018.json). | Three printed rows, representation129vs137, curve provider/raster words and matched speed remain open. |
| Supplemental unbounded endpoints | Fresh native3438hash proof reproduces EXPnativefailure and XEXPnativepass/JAXfailure; matched maps and coefficient evaluation exact. [Evidence](unbounded_endpoint_provider_cpu_20261018.json). | Callback differences plus remaining constructor/transform differences persist after native callback replay. No repair or bound change. |
| Fixed48 PDE | Published scalar33 real CPU transform repair makes the unchanged authored fixed48 test pass;21parent controls qualify that scope. [Evidence](fft64_pde_cpu_20261017.json). | Broader FFT/PDE plans and latest-main CI remain required. |
| Complex rational fit | Published FFT602/C2R integration qualifies129final-source controls and exactD/V/Z assembly. Pole gap is2.076e-11; denominator gap5.739e-12. [Evidence](fft602_rational_integration_cpu_20261017.json). | Fresh unchanged ThreeBody415coefficients equal retained input; full run104.57s,1380imports preguarded. Same-input JAX default/QR SVD both retain5.739e-12vector discrepancy. Full native ODE/second-fit/figure/performance parity remains open. |
| Dense Chebfun3T constructor7 / representation | Original7 now passes normal markerless parent pytest at unchanged100eps; sealed candidate1.0103e-14 <2.2204e-14. Actual parent51checks pass. [Evidence](rader724_parent_cpu_20261018.json). | Native/candidate crops261/265/263 vs261/261/263 and coefficientwords still differ; HC2C4/three180convolutiongraphs require further source qualification. Full API/JAX/speed/CI remain open. |
| Small Hermite ASY | Public full30-node/all18valid small orders meet original bounds/storage; actual parent47checks include cold cache-disabledn2 and child coverage merge. [Evidence](hermite_public_parent_cpu_20261018.json). | Small whole-call JIT/AD, nonfinite storage, full API and matched CPU speed/memory remain open. Earlier180s component censors are historical, not current integration blockers. |

| C1 scalar eigs / shared dense provider | JAX generalized eig and native C1 constraint grids pass parent integration at unchanged bounds. [Evidence](generalized_qz_parent_cpu_20261018.json). | Latest-main CI reproduction, full grammar/providers/API/JAX transformations and matched CPU speed remain open. |

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

## Historical real-even C2R diagnostics — inverse family superseded below

[Source-bound comparison](transform_c2r_diagnostics_cpu_20261017.json) rejects
a production patch from these observations. Dense363cubed C2R matches fewer
native coefficient words; both crops fail100eps on the normalized diagnostic
grid. This does not replace the literal original constructor failure receipt.
Inverse C2R matches native8630coefficient count while raw coefficient agreement
is not improved. Native724/32768plans and fresh packed profiling are next.


## Real CPU radix-two C2R family integration

The [parent-reviewed integration](inverse_c2r_parent_cpu_20261017.json) follows
actual native32768plan evidence and a general power-of-two predicate, preserving
64/602 precedence and nonfinite/complex/nonCPU fallback. All254scientific and
21parent CIprotocol checks pass. Actual inverse now returns8630coefficients
(native8630, previous9030), native129value error3.886e-16, both independent
257compositions nearroundoff. Raw coefficient words still differ; no universal
bit-equality claim. Warm2.327s vsbase2.336s leaves native speed open. Dense724
constructor7 remained strictxfail at that historical checkpoint; the later
Rader724 integration repairs its original bound as recorded in the current
table above. Full API/page/figure/latestCI qualification remains open.


## Ordered Clenshaw CPU overhead

[Four-stage source scheduling](clenshaw_four_stage_cpu_20261017.json) reduces
same-host wholeinverse medianwarm2.3134s to1.7380s while preserving all20output
artifacts exactly. Ninety Clenshaw controls and64original-source checks pass.
The matched MATLAB speed qualification and native low-word/API/page/figure
work remain open; historical timings above describe their measured source.


## Current Maxwell repair and matched inverse timing

[Projected system eigs](maxwell_system_eigs_parent_cpu_20261017.json) removes
the discrete spurious second zero and preserves complex eigenfunctions. Both
actual CI failures and all four original default C2 predicates pass. This
supersedes earlier open duplicate-zero reports; full APIs/JAX generalized
eigensolver/matched performance/latest CI remain required. C1 adaptive preferences
and all six default original cell/autonomous predicates were subsequently
verified; see [original predicate evidence](original_system2_polyfit8_cpu_20261017.json).

[Matched single-CPU inverse benchmark](inverse_single_cpu_20261017.json):
warm medians native0.642599s/JAX1.710633s (2.662x); first-after-setup
0.972099s/31.864010s. Both produce8630coefficients. Cold profile has179backend
compiles taking25.62s; profile categories overlap. Measurements are descriptive
on a shared host. Performance parity and complete inverse qualification remain open.


## Current EQUI correction and sampled-root runtime

[Barycentric rounding repair](bary_rounding_joint_cpu_20261017.json) fixes
the actual74vs72periodic EQUI coefficient-count failure. Larger-query sampled
values now match902native words; small-query and FFT coefficient-word gaps
remain. No tolerance or length cutoff changed. The old unbounded ABS failure
passes locally before this repair; current CI is still required.

[Sampled-root four-stage recurrence](inverse_sampled_four_stage_cpu_20261017.json)
preserves20inverse outputs and lowers descriptive warm median1.752641→1.611100s.
Three capacity compile times worsen6%; no cold compilation or matched-native
performance closure follows. Existing full inverse/native/API gaps remain.

## Further primitive observations, still open

The [native180 packet](rader180_primitive_cpu_20261018.json) reproduces all
observed complete180transform words with a diagnostic JAX composition. It
separates generic180 arithmetic from the additional HC2C4 coordinate difference.
These diagnostic GPL transcriptions are not production imports. No full tensor,
crop, API, figure or performance acceptance follows.

The [dot/bary packet](dot_bary_scalar_provider_diagnosis_20261018.json) preserves
17native dot and6bary controls plus two scalar eigenvalue/vector observations.
The ordinary installedMKL APIs and OFF/AUTO/COMPATIBLE modes tested do not
uniformly reproduce MATLAB matrix-product words. A guessed reduction was not
accepted. Checkmark remains open.

## Shared dense eig and further primitive evidence

[Generalized eig parent evidence](generalized_qz_parent_cpu_20261018.json)
records60checks across055arithmetic and059scalar-storage stages with a reviewed
scalar-only branch bridge. Native20controls are separately qualified with an
11module executable AST bridge. Full APIs, transformations and speed stay open.

[HC2C4 observations](hc2c4_primitive_cpu_20261018.json) separate packing and
endpoint coordinates from180arithmetic; external coefficient reconstruction
uses captured native180buffers strictly as diagnostics. [Complex QR evidence](complex_poles_svd_cpu_20261018.json)
localizes the vector difference mainly to tall QR. JAX first norm matches a
correctly rounded high-precision reference; a native-provider difference alone
does not establish a mathematical accuracy defect.
