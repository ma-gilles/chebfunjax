# Disk Helmholtz banded coefficient solves — 2026-10-18

The parity solves now pack the exact five nonzero diagonals and use the existing
JAX pivoted band kernel, including distance-two pivots and both source RHS.
Dense source assembly and rank-one correction order remain unchanged. All81
parent checks pass:68focused/protocol and13original/rank. Original tests and
native fixtures retain their bounds. The separate worker addendum passes36
complex derivative equation controls; its source bridge is exact.
Matched shared-host observations at public m257/n256: warm .37590s dense versus
.15073s banded. Small cases show no benefit; cold compilation costs more and
dense assembly remains a bottleneck. No universal or native speed claim.
All1108prior CI owners retained, one new file appended;13lanes, dedicated
Hermite lane and79%coverage gate unchanged. Full API/nonfinite/status/native
ordering/JIT/AD/vmap and latest-main CI remain open.
Evidence: docs/disk_banded_parent_cpu_20261018.json.

# Hermite GW source normalization and guarded endpoint diagnosis — 2026-10-18

GW retains raw weights through symmetry and normalizes once afterward. Same12
native controls6FAIL6PASS become12PASS; all78nodes/78weights exact. Example
uses existing source_grid for108exact trapezoid coordinates. Actual parent70
checks pass; complete child72.41s page includes every original order and actual
stdout, unchanged600x270PNG. Only unrelated disk code differs in source bridge.
Full page/API/provider/raster/speed/latestCI remain open; three printed rows and
native129vsPython137representation differ. No core owner/config changes.
Fresh endpoint native3438hash capture reproduces EXPnative supplemental failure
and XEXPnativepass/JAXfailure; maps and both matched-coefficient evaluations
exact. Native callback replay still leaves coefficient gaps. No endpoint library
repair, source assertion or bound change.7145parent archivepayloads streamverified.
Evidence: docs/hermitequad_gw_parent_cpu_20261018.json and
 docs/unbounded_endpoint_provider_cpu_20261018.json.

# Disk Helmholtz source coefficient provider in JAX — 2026-10-18

Disk Helmholtz ultraspherical assembly/forcing/boundary transforms and parity
batched dense solves use JAX. Native exact complex K² retained; source correction
multiply/divide/subtract order preserved; K promoted before JIT entry, including
Python integer10**20. Actual parent47distinct checks pass:13original/rank plus
34focused/protocol. All10default original native predicates independently true;
source bounds/resolutions retained. Historical k7 already passed; current L2
5.56488e-12 versus unchanged2.22045e-8. All1106prior core owners retained+2;
13lanes/HermiteCore12/79%unchanged.3590archivepayloads streamverified.
Inherited constructor/coeffs2/harmonic providers, full singular/nonfinite/API/
JIT/AD/vmap, native sparse complexity/matched speed and latest-main CI remain
open. Evidence: docs/disk_helmholtz_parent_cpu_20261018.json.

# ASINH axis/underflow and large finite derivatives — 2026-10-18

ASINH retains native positive imaginary zero on the real axis, preserves tiny
real values and binary64 subnormal imaginary components, and scales large
finite analytic derivatives with signed branch-side preservation. Four ASINH
expected-failure entries removed; native fixtures/assertions/bounds retained.
Actual parent broad1392PASS35XFAIL precedes a JVP-only signed-root correction;
final50runtime/public checks pass.33axis native inputs and4Pell controls captured.
Pell imaginary words match native; real values differ1ULP. All3807archive
payloads stream-verified. Existing test files retained; core ownership unchanged.
Full API/nonfinite/branchpoint/IEEE words/extreme higher derivatives/speed and
latest-main CI remain open. Evidence: docs/asinh_axis_parent_cpu_20261018.json.

# HermiteQuad source plotting and checklist reconciliation — 2026-10-18

HermiteQuad uses public plotData for the restricted Chebfun rather than manual
1200-point sampling. Complete37.61s CPU run retains all original orders and
actual stdout,9input blocks,7output blocks and one600x270figure. Root verified
all2893archive payloads, identical library source bytes and viewed the figure.
Final table digits/native curve coordinates/font/raster/matched timing remain
open. Twelve historical checklist claims now link their superseding scoped
published evidence. Latest-main CI remains required; no full-page/API closure.
Evidence: docs/hermitequad_page_cpu_20261018.json.

# JAX generalized eig and C1 source constraints — 2026-10-18

Shared blocklinop._geig now uses JAX real/complex QZ and Hermitian/SPD paths;
caller sorting/filtering/adaptivity retained. C1 constraints instantiate their
functional stacks on C1 values grids.60actual-parent checks across055/059 stages
pass with a reviewed scalar-only AST bridge;20child native controls are separate.
Exact tiny-result rounding, Inf-times-subnormal classes and eligible scalar
real storage verified. All1103prior core owners retained+3;13lanes, dedicated
HermiteCore12 and79%gate unchanged. Earlier97e CI passed46jobs/82%reportedcoverage;
latest-main scientificCI remains required. Whole-callJIT/AD/vmap, separate
legacy providers, full APIs/CPU speed,322pages/1305figures and remaining gaps
stay open. Evidence: generalized_qz_parent_cpu_20261018.json.

# Dense Chebfun3T constructor bound repaired — 2026-10-18

Real finite CPU length724 transforms use a source-motivated C2R/Rader181
family with existing JAX180 convolutions. The unchanged original constructor7
now passes normal pytest without its obsolete expected-failure marker; sealed
candidate error1.0103e-14 is below original100eps2.2204e-14 (baseline2.8644e-14).
Actual parent51checks pass in2fresh processes, including27newcontrols and
23CIprotocol.1102prior owners retained+1Core1;79%gate unchanged. Child146scoped
checks are separately recorded. Native/candidate crops and words still differ;
full API/JAX/performance/pages and latestCI remain open.
Evidence: docs/rader724_parent_cpu_20261018.json.

# Polynomial minimax zero-scale IEEE status — 2026-10-18

Polynomial minimax retains the native zero norm and IEEE divisions, replacing
an artificial EPS floor. Actual native degree1zero target has NaN correction,
1e-19 exchange error and no warning; two public/input-adapter controls reproduce
both baseline failures and pass with the repair.40candidate scientific checks
and25actualparent checks pass;2overlap,63distinct. All source/test/fixture bytes
match across candidate and parent.1101prior owners retained+1Core1;79%unchanged.
Full minimax API/JAX/zero-storage/speed, Checkmark and latestCI remain open.
Evidence: docs/minimax_zero_parent_cpu_20261018.json.

# Polynomial minimax source normalization — 2026-10-18

Polynomial minimax uses the source continuous L2 norm rather than a sampled
maximum for its stopping scale and returned correction status. Three fresh
native controls and independent exchange/integral formulas expose6baseline
failures; all6pass with the source policy. Actual parent61checks pass in2fresh
processes, including12unchanged original clauses and23CIprotocol checks.
1100prior owners retained+1Core1;79%gate unchanged. Existing zero-norm adapter,
fullAPI/JAX/speed and finalCI remain open. Nine Checkmark probes are unchanged;
this does not repair or qualify the Checkmark page.
Evidence: docs/minimax_norm_parent_cpu_20261018.json.

# Disk zero-result source storage — 2026-10-18

Disk scalar-zero and equality cancellation retain native first factors,
pivots and flags; canceled-pole and zero-remainder addition follow source
branches. Same21native/public controls change19FAIL2PASS to21PASS.
Actual parent79checks pass in3fresh processes, including original31/32 and
23CIprotocol checks.1099prior owners retained+1Core7;79%gate unchanged.
Native rank2 stale-index errors are preserved; inherited host warnings,
complex/nonfinite metadata, fullAPI/JAX/speed and latestCI remain open.
Evidence: docs/disk_zero_storage_parent_cpu_20261018.json.

# Native root normalization and CI lock metadata — 2026-10-18

Real CPU Chebtech roots now retain source coefficient division rather than
compiler reciprocal multiplication. Fresh native9word controls expose2baseline
failures; final123focused controls and38unchanged original roots checks pass.
Actual parent184checks pass in two fresh processes, including23CIprotocol.
All1098prior owners retained+1;13lanes and79%gate unchanged. No tolerance change.
CI3396 failed before numerical tests because editable pytest-cov metadata was
stale. Pixi-generated metadata is refreshed; every locked dependency/version
is retained and pixi lock --check --no-install passes with no changes.
Checkmark construction, full roots/API/JAX/speed, pages/figures and latest-main
scientificCI remain open. Evidence: docs/roots_normalization_parent_cpu_20261018.json.

# Disk reciprocal and zero-pivot evaluation — 2026-10-18

Disk CDR and its verified consumers now follow the source reciprocal policy:
infinite reciprocals become zero and NaNs remain NaNs. Local JAX binary64
arithmetic also preserves subnormal inputs/results; all16native captured
words and22,130signed exact-rational controls pass. A CPU-only compiled helper
avoids a measured eager dispatch regression. Shared providers are unchanged.
Actual parent104checks pass in two fresh processes, including original31/32
at unchanged EPS. All1096prior owners retained+2;13lanes and79%gate unchanged.
Zero-result storage/flags, inherited host iszero debt, full APIs/IEEE/JAX,
matched CPU speed, pages/figures and latest-main scientificCI remain open.
Evidence: docs/disk_cdr_parent_cpu_20261018.json.

# Source pole cancellation and callable Bernoulli columns — 2026-10-18

Strict original disk31/32 now pass at unchanged EPS after source addPoles
coefficient/L2/extracted-scale correction.13newcontrols+17existing+3structural
checks pass; no externalcallback/provider/trim/tolerance substitution. Canceled-
only-zero storage/pivot and remaining providerwords stay open.
Bernoulli now returns a callable column-valuedChebfun; prior recurrence and
column indexing preserved. Fresh native N0/1/4/8 capture and samefinal5failing
APIcontrols become6PASS including unchangedoriginalsmoke. Actual combined
parent64checks pass in2freshprocesses. All1094owners retained+2;79%unchanged.
Fresh unmodified nativeHermite test15/15true;16assignments/overwritten11mapped
to4existingpassedPythonmethods. No newwholeAPI/page/figure/speed/CIclaim.
Evidence: docs/disk_poles_bernoulli_parent_cpu_20261018.json.

# Supplemental cosine L1 test contract — 2026-10-18

External JAX cosine now tests finite continuous L1 optimality at original1e-8,
with unchangedcallback/degree/20queries. A separate source-compatiblecallback
asserts15native constructorwords+2endpointcallbackwords before retaining exact
MAXITER/20NaN/noInf checks. Actual parent25checks pass; original8predicatebytes,
all production and all other test ASTs unchanged. Prior incorrect warningtest
and its actual failure preserved. This is a supplementaltest correction, not a
library repair or externalproviderword-parity claim. Existing1094owners retained.
Evidence: docs/polyfit_l1_cosine_parent_cpu_20261018.json.

# Public small Hermite, disk maps and original query adapters — 2026-10-18

Explicit small Hermite ASY now uses the source-derived pure-JAX trajectory.
Full public30-test file passes all18valid small orders at original bounds and
native finite storage; n3 source error retained. Actual parent47checks pass,
including cold publicn2 with cache disabled and real child coverage/merge proof.
Small whole-call JIT/AD and performance parity remain open. All1092old owners
retained; disk file appendsCore7, Hermitefile dedicatedCore12;79%gate unchanged.
Disk maps now follow native six-operation factor maps;19new+17existing+original31
parent checks pass. Strict original32 remains failing; residual1ULP functional
word difference retained. Legacy unbounded ports now use actual original seed
queries and exponents: both14-clause adapters pass; endpoint accuracy is not
repaired. Combined112distinct parent checks pass across separate owned stages.
Evidence: docs/hermite_public_parent_cpu_20261018.json and
 docs/disk_unbounded_parent_cpu_20261018.json.

CI251 completed with known failures plus C1 spectrum3.8022e-9>1e-9; current
local6.9063e-10 pass does not repair CI. All APIs/pages/figures/matched speed and
latest-main green CI/coverage remain required. No overall completion claim.

# Scalar simplify honors session tolerance — 2026-10-17

Omitted scalar Chebfun.simplify tolerance now uses existing global ChebfunPref,
as native simplify.m36 requires. Four failing C1/C2 normal/transposed cases
are repaired;6new+64existing scientific checks and21parentCIprotocol checks
pass (91unique nodes;6overlap parent/child). All1089prior core owners retained
plus1new file/6selectors. Explicit tolerance/chopping/scaling/metadata unchanged.
Full simplify API/JAX/performance, Checkmark page and latestCI remain open.
Evidence: docs/scalar_simplify_parent_cpu_20261017.json.

Parent also corrected agent patch transport path prefixes without altering
executed source or output. Frozen original archives are preserved; intended
publication path sets and standard-git replacement patches are verified.
See docs/patch_transport_review_cpu_20261017.json.

# AbsoluteValue source computations and tracked gaps — 2026-10-17

The complete AbsoluteValue example now uses public Chebfun plotData and the
source error function abs(r-abs(x)); artificial samplewise error clipping was
removed. Actual application132.33s, five600x270figures, empty stdout and zero
scientific warnings. All1380actual import files were preguarded. Fresh pinned
MATLAB executes all five cells; representation lengths and visible error
patterns differ. Parent reviewed source/evidence archive3956payloads and exact
presentation replay. This is a source/presentation repair, not page/figure parity.
Evidence: docs/absolute_value_parent_cpu_20261017.json.

Paired supplemental cos4 controls now establish that source-compatible cosine
reproduces native MAXITER/20NaNs with15/15native constructor coefficient words;
unchanged external JAX cosine gives20finite outputs and L1momentnorm2.61e-12
below original1e-8. Strict external warning regression remains unresolved;
no solver/test/callback rewriting. Evidence:
docs/cosine_watson_paired_controls_cpu_20261017.json.

Current checklists supersede stale nonfinite-mask/no-x callback entries.
All322final-source page qualifications, full API/original tests/JAX/speed,
scientific figures and latest-main CI/coverage remain required.

# Original cell-system and L1 test coverage — 2026-10-17

All6default original test_eigs_system2 predicates now run with actual cell
callbacks, piecewise domain mutation and the no-x C1 callback. Maximum error
1.6689e-12<unchanged1e-9. Existing constructor support already handled this;
the previously stated unsupported callback gap was stale. No library edit.
New regression plus21CIprotocol checks pass;1088prior owners retained+1new.

All8original polyfitL1 numerical predicates pass on frozen36e at unchanged
1e-8; largest moment1.32e-12. Fresh unmodified native test_polyfitL1() also
returns all8true (13.3182s actual function). Historical harness-only caveat is
superseded for native invocation; Python evidence remains bound to36e.
Supplemental external cosine warning, full API/backend/JAX/speed, all pages/
figures and latestCI/coverage remain open. Evidence:
docs/original_system2_polyfit8_cpu_20261017.json.

# Native nonfinite transform components — 2026-10-17

Three previously reported ABS Inf/NaN mask discrepancies are repaired in
shared C1/C2 primitives. Direct complex construction avoids zero*Inf cross
terms; C1 nonfinite real transforms retain the infinite constant through a
real inverse-prefix route. Established finite FFT arithmetic remains unchanged.
New24native-stage tests repair7baseline failures;16existing exact C1/JIT/AD
and44original transform nodes pass. Parent111tech/native/protocol+3unbounded
and20complex DCT/602 AD nodes pass. All1087prior core owners retained+1new.
Eight captured cases do not qualify full nonfinite/native FFT/API behavior.
Matched speed, all pages/figures, cos4warning and latestCI/coverage remain open.
Evidence: docs/nonfinite_transform_native_cpu_20261017.json.

# Bounded source cosine derivatives and polyfit diagnosis — 2026-10-17

Bounded source cosine retains its exact primal AST and now uses analytic
JAX tangents. Twelve failing derivative controls are repaired;15new checks,
8existing source controls and the unchanged fixed48 PDE regression pass.
Parent21CIprotocol checks pass;1086prior owners retained+1new. Native1040
independent vector/scalar controls and50actual constructor samples match words.
The supplemental polyfitL1(cos,4) warning failure remains: two external JAX
cosine samples differ1ULP from native, whereas transforms of identical samples
match exactly. No callback routing, root filtering or synthetic warning added.
Full APIs, pages/figures, matched speed and latestCI/coverage remain required.
Evidence: docs/bounded_cosine_parent_cpu_20261017.json and
docs/polyfit_cos4_first_divergence_cpu_20261017.json.

# Existing-grid absolute value representation — 2026-10-17

C1/C2 direct absolute value now follows native real/pure-imaginary storage
dispatch, preserving the existing coefficient count and happiness. Empty
arrays retain shape and realify storage. General complex fallback is unchanged.
Parent87tech/original/protocol checks plus3original unbounded assertions pass;
all1085prior core owners retained plus1new. Child79checks overlap these gates.
Three native nonfinite transform-mask discrepancies remain open primitive debt;
no complete ABS API/output, speed, page/figure or latestCI claim.
Evidence: docs/abs_existing_grid_parent_cpu_20261017.json.

# First-kind adaptive eigenproblems — 2026-10-17

C1 now shares the native targeting/refinement policy with C2 and retains
first-kind output functions. Explicit system C1/C2 preferences are forwarded.
Three failing baseline checks are repaired; six final focused checks and39
existing regression nodes pass, with21CIprotocol checks separately passing.
Fresh pinned MATLAB system2 returns all6true; captured C1 block residual is
1.9913e-13. Existing1084core owners stay fixed; one selector is appended.
Broader cell/BC grammar, full eigs API/JAX conversion, matched CPU speed,
all pages/figures and latest-main scientificCI/coverage remain open.
Evidence: docs/c1_eigs_adaptive_cpu_20261017.json.

# Twelve interrupted script edits reconciled — 2026-10-17

The per-script disposition table is now in
docs/handoff_twelve_disposition_review_cpu_20261017.md. All twelve current
replacements are committed, and clean historical full-run receipts exist for
all twelve. Original dirty payloads are preserved and superseded, with no
claim of accepting them verbatim. Later LaplaceBall, FermiDirac and DDE runs
supersede older partial-execution notes. Every row lists remaining numerical,
stdout and figure gaps plus its next step. No new final-source page pass.

# Native barycentric rounding and sampled-root runtime — 2026-10-17

The periodic EQUI CI failure is repaired: the constructor returns the native
72coefficients rather than74. Barycentric node-loop multiplication/addition
are separated, and final elementwise division retains its source rounding.
All902captured native values at65/129/257queries are word-identical. Final
39bary/native,12FUNQUI/EQUI and35original interp1checks pass unchanged bounds.
The combined gate has41passing nodes, including21CIprotocol checks; all1083
existing core owners remain and one native regression selector is appended.

The sampled-root recurrence now uses four ordered instructions per pair;
141child pytest checks and48direct controls pass, all20inverse output files
are unchanged. Local warm median1.752641s→1.611100s is descriptive; its three
capacity compile times worsen about6%, so cold compilation remains unresolved.
Native low words/full APIs/pages/figures/matched speed/latestCI remain open.
Evidence: docs/bary_rounding_joint_cpu_20261017.json and
docs/inverse_sampled_four_stage_cpu_20261017.json.

# Maxwell system eigenmodes and matched inverse CPU measurements — 2026-10-17

Coupled C2 eigenproblems now use existing exact JAX AD extraction and native
rectangular projection, removing the spurious second zero mode and preserving
complex eigenfunctions. Both actual CI regressions pass; the four literal
native smooth/piecewise predicates pass at unchanged1e-9 bounds. The reviewed
child packet has38unique scientific selectors plus1structural observer;
parent integration has24passing nodes (21protocol,3scientific selectors).
Explicit n/n_default precedence and genuine repeated modes are controlled.
All1082existing core owners remain; one new selector is appended. Existing
generalized-eig host boundary and broader C1/full API qualification remain open.
Evidence: docs/maxwell_system_eigs_parent_cpu_20261017.json.

Matched retained-input single-CPU inverse timing: MATLAB warm median0.642599s,
JAX1.710633s (2.662x); first-after-setup0.972099s vs31.864010s. Both return
8630coefficients with native sample discrepancy3.886e-16. Serial shared-host
descriptive measurement; no speed-parity claim. Cold profiling attributes
25.62s to179backend compilation calls; cumulative profile categories overlap.
Full inverse API/low words/pages/figures and other CPU paths remain open.
Evidence: docs/inverse_single_cpu_20261017.json. Latest-main CI/coverage required.

# Four-stage ordered CPU Clenshaw — 2026-10-17

Each coefficient pair now uses4instruction iterations instead of6, retaining
separate multiplication and ordered add/subtract groups. Ninety existing
Clenshaw/native/IEEE/AD/chunk controls and64unchanged original Clenshaw/inverse/
interp1checks pass. Inspected98optimized LLVM files have noFMA intrinsic.
Same-input wholeinverse medianwarm2.3134s→1.7380s (3calls per arm); all20output
artifacts byte-identical,8630coefficients retained. This is descriptive sharedhost
CPU40-43 evidence, not matched native speed closure. Other363tracked library
files/tests unchanged; no selector/owner change. Batch4variant was slower and
rejected. Full API/low-word/page/figure/latestCI/coverage remain open.
Evidence: docs/clenshaw_four_stage_cpu_20261017.json.

# Real radix-two inverse transform family — 2026-10-17

CPU finite realfloat64 DCT-I now uses C2R for power-of-two FFT lengths,
following native mirrored real input and observed native32768 C2R plan.
Existing64/602 specializations retain precedence; other routes unchanged.
All254scientific controls pass unchanged bounds; parent21CIprotocol checks
pass with1081prior owners retained plus1new selection. Actual inverse now
returns8630coefficients like native (base9030); native129error3.886e-16,
independent257compositions nearroundoff. Warm2.327s vsbase2.336s is descriptive,
not speed qualification. Coefficient low words/full APIs/pages/figures and
matched native speed remain open. Dense724 is unchanged; latestCI required.
Evidence: docs/inverse_c2r_parent_cpu_20261017.json.

# Native complex barycentric weights — 2026-10-17

Complex nodes now use native baryWeights.m capacity C=1. Explicit complex
phase z/abs(z) also retains the analytic rotation JVP (JAX sign gives zero).
Two initial phase failures and two subsequent derivative failures repaired;
final12focused checks pass, including4new closed-form controls. Four fresh
pinned-native controls pass and are compared directly with JAX. First native
run crashed in MATLAB ProductsInit after producing outputs; rawfailure retained,
clean retry required. Real arithmetic/other functions and old tests unchanged.
Full API/large-n/native performance and latest-main CI/coverage remain open.
Older CI also exposes Maxwell duplicate-zero eigenmodes; these remain unresolved.
Evidence: docs/bary_complex_cpu_20261017.json.

# Interpolation and minimax CI corrections — 2026-10-17

Actual interpolation predicate25 failure reproduces at exactly100eps. Pinned
native passes1.5765e-14; source-loop diagnostic passes1.4877e-14. Bary now follows
native point/node accumulation branches instead of one matrix product.
Final65interpolation/bary checks pass unchanged bounds; original35test bytes
unchanged. Two additional complex-JVP loop-boundary checks pass.
Five minimax CI checks use guarded scalar/shape comparison adapters; no operand
or tolerance changes. Baseline5fail→5pass, then5pass on final bary source.
Native low-word/weights/full API and latest-main CI/coverage remain open.
Evidence: docs/interpolation_minimax_ci_cpu_20261017.json.

# C2R transform diagnostics remain unqualified — 2026-10-17

Same-input363cubed Chebfun3T diagnostic C2R transform matches fewer native words
than current; both selected crops exceed100eps on retained normalized query
coordinates. This is not a fresh literal physical-map constructor gate.
Inverse C2R gives8630coefficients like native, but raw coefficient agreement
is not better; current gives9030. Count matching is not a production criterion.
No library/tolerance/crop change. Actual native724/32768plans and fresh packed
inverse profile are next; no matched native speed or full parity claim.
Evidence: docs/transform_c2r_diagnostics_cpu_20261017.json.

# Coupled complex derivative towers use JAX native IVP — 2026-10-17

Supported unit-diagonal coupled higher derivatives and complex endpoint states
now map to the existing JAX Adams engine. Source preserves full-tower fitting,
position-only maxnorm and final variable projection. Unsupported numeric
higherorder endpoint vectors are rejected per native parseBC. Final44controls
pass, with56earlier controls and source bridges; parent21CIprotocol checks pass.
Original full ThreePlanets completes221.80s with1native/0SciPy integration calls,
4frames and all1584runtimefiles preguarded. Parent verifies145child payloads.
No warning synthesized and no page/script/published-image changes. Native
trajectory/raster, full grammar and matched speed remain open.
Evidence: docs/coupled_ivp_parent_cpu_20261017.json.

# Core7 deadline and explicit exchange roots — 2026-10-17

Completed olderCI39071892475 hit90minutes inCore7 after10finished processes.
The first four processes took2633.31pytest seconds. Split originalCore7 after
those four complete batches into lanes7/11; all1080selectors, every process
boundary, other lanes, timeouts and coverage>=79% remain unchanged.
ActualCI also exposed stale default-root assumptions: polynomial exchange now
requests nobreaks explicitly, as pinnednative trigremez.m377. Default stored
breakpoint roots remain intact. Current baseline1fail/5pass; candidate29checks
pass24.15s. Latest-main actual runtime/scientific green/coverage remain required.
Evidence: docs/ci_core7_cpu_20261017.json.

# Public ChebPadé JAX padding — 2026-10-17

Both public padding branches now use the existing shared JAX normal provider
instead of resetting a NumPy RNG. Four baseline failures repaired;51regression
checks pass, including the six literal original predicates. Six additional
public-adapter checks reproduce retained native padding fixtures with supplied
normal draws. All bounds unchanged. This does not establish MATLAB RNG bits.
Other361library files and existing test bodies unchanged; public warning,
full API, remaining host CF numerics and matched CPU speed remain open.
Evidence: docs/public_chebpade_padding_cpu_20261017.json.

# Legacy ChebPadé JAX port — 2026-10-17

The direct utils.ratapprox.chebpade wrapper now preserves complex data,
uses JAX fitting/evaluation and maps physical Chebfun domains. Eight initial
contract failures repaired; final15CPUchecks pass, including four unchanged
authored tests. Seven fresh pinned-MATLAB controls meet their numerical bounds.
The first candidate's denominator-length assertion was wrong; native confirms
three coefficients. Raw failure retained. Rank-warning sequence still differs;
legacy sampling/zero-padding/K extensions and full APIs remain unqualified.
Other361library files and other module/test ASTs unchanged. Public cj.chebpade
uses a separate frontdoor; no full original-test or CPU speed closure claimed.
Evidence: docs/legacy_chebpade_cpu_20261017.json.

# Latest ThreePlanets actual warning visibility — 2026-10-17

Blanket warning suppression removed; only actual solve warnings captured/printed
with explicit slot1. Full actual application219.22s completes4frames equal to
publishedPythonbytes,0warningattempts/captured,3outputs256happyFUNs/4229coeffs.
All1582runtimefiles preguarded; originalnumerical/RNG/plotAST unchanged.
Raw supervisor flagroot_exited_with_live_descendants preserved: parent source/
sample/identity review attributes it to root-only exit-observation race (exit0,
finalrootRSS0,nosignals/events/descendants/survivors,belowlimits). No bare
uncensoredclaim. ExistingSciPycomplexhighorderLSODA provider remains JAX/native
gap; no freshnative trajectory/warning/raster parity. Page/images unchanged;
cached65537warning stillunmatched, no warning synthesized. Source361files equal
currentparent; differingPadéfunction unrelated, allothermoduleAST equal.
Evidence: docs/threeplanets_parent_cpu_20261017.json.

# Latest Padé empty optional defaults — 2026-10-17

Native isempty defaults restored for optional tolerance/radius/FFTlength.
None, emptylist and emptyJAXarray work for coefficients and callable input;
zero tolerance remains nonempty and disables robustification. Six baseline
failures nowpass; final23CPUchecks include all17priorchecks unchanged. Fresh
pinnedMATLAB numeric/callableemptydefaults and zero-tolerance controls pass.
Stable declaredinputs/no survivors; other361libraryfiles byteidentical and
allothermodulefunctions/oldtestbodies ASTunchanged. FullAPI, exactnumerical,
matchedperformance/latestCI/coverage remainopen.
Evidence: docs/pade_optional_defaults_cpu_20261017.json.

# Latest Lorenz warning slot and TransientGrowth diagnosis — 2026-10-17

Full unchanged Lorenz numerical/plot operands run33.63s, all1381runtimefiles
preguarded, stable/no survivors. Actual complex-plot warning now reaches stdout
and the correct fourth page outputslot using explicit source markers. Four
600x270figures complete;2PNGs changed from priorpublication, all4differ from
referencepixels. Allfour rational records andPNGbytes equal preceding run;
non-outputMarkdown exact. Native ODE/poles/numericstdout/rendering remainopen;
warningterminalpunctuation differs fromcache. One formerlyblankslot closed;
six other targetedblanks/three warning-countdifferences remain unresolved.
TransientGrowth originalconstructor alone resolves28happyFUNs/2032coefficients
with0actualwarnings; source362matchesparent, no forced native6050warning/library
fix justified. This doesnot qualifyfullpage/energy/figures/freshnative behavior.
Evidence: docs/lorenz_warning_cpu_20261017.json and
 docs/transient_growth_constructor_cpu_20261017.json.

# Latest output-slot and unresolved-test audit — 2026-10-17

Parent reviewed72payloads and68unchanged per-file snapshot bindings for ten
page/output discrepancies. All seven blanks and three count differences are
warning/environment diagnostics; no omitted numerical print among these slots.
Four JOGL graphics warning blocks remain unmatched; solver/constructor/plot
warnings and version differences remain explicit. No warnings synthesized.
Ten page ledger rows now link scoped mapping, preserving all item statuses.
Separately, unchanged Chebfun3T constructor predicate7 still fails after shared
inverse-FFT changes:2.864e-14 >2.220e-14, shape249/259/263;34.48s, stable/no
survivors. Strict xfail retained; no numerical repair claimed by this audit.
Evidence: docs/output_slot_reconciliation_cpu_20261017.json and
 docs/chebfun3t_current_transform_cpu_20261017.json.

# Latest Bloodhound scalar C2 corrections — 2026-10-17

Coefficient-induced Jacobian breaks now use the existing JAX block correction;
C2 RHS evaluates source equation points without rejecting tiny merged FUNs.
Scalar callable RHS retains broadcast behavior. Original7joint controls passed
before that narrow final branch; final4focused controls pass, with an explicit
unchanged-vector-branch bridge to actual Bloodhound. Whole solve completes233s,
7Newton iterations/error4.842e-11<1e-10, zero actual warnings and two figures.
Cached native giveUp warning remains different; no native convergence/page or
figure parity claim. Parent verifies148child payloads and361matching library
files (unrelated Padé module differs), appends one core selector1079→1080while
retaining every old owner;20protocol checks pass. Padé evidence attached to two
inventory rows without changing any full-item statuses. Full static gates pass.
Evidence: docs/bloodhound_parent_cpu_20261017.json.

# Latest CPU Padé JAX port — 2026-10-17

Padé array numerics now use JAX. Native original3predicates all true; final
17Python checks pass with unchanged original bounds, stable declared inputs
and no survivors. Four baseline contracts failed: zero degree (-Inf), tiny
imaginary numerator preservation, NumPy-free fit and jit/complex-JVP evaluation.
All four now pass. Source order forms numerator before cancelling shared leading
powers; genuine complex Taylor data is never discarded by absolute threshold.
Other module functions are AST-identical;361other library files byte-identical.
Legacy trig fitting, full API/parameter audit, exact FFT/SVD and matched native
performance remain open; latest-main CI/coverage still required.
Evidence: docs/pade_jax_source_cpu_20261017.json.

# Latest reference prose/structure audit — 2026-10-17

All322pages now match cached reference parser-normalized visible prose,
equations and headings after27implementation notes moved out of21example
pages. Exact2408MATLABinputs and1305figure-reference/interleaving positions
agree. Historical notes/provenance remain in docs/example_evidence. This
closes the measured structural discrepancies only;7blank outputslots,
3historicalwarning-count differences, computations and figure science/raster
remain open. [Evidence](page_structure_prose_parent_cpu_20261017.json).

# Latest ThreeBody after FFT602 integration — 2026-10-17

The complete unchanged CPU script runs in104.57s with all1380observed imports
preguarded, stable declared inputs and no survivors. Its newly constructed415
coefficients equal retained input words exactly, and both fits complete with
degrees151/8and157/156. The first fit's retained same-input native pole gap is
2.076e-11. Five actual stdout blocks and two refreshed PNGs are published.
All four600x270figures differ from official reference pixels. Full nativeODE,
second-fit numerical/residue, renderer and matched performance parity remain
open. See [evidence](threebody_fft602_current_cpu_20261017.json). This report
supersedes earlier production pole/FFT diagnostic-only status below.

# Latest LaplaceBall whole-script scope — 2026-10-17

The complete unchanged script now runs on current CPU source, including its
renderer, in34.994s with stable guards and no survivors. Eleven actual outputs
are retained and the page reflects the current Oxford boundary value. All three
600x253 PNGs equal previously published bytes; none equals reference pixels.
All12interrupted script hashes still match the latest retained source bindings.
Boundary65x65vsnative65x63, general RNG, final digits, Cartesian Spherefun API,
slice lighting/colorbar/fonts/raster and latest-main CI remain open.
Evidence: `laplaceball_current_cpu_20261017.json`.

## Current numerical packages

Latest source has corrected the earlier subspace spectral-norm claim: native
default residual norm is Frobenius. Subspace/Deltafun, JAX LU, scalar CPU FFT/PDE
and complex extrema scopes are tracked in parity_checklist.md and their reports.
Earlier entries below retain historical scope and do not imply current full parity.

# Latest subspace empty/rank-zero scope — 2026-10-17

Subspace now matches captured empty/rank-zero outcomes and preserves four
rank-deficient column/argument orderings. Final nine focused controls pass;
the earlier16-node gate includes the original subspace predicates. Native
command returned0 but its broad guard failed because moving main was guarded;
actual native input/source hashes remain equal. This is a reconciled source-bound
observation, not a clean native supervisor pass. Tested function/test bytes are
retained with an explicit current-source bridge. Broader API and near-rank
quantitative fidelity remain open. See subspace_empty_rank0_cpu_20261017.json.

# Latest first-kind Chebfun2 and string-operator scope — 2026-10-17

Genuine first-kind grids, transforms, factors and native tripling/pivot nesting
now serve adaptive callable Chebfun2 preferences. All three original technology
predicates pass with unchanged bounds. Raw string multiplication preserves
matrix versus dotted elementwise behavior; all seven original no-dots predicates
pass, including the exact MATLAB:innerdim marker. Joint15 CPU controls pass.
Broader API/grammar, existing GE host numerics, sample-test ordering and full
parity/CI/coverage remain open. Evidence: c1_nodots_cpu_20261017.json.

# Latest subspace and unbounded inner-product scope — 2026-10-17

The joint CPU gate passes19 nodes: six original subspace predicates plus
API/numerical controls and three original unbounded inner-product predicates
plus two complex controls. Subspace now follows native QR/SVD with JAX numerics,
Hermitian products and residual spectral norm. Unbounded inner products now
conjugate their first operand. A final domain-check correction follows native
max(hscale(A),hscale(B)); the affected domain/orientation control passes.
Near-rank quantitative angle fidelity, remaining API cases, full parity and
latest-main CI/coverage remain open. Evidence: subspace_unbnd_inner_cpu_20261017.json.

## ComplexArcLength graphics adapters and tracking repair — 2026-10-17

The example now applies existing MATLAB layout/equal-axis adapters at its final
600×270 canvas and uses numeric matlab_plot for the original marker sizes 16/8.
A full unchanged-computation run completes in 68.635s, with distinct 64-point
P and Q groups captured. Inverse input/output arrays equal the previous whole
run exactly. Figure 4 mean channel difference improves from 12.3598 to 10.9031;
all four images retain reference sizes, but raster/marker/typography differences
remain. The earlier Q-only exploratory comparison is not parity evidence.
The failed interpreter-path launch occurred before Python and is retained.
The only imported source difference from integration main is the unrelated
trigremez presentation body; the scoped source bridge is recorded explicitly.

The checklist's missing historical handoff-audit link is repaired by preserving
its original b16 report and hashes. Four grouped exporter cohorts missed by
basename matching now have reviewed navigation entries; all 64 demo fixtures
match native bytes, and five Python controls pass. No native exporter invocation
or complete original-test/API parity is claimed by this package.

[Renderer evidence](complex_arc_renderer_cpu_20261017.json) and
[exporter mapping](chebgui_inventory_mapping_20261017.json).
Latest-main CI/coverage and the full goal remain open.

## Packed Clenshaw CPU optimization and iteration colors — 2026-10-17

Packed registers and branch-local reads remove redundant vector copies while
preserving each native Clenshaw arithmetic operation. Independent points use
1024-point chunks without padding; global finiteness decisions remain global.
All 95 component controls pass, including cross-chunk AD; 59 retained native
Brent states retain exact raw bits. Four actual retained-input inverse calls
have byte-identical outputs to the current-main baseline. The baseline alone
also has a fifth profiled call; earlier summaries conflated those counts.

Warm inverse median falls from 6.9312s to 2.2687s (3.055x). The same-core,
serialized native median is 0.6536s, so a 3.47x gap remains. These shared-host
measurements use native single-thread execution versus JAX backend threading;
they are not a full performance acceptance result. Native inverse length
8630 versus Python 9030 remains unresolved despite near-roundoff value gaps.

The unchanged full ComplexArcLength script completes in 66.112s with stable
inputs and no survivors. Its inverse input matches retained bytes exactly.
Printed arc lengths differ from official HTML by at most 1.1e-14; forward
length is 2092 versus the historical page's 2762. All four refreshed figures
are 600x270, with visible axes/marker/layout and raster differences remaining.
The generated page now records actual Python stdout, including timings.

Trigremez iteration plotting no longer double-advances the color cycle for
the black named-Color line. All 24 combined plotting controls pass. An earlier
24-pass run's broad guard failed on concurrent generated PNG writes; both
receipts and scoped reconciliation are retained. A malformed review JSON is
preserved beside its validated serialization repair. Test formatting has an
AST/import-multiset bridge; original operands and bounds are unchanged.
Incidental test/stdlib imports are observed rather than all preguarded; no
full external-runtime freeze is claimed. PDE tolerance failure, broader API,
original tests, pages/figures and latest-main CI/coverage remain open.

[Source-bound review](inverse_packed_color_cpu_20261017.json).

## JAX spin comparison and complex C2 construction — 2026-10-17

Public `spincomp` now compares 1D/2D/3D schemes with source sampling,
normalization, fresh preferences and internal stepping times. Public 1D spin
uses the JAX integration engine. Source checkpoint behavior is preserved:
one-output short/nonaligned solves retain the initial output; timed calls
reproduce the missing-third-output error when no endpoint was saved. The
loop stops immediately on the native checkpoint tolerance. Final gates pass
39 1D/fixture checks, four affected 2D/3D checks and six root adapter/render
checks. Six native N=8 cases confirm the output distinction, including a
nonzero initial tspan entry. NaN plot reductions now follow native omission.
Full spin API, long examples, graphics and matched performance remain open.

The default nonperiodic C2 Chebfun3 constructor now carries complex callback
values through native ACA/QR/DEIM/Tucker construction rather than reconstructing
real/imaginary parts separately. All five existing std3-file controls and two
new real/complex constructor controls pass (59.09s, sampled peak 3,008,872 KiB).
These include one literal original MATLAB constant predicate; they are not
seven original MATLAB predicates. Native complex polynomial/conjugate/product
controls have matching core sizes and analytic errors below 1.6e-14. C1 and
periodic complex construction retain their existing paths.

The Chebfun3 wrapper exits 4 because its final runtime auditor found one
unpreguarded dynamic JAX dialect source, despite seven passing tests and
stable source inputs. Post-run inspection proves it is the same file as the
pinned wheel alias; this does not establish a prelaunch freeze. No full runtime
closure, complete Chebfun3 parity, earlier CI shutdown cause, whole-suite green,
CPU performance parity or coverage acceptance is claimed.

Evidence: [parent review](spin_complex_integration_cpu_20261017.json).
The receipt retains failed/superseded observers and the root mock correction;
external dependency preimages remain referenced rather than all embedded.

## PDE option consumers and source color cycle — 2026-10-17

Basic pdeset options now reach polynomial pde15s: Eps, N, scalar/vector AbsTol,
RelTol and AdjustBCs. The sixth positional grid size overrides options.N;
the wrapper overrides options.ODESolver as native pde15s does. Explicit Python
keywords retain precedence. Unsupported active options raise explicitly;
Plot='on', other ODE controls, periodic/complex/mass and native output grammar
remain open. No fixed-48 advection workaround or existing-bound change is made.
Four authored numerical controls, 17 factory controls and three unchanged
heat/Neumann/Robin regressions pass. Four native fixed-grid controls also pass;
heat errors are 1.10e-9 / 1.21e-8 and nominal boundary error with AdjustBCs=False
is 3.12e-17. Spatial Eps forwarding is source-reviewed; full adaptive option
behavior is not qualified by these fixed-grid controls.

Plotting now distinguishes a named Color property from a color in LineSpec.
Only the former consumes a color entry when LineSpec has no color; source
marker/jump artists do not add entries. Native Chebfun and numeric probes
confirm the distinction. The 21-check render/API gate passes; one subsequent
comment-only edit has an independently verified executable-AST bridge. The
exact prior comment preimage was recovered and matches its original guard SHA.
Uppercase Python Color alias and full native graphics grammar remain open.

ThreeBody's two numeric blue-root plots now use the public adapter. Its whole
CPU run passes in 94.311 seconds with stable inputs, all 1,290 actual imports
preguarded and no survivors. All retained rational/pole/residue/grouping
artifacts equal the prior Python run byte-for-byte. Figures 2 and 4 now use
the source orange ellipse; all four remain 600 × 270 and differ in decoded
pixels. MATLAB accuracy, fonts/axes/subplot layout, performance and latest-main
CI/coverage acceptance remain open. The inventory preserves prior figure
observations and adds the new hashes without promoting full-page statuses.

Evidence: [PDE consumer](pdeset_consumer_cpu_20261017.json),
[color cycle](colororder_cpu_20261017.json),
[whole example](pde_color_whole_cpu_20261017.json).
Earlier entries retain their historical source/evidence scope.

## Latest CPU integration and complete figure references — 2026-10-17

The ThreeBody example now calls the public six/seven-output rational API,
public plotregion, and numeric point-quiver adapter. The residue polynomial
recurrence uses a fixed-shape JAX scan; 168 retained cases and 24,205 recurrence
states match the prior implementation exactly, with 105 component regressions.
Optimized CPU HLO removes the explicit barrier: this is evidence for the pinned
CPU cases, not a universal compiler-rounding guarantee.

The combined-source gate passes 135 tests with no skips or failures. All 1,421
loaded runtime files were frozen before execution. The whole example exits
successfully in 94.622 seconds, with all 1,290 runtime imports covered, stable
inputs and no surviving owned processes. Its saved rational inputs, p/q,
poles/residues and pole grouping are byte-identical to the prior Python
observation. This is not MATLAB accuracy or matched performance parity. Four
regenerated figures have the required 600 × 270 dimensions; all differ in
pixels. The cached pole discrepancy, automatic axes, color cycle, subplot
layout, fonts and raster parity remain open.

GET now covers all 38 original predicates plus 45 existing value controls and
one singleton control. The production correction preserves native per-column
delta cells and collapses a singleton to its matrix. Broader GET API and type
coverage remain open. Plotregion now draws source artists by default; optional
`decorate=True` retains the earlier presentation. Complex Chebtech curves use
the paired real/imaginary source plotting route. Numeric point quiver uses
native scaling with readable compatibility heads; eight native metadata cases
and 54 component tests qualify that scope. Native automatic limits and opaque
renderer equivalence remain open.

All 322 official HTML pages and 1,305 official reference PNGs are now archived;
current HTML is byte-identical to the cached target. At source 43e5a5b7 all
1,305 committed PNG dimensions match the reference, and none is decoded-pixel
identical. The new four ThreeBody images have their own source-bound audit.
Reference acquisition and matching dimensions do not verify figure content or
computations. Latest-main CI and coverage acceptance remain required.

Evidence: [combined CPU review](residue_render_joint_cpu_20261017.json),
[GET](get_remaining_cpu_20261017.json),
[residue scan](residue_poly_scan_cpu_20261017.json),
[plotregion](plotregion_native_cpu_20261017.json),
[point quiver](quiver_points_cpu_20261017.json),
[complex plotting](complex_plot_cpu_20261017.json),
[reference archive](official_example_figures_cpu_20261017.json),
[dimensions](reference_figure_dimensions_cpu_20261017.json),
[pixels](reference_figure_pixels_cpu_20261017.json).

Earlier entries below retain their historical source and evidence scopes.

## Running parity checklist and complete source inventory — 2026-10-17

Tracking now has a repository [completed/to-complete checklist](parity_checklist.md),
a [scope/method summary](parity_inventory.md), and machine-readable
parity_inventory.json with2333native library .m sources,1102original test files,
and322cached example pages.3707library/1256test regex declaration counts are
not public API counts. Every full-item parity status begins unverified; candidate
path mappings never confer qualification. Historical scoped evidence is linked
for selected original tests and the ThreeBody observation; other evidence needs
reconciliation into rows. The source-only generator records source pins/hashes,
requires clean mapping inputs, separates MATLAB/Python script paths and refuses
to overwrite the live ledger. Existing checked scopes remain in the checklist.

ThreeBody's optimized whole run now completes131.896s/3.75GiB withboth6/7output
calls and4figures600x270, stable inputs/no survivors. This is an observation,
not page/native parity: cached first8poles differ1.26814e-5 and one actual
stdlib runpy.py import was not preguarded. Native rendering must replace its
local plotregion; highdegree correctness, printed output and pixel parity
remain open. Evidence:three_body_optimized_observation_cpu_20261017.json.

## Spinop2 GL authored test repair

Identical-input MATLAB and JAX normalized norm reevaluations both reject the
old authored32eps expectation. The check now uses512eps with both measured
values adjacent; the2e-14 pointwise comparison is unchanged. Fresh instrumented
focused test passes on isolatedf328 source, all1500actualimports preguarded;
fullstatic checks pass. Native diagnostic has one unavailable incidental prior
guard preimage; full norm/RNG equivalence and latest-main CI remain open.
[Scoped evidence](spinop2_gl_norm_cpu_20261017.json).

## Latest scoped package: polynomial conversion and CI process lifetime

CPU-only qualification:43 polynomial/rational integration checks and6 runner
protocol checks pass. Physical-domain conversion retains intermediate/final
bytes on retained cases; observed shared-host polynomial-pair time79.343s to
28.655s is not final MATLAB performance parity. Five existing3D shards use a
fresh process per file with identical selection/timeouts. Root archive and
runtime-import coverage verified. See [evidence](poly_port_joint_cpu_20261017.json).
Whole ThreeBody, all322 actual pages/figures, remaining original assertions and
API/correctness issues, matched performance and green latest-main CI remain open.

# MATLAB Chebfun parity checklist

Updated 2026-10-17. **Active and incomplete; CPU only.** Reference:
`7574c77680d7e82b79626300bf255498271a72df`. Checked entries certify only the
scope stated. There is no verified overall completion percentage or current
remaining-page count.

## Completed, scoped work

- [x] Pinned JAX/JAXlib 0.11.0 CPU/x64 environment and direct-main publication.
- [x] C1/C2 source coefficient-grid composition; all 44 original Tech compose
  predicates and six one-column controls now pass. The former 22 adapted
  controls remain supplemental. See [composition evidence](chebtech_compose_original44_cpu_20261017.json).
- [x] Minimax QR SVD and native exchange-error reporting; eight combined
  minimax/power/endpoint regression checks pass with the composition changes.
  See [integration evidence](chebtech_compose_original44_integration_cpu_20261017.json).
- [x] Scoped mixed linear-system, source AD, parameter boundary-condition,
  derivative-order, roots, Airy, public Trigtech and Spherefun corrections.
  These are not whole-module completion claims; see HANDOFF_CODEX.md.
- [x] Original mapped PDE technology-session predicate passes unchanged;
  this differs from the still-failing authored fixed-48 advection control.
- [x] Cached structure for 322 pages, 2,408 code blocks and 1,305 figure slots
  aligned. This does not qualify computations, printed output or image pixels.
- [x] All 13 original handoff issues and eight smaller subitems reconciled
  against source/evidence: [issue-by-issue list](handoff_issue_reconciliation_cpu_20261017.md).
- [x] CI timeout repartition preserves test membership and existing gates;
  actual latest-head runtime and success remain unverified.

- [x] Public C1/C2 nonperiodic composition preference propagation: 52 scoped
  controls across preserved runs; source first-piece fallback, raw-operator
  fixed/Turbo behavior and custom refiners retained. See
  [preference evidence](nonperiodic_compose_preferences_cpu_20261017.json).
- [x] Exactly-real complex negative-axis Airy correction: 39 first-candidate
  controls, final affected-control repeats and current public integration.
  See [joint integration](compose_airy_joint_cpu_20261017.json).

- [x] Public smooth Bndfun.compose and constructor DATA scales, with nine
  restored original predicates and preserved supplemental legacy controls.
- [x] Cross-kind C1/C2 outer composition and native minimum-length/column
  checks. Combined qualification: 81 controls, including the nine originals.
  See [combined evidence](bndfun_crosskind_integration_cpu_20261017.json).

- [x] Source inverse callbacks return raw algorithm results; optional final
  rangeCheck remains. 52 test nodes pass across separate frozen processes;
  this is not 52 literal MATLAB predicates. All actual callback samples match
  native for the retained input; transform length/speed parity remains open.
- [x] Real-valued Fourier CDR outputs retain the dtype of evaluated products;
  20 distinct controls include all three prior CI failures and complex/JIT/AD
  controls. See [joint integration](inverse_cdr_joint_cpu_20261017.json).

- [x] Large-rank CDR evaluation used by Poisson: all nine original Poisson
  assertions pass unchanged in 193.16 seconds, within the existing 900-second
  timeout. Equal genuine C2 slices use ordered matrix Clenshaw; 32-column
  stacks and a final block concatenation avoid the large assembly cost.
  All 26 authored value/AD/fallback controls pass. See
  [Poisson evidence](poisson_grid_chunk_cpu_20261017.json).

- [x] All 32 original Chebfun Subsref predicates covered through frozen
  component gates and an exact source bridge. Corrected JAX-only record
  helper passes 54 current controls; total 88 distinct component controls.
  Root 12 latest-source integration controls pass in 13.58 seconds.
  [Evidence and remaining record API gaps](subsref_jax_source_cpu_20261017.json).
- [x] Current source bindings for the original 12 interrupted scripts
  reconciled with their retained acceptance evidence. This is a source-only
  audit, not new numerical or figure qualification. See
  [script dispositions](handoff_twelve_current_bindings_cpu_20261017.json).

- [x] Native happinessCheck local DATA scale derives from the coefficient
  representation, preserving independent supplied-value scales and running
  constructor metadata. 153 component controls pass; 27 latest-source
  integration controls pass in 30.48 seconds. Native FFT and inverse length
  parity remain open. See [scale evidence](happiness_representation_vscale_joint_cpu_20261017.json).

- [x] Classify source AD linearity before optional mixed-linear preparation;
  the original natural C2 nonlinear parameter setup now passes adaptively at
  the unchanged bound. Final component13 nodes include11 routing controls,
  fixed48 supplemental and one literal adaptive original. Root latest-source
  routing11 and original-adaptive1 pass separately; C1/US remain open. See
  [nonlinear dispatch evidence](nonlinear_parameter_dispatch_joint_cpu_20261017.json).

- [x] Native programmatic record column selection now preserves max/if
  ordering, logical/complex/NaN quirks and computed outCols validation.
  Component116 controls pass; latest-source68 integration controls pass in
  18.89 seconds. Native48 comparisons qualify selectors/errors/shapes, not
  constructor/output-bit parity. Remaining query-type/recursive/property
  gaps are explicit in [index evidence](subsref_record_indices_cpu_20261017.json).

- [x] Natural adaptive first-kind numeric parameter backend: original C1
  passes at unchanged5e-10; original C2 regression unchanged. Root C1
  repeat44.028s, exact143runtime-source bridge into combined checkout.
- [x] Source numeric query classification:122component controls and fresh
  combined-source68integration controls pass19.075s. Logical/foreign
  nonnumeric queries reject without conversion hooks; direct Python calls
  retain their separate convention. [Combined evidence](c1_query_joint_cpu_20261017.json).
- [ ] Broader parameter/C1/US APIs, C2 factor eligibility policy, GET values
  and cell/foreign/property dispatch beyond qualified numeric recursion.

- [x] Skip redundant scalar AD extraction for known-local one-sided solves;
  two-sided source routing and nonlocal behavior retained. Fourteen unchanged
  metadata/source-route controls pass, plus the existing numeric system BC
  vector/length-error regression with consistent Python error text. Initial
  memory censor preserved, no benchmark/native-message parity claim.
  [CI routing evidence](ci_routing_controls_cpu_20261017.json).

- [x] Natural typed ultraspherical scalar-parameter backend: literal adaptive
  originalUS and unchanged C1/C2 regressions pass, with native tolerances and
  schedules. Latest-source originalUS repeat41.351s; classification6repeat
  passes19.167s, preserving601 CI fixes.
- [x] Source builtin numeric-result recursion:194distinct component controls,
  native65API capture and fresh latest-source72controls pass15.955s. Native
  public patterns are scoped to errors/shapes/selection; constructor bits,
  cell/foreign recursion and GET values remain open.
  [Combined qualification](ultras_recursion_joint_cpu_20261017.json).

## Correctness and original test parity

- [ ] Every original function and test predicate; remaining omissions, adapted
  assertions, skips, strict xfails and host NumPy/SciPy numerical routes.
- [ ] Singfun/Unbndfun/heterogeneous composition and native warning/API
  gaps; remaining invalid-input and future JAX list-DATA behavior.
- [ ] Authored fixed-48 advection control and chemical PDE accuracy.
- [ ] Small explicit Hermite ASY, nonlinear parameter/backend cases,
  interpolation norms/grids, broader Subsref records and Chebfun3T predicate.
- [ ] Strict composition/division representations, general rational poles,
  conformal inverse counts, IVP representation, QR/LU/QZ and other API gaps.
- [ ] Native normal RNG values and production integration/draw ordering.

## Pages, figures and speed

- [ ] All 322 actual computations and printed outputs; all 12 handoff scripts
  on the final affected source. Historical 295/322 and 27-page counts are stale.
- [ ] Current complete Tricky/Krylov qualification; DDE meshes/output/clipping.
- [ ] Every figure's data, RNG, cameras, axes, markers, fonts and raster at
  reference sizes; final regeneration and full audits.
- [ ] Matched native CPU inverse, Hermite, QR/LU, BVP, Bartels-Stewart and
  roots performance. Historical speed improvements are scoped observations.
- [ ] Green latest-main CI with coverage at least 79%, final JAX-only,
  provenance/lint checks and requirement-by-requirement acceptance audit.

Do not mark the goal complete while any required test, page, figure,
correctness, performance or CI requirement is unresolved.

## Interpolation / GET / rational output integration (2026-10-17)

- [x] All35 literal interp1 original predicates plus13authored/11supplemental controls; row/Y-Chebfun/trig JAX source paths.
- [x] Scoped GET120controls,33literal originals; root78repeat.
- [x] Rational explicit output-count/symmetry57controls, covering23original clauses; native dispatch/copy diagnosis and root57repeat.
- [x] Combined-source root48/78/57 allpass, stable8228inputs/allimports pregated/no survivors; fullstatic and archives reviewed.
- [ ] GET originals31/35–38 and broader cells/types/empty/family APIs.
- [ ] Actual high-degree ThreeBody/conformal computations, figures and remaining fitting/provider differences.
- [ ] Complete original function/test inventory, all322scientific/output/figure gates, CPU performance and green latest-main CI/coverage.

See [joint evidence](interp_get_rat_joint_cpu_20261017.json). Older open interpolation/GET/rational entries are superseded only within the scoped qualifications above.

## PDE option factory / recursive3D adapter (2026-10-17)

- [x] Public32field pdeset/display factory,17scopedcontrols; captured native NaN/complex/Inf metadata and per-function provenance. Parent fixes whole-vector conversion and pre-parser error order.
- [x] 3D full-slice token translation at recursive delegation; exact pinned badDomain authored expectation;42controls pass after two baseline failures.
- [x] Latest combined20nodes pass23.004s,8236stableinputs/all1440imports pregated/no survivors; fullstatic and completearchive reviews pass.
- [ ] PDE solver mapping consumption and fullfactory invalid MATLAB class/grammar behavior.
- [ ] Original3D1–5/12–18 latest qualification (whole-module attempt censored3GiB, no completedgate credit).
- [ ] Full originaltest/function/page/figure/performance/greenlatestCI requirements remain.

See [joint evidence](pdeset_core4_joint_cpu_20261017.json).
