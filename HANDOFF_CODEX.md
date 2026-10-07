# chebfunjax: handoff to a Codex agent (2026-10-02)

## Sphere real addition, CDR and default singular values (2026-10-07)

Real Spherefun addition now follows source pole extraction, parity grouping and
continuous QR/small-SVD compression. Structural negation and partition/combine
preserve source factor metadata and pole flags. CDR evaluation masks only infinite
reciprocal magnitudes, including zero/overflowing pivots; NaNs remain unchanged.
The sampled scale/iszero paths use that same mask while source raw reciprocal
checks and compression retain their original ordering. Default singular values
use JAX weighted column QR and source row QR/inner-product branches, including
complex pivots and the source nonconjugating transpose.

Combined CPU gate: **69 passed**, no failures/errors/skips,242.17s pytest,
peak sampled summed RSS4,708,516KiB. Exact collection/JUnit/runtime origins and
unchanged source/runtime inputs reconcile; no owned processes survived. Separate
46-case addition/CDR and23-case SVD gates also passed; earlier zero-pivot failure
and observer-binding failures remain archived. Full configured repository Ruff,
F821 and NumPy/provenance presence policies passed before publication. Existing
core CI discovers every new independent control automatically.

Qualification covers the selected real-addition/CDR/default-SVD behavior, not
full Spherefun parity. Complex addition and singular-function U/V retain legacy
paths. The public stored-factor rank property still differs from MATLAB spectral
rank; source plus clause4 presently uses that property and is not yet spectral-rank
qualification. Its source numerical-rank migration is a separate pending gate.
Variable-rank host dispatch is explicit; no whole-addition JIT guarantee is made.
Literal partition/plus bounds are retained; deterministic query adapters do not
establish MATLAB random-stream equivalence. Other legacy NumPy consumers remain.

Evidence root: `/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/`.
Archive `sphere_addition_svd_combined69_reference_v1_20261007.zip`, SHA256
`567be0b61c0cc57b71abfae81bf18e2b15e1f5c3000b02c147c9d73747c4e5be`,
verified52,432payloads with pointers to the separate and prior-failure archives.
It requires full base `corrected_runtime_union_v3_full_source_20261006.zip`, SHA256
`1e7c77153df60503d57a940869e2922174db86f7b55ab633d16a6225be2d90a7`.
Green CI on this commit and complete library/test/page/figure/speed parity remain open.

## Chebfun2 ACA rank budget and Localization output (2026-10-07)

Complete ACA now forces source grid refinement when the rank reaches its sampling
budget, including when the final residual is already small. This corrects the
Localization final field lengths from112/112 to the MATLAB/reference103/103.
The constructor port also restores whole-row quasimatrix reconstruction, the
source100-point query grid and repeated construction clause; bounds are unchanged.

CPU evidence:5 focused rank-budget controls and all4 actual example fields passed;
a separate8-test constructor/regression gate passed in130.15s with no skips/errors.
Collections, JUnit, runtime origins and unchanged inputs reconcile; no owned
processes survived. The first regression run's7pass/1failure is preserved: its
ported clause13 incorrectly reconstructed just one scalar row. The corrected test
uses MATLAB's whole array-valued operand. Arbitrary MATLAB preference-object API
syntax remains unqualified.

Actual example stdout matches all6 cached HTML output blocks exactly. Markdown
changes only the two final lengths. Both candidate images are610x276, but residual
rendering differences remain; this package does not establish figure parity or
complete disposition of the original12 interrupted scripts. Existing image/style
work remains under separate review.

Evidence root: `/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/`.
Archive `localization_aca_regression8_with_failure_page_reference_v1_20261007.zip`,
SHA256 `4e923795b37e7c5e713ac7f892e12194d922196d8c36e908a91d7609306f4403`;
verified72,339payloads including prior failure and full page evidence. Prior5-control
and4-field evidence: `localization_aca_rank_budget_pass_reference_v1_20261007.zip`,
SHA256 `53fa9e831fac70b035f1494f26cae94949ada8e57e6704f540ebbaf113f80667`.
Both require full base `corrected_runtime_union_v3_full_source_20261006.zip`, SHA256
`1e7c77153df60503d57a940869e2922174db86f7b55ab633d16a6225be2d90a7`.
Full library/test/page/speed parity and green CI on this new commit remain open.

## Sphere projection, refinement and norms (2026-10-07; verified local gate)

JAX BMC-I projection now applies source common coefficient simplification before
parity projection and handles the even Nyquist coefficient. PhaseOne rejects a
rank that exhausts its sampling budget, forcing the source grid refinement; this
repairs the reproduced partition regression without deleting small odd factors.
Scalar norm dispatch restores numeric infinity/even powers and numeric empty[];
Spherefunv norm is the source global component norm, with pointwise magnitude
kept as a separately labelled extension. Scalar/vector port CI runs in separate
processes; core shards discover all independent controls.

Local CPU qualification: **159 unique tests passed**, with no failures, errors
or skips: scalar101 in1024.47s, vector41 in1785.78s, and BMCI17 in36.67s
(pytest times). Exact collections/JUnit/runtime origins reconcile; guarded inputs
were unchanged and no owned processes survived. Vector peak sampled summed RSS
was7,696,348KiB. Its14warnings include unresolved/high-rank constructor controls,
pointwise magnitude and plotting; passing assertions do not establish global
resolution of those fields or visual acceptance. Full configured repository Ruff,
explicit F821 and NumPy/provenance presence policies passed on the publication
candidate. Fresh CI on this commit is still required; preceding P120 CI failed
its sphere rank timeout.

Evidence root: `/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/`.
Archive `sphere159_pass_reference_v1_20261007.zip`, SHA256
`1037b00cd7791d21bbf6d9c685cb4db32dedb6cf1223fbdd9c782aef9a03f74a`,
verified52,528payloads. This reference archive requires the full base
`corrected_runtime_union_v3_full_source_20261006.zip`, SHA256
`1e7c77153df60503d57a940869e2922174db86f7b55ab633d16a6225be2d90a7`.
The union guard is archival only; original per-process guards/receipts and prior
failed/censored run pointers are preserved. See
`sphere159_archive_prepared_v1_20261007/terminal_review.json`.
Focused evidence already established37 BMCI checks,25 norm checks and7 rank-budget
checks, with original partition/rank regressions unchanged. The failed/censored
full102 attempt and its reference-input mutation incident remain preserved; they
are not accepted as a pass. One instrumented rank comparison observed about11.88x
faster constructor work; this is not repeated MATLAB performance qualification.

Scope remains incomplete: legacy NumPy SVD/constructor paths, literal source
addition/partition/Helmholtz test gaps, all-page numerical/visual acceptance and
other library issues remain open. Localization's pinned MATLAB103/103 lengths
versus Python112/112 is a separate confirmed ACA refinement gap under diagnosis.

## Rational minimax numerator/denominator outputs (2026-10-07)

MinimaxRationalResult.as_chebfuns() now returns source numerator/denominator
Chebfuns. Conversion follows physical-domain sampling, denominator sign,
numerator-only stored scaling, exact support collisions and source simplify
cutoff applied to original coefficients. Polynomial and zero/odd branches are
covered. New arithmetic uses unconditional JAX transforms even with JIT disabled;
legacy solver NumPy dispatch remains an unresolved migration.

Exact candidate CPU gate: 62 passed, no failures/errors/skips/xfails, 74.32s pytest.
This includes11 new conversion controls and51 adjacent minimax/transform checks.
Runtime origins and input hashes passed, no surviving owned processes;
supervisor149.592s, sampled summed peak1,723,908KiB. Existing convergence and
unused CUDA-plugin warnings were preserved. Independent review confirms exact
collection paths/nodeids, candidate bytes and source bounds. The unchanged new
test file is published under misc rather than scratch utils so existing CI
shards include it; both inherit the same port-tree conftest.
Evidence: minimax_rational_pair_v2_reference_20261007.zip SHA
c429ced70a0bfd018ae19c6683246c32702e9fb1840b35aa110a9e58c21c8a43,
requires preserved full base/receipt below. This pair may be ill-conditioned;
use result.r for stable evaluation. CF initialization, CDF retries, source
messages and full BestApprox page parity remain unresolved. New CI pending.


## Keyhole source computation and observed plot defaults (2026-10-07)

KeyholeContour now constructs the joined complex Chebfun with the source's
separate complex powers, then integrates `log(z)*tanh(z)*diff(z)` through library
operations. It replaces the separate hand-derived segment derivatives. The
source-observed MATLAB line widths are 0.5 points; the reference blue is retained.
Actual isolated CPU output: I = 2e-15 + 5.674755637702230i, absolute error
6.498891186147061e-15. The actual stdout and 510x388 figure are published.

Run passed with source/runtime origins verified, stable inputs and no survivors;
wall153.545s, sampled summed peak1,986,756KiB. Evidence layer
697ba91949f8760ccc4334df6134f6a49f0334819d183fa35965863dff2796fa
requires the preserved full base archive/receipt named below. Actual MATLAB
metadata is in explain_v1_matlab_source_evidence_20261007.zip (SHA
ace59b5ada86dca8e47ccd19e4dde655bb06db4c250982a5b1047ea379718d54).
Reference/source-runtime color, raster stroke, axis and roundoff differences
remain documented; this is source-computation progress, not full pixel parity.
New publication CI needs verification. Full goal remains incomplete.


## Quiver source scaling and vector calculus page (2026-10-07; takes precedence below)

Two- and three-component planar quiver plots now share JAX scaling and open-head
geometry from MATLAB's readable compatibility routine. The source default factor
is .9; zero disables autoscaling; a one-point grid uses the upper endpoint.
Three-component quiver preserves the source wrapper's consumed-numpts behavior
and uses 20 points, while direct quiver3 honors its own count. Explicit native
Matplotlib scaling options retain their separate interpretation. Twelve actual
R2025b preflush ScaleFactor records support the arithmetic; this does not prove
opaque renderer or pixel equivalence. Surface-coordinate overloads remain open.

CPU qualification: 40 scientific tests passed, zero failures/errors/skips,
including 35 new source/geometry/API controls and five adjacent plotting checks.
Runtime origins and input stability passed; no owned processes survived. The
controller exited 4 because pytest omitted the filename prefix for two external
scratch cases. That result is preserved. A separate verifier and independent
review establish those cases from exact bound source/parameters, module origin,
command selection, nodeids and JUnit; no test was changed or repeated to hide it.
Supervisor wall 243.99s; peak sampled summed RSS 1,449,064 KiB.
Test evidence layer: d230d6189362cf7fcb9bb121b974a2f953bccf72d5c89afe5c957e6b147fde9b.

CheckingVectorCalculus now uses source constructors, vector arithmetic/divisions,
library quiver and curve plotting, and continuous Frobenius/L2 norm instead of a
maximum over 30 diagonal samples. Its isolated CPU run passed and produced the
source 600x400 figure. Actual final norm: 2.047447252784797e-15. An independent
x-y control gives L2 1.6329931618554523 despite nearly zero diagonal samples.
Actual stdout and image are published, not copied reference results. Runtime
origins/input stability passed; no survivors; wall 195.63s, peak 2,165,904 KiB.
Page evidence layer: ea4eb3425769997d42e450e6474f52539d94138f7b0e12ab74197e1070e0b11a.
Small reference-image position/stroke differences and roundoff-output differences
remain; full page/figure parity is not claimed.

Five undefined guide02 plot_pieces calls now use the public plotting library;
missing Chebfun2/Chebtech2 annotation imports are fixed. Full configured Ruff,
global F821, whitespace, provenance and NumPy policy checks pass. New publication
CI requires verification. Full library/test/page/performance parity is incomplete.

Both evidence layers require preserved full base
1e7c77153df60503d57a940869e2922174db86f7b55ab633d16a6225be2d90a7
and receipt d3880112810f60a7945e1625dc9030de254bf1e6686111c755d8e050363dc826.
They are not self-contained. Current backlog: parent CURRENT_GOAL_STATUS.md.


## TYPE2 rational source evaluation (2026-10-07; takes precedence below)

The TYPE2 nonconstant-denominator handle now follows pinned `ratinterp.m`
first-kind barycentric evaluation on Chebyshev second-kind nodes. It preserves
source node bypasses, singleton weight behavior, complex component arithmetic
and observed nonfinite outputs. New evaluator math is JAX-only; the existing
fitting implementation is unchanged and is not claimed fully migrated to JAX.

Four fresh serial CPU processes passed **197 tests**, with zero failures, errors
or skips (45 source controls, 27 MATLAB ports, 80 rational utilities, 45
trigonometric utilities). Actual runtime origins were checked at each group's
start/end; all bound inputs stayed stable and no owned processes survived.
Supervisor wall time was 409.96s; peak sampled summed RSS was 3,844,880 KiB.
Configured full Ruff, changed-file F821, whitespace, provenance, NumPy import
policy and required golden-reference presence passed.

The nonsmooth MATLAB test now restores the original `f-p/q` norm. Three added
Python analytic checks retain their original accuracy bounds on polynomial
ratios, while separate source controls test returned-handle behavior. Actual
MATLAB R2025b public calls confirm complex TYPE2 mu=0 returns NaN components
and the high-degree real mu=0 case returns negative infinity; these outcomes
are preserved rather than replaced by the ideal finite rational function.
Source observations and fixture bytes were independently reviewed.

Nonfinite complex adaptation is bounded to the observed scalar off-axis cases;
axis/zero/nonfinite factors and general BLAS bit identity remain unqualified.
Query JVP/VJP controls pass at finite continuous points; construction remains
eager, and traced invalid inputs have an explicit NaN convention. Sequential
real denominator accumulation preserves the observed cancellation control but
has no general MATLAB reduction or performance-equivalence claim.

Verified evidence reference layer SHA256:
`1ca454634549ebb1e9497915784416b825ad3cd35151ad6f6d0eb8312f50bc76`.
It is **not self-contained**: retain full base
`1e7c77153df60503d57a940869e2922174db86f7b55ab633d16a6225be2d90a7`
and receipt `d3880112810f60a7945e1625dc9030de254bf1e6686111c755d8e050363dc826`.
Actual MATLAB source archive hashes are recorded in the fixture provenance.
New publication CI still requires verification. Full library/test, example,
figure and performance parity remain incomplete; current backlog and evidence
are in the parent shared scratch `CURRENT_GOAL_STATUS.md`.


## Coefficient-input complex dtype repair (2026-10-06; takes precedence below)

`Chebfun.from_coeffs` no longer forces coefficients to float64 and discards their
imaginary parts. Existing downstream dtype promotion preserves float64 for real
inputs and complex128 for complex inputs. This matches the public coefficient
constructor and pinned MATLAB coefficient-input semantics.

Isolated CPU evidence: **17 tests passed**, covering real/complex vectors,
singletons, one/multiple columns, JIT enabled/disabled execution and integer
promotion. Exact coefficient checks and independent degree-two polynomial
values pass without changing bounds. Pytest took 27.65s; sampled peak summed RSS
was 889,616 KiB. Runtime start/end checks passed; all 49,524 inputs stayed stable
and no owned processes survived.

The frozen driver incorrectly expected 33 cases and therefore exited 1 after
pytest passed all 17. This bookkeeping error is preserved in the evidence. A
separate verifier checks the actual 2*4*2+1 parametrization and all 17 JUnit names;
no scientific test was altered or rerun to hide the failure. This is narrow
scientific qualification, not a claim that the old driver or full core suite passed.

Evidence reference layer SHA256:
`4ce1b12814da193c68ce008a23b7e724f1558b4826e185c2c7cf42da03d7ab30`.
It is **not self-contained**: reconstruction requires the verified full base
`1e7c77153df60503d57a940869e2922174db86f7b55ab633d16a6225be2d90a7`
and its receipt SHA `d3880112810f60a7945e1625dc9030de254bf1e6686111c755d8e050363dc826`.
All logical payloads remain required. New publication CI must still be checked.

The separate rational evaluator remains unqualified and excluded from this
publication. Its source-backed nonsmooth test correction and complex arithmetic
controls are in progress; a reverse-mode undefined-node derivative failure was
found and archived. Full library/test, 322-page/figure and performance parity
remain incomplete. Current work and exact evidence are in CURRENT_GOAL_STATUS.md
in the parent shared scratch directory.


## Adaptive scalar and complex boundary repair (2026-10-06; takes precedence below)

The cohesive operator candidate passed a CPU union of **333 tests**, with one
existing strict Carrier expected failure, zero unexpected failures/errors and
nine warnings. Original source assertions and bounds are retained. Actual pytest
module origins and native library paths were checked at start and finish against
prebound runtime files; all 63,913 declared inputs remained stable and no owned
process survived. Pytest took 1500.72s; peak sampled summed RSS was 5,936,288 KiB.
This establishes the tested package, not complete library or example parity.

Verified full-source/runtime evidence archive SHA256:
`1e7c77153df60503d57a940869e2922174db86f7b55ab633d16a6225be2d90a7`.
The runtime bindings correct an earlier environment-origin gap. Older archived
runs are not retroactively claimed to have complete actual-runtime bindings.

Adaptive scalar solves follow source defaults 32/4096/5e-13, order-adjusted
Cheb2-to-Cheb1 projection, boundary-first scaled assembly and source output
chopping. Supported operator trees use discrete Chebyshev coordinates, retaining
finite-grid multiplication, interpolation and tree composition. This is a
floating-point representation adaptation, not MATLAB nodal roundoff identity.
Unknown capabilities retain the nodal path. Domain and point-value checks guard
dispatch. Complex boundary/forcing promotion also affects fixed-size solves;
eigenvalue routing is unchanged. Chebtech1 singleton conversion retains complex
values. All ten original boundary-condition clauses pass at their source bounds.

Twelve serial CPU observations at base dimensions 32/64/128 measured one first
call after setup and three warm calls per fresh process. Coordinate cubic warm
medians were 0.175–0.188s versus nodal 0.032–0.033s; variable-composition medians
were 0.130–0.138s versus 0.013s. Every coordinate chopped query/equation/boundary
check passed at 1e-10. Nodal cubic results failed query/boundary checks; both
variable-composition paths passed. Unchopped cubic residuals failed even on the
coordinate path, so source output chopping remains essential to these results.
These are shared-host observations, not a statistical performance qualification,
a MATLAB timing comparison or evidence for dimension 4096. The coordinate path
has measured small-size overhead; optimization remains open.
Performance evidence archive SHA256:
`033df87baea338dd70c7dfce489ee3ee0ca687f06e78a37f33a63b3d38f5579f`.

Configured repository lint, changed-file F821, whitespace, provenance, NumPy
import rules and required golden-reference presence passed. Exact publication CI
must be checked after pushing; the preceding main commit ecbb324 had green CI.

Carrier damping, general session preferences, whole-block construction JIT/AD,
high-dimension 4096 memory/time, constructor warnings and full library/test
parity remain open. No new page or figure is qualified by this package. All 322
pages still require full prose/computation/output/figure acceptance; the twelve
original interrupted edits retain their documented unresolved disposition.
Separate rational-evaluator and plotting drafts remain uninstalled.

Use shared scratch, CPU only, serial heavy runs, at most two helpers, explicit
staging, ordinary SSH push and the GPT-6 trailer. Current evidence and next work
are in parent CURRENT_GOAL_STATUS.md.


## Initial-guess source repair (2026-10-06; takes precedence below)

A narrow correction on published P113 is CPU-qualified: **17 passed**, no
failures, errors or skips, 135.802s driver. This covers all seven original
`tests/chebop/test_linearize_init_fails.m` clauses, eight independent error
boundary controls and two valid coupled-system regressions. All 19,989 frozen
inputs stayed unchanged; no owned process survived. Six constructor warnings
remain. These tests establish this package, not full-suite parity.

The old clause1 sampled maximum was an incorrect port. The original uses
continuous L2 at the unchanged 1e-10 bound. Fresh pinned MATLAB passes all
seven source clauses: clause1 L2 is1.552e-11, while the same sampled maximum
1.196e-10 also fails on MATLAB. Source clause6 keeps the chebmatrix Frobenius
norm and original 1e-8 bound. Missing clauses3/6/7 are restored.

The system solver translates only the two exact source domain error
identifiers during initial operator evaluation. Boundary callbacks, unrelated
errors and later finite-difference probes retain their errors. MATLAB maps
these IDs on each linearize invocation; later FD/AD equivalence is unqualified.

Full-source qualification archive SHA256:
`3b32dab30306c7b259248865473e9410b1a1ef32f885f17373d1b33c44ed29f8`.
A separate 16-case candidate gate, including the actual CI mirror, passed;
its archive is `0ffd5c23112c20fd09a0f59f0bea33aa4d39b403410ebc3cbb8eabacc6682f5a`.
The fresh MATLAB observer exits1 during later metadata/JSON capture; its
seven-clause source result and partial numeric captures are preserved in
`bcba6925728f7f1c98d14d61881ef27d1957edf54fd96a144c64ac0a45a1a747`.
Do not label that observer a completed batch pass.

This publication excludes the unqualified adaptive/complex operator candidate.
Its default cubic accuracy failure remains open, as does one variable-coefficient
equation residual in a same-dimension coefficient diagnostic. Original BC10 and
23 complex functional controls pass separately; full candidate acceptance is
pending. P113 CI failed only the duplicate incorrect port assertion; the new
publication's CI is pending. All library/test, 322-page/prose/output/figure and
performance parity remains active and incomplete.

Use shared scratch, CPU only, serial heavy tests, at most two helpers,
explicit staging, ordinary SSH push and the GPT-6 trailer. Current evidence
and next steps are in parent `CURRENT_GOAL_STATUS.md`.

## Qualified CPU package (2026-10-06; takes precedence below)

P113 operator correction is qualified on CPU: **16 focused tests** pass in
362.941s, followed by **110 passed, one existing skip** across 27 broader
regression files in 893.536s. No failures or errors; all 1,668 Python files
and 15,248 outer inputs stayed stable, with no surviving owned processes.
The gates overlap; these are not 126 unique tests or full-suite parity.

General scaled-jump constraints now use the complete residual Jacobian and
reject nonfinite Newton candidates. The original jump predicate and an
independent analytic solution with all four constraints pass at 1e-10.
Supported real scalar IVPs select the actual native113 default; failures
propagate. Restarted pulse representations retain their breaks; restart-off
behavior matches the original MATLAB no-pulse comparison at 1e-10. All
three source domain checks use exact equality. Explicit LSODA controls
retain the previous adapter agreement bound without changing assertions.

All ten original integral-operator clauses are restored, including the
source L2/infinity norms and periodic Fredholm weighted-kernel assembly.
The earlier sampled-max Volterra port was incorrect; its source L2 error
2.608e-11 passes unchanged 1e-10. Source case5 assigns a numeric expression
to a logical pass vector, with no accuracy bound. Fresh MATLAB confirms
its 0.528 total is an interior value term, not an equation/left-BC error;
equation L2 residuals are 5.743e-14 MATLAB and 2.064e-13 Python.

Final broader full-source archive SHA256:
`21af4521bfb724a2e9f4124a7d4577b6c496aa7eaddc52daad5e5f0f9102381e`.
Focused archive:
`414aac85b60004fc2d4c367212d3c4c05a7dcf940212d5e23b46af3dcf301393`.
Fresh case5 MATLAB archive:
`f1e67683224ba371b8048d57b3d3061b79a07a0e2089ee08bb8e15e2cca7e13b`.
After archiving, a TYPE_CHECKING-only Chebfun import resolves two preexisting
F821 annotations in integral.py; every function/class AST remains identical.
This amendment is checked statically, not relabelled as the earlier run.

P112 d2906a406c2fc6ca129f409cb91c21cfbcc2a25a is pushed. Its three core
CI shards and final combined **79%** coverage gate passed. Operator-b/c
failed only the known integral, jump and pulse cases addressed here; techs
and the new publication's CI remain unresolved. Prior periodic package:
129 focused and 617 broader CPU cases passed, with all 23 original MATLAB
multiplication predicates checked on captured primitive RNG inputs.

Next: qualify the complete ten-clause boundary-condition port, replacing an
incomplete loose port and its stale Neumann-string skip. One 65,537-point
unhappy-constructor warning in intops remains unlocalized. General BC AD,
public IVP operator-break routing, remaining options, full integral APIs,
all MATLAB functions/tests, twelve protected script edits and all 322 page/
output/figure/performance parity remain **active and incomplete**.

Use shared publication_checkout, ordinary SSH push, explicit staging and
GPT-6 trailer. CPU only; heavy runs serial; at most two helpers. See parent
CURRENT_GOAL_STATUS.md for immutable evidence and the next gaps.

## Latest shared checkout status (2026-10-06; takes precedence below)

The next bounded singular/Gamma correction is qualified:72focused CPU cases
pass64.531s and131unchanged prior regressions pass302.101s, no failures,
errors or skips; all1659Python files and frozen inputs stable. Exact tiny
locator grids preserve gradual-underflow/signedzero with JAX integer
arithmetic;51controls include actual MATLAB bit comparisons. Singular
outer construction captures callback pointValues and merges introduced
breaks only, preserving given boundaries and existing merge preferences.

Gamma port inputs now match fresh MATLAB's +Inf at finite nonpositive
integers (both signedzeros), while retaining SciPy off-pole bytes. Original
nineGamma assertion ASTs/bounds are unchanged; six actual primitive pole
controls, two off-pole controls and independent analytic values pass. This
is a test-input compatibility adapter, not a general library Gamma primitive
or universal finite/complex equivalence claim. Fresh pinned MATLAB gives
fivehappy pieces and NaN/Inf/14.043323986892394 integrals; rawMAT/JSON/input
words and source hashes were independently verified. Original SciPy callback
caused spurious tiny panels and scale pollution; failed21/9piece versions
remain preserved. Source callback and constructor counts are not inputs.

Focusedfullsource archiveSHA
81a66aeaf6f7158e1946b5d44ca0e2363a5271c983af6cce0a32462419c14e5f;
regressionarchiveSHA
d6eb2c18e1fad1b052fa718d7bfd8551e7f0df83c326d440451bb62cb088c9ef.
A threeway operator CI partition covers all99testfiles exactly once33/33/33,
retains assertions/900s test timeout; runtime balance remains unqualified.
CI on this new commit, Gamma page's callback update/standalone run, all322
page/figure parity, C1 and wider singular policies remain open. NonsmoothFOV
V12 ran226.769s/all14figures600x253, but cropped labels/outputcounts/pixels
are unaccepted; rendering-only V13 and observational audit await cleanhead.
Fullgoal remains active; CURRENT_GOAL_STATUS.md points to current evidence.

Full MATLAB7574c77 and322-page parity remains active and incomplete. Work in
`/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/publication_checkout`.
Read `CURRENT_GOAL_STATUS.md` in its parent for current evidence and next gaps.
The original home checkout and twelve interrupted scripts remain protected;
write only under shared scratch. Normal SSH push and licensed CPU MATLAB work.
Use gh-cli, explicit staging, GPT-6 trailer and full Ruff gates. CPU only;
heavy examples/tests serial; at most two helpers, Astra medium for numerical
review as requested and cheap routine execution. Freeze qualification inputs.

P105 535fef0e89a44c7bcdbee8f94d15e2bc354e846f is pushed. Combined48
coast/source-grid colorbar cases pass37.971s; archive verified. New API retains
corrected C and a shared norm snapshot. V1 new exact-edge assertion failed
because Matplotlib inverse normalization differs oneULP from fresh MATLAB
CLim; gamma2 operation bound documented, production unchanged.

P106 fc686d969375741547bb785126278bb663f2863b is pushed:117CPU checks pass,
including originalIVP1e-14/exactsyntax equality, metadata/typedscalar/source
sampleTest controls. Fresh MATLAB6IVP assertions and pointValues preservation
pass68.684s. Existing pointValues survive simplify/common padding; accepted
integer0Dscalars adapt atChebfunboundary; lowertech diagnostics remain. Exact
P105 baseline reproduced all4historicalCI failures and3metadata failures.

The next bounded singular source correction passed114CPU checks in218.757s,
including the original strict oscillatorycube, sourceconstructor13/15/16/17,
sourcepowers32-38/sqrt24, Singfun, complex and ordinarydetector controls.
All1643qualifiedPython files and selected inputs stayed unchanged. Fresh
MATLAB originalpass23 passes88.690s; its100seed6178sites are bit-identical to
our existingadapter, now checked against an exacthexfixture without replacing
samples. Sourceedgeprobe passes69.562s: blowupTrue/globalvscale0 chooses a
finitepeak whileFalse orscale1 bisects. Candidate selects the same first
bracket/edge within1ULP. Literal nestedblowup/globalvscale and third-point
endpointzoom replace the median-scale wholeinterval prepass. New math isJAX;
all original numericalbounds unchanged. No fullpartition/rounding guarantee.
Legacy dead NumPylocator cleanup, nondefaultcaps/tinyinterval/budget/merge
semantics and C1 remain open. Full latest-main greenCI and Atmospheric/pixel
parity are unverified; P106basicchecks/docs pass, numericalCI remainsactive.

The source singular/unbounded merge correction is qualified: 15 focused
checks pass in 91.303s, 331 existing regression checks in 422.262s, and two
independent public complex-query controls in 4.874s. No failures or skips;
all frozen inputs are stable. Original source assertions12/13 and their
captured MATLAB RNG sites/bounds are unchanged. The clean baseline exposes
an unbounded error0.7662 and unsupported singular breakpoint removal; a
missing new helper is reported separately. Source12 alone admits a no-op,
so actual analytic breakpoint removal is also required and passes.
Literal outer exponents/full endpoint limits, fresh union-owned maps,
global unbounded hscale1 (including bounded intermediate trials), Inf-only
discovery and source exponent negation are retained. New math is JAX;
root metadata is zeroed only after new breaks. Unbounded queries preserve
complex coordinates, checked eagerly/JIT on both semi-infinite domains.
The unchanged pinned MATLAB merge test passes14/14. Earlier qualified files
stay byte-identical; the last gate freezes1650Python files including the
new query test. Full-line complex behavior, unsupported/periodic adapters,
generic constructor policies and full suite/page/CI parity remain open.

The exact-computed-repeat residue correction passes151CPU checks in94.851s
on final publication bytes, zero failures/skips/errors;1652Python files and
selected inputs stay stable. It computes the first Laurent coefficient by
JAX synthetic deflation/Taylor division, preserving denominator scaling,
and copies A1 across exact-equal groups as the source wrappers do. Ordinary
simple/no-pole behavior and six-versus-seven-output trig definitions remain
qualified. Twenty-two new analytic cases include real/complex/improper/
triple/physical scaling, actual zero-root wrapper reachability and separation
of nearby distinct poles. Clean baseline22failures are19missinghelper imports
and3existing unsupported wrapper operations, recorded separately. Fresh
MATLAB source/builtin probe passes70.116s and confirms A1 equations. It also
records TYPE2mu0 division and complex trig-conjugate source limitations;
those diagnostics are not substituted as analytic acceptance answers.
Builtin near-root clustering, regrouped physical poles and high-degree basis
conditioning remain open. Installed R2025b builtin has two grouping stages;
no guessed threshold is added. V1numeric151pass evidence and failed I001lint
are preserved; only importorder/provenance docs changed, AST/symbol proof
matches, and finalV2 rerun passes. Full residue/API/CI parity is not claimed.

The opt-in source coefficient plotting policy passes27CPU checks in34.818s,
zero failures/errors/skips, all1654Python and frozen input hashes stable.
Fresh pinned MATLAB12literal cases captured71.064s verify coefficient line
values, Fourier global-domain normalization, marker formula, labels/grid and
explicit X limits. Source hold defaults off (clears supplied axes); hold=True
keeps artists and axis scales. Each tech updates limits before the next piece
inherits held/manual limits. Arrays separate columns before zero-scale checks;
Singfun forwards its smooth part. New coefficient/marker arithmetic is JAX,
conversion to NumPy occurs at drawing. Original11source smoke assertions,
legacy data/envelope/default colors and existing adapters pass. Clean baseline
12failures are missing-API availability evidence, not12numericbugs. Default
palette maps the graphics root to Matplotlib rcParams; explicitNone defaults,
autoYlimits/ticks/pixels/barplot/long-format adapters remain scoped limits.
One endpoint-discontinuous constructor used by the color control emitted an
unhappy65537-point warning; this is not constructor acceptance and requires
separate source diagnosis. Full page/figure/CI parity remains unfinished.

A matched single-CPU current ComplexArcLength inverse diagnosis completed:
MATLAB1.570572/0.869868s and JAX23.139/0.745728s first-after-setup/warm.
Same100interior-target inverse values differ at most2.22e-15; JAX roundtrip
3.05e-15. Single observations, not fullpage or repeated performance claims;
the old800x report predates current Brent. The outerMATgroup monitor missed
GNUtimeout's worker group, so its18MiB reading is invalid; inner GNUtime
reports1308832KiB. Preserve evidence and fix future descendant supervision.
RegulaFalsi/Illinois source-policy correction is drafted, not qualified.
NonsmoothFOV freshsource/input-replay candidate still needs actual execution
and14reference-sized figure/page audits after the plotting dependency.

Exact MATLAB rng(1);randn(60) primitive input capture passes71.568s with
independent reseeding/state/bit checks. All3600column-major binary64 values
match raw MAT and JSON independently; generator twister/transform Ziggurat.
It contains no FOV answers or coefficients. NonsmoothFOV source sequence
and primitive replay drafts are not yet executed or accepted; a general
MATLAB-compatible normal generator remains unfinished.

P104 8e0155ec666a4aa05db0196b0076664ff7d3c9c1 is pushed with the62-case
short-sum/channel and camera qualification below. New CPU exp diagnostic13
selected inputs/12paths/threeflagsettings found identical output bits; HIGHEST
requested in HLO did not repair one-ulp errors. C1 remains open.

A coastline renderer correction passed41focused CPU cases in30.062s;
new general colorbar adapter is under combined frozen qualification: actual registered
opaque sphere artists govern masking; hidden/removed/transparent artists and
user zorder are respected. Conservative JAX ray arithmetic retains ambiguous
front boundary points without moving CoastData. P104 actual-artist baseline
two true rendering failures and one harness import error in first gate;
corrected-import isolated baseline and complete candidate gates are recorded. This is analytic sphere
visibility, not exact faceted-mesh depth; near-limb segments, source framing,
full pixel parity remain open. The opt-in scalar mappable uses the exact
corrected plotted tensor C and same normalization; it is documented as a
snapshot, preserving explicit/bumpy surface colors. Baseline7API controls fail
before this adapter. Constant CLim/face interpolation/custom norms remain open.

P103 ab9b4feaf51c1db30f1128c035ab7ec9d17ad541 is pushed. It replaces dense
adaptive sphere output with literal northern-grid construction using JAX FFTs
and preserves tensor plotting arguments. Frozen29small plus11solver cases pass,
including fresh5MATLAB API outputs, Poisson, four Helmholtz grids and source
Spinsphere trajectory. The existing host from_values constructor remains open.
Bounded warm constructors improved36.55/38.94ms to31.80/33.90ms, while cold
costs increased3.64/1.24s to4.91/5.53s; no full Atmospheric timing claim.

The next source short-array sum/channel correction passed62frozen CPU cases,
including7independent short-real controls,14camera controls and retainedP100
source/regression cases. Fresh MATLAB5public CDR fixtures pass68.909s. Source
SUM without a dimension can reduce across factors; scalar-times-D then retains
multiple channels. An earlier independent zero-pivot expected2pi was wrong:
MATLAB gives4pi, now corrected with the original bound. Source quirks retained.
Camera helper converts source azimuth50 to Matplotlib-40 and preserves projection
and layout. Coast clipping/framing/full pixel parity remain unqualified.
Generic complex addition, MATLAB query reshaping and full per-channel policy
remain open. Source-body/import-set proofs cover formatting-only post-gate edits.

AtmosphericTemperature V11 failed signal9 after1093s/four of ten figures at
407GiB peakRSS; immutable failure archive is preserved. Rank185 and three
scalars computed, with final digit differences. Next rerun needs verified
rendering fixes and a fresh frozen full execution, page/stdout/figure audit.
MATLAB graphics metadata captured in96.799s, actual source/properties stable;
its software WebGL PNGs are nearly black and unusable as visual references.
Official cached1305figure slots remain available. No arbitrary95%viewport
fitting or source asset substitution is accepted.

P97 stable realatanh9passes, P98 compensated extrapolation72passes, P94Fejer
original18bounds pass; Fejer warm n100000 cost4.781x remains recorded. Focused
passes overlap; no full-suite or universal MATLAB rounding identity claim.
CI P98–P103 snapshots remain in progress, completed checks have no failures,
docs allsuccess. Latest fullgreen is not verified. C1 detector remains open:
a JAX exp one-ulp bump triggers the literal source zero-slope rejection; no
cutoff/tolerance/xfail change justified. All322orderedprose/2403inputs match,
11output-count gaps and154figure-size mismatches remain; computations/pixels/RNG
and exact-byte disposition of the12protected originals are unfinished.

Next: full Atmospheric run/figures, detector backend accuracy, remaining merge policies,
repeated-residue correctness, full functions/tests/JAX/RNG/performance and CI.
Drafts and failed/unrun revisions are preserved; no external blocker.

This document takes over from the Claude session that ran the parity
campaign from 2026-07 to 2026-09. `HANDOFF.md` (2026-07) is older and partly
stale: where they disagree, this file wins.

## 1. Mission and standing rules

The goal is full parity with MATLAB Chebfun (commit `7574c77`, source at
`/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref`). That covers every library
function and every MATLAB test, and every chebfun.org example page must be
replicated exactly: same prose, same computations, the same printed outputs
and the same figures at the same sizes.

Rules the user has set (all still in force):

- **Push directly to `main`; CI is the only reviewer.** Never force-push. Use
  `gh-cli` (not `gh`) for GitHub.
- **Stage with explicit pathspecs.** Never `git add -A` on the whole tree,
  since other agents may share it.
- **Library code is JAX-only.** Any file that imports numpy needs a
  `# uses-numpy: <reason>` comment, which CI checks. Every public function
  needs a `Provenance` docstring section naming the MATLAB source and the
  `Chebfun commit: 7574c77` line.
- **Tolerances.** Never widen a test tolerance without a comment explaining
  why, with the MATLAB value and ours.
- **Lint.** `ruff check src tests scripts examples` must pass before every
  push; CI lints `examples/` too.
- **Commit trailer.** End every commit message with
  `Co-Authored-By: <your model> <noreply@...>`.
- **Storage.** Never write data under `/home/mg6942`. Put scratch in
  `~/myscratch/tmp/<dated-name>/`, or `/scratch/gpfs/GILLES/mg6942/_agent_scratch/`
  for work that spans sessions.
- **Token budget.** Be economical: at most 2 parallel sub-agents, on a cheap
  model or low reasoning effort for routine script fixes.
- **Examples use the library.** They are translations: never re-implement a
  library feature in numpy inside an example. When the library is wrong or
  missing something, fix the library (with tests) or log it in section 6.

## 2. Environment: fix this first

**The pixi environment is gone.** `.pixi` is a symlink to
`/scratch/gpfs/CRYOEM/gilleslab/chebfunjax_pixi_env_20260702/.pixi`, which
has been deleted. Recreate it:

```bash
cd ~/chebfunjax
rm .pixi                                   # dangling symlink
mkdir -p /scratch/gpfs/GILLES/mg6942/_agent_scratch/chebfunjax_pixi_20261002
ln -s /scratch/gpfs/GILLES/mg6942/_agent_scratch/chebfunjax_pixi_20261002 .pixi
pixi install                               # ~/.pixi/bin/pixi
pixi run smoke
```

Always run Python as
`JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg .pixi/envs/default/bin/python ...`.
Without `OMP_NUM_THREADS=1`, linear algebra on the login node is about 70x
slower.

MATLAB R2025b is available for reference runs:
`module load matlab/R2025b; matlab -batch "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); run('x.m')"`
(about 1 minute of startup).

The MATLAB example sources are at `/scratch/gpfs/GILLES/mg6942/chebfun_examples/<cat>/<Stem>.m`.

## 3. State at handoff

- `main` is at `ecfda8fc`, and CI passed on the last three pushes.
- 322 example pages (`docs/examples/<cat>/<Stem>.md`) are generated from the
  chebfun.org HTML plus our script output.
- A structural audit finds 295 of 322 pages whose printed output blocks line
  up with the original. The 27 that don't are listed in section 5.
- Example figures are saved with
  `chebfunjax.plotting.save_chebfun_figure(fig, path, size=(w, h))`. The
  default is 600x270; other pages use the size of the reference PNG in
  `/scratch/gpfs/CRYOEM/gilleslab/chebfunjax_audit_20260702/refs/docs/images/<cat>/`.
- `TEST_CHECKLIST.md` (generated by `scripts/gen_test_checklist.py`) records
  MATLAB test-suite coverage: 1069 ported, 26 skipped by policy (chebgui,
  adchebfun), 7 missing (chebgui).

### Uncommitted work in the tree: verify or revert

Interrupted agents left edits that have not been verified, in 12 example
scripts and their images:

```
examples/approx/best_approx.py              examples/pde/trapezoideigs.py
examples/approx/fermi_dirac.py              examples/roots/marching_squares.py
examples/approx2/localization.py            examples/sphere/atmospherictemperature.py
examples/fourier/fourier_based_chebfuns.py  examples/sphere/laplaceball.py
examples/linalg/vandermonde_arnoldi.py      examples/stats/resampling_random_variables.py
examples/ode-nonlin/delay_differential_equations.py
examples/ode-random/randomswitching.py
```

Modified images also appear under `docs/images/` for GlobalMinimum and
WhiteCurves, whose scripts are unchanged; those were overwritten by a test
run, so revert them with `git checkout`. For each script, run it alone, run
`unmatched.py` on its page (section 4), then commit or
`git checkout -- <file>`.

Four scripts are slow or failed when run in parallel: VandermondeArnoldi
(over 30 min), DelayDifferentialEquations (over 1 h, because piecewise delay
chebop solves are slow), RandomSwitching (crashed at about 25 min) and
AtmosphericTemperature (killed, probably out of memory, while other jobs
ran). Run them one at a time, or as a CPU Slurm job on a general partition
with about 32G of memory.

## 4. Tooling

The durable copies are in `/scratch/gpfs/GILLES/mg6942/_agent_scratch/chebfunjax_handoff_20261002/`
(called `$D` below):

| Item | What it does |
|---|---|
| `scripts/gen_example_pages.py` (in the repo) | Builds `docs/examples/<cat>/<Stem>.md` from the chebfun.org HTML (prose, headings, MATLAB cells, figure slots), our stdout and our images. `--stdout-dir $D/stdout --html-cache $D/pages_html [cat/Stem ...]`; with no pages it does all of them. It reports missing images and output blocks it couldn't match. |
| `$D/pages_html/<cat>_<Stem>.html` | Cached chebfun.org pages. This is the target for the print format; the pages were published with `format long`. |
| `$D/stdout/<cat>_<script>.txt` | Our latest stdout per script, which feeds the page generator. Regenerate it by running the script with stdout redirected there. |
| `$D/regen_all.py [substr ...]` | Runs example scripts 6 at a time with a 30-min cap, writing stdout and stderr to `$D/stdout/` and rc lines to the log (`REGEN_LOG` env). |
| `$D/run_slow.sh <script>` | Runs one script with a 4-h cap and appends to `slow_runs.log`. |
| `$D/unmatched.py <cat>/<Stem>` | Lists the page's output blocks that our stdout lacks, with the MATLAB input cell that produced each. |
| `$D/structure_audit.py` | Audits every page: blocks whose line counts differ, and extra lines we print. Writes `structure_audit.csv`. |
| `$D/refs_matlab/`, `$D/compare_vs_matlab.py` | MATLAB R2025b reruns of the examples (diary output). Use them only to compare numbers; a fresh rerun defaults to `format short`. |
| `$D/batch_prompt.md` | The prompt template used for the page-fidelity sub-agents. |

The workflow for one page:

1. Read `$D/pages_html/<cat>_<Stem>.html` and the `.m` source.
2. Edit the script so every output cell prints the same labels and layout and
   every `img/<Stem>_NN.png` is saved in order. Note that a MATLAB `hold on`
   cell produces separate before and after images.
3. Run the script, writing stdout to `$D/stdout/`.
4. Run `unmatched.py`, then `gen_example_pages.py <cat>/<Stem>`.
5. Run ruff, then commit the script, its images and its page.

Library display helpers that already match MATLAB: `repr(chebfun)`,
`Chebfun2.disp()`, `Diskfun.disp()`.

## 5. Remaining example work

Pages the audit marks as not matching (as of 2026-09-24). Many are blocked
only by random draws, which MATLAB's `randn` makes impossible to reproduce,
or by the library issues in section 6. Fix what is fixable in the script and
note the rest.

```
approx/EdgeDetection approx/NoisyNonsmooth approx3/Wagon cheb/ChebExplain
complex/ConformalL fourier/FourierBasedChebfuns geom/Ellipses
linalg/ConstrainedLeastSquares linalg/EigsViaDet linalg/FieldOfValues linalg/SOR
linalg/VandermondeArnoldi ode-eig/Drum ode-eig/Eigenstates ode-linear/DawsonIntegral
ode-linear/Krylov ode-nonlin/Blasius ode-nonlin/Carrier
ode-nonlin/DelayDifferentialEquations ode-nonlin/GulfStream ode-nonlin/IVPCapabilities
ode-random/RandomSwitching opt/Catenary quad/HermiteQuad roots/RandomPolys
sphere/AtmosphericTemperature stats/ResamplingRandomVariables
```

Some audit hits are false positives where the output is already correct
(Blasius, Catenary, Carrier apart from its iteration table).

Other known example gaps:

- **Checkmark** misses the local minimum at alpha = 0. Our E_3 piece on
  [0, 0.48] has 5 coefficients where MATLAB's has 12. Using
  `norm(p - f, inf)` instead of `r.err` would probably fix it, at about twice
  the runtime.
- **LaneEmden** now matches MATLAB (radius 3.653753736219 against
  3.653753736220) but takes about 28 minutes, because the nonlinear system
  Newton step uses a finite-difference Jacobian.

## 6. Library issues found by the examples, not yet fixed

These are ordered by value. Each item names the example that exposes it, and
should be fixed in `src/` with a MATLAB-port test.

1. **`Chebfun.inv` is about 800x slower than MATLAB.** It bisects pointwise:
   ComplexArcLength took 2018 s against MATLAB's 2.5 s. Port `@chebfun/inv.m`,
   which uses a roots-based inverse.
2. **`hermpts` uses only Golub-Welsch.** n = 10^4 takes 136 s, and 10^5 would
   need an 80 GB matrix. Port the GLR and asymptotic methods from `hermpts.m`
   (HermiteQuad).
3. **`ratinterp` returns wrong poles for complex f.** `utils/ratapprox.py`
   (`_chebyshev_roots`, about line 957) takes the real part of the
   coefficients. Its `r_fn` also casts x to float (ThreeBodyProblem).
4. **Building a periodic chebfun2 from complex values is wrong:**
   `chebfun2(complex matrix, trig=True)` gives O(1) errors (GinzburgLandau).
5. **MINRES stalls on indefinite operators.** On `-u'' - 100u` it reaches a
   relative residual of 5.8e-4, where MATLAB reaches 1.85e-15, and it
   diverges on the piecewise case (ode-linear/Krylov).
6. **Nonlinear `Chebop` solves.**
   - Newton starts at 16 points and doubles, restarting each time, while
     MATLAB starts at 64 and refines once to 128. As a result, Carrier's
     starting guesses 2 and 3 converge to different solutions.
   - There is no `'iter'` display, so Carrier's iteration table can't be
     printed.
   - `info["normDelta"]` concatenates the Newton histories of every
     refinement level.
7. **Splitting creates spurious breakpoints.** `_split_edge_fd` zooms to about
   1e-15 regardless of `eps`, which leaves clusters of breakpoints 1e-15 apart
   at eigenvalue-max kinks (EdgeDetection, NoisyNonsmooth).
8. **Backslash on an ill-conditioned basis is far less accurate than MATLAB.**
   The quasimatrix `\` gives a vertical scale of 7.7e7 where MATLAB gets 1.5
   (VandermondeArnoldi).
9. **The IVP marcher returns 16 pieces** where MATLAB returns one smooth piece
   (IVPCapabilities).
10. **The AAA conformal map keeps about twice MATLAB's spurious poles**
    (ConformalL).
11. **There are three `randnfun` implementations.** Only
    `chebfun1d/randfuns.py` follows `randnfun.m`. `cj.randnfun`
    (`utils/random.py`) returns a numpy callable and scales `'big'` wrongly.
    Unify them on the faithful one.
12. **`spin2` only returns arrays.** It doesn't accept a multi-time `tspan` or
    return a chebfun2, and `SpinOp2('gl')` uses a deterministic initial
    condition.
13. **Smaller issues:**
    - `Spherefun.max2` returns the antipode for even functions.
    - `plot_ball_slices` shading and depth order are poor.
    - `plotregion` estimates rho differently from `plotregionData`.
    - Chebfun has no `__rpow__` (`complex ** chebfun` fails).
    - `matlab_plot` gives each piece of a complex piecewise chebfun its own
      colour.
    - `roots(..., 'ms')` finds points MATLAB misses (ResultantMethod).
    - Disk Helmholtz at k = 7 has an error of 3.8e-8 from laplacian noise; the
      test is xfail.
    - There is no phase-portrait plot for complex chebfun2.

## 7. Fixed in September 2026 (for context; don't redo)

- **chebfun2 constructor.** It now uses MATLAB's sampleTest (Halton points),
  standardCheck happiness and global simplify. The Greeks vega matches MATLAB
  to 5e-12.
- **Splitting.** The length cap is now 129 points, the global hscale and
  vscale are threaded through, and detectEdge snaps to endpoints.
- **Small 1-D methods.**
  - `angle` uses atan2 with branch-cut breakpoints.
  - New `chebfun1d.fov`.
  - `restrict` snaps to piece endpoints.
  - `flipud` is exact.
  - `min(f, 'local')` finds kinks.
  - Fractional powers use the principal branch.
- **Chebop.**
  - The nonlinear Newton step uses rectangular projection with cached
    boundary rows, which fixes LaneEmden for n >= 2.
  - Systems accept scalar unknowns.
- **Disk and sphere solvers.**
  - Diskfun Helmholtz and Poisson are now MATLAB's ultraspherical port, about
    90x faster.
  - `Spherefun.rotate` uses MATLAB's ZXZ convention.
  - `randnfunsphere` is stable at high degree.
  - Spherefun `coeffs2vals` is vectorized.
- **Plotting.** New `plot_earth`, and `save_chebfun_figure` produces exact
  pixel sizes.
- **Examples.** The `_replica` suffix is gone, about 150 scripts were
  rewritten cell by cell, and every page figure is present.
