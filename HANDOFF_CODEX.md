# chebfunjax: handoff to a Codex agent (2026-10-02)

## Relational/sign source semantics (2026-10-10; latest)

All20 original lt/le clauses now pass (baseline16failed), with true <=/>=
identity behavior, crossing pointValues, empty/array error contracts and
singular/unbounded cases. Sign delegates through native FUN sign and merge;
Singfun.sign is implemented. addBreaks now restricts once to the complete
breakpoint vector; join previously shifted a root and lost its zero impulse.
Root40 tests pass, including9 supplemental controls and existing regression
modules. Explicit MT uniform adapter is checked against two captured blocks.
Evidence: docs/relational_sign_cpu_20261010.json. Full parity/CI remain open.

## Chebop quiver source renderer (2026-10-10; latest)

Source Euclidean grid scaling, held axes color cycling and open arrowhead
geometry replace Matplotlib's incompatible quiver defaults. Normalized zero
vectors retain native NaNs and are omitted from drawable geometry. Root28
focused tests pass, including independent scale and head-coordinate controls.
Only Chebop.quiver changes in production. Full43-solve page and image/content
parity remain open. Evidence: docs/chebop_quiver_renderer_cpu_20261010.json.

## Full native Jacobi sweep and restrict error semantics (2026-10-10; latest)

All625 native Jacobi parameter tuples and3 original513x2 cases pass at
original matrix-infinity-norm bounds, with explicit unmatched NumPy RNG
adapter. Root verified321 bindings and all28 clean runtime audits;9 reachable
Jacobi function ASTs match current code. Root7 integration tests pass, including
all3large cases and restrict error slots4-6. Prior oracle-grouping failure is
preserved; no production changes. Reduced composition tests remain supplemental.
Evidence: docs/native_jacobi_restrict_tests_cpu_20261010.json.

## Parametric source plotData and limits (2026-10-10; latest)

Bounded polynomial parametric curves now overlap breakpoints and sample both
components at their common source degree. Representation markers, NaN piece
separators and endpoint-inclusive axis limits follow source. Arrowplot uses
this route. Root25 tests pass; static checks pass. ChebopQuiver serial rerun
and unsupported-tech/interval rendering remain open.
Evidence: docs/parametric_plot_source_cpu_20261010.json.

## Arrowplot source sampling and held colors (2026-10-10; latest)

Arrowplot now uses native polynomial plotData grids/markers after combining
real components, follows the held axes color cycle, retains stationary-endpoint
annotations for nonzero functions, and rejects a single real argument. Root18
tests pass, including six source controls; static checks pass. Only arrowplot
changed. ChebopQuiver fullpage and general parametric sampling remain open.
Evidence: docs/arrowplot_source_cpu_20261010.json.

## Subtraction and negation preserve breakpoint values (2026-10-10; latest)

Native plus/uminus dispatch now retains pointValues, orientation and delta
metadata. Actual RandomSwitching arithmetic was wrong at17 switching points;
retained-coefficient replay now matches both direct residuals at all19 points.
Root27 focused tests pass; agent17+11 overlapping tests and runtime hashes
verified. Only3 arithmetic methods changed; static gates pass. Full page and
global regression remain open. Evidence: docs/pointvalues_arithmetic_cpu_20261010.json.

## Native ODE113 event location and terminal history (2026-10-10; latest)

Events now use the native Adams interpolant and source terminal phi/psi
rebasing; xe/ye/ie and truncated dense output are exposed. Shared ODE45
event bracket arithmetic is unchanged after callback extraction. Root43
CPU tests pass, including8 independent complex polynomial-history controls.
Adaptive step and interpolation kernels are unchanged; static checks pass.
No native MATLAB trajectory capture is claimed. RandomSwitching system
routing, coefficient breakpoints and fullpage remain outstanding.
Evidence: docs/native_ode113_events_cpu_20261010.json.

## Native differentiation dimensions and fractional metadata (2026-10-10; latest)

Chebfun/Quasimatrix diff now dispatches continuous and finite dimensions by
orientation, including row finite differences and fractional kind. fracInt
materializes native endpoint metadata. All11 original Chebfun diff expressions
and bounds are represented with explicit Python RNG/grid adapters. Root38
distinct cases pass across3 gates; initial missing Quasimatrix dependency and
JIT/fixture failures remain documented. Final static checks pass.
Evidence: docs/diff_source_cpu_20261010.json. Global tests/CI remain open.

## LevelHopping completes both source solves (2026-10-10; latest)

The complete page now finishes with53.919777s captured timing, both original
full-domain solves and two fresh600x269 figures. Root5 portable tests pass;
fullpage1 test passes with2968 runtime files independently rehashed.
Prior first-solve payloads and second-solve replay match exactly. Updated
whole figure-size audit:1305 mapped,33 mismatches across6 pages, none missing.
JAX/MATLAB RNG and historical font/grid/raster differences remain unresolved.
Evidence: docs/levelhopping_cpu_20261010.json.

## ODE113 history assembly bottleneck fixed (2026-10-10; latest)

Bounded JAX stack/concatenate groups preserve ordered history and all existing
controller/interpolation AST. Root18 tests and agent12 exact history/provider
controls pass. Actual LevelHopping second IVP now returns in18.38s,37.21s
including fit, with26330 accepted steps and207 retained checkpoints exact;
previous unchanged provider timed out at600s. Both sequential forcing draws
match prior evidence. Full page, MATLAB RNG/trajectory and CI remain open.
Evidence: docs/ode113_history_cpu_20261010.json.

## Ultraspherical coefficient utility preserves complex matrices (2026-10-10; latest)

The coefficient-array ultracoeffs helper now uses JAX nativegamma scaling,
rowwise matrix scaling, and complex-preserving second-kind coefficient shifts.
Nine distinguishing complex/matrix failures are fixed. Qualification covers
35 distinct passing cases across two gates; two initially incorrect empty
fixtures were corrected to explicitly complex inputs and all six affected
cases rerun at unchanged bounds. Other module AST and public Chebfun method
are unchanged; static checks pass. Evidence: docs/ultracoeffs_utility_cpu_20261010.json.
Full native public-test and large-N qualification remain open.


## Native ultraspherical conversion wrapper restored (2026-10-10; latest)

Ultra2ultra now uses native JAX gamma/cumulative-product scaling around the
public Jacobi converter, preserving complex values and matrix columns. Its
last NumPy Jacobi quadrature dependency is removed. Three of four original
native predicates failed before; all four now pass at unchanged100eps.
Root23 combined cases pass, including12 independent JIT/value controls; static
checks pass. Evidence: docs/ultra2ultra_jax_cpu_20261010.json. The separate
utility ultracoeffs hostscaling/complex-lam1 path remains an identified gap.


## Original fractional predicates and public Jacobi route pass (2026-10-10; latest)

Direct jac2cheb now uses native Chebtech1 FFT coefficients; with corrected beta,
all seven original fractional predicates pass unchanged bounds. Public jac2jac
now dispatches to the JAX source stages and preserves complex/matrix inputs.
Root70 combined cases pass; static checks pass; only those two transform
functions changed. Evidence: docs/fractional_fft_public_jacobi_cpu_20261010.json.
Full625+3 native Jacobi sweep and diff thirdargument dispatch are underway.
Source review found that existing Python fractional row assertions differ from
native row warning/rescaling/storedendpoint behavior; diff repair owns that gap.
No full fractional/public-API or current-head CI parity claim.


## Fractional beta scaling corrected (2026-10-10; latest)

Installed JAX0.11 beta uses the wrong geometric ratio in its algdiv remainder.
A local JAX helper restores the upstream ratio while retaining native beta/gamma
scaling. Root30 cases pass, including the original repeated-integral bound.
Independent80-digit checks cover selected indices up to10000 and mu near0/1;
the failed SciPy reference gate is preserved and diagnosed as oracle error.
Evidence: docs/fractional_beta_cpu_20261010.json. Repeated differentiation
still needs the separate sourceFFT correction; no tolerance changes.


## Native high-order Jacobi conversion helpers restored (2026-10-10; latest)

Chebyshev/Jacobi conversions now handle matrix columns and use the corrected
source integer/fractional chain at high order. Native pivot/diagonal/first-row
indices are restored. Cached JAX factors retain actual rank (21–26 at tested
N513 parameters); per-column FFT workspaces avoid rejected square buffers.
All63 root integration cases pass, including restored native matrix predicates;
static checks pass. Evidence: docs/jacobi_conversion_cpu_20261010.json.
Public jac2jac remains the old host quadrature route pending a separate patch.
Small-N FFT, beta correction, exact MATLAB RNG and large-N speed remain open.


## Fractional public column dispatch restored (2026-10-10; latest)

Chebfun fractional integrals/derivatives split array input into scalar columns;
Quasimatrix exposes both methods. Scalar integral output preserves orientation.
Native breakpoint error IDs/order and non-Caputo RL fallback are restored.
Nine distinguishing controls failed before the fix; final34 cases pass
(11public controls,19kernel controls,4existing compatibility tests). Static
checks pass; other method ASTs are unchanged. Evidence:
docs/fractional_public_api_cpu_20261010.json. Original sequential fractional
failures, full diff dimension/kind dispatch and remaining edge semantics are
still open; no full fractional parity claim.


## Public Singfun predicates restored (2026-10-10; latest)

Added isfinite/isinf/isreal/any with native current-preference strict exponent
boundary, NaN complement and smooth-factor delegation. Original finite/isnan
tests now call public methods instead of test-only proxies. All21 integration
cases pass; existing module AST is unchanged outside the four added methods.
Agent runtime audit independently checked2946 files; root focused CPU gate and
static checks pass. Evidence: docs/singfun_predicates_cpu_20261010.json.
Numeric-empty behavior remains a labeled Python adapter. Restoring all seven
fractional predicates exposed two original-bound failures (sequential integral
and derivative); five pass, diagnosis active, no tolerance relaxation.


## JAX fractional integral coefficient path restored (2026-10-10; latest)

Tech/Singfun fracInt now expose native smooth-factor and exponent operations.
Chebfun delegates through them, removing real NumPy coefficient casts and
SciPy special/sparse kernels. Complex inputs are preserved; the half-integral
triangular solve uses linear-storage JAX back substitution. All23 focused
cases pass; static checks pass. Evidence: docs/fractional_jax_cpu_20261010.json.
The old fractional-calculus port omits native predicates and changes bounds;
all7original predicates, quasimatrix support and remaining derivative host
operations remain open. No full native fractional-calculus parity claim.


## Native low-rank Chebyshev-to-Legendre branch restored (2026-10-10; latest)

cheb2leg now uses the native pivoted-Cholesky/FFT algorithm at513 rows,
with static-N JAX plans and runtime arrays sized by retained rank. N513
retains26 columns (106496 factor bytes); the rejected square-buffer candidate
is preserved unmerged. Matrix conversion, native row early returns and
normalization order are restored. Root corrected the N1000 normalization
bound to native tol and literal row fixtures, then qualified30 combined
transform/coefficient API cases. Static checks pass; _dct1 is unchanged.
Evidence: docs/cheb2leg_lowrank_cpu_20261010.json. Random fixtures are not
MATLAB-identical; large-N scaling and matchedMATLAB timing remain open.


## Public endpoint-decay methods implemented (2026-10-10; latest)

Chebtech1, Chebtech2 and Singfun expose native isdecay; unbounded integral
warnings dispatch through these methods. The shared JAX helper now uses
MATLAB real-part ordering for complex constants, replacing the earlier
exact-zero-only adapter. Eight new controls and all24 unbounded-sum cases
pass, with two previously recorded warnings. Static checks pass. Evidence:
docs/isdecay_public_cpu_20261010.json. Nonconstant JIT, Trigtech decay and
no-input Singfun numeric-empty dispatch remain unqualified.


## Singfun reflection and derivative empty metadata fixed (2026-10-10; latest)

Empty flipud reverses the stored exponent tuple; fliplr is the native identity;
diff preserves input for empty and order-zero cases. This completes the
remaining explicit-empty unary metadata corrections found after the new
no-input representation. Twenty focused/native reflection and derivative
cases pass; static checks pass. Evidence: docs/singfun_empty_flips_cpu_20261010.json.
Existing deterministic query grids are not native RNG fixtures.


## Exact Singfun smoothness and unary demotion restored (2026-10-10; latest)

issmooth now uses the native exact-zero exponent or zero smooth-factor
predicate. real/imag/conj transform first, then demote a smooth result; empty
inputs preserve their original exponent representation. All eight new
regressions fail on the preserved baseline; the corrected package passes
46 focused/native unary and arithmetic cases with unchanged numerical bounds.
Static checks pass. Evidence: docs/singfun_smoothness_cpu_20261010.json.
This resolves the previous nonempty issmooth item; constructor14 and broader
Singfun/library/page qualification remain open.


## Singfun no-input empty representation restored (2026-10-10; latest)

No-input Singfun stores empty exponents; explicit empty input retains the
zero exponent pair. Native roots, smoothness, sum/cumsum and negation empty
routes are restored with Python storage/adapters explicitly documented.
32 cases pass on the isolated snapshot. Root verified 41 bindings, 2956 runtime
files and byte-identical integrated payloads; ten nonempty method tails and
the module outside twelve scoped methods are unchanged. Static checks pass.
Evidence: docs/singfun_empty_cpu_20261010.json. Constructor assertion14 still
fails on the adapter query grid; nonempty issmooth source semantics remain open.


## Native Chebtech logical API implemented (2026-10-10; latest)

Both Tech classes now expose logical via source coefficient-to-value
conversion and explicit column reduction, retaining happiness and boolean
coefficients. Eight CPU controls pass, including NaN/value-versus-coefficient
semantics, complex columns, empty shapes and JIT. Only two new method ASTs;
full static checks pass. Evidence: docs/chebtech_logical_cpu_20261010.json.
Native root-free precondition remains; full qualification is incomplete.


## Unbounded integration slow-decay warning restored (2026-10-10; latest)

Unbndfun.sum now emits the native slowDecay ID/message when endpoint decay
is insufficient, including one warning for array-valued input. Existing
divergence thresholds and integration paths are unchanged. Root corrected the
candidate residual mask to match MATLAB across all columns when any initial
root triggers extraction, with a distinguishing regression. Full module:
24 passed, two existing warnings; static checks green and source/test hashes
stable. Evidence: docs/unbndfun_slow_decay_cpu_20261010.json. Public isdecay
wrappers and complex unbounded integration remain unqualified.


## Chebtech coefficient APIs and complex Legendre conversion (2026-10-10; latest)

Both Tech classes now expose chebcoeffs, legcoeffs and jaccoeffs with native
conversion/padding order and matrix-column behavior. Public native legcoeffs
assertions replace utility proxies and use the native matrix infinity norm.
A complex control exposed unconditional real FFT projection; the helper now
preserves complex data. Root gate: 16 passed, stable source/test hashes, static
checks green. The initial two complex failures remain preserved. Evidence:
docs/chebtech_coefficient_apis_cpu_20261010.json. Large transform algorithms,
full current-head regression and MATLAB/remote CI qualification remain open.


## Native Chebtech any implemented (2026-10-10; latest)

Both Tech classes expose any with native coefficient reduction and arbitrary
point dim2 behavior. Tests now call the API instead of duplicating it. Root
review additionally restored NaN ignoring, first-nonsingleton row behavior,
zero-row columns and dim2 happiness metadata. All 14 cases pass; only two new
methods change library AST. Evidence: docs/chebtech_any_cpu_20261010.json.
Native dim2 root-free precondition remains; global qualification is incomplete.

## Singfun factory preferences and scalar columns restored (2026-10-10; latest)

Native constructor(op,data,pref) and make now forward endpoint hints, copied
preferences, scales, selected tech and constructor options. Numeric input means
smooth-factor samples; existing smooth factors retain identity. Scalar-column
construction/evaluation avoids unintended outer broadcasting, with complex and
JAX gradient controls. Root verified 88 bindings and six runtime inventories
(2929–2948 files), including the preserved baseline shape failure.
75 distinct logical cases pass across qualified snapshots; 36 final-v2 cases
cover the shape fix and actual construction. Evidence: docs/singfun_factory_cpu_20261010.json.
Remaining gaps include constructor 1–18 random-input/assertion fidelity, native
empty exponent representation and unqualified Trig arithmetic. Full current-head
suite and MATLAB numerical parity remain open.

## Large QR reconstruction error corrected (2026-10-10; latest)

QR now selects the existing source transpose-NDCT inverse transform throughout
its n>4000 branch. The general inverse transform retains its 5000 crossover.
The original n=4001 cos/exp regression now passes the unchanged 5e4eps-scaled
bound for both Tech classes and pivoted/unpivoted outputs. Root verified 14
cases including existing large Legendre blocks and permutation controls.
Captured matrix-infinity residuals improve from about 1.263e-9 to
5.13e-12–1.17e-11. Evidence: docs/chebtech_qr_reconstruction_cpu_20261010.json.
The direct failure remains preserved. This source-equivalent algorithm choice
differs from MATLAB’s direct floating-point path below 5000; no exhaustive
length-range, paired MATLAB speed or full-suite parity claim.

## Pitchfork source computations and figures integrated (2026-10-10; latest)

All five native [0,600] solves completed using the existing JAX ode113 state
tower, now enabled for higher-order real scalar IVPs. The page restores
sequential nonperiodic forcing and the source damped -f1/f2 signs, actual
600x269 figures and timing-only stdout. Root verified117delivery bindings,
five runtime inventories (2953/2970/2953/2980/2971files),11exact probe members
and11signed array comparisons. Existing first-order controls passed.
Evidence: docs/pitchfork_full_page_cpu_20261010.json.

The full-page observer failed after computations on optional None metadata;
its failed status is preserved. Artifact-only recovery qualified retained
meshes, arrays and figures without repeating solves. Missing full-page artist
metadata, endpoint/interior residual and fitted initial-derivative accuracy,
MATLAB RNG and historical rendering remain open. These figures do not establish
full visual parity. A fresh whole-gallery audit maps all 1,305 figure pairs with
no missing files or changed reference bytes: 35 size mismatches across 7 pages
remain. Evidence: docs/figure_size_audit_20261010.json.

## Missing Chebtech predicate APIs implemented (2026-10-10; latest)

Both Tech classes now expose JAX isfinite, isinf and isreal. Native tests call
these APIs and existing isequal, replacing copied predicates. Inf fixtures now
use coefficients, matching native make({[],y}); duplicate native scalar inputs
and separate array controls are retained. All 44 cases pass, including JIT,
NaN, complex-zero dtype and empty controls. Only six new methods change library
AST. Evidence: docs/chebtech_public_predicates_cpu_20261010.json. Global suite
qualification, remaining native test fidelity and publication remain open.

## Public Chebtech iszero implemented (2026-10-10; latest)

Both Tech classes now expose the source exact per-column zero predicate in JAX.
Native tests previously used a coefficient proxy and did not exercise this API.
All ten native cases, two JIT/complex/nonfinite/empty controls and two simplify
zero regressions pass. Only the two new methods change the library AST.
Evidence: docs/chebtech_iszero_public_cpu_20261010.json. Remaining simplify
source/input gaps and full-suite qualification stay open.

## Built-in three-output QR pivoting restored (2026-10-10; latest)

Both weighted QR branches now use JAX column pivoting when a permutation is
requested; two-output and Householder behavior remain source-correct. Matrix
and vector encodings use the native A[:,p]=Q*R convention. Twelve integration
cases pass, including a three-cycle control that distinguishes inverse
permutations. Newer restriction fixes and QR scale assertion are preserved.
Evidence: docs/chebtech_qr_pivot_cpu_20261010.json. The pre-existing high-length
cos/exp reconstruction error (~1.05e-9 versus3.02e-11 bound) remains open and
is under source diagnosis; high-branch pivot checks do not qualify accuracy.

## Native Chebtech restriction validation restored (2026-10-10; latest)

Both Tech classes now return empty inputs before validation, reject endpoints
strictly outside [-1,1], validate the complete breakpoint ordering, preserve
exact full-interval identity, and emit the native badInterval identifier/message.
Prolong/restrict tests now use native vector/matrix infinity norms and public
empty/equality predicates. All 40 focused cases and six downstream Singfun
restriction cases pass; tolerances are unchanged. Only the two restrict methods
changed in library AST. Evidence: docs/chebtech_restrict_native_cpu_20261010.json.
Full current-suite, page qualification and publication remain open.

## Native public isnan tests restored (2026-10-10; latest)

All ten native Chebtech isnan cases now exercise the public method, with exact
case-insensitive exception-message checks after removing Python’s native-ID
prefix. All ten pass; library code is unchanged. Evidence:
docs/isnan_public_native_cpu_20261010.json. Source/test hashes and JUnit verified;
no full observed-module runtime audit or whole-suite claim.

## QR per-column scale assertion restored (2026-10-10; latest)

Native QR pass 20 now checks the three-entry vscale_columns vector. All four
Chebtech/method combinations pass; root verified the single added assertion,
unchanged library, source/test hashes and JUnit. Existing aggregate checks are
retained as supplemental controls. Evidence: docs/qr_vscale_columns_cpu_20261010.json.
The gate enforced CPU affinity and timeout but did not measure or cap RSS.
Builtin three-output pivoting remains an independent library gap under repair.

## Singfun typed and partial exponent construction restored (2026-10-10; latest)

Callable construction now accepts endpoint sing_type hints: pole selects
integer detection, sing/root select fractional detection, and none gives zero.
Empty exponents trigger detection; NaN entries are filled while finite supplied
entries remain unchanged. Case-insensitive native dispatch and numeric helper
behavior are preserved. Existing detection kernels and other methods are
unchanged. All 21 cases pass, including six original restriction cases; root
verified 2,945 runtime files and the source AST. Evidence:
docs/singfun_constructor_hints_cpu_20261010.json. Factory/preference forwarding,
other constructor forms, full current-suite qualification and CI remain open.

## Chebtech division public NaN assertions restored (2026-10-10; latest)

Native division passes 2, 4 and 6 now check public isnan, instead of accepting
any Inf/NaN coefficient contamination. The complete 30-case module passes;
source/test hashes and exact three-assertion AST additions were root-reviewed.
Library behavior was already correct. Evidence: docs/rdivide_public_isnan_cpu_20261010.json.
Native seeded query points remain unmatched. This test-only gate enforced CPU
affinity and timeout but did not sample/enforce RSS; no memory claim is made.

## Native test qualification and restored Singfun assertions (2026-10-10; latest)

All 55 existing native Chebtech port modules passed: 825 unique cases, no
skips/failures/errors, on frozen f1cb17fd. Root verified exact JUnit coverage,
delivery hashes and all four runtime audits. This is execution qualification,
not proof that every port retains all native predicates; source fidelity
review continues. Evidence: docs/chebtech_native825_cpu_20261010.json.

Singfun restriction's six old tests only called restrict and discarded the
results. The original 12 subinterval accuracy checks are now restored with
100-point grids, endpoint exclusions and unchanged 5e4eps-scaled bounds.
All six cases pass on the unchanged library; root audited 2,943 runtime files.
Evidence: docs/singfun_restrict_native_cpu_20261010.json.

A file-level inventory maps all 1,102 native test files: 1,098 matching port
paths and four combined GUI exporter cohorts. This establishes representation
only. Missing assertions and changed operators/order/masks can remain inside
those files, as the Singfun and Trigtech roots reviews demonstrate.

## Trigtech Fourier transforms now use JAX throughout (2026-10-10; latest)

Concrete and traced public transforms share the existing JAX kernels; the
two unused NumPy transform mirrors are removed. Kernel and other method ASTs
are unchanged. All 123 focused/native/construction cases passed, plus one
cumulative resource control covering 832 round trips across 416 lengths.
Root audited all four runtime gates and source equivalence. Cumulative peak
RSS was 2.87 GiB; cold compilation retained 416 entries per kernel and the
warm sweep added no entries. Native 100eps absolute bounds were unchanged.
Evidence: docs/trig_transforms_jax_cpu_20261010.json. Legacy host Horner and
construction paths remain; no full expensive-solver or matched speed claim.

Latest full dimension audit maps 1,305 figure pairs with no missing files or
changed reference bytes: 37 size mismatches across 8 pages remain after
Consensus. Source: figure_size_current_d85340c7_20261010.json in shared evidence.
Matching dimensions do not establish figure content or historical rendering.

## Consensus source forcing and full page restored (2026-10-10; latest)

All three native [0,40] solves completed in 596.6s with 4.26 GiB peak RSS.
Nonperiodic random forcing uses the native 48-unit embedding/481 coefficients,
then restricts to the original physical interval. Source operators, initial
values and strengths 0/3/1 are retained. Seven portable controls pass; root
audited all four gates (2,951-2,975 runtime files each) and 69 delivery bindings.
Three actual figures are 600x269; stdout is empty as in the source. Evidence:
docs/consensus_full_page_cpu_20261010.json. Sequential JAX random draws differ
from MATLAB rng(3); literal source 32pt labels and 2.5 linewidth differ visibly
from historical website rendering. Full visual/controller parity stays open.

## AAAtrig input infinity constraints restored (2026-10-10; latest)

Source constraint extraction, appended Loewner rows, limit residuals and
cleanup refitting now preserve supplied values at imaginary infinity.
All 74 CPU cases passed (20 new, 54 unchanged prior controls); root verified
69 delivery bindings and 2,935/2,946 runtime files. Full static checks pass.
Evidence: docs/aaatrig_sample_constraints_cpu_20261010.json. Integration
adds four JAX operand conversions solely in an unexecuted source edge branch;
entire-module AST equivalence outside those conversions was checked.
Negative-only even, duplicate constraints and no-finite-sample branches
remain unqualified. Native options, host eigenpaths and full current CI
remain open. Earlier sections describe historical package boundaries.

## Core sweep ledger and deferred examples verified (2026-10-10; latest)

Historical frozen core partitions provide 8,800 distinct passing test IDs,
11 remaining skipped IDs and one retired stale-contract baseline ID. Root
verified the artifact bindings and the separately resolved AAAtrig skip.
All 13 deferred example tests also passed in a clean frozen d674129e run;
root rehashed 3,018 runtime files and 12 fresh declared PNG outputs. The
preceding guard-failed run remains preserved. Evidence:
docs/core_sweep_completion_ledger_20261010_v2.json and
docs/core_sweep_and_examples_review_20261010.json. These are separate
snapshots, not current-head full CI or page/figure parity. Native-port
coverage, actual MATLAB reference fixtures and broader semantic gaps remain.

## Native AAAtrig greedy nullspace weights restored (2026-10-10; latest)

JAX row/column scaling preserves the native two-product Loewner expression
and removes the dense sample-count-squared diagonal allocation. Wide and
zero-row matrices retain the full right singular basis, selecting native
column m. The original three-sample failure used three supports instead of
two; corrected odd/even public and analytic controls now pass. All 54 CPU
cases passed, including 42 unchanged prior controls; root audited both
successful gates and the preserved baseline failure, plus 64 bindings.
Evidence: docs/aaatrig_greedy_cpu_20261010.json. Lawson is unchanged. Infinity
input constraints, AutoZ/string inputs, host construction/eigenproblems,
full native tests and current-head CI remain open. No measured performance
improvement is claimed from the allocation change.

## Native Chebfun3 tensor grids, Tucker and rank tests restored (2026-10-10; latest)

Eager tensor-grid evaluation follows all four native recognition patterns,
permutations and contractions; generic and traced evaluation are unchanged.
All original Tucker3 predicates passed on full 100-cubed grids, including
the original nonunit box; all rank6 predicates passed for rational and Airy
functions at original bounds. Eight CPU gates passed 23 cases, including
13 grid/fallback/AD controls and five retained legacy tests. Root verified
128 delivery bindings and each gate runtime (2,936-2,960 files). Evidence:
docs/chebfun3_tensor_rank_cpu_20261010.json. Empty HOSVD output arity, matrix
grid optimization, matched performance and complete current-head CI remain
open; no measured baseline memory improvement is claimed.

## Conformal plotting/options and full page restored (2026-10-10; latest)

Native plots/numbers options and continuous-boundary page computations now
use source geometry, 10,000 mapped disk points and 1,001 boundary probes.
The complete page and five portable option controls passed on CPU; root
verified 2,971 and 2,978 runtime files and all 80 delivery bindings. Both
figures are 600 by 253 pixels. Four actual output cells were refreshed;
original prose and MATLAB cells are unchanged. Evidence:
docs/conformal_public_page_cpu_20261010.json. MATLAB rng(0) parity remains
open (51/40 poles versus historical 59/46), as do historical fonts/ticks/
rasterization, matched performance and current-head CI. Preserved failed
preparation/observer/guard attempts are documented in the evidence.

## Native AAAtrig degree and Lawson options restored (2026-10-10; latest)

Explicit degree or mmax now enables native adaptive Lawson iteration, unless
lawson=0. JAX IRLS preserves source support rows, ordered sums, iteration
limits and rollback. Native two-sample antisymmetric weights remove the old
spurious midpoint pole. All 42 CPU cases passed, including original predicates
9 and 18-23 in both forms, finite-step/option controls, a wide-matrix nullspace
control and 28 prior cases. Root audited both successful gates and the missing-
option baseline failure; independent source review confirmed the SVD shape
semantics. Evidence: docs/aaatrig_lawson_cpu_20261010.json. Infinity-sample
constraints, AutoZ/string inputs, native RNG fixture, inherited host greedy/
eigenproblem paths, full native tests and current-head CI remain open.

## Native AAAtrig evaluation at imaginary infinity restored (2026-10-10; latest)

Both trigonometric bases now return the native analytic limits at positive
and negative imaginary infinity using JAX. The baseline returned NaNs for
independently derived finite limits. All 24 cases passed, including closed-form
limits, NaN/real-infinite/support evaluation, original public special-value
predicates and the previous 20 controls. Root verified 2,940 runtime files;
all constructor and pole/zero code is unchanged. Evidence:
docs/aaatrig_evaluation_cpu_20261010.json. Missing degree/Lawson/autoZ and
infinity-sample constraints, native RNG fixtures, inherited host construction,
complete native tests and current-head CI remain unresolved.

## Native Chebfun3 get and empty properties restored (2026-10-10; latest)

All five valid properties of an empty Chebfun3 return native numeric empty
arrays through get and dot subsref. Invalid names still raise the native
error. All five original get predicates now use source continuous norms and
factory tolerances; the previous sampled test is retained. All 20 cases
passed, with 2,966 runtime files and 31 delivery bindings checked by root.
Nonempty executable AST is unchanged. Evidence: docs/chebfun3_get_cpu_20261010.json.
Native object-array/cell recursion has no Python representation yet; that
branch, full current suite and CI remain open.

## Trigtech inner-product test reference corrected (2026-10-10; latest)

The real-column control now uses native prolonged stored values and trapezium
weights. Its old coefficient shortcut ignored real-column projection. Exact
array equality is retained; library code is unchanged. All 45 cases in the
original module passed, with 2,948 runtime files independently verified.
Evidence: docs/trig_innerproduct_reference_cpu_20261010.json. The earlier
batch failure remains preserved; broad-suite and CI qualification remain open.

## AAAtrig infinity mapping and cancellation restored (2026-10-10; latest)

Odd/even inverse transforms now apply native infinity thresholds and cancel
matched poles/zeros in source order. JAX componentwise projection avoids NaN
real parts for imaginary infinities. All 20 CPU cases passed; root rehashed
2,939 runtime files and checked unchanged production bytes against the prior
diagnostic. The baseline cancellation failure is retained. A new test's
unsupported odd-form negative-infinity expectation was corrected using actual
generalized eigenvalues and the native isinf filter; both signed branches
remain covered by source controls. Evidence: docs/aaatrig_infinity_cpu_20261010.json.
Inherited host eigensolver/greedy loop, missing options, complete native tests,
full current suite and remote CI remain unresolved.

## Continuous conformal mapping restored (2026-10-10; latest)

Conformal mapping accepts native continuous Chebfun boundaries, uses native
arclength and inverse composition, and follows the source Kerzman-Stein and
polynomial construction with JAX arithmetic. The complete original conformal
and conformal2 tests passed together, followed by five analytic/array controls.
Root verified all three runtime records; independent review checked 90 delivery
bindings, unchanged canonical tests and executable AST identity after a doc-only
correction. Evidence: docs/conformal_continuous_cpu_20261010.json. A prior 4 GiB
resource-cap failure is retained; the complete gate passed at a justified higher
limit. Plot/numbers options, page geometry/RNG, exact boundary classification,
inherited host-array dependencies, full current suite and CI remain open.

## Native Chebfun3 indexing and slices restored (2026-10-10; latest)

Parenthesis, property and brace dispatch now follow the native source.
Colon slices contract existing factors; a scalar-column adapter corrects
plane broadcasting and is shared with restriction. All 67 cases passed in
six serial CPU gates, including all 18 original subsref predicates and all
10 restriction predicates through an independent brace route. The prior
analytic test is retained. Root rehashed all six runtime records and 110
delivery bindings; the numeric evaluator AST is unchanged. Evidence:
docs/chebfun3_subsref_cpu_20261010.json. Native test sections were qualified
in separate processes. Vector-fixed colon slices, empty properties,
arbitrary recursive indexing, full current suite and remote CI remain open.
Earlier scalar-plane and sphere-domain failures are preserved.

## Trigtech finite/infinite predicates restored (2026-10-10; latest)

isinf and isfinite now inspect the stored source-grid values as native
Trigtech does. An isolated Inf sample can produce NaNs in Fourier
coefficients, so testing coefficients missed the actual infinity.
All14 cases in the original predicate module passed; its failed assertion
is unchanged and an added check proves constructed values retain Inf.
Root independently checked2,937 runtime files and confirmed only these two
library methods changed. Evidence: docs/trigtech_predicates_cpu_20261010.json.
The prior237-pass/one-failure batch and two harness-preflight failures are
preserved. Remaining broad batches, full current suite and CI remain open.

## AAAtrig cleanup order and JAX arithmetic restored (2026-10-10; latest)

Cleanup now uses native truncation toward zero and removes each selected
support before considering the next spurious pole. It removes the artificial
distance regularizer and uses JAX for cleanup arithmetic and reduced SVD.
The previous wrong-support failure is preserved. All16 CPU cases passed:
four source branch controls, two public analytic small-pole cleanup runs,
and ten existing residue/public controls. A prior test input that did not
trigger even-form cleanup is preserved; its replacement has known poles
and retains the same numerical bounds and required cleanup warning.
Root verified runtime evidence and unchanged AST outside cleanup. Evidence:
docs/aaatrig_cleanup_cpu_20261010.json. Infinity mapping/cancellation,
inherited host eigensolver/greedy loop, full native tests and CI remain open.

## AAAtrig complex-pole residues corrected (2026-10-10; latest)

Residues now evaluate the quotient numerator and derivative at the full
complex pole using JAX, matching native prztrig.m. The prior implementation
discarded imaginary pole coordinates and could return a near-zero residue
instead of the analytic -4i/3. That failing baseline is preserved.
All ten candidate cases passed: independent closed-form and public both-form
controls, original native residue predicates16/17, and five existing controls.
Root checked source AST boundaries and both runtime records. Evidence:
docs/aaatrig_complex_residues_cpu_20261010.json.
Infinity mapping/cancellation, cleanup order, inherited NumPy/SciPy paths,
missing AAAtrig options, full native tests and remote CI remain unresolved.

## Obsolete AAAtrig even-form skip removed (2026-10-10; latest)

All five existing AAAtrig controls pass, including the formerly skipped cot
basis case, with original assertions and library bytes unchanged. Root
independently rehashed 2,933 runtime files and checked the only test AST
change was removal of that skip. Evidence: docs/aaatrig_even_skip_cpu_20261010.json.
This closes the known skip in the 304-pass core suffix. It does not establish
full native AAAtrig parity: complex-pole residues, infinity handling, cleanup,
missing options, native test fidelity and JAX-only construction remain open.
Four existing pole-projection warnings were retained in the passing run.

## Factor-preserving Chebfun3 restriction restored (2026-10-10; latest)

Restriction now contracts the existing core and restricts its factors using
native branch order, preserving complex point values, scalar line shapes,
and cuboid domain errors. It no longer reconstructs slices by resampling.
All 21 CPU cases passed: 19 controls, one case containing nine native
restriction predicates plus one adapted predicate, and the unchanged analytic
regression. Root independently checked all three runtime records and 46
source delivery bindings. Evidence: docs/chebfun3_restrict_cpu_20261010.json.
The explicit subsref API and its independent test route remain open, as do
native rank stress tests, full current suite and remote CI. Prior failing
shape controls and the Python scalar-shape regression are preserved.

## Public ODE45 output modes restored (2026-10-10; latest)

The explicit outputs keyword now provides native solution-record, T/Y and
three-to-five-output branches while preserving the existing T/Y default.
Raw accepted mesh outputs and missing-event-field errors follow executable
source; empty explicit multiple-output calls report the native arity fault.
All 42 CPU cases passed. Root independently checked runtime hashes, 21
delivery bindings, unchanged controller bytes and the entire existing
nonempty numerical AST. Evidence: docs/ode45_outputs_cpu_20261010.json.
The private record still lacks native dense interpolation and continuation
metadata; complete SOL/deval equivalence, full suite and remote CI remain open.

## Laguerre dispatch regression expectation corrected (2026-10-10; latest)

A broad batch exposed an outdated test expecting GW for alpha=1.5 at n=3000.
Pinned lagpts.m selects RH for every static alpha at n>=3000; the library
already follows it. The expectation now matches the source. All 13 focused
cases passed, including numerical default-RH checks and the separate traced
alpha adapter. Root independently checked 2,954 runtime files.
Evidence: docs/laguerre_dispatch_cpu_20261010.json. The preceding 99-pass,
one-failure batch remains preserved; its unexecuted suffix is running.
Full current suite and remote CI remain unresolved.

## Native ODE45 terminal events implemented (2026-10-10; latest)

The JAX controller now locates directional events using the source quartic
interpolant and safeguarded Illinois brackets, records event metadata, and
truncates the public solution domain. All 30 CPU cases passed across six
serial gates, including the original full-interval projectile and unchanged
controller regressions. Root independently verified runtime hashes and all
49 delivery bindings. The projectile triggers no events under the native
domain mask; separate controls exercise actual crossings and public outputs.
Evidence: docs/ode45_events_cpu_20261010.json. No-event numerical kernels and
the complete DynamicalSystems page source remain unchanged.
Public SOL/five-output adapters, broader event corners, remaining solver
options, native executable trajectories and complete suite/CI remain open.

## Native complex construction and realness restored (2026-10-10; latest)

Chebfun3 complex now checks real inputs before arithmetic, then uses the
native sum of real and imaginary components. isreal follows the stored core
and factors, including complex-zero storage and the Trigtech real flag;
real preserves empty input before composition. Only these three library
methods changed. All 53 CPU cases passed in four serial processes, including
the four original complex/isreal predicates and all three preserved analytic
regressions. Root independently checked runtime hashes and 58 delivery
bindings. Evidence: docs/complex_real_cpu_20261010.json.
Global representation normalization, full consumer coverage, the complete
current suite and remote CI remain unresolved.

## Subnormal breakpoint alignment fixed (2026-10-10; latest)

Static domain comparisons now preserve subnormal differences using Python
doubles, matching native tweakDomain before overlap restriction. Previously
XLA flushed the difference to zero, causing a zero-width restriction in the
original constructor-splitting test. Coefficient and remap arithmetic remain
JAX. Half-away integer rounding also retains values just below one half and
large exactly representable odd integers.
All 18 focused CPU cases passed, including unchanged native constructor,
tweakDomain and overlap assertions; root independently checked 2,955 runtime
files and confirmed only tweak_domain changed in the library module AST.
Evidence: docs/breakpoint_subnormal_cpu_20261010.json. Earlier failed control
observers are preserved. Full current suite and remote CI remain unresolved.

## Native cumulative integral and extrema tests restored (2026-10-10; latest)

All 16 original cumsum3/min3/max2/min2 predicates now use the native
functions, construction order, continuous norms and tolerances. Four CPU
cases passed in three serial processes; root independently checked runtime
hashes and all source delivery bindings. This replaces sampled-grid
surrogates and restores the original shifted quadratic for min3.
Evidence: docs/native_cumsum_extrema_cpu_20261010.json.
Full extrema algorithm parity, the current complete suite and remote CI
remain unresolved.

## Public JAX ODE45 and complete DynamicalSystems run (2026-10-10; latest)

Chebfun2v.ode45 now uses a JAX finite Dormand–Prince accepted-mesh controller
following the separately versioned R2017a built-in source and pinned7574c77
wrapper: native defaults, domain indicator, accepted mesh and complex return.
Source/wrapper controls and the preserved analytic regression passed.
All55 public example integrations completed and11 fresh500x400 figures
are integrated. Root checked2,963 runtime files, all110 mesh arrays exactly
against the accepted computation, and all10 unchanged Markdown stdout cells.
Public plotData grids and source-backed72dpi export fix the clipped final
title while retaining literal font/marker sizes and canvas dimensions.
Evidence: docs/dynamical_ode45_cpu_20261010.json.

Terminal events, mass/output options, broader controller qualification,
native executable trajectories and historical website styling remain open.
The shortened projectile control is explicitly separate from nativepass1.
The earlier100dpi layout failure is preserved, along with its completed
55 numerical solutions. Full current tests and remote CI remain unresolved.

## Native contractions and directional assertions restored (2026-10-10; latest)

Chebfun3 now exposes source mtimes through `.mtimes()` and Python `@`:
scalar/zero branches, continuous mode-one contraction with Chebfun and
Chebfun2, and the native forbidden Chebfun3 contraction error. New arithmetic
uses JAX inner products, tensor products and SVD. Python `*` retains its
elementwise convention. All four original mtimes predicates, two unchanged
additional Python tests and four scalar/shape controls passed in ten cases;
root verified 2,976 runtime files. Evidence: docs/mtimes3_cpu_20261010.json.
General complex/periodic/rectangular product qualification remains open.

All 44 original diffx/diffy/diffz continuous norm predicates are restored
with the actual functions, domains, constructor order and native bounds.
Twenty-two cases passed across nine serial gates, each independently checked
against 2,940 runtime files. Evidence: docs/native_directional_diff_cpu_20261010.json.
The whole current suite, remaining source gaps and remote CI remain open.

## Native complex routes and assertions restored (2026-10-10; latest)

Permute now rearranges existing factor functions and transposes the JAX core,
retaining periodic factors and avoiding constructor resampling. Conjugate
uses native compose with its empty guard; imag also preserves empty input.
All fifteen original conj/imag/permute predicates now run with the actual
complex functions, continuous norms and original bounds. These and four
periodic/factor/empty controls passed in ten CPU cases. Root independently
verified the runtime and source bindings. Evidence:
docs/chebfun3_complex_routes_cpu_20261010.json. Invalid permutation adapters,
global constructor host paths, full current suite and remote CI remain open.

## Native subtraction and Carrier gate restored (2026-10-10; latest)

All three original subtraction predicates passed, including the varying
pi domain and nonzero near-cancellation norm, with native tolerances. Root
independently rehashed 2,940 runtime files in each of three serial gates.
Carrier now passes its existing MATLAB fixture comparison without the stale
strict-xfail marker; its body, inputs and bounds are unchanged. Root verified
3,037 runtime files. The preceding broad run remains recorded as 251 passes
and one strict XPASS; its unexecuted suffix still needs completion.
Evidence: docs/native_minus_cpu_20261010.json and
docs/carrier_xpass_cpu_20261010.json. Full current tests and remote CI remain open.

## Six native unary contracts restored (2026-10-10; latest)

Exp/cos/tanh now use the native default constructor. Abs returns positive
real input, negates negative input and rejects sign changes. Sqrt/log
apply the source sign test and principal complex promotion for negative
samples, preserving generic-compose periodic behavior. All five original
abs predicates, including the previously omitted error case, and fifteen
periodic/sign/complex/empty controls passed:18CPUcases,2,950runtime
hashes independently verified. Evidence: docs/chebfun3_unary_source_cpu_20261010.json.
Generic compose, existing isreal, constructor/evaluation host paths and
fully traceable adaptive dtype selection remain separate audit scope.

## Native guide predicates1–10 restored (2026-10-10; latest)

The first ten guide assertions now use native public construction/length
routes and the actual three-column Chebfun helix curve. All five cases
passed in four serial gates; root rehashed2,939/2,938/2,946/2,940 files.
HOSVD assertions11–19 remain unchanged from their accepted qualification.
The earlier positional-domain Python adapter failure is retained; v2
changed only the required domain keyword. Native predicates/tolerances
remain. Evidence: docs/chebfun3_native_guide_cpu_20261010.json. Full extrema
and line-integration algorithm equivalence is not established by this gate.

## Norm JIT interval check repaired (2026-10-10; latest)

Interval overlap now checks static Domain endpoints as Python metadata,
avoiding a traced boolean conversion while retaining the native scaled
tolerance and unbounded-domain handling. No coefficient is host-converted.
All23 existing focused tests passed, including the original norm value
and derivative assertions, restriction and mismatch controls; root
verified2,964 runtime files. Evidence: docs/normjit_overlap_cpu_20261010.json.
The preceding79-pass/1-failure broad run remains recorded; its remaining
cases still need completion.

## Native sine constructor dispatch fixed (2026-10-10; latest)

Chebfun3.sin now resamples through the default constructor as native
sin.m requires; generic compose still preserves periodic technology.
All nine original sine predicates and new periodic/empty controls passed
in six CPU cases. Root independently verified2,945 runtime files and
that all non-sine module/class AST remained unchanged. Evidence:
docs/chebfun3_sin_dispatch_cpu_20261010.json. This closes the separately
recorded periodic-object sine dispatch gap; other unary methods and
global constructor/evaluation host paths remain under audit.

## Native max3 assertions restored (2026-10-10; latest)

Both original cosine/sine predicates now run with the shared constant
Chebfun reference and native continuous norm threshold. One CPU case
passed; root independently verified2,947 runtime files. Evidence:
docs/chebfun3_native_max3_cpu_20261010.json. This replaces the surrogate
quadratic-only native test; the current extrema implementation still uses
a different algorithm, so full extrema/source parity remains unresolved.

## Supported preference test corrected (2026-10-10; latest)

The obsolete unsupported-trig expectation is replaced by a positive
periodic type/evaluation control; the focused CPU case passed and root
verified2,939 runtime hashes. Two obsolete domain/exps negative cases
are removed with exact existing positive nodes documented. This corrects
a Python contract test, not a native MATLAB assertion, and does not claim
equal assertion counts. Evidence: docs/constructor_preference_test_contract_cpu_20261010.json.
Full broad remainder and CI remain open.

## Atmospheric presentation replay integrated (2026-10-10; latest)

The full CPU page completed with ten fresh600x270 figures. Discrete jet64,
figure01 sphere/colorbar placement and default single-contour color now
follow the supported source/reference evidence. Stdout and non-timing
events match the previous84-baseline run; north-pole output cell now
records0.624920062687918. Root accepted the rendering controls and
3,067-file fullpage audit. Evidence: docs/atmospheric_presentation_cpu_20261010.json.
Historical Poisson field, exact parula and minor styling remain unresolved;
this is not complete visual parity or a current whole-suite/CI claim.

## Native continuous HOSVD restored (2026-10-10; latest)

HOSVD now follows continuous factor QR and discrete mode SVD in JAX,
removing the old real-only host arrays and Cholesky jitter. The five-output
adapter supplies continuous singular factors. Seven CPU cases passed:
all nine original HOSVD predicates, guide11–19 and complex, periodic,
dependent-factor and zero controls. Original tolerances remain; native
guide11–13 tautologies are retained literally. Root rehashed 2,972 and
2,968 runtime files. Evidence: docs/chebfun3_hosvd_cpu_20261010.json.
Empty-output semantics, earlier guide predicates, global host paths and
full current-suite/remote CI remain open.

## All native Chebfun3 sine assertions restored (2026-10-10; latest)

All nine original predicates now run, including fiberDim1/2/3, coordinate
object construction on the varying domain, trig evaluation and eps alias.
Four pytest cases passed in three serial CPU gates; root independently
verified 2,940/2,942/2,940 runtime hashes. Original thresholds remain.
Evidence: docs/chebfun3_native_sin_cpu_20261010.json. Native test8 checks
scalar sine, so periodic-object sine dispatch remains separately open.
Full suite and remote CI remain unresolved.

## All native Chebfun3 plus assertions restored (2026-10-10; latest)

The canonical plus test now contains all10 original MATLAB predicates:
three domains with continuous norm, scalar-output rank comparisons, tiny
and large scales, and mixed technologies. Eight pytest cases passed in
five serial gates; root independently verified2,940 runtime files per gate
(2,942 for mixed technologies). Original thresholds and construction order
remain. Evidence: docs/chebfun3_native_plus_cpu_20261010.json. This restores
previously omitted assertions; the82-file native inventory still contains
semantic gaps in sin/max3/guide and is not full parity.

## Empty abs dispatch fixed (2026-10-10; latest)

Chebfun.abs now returns the empty input before inspecting array metadata,
as native abs.m requires. The original empty propagation test and added
identity check pass. A separate stale Chebfun3 error regex now checks the
native plus identifier/message; its original mismatch case passes. Root
verified2,942 and2,939 runtime hashes. Evidence:
docs/empty_abs_and_domain_contract_cpu_20261010.json. Prior broad failures
and their unexecuted remainder remain recorded; full CI is not qualified.

## Chebfun3 active arithmetic integrated (2026-10-10; latest)

Addition now follows native condition-scaled adaptive resampling; power
follows native empty/type/sign dispatch. Empty ACA pivots and Cartesian
callback handling are corrected. All78 merged cases across12 gates passed,
including all nine native multiplication cases and all eight original
repeatedArithmetic clauses. Root independently reviewed every gate and
verified the seven integrated files against the qualified snapshot. Current
sum/sum2/sequential sum3 methods are retained unchanged. Evidence:
docs/chebfun3_active_arithmetic_cpu_20261010.json. Repeated clauses used
separate processes; original loop counts and tolerances remain. Full native
constructor/HOSVD/evaluation parity, inherited host paths, broader tests,
performance comparison and remote CI remain unresolved.

## Native test expectations corrected (2026-10-10; latest)

Two broad-run failures were obsolete test expectations, confirmed against
7574c77 source. Two-endpoint Singfun integration now checks both returned
primitives against an independent semicircle integral;29 focused cases pass.
Invalid Chebfun restriction now checks the native error identifier/message;
the focused case passes. Production code is unchanged. Root independently
verified2,943 and2,940 runtime files. Evidence:
docs/native_test_contract_corrections_cpu_20261010.json. Both prior failed
broad receipts remain retained; their full reruns are still required.

## Numeric sequence constructor restored (2026-10-10; latest)

Numeric lists/tuples now convert to JAX arrays before atleast_1d in the
coefficient, trigonometric-values and Chebyshev-values branches. Adapter
dtypes are preserved. All27 focused cases passed, including both Chebtech
kinds, complex inputs, trig samples and native constructor-input tests;
root independently rehashed2,953runtime files. Evidence:
docs/numeric_list_constructor_cpu_20261010.json. Broadpart1 previously
stopped at119passes/1failure; that failed receipt is retained and rerun
is still required. Corepart0 separately ended unexpectedly with tool exit143
at roughly41%, without finalreceipt/JUnit; it remains unqualified.


## Sphere rendering integrated (2026-10-10; latest)

Default sphere surfaces now interpolate vertex colors and use source camera
orientation. The complete SphereHeat rerun produced ten610x276 figures and
all eight visible time titles, with byte-identical numerical stdout.
Root independently verified2,978runtime hashes;52 global regressions with
nonempty geometry checks and seven portable controls also passed. Evidence:
docs/sphere_rendering_cpu_20261010.json. Website layout is an explicit
measured presentation policy, not a port of opaque HG2 layout. Gaussian
MATLAB RNG remains unmatched; custom plotting kwargs retain the inherited
flat-color renderer. AtmosphericTemperature is next for actual-page impact
verification. Global CI and full pixel-level parity remain unresolved.


## AnalyticSVD computation completed (2026-10-10; latest)

All86 constructors completed:427 pieces finite and happy, five608x271
figures. Root independently verified2,946runtime hashes, constructor events
and all images. The example uses the exact checked-in historical Python
matrices by default; --matrix-input permits a future MATLAB capture.
Native rng(10) matching, visual/numerical parity and printed timing remain
unverified. The observer discarded timing stdout; the page now identifies
that missing capture instead of retaining an older output. The former4GiB
cap failure remains recorded; this distinct9GiB completion is not a speed fix.
Evidence: docs/analytic_svd_completion_cpu_20261010.json.

The full1,305-pair dimension audit now has53 mismatches across11 pages.
Dimensions alone do not establish figure content parity.


## Callable coefficient input validation fixed (2026-10-10; latest)

The public constructor now rejects callable input with coeffs=True before
callback normalization or coefficient-array construction. This restores
the Python rejection adapter for the native invalid bare coeffs marker.
Existing error assertion, callback-not-called control and native constructor
input module pass (three pytest cases); root rehashed2,950runtime files.
Evidence: docs/callable_coeffs_cpu_20261010.json. The earlier broad failed
shard remains recorded; full rerun and remote CI are still required.


## Golden fixture inventory (2026-10-10; latest)

A static scan of literal test reference names finds three missing files:
chebfun.mat, chebfun_ops.mat and chebfun_specfun.mat. The latter two now
have source-reviewed capture generators; the first already had one. No
new MATLAB capture has run. Two specfun tests still have unconditional
missing-fixture skips, and five ops tests skip at runtime. This inventory
is not complete dynamic test coverage. Evidence: docs/missing_matlab_fixtures_20261010.json.


## Ballfun scalar contract corrected (2026-10-09; latest)

The stale Python float assertion now checks the intended rank-0 JAX scalar;
sum/integral documentation agrees. Numerical values and tolerances are
unchanged. All24 scoped cases passed; root independently rehashed2,946
runtime files. Evidence: docs/ballfun_scalar_contract_cpu_20261009.json.
The failed broad shard is retained and still requires a complete rerun.


## Missing golden operation fixture (2026-10-09)

The five corepart2 skips require tests/references/chebfun_ops.mat, which
is absent. The old advertised generator was also absent; the restored
matlab_harness/refs/chebfun_ops_refs.m captures the five consumed fields,
checks the pinned clean Chebfun source and records MATLAB provenance.
It has only been source-reviewed, not executed. No reference values have
been fabricated; all five comparisons remain unresolved pending capture.


## SphereHeat public computation integrated (2026-10-09; latest)

Replaces the private harmonic shortcut with both original 100-step public
Helmholtz trajectories. The completed CPU run generated ten 610x276 figures
and all eight visible Time titles; independent audit rehashed 2,977 runtime
files. Actual output is now on the page; original prose and MATLAB cells
are unchanged. Evidence: docs/sphere_heat_public_cpu_20261009.json.
Camera, interpolated shading, layout, contour colors and MATLAB RNG remain
open: this is computational completion, not full page parity.

All 1,305 mapped figure pairs were reread and hashed: 58 dimension mismatches
remain across 12 pages (shared figure_size_current_sphereheat_20261009.json).
This dimensional audit does not establish visual/content parity.

Broad CPU shards stopped at first failures: part1 46 passed/1 failed
(Ballfun Python float assertion versus intended JAX scalar); part2 234
passed/5 skipped/1 failed (callable input with coeffs=True exception).
Both failures are being diagnosed; neither shard is qualified as passing.


## Partial Chebfun3 reductions qualified (2026-10-09; latest)

Partial sum/sum2 now preserve complex factor integrals, physical factor
scaling and native contraction order; invalid dimensions raise. All25
original sum/sum2/mean/mean2 norm assertions pass at their native tolerances,
plus12 complex/dimension controls. Independent runtime reviews verified
2941/2952 observed hashes. Evidence: docs/chebfun3_partial_reductions_cpu_20261009.json.
Inherited output reconstruction, global tests and remoteCI remain open.

SphereHeat fullpage has separately completed both100step trajectories and
all10 reference-size610x276 figures. Missing Time titles were found during
visual review; page rendering repair is underway, so fullpage parity is not
qualified. Evidence: shared sphere_heat_fullpage_root_runtime_review_20261009.json
and sphere_heat_fullpage_root_values_review_20261009.json.

## Sphere addition staging qualified (2026-10-09; supersedes caps below)

Persistent JAX stages preserve source sampling, ordered products, compression
and exceptional pole behavior. 102 numerical cases, compiler review, 16
portable controls and exact baseline/candidate initialization+two-step state
comparisons passed. The original Gaussian trajectory now completes all100
solves in a distinct6GiB completion run (peak3.56GiB). The older3GiB gate
failed after38solves and remains a performance failure. Fullpage/tenfigures,
MATLAB RNG matching and matched performance remain open. Evidence:
docs/sphere_addition_stages_cpu_20261009.json.

## Sequential sum3 qualified (2026-10-09; supersedes contraction gap below)

Native physical factor scaling and x→y→z contraction order are now restored.
Four focused executions (JIT enabled/disabled) and 24 reduction regressions
pass independent runtime review; all other library files are unchanged.
Evidence: docs/chebfun3_sum3_order_cpu_20261009.json. This does not establish
MATLAB bitwise equality, full-suite completion, or remote CI.

## Latest accepted CPU packages (2026-10-09; supersedes status below)

Local implementation head efc54edf includes:

- Chebfun3 sum3 complex factor integrals (efc54edf): removes three real-only
  casts; restores both original mean3 assertions at their native tolerance.
  All 24 scoped CPU cases pass, with 2,945 runtime hashes independently
  verified and full static gates passing. Existing multioperand contraction
  and post-contraction scaling remain a source-order parity gap. Evidence:
  docs/chebfun3_sum3_cpu_20261009.json.
- Chebfun3 std3 (7b63aed3): preserves complex mean and centered conjugation;
  original constant assertion plus five controls pass. Independent runtime
  audit rehashed 2,939 files; full static gates pass. Evidence:
  docs/chebfun3_std3_cpu_20261009.json. Complex-factor sum3 fixed below at efc54edf.
- Diskfun native weighted continuous SVD and default/2/fro norm formula
  (462e3342): 14 focused controls and six original deterministic native
  assertions passed. Independent runtime reviews rehashed 2,950/2,951 files.
  Public empty norm wrapper bypass fixed. New arithmetic is JAX; inherited
  public evaluation/adaptive construction host paths remain a library gap.
  Evidence: docs/disk_svd_cpu_20261009.json.
- Two-endpoint singular antiderivatives, physical split scaling and parent
  continuity carry (f0aa9f83): 25 controls and six unchanged native cases pass;
  all four runtime groups independently verified. Corrects first-kind endpoint
  carry and restores logarithmic-term rejection. Two cases emit nonconvergence
  warnings despite passing sampled accuracy checks; happiness is unqualified.
  Both broader Chebfun11/12 unmatched-input diagnostics passed sampled checks,
  but returned unhappy length65537 pieces. They do not qualify native RNG,
  uniform accuracy or convergence.
  Evidence: docs/singfun_cumsum_split_cpu_20261009.json.
- Chebfun3 empty outputs (6a34347a): restored the original 22-statement
  MATLAB sequence; 20 pytest cases passed with independent runtime review.
  Nonempty power remains under implementation. Evidence:
  docs/chebfun3_empty_cpu_20261009.json.
- Sphere factor selection (435dcffd): 34 checks passed; the full100 consumer
  still capped during solve six at 3 GiB after five completed solves.
  Evidence: docs/sphere_selection_cpu_20261009.json.
- Chebyshev source barplot data (eb091960): six literal-source controls and
  24 existing source plotting checks pass, with 2,948 runtime hashes reviewed.
  Raster/tick/backend parity remains open. Evidence:
  docs/plotcoeffs_barplot_cpu_20261009.json.

The latest full AnalyticSVD diagnostic on frozen22f73f99 still capped at4GiB,
now after62 completed constructors/three figures. Start-observed/prebound file
hashes are clean, but no final runtime report/JUnit survived the cap. Unmatched
historical RNG inputs and complete page/figure parity remain unresolved.
SphereHeat still caps during its sixth m150 solve after selection compilation
changes. Isolated broader addition staging has 74 exact controls passing;
compiler review and 28 unchanged regressions now pass independent review.
An actual Gaussian initialization plus two-step baseline/candidate state
comparison is running serially; full100 remains pending.
Chebfun3 power passed 48 added controls but failed original multiplication
statement five: 1.718e-9 versus 2.220e-12. A separately audited diagnostic
localizes that error to inherited subtraction compression (raw difference
about 7e-15). Literal active native addition has 28 helper controls passing. Its added
near-cancellation test initially expected an analytic residual where native
tolerance scaling predicts zero; that failed run is preserved. Corrected
source-truncation and resolvable-cancellation cases are in progress; neither
addition nor power is integrated. See shared chebfun3_power_candidate_20261009 evidence. Full CPU suite, all322 verified pages,68 known figure-size
mismatches, remaining source/RNG gaps, publication and exact-head green CI
remain open. Current worker handles live in the shared checkpoint JSON.


## Latest CPU qualification and open failures (2026-10-09)

Local implementation head 22f73f99 includes these qualified packages:

- Persistent real-Horner loop (5c191a08): 10 exact-eager fixtures and 23
  existing regressions; independent full-module AST identity after helper
  inlining and runtime reviews. Evidence: docs/trig_real_horner_cpu_20261009.json.
- Persistent polynomial endpoint helper (22f73f99): 49 scoped checks,
  independent runtime/IR review, 44 portable numerical cases and unchanged
  generic dispatch. No full AnalyticSVD memory/completion claim. Evidence:
  docs/endpoint_limit_fusion_cpu_20261009.json.

- Diskfun norm dispatch (4a07afe9): empty-before-dispatch, numeric infinity,
  native errors and even-order powers, including zero/negative orders.
  24 dispatch controls and six actual analytic cases passed with independent
  runtime reviews. Default 2/fro still uses inherited sampled quadrature;
  native Diskfun SVD/default norm parity remains open. Evidence:
  docs/disk_even_norm_cpu_20261009.json.
- Compressed Chebyshev quadrature (6b5a7101): native n-1 moment layout and
  shared endpoint assignment, with exact integer division for inverse-DFT
  scaling. All 18 unchanged native port cases and ten direct consumers pass;
  six source controls, two IR captures and six portable controls also pass
  (42 executions, including repeated controls). QR reconstruction assertions
  now use the native matrix infinity norm; matched seed6178 inputs remain
  pending. Five runtime groups were independently reviewed. The earlier
  strict n10 failure is retained. MATLAB FFT bit parity is not claimed.
  Evidence: docs/compressed_quadrature_cpu_20261009.json.

SphereHeat's two m150 solves complete under 3 GiB, but the attempted full100
trajectory completed only five solves before the sixth hit the limit. A
separate compiler/shape/live-array diagnostic capped during the fifth solve;
its four completed steps expose repeated real-Horner loop compilation and
about 1 MiB live array payload. Full trajectory, native RNG and figures
remain unqualified. The accepted Horner extraction still capped during the
sixth solve in a full100 consumer replay; five solves completed. Its numerical
qualification does not establish a complete memory fix. Shared evidence:
sphere_heat_public_source_20261009/
gaussian_full100_classification_v1/ROOT_REVIEW.json and
gaussian_steps_compile_v1/ROOT_REVIEW.json.

The full native Newton C2 scalarODE damping case now PASSES on 6b5a7101:
original residual predicate <1e-9, unchanged 180-second/3-GiB/width1026
limits, 56.257 seconds for the test and 3,057,940 KiB sampled peak. Independent
runtime review rehashed 2,970 observed files cleanly; no censoring or survivors.
Earlier capped trials remain preserved. This resolves this particular C2 gate,
not all Newton cases or general memory behavior. Evidence:
docs/newton_c2_compressed_quad_cpu_20261009.json.

AnalyticSVD's full diagnostic remains capped at 4 GiB after 58 constructions
and three figures, using unmatched historical NumPy matrices. A distinct
20-construction/three-figure attribution prefix completed with 2,035 compiler
calls and under 47 KiB sampled live-array payload. It supports investigation
of endpoint and evaluation compilation; it does not qualify a complete page,
MATLAB inputs, visual parity or executable memory ownership.

Full CPU suite, all 322 verified pages, 68 outstanding figure-size mismatches,
remaining library/source gaps, matched MATLAB captures, push and exact-head
green CI remain unresolved. Shared checkpoint records current owned handles.


## Latest CPU integration (2026-10-09; supersedes prior checkpoint below)

Local implementation head 0f120b1e includes native continuous factor QR/JAX
core SVD and BMC-I projection compilation. Full parity and remote CI remain open.

- SVD: 54 distinct scoped checks passed, plus six repeated Trig integration
  checks against the current FFT kernels. Includes all ten executed native
  chebfun2 norm assignments and 18 explicit capability-boundary controls.
  All seven runtime audits and independent root reviews passed. Nonfinite
  MATLAB behavior, extreme finite norms, custom zero constructors, generic
  axes and full JIT/AD remain unqualified. Evidence: docs/separable_svd_cpu_20261009.json.
- Sphere BMC-I: eight exact eager-byte cases and 17 unchanged source controls
  passed. The actual five-Gaussian initial condition now builds at about
  2.47 GiB process highwater, and the first m150 Helmholtz solve completes
  with mean drift 5.55e-17. The second step still exceeds the 3 GiB guard.
  No full-page or matched performance claim. Evidence:
  docs/sphere_bmci_fusion_cpu_20261009.json.
- A distinct first-solve compiler diagnostic completed at 3,038,960 KiB
  sampled tree peak. Initialization produced 1,333 compile starts; source
  projection, real conversion, factor scaling and prolongation remain
  prominent. Evidence: shared scratch sphere_heat_public_source_20261009/
  gaussian_firstsolve_compile_v1/ROOT_REVIEW.json.
- Newton fusion: 38 fixture tests/39 captured outputs remain byte-identical;
  inherited differentiation regressions are running. Some inherited tests
  substitute deterministic inputs or weaker assertions, so they do not
  establish complete native assertion parity. Separately, a literal native
  compressed quadrature candidate is entering source-stage/IR qualification.

No page/figure inventory reduction is claimed: 68 size mismatches across
13 pages remain. Publication remains local; push and exact-head green CI,
full CPU suite, missing MATLAB captures and all-page verification are open.


## CPU numerical checkpoint (2026-10-09; supersedes older status below)

Full MATLAB/library/test/page parity is still incomplete. Latest qualified
implementation checkpoint: b90aa677, local only; exact-head remote CI remains
unverified. No GPU work was used.

- Native periodic inner products now use summed-length trapezoid quadrature,
  fixing even-Nyquist norms. All eleven original assertions, ten focused
  controls and two scalar QR regressions passed (23 total). Evidence:
  docs/trig_inner_product_cpu_20261009.json.
- Two persistent JAX FFT transform kernels passed 25 unchanged native
  assertion bodies directly on JAX, plus eight empty/nonfinite/AD controls.
  Evidence: docs/trig_transform_fusion_cpu_20261009.json. These root gates
  used source input guards; they do not establish full dependency/runtime parity.
- Native active-set direction/blocking primitives passed 16 controls. The
  finite-box QP working-set/phase-I and explicit-provider SQP engines passed
  another 19 controls, including backtracking, curvature repair and budget
  restoration. Independent runtime audits were clean. Public extrema are not
  switched to this engine: protected finite differences, native RNG and
  degenerate branches remain unresolved. Evidence:
  docs/active_set_qp_primitives_cpu_20261009.json and
  docs/active_set_box_qp_sqp_cpu_20261009.json.

An isolated SphereHeat rewrite completed the first 100 public Helmholtz solves
and five 610x276 figures. Its error norm was 2.3252808304058767e-05 versus cached
website 2.325280830910560e-05. The website uses flipud(hot) and different limits
from the pinned example .m; the draft now follows the website colors, but fresh
visual verification remains open. With the FFT kernels, the five-Gaussian
initial condition finishes; the first m150 solve still hits the 3 GiB guard.
The draft and figures have not replaced the published page. Figure inventory
therefore remains 68 size mismatches across 13 pages, with broader visual and
printed-output parity unresolved.

Actual Newton trajectory instrumentation stopped at 2.7 GiB during the second
update. Differentiation and quadrature accounted for 98 of 187 new compilations
in that update; final live arrays were zero. Stable helper kernels are under
qualification, with no claimed full-solve memory fix. SVD/norm work is also
under native regression qualification, not integrated or complete.

Continuing-work evidence and owned handles are in shared scratch:
/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/
current_parallel_checkpoint_20261008.json. Do not rerun a live handle or treat
a capped/partial diagnostic as native acceptance. Full CPU suite, all322 fresh
pages, complete function/assertion coverage, MATLAB captures, push and green CI
remain outstanding.

## Fixed sphere surface color limits (2026-10-09)

Public sphere plotting now accepts clim=(low, high), applying the fixed scale
to the surface facecolors and colorbar snapshot together. Map projections use
the same limits. This supports the source heat-page caxis commands without
changing only a detached colorbar. Omitted limits retain automatic scaling.

Twelve CPU controls passed: seven existing mappable/lighting checks and five
fixed-limit/validation checks. Actual surface color inputs, map limits and
colorbar normalization were checked. Peak RSS was 619168 KiB; source inputs
were stable with no censoring or survivors. Whole-page and pixel parity remain
open. Evidence: docs/sphere_fixed_color_limits_cpu_20261009.json.

## Sphere contour overlays preserve held plots (2026-10-09)

Public Spherefun.contour now accepts explicit hold=True, representing the native
ishold branch. It adds contour lines without an extra background sphere and
preserves supplied axes camera, limits, aspect and layout. Default standalone
contours retain their background and layout behavior.

Four CPU checks passed: held surface/geometry preservation, standalone backing
sphere and two unchanged line-property cases. Peak RSS was 861460 KiB, with
stable guarded source inputs, no censoring and no survivors. This qualifies
artist behavior, not rendered pixel or whole-page parity.

The SphereHeatConduction page still uses a private harmonic stepping surrogate,
omits source colorbars/mean contours and has ten wrong-size figures. Restoring
its public Helmholtz workflow and fixed color scale remains open. Evidence:
docs/sphere_contour_hold_cpu_20261009.json; source page audit in shared scratch
sphere_heat_source_audit_20261009/SOURCE_AUDIT_v1.json.

## Existing Chebfun3 constructor identity (2026-10-09)

The public constructor now returns an existing Chebfun3 directly, matching
native constructor.m before fixed-rank processing. It no longer resamples the
object or changes its domain when construction flags are supplied. Native
EQUI validation still precedes that return.

Five bounded CPU/x64 controls passed: empty and nonempty identity, domain/rank
flags, fiber/technology flags, and EQUI validation order. The process closed
uncensored with stable source inputs, no survivors and 496544 KiB peak RSS.
Loaded library source hashes and exact applied bytes were checked; this was
not a full third-party runtime audit or a numerical approximation test.
The larger Chebfun3 algorithm/dispatch gap remains open.
Evidence: docs/chebfun3_identity_cpu_20261009.json.

## Chebfun3 constructor factory preference (2026-10-09)

Restored cheb3Prefs.constructor='chebfun3f' from native chebfunpref.m740.
Seven direct preference checks passed: complete factory fields, session
selection, independent factory access, nested factory reset, partial structure
merge, copy isolation and full reset. These used a standard-library module
load; no numerical solver or pytest suite was run for this small state change.

Chebfun3 still does not consume this field to select the native classic versus
chebfun3f algorithms. That dispatch/algorithm gap remains unresolved; restoring
the field does not establish constructor parity. Evidence:
docs/chebfun3_factory_preference_20261009.json.

## Retained 2D pivots and complex norm correction (2026-10-09)

Native construction now retains original raw pivot values alongside legacy
reciprocal CDR weights. Supported arithmetic propagates that metadata; adapters
without native metadata mark it unknown. Source extrema uses retained pivots.
Scalar division preserves the native division/reciprocal operation boundaries.

Corrected the complex Hermitian norm formula to conjugate the first CDR weight
in both Chebfun2 and SeparableApprox. Restored original native imag/complex and
vector conjugation predicates, including function norms and expression order,
without widening bounds. Eighty-six CPU cases passed in thirteen serial groups;
root independently checked exact payloads, results and runtime evidence.

Full native norm/SVD parity remains open: the inherited Gram norm and sampled
NumPy QR/SVD compression still differ from the native continuous algorithm.
The legacy pivots property still exposes CDR weights; pivot_values exposes raw
values. Native normalized getter migration, active-set extrema and the full
Gibbs example remain unresolved. No MATLAB complex bit-identity claim is made.
Evidence: docs/pivot_complex_norm_cpu_20261009.json.

## Periodic QR source restoration (2026-10-09)

Built-in two-output Trigtech QR now uses JAX QR, native weighting/sign order,
and authoritative value caches. Public QR preserves original array-valued FUNs;
true quasimatrices still undergo native restriction before concatenation.
The eight native two-output tests use exact accepted seed6178 query values and
the source matrix-infinity residual predicate, with unchanged bounds.

Twenty-eight CPU cases passed: sixteen focused controls, eight native periodic
cases and four polynomial regressions. Root verified all tested payloads,
JUnit outcomes and runtime files. An earlier failed public periodic control
is preserved; its array-provenance bug was fixed without bypassing restriction.

Three-output pivoting and Householder selection remain incomplete. Inherited
single-column and generic fallback host/NumPy paths are not fully JAX-qualified.
This does not complete 2D SVD/norm, full-suite, page or CI parity.
Evidence: docs/periodic_qr_cpu_20261009.json.

## Newton damping preference consumption (2026-10-09)

Scalar, parameter and coupled Newton solvers now capture ChebopPref.lambdaMin
once per solve and use it in the native strict damping comparison. Six actual
solve controls passed, covering reduced steps and full-step fallback in all
three routes. Root verified the exact tested payloads, JUnit cases and runtime
files. Default numerical module ASTs match the previous implementation after
substituting the factory value 1e-6 and removing the new preference read.

The unchanged native C2 damping regression remains UNRESOLVED: its run hit
the 3 GiB RSS cap (3151380 KiB peak), without a result, JUnit or end manifest.
This is not a passing regression. Inputs were stable and no processes survived;
only the startup runtime could be audited. Width observations reached 258.
The next step is memory/compilation diagnosis, with no unchanged retry or cap
increase. Callable/parser parity and the full CPU suite remain open.

Evidence and exact scope: docs/newton_lambda_min_cpu_20261009.json.

## PTDecomposition source plot structure and dimensions (2026-10-09)

The example now calls public Ballfunv.quiver and Ballfun.plot. This replaces
invented sparse blue arrows and flat scalar images with four magnitude-colored
vector plots and two five-surface 3D scalar plots. All three generated figures
have the reference 600x253 dimensions. Fresh printed outputs are recorded in
the page. One full CPU example run passed its structure/size checks (69.39s
pytest, 2871584KiB peak); root audited 2960 runtime files and inspected all
three generated images against the references.

Full visual parity is NOT established: lighting/depth sorting, arrow scaling,
ticks and spacing still differ; some edge tick labels are clipped. Numerical
outputs differ from the retained MATLAB rerun, with no new tolerance invented
to declare a match. The original computation sequence is unchanged. The page
remains open for numerical and renderer parity despite corrected plot types
and sizes. See docs/ptdecomposition_cpu_plot_refresh_20261009.json.

## Operator preference factory and complete native test (2026-10-09)

Restored top-level scale=NaN, lambdaMin=1e-6 and happinessCheck factory fields.
The happiness selector retains the existing Python string representation;
independent operator defaults no longer inherit technology-check overrides.
The canonical preference test now preserves all ten native predicates, including
an explicit NaN-aware comparison of represented preference state.

Nine CPU cases passed: eight focused controls and the ten-predicate canonical
case. Root verified exact tested payloads, JUnit names and 2929 runtime files;
the process closed uncensored with no survivors. Full Ruff/F821 and provenance
checks passed. These tests exercise preference state, not numerical solvers.

Native callable identity, entry-specific parser rules, flat object shape and
solver consumption of scale remain incomplete; lambdaMin is addressed above. Existing default
solver read-values are unchanged; no new combined solver run is claimed.

Evidence: cheboppref_factory_source_plan_20261009/DELIVERY_v1.json and
cheboppref_factory_root_runtime_review_20261009.json /
cheboppref_factory_root_results_review_20261009.json in shared scratch.

## CPU inverse compilation reduction (2026-10-09)

Default colleague roots now compile matrix construction with input checks, then
eigenvalues with output metadata, in two JAX phases. Host failure boundaries,
provider ordering, matrix arithmetic and writable output ownership are retained.
Qualification: 45 helper cases and 30 inverse/root regressions passed with the
original bounds, including the full flower inverse. Retained compiler IR was
reviewed for source operation order and unchanged LAPACK calls.

A matched CPU run measured first inversion after setup at 24.437788s baseline
and 17.574701s candidate (1.3905x observed), with 544 versus 370 compilation
calls. Five warm calls had medians 0.614889s and 0.608158s; this small difference
is not a general warm-speed claim. Setup was 19.321124s versus 19.513769s.
All 80 captured arrays per arm were byte-identical, including inverse outputs;
roundtrip error remained 1.0547118733938987e-15. Both processes and runtime
audits closed cleanly. Root independently rehashed 2964 runtime files per arm.

This is one instrumented process per revision, after setup that itself invokes
roots. It is not a pristine-cold or matched MATLAB comparison. Full CPU-suite,
all-page/figure parity, remaining correctness issues, publication and CI remain
open. Complex separable norms also have a newly confirmed weight-conjugation
bug; its repair is recorded above and is separate from this inverse change.

Evidence in shared scratch: roots_two_phase_fusion_plan_20261009/
QUALIFIED_TIMING_PACKET_v1.json, QUALIFIED_CORRECTNESS_PACKET_v1.json,
QUALIFIED_HELPERS_PACKET_v1.json; roots_fusion_timing_root_runtime_review_20261009.json
and roots_fusion_timing_root_values_review_20261009.json.

## Native coupled RHS parsing and restart preference (2026-10-09)

The native coupled IVP route now validates exact RHS endpoint domains before
selection, broadcasts only numeric scalars, and reads numeric/ChebMatrix entries
in native column-major order. Array-valued Chebfun columns and RHS interior
breakpoints are retained. Insufficient entries raise; surplus entries follow
the pinned source indexing. Domain errors cannot silently fall back to a legacy
solver. Row Chebfuns and complex/logical/higher-rank RHS remain unsupported.

The native ivpRestartSolver=True factory preference is now a top-level field,
so overrides, copying, session defaults and factory reset use the right storage.
The prior solver fallback was already True; default numerical options are equal.

Qualification:19 RHS cases (16 focused, native initialConditions9/10 through
numeric-column RHS, one independent analytic solve) plus5 preference cases
(4 focused and the canonical7-predicate preference test). Original bounds are
unchanged. All five CPU processes closed cleanly; root verified runtime hashes,
JUnit, exact payloads and source applicability. Combined read-value equivalence
is source-reviewed, not a new combined solver run. Previous Lorenz/Brusselator
expression trees and solver/provider calls are unchanged.

Source provider remains R2025b. General solver grammar, callable solver tokens,
remaining preference factory/NaN-comparison gaps, full Consensus, fresh MATLAB,
full-suite, page parity, publication and CI remain open.

Evidence: rhs_preference_root_delivery_review_20261009.json,
coupled_ivp_rhs_source_20261009/DELIVERY_v1.json and
ivp_restart_preference_source_20261009/QUALIFIED_PACKET_v1.json in shared scratch.


## Full native Brusselator cell-syntax test restored (2026-10-09)

The canonical test now preserves all six pinned predicates, the full [0,5]
interval, sequential operator reuse and boundary-condition updates. Continuous
solution differences use the public ChebMatrix Frobenius norm, replacing the
previous maximum-block surrogate. Cell syntax and multi-argument syntax both
select the accepted native ode113 route.

Six focused syntax/options/norm controls passed (27.14s), followed by one
sequential canonical test containing six native predicates (48.96s). Original
1e-14 endpoint bounds and four exact-zero difference comparisons are unchanged.
Root independently checked JUnit, all six post-assertion markers, both test
payloads, current production bytes and 2959/2960 observed runtime files. Peak
canonical process-tree RSS was1576620KiB. The CUDA plugin compatibility warning
is preserved; these were explicitly CPU-only runs.

This qualifies the pinned Brusselator test with the ported R2025b ode113
provider, not fresh MATLAB execution or general cell grammar/preference parity.
Numeric-array forcing, callable solver preferences, higher-order/complex/events,
full Consensus rendering, full-suite and CI remain open.

Evidence: brusselator_root_results_review_20261009.json,
brusselator_focused_root_runtime_review_20261009.json and
brusselator_source_root_runtime_review_20261009.json in shared scratch.


## Coupled native IVP route and restored Lorenz predicate (2026-10-09)

Structurally recognized real first-order coupled initial-value problems now
use one vector ode113 solve through the public Chebop interface. The route
preserves reordered initial conditions, forward/backward spans, forcing
breakpoints and native solver preferences. Once selected, native solver errors
propagate. Unsupported default grammar retains the existing route; explicit
unsupported ode113 requests raise.

Four bounded CPU gates passed 17 unique cases: 14 structural/options/error
controls, two analytic forward/backward solves with original 100eps bounds,
and the native Lorenz endpoint predicate on [0,5] with its original 1e-14 bound.
The canonical Lorenz test replaces the weakened [0,3]/LSODA/1e-6 adapter.
Root independently checked runtime evidence, receipts, payload hashes and
AST identity when moving the qualified Lorenz predicate into its canonical file.
The duplicate focused Lorenz case was removed without numerical replay.

This uses the ported R2025b ode113 provider with the pinned Chebfun wrappers;
it is not fresh MATLAB execution or complete treeVar/options parity. Mixed or
higher derivatives, complex systems, events and broader forcing/solver forms
remain unsupported or on the legacy route. Actual cell-style Brusselator tests,
full Consensus rendering, the full suite and CI remain open.

Evidence: coupled_native_ivp_source_20261009/DELIVERY_v2.json,
coupled_ivp_root_delivery_review_20261009.json and the A/B/Lorenz root runtime
and results reviews in shared scratch.


## Public separable extrema source path (2026-10-09)

Real Chebfun2 extrema now use the pinned empty/zero checks, fixed4000 factor
construction with selected preferences, signed scaling, rank-one extrema and
the previously qualified higher-rank fallback. Separate min/max requests each
compute both extrema as the source does. Native empty outputs are numeric.
The inherited complex/nondefault optimizer body is unchanged.

Qualification: 28 focused controls plus one native empty regression passed in
four bounded CPU processes. Root independently verified runtime files, all
payloads/JUnit counts, source scope and unchanged legacy optimizer AST. The
original turbo accuracy control remains archived as failed: exact summation
showed cancellation already in its large-ellipse samples. Corrected controls
check turbo stage semantics; two separate non-turbo controls retain the original
analytical bound. No native tolerance was changed.

This is incomplete native extrema parity: original pivot bits are still lost
by reciprocal storage, and a source-equivalent active-set optimizer is absent.
The explicit unavailable capability selects the source fallback. Raw-pivot
retention is the next representation package; full Gibbs rendering, historical
optimizer behavior, fresh MATLAB, full-suite and CI remain open.

Evidence: gibbs_frontend_root_delivery_review_20261009.json,
gibbs_frontend_root_runtime_review_20261009.json and
gibbs_public_extrema_source_20261009/FRONTEND_DELIVERY_v3.json in shared scratch.

## Current CPU inverse measurements (2026-10-09)

Measured actual flower construction at d7c2c8d0: default inverse first call after
setup 22.272s, warmed median 0.612s; derivative roots 16.230s / 0.111s. Separate
setup costs were 18.488s / 18.847s. Repeated outputs were bit-identical; inverse
roundtrip error was 1.055e-15. First calls recorded 544/411 backend compilations,
while all warm calls had none. These are instrumented absolute costs, not a
matched speedup or a fresh MATLAB comparison.

Both CPU processes and independent runtime/capture audits passed. See
[measurement report](docs/inverse_cpu_profile_20261009.md) and its compact JSON
record for timings, scope, provider caveat and evidence hashes. Cold compilation
attribution is next; no optimization is inferred from overlapping profiler totals.
Full-suite, page/figure parity, publication and final CI remain open.

## Native trigonometric extrema delegation (2026-10-09)

Trigtech minandmax now constructs a full-array Chebtech1 from its evaluation
callback and delegates to the accepted Chebtech extrema implementation, as the
pinned MATLAB source does. This replaces the direct Fourier derivative shortcut.
Canonical tests preserve complex results, paired location evaluation, native
vscale factors and the complex matrix infinity norm.

All seven native extrema predicates passed across five bounded CPU processes.
Independent cross-review rehashed runtime files and checked commands, output,
source/payload bindings and process closure. Original commands did not request
JUnit; exact nodes and captured pytest counts establish their reported scope.
The integrated expression is an exact inline expansion of the tested conversion
helper. Whole-module AST comparison verifies that only minandmax changes.
Later dependency changes were reviewed; this is not a new-head runtime claim.

The strict complex-root source4 failure remains unresolved and its proposed
root implementation is not part of this commit. Empty representation parity,
full root options/backend behavior, fresh MATLAB and full-suite/CI gates remain
open. Evidence: trig_minandmax_independent_inverse_review_20261009/
INTEGRATION_REVIEW_PACKET_v1.json, trig_mthree_inverse_cross_review_20261009.json
and trig_remaining8_inverse_cross_review_20261009.json in shared scratch.

## Complete pinned inverse test battery at default preferences (2026-10-09)

Restored the native test setup: sine of the identity Chebfun, cumulative
arithmetic for the degree-nine sausage map, and explicit preference arguments.
All six inversion algorithms pass the original continuous-norm, composition,
range and endpoint predicates. Shared nonmonotonic-error, jump and decreasing
cases also pass. Native tolerances are unchanged.

Eight separate CPU processes passed 24 pytest cases: 21 cases exercise 33
distinct native predicates (48 pass-array cells), plus three labelled API/domain
controls. Independent cross-review checked runtime hashes, receipts, source
and JUnit bindings, and every snapshot production file against tested 94a1014a.
Root separately reviewed the source predicates and payload bindings. The later
Ball-only commit does not change inverse dependencies; no new-head run claimed.

This qualifies tests/chebfun/test_inv.m at default preferences, not arbitrary
preference contexts, fresh native execution, full-library parity or a speedup.
Current CPU profiling, remaining library tests, page audits and CI remain open.
Evidence: inverse_native_root_source_review_20261009.json and
inverse_native_cross_review_greeks_20261009/CROSS_REVIEW_FINAL_v1.json in shared
scratch, with inverse_native_predicates_plan_20261009/qualified_packet_v1.json.

## Ball volume integrals and norms (2026-10-09)

Integrated JAX coefficient-space volume integration and the source norm
construction, including literal radial/Fourier weights, source prolongation
sizes and real/complex return semantics. Only Ballfun norm/sum methods and a
new private integration module change library behavior.

Qualification includes 42 scoped controls, all 121 real spherical-harmonic
modes through degree 10 (363 norm observations), and 11 complex/normalization
cases. Root independently audited all 48 bounded CPU processes and verified
identical code/test payloads across their frozen snapshots. Native tolerances
were retained; the real-mode aggregate was run in partitions.

Four original SolidHarmonics scalar cells were executed and their printed
values updated. This is not a new full-page render or a degree-150 accuracy
qualification. Existing Ball adaptive/differentiation host paths, image
lighting/camera/interpolation differences, fresh MATLAB captures and final
full-suite/CI qualification remain open.

Evidence: ball_norm_root_delivery_review_20261009.json and
ball_norm_source_20261009/FINAL_HANDOFF.json in shared scratch. Source
applicability was reviewed through 94a1014a; no new-head runtime claim.

## Public constructor context and regression milestone (2026-10-09)

Integrated the source construction context: selected technology and private
preferences, full-domain scale, operator normalization, literal affine maps,
endpoint metadata and shared recursion through splitting, doubleLength and
cells. Session-selected Trigtech is now distinguished from the explicit
periodic parser flag; numeric breakpoint construction no longer raises the
wrong periodic error. The original failed regression is preserved.

Qualification: 29 helper controls, 53 focused public controls and 64 affected
regression cases pass. Root independently audited all process receipts,
observed runtime files and frozen payloads. Regressions include original
vectorCheck/equi norms, explicit-periodic rejection, doubleLength, actual
singular cells/autodetection at the original bound, and both unbounded maps.
Native clauses exceed pytest body counts; 146 controls are not complete
constructor coverage. New numerical implementation uses JAX.

Remaining: numeric-empty cell/native null-FUN semantics, command service
options, broader singular/unbounded modifier combinations, full MATLAB RNG
and full-suite/page/performance/CI verification. SciPy quad in one test is a
callback adapter, not a quadgk library port. Existing Python shape conventions
and the native even-length conjugation quirk are explicitly recorded.

Evidence: public_constructor_root_integration_20261009.json and
trig_constructor_r2_source_20261009/qualified_public_constructor_packet_v8.json
in shared scratch. Native MATLAB execution and publication remain unresolved.

## High-order derivatives and polynomial preprocessing (2026-10-09)

Chebtech1/2 now return a fresh resolved real-double zero when derivative order
reaches the stored coefficient count, matching diff.m. The ordinary recurrence
is unchanged. Ten controls pass, including huge orders bypassing recurrence,
column shape, happiness reset, JIT and coefficient derivatives. Root verified
2,929 runtime files; peak RSS was 638,504KiB. Finite-dimension exhaustion's
empty metadata remains an inherited gap.

Added an eager JAX polynomial-root preprocessing helper: finite/vector checks,
exact leading/trailing-zero handling, overflow-leading removal, companion
construction and zero-root prefix. Native zero/constant allocations preserve
real single/double class; eigenvalue concatenation promotes normally. Twenty-eight
distinct controls and four dtype rechecks pass; root verified 2,932 runtime files
per process and unchanged original control ASTs. This helper is not yet wired
into Trigtech. JAX eigenvalue rounding/order, rank>2 input handling and the
strict complex-root residual failure remain outside this qualification.

Evidence: diff_polynomial_root_integration_20261009.json and the corresponding
runtime/source reviews in shared scratch. Full-suite, page and CI gates remain
open; no native MATLAB execution or complete root parity is claimed.

## Differentiation early dispatch (2026-10-09)

Chebtech1/2 now return empty inputs before inspecting derivative arguments,
return the original object for zero-order differentiation before dimension
selection, accept None as the default first derivative, and use column finite
differences for every dimension other than one, matching the pinned source.
The coefficient recurrence is unchanged. Ten analytical controls pass,
including empty shape/dtype preservation, scalar and array identity, exact
polynomial and column differences, JIT and coefficient derivatives. Root
verified all 2,929 observed runtime hashes and frozen payloads; peak RSS was
679,156KiB. Factory-empty representation and high-order happiness-reset gaps
remain open; this does not qualify the full differentiation suite.

Evidence: chebtech_diff_dispatch_source_20261009 and
chebtech_diff_dispatch_root_runtime_review_20261009.json in shared scratch.
Full-suite, page, native MATLAB execution and CI gates remain incomplete.

## JAX array-root result assembly (2026-10-09)

Chebtech1/2 now assemble per-column root results and NaN padding with JAX
arrays, removing NumPy transfers from that wrapper. Each original scalar
technology, stored coefficient length, happiness flag, option and root order
is preserved. AST comparison proves scalar algorithms and unrelated code
unchanged. The helper remains eager because root counts vary.

Six independent polynomial controls pass for both technologies: real roots,
complex roots and zeroFun=false, with unequal column counts and NaN padding.
Root verified all 2,932 observed runtime hashes, frozen payloads and JUnit
counts. The bounded CPU process peaked at 704,528KiB. Inherited scalar QZ,
evaluation and other host paths, empty representation gaps and broad root
parity remain open; this is not a whole-library JAX claim.

Evidence: chebtech_array_roots_jax_20261009/SOURCE_REVIEW.json,
candidate/manifest.json and chebtech_array_roots_root_runtime_review_20261009.json
in shared scratch. Full-suite, example and CI qualification remains incomplete.

## Missing FUNQUI factory preference (2026-10-09)

Restored the native top-level enableFunqui=false factory field. Its omission
caused the first source-constructor control to raise AttributeError and routed
explicit writes into technology overrides. Four controls verify factory default,
private copy, session inheritance and field reset without losing unrelated
technology overrides. All pass; root verified 2,922 observed runtime hashes,
JUnit counts and the one-field-only production change. The failed constructor
run is preserved; its unchanged forty controls are being rerun separately.

Evidence: trig_constructor_r2_source_20261009/qualified_funqui_factory_packet_v1.json
and funqui_factory4_root_runtime_review_20261009.json in shared scratch. This
fix does not establish full constructor/preference, example or CI parity.

## Native trigonometric zero-target truncation (2026-10-09)

Nonempty Trigtech truncation to zero now reaches the native coefficient
indexing error after deleting all rows, including even input lengths after
Nyquist expansion. The shared helper's initial guard is the only production
AST change; positive-target arithmetic is unchanged. Already-empty zero-target
construction retains its object and stored empty shape/dtype.

Eight focused controls and all eleven original prolongation predicates pass.
The canonical array tests now use MATLAB's matrix infinity row-sum norm and
recompute values from coefficients; same-length equality also compares the
recomputed transforms. Original numerical bounds are unchanged. Root verified
all 2,929 and 2,933 observed runtime hashes respectively, source scope and JUnit
counts. Sampled peaks were 708,432 and 1,450,328KiB.

Negative targets and empty-to-positive prolongation still have inherited source
gaps. No native MATLAB error identifier or complete constructor qualification
is claimed. Public constructor tests are running separately. Evidence:
trig_constructor_r2_source_20261009/qualified_trig_prolong_zero_packet_v1.json,
trig_prolong_zero_root_source_review_20261009.json and both root runtime reviews
in shared scratch. Full-suite, page, speed and CI gates remain open.

## Trigonometric empty numeric storage (2026-10-09)

Fixed-zero Trigtech population preserves the sampled empty array's columns and
complex dtype. The values getter returns an explicit stored cache before using
the cacheless empty fallback, matching native stored-value observation. Seven
unchanged controls pass, including actual callback order and PyTree/JIT storage;
root verified all 2,930 runtime hashes and the two-function change scope. The
first run exposed the values getter's complex-storage loss and is preserved as
a failed run. Numeric empty population differs from the initial null shortcut,
which is unchanged. Empty logical-mask dimensions remain a Python adaptation.

The public constructor's 53 draft controls are still unqualified; source review
also found a separate numeric-to-zero prolong error case requiring correction.
This package does not establish full constructor, full-suite or CI parity.
Evidence: trig_constructor_r2_source_20261009/qualified_trig_empty7_packet_v2.json
and trig_empty7_root_runtime_review_20261009.json in shared scratch.

## Separable extrema fallback transaction (2026-10-09)

Added private JAX helpers for original CDR evaluation, column-major seed
selection and the source optimizer/fallback transaction. Literal physical maps,
partial assignments after exceptions and source Nelder–Mead fallback are
preserved. Seventeen distinct controls pass; two targeted reruns verify the
standalone helper extraction. Root checked all 2,943 observed runtime hashes
in each successful process, unchanged function ASTs and identical test bytes.
The intervening fixed-zero polynomial changes affect neither the positive-length
nonempty fixtures nor their executed algorithms.

This is a dependency for public extrema, not completed public dispatch. Native
active-set optimization, fixed-4000 constructor routing, complex/nonfinite cases,
Gibbs2D page qualification and native MATLAB execution remain unresolved.
Evidence: gibbs2d_extrema_source_20261009/FALLBACK_DELIVERY_FINAL_v2.json,
extrema_fallback_root_delivery_review_20261009.json and the two root runtime
reviews in shared scratch. Seventeen controls plus two rechecks are not nineteen
distinct controls. No full-suite, speed, figure or CI claim follows.

## Fixed-zero polynomial callback and empty scale (2026-10-09)

C1/C2 fixed-length zero construction now calls the operator on the empty grid,
populates its actual numeric result and prolongs to zero. Empty callback-result
shape/dtype and nonempty constant/array column counts are preserved; callback
exceptions propagate. Scalar vscale returns zero for empty coefficient storage,
including the existing null adapter, matching the native empty scale branch.

Fourteen source controls pass in a fresh CPU process (28.75s pytest,
640,424KiB sampled peak). Root verified all 2,931 runtime hashes and proved
positive-length construction and unrelated Chebtech bodies unchanged by AST.
The first draft passed its fixtures but incorrectly collapsed explicit empty
shapes/dtypes; it is preserved and excluded from source qualification. Corrected
v2 follows the native vals2coeffs n<=1 identity branch. One-dimensional empty
arrays retain the established Python vector adapter.

Public retained-empty-FUN/domain behavior awaits the constructor package.
Default null ishappy representation, negative-length legacy behavior and general
nonfinite extrapolation remain outside this qualification. Evidence in shared
scratch: chebtech_zero_fixed_source_20261009/DRAFT_REVIEW_v3.json,
SOURCE_CORRECTION_v1_v2.json and chebtech_zero_fixed_root_integration_20261009.json.

## JAX two-dimensional Nelder–Mead fallback dependency (2026-10-09)

Added the finite real two-variable fminsearch policy used by the native
separableApprox extrema fallback, following the available R2017a source.
Initial simplex, stable ties, reflection/expansion/contraction/shrink rules,
AND convergence test and complete-branch evaluation-budget behavior are
preserved. Twelve analytical and source-clause controls pass in a bounded
CPU process; root independently verified all 2,932 observed runtime hashes.

This private dependency is not yet wired into public extrema. The active-set
optimizer, historical page branch, nonfinite inputs and general dimensions
remain unqualified. A native protected finite-difference probe is prepared in
shared scratch but unrun because MATLAB IPC remains unavailable. No full-page,
full-suite, speed or CI claim follows. Evidence: gibbs2d_extrema_source_20261009/
FMINSEARCH_DELIVERY_FINAL_v1.json and fminsearch_root_runtime_review_20261009.json.

## Shared JAX polynomial extrema policy (2026-10-09)

Chebtech1/2 minandmax now follow the pinned source's constant midpoint,
[-1; roots; 1] candidate order, strict grid-minimum safeguard and complex
magnitude ordering with actual complex output values. Stored zero-column
arrays return empty outputs. The source's maximum-grid MIN quirk is retained.
The native complex-array test now uses its original matrix infinity norm,
with the original tolerance unchanged.

All 16 original source predicates pass across C1 and C2, plus 12 focused
controls. Three fresh bounded CPU processes terminated cleanly; root verified
runtime hashes, unchanged frozen payloads and unchanged code outside the two
minandmax methods. New arithmetic uses JAX. Native MATLAB execution, broad
JIT/AD behavior, NaN/complex-storage edges and ordinary marker-only Tech.empty
representation remain unqualified. Gibbs2D constructor and optimizer work is
separate and unfinished. No full-suite, plot, performance or CI claim follows.
Evidence: gibbs2d_extrema_source_20261009/CHEBTECH_DELIVERY_FINAL_v1.json and
chebtech_extrema_root_integration_20261009.json in shared scratch.

## Shared Trigtech constructor and composition core (2026-10-09)

Added the source preference/data, refinement, happiness and sample-check core
using JAX, including classic checks, callback order, fixed/numeric bypass and
literal refinement lengths. Trigtech.compose now uses that core with source
operand checks, length overrides, epsilon floor and disabled sample test.
All nine original low-level constructor clauses are restored.

Qualification: 38 analytical controls, nine native constructor clauses, ten
composition controls, eleven original composition clauses and fourteen affected
regressions (including JIT/AD and Frechet propagation): 82 distinct passing
bodies. Six bounded CPU processes terminated cleanly with stable input hashes
and no survivors. A combined 25-body run hit its 3GiB memory cap and is excluded;
the same tests passed in fresh 11/1/13 partitions with unchanged caps/assertions.
Root verified all runtime hashes, JUnit counts, source applicability and the
final documentation-only change. Default numeric factory bodies used by the
solid-harmonic package remain identical.

Full public Chebfun constructor parity is still unfinished: shared preference
and callback context, vectorCheck, existing-object technology selection,
recursive modifier order, full-domain hscale and endpoint metadata remain the
next package. Low-level malformed minSamples and integer endpoint storage,
complex-zero observation, broader Trig arithmetic and native MATLAB capture
also remain open. No full-suite, speed or CI claim follows from these checks.
Evidence: trig_constructor_r2_source_20261009/qualified_core_compose_packet_v2.json
and trig_constructor_core_compose_root_packet_review_20261009.json in shared
scratch. Publication remains local pending network access.

## JAX solid harmonics and radial restriction (2026-10-09)

Ballfun.solharm now follows the native modified-forward-column recurrence,
real/complex coefficient branches and conjugacy predicate using JAX. Radial
restriction uses paired Clenshaw contraction and the public Spherefun coefficient
constructor. All 14 original native test predicates pass: 121 real modes with
363 continuous norm observations, nine complex closed forms and two final
normalization cases. Their original tolerance is unchanged. Twenty-five
independent controls also pass; grouped execution uses identical source payloads.

The affected full SolidHarmonics page was rerun, including all 13 constructions,
degree 150/order 50, four scalar outputs and 11 plots/55 surfaces. Both images
are 600 x 253. The page body took about 16.17s with a sampled peak of 1,653,060KiB.
The observed degree-150 time is 1.202674s including completion/validation, not
an isolated benchmark or MATLAB speed comparison. Actual outputs and images
replace the previous implementation's artifacts; source prose and cells remain.

Root verified every qualification group, all 14 predicates, runtime hashes,
current-head applicability and both generated images. Inherited Ballfun norm,
differentiation and arithmetic still contain host numerical code. Native
lighting, camera/layout and exact figure parity remain open, as do native MATLAB
capture, publication and CI. Evidence under shared goal scratch:
ball_solharm_source_20261009/FINAL_HANDOFF.json, ROOT_NATIVE14_AGGREGATE.json,
and ball_solharm_affected_page_root_runtime_review_20261009.json.

## C2 eigenvalue policy and original basic predicates (2026-10-09)

Restored native C2 projection, automatic target selection, adaptive refinement,
mode filtering and continuous normalization. Smooth scalar Chebop and internal
Linop share the policy while retaining their explicit-size API conventions.
ChebMatrix differentiation now preserves continuous block semantics. All four
original basic-eigs predicates pass: both public routes, ten eigenpairs,
spectrum tolerance 1e-10 and continuous residual tolerance 1e-7.

Qualification comprises 14 controls, four original predicates and four affected
consumer/dispatch checks, in eight bounded CPU processes. Root independently
verified runtime hashes, native test scope, carried polynomial-path applicability
and documentation-only delivery changes. Full Ruff/F821 and provenance checks
are required at integration. An earlier combined control run hit its memory
cap; its partial dots are excluded and the unchanged tests passed in partitions.

The new policy is JAX; the inherited generalized eigendecomposition still uses
SciPy/NumPy. Alternative C1/ultraS defaults retain legacy fixed size 65, and
periodic/system/piecewise/general-boundary adapters remain separate parity gaps.
This resolves the basic-eigs issue described in older entries below, not full
eigensolver parity. Evidence: eigs_basic_source_20261009/DELIVERY_FINAL_v1.json,
eigs_delivery_root_documentation_review_20261009.json and the native/regression
root runtime reviews under shared goal scratch. Publication and CI remain open.

## Trigtech per-column realness (2026-10-09)

Static per-column masks now propagate through constructors, arithmetic,
calculus, column operations and existing-object factories. Public evaluation
retains the native aggregate projection; direct Horner supports explicit static
column masks. Source conjugation and scalar-row division semantics are covered.
45 focused checks and 14 unchanged regressions passed on CPU. Root independently
verified 2,942 focused and 2,950 regression runtime hashes; both runs terminated
cleanly with stable inputs and no resource censoring or surviving processes.

This qualifies the representation/propagation package only. Full constructor
preference/refinement behavior, inherited arithmetic formula gaps and native
complex-zero storage observations remain open. Evidence under shared scratch:
trig_real_columns_source_20261009/qualified_packet_v2.json and
trig_regression_root_runtime_review_20261009.json.

## Ballfun subplot preservation (2026-10-09)

Removed one unconditional figure-wide subplots_adjust call from the public
Ballfun renderer. Native plot changes the current axes; it does not reposition
all sibling subplots. Five actual-artist controls pass for default/wedge styles,
ten native subplot indices and standalone rendering, with exact position and
figure-parameter invariance. Peak831,672KiB; terminal0/stable/uncensored and no
survivors. Root independently verified2,954 runtime hashes, the exact one-line
change and saved subplot/standalone images.

No mathematical/data or camera algorithm changed. Supplied-axis camera reset,
lighting/interpolation and native default layout remain open. SolidHarmonics
images now use the corrected subplot behavior: a rendering-only source replay
verified 11 finite fields, 55 surfaces and two 600 x253 images. Root verified
2,949 runtime hashes and viewed both images. Original full-source scalar and
degree-150 evidence is carried unchanged; this is composite qualification,
not a repeated full-page computation. Evidence: ball_subplot_root_runtime_review_20261009.json and
ball_subplot_position_source_20261009/HANDOFF.json in shared scratch.

## WaveDecay source execution and reference dimensions (2026-10-09)

Both source40-mode eigenproblems now use the public adaptive eigensolver,
continuous infinity-norm normalization and public plotting. The fixed256
override, sampled normalization, warning suppression and synthetic stdout
were removed. The full CPU page ran in59.60s (3,351,892KiB sampled peak);
all eight actual eigenvalue labels match the historical three-decimal labels.
Both figures are600 x480 at72dpi. Source stdout is empty; prose and the two
MATLAB input cells remain byte-identical.

Root independently verified2,973 runtime hashes and viewed both actual/reference
image pairs. Maximum observed pencil was240 x240. Unchanged source algorithms
were bound across later Ball-only library changes. Historical eigenfunction
signs, second-figure color, subplot spacing, ticks and line widths still differ.
The inherited piecewise solver and canonical BlockLinop adaptive/targeting
semantics remain library parity gaps; this page does not qualify them.
The native basic-eigs test also needs its four original predicates restored.

Evidence: wavedecay_page_root_runtime_review_20261009.json,
wavedecay_root_integration_20261009.json and
wavedecay_source_20261009/DELIVERY.json in shared scratch.

## SolidHarmonics full source execution (2026-10-09)

The page now calls public Ballfun.plot for all eleven source plots, producing
55 surfaces and two correctly sized 600 x 253 images. The full source sequence
ran on CPU, including all thirteen finite constructions and degree150/order50.
The body completed in about9.48s with a1,242,736KiB sampled peak. Its observed
high-degree construction took0.207955s including completion/finite validation;
this is not an isolated benchmark or a MATLAB speed comparison.

Actual Laplacian norm:2.315864701948616e-14; squared norms:
0.999999999999999 and1.0; cross integral:1.118945367695923e-17.
Original prose/MATLAB input cells are preserved; output blocks use this run.
Root independently verified2,955 runtime hashes and viewed both actual and
reference images. Lighting/interpolation, visible surface seams, overall
framing and subplot positions/scales remain visibly different. This is source
execution and dimension qualification, not full figure parity. Inherited
NumPy numerical paths also remain part of the JAX-only goal.

Evidence: solid_harmonics_page_root_runtime_review_20261009.json,
solid_harmonics_root_visual_review_20261009.json and
solid_harmonics_page_20261009/HANDOFF.json in shared scratch.

## Ballfun default plot data and coefficient emptiness (2026-10-09)

Default public plot/surf now use the pinned coefficient prolongation,
radial-first transform, theta reordering, longitude closure and five source
slices. The new numerical helper is JAX-only. Ballfun.isempty now also detects
zero-size coefficient tensors; the empty factory remains supported. Native
empty-input test coverage and mixed zero-axis controls are included.

Qualification comprises 27 distinct passing checks across staged runs: three
grid rules, nine transform/slice controls, five public artist/wedge controls,
and ten empty/complex/canonical controls. Two earlier failures are retained:
a signed-zero error in the independent alias oracle, and the real public
isempty bug fixed here. Root independently rehashed 2,932/2,933/2,966/2,945
runtime files and verified applicability of carried checks. All three actual
192px public artist images were reviewed. A separate actual-plot check passes
with the scratch artifact environment variable absent; tests use pytest's
temporary directory in ordinary CI.

This qualifies default plot data and routing, not native lighting or surface
interpolation. Wedge empty/complex handling remains open. The full
SolidHarmonics execution is qualified in the entry above. The accepted sphere
contour viewport fix is preserved. Evidence: ball_plot_composite_root_{runtime_review,applicability}_20261009.json,
ball_plot_ci_fallback_root_runtime_review_20261009.json and
solid_harmonics_source_gap_20261009 in shared scratch.

## Atmospheric explicit-camera page replay (2026-10-09)

Integrated all ten 600 x 270 source-computation figures with explicit captured
camera properties. Full CPU replay completed in 197.95 s (4,323,316 KiB peak);
root independently verified 3,064 runtime hashes. Computed stdout and generated
prose are unchanged. The full replay exposed an overflowing contour viewport;
after the library fix, the exact figure 3 source cells were rerun in isolation
(18.16 s body, 1,517,916 KiB peak), with 3,053 runtime hashes verified by root.
The other nine images are byte-identical to the full replay. This is a qualified
composite, not another full execution on the final commit.

Figure 3 now fits the canvas. Historical colors/line widths, surface
interpolation, figure 1 colorbar framing, typography, and the historical
figure 4/6 scientific-reference differences remain unresolved. No complete
pixel parity or isolated visibility speedup is claimed.
Evidence: sphere_contour_viewport_source_20261009/PAGE_DELIVERY.json and
atmospheric_source03_root_runtime_review_20261009.json in shared scratch.

## Sphere contour viewport (2026-10-09)

Newly created sphere contour axes now use the existing non-overflow setup,
matching unit-sphere plotting. The sole library change is fill_canvas=False
at the contour_sphere setup call. Source grids, contour extraction, coordinates,
visibility and supplied-axes identity are unchanged. This fixes the legacy
oversized viewport exposed by explicit native camera mapping.

Eight focused checks pass in 38.67 s, including a saved analytic contour with
camera/layout, source-coordinate restoration, existing samples and line styles.
The run was terminal 0, stable and uncensored, with no survivors and a sampled
peak of 1,513,036 KiB. Root verified 3,053 loaded-file hashes, the one-keyword
AST change, and the saved image. The Atmospheric figure 3 replay is now qualified in the entry above;
historical pixel parity remains unresolved.
Evidence: sphere_contour_viewport_root_{runtime_review,integration}_20261009.json
and sphere_contour_viewport_source_20261009/DELIVERY.json in shared scratch.

## Periodic composition preference forwarding (2026-10-09)

Public periodic compose now copies raw preferences, applies source splitting
and multi-piece overrides, selects the actual operand technology, then resolves
its defaults. Explicit values equal to polynomial factory defaults are retained.
Unary periodic compose without an explicit preference now honors session
preferences. Low-level Trigtech numerical policy is unchanged.

All 21 focused checks pass in 31.94 s, including actual fixed-length and typed
compositions. The frozen run was terminal 0, stable and uncensored, with no
survivors and a 1,236,284 KiB sampled peak. Root rehashed 2,949 loaded files and
verified that only Chebfun.compose changed in its module. The first run's
invalid tuple-domain fixture and failure evidence are preserved; the correction
used the actual Domain storage type without changing production or tolerances.

Evidence: compose_preference_root_{runtime_review,integration}_20261009.json
and compose_preference_provenance_20261009/qualified_packet_v1.json in shared
scratch. Polynomial/singular forwarding, direct sin/cos shortcuts and general
mixed-technology composition remain open; this is not complete compose parity.

## Explicit native sphere camera mapping (2026-10-09)

The opt-in matlab_explicit_camera adapter maps explicit camera position,
target, up vector, data aspect and view angle into the actual viewport.
It corrects Matplotlib's default aperture and offset using source camera
geometry; no image-fitted zoom is used. Supplied axes and the angle-only
matlab_view helper retain their existing behavior unless explicitly opted in.
Later explicit view, projection or box-zoom changes opt out of this mapping.

All 31 controls pass, including two captured views, independent landmark
equations, resize/layout, depth and coastline visibility. The run completed
in 25.16 s with a 784,084 KiB sampled peak, stable inputs and no survivors.
Root verified 3,023 loaded-file hashes, matched camera fields to the native
capture, and viewed the saved test image. Its centered globe and coastlines
are visible; flat-face banding remains. This does not establish automatic
camera behavior, historical pixel parity, or a completed Atmospheric replay.

Evidence: sphere_explicit_camera_root_{runtime_review,integration}_20261009.json
and sphere_camera_scale_source_20261009/DELIVERY.json in shared goal scratch.

## Preference provenance and complete native preference test (2026-10-09)

ChebfunPref now stores explicit technology overrides separately from resolved
defaults. Reads use the selected technology's defaults; copies and session
preferences preserve omission, including an explicit value equal to a factory
default. Polynomial and periodic defaults stay distinct. Ordered default
updates and two-level field assignment/reset now follow the native manager.

Verification: 45 preference-object checks, 34 unchanged constructor checks,
and an 11-body followup covering all 28 predicates from native
`tests/chebpref/test_chebfunpref.m` plus 10 focused setter controls. These are
90 executions across three frozen runs, not 90 distinct native tests. All
runs were CPU-only, terminal 0, stable, uncensored, and left no survivors.
Root independently verified 2,928 / 2,941 / 2,925 loaded-file hashes and the
JUnit record of predicates 1–28. Only setDefaults changed after the first
two runs; the final followup covers that change and the complete native test.

Evidence: shared_preference_combined_root_integration_20261009.json and
shared_preference_provenance_20261009/combined_qualified_packet_v1.json in
shared goal scratch. Unknown-preference warning behavior, general callable
constructor forwarding, periodic composition, and broader API parity remain
open. The ChebopPref nested-field check covers Python compatibility only.

## Sphere visibility compilation adapter (2026-10-09)

The existing conservative ray test and final display mask now compile together.
Renderer inputs use eight bounded row shapes (32 through4096), preserving
vertex order, separators and original stored geometry. Camera and radius stay
dynamic; no camera-result cache or geometric offset was introduced. The outward
arithmetic-bound helper is unchanged and the frozen eager test oracle matches
its accepted source AST.

All29 focused geometry/painter/compiler controls pass in26.78s, peak0.84GiB;
terminal0/stable/no survivors. Root independently rehashed3024 observed runtime
files and verified the three integrated file hashes. Evidence:
sphere_visibility_root_runtime_review_20261009.json and
sphere_visibility_root_integration_20261009.json in shared goal scratch.
The matched synthetic baseline hit its 2 GiB RSS cap (original handle 63499;
2,101,996 KiB sampled peak, terminal 137, stable inputs, no survivors).
The candidate timing arm was not run. No speed ratio, full-page speed, or
figure-parity claim is supported by this censored comparison. Evidence:
sphere_visibility_compile_plan_20261009/FINAL_HANDOFF.json.

## Atmospheric current painter/layout replay (2026-10-09)

A fresh full source run on226ba700 produced all10 figures at600x270 with
accepted coastline/contour visibility and final-canvas layout for02/04/05–10.
Only the save-layout keyword changed in the script. Numerical stdout is
byte-identical to the prior successful run; page prose, MATLAB cells and
output text are unchanged. Root viewed all10 fresh images and independently
rehashed3063 observed runtime files; terminal0/stable/no survivors,257.10s
page body and4.62GiB peak. Current PNGs now show the previously clipped
labels/titles and visible contours/coastlines.

Full figure parity is still open: globe framing/scale (the native-inset
adapter makes current spheres smaller), flat-face striping, typography,
ticks/colorbars, and historical reference04/06 differences. The unqualified
Gouraud candidate is NOT included. Figure03 save took53.43s for183lines;
shape-specific eager JAX visibility work is being investigated. This timing
is an observed stage cost, not a controlled speed comparison.
Evidence: atmospheric_layout_full_root_runtime_review_20261009.json and
atmospheric_layout_root_integration_20261009.json; package
atmospheric_layout_optin_20261009/full_page_layout_v1.


## Trigtech fixed-length callback order (2026-10-09)

Fixed-length Trigtech construction now samples only its requested fixed grid
and endpoint before converting values, matching native constructor136–150.
The adaptive pseudo-random probe is no longer called for fixedLength; the
adaptive path and fixed-grid transform are otherwise unchanged. This restores
valid fixed-grid-only callbacks and callbacks nonfinite only at that unrequested
adaptive point. Thirteen focused controls pass in23.81s, peak0.76GiB;
terminal0/stable/no survivors, root2929 observed-file hashes clean.
Evidence: trig_fixed_probe_root_runtime_review_20261009.json and
trig_fixed_probe_source_20261009/focused_v1 in shared goal scratch.
General Trigtech preferences, native per-column realness thresholds and full
constructor parity remain separate open work.


## Numeric zero operator preference forwarding (2026-10-09)

Unbounded numeric zero construction now forwards C1/C2 technology preferences
through its native zero-valued operator boundary. Explicit keyword values win
over session values, including explicit False; omitted turbo/extrapolate retain
existing defaults on ordinary callable routes. Polynomial numeric input with
explicit adaptive preferences keeps the native direct-value transform. No
finite transform arithmetic or other constructor branch bodies changed.

All34 focused/regression controls pass in39.34s, peak1.39GiB; terminal0,
stable inputs/no survivors and root2943 observed-file hashes clean. Controls
include actual refinement callbacks, fixedLength/useTurbo and real-double
multi-column zeros. One prior test expecting numeric sample_test rejection was
corrected from pinned @chebtech/populate.m65–84 (numeric data returns before
adaptive refinement), without changing numerical bounds.

Trigtech general preference support remains incomplete; this package retains
its existing fixedLength behavior and explicit unsupported-override errors.
It does not establish complete constructor or callable-preference parity.
Evidence: numeric_zero_preferences_root_{runtime,integration}_review_20261009.json
and numeric_zero_preferences_candidate_20261009/controls_v1.


## Full original degree80 Vandermonde/Arnoldi execution (2026-10-09)

The example now uses actual public Chebfun powers, continuous least squares,
all80 literal Arnoldi iterations and an independent80-step evaluation recurrence.
All12 preamble outputs plus14 total scalars, both81-column arrays and81x80 H
were captured from the original source order. Source body369.30s CPU-only,
peak4.93GiB, terminal0/stable/no survivors; root rehashed2967 observed files.
The fresh600x253 figure and all output fences come from that run. Published
prose/MATLAB cells are preserved; previously scrambled output cells are repaired.

This replaces the sampled surrogate but does NOT establish page parity.
Ill-conditioned monomial max is1.0450 versus cached MATLAB1.5596; coefficient
infinity norm1.0862e14 versus3.5308e14. Several condition numbers and the native
near-singular warning also differ. The stable yA display matches cached text,
but source computations/figure pixels still need numerical/reference review.
Root viewed both images: same canvas size, different oscillations, axes/ticks
and framing. No source coefficients or limits were fitted to the reference.
Evidence: vandermonde_full80_root_{runtime,source}_review_20261009.json and
vandermonde_full80_source_20261009/{full80_v2,payload_manifest_v2.json}.


## Source-derived opt-in figure layout (2026-10-09)

The single-axes `matlab_axes_layout` helper and
`save_chebfun_figure(..., layout="matlab")` now measure annotations at final
canvas size/DPI and apply normalized outer-position/loose-inset constraints.
No camera, function data, font size or output dimensions are fitted to reference
pixels. Multiple axes/colorbars, legends and active layout engines are rejected.
Nine controls pass, including actual saved label/title bounds and unchanged
sphere geometry/camera. Root independently rehashed2952 observed runtime files;
terminal0, stable inputs and no survivors. The first run's -7.1e-15 pixel
boundary failure is preserved; final checks use a symmetric binary64 roundoff
bound. Existing save behavior remains the default, and no page is yet opted in.

This is a layout policy adapter, not verified historical typography or a
complete MATLAB graphics engine. Surface interpolation is separately in progress;
combined renderer/layout checks and a fresh Atmospheric replay remain required.
Evidence: sphere_layout_root_runtime_review_20261009.json and
sphere_source_layout_policy_20261009/controls_v2 in shared goal scratch.


## Atmospheric full source execution and numeric construction (2026-10-09)

AtmosphericTemperature now executes the original dataset through public
Spherefun construction, mean/slices, Poisson and three Gaussian filters.
The complete fresh CPU run generated all10 figures at600x270 in194.23s,
peak4.27GiB; root rehashed3053 observed files, stable inputs/no survivors.
A prior run crashed after9figures during a periodic traceback dump; matched
replay without that watchdog passed. This supports an observer-race hypothesis,
not a proven causal diagnosis. Six mocked CLI/cache controls also pass; root
rehashed2999 files. Noargs cache/download behavior is retained with immutable
input URL/SHA verification; numerical run/display ASTs are unchanged.

This is a verified source computation replacement, NOT full page parity.
All published MATLAB cells/prose remain, four output cells use actual stdout,
and the missing09 image is restored. Coast/contour painter order, clipped
labels/titles, globe framing and surface striping remain under repair.
Historical MATLAB mean commit eecdfb505040f941b324b6468316088088a88e43
corrected s/2*pi to s/(2*pi), explaining a plausible pi-squared reference04
scale discrepancy; exact historical image-generation revision is unbound.
Poisson historical changes/reference06 remain under investigation. Do not
rescale current correct computations merely to fit historical reference PNGs.
Evidence: atmospheric_page_source_20261009/FULL_PAGE_EXECUTION_HANDOFF.json,
CLI_HANDOFF.json and atmospheric_page_source_root_integration_20261009.json.

Numeric constructor commit bc365398 passes93 composite controls: selected
session technology, full-domain numeric samples, own-grid nonfinite
extrapolation, native unbounded nonzero rejection and realdouble zero
operators (including Trigtech). Existing modifier routes and finite low-level
Tech transforms remain unchanged. Zero-operator preference forwarding and
low-level nonfinite API still have documented gaps. Fullsuite, all322pages,
native execution, publication and exact-head CI remain open.


## Continuous Lebesgue, concatenation and contour follow-through (2026-10-09)

The new public `chebfunjax.lebesgue` constructs a continuous piecewise
Chebfun and optionally returns its continuous infinity norm. It follows
polynomial fixed-length construction/manual simplify and the trigonometric
barycentric branch, including periodic endpoint removal. Supported C1/C2
session preferences reach the actual technology; polynomial construction
retains native fixedLength/sampleTest overrides. All11 literal native test
slots and23 parser/preference/public-entry controls pass together (34 total)
in final_public_v2,180s/3GiB CPU-only, stable inputs and no survivors.
Root independently rehashed2941 loaded files (consult the runtime JSON for
its authoritative count). No original test bounds were widened. Earlier
static/fixture failures and separate successful gates are preserved.

Limits: non-Chebtech session preferences, complex/large-weight edge cases,
and legacy sampled lebesgue_constant/lebesgue_function helpers remain
unqualified or unfinished. The new API does not establish all such parity.
Evidence: lebesgue_source_20261009/final_public_v2 and
lebesgue_final_public_root_runtime_review_20261009.json in shared goal scratch.

Accepted prerequisites: dd091ade native sequential vander/column reversal
(52 controls);98a2b70c source FFT contour sampling/seam closure (6 composite
controls);f3f91c23 public horzcat source return/dispatch (22 focused plus9
native slots);f4ff1ecf C1 tolerance/turbo/check and C2check forwarding
(10 actual-technology/analytic controls). Root runtime/scope/full lint
checks passed for each. Numeric promotion on unbounded/nonfinite inputs and
session technology remains active follow-up work; no full constructor claim.

Full Atmospheric replay on98a2b70c saved9figures, then SIGSEGV during the
last smoothing reconstruction below its6GiB cap. A matched observer-only
replay is testing a periodic traceback watchdog race hypothesis; the cause
is unproven. Independent visual review found contour/coastline/layout and
possible reference-version numerical gaps. Neither full page nor figure
parity is established. Degree16 Vandermonde/Arnoldi checks pass; degree32
and the original degree80 page remain unrun. All322-page/full-suite/native
MATLAB/publication/exact-head CI completion remains open.


## Sphere subtraction accepts public JAX scalar results (2026-10-09)

Preparing the actual AtmosphericTemperature input exposed a dispatch bug:
mean2(f) returns a JAX scalar, but sphere subtraction recognized only NumPy's
isscalar predicate. The native f-g=plus(f,uminus(g)) route was bypassed for
zero-dimensional arrays, triggering adaptive resampling of the whole function.
Real zero-dimensional NumPy/JAX scalars now use the existing addition route.
Complex zero-dimensional arrays and non-scalar arrays retain their prior paths.

The frozen baseline reproduces the wrong route with an explicit resampling
sentinel. The corrected candidate passes6 controls: actual mean2 subtraction
and five scalar representations, sampled analytical values and zero-mean
checking. The source condition is the only production change; numerical tests
retain their original bounds. These controls do not establish arbitrary data
or complex-scalar parity. Both runs are stable/uncensored with no survivors.

Evidence in shared goal scratch: sphere_scalar_minus_20261009/{baseline_v1,
candidate_v1}, sphere_scalar_minus_independent_source_review_20261009.json
and sphere_scalar_minus_root_runtime_review_20261009.json.
Actual Atmospheric data construction and full page verification remain pending.


## Native array-valued QR/SVD returns and scalar-zero branch (2026-10-09)

Smooth/scalar QR now returns an actual Chebfun array, retaining factor panels
and native matrix multiplication semantics. SVD preserves this representation;
backslash, subspace, orth and pinv consumers use explicit column access.
Public Chebfun normest/cond/rank expose the native estimate/tolerance formulas.
The scalar-zero branch uses literal0*f+1/sqrt(diff(domain)), preserving C1/C2
and source piecewise numeric-vector expansion. Orth's broader native return
API and existing noncollatable fallback behavior remain separate gaps.

Forty-nine passing instances across seven bounded serial CPU gates cover48
unique case IDs, including all15 original QR slots and mldivide5/6 at their
unchanged bounds. QR15 is repeated after the separate zero-branch repair.
Other checks include array and complex-row SVD, consumers and legacy adapted
regressions; these do not establish all native SVD/subspace clauses. A first
missing-import failure is preserved; subsequent gates explicitly checked F821.

Twelve final payload hashes match the last frozen snapshot. Migration and
zero-branch patches/evidence are separately attributable. Three example
consumers and one Chebfun2 test have column-access compatibility edits; their
full numerical/page runs remain pending. Degree80 VandermondeArnoldi, full
suite/native execution, publication and exact-head CI remain unresolved.
Earlier degree8 diagnostics apply to their earlier accepted base.

Evidence in shared goal scratch: qr_native_array_source_20261009/
final_delivery_v1/HANDOFF.md, migration_delivery_v1, zero_followup_delivery_v1,
and qr_native_array_root_runtime_review_20261009.json.


## Committed sphere backend regression coverage (2026-10-09)

Five reusable test files retain the previously qualified numerical controls,
now importing the committed library:23 pivoted band-LU checks,12 Fourier
assembly/convolution checks,6 nonzero-mode composition checks and14 bordered-QR
checks. All55 pass in five fresh serial CPU processes, each120s/2GiB; original
bounds retained. These are backend regression controls, not55 additional native
MATLAB test slots. Public/native sphere assertions remain documented below.

Root verified numerical test ASTs unchanged after package-import adaptation
and one artifact destination change to pytest tmp_path. Stale prototype-only
helper descriptions were corrected with no numerical AST change. Full Ruff,
F821, provenance/NumPy policy and whitespace checks pass.

Evidence in shared goal scratch: sphere_helper_regressions_20261009/
(root_scope_review.json and five gate directories), sphere_helpers_first_root_
runtime_review_20261009.json, sphere_helpers_remaining_root_runtime_review_
20261009.json and sphere_helper_documentation_scope_20261009.json.
Large-grid scaling, actual AtmosphericTemperature inputs/reconstruction/page,
full-suite qualification, native MATLAB execution, publication and CI remain open.


## JAX sphere Fourier band solvers (2026-10-09)

Poisson and Helmholtz now use compact JAX pivoted band LU for nonzero longitude
modes and compact bordered QR for the integral-constrained zero mode. The
source's actual Fourier multiplier coefficients, including tiny odd bands,
are retained. No dense PDE factorization or principal-minor inversion is used.
The public callers preserve explicit odd Poisson sizes, source grids, original
forcing integral, complex Helmholtz parameters and native eigenvalue checks.
Gaussian filtering now passes longitude/latitude lengths in native order.

Original-source CPU predicates pass:40 Helmholtz harmonic checks across all
four original size cases,6 Poisson assertions, and6 Gaussian-filter assertions
in3 tests. The latter test port restores full function norms and the original
spherical-harmonic construction. Fourteen independent public controls also
pass. Root/independent runtime audits found no hash/origin faults, changed
inputs, censored gates or surviving processes. Helper evidence separately
covers23 band-LU,12 assembly,6 composition and14 bordered-QR checks.

An initial near-eigenvalue analytic control failed: the actual source-rounded
border has condition number6.95e13. Independent dense and compact solutions
agree to1.11e-16 but differ from the ideal harmonic by2.003e-5. The preserved
diagnostic and final controls compare the complete discrete equations at the
original bounds; a well-conditioned case additionally checks the analytic
harmonic. No tolerance widening or tiny-coefficient deletion was used.

Limits: no fresh MATLAB execution, release-specific linspace bit identity,
large-grid scaling, AtmosphericTemperature replay or performance claim. General
latitude sizes below3 and native singular-system warning/exception identity
remain unqualified. Existing native real projection in coeffs2spherefun is
preserved. These scoped source predicates do not establish all sphere parity.

Evidence in shared goal scratch: sphere_public_source_20261009/
(native_helmholtz_qualified_packet.json, caller_qualified_packet.json,
poisson_native_v1, gaussfilt_native_v1), sphere_helmholtz_root_runtime_review_
20261009.json, sphere_gaussfilt_root_runtime_review_20261009.json and
sphere_public_independent_review_20261009.json. Publication and CI remain open.


## JAX linear roots and literal boundary rejection (2026-10-09)

The n=2 roots branch now computes, filters and clips with JAX. Native linear
rejection retains equality at the imaginary threshold and expanded domain
bounds; the distinct strict eigenvalue-leaf predicate remains unchanged.
A writable host return adapts to the inherited recursive engine.

The focused CPU gate passed48 checks (24 direct boundaries, scaled/public
and array dispatch, solver selection and8 dtype/ownership checks). An exact
Fraction diagnostic on represented normalized coefficients independently
explains21 imaginary-component one-ULP changes: JAX rounded all21 correctly,
including4 changed acceptances whose exact imaginary magnitudes exceed the
threshold. This is not a general correctly-rounded complex-division claim:
only18/21 real components were correctly rounded. Native MATLAB division
and observed acceptance remain unmeasured; the exact-input MATLAB probe is
prepared but unexecuted. No tolerances were widened.

Root independently verified2928/2924 observed runtime files for the passing
boundary and diagnostic gates, exact rational results and payload hashes.
Only the n=2 production branch changed. Host trimming, eig/QZ migration,
full regressions, native execution, publication and CI remain open.
Evidence in shared goal scratch: roots_linear_boundary_source_20261009/
qualified_caveat_packet_v3.json, roots_linear_root_runtime_review_20261009.json
and roots_linear_fraction_root_review_20261009.json.


## Polynomial FUN QR and column backslash (2026-10-09)

Column Chebfun/Quasimatrix backslash now uses continuous QR and inner products,
then a JAX triangular-system solve. Smooth polynomial columns collate onto
unified domains and delegate to technology QR with physical-interval scaling;
piecewise panels use stacked-factor QR. First C1/C2 technology is preserved.
Public Chebfun.qr accepts array-valued inputs through column dispatch.

Seven passing test instances cover seven native source slots (mldivide5/6;
QR9/10/13/14/15) plus two physical-domain controls. Root independently verified
2971/2959/2971 loaded files across three gates; all terminal0, stable, uncensored,
no survivors. Original input/tolerance predicates retained. A first interval-
adapter error and an invalid last-column normest adapter are preserved; the
final gate uses native array-valued all-column normest semantics.

The Python Q return remains Quasimatrix; native smooth Q is array-valued
Chebfun. Return type/shape/normest and numeric-multiplication order remain an
explicit API/behavior gap, not covered by these factorization checks. Row/
singular backslash, other original QR slots, degree80, the full Vandermonde
page, broad regressions and performance remain unqualified.

Evidence: vandermonde_qr_source_20261009/delivery_v1,
vandermonde_qr_root_runtime_review_20261009.json and
vandermonde_qr_root_integration_20261009.json in shared goal scratch.
Full suite, all-page parity, native execution, publication and CI remain open.


## Resampling histogram page replay (2026-10-09)

The full ResamplingRandomVariables script now uses the accepted source JAX
histogram helper and native-default black0.5pt bar outlines. A complete CPU
replay on roots-vscale commit2e572ea6 passed in86.48s test time, with stable
inputs, no survivors and peak2.80GiB. Root independently rehashed2971 observed
runtime files, checked all8 delivered payload hashes and compared19 nonimage
captures byte-for-byte with the earlier run. Actual stdout is unchanged:
12 inverse breakpoints and missing2.449485059230483e-10; no reference injection.

Both10000-sample histograms match independent R2017a source counts and all
center words. All6 figures are600x270;9 MATLAB cells and original prose are
preserved. All6 reference pairs were reviewed, including new03/06 outlines.
Native RNG, historical2014 runtime, precise historical numerical output,
fonts/titles/legends/axes/ticks and exact bar geometry/pixels remain open.
This qualifies a source computation replacement, not full historical parity.

Evidence: resampling_hist_source_20261009/page_delivery_v2,
sphere_border_resampling_v2_root_review_20261009.json and
resampling_page_v2_root_integration_20261009.json in shared goal scratch.
Seven of twelve interrupted scripts now have reviewed source replacements;
original home-worktree edits remain preserved. Full suite, push and CI open.


## Original-grid roots normalization (2026-10-09)

Public Chebtech1/2 roots now divide by native vscale from the original
representation grid, separately for each array column. Constant shortcuts
precede scale evaluation. No denominator floor or fallback is introduced.
The coefficient-only private resultant caller retains its previous policy;
inherited host trimming/eig/QZ and other roots options remain open gaps.

145 acceptance checks passed across11 gates, plus2 explicitly diagnostic
nonfinite cases. These cover original-grid/constant/column behavior, source
roots, count boundaries through4001, mapped roots and inverse behavior.
Root independently rehashed each gate's observed runtime files; all final
gates terminal0, stable, uncensored, no survivors. Native tolerances unchanged.

Four matched CPU measurement arms were run serially while other numerical
workers paused. First calls AFTER setup, inverse baseline/candidate:
16.8709/16.6170s; derivative roots8.9189/9.0256s. Five-call warm medians:
inverse0.727106/0.701232s; derivative roots0.130914/0.124562s. Samples overlap;
these observations do not establish a speed gain. Setup had already compiled
shared work; these are neither pristine cold timings nor MATLAB comparisons.
Both inverse roundtrips retain1.0547118733938987e-15 error (<1e-10 bound).
Root independently verified2958 runtime files for each measurement arm.
A separately bound posttiming comparison confirms62 inverse and8 derivative
pipeline capture arrays are byte-identical across arms; root also compared
these raw bytes independently. Normalized engine inputs differ intentionally
under the corrected scale, changing some internal trimming counts without
changing these measured outputs. The unchanged native correctness bounds,
not cross-arm identity, remain the acceptance criteria.

Evidence: roots_vscale_source_20261009/correctness_packet_v1.json,
roots_vscale_full_root_review_20261009.json and
roots_vscale_measurement_root_review_20261009.json in shared goal scratch.
Full-suite regression, all-page parity, native execution, push and exact-head
CI remain unresolved.


## Native anonymous multiplication assignment (2026-10-09)

Chebop.nativeAnonymous explicitly preserves native * versus .* syntax and captured bindings. Op/BC assignment compiles tags using the instance vectorize flag, initialized from current ChebopPref. Ordinary Python callbacks and legacy constructor strings retain their algorithms. Actual AD mtimes now exposes its literal identifier through a compatible ValueError subclass.

Original autoVectorize clause6 passes through public solve with the original expression, BCs, domain and expected identifier. Composite34 passing instances qualify29 unique checks:22 historical ordinary passes,3 focused corrections,9 final preference/constructor checks. The first gate's three failures are retained: one test domain mismatch and two overly small resource guards; no tolerance or solver defaults changed. Final negative solve allows only initial descriptor32 and forbids adaptive correction assembly. All final gates stable/no survivors; runtime file hashes verified.

Limited multiplication grammar only; not general MATLAB parsing, arbitrary Python rewriting or AD array expansion. Earlier source1–5 remain their existing execution-only qualification and were not rerun. Full suite/native execution/pages/performance/publication/exact-head CI remain open. Evidence: autovectorize_native6_source_20261009/qualified_packet.json.

Root independently reviewed source/dispatch/error scope and rehashed2972,2972
and2971 observed runtime files for the three composite gates. Earlier raw
failures remain preserved. See native_anonymous_root_review_20261009.json
and native_anonymous_root_integration_20261009.json in shared goal scratch.

## Source histogram edges and CPU division (2026-10-09)

A JAX helper now implements the R2017a scalar-bin hist source for nonempty
finite real vectors. It retains37 edges for36 bins, half-bin centers and
edge+eps(edge) classification, including negative powers of two, signed zero
and already represented subnormal data. Other MATLAB hist overloads remain
unimplemented; R2013a arithmetic differs, and the2014 page runtime is unproven.

Qualification:39 strict controls, including all23 unchanged original controls
and16 division/shape/overflow checks. Root rehashed2949 observed runtime files;
terminal0, stable inputs, uncensored, no survivors. Original failed strict
asymmetric case is retained. Compiler IR proves XLA changed vector division
into reciprocal multiplication:8 divisions moved by1ulp, changing14 probe bins.
Strict CPU compiler options did not fix it. A full broadcast-operand barrier
preserves vector division with default options; all diagnostic stage words and
counts match the independent source-order oracle. Root reviewed both IR arms
and independently rehashed2938 files per diagnostic. No tolerances were widened.

This barrier is private to the new histogram helper, with no global compiler
flags. Extreme subnormal grid arithmetic, whole-function AD/JIT, and asymmetric
runs with global JIT disabled remain unqualified. The Resampling example has
NOT yet been switched: a full page run is still required, and native RNG and
historical plot/output equivalence remain open.

Evidence in shared goal scratch:resampling_hist_source_20261009/library_delivery_v1,
vscale_histogram_root_review_20261009.json, histogram_controls_root_review_20261009.json
and histogram_root_integration_20261009.json. Full suite, publication and CI remain open.

## 2026-10-09: periodic promoted functional source7–10

Periodic `u'' + sum(u)` now follows native trigcolloc values or trigspec coefficient assembly, with exact AD blocks, full values-stack conversion, source dimension schedules, repeated toFunctionOut conversions and strict continuous imaginary-part projection. Narrow proxy `sum` markers retain existing differential callback and nonlinear routes. Existing first-kind capability arithmetic is unchanged.

Evidence: `periodic_nonlocal_source_20261009/qualified_packet.json`. Four original slots7–10 pass in two backend cases at native initial dimension32; continuous residual <1e-10 and actual Trigtech class, plus independent analytic/mean checks. Composite31 passing test instances =29 ordinary controls +2 source backend cases. Final source snapshots use accepted roots base11682da24 plus only six owned files; historical ordinary base9c8d1a07 applies by scoped proof.

Preserve limitations: pinned odd-size trigspec conversion is C*M*V.T, not a Fourier similarity; literal source quirk retained and independently tested. Original failed ordinary gate retained (six member-lookup errors fixed, one invalid odd-diagonal hypothesis corrected), with22 prior passes plus7 focused passes. The bounded n=8 initial-scale control emits native noConverge and does not qualify adaptation. No native MATLAB capture (startup IPC restriction). Other periodic nonlocal primitives/systems and legacy NumPy/nonlinear algorithms are outside this package.

Root independently rehashed2972 loaded files for each source backend, and
2968/2970 for the ordinary composite gates. All final source gates are
terminal0, stable, uncensored and have no survivors. Full-suite regression,
publication and exact-head CI remain unresolved. Root evidence:
periodic_root_source_review_20261009.json, periodic_values_root_review_20261009.json,
periodic_coeffs_root_review_20261009.json and periodic_root_integration_20261009.json.

## TrapezoidEigs source computation restored (2026-10-09)

The full example now uses public JAX Bessel/SVD boundary matrices, source
n=4..7 and domain[3,7], adaptive splitting, and public local minima with the
literal x>3 filter. Private numerical surrogates/prominence selection are gone.
A complete standalone CPU run on11682da2 passed all12 displayed labels and
all5 reference-sized600x270 figures with72.009dpi metadata. It finished with
stable inputs, no survivors, and peak sampled RSS5.10GiB; root independently
rehashed2978 observed runtime files. Actual stdout is empty. Construction
titles report actual instrumented timings (observer I/O included), not native
MATLAB timing or an isolated benchmark.

The generated page preserves all5 cached MATLAB cells and source prose.
Delivered code differs from the executed script only by import ordering and
an explicit source-default black polygon edge, redrawn from the saved figure;
root AST and output-hash checks establish applicability. Numerical labels agree
to the historical five displayed decimals, not an unrounded MATLAB capture.
Native plot sampling counts are matched; automatic axes/ticks, fonts, title
weight, grid styling and exact pixels remain open. This is a verified source
computation replacement, not a claim of complete page visual parity.

Evidence in shared goal scratch:trapezoid_page_source_20261008/delivery_v2,
trapezoid_standalone_root_review_20261009.json and
trapezoid_root_integration_20261009.json. Full suite, publication and CI remain open.


## Native roots subdivision regimes and measured CPU tradeoff (2026-10-09)

Roots now use the pinned coefficient-count cutoff50 (including the formerly
incorrect51-coefficient boundary), cached513-point triangular transforms
through count513, paired Clenshaw sampling through4000, and public source-policy
NDCT above4000. New subdivision arithmetic/transforms are JAX. Fixed transforms
are cached as concrete arrays by device/x64 configuration; no traced inputs are
cached. Native affine operation order and NDCT pure-imaginary behavior remain.
Inherited host trimming/eig/QZ and coefficient-max rather than native-vscale
normalization remain explicit gaps; this is not full roots/JAX parity.

Qualification:124 scoped checks across ten frozen gates, including37 helper
checks,28 technology-root checks (four restore literal native output order),
six actual boundary/trim checks,14 mapped-root checks, and39 inverse checks.
All are terminal0/stable/no survivors. Root independently verified observed
runtime hashes and unchanged acceptance-test ASTs after diagnostic-only code
extraction. Original numerical root/inverse bounds are unchanged. Earlier exact
matrix-builder comparison failed by1.11e-16 across compiler contexts; that raw
failure is preserved, with a separately reviewed helper-only finite-arithmetic
bound. The precise compiler cause is unproven; no native bitwise matrix claim.

Four separate CPU measurement processes used the same affinity and setup,
with all other numerical workers paused. Both arms were profiled, with an extra
subdivision observer on the candidate; five warm calls per arm are observations,
not a machine-exclusive or fresh MATLAB comparison. All compared raw forward,
derivative, target, Brent, domain and result arrays match bitwise. Inverse
roundtrip error is1.054711873e-15 for both. First means first call AFTER identical
flower/arc-length/derivative setup; the fixed matrix cache was built in setup.

| Operation | First after setup, baseline → candidate | Warm median, baseline → candidate |
| --- | --- | --- |
| Inverse | 12.205127 s → 15.996914 s | 0.822081 s → 0.720799 s |
| Derivative roots | 3.658862 s → 8.336681 s | 0.251248 s → 0.153377 s |

First-call compilation costs increased (inverse212→292 and derivative76→156
profiled backend compile calls); warm calls compiled zero times. Warm inverse
latency is about12.3% lower and derivative roots about39.0% lower in these runs,
but first calls are slower. The large MATLAB inverse performance gap remains.
The failed v2 measurement capture (tuple metadata handling) is preserved; v3
changes only capture conversion outside the timed section.

Evidence:roots_correctness_root_acceptance_20261009.json,
roots_measurements_root_review_20261009.json and
roots_subdivision_source_20261008/measurement_comparison_v3.json in shared
scratch. Frozen evidence predates unrelated operator changes; publication tech
matches the implementation base and intervening changes are disjoint. Full-suite
regression, all pages, native execution, publication and exact-head CI remain open.


## Coupled nonlocal Newton and second-kind adaptive assembly (2026-10-09)

Original promote_functional slots3–6 now use exact coupled AD, native zero-state
public linearization, source boundary fitting and error-controlled Newton.
Both C2 and C1 solve the two original systems at the unchanged continuous
infinity-norm bound1e-10; independent analytic solution and absolute boundary
checks pass. The second system starts without an inherited explicit init.
New C2 assembly uses native second-kind function points and first-kind equation
points through the shared adaptive solve. Scalar algorithms remain unchanged.
Numeric-zero AD multiplication now preserves native zero/order metadata while
zero Chebfun multiplication retains its distinct structural behavior.

Composite qualification has54 checks:42 ordinary/source-matrix/AD checks,
8 routing/applicability checks and4 original coupled solves across C2/C1.
Six historical routing comparisons remain scratch-only evidence; permanent
routing tests have no scratch environment dependency. Known scalar parameter
columns delegate to existing routes; alternate routing has code-scope proof,
not a claimed runtime parameter test.

One C2 receipt retains a root_exited_with_live_descendants label. Root review
found only the root process identity, no signals or observed descendants,
terminal exit0, stable inputs and no survivors. Supervisor discovery-before-poll
explains a stale root-only exit sample. The numerical result is accepted with
this explicit caveat; the raw receipt remains unchanged. Other gates are
uncensored. Final parameter guards are qualified by8 focused checks and code
scope proofs; historical C2 runs bind the earlier guarded-route implementation.
This is composite evidence, not a full-suite run on the integrated tree.

Evidence:root_coupled_roots_receipt_review_20261009.json,
coupled_c1_root_receipt_review_20261009.json,
coupled_root_final_algorithm_scope_20261009.json and the source mapping,
algorithm review, routing extraction and frozen gates under
coupled_nonlocal_source_20261008 in shared scratch. Original periodic slots7–10,
full regression/pages/native execution/performance/push/exact-head CI remain open.


## Native C1 nonlocal equations and AD scalar matrix multiplication (2026-10-08)

The scalar finite-interval nonlocal route now realizes the original operator
stack on genuine first-kind nodes, preserving intermediate multiplication,
complex values, coefficient breakpoints and global cross-interval functionals.
Existing differential assembly and the adaptive solver loop are unchanged.
Unsupported ultraS nonlocal equations retain the native rejection.

AD matrix multiplication now accepts numeric scalars, including size-one JAX
arrays, and preserves coefficient, coordinate and native-values metadata.
Chebfun/AD operand precedence follows the native AD class. Functional matrix
multiplication raises the native dimension error. Numeric AD array expansion
and native anonymous-function vectorization remain unfinished.

Qualification is composite: 27 AD controls, nine nonlocal controls, four
instances of original promote_functional slots 1/2 (C2 and C1), and 20 endpoint
controls. The original continuous infinity-norm bound remains 1e-10. Root
independently rehashed every observed runtime file in all seven gates. All
receipts are terminal zero with stable inputs and no surviving processes.

Root review caught and corrected interval-direction corner cases before
integration. Historical gates bind the earlier helper; AST review proves only
FirstKindDisc.evaluation changed, the scalar source solves use unchanged paths,
and the final endpoint controls cover the corrected branches. This is not a
single full-suite run on the final tree. Source slots 3-10, periodic/coupled
nonlocal workflows, full page parity, native execution and final CI remain open.

Evidence in shared scratch: nonlocal_root_runtime_review_20261008.json,
nonlocal_root_endpoint_runtime_20261008.json, nonlocal_root_scope_review_20261008.json,
nonlocal_root_assembler_scope_20261008.json, nonlocal_root_endpoint_scope_20261008.json,
and ad_mtimes_root_review_v2_20261008.json. Atomic payload comes from
nonlocal_c1_source_20261008 and autovectorize_mtimes_source_20261008/delivery_v3.

## Interrupted-script disposition refresh and inverse experiment (2026-10-08)

A fresh hash inventory at4d5ff9b7 confirms five interrupted scripts have
separately qualified source replacements in shared main: BestApprox, FermiDirac,
Localization, FourierBasedChebfuns and MarchingSquares. Current payloads match
their accepted records/commits. Their documented numerical/RNG/raster gaps
remain; this is not full page parity. All12 original home-worktree files remain
unchanged and none is byte-identical to the current publication script.

Seven entries still need current source/page disposition review: VandermondeArnoldi,
DelayDifferentialEquations, RandomSwitching, TrapezoidEigs, AtmosphericTemperature,
LaplaceBall and ResamplingRandomVariables. Existing partial evidence is retained.
Do not repeat the stale blanket count of11 unverified edits as a current audit.
Evidence:handoff_twelve_current_disposition_20261008.json in shared scratch.
Next root page package:trapezoid_source_20261008/PLAN.md; restore public local
minima, original boundary matrices and JAX Bessel/SVD, with bounded kernel
qualification before adaptive whole-page execution. No run is claimed yet.

The single256-point inverse evaluator-blocking experiment is rejected: matched
fresh-process warm medians0.704283s baseline versus0.717561s blocked (1.01885x);
cold9.60261s versus9.60349s. Nine controls and both actual-flower arms pass;
all captured inputs, Brent outputs and inverse coefficients match bitwise.
Root rehashed2933/2955/2955 runtime files and packet artifacts; all three
receipts terminal0/stable/no survivors. No production change or speed gain.
Evidence:inverse_point_block_root_review_20261008.json and
inverse_point_block_20261008/qualified_packet.json.

Active correctness work: additive native C1 nonlocal equation realization and
AD scalar-multiplication metadata preservation require coordinated qualification
and integration. Neither is accepted yet. A read-only roots source audit found
count-threshold/subdivision branch differences; its proposal must precede further
evaluator optimization. CPU only; full tests/pages/native captures/push/CI open.

## Restored autoVectorize execution setups (2026-10-08)

Canonical tests now reproduce original setups1–5, including late operator
assignment, coefficient closures, general BCs and exact coupled initial values.
All five original completion-without-exception predicates pass. Two independent
assignment/action controls pass; no constructor or assignment defect was found.
Prior four endpoint controls remain separately named legacy tests and were not
rerun in this package. No library files changed.

These are execution checks, not native vectorization or algorithm parity. Python
pointwise operators spell the already-vectorized expressions; original clause6
stays pending. Cases1/2 run SciPy LSODA instead of native ode113. Coupled5 retains
the existing fixed48 solver. Source3/4 use the accepted scalar Newton path.

Root rehashed1789 imported files for assignment/1/2/5 and1797 for3/4, source
hashes and delivered tests; six receipts terminal0/stable/no survivors. The
source/test/runner guards are prebound, but1716–1724 imported dependencies per
gate were only hashed after execution, not prebound. This is not a fully bound
runtime-environment qualification. Evidence:autovectorize_root_review_20261008.json
and autovectorize_scalar_source_20261008/HANDOFF_FINAL_v1.json in shared scratch.
Full native callable rewriting, solver algorithms, all tests/pages and CI remain open.

## Explicit contour level counts (2026-10-08)

Scalar contour counts now use the R2017a contourobjHelper source rule: exactly
N interior evenly spaced levels, midpoint for N=1, truncation toward zero,
finite-data extrema and minimum boundary for filled counts. Explicit vectors
and repeated single-level vectors keep their value semantics. This replaces
Matplotlib's different integer tick-locator behavior; sampling is unchanged.

All23 checks pass (13 focused controls plus10 existing sampling/plot checks).
Root rehashed2948 observed files and source hashes; terminal0/stable/no
survivors. Only contour changes in plotting.py. Evidence:
contour_levels_root_review_20261008.json and contour_levels_source_20261008/controls_v2
in shared scratch.

Automatic level selection is NOT resolved: the existing default12 still
substitutes an explicit count for MATLAB auto. R2013a HG1 automatic source
is readable, but equivalence with later HG2 p-code is not established.
Constant/all-nonfinite data keep the inherited renderer path. Gibbs automatic
contours, exact image parity, native execution and exact-head CI remain open.

## Scalar linear alternative backends (2026-10-08)

Scalar linear C1/ultraS solves now use exact AD differential/constraint blocks
and adaptive requested-backend correction, including supplied initial guesses,
coefficient breakpoints, interior evaluations and integral boundary conditions.
The default-solver seed, fixed65 reconstruction and finite-difference probes
are removed from this route. Shared matrix assembly and scalar Newton are unchanged.

All56 instances pass compositionally:44 source instances covering28 original
slots/54 unique scalar comparisons, plus12 controls. Canonical linearScalarODEs
and linearInit tests restore source continuous norms, original tolerance factors,
piecewise cases and nonidentity checks. Native signed/repeated-variable quirks
are retained with independent absolute-boundary controls. Linear bc1–8 pass
under all three backends. Max observed RSS3.33GiB; no guard activation.

Root rehashed nine gates (2963–2969 observed files each), checked source hashes,
payloads and AST scope/equivalence. Initial controls_v1 preserves two wrong
exception-class expectations; its six passing controls and focused corrected
two-check rerun qualify identical numerical code. Every receipt is stable with
no survivors. Evidence:scalar_linear_root_review_20261008.json and
scalar_linear_altdisc_source_20261008/qualified_packet.json in shared scratch.

Nonlocal equation realization, periodic and explicit scalar parameter routes
remain unsupported here. This does not qualify a full exact-head suite, fresh
MATLAB execution, all pages or CI.

## Historical matrix spy artist behavior (2026-10-08)

Matrix spy now follows the installed R2016b/R2017a source: one-based
column-major nonzero coordinates, native limits/orientation and nz label,
physical-point automatic marker sizing and clamps, and axes-first color.
Historical axes Position and one-third point diameter follow installed
R2017a/R2013a documentation. Linop discretization is unchanged.

Thirteen checks pass: eleven independent artist controls, one existing test
and a scratch Gibbs source-matrix render. Root rehashed2946 observed runtime
files; terminal0/stable/no survivors. Only spy changes in plotting.py.
Evidence:spy_root_review_v5_20261008.json and spy_source_20261008/controls_v5
in shared scratch. controls_v4 ran zero tests due to an incorrect selector;
its terminal failure is preserved. Axes color inspection uses Matplotlib3.10.8
private cycle representation and has an explicit custom-cycle control.

This does not establish cached-image parity: Gibbs spy still differs in ticks,
plot box, dot rasterization and label layout. Full Gibbs camera/lighting/contour
work, native execution, full-head tests and exact-head CI remain open.

## NonsmoothFOV complex-pole diagnosis (2026-10-08)

No extraction defect is demonstrated on the captured current data. All59
poles and upstream polynomial/trig coefficient arrays are byte-identical to
the earlier accepted Python page artifacts. Maximum normalized denominator
residual2.056e-15 and pencil residual1.720e-17; known complex-pole controls
are consistent with conditioning. A small Loewner singular gap indicates
sensitivity but does not establish the historical discrepancy's cause.

Root independently rehashed2935/2923 runtime files plus packet artifacts,
confirmed terminal stable receipts/no survivors. Evidence:aaa_fov_root_review_20261008.json
and aaa_fov_pole_audit_20261008/qualified_packet.json. No production numerical
change. Native matched F/Z/support/weight data remain unavailable; the native
capture script is explicitly UNRUN. Historical page comparison remains open.

## Literal Carrier solver qualification (2026-10-08)

All six original Carrier predicates now pass with source inputs, adaptive
solves and exact backend-specific preferences. C1/C2 restore bvpTol1e-10 and
bounds1e-10; ultraS retains factory5e-13 and bounds5e-11. Fixed256/384 alternate
resolution substitutions are removed from canonical tests. Prior variants remain
explicit legacy controls. This package changes seven test files only.

C2 errors1.699e-11/1.167e-11, C1 errors1.699e-11/1.166e-11 and ultraS errors
5.463e-13/1.512e-13 satisfy both original comparisons per backend. Solution
lengths157/157/185; updates9/9/10; maximum matrix width258. Resource guards never
activated. Root rehashed1804/1805/1805 observed runtime files and pinned source
hashes, verified exact payload and terminal stable receipts/no survivors.
Evidence:carrier_root_review_20261008.json and
 carrier_scalar_source_20261008/HANDOFF_FINAL_v1.json in shared scratch.

Qualification used4e431f08 plus the exact generalBC overlay; newer157d source
differs only in AAA. No full-head suite, fresh native execution or isolated
performance claim. Full autoVectorize/operator API, all pages and CI remain open.

## AAA explicit-sample string expressions (2026-10-08)

AAA now accepts MATLAB expression strings with explicit sample coordinates,
reusing the existing single-variable expression parser. Sampling, derivatives,
SVD/Lawson/cleanup and pole kernels are unchanged. Omitted-sample strings remain
rejected; native parseInputs establishes conversion only in its explicit-Z arm.

Sixteen final checks pass: eight independent expression controls plus all eight
original derivative predicates33-40. Controls include the original16 sample
count10001 with deterministic queries, exact scalar/vector outputs and all six
metadata outputs, alternate variable names, complex/nondefault domains and
MATLAB elementwise syntax. Original16's exact random query remains unavailable;
the API gap is repaired but its native random input is not claimed captured.
Root rehashed2931 observed runtime files and verified terminal stable receipt/no
survivors. Evidence:aaa_string_root_review_20261008.json and
 aaa_string_source_20261008/qualified_packet.json in shared scratch.
The shared parser's existing syntax limits and earlier AAA endpoint cancellation
and complex-pole questions remain. Full suite/native reruns/exact-head CI open.

## Scalar general boundary conditions (2026-10-08)

Finite single-interval scalar nonlinear general BCs now use the source scalar
Newton adapter and preserve public tolerance/resolution preferences. Scalar
linearity classification includes AD boundary-condition flags at the explicit
initial guess or source zero. This repairs two independently reproduced
misroutes without changing AD, shared matrix assembly or parameter Newton.

Literalbc9-10 and exactInitial1-4 all pass unchanged, with five independent
controls. The canonical exactInitial test restores general BCs, continuous norm,
adaptive alternate backends and the source restart sequence; prior sampled/fixed
grid controls remain explicitly named legacy controls. Root rehashed1795 files
per control run and1805 for the source gate, checked source hashes and the exact
two-method scope. Accepted runs are terminal/stable/no survivors. Evidence:
 scalar_general_bc_root_review_20261008.json and
 scalar_general_bc_source_20261008/HANDOFF_FINAL_v4.json in shared scratch.

Unresolved diagnostic retained: u''=0 with nonlinear BC[u(-1)^2-1,u(1)-1] and
explicitinit2 stalls with damping warning and distance norm0.0204124. Adding a
BC-satisfying nonconstant-init control does not resolve that failure. Its runnable
reproducer and unrun native MATLAB probe are in the package diagnostics folder;
frozen-constraint recurrence is a hypothesis, not a verified native trajectory.
Periodic/piecewise/coupled/parameter cases, Carrier restoration, full-suite and
exact-head CI remain open. Timings are qualification observations, not benchmarks.

## Adaptive alternative ODE discretizations (2026-10-08)

Coupled linear Chebcolloc1/ultraS solves now use exact AD coefficient/functional
blocks, native input dimension offsets, projection, backend factor policies and
adaptive per-interval happiness checks. Only unhappy intervals advance. Native
C1 output construction is included before convergence testing and final output.
Scalar nonlinear alternate backends now use those genuine corrections through
the accepted scalar Newton engine; neither uses a default-backend seed.

Qualification:60 linear checks (all33 original System1/2 slots,49 scalar
comparisons,27 independent controls), four original scalar alternate clauses,
and eight scalar representation-equivalence checks. All original bounds remain.
This is composite qualification with source/AST equivalence for final guard/domain
edits, not one exact-head full-suite run. Root rehashed all ten linear gates,
both1805-file scalar gates and the1664-file equivalence gate; every accepted
receipt is terminal/stable with no survivors. Evidence:linear_adaptive_root_rehash_20261008.json,
 scalar_alternate_root_review_20261008.json and scalar_helper_equivalence_root_review_20261008.json.

Review found and fixed coefficient-only breakpoints missing from discretization
domains. An initial suspicion of raw periodic coefficients was disproved:
existing restriction already converts periodic data as native restrict does.
Four incorrect rejection expectations are retained as failed diagnostic evidence;
positive conversion/action tests pass. Singfun data remaining after restriction
are explicitly unsupported. Scalar-linear legacy paths, general/coupled nonlinear
constraints, scalar parameters and eigenproblems retain documented gaps. Native
reruns, full-suite coverage, JAX migration and exact-head CI remain open.

## Remaining deterministic AAA source predicates (2026-10-08)

A literal companion restores original14,17,18,21-23,25-32 with native autoZ,
full complex residuals, original norms/options and unchanged bounds. All14
pass across three isolated CPU gates (12+1+1); no library edit was needed.
Original17 uses the already captured native real-gamma pole convention; finite
values use JAX gamma. Original31 uses public mapped chebpts; node arithmetic
is not claimed bitwise native. No fixed-grid substitution for original21/22/32.

Together with accepted earlier and derivative packages,40 of42 original AAA
predicates are now qualified. Original15/16 native random inputs remain pending;
16 also needs the string-expression API. This is composite coverage, not a single
42-case run, and does not establish general complex-pole or endpoint accuracy.
Evidence:aaa_remaining_literal_20261008/qualified_packet.json and
 aaa_remaining_root_rehash_20261008.json in shared scratch. All three receipts
are terminal/stable/no survivors; root independently checks2933 observed files
per gate. Full suite, native reruns and exact-head CI remain open.








## AAA derivative API and source recurrences (2026-10-08)

AAA deriv_deg now returns a list of derivative callables as its first output,
retaining the other six outputs. JAX Schneider-Werner recurrences follow native
ordinary/support-node formulas, nonconjugate products and sequential denominator
accumulation. Autosampling, greedy/SVD, cleanup, Lawson and pole kernels are
unchanged. Constant derivatives retain native scalar zero. Numeric singleton
arrays now follow MATLAB isscalar semantics; root caught and corrected an
initial ndim-only parser. Boolean/nonnumeric/multielement options remain ignored.

All eight original derivative predicates33-40 pass unchanged, including both
1e-112 bounds. Final parser/derivative gate passes28 checks. Composite coverage
is113 acceptance bodies, including85 prior nonderivative checks carried by
unchanged function AST and default-zero parsing; not one final113-test run.
Root rehashed2938/2999/2926 observed files across three accepted gates, verified
terminal stable receipts/no survivors and native source/payload/scope hashes.
Evidence:aaa_derivatives_root_acceptance_20261008.json and
 aaa_derivatives_scalar_correction_20261008/qualified_packet.json in shared scratch.

Known limitation: a new analytical autoZ-exp endpoint bound1e-11 failed and is
retained as unaccepted diagnostic evidence. At+1 the error is7.66e-8 near an
inset support node; identical-data literal scalar/vector/JIT recurrences agree.
This is cancellation evidence on those data, not fresh native MATLAB behavior.
No snapping, sample changes or bound relaxation were introduced. Native complex
order-option behavior, full AAA42/API and exact-head CI remain unqualified.

## Orthographic surface viewport (2026-10-08)

New surf axes use an orthographic rectangular viewport instead of Matplotlib's
forced physical square and perspective projection. This preserves sampled data
and Matplotlib depth ordering. The native two-angle viewmtx source is explicitly
orthographic. Other3D renderers retain their existing policies; supplied axes
retain their aspect handler. Exact native plot-box aspect, zoom, lighting and
rasterization remain open, so this is not a complete rendering parity claim.

Ten focused controls and eight existing plotting regressions pass (18 total),
including wide/tall viewport and homogeneous projection checks. Root rehashed
2970 observed bound runtime files and checked terminal stable receipt/no
survivors. Evidence:gibbs2d_source_20261008/frame_root_review_v1.json.
Gibbs2D scientific replay completed all extrema and eight figures, but its page
is not integrated: native lighting, contour levels, spy styling and final camera
matching remain under review in NEXT_RENDERING.md. No size-only acceptance.

## Mixed Chebyshev technology arithmetic (2026-10-08)

Chebtech1 and Chebtech2 addition now accept either polynomial technology,
matching their common native chebtech coefficient representation. The change
is exactly two type predicates; addition formulas, left output class, happiness,
stored point values and each original technology's transform of prolonged
coefficients are unchanged. Subtraction uses the existing addition path.
This fixes the type error when genuine Chebcolloc1 corrections meet a
Chebtech2 initial iterate or analytic reference; no solver-local resampling.

Seven independent controls pass, including both operand orders, subtraction,
unequal lengths and cancellation using different grid scales. Root verified
exact scope, payload identity,1766 observed runtime hashes and terminal stable
receipt/no survivors (4.90s pytest,757748KiB peak). Full suite and exact native
last-bit identity remain unclaimed. Evidence:mixed_chebtech_root_acceptance_20261008.json
and scalar_nonlinear_altdisc_source_20261008/MIXED_HANDOFF_v1.json in shared scratch.

## Inverse Regula Falsi and Illinois source behavior (2026-10-08)

The JAX false-position loop now follows source absolute-eps collective
stopping, signed-residual updates, NaN-only step cleanup and sequential
Illinois logical indexing. Non-source Brent rescue is removed. The explicit
512-step safety cap raises if native continuation remains open. Native
Illinois uses a compressed logical prefix mask; this unusual source behavior
is preserved, with an unrun MATLAB probe retained for fresh native checking.

All24 canonical inverse test bodies (33 distinct original predicates plus
three extras) and15 focused controls pass:39 acceptance bodies, plus four
before/after diagnostics. Root rehashed2950/2949/2927 observed runtime files,
verified three terminal stable receipts/no survivors, native source hashes,
unchanged canonical test bytes and exact method scope. Default Brent,
tol_union and evaluator code are unchanged. No full-suite or fresh native
trajectory claim. Evidence:inverse_false_position_root_acceptance_20261008.json
and inverse_cpu_source_20261008/qualified_packet.json in shared scratch.

Default flower timing remains first11.306s/warm0.738-0.753s on this CPU setup.
A static evaluator unroll trial was slower warm0.811-0.846s and was rejected.
Matched65-target sine diagnostics improve Regula Falsi7.54-8.14ms to0.156-0.219ms
and Illinois3.43-3.66ms to0.274-0.418ms at unchanged1.11e-16 error; these are
combined-process observations with changed source policies, not a general
speed guarantee or a native MATLAB comparison. The old800x finding is stale.

## Separable surface and contour sampling (2026-10-08)

Public surf/contour now use the source200x200 default grid (formerly100/150).
Scalar surf(f) follows the source matrix-infinity-norm near-constant correction;
contour colorbar=True now attaches a colorbar to line contours as well as filled
contours. Eight focused grid/override/matrix-norm/colorbar controls and eight
existing plotting regressions pass (16 total,39.03s pytest). Root independently
rehashed2970 observed bound runtime files; terminal receipt reports stable
inputs and no survivors. Full source rendering is not established: parametric
surface color corrections, automatic contour levels and native lights remain
open. Gibbs2D page replay is separate and is not claimed complete here.
Evidence:gibbs2d_source_20261008/plotting_root_review.json in shared scratch.

## AAA source scaling and infinite-query correction (2026-10-08)

AAA now preserves scaling after the first conditioning trigger, uses exact
minimum-singular-value multiplicity, follows the sign option's zero-minimum
rule, and evaluates callbacks on their original real/complex sample type.
Infinite queries now use the source barycentric limit; a baseline complex
rational example returned NaN instead of1+i. Source NaN-only support repair
is restored. New helper/evaluation math uses JAX; inherited NumPy/SciPy
SVD, greedy loop, cleanup and pole/residue kernels remain migration gaps.

Eighty-two acceptance checks plus two diagnostics pass, including18 literal
original slots. Root independently rehashed2936 and2999 observed bound files,
verified both terminal stable receipts/no survivors, native source hashes,
payload identity and exact function scope. Full AAA42/API coverage, native
random fixtures and derivative clauses33-40 remain outside this package.

Source42 now uses50 SVD calls instead of72 at unchanged49 supports; no elapsed
performance claim. Baseline/candidate poles are exactly unchanged for both
source diagnostics and the derived Python FOV replay. Thus these corrections
do not resolve the historical FOV remote-pole discrepancy; matched native AAA
sample data remains needed. Evidence:aaa_root_acceptance_20261008.json and
aaa_complex_source_20261008/qualified_packet.json in shared scratch.
No full-suite, complete JAX migration, page parity or exact-head CI claim.

## Scalar Newton source policy and GulfStream correction (2026-10-08)

The finite single-interval scalar Chebcolloc2 path now follows source initial
boundary fitting, adaptive correction solves, L2 damping/stopping and final
simplification. New numerical code uses JAX. Nineteen distinct checks pass:
nine original source predicates and ten focused controls. Four original
Chebcolloc1/ultraS nonlinear clauses remain explicitly pending. Retained legacy
controls were not rerun; this is not full-suite qualification.

GulfStream now produces five Newton updates and45 coefficients, with actual
integral error8.476552793013070e-14 versus cached8.482103908136196e-14.
Its three600x253 figures and printed output come from the same completed CPU
run; fonts, plotting grids and final digits are not exact native parity.
This supersedes the older35-update/57-coefficient numerical status below.

Root verified payload/source/artifact hashes, four terminal stable receipts
with no survivors, and1799/1796/1798/1669 observed runtime paths respectively.
Final executable AST matches qualified snapshots; observed paths do not imply
recursive shared-library closure. Failed and memory-censored attempts remain.
Evidence:scalar_newton_root_acceptance_20261008.json and
scalar_nonlinear_source_20261008/HANDOFF_v7.json in shared scratch.

Limits: general-BC dispatch, coupled/piecewise/periodic/alternate nonlinear
paths and nondefault happiness preferences remain outside this qualification.
Legacy newton_tol is accepted but ignored on this path; public tol supplies
native bvpTol and stopping uses200*bvpTol. Reusing identical frozen LU factors
is algebraically equivalent but differs from native cache-state policy; see
SOURCE_CACHE_NOTE.md. No AD multiplication defect is claimed. Parameter Newton
code is unchanged. No fresh MATLAB, isolated performance or final CI claim.

## Coupled linear-system backend qualification (2026-10-08)

All21 original linearSystem1 slots (30 scalar comparisons) pass across
Chebcolloc1, Chebcolloc2 and ultraspherical discretizations on smooth and
piecewise domains, retaining continuous norms and exact source tolerances.
Six independent controls also pass. Alternative coupled-linear calls now
solve on the requested backend instead of returning a default-backend seed;
source AD linearity flags and supplied initial functions determine routing.
New solve math uses JAX. A suspected AD zero-product defect was disproved
by structural source flags and executed controls; AD code is unchanged.

Root checked all seven qualified terminal receipts, rehashed the observed
runtime files, and reviewed source/branch equivalence. The source test bytes
are unchanged across staged gates. Baseline dispatch failures and an incorrect
exploratory control expectation remain recorded. No native RNG is required.
Inherited fixed alternative grid65 and finite-difference Frechet assembly
remain explicit source/JAX gaps; the next package addresses those algorithms.
Evidence:linear_system1_root_acceptance_20261008.json and
chebop_linear_system1_source_20261008/qualified_packet.json in shared scratch.
No complete-suite, native performance or exact-head CI claim.

## Object and binary composition, stored-factor interpolation (2026-10-08)

All30 original Chebfun-object composition predicates pass in one final CPU
gate with30 additional controls/regressions (60 total,150.73s pytest).
Typed Chebfun/quasimatrix/2D/3D/vector dispatch now preserves source errors,
periodicity, breakpoint preimages and stored point values; g(f) shares the
public composition route. Binary composition now follows the source splitting
fallback and preserves old endpoints through newly split intervals. All10
available original binary predicates pass; source11 is explicitly skipped
for missing subsequent native seed6178 inputs. The old nonsmooth proxy had
omitted splitting; its baseline failure remains recorded, and its replacement
uses the original source preferences, matrix norms and strict tolerances.

The strict 3D periodic composition failure was constructor representation
roundoff. A JAX correction enforces the same source DEIM equations using the
stored polynomial factors, with original column scaling removed during solves
for conditioning. This is a mathematically equivalent numerical adaptation,
not MATLAB's literal floating-point order. Source28 passes its unchanged10eps
bound. Independent polynomial/exponential/small-component controls and19
existing constructor/evaluation/integration regressions also pass; exponential
roundoff does not uniformly improve. Original factor endpoint storage remains
a separate fidelity gap.

Composite coverage is97 distinct passing checks, not a full repository suite.
Root rehashed the final runtime manifest and both binary gates, plus the two
constructor gates. All accepted runs were terminal with stable inputs and no
survivors. Mapped native inputs for object11/binary10 are explicitly derived,
not fresh MATLAB captures. Evidence:composition_root_acceptance_20261008.json,
compose28_root_acceptance_20261008.json and
chebfun_compose_source_root_20261008/staged_review.json in shared scratch.
The latest figure-size audit is76 mismatches across17 pages (1304 mapped
pairs), plus the existing AtmosphericTemperature mapping hole. Full parity,
publication and exact-head CI remain open.

## GulfStream actual-output and figure refresh (2026-10-08)

Source script rerun on CPU completed at83.09s supervised/3.23GiB peak,
stable inputs and no survivors. The earlier3GiB capped run is retained.
All three figures now have the source600x253 canvas and use public source
plot/plotcoeffs adapters; printed output is the actual completed computation.
Root rehashed1660 observed Python/native-extension files, checked source/run
AST identity after styling, preserved prose/MATLAB cells, and viewed all three
reference/current pairs. Fonts, sampling grids and pixels are not fully equal.

Scientific parity remains open:35 Newton updates versus reference5,57
coefficients, integral error1.605131083604050e-9 versus cached8.482e-14,
left derivative boundary residual1.458e-9. The differential residual is
1.547e-12 (cached4.581e-10); comparisons are not uniformly worse. Actual
history is retained. Source initial-guess, stopping-norm and adaptive-solve
policy gaps are documented for the nonlinear solver worker; no tolerance
or iteration-history tuning was performed. Evidence:gulf_stream_root_acceptance_20261008.json
and gulf_stream_source_20261008/SOLVER_DIAGNOSIS.md in shared scratch.

## Restriction and Ballfun Helmholtz source qualification (2026-10-08)

Chebfun restriction now preserves breakpoint vectors, existing point values,
row orientation and periodic-to-Chebyshev conversion. Overlap follows source
finite hscale for unbounded domains, domain tweaking and exact breakpoint union.
Scoped CPU gates pass 28 of 29 original predicates plus 12 controls and two
existing regressions (42 distinct). Original24 remains explicitly skipped for
missing subsequent native random inputs. Original23/27 constructor warnings
remain visible; no tolerance was changed. Root independently rehashed all seven
observed runtime manifests (1771–1779 files each); these are Python and native
extension observations, not recursive shared-library closure. The final module
provenance heading was added during review; executable AST is unchanged.

Ballfun Helmholtz passes all38 original slots (79 scalar comparisons), eight
controls and nine unchanged regressions (55 distinct). Fixes include scalar
constructor broadcasting, explicit grid parity/single-size dispatch, Spherefun
boundary data, solution-based reality and source early-empty return. Root
rehashed all six contributing runtime manifests (2930–2950 files each), checked
terminal stable-input receipts with no survivors and reviewed the early-return
AST equivalence. Inherited NumPy/SciPy solver/transforms, undersized coefficient
aliasing, native arithmetic trajectories and performance remain unqualified.

Evidence: restrict_root_acceptance_20261008.json and
ball_helmholtz_root_acceptance_20261008.json in the shared directory below.
Qualification is composite across immutable scoped gates, not a complete-suite
run. Chebfun-object composition has29 original passes and one strict numerical
failure under diagnosis; its candidate is not committed. Publication and
exact-head CI remain unresolved.

## Trigonometric Pade source qualification (2026-10-08)

Public trigpade now follows the source Laurent coefficient ordering, first
null-space vector, degree-defect reduction, zero-denominator diagnostic,
n=0 values-constructor branch, final simplify and conditional imaginary-part
warning. New numerical kernels use JAX; the separate legacy ratapprox API
and trigremez are unchanged. Four staged terminal gates provide36 distinct
passes:27 original deterministic predicates,7 independent controls and2 legacy
regressions. Original random-handle predicates5/10/15 remain explicitly skipped
pending native unseeded tt inputs; the MATLAB capture harness is unrun.
Root rehashed2939/2939/2939/2936 observed runtime files and verified payload
identity. Full Ruff/F821, provenance/NumPy policies and diff checks pass.
No native singular-vector/roundoff identity or performance qualification.
Evidence: trigpade_root_acceptance_20261008.json in the shared directory below.

Active: Ballfun Helmholtz original38; source restrict29 and unbounded overlap;
root Chebfun-object composition30 (28predicates pass, one unbounded pending,
one strict periodic numerical discrepancy under diagnosis). No complete-suite,
remote publication or exact-head CI claim.

## Fourier and NonsmoothFOV page reruns (2026-10-08)

Both scripts completed alone on CPU against1fbf3a2a. Fourier:51.07s,
2.79GiB peak,8figures600x270; actual adaptive composition now161terms,
matching the cached reference length without tuning. NonsmoothFOV:153.93s,
4.09GiB peak,14figures600x253; all source stages including final inverse plot
restored. Both receipts show stable inputs/no survivors; root independently
rehashed1661/1652 observed Python/native-extension files and reviewed all22
reference/current figure pairs. Final rendering-only edits preserve line arrays
and numerical AST; their lightweight renderer has no RSS supervisor receipt.
Prose and MATLAB code cells unchanged; published output is actual execution.

Historical numerical/pixel parity is still incomplete: Fourier native seed0
Gaussian input is unavailable, extrema select different ties, and area residual
is-4.020849375084742e-16. Nonsmooth actual lengths805/545/5546/3671 differ
from803/545/5636/3509; remote AAA poles and renderer differences remain open.
Fresh audit of1304 mapped figure pairs leaves79size mismatches across18pages
(down from93/19); AtmosphericTemperature still has a historical slot10 mapping
hole. Matching dimensions do not certify content or pixel parity.
Evidence: fourier_page_root_acceptance_20261008.json,
nonsmooth_page_root_acceptance_20261008.json, and
figure_size_current_fourier_nonsmooth_20261008.json in the shared directory.

## Disk constructor qualification (2026-10-08)

All34 original disk constructor predicates now pass across staged CPU gates,
including the original high-rank field (129.85s pytest,4.07GiB peak). Composite
coverage is72 distinct checks:34 originals,9 constructor controls,8 numerical
rank controls,3 failure-state controls,16 smooth and2 nonsmooth regressions.
The first extended gate's two test-import mistakes were corrected and retested;
no failed assertion or resource-censored run is counted as a passing gate.

The public diskfun factory now supports the source constructor forms covered
by those predicates. Fourier projection preserves even/odd stored dimensions.
Diskfun.numerical_rank(tol=0) exposes source singular-value threshold semantics;
legacy .rank remains the stored factor count. PhaseTwo failure now prevents
post-cap sample-test restarts, as in MATLAB. Original radial/angular regressions
pass unchanged with peaks0.87/0.96GiB; source nonconvergence warnings remain.
Root rehashed every observed file in eight qualification gates and checked
final payload identity and unchanged original assertion bodies. Evidence:
disk_constructor_root_acceptance_20261008.json in the shared directory below.
Inherited NumPy/SVD kernels, unsupported constructor forms, native trajectories,
full combined tests and exact-head CI remain unqualified.

## Latest norm and composition review (2026-10-08, publication pending)

Full CPU parity remains incomplete. The current local batch restores:

- Chebfun norms: all34 original predicates and22 additional controls/regressions
  pass in staged runs. Source24 still emits an inherited65537-point constructor
  convergence warning despite satisfying its original norm bound. Stored-point
  extrema comparisons remain incomplete. Finite unbounded restrictions now
  retain the bounded internal piece protocol without resampling.
- Trigtech composition: all11 original predicates plus12 controls pass. Source
  adaptive nested/resampling replaces the public fixed-grid approximation;
  cos(80*sin(x)) error falls from1.877 on64 samples to3.40e-14 on249 terms.
  Empty scale metadata and empty optional operands follow source defaults.
  Classic/plateau/custom happiness and empty-receiver composition remain open.
- Chebfun3v composition: all10 original predicates plus29 controls/regressions
  pass; source range/periodic/error/empty-component dispatch restored. Independent
  review rehashed2947 observed runtime files and19 pinned MATLAB inputs.

Evidence in the shared directory below: chebfun_norm_root_acceptance_20261008.json,
trig_compose_root_acceptance_20261008.json, and
chebfun3v_compose_independent_review_20261008.json. No fresh native MATLAB,
full combined suite, remote publication, or exact-head CI is established.
Fourier/NonsmoothFOV pages await computation reruns after this integration.
Disk constructor final public-rank/failure-propagation qualification is active;
both inherited nonsmooth regression assertions now pass in isolated runs.

## Latest reviewed arithmetic/composition (2026-10-08, local b6ce6212)

Full parity remains incomplete. New local packages:

- `194e2d07`: Chebfun addition orientation, numeric-column expansion,
  breakpoint threshold and quasimatrix constants. Composite114 passing checks
  include33 original predicates, six controls and75 regressions; source15 is
  vacuous and independently controlled. Source29 is explicitly skipped pending
  native source-order RNG; source28 uses mapped recovered primitive uniforms,
  not a fresh native capture. Its strict bound passes despite two retained
  constructor convergence warnings. Three accepted gates have terminal receipts,
  stable inputs/no survivors; root rehashed2942/2947/2939 observed files.
- `b6ce6212`: all35 original Chebfun3 composition predicates plus27 controls
  and13 regressions (75 distinct staged passes). Full64-check gate903.01s,
  peak8.20GiB; sample/arity followup40pass and complex-vector guard12pass.
  Source sample defaults/empty/factor output, N25 range estimation, periodic
  dispatch and diagnostics restored. Numerical contractions/typed dispatch
  unchanged across followups; original test bytes identical. Root rehashed
  all three gates' observed runtime files. Inherited constructor/fiberDim/RNG
  trajectories and broader Chebfun3v API remain open.

Evidence: `chebfun_plus_root_acceptance_20261008.json` and
`chebfun3_compose_root_acceptance_20261008.json` in the shared directory below.
Full Ruff/F821, source provenance/NumPy policies and diff checks pass locally.
No full-suite, native MATLAB, remote publication or exact-head CI claim.

Active: Chebfun norms (heavy singular/slow-decay cases); remaining Chebfun3v
composition; Fourier/NonsmoothFOV final page review and unresolved output gaps.
Root disk constructor has33/34 original assertions qualified in disjoint groups;
high-rank10 queued. Two inherited nonsmooth disk regressions remain unqualified
following a bounded RSS stop; no weakened assertions or tolerance changes.
Updated static inventory atbf21897f reports1094 PRESENT/8 MASKED/0 missing out
of1102 source files, which establishes mapping only, not semantic coverage.

## Latest reviewed packages (2026-10-08, local eabdbfa0)

Full parity remains incomplete. This checkpoint supersedes prior active-work
lists; earlier evidence and limitations remain applicable.

- `f3c27d55`: source ultraspherical REC/ASY/GW rules and all 40 original
  predicates. Worker gate: 70 passing checks, one live-MATLAB fixture
  deselected. Root review: nine additional controls and three bitwise interior
  equivalence comparisons pass. This is composite evidence, not one final
  79-test run. Mapped-interval JIT, source uncapped interior stopping, small-order
  stopping and boundary underflow warning repaired. Both gates have stable
  inputs, terminal receipts and no survivors; 2,950 and 2,934 observed runtime
  files independently rehashed. Dynamic interval validity requires caller
  validation before tracing. No general native last-bit/performance claim.
- `eabdbfa0`: FermiDirac original four minimax fits and six 600×270 figures;
  actual elapsed-time cells and conditional diagnostics. Numerical run 96.75 s,
  peak sampled 3.86 GiB; saved-data rendering 1.14 s. All six reference/candidate
  pairs visually reviewed, line-array/pickle/source hashes bound and 1,654
  observed runtime files rehashed. Initial fplot still uses uniform 2,000-point
  sampling; native adaptive sampling, CF rank-loss paths, font/dash/raster
  identity remain open. The protected original dirty script is preserved;
  its unsupported diagnostics are not accepted by this replacement.

Root evidence in the shared directory below:
`ultrapts_root_acceptance_20261008.json` and
`fermi_dirac_root_acceptance_20261008.json`.
Full Ruff/F821 and diff checks pass. Quadrature source provenance/NumPy checks
pass; the following page-only commit does not change library policy results.

Current parallel packages: all 35 Chebfun3 composition predicates plus controls
(full 64-check gate, heavy lane); Chebfun addition source assertions/regressions
(singular clause 28 queued); FourierBasedChebfuns source plotting/prolongation.
Three workers plus root use CPU only. Figure-size audit remains 93 mismatches
on 19 pages plus the historical mapping hole; Fermi dimensions already matched
before this update. No full integration suite, fresh native MATLAB, remote push
or exact-head CI has been established.

## Latest reviewed packages (2026-10-08, local ce0f8c27)

Full parity remains incomplete. These additions supersede the older checkpoints:

| Commit | Change | Verified CPU evidence |
| --- | --- | --- |
| 456be0c4 | Rational minimax CF/AAA-Lawson/CDF initialization, failure status, iteration rounding and constructor dependencies | 83 distinct passing checks across three staged gates; four native polynomial fixtures skipped. Not one final-tree suite; selected import origins, incomplete native dependency closure |
| 3052acbe | Chebfun3 default/inf/even-order norms and source selector errors | All six original predicates plus ten controls; 12 + four disjoint checks. Default Frobenius loop unchanged; infinity uses existing optimizer, not MATLAB optimization trajectories |
| fd083258 | BestApprox source computations, genuine diagnostics and five reference-sized figures | Numerical page 83.33 s, 3.56 GiB; saved-data tick correction 0.95 s. All five 600×269, numerical run AST/line arrays unchanged, all pairs visually reviewed; no exact pixel/native last-digit claim |
| ce0f8c27 | Periodic boundary dispatch, piecewise generalized eigenproblem and explicit Fourier coefficient assembly | All 41 original predicates; 49 distinct checks across 45 + 14 gates with ten overlaps. Source bounds/domains retained, independent endpoint/residual controls included |

Root acceptance records in the shared evidence directory below:
`minimax_initialization_root_acceptance_20261008.json`,
`chebfun3_norm_root_acceptance_20261008.json`,
`bestapprox_page_root_acceptance_20261008.json`, and
`chebop_periodic_root_acceptance_20261008.json`.
All supervised runs above have terminal receipts, stable bound inputs and no
surviving owned processes. The norm and periodic gates also bind observed native
and Python dependency origins. Minimax/page origin limits are recorded explicitly.
Full Ruff/F821, provenance/NumPy policies and diff checks pass at integration.

Fresh `figure_size_current_bestapprox_20261008.json` rehashes all 1,304 mapped
PNG pairs: **93 size mismatches on 19 pages**, down five after BestApprox.
The historical AtmosphericTemperature slot-10 mapping hole remains; reused slot
mapping and dimensions do not establish full figure/content/pixel parity.

Periodic generalized eig uses JAX inverse-pencil eig and requires invertible A,
positive differential orders and a shared domain. Singular/arbitrary pencils,
all selectors and native QZ arithmetic remain open. Explicit scalar linear
trigspec is qualified; nonlinear trigspec retains its regression-tested fallback.
BestApprox CF(16,16) still takes a documented source-prescribed fallback after
an inherited unsupported CF branch. Full CDF numerical hard-case parity remains
open despite dispatch controls. Chebfun3 extrema remain an inherited algorithm.

Active worker packages: Chebfun addition original predicates; ultraspherical
REC/ASY and original 40 predicates with final JIT qualification; FermiDirac
source page preparation awaiting the serial heavy lane. Root reviews deliveries.
CPU only, three workers plus root; no extra agent slots. Publication/fresh native
MATLAB blockers below are unchanged. No exact-head CI or full-suite claim.

## Latest accepted packages (2026-10-08, local 6e62af23)

Full parity remains incomplete. These additions supersede the table below:

- `6e62af23`: all 36 original Chebfun sum predicates restored; 114 CPU tests
  pass in 240.31 s pytest / 287.54 s supervised, peak 3.79 GiB. Variable
  limits, dimensions/orientation, unbounded array columns, and source singular
  multiplication cancellation/simplification are covered. Root rehashed 2,954
  observed runtime files and verified all seven payload files byte-identical
  to the qualified snapshot. See `chebfun_sum_root_acceptance_20261008.json`.
- `5b6ca0bf`: supported diffmat error identifiers/messages and breakpoint
  warning restored; ten focused checks pass. Numerical paths are AST-identical
  to the previous qualified implementation. See
  `diffmat_diagnostics_root_acceptance_20261008.json`.

Evidence files are under the shared directory given below. Full lint/F821,
NumPy/provenance policy and diff checks pass. These are local commits; remote
publication, exact-head CI and fresh MATLAB remain blocked as documented below.
No full-suite or complete API parity is inferred from focused gates.

Three worker slots are occupied: periodic solver's 41 original predicates;
BestApprox/minimax initialization and page figures; ultraspherical quadrature's
40 original predicates and missing REC/ASY methods. Root review found a minimax
Python-versus-MATLAB rounding mismatch and failed-trial status discrepancy;
those are being corrected before acceptance. The periodic worker holds the
serial heavy lane; BestApprox page is queued next. Small bounded tests and
source work proceed concurrently on separate CPU sets.

## Latest CPU qualification (2026-10-08, local b3b0258d)

Full parity remains incomplete. This checkpoint supersedes earlier counts.
Shared evidence directory:
`/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/`.

| Commit | Verified change | CPU evidence and limits |
| --- | --- | --- |
| c9edc577 | Singular addition scales, endpoint reconstruction, zero identity and diagnostics; original cumsum clauses | Composite gates: 30 + 2 prior checks, 31 final-library checks and seven strengthened controls; see acceptance record, not one combined final-tree run. Native seed666 identity and difficult right-pole happiness remain open |
| f6338189 | 3D scalar callbacks, slices, paths, continuous factor norm, empty integration and Runge entry | 64 pass: all 59 original divide/feval/sum3 predicates plus five controls; full 100³ grids, 134.37 s supervised, 2.48 GiB; native RNG identity and direct factor-transfer slicing remain open |
| b7b2a320 | Greeks source computations, actual relative-error output, reference dimensions and clipped interpolated surface rendering | Full page 107.75 s, 114.76 s supervised; corrected rendering 45.40 s from bound saved figures without recomputing numerics; all five 600×268; font/projection/triangulation and native last digits remain open |
| 75fc23cf | Oriented Chebfun matrix products, stored outer factors, singular conjugation and public sizes | 94 pass including 75 original predicates; five original random clauses await native source-order fixtures; 169.44 s supervised, 2.67 GiB |
| b3b0258d | Rectangular/mixed-grid/periodic differentiation and boundary matrices | 199 + five distinct checks pass, including all 62 original predicates; one unrelated integration-degree skip; final source differs only in docstrings from main gate |

Acceptance records: `singfun_plus_cumsum_root_acceptance_20261008.json`,
`chebfun3_source_root_acceptance_20261008.json`, and
`greeks_page_root_acceptance_20261008.json`,
`chebfun_mtimes_root_acceptance_20261008.json`, and
`diffmat_source_root_acceptance_20261008.json`. The 3D gate records complete
observed dependency/native origins; the Greeks page records only selected
import origins and its frozen source/render inputs. Earlier Greek renders
with off-window surface walls were rejected and superseded by v4.

Fresh inventory `matlab_test_static_inventory_b3b0258d_20261008.json` reports
**1,094 present, eight masked, zero module-skipped or missing**, with all 1,102
source/port hashes revalidated. These are file/marker counts, not assertion,
RNG, runtime or API parity. Manual audit found 80 original Chebfun mtimes
predicates behind four old assertions and 62 diffmat predicates behind three.
`source_assertion_triage_v2_3b20c740_20261008.json` guides further review;
static assertion counts alone also cannot establish parity.

Fresh figure audit `figure_size_current_b7b2a320_20261008.json` has **98 size
gaps on 20 pages**, plus the historical AtmosphericTemperature slot-10 mapping
hole. It rechecks all 1,304 mapped PNGs; matching sizes do not qualify pixels,
prose, computations or stdout. The Greeks relative error now follows source;
its call-vega approximation error about 6.431e-8 also occurs historically.

The cached prose/MATLAB-cell regeneration audit covers all 322 pages:
`prose_cell_audit_75fc23cf_20261008/report.json` has 321 strict matches after
whitespace normalization and excluding stdout/figure paths. Manual review of
ThreeBodyProblem found only an original-site versus local bibliography link;
no prose correction was needed. This uses the existing generator, not an
independent HTML semantic proof, and does not qualify Python computations.

The fixed-grid diagnostic `highfreq_integral_diagnosis_20261008/conclusion.json`
reproduces the remaining oscillatory integral error at 65,537 samples.
Compensated coefficient reduction changes only about 1e-17; extended-precision
sampling with the same float64 transform reduces error to 4.75e-17. This
localizes the dominant discrepancy to sampling arithmetic. No production
precision/refinement change or tolerance relaxation was made; native sample
trajectories remain needed for a justified source-parity repair.

All three workers remain occupied: Chebop periodic (41 source predicates),
Chebfun sum (36, with captured seed7681 inputs), and BestApprox/minimax source
initialization/fallback semantics plus five figures. The session supports four
agents including root, and the user's newer request authorizes all workers.
CPU only. Coordinate the single heavy lane with BestApprox/minimax; small
bounded matrix gates run independently. Diffmat's canceled duplicate preflight
has no final receipt and cleanup is uncertified; it is excluded from acceptance.
Its accepted main and focused runs both certify stable inputs/no survivors.
MATLAB error identifiers/text and broader source Legendre quirks remain open.

Previously accepted f31f378a plain elliptic/composition has 93 passing checks;
561eef04 classic multiplication has 44 passing checks and six native-pending
clauses. Bounded calculus 5071198f actually ran 39 pass/one failure:
cos(1e4*x) integral error 6.519329347198788e-14 exceeds unchanged bound
2.2204460475929106e-14. A forward-map diagnostic still failed and was excluded;
the final tree preserves the existing strict xfail by a marker-only change.

Native startup remains blocked: `matlab_startup_diagnosis_20261008.json` records
long-TMP plugin assertion, short-TMP and -nojvm exit1 without scientific output,
and independent AF_UNIX/AF_INET socket creation EPERM. IPC restriction is a
supported inference, not a traced MATLAB syscall diagnosis; ptrace is denied.
Required change: authorized execution setting permitting local IPC, with a
short shared TMP. Publication remains blocked by system SSH configuration and,
with user configuration, unavailable GitHub DNS/API. No new exact-head CI.
Do not repeat unchanged attempts; continue local work and preserve bundles.

## Earlier CPU qualification (2026-10-08, local 7bb59fb8)

Full parity remains incomplete. This section supersedes the older checkpoint
below. Fresh static inventory at `87a4f4b0` revalidated all 1,102 MATLAB test
mappings: **1,087 present, 15 with skip/xfail markers, zero module-skipped or
missing**. These are file/marker counts, not complete assertion or runtime parity.
The following constructor commit does not change mapped source-test files.

| Commit | Verified change | CPU evidence |
| --- | --- | --- |
| c1cbba95 | JAX Bessel primitive and source AD | 40 passing checks, including six original clauses |
| fa4748af | Classicfun division, public NaN predicates and complex singular cancellation | 31 passing checks; all 15 original division clauses restored |
| c8cbc1c1 | JAX Jacobi elliptic primitive and AD | 60 passing checks; plain tolerance/preferences/composition still being repaired |
| 456ca646 | AD deflation and linearity detection | 66 passing checks, including 48 original clauses |
| cb1e7e03 | Source mixed Newton damping and collocation projection | Ten passing controls; LaneEmden full run 163.30 s, two 600×269 figures |
| 87a4f4b0 | Singular extrema layouts and complex-root order | 52 passing checks; all 29 original extrema/root clauses restored |
| 7bb59fb8 | Adaptive construction shares source-ordered JAX ACA | 106 passing checks; corrected overlapping rank-budget fixture separately passes |

Evidence remains in
`/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/`:
`ad_bessel_root_acceptance_20261008.json`,
`classicfun_rdivide_root_acceptance_20261008.json`,
`ad_ellipj_root_acceptance_20261008.json`,
`ad_remaining_root_acceptance_20261008.json`,
`nonlinear_projection_root_acceptance_20261008.json`,
`classic_extrema_roots_root_acceptance_20261008.json`, and
`adaptive_aca_root_acceptance_20261008.json`.
Static inventory: `matlab_test_static_inventory_87a4f4b0_20261008.json`.

The corrected LaneEmden run prints actual L2 error `1.602e-14` and radius
`3.653753736220`. Newton uses the source update/error convergence criterion;
white-dwarf continuous residual `1.0864e-6` and derivative boundary error
`7.458e-10` still need native comparison. Only one finite-interval function with
explicit scalar parameters is qualified. No full nonlinear-system parity claim.

VanillaOptions remains held: the ACA repair does not close its out-of-domain
zero curve. A saved-state analytic diagnostic shows the source endpoint tangent
snap and subsequent Newton step can leave the domain even for the exact field.
Independent interpolation/snap replay is ongoing; no clipping or relaxed bound
has been introduced. The figure audit still has 103 dimension gaps on 21 pages,
plus its historical mapping hole; prose/output/pixel parity also remains open.

Three worker slots are active: plain elliptic/composition semantics, literal
bndfun differentiation/inner-product/integration clauses, and option-contour
source diagnosis. The session limit is four concurrent agents including root;
new user instructions permit all three worker slots. Heavy examples use one
reserved serial lane. CPU only. Root integrates reviewed patches separately.

All commits above are local. Normal SSH push fails the system hostbased-config
ownership check; explicit user SSH config loads but GitHub DNS fails in this
restricted session. No exact-head CI result is available. Native MATLAB startup
also currently exits without output. Do not repeat unchanged failed launch or
network attempts, claim publication, or mark the goal complete.

## Earlier CPU qualification (2026-10-08, local 89a9c49a)

Full parity remains incomplete. The static inventory revalidated all 1,102
source/port mappings: **1,079 present, 19 with skip/xfail markers, 4 module-skipped**.
File presence does not establish complete assertions or runtime parity.

| Commit | Verified change | CPU evidence |
| --- | --- | --- |
| ca70c195 | Source repmat forms, point values and orientation | 64 passing tests; native seeded source inputs |
| ea3fb9e2 | Exact singular equality, source cancellation, complex-transform conjugacy | 207 passing tests; fresh MATLAB equality/complex predicates pass |
| 73772a94 | Explicit small Hermite REC source outcomes, including source defects | 60 passing tests against captured native outcomes; bounds unchanged |
| 6cda66b9 | AD product integrals and plain cumprod | 65 passing tests plus 12 integration checks; six original clauses |
| d14c813a | JAX Airy primitive, complex public wrapper and source AD | 44 passing checks, including 14 original clauses at source bounds |
| d74670ee | JAX-only Chebtech2 value/coefficient transforms | 102 passing source, inverse, cancellation and JAX controls |
| 53d7efb4 | EquispacedData source computations and historical figure layout | Seven 598×273 figures, four observed output blocks; 39.06 s run |
| 8a33d4ce | AD Fredholm/Volterra operators and source array norms | 67 passing tests, including six original AD clauses |
| 702456ef | Complex-preserving mixed scalar AD solver and LaneEmden page | Four solver controls; full page 270.95 s, L2 residual 1.563e-14, two 600×269 figures |
| 31d78758 | Fused JAX Chebyshev nodes | 89 passing tests plus four mixed solver controls; profiled first flower inverse 9.17 s |
| 16b62ba2 | Source endpoint Dirac masses, stems and EuropeanOptions | 26 strict integration tests; seven 600×268 figures and seven observed output blocks |
| 89a9c49a | AD numeric evaluation, jumps and domain metadata | 98 passing tests, including all six original clauses |

Evidence directory: `/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/`.
Acceptance records: `repmat_root_acceptance_20261008.json`,
`singfun_equality_root_acceptance_20261008.json`,
`hermite_small_root_acceptance_20261008.json`,
`ad_cumprod_prod_root_acceptance_20261008.json`,
`airy_root_acceptance_20261008.json`, `dct_root_acceptance_20261008.json`,
`equispaced_root_acceptance_20261008.json`,
`ad_integrals_root_acceptance_20261008.json`,
`lane_scalar_root_acceptance_20261008.json`,
`chebpts_fused_root_acceptance_20261008.json`,
`europeanoptions_root_acceptance_20261008.json`,
`ad_feval_jump_root_acceptance_20261008.json`.
Inventory: `matlab_test_static_inventory_89a9c49a_20261008.json`.

Fresh figure inventory `figure_size_current_89a9c49a_20261008.json` reports
**103 dimension mismatches on 21 pages**, among 1,304 mapped slots plus one
historical mapping hole. Matching dimensions do not establish matching pixels
or computations. EquispacedData RNG and last digits remain open. EuropeanOptions
now uses public library payoffs/Dirac masses; the digital price agrees with its
analytic check to about 2e-15. A cubic endpoint root rounding diagnostic remains
open; source exact endpoint comparisons were retained. No fresh native MATLAB
EuropeanOptions output is claimed.

Airy uses JAX ODE continuation, contour quadrature and asymptotics. Independent
sampled complex arguments through radius 100 qualify that range only; no fresh
native Airy outputs, universal complex-plane accuracy or extreme scaling claim.
Source AD covers kinds 0,2. Fredholm/Volterra inner quadratures remain fixed-order
adapters, so general kernel accuracy remains open. AD evaluation/jump side flags,
character endpoints and callable composition remain unqualified.

The public Chebtech2 transforms no longer use eager NumPy mirrors. Two real
FFTs preserve complex conjugacy and pure components. Other transform families
still contain legacy NumPy/SciPy paths. Fused nodes reduced the profiled flower
first inverse from 21.91 s to 9.17 s, with warmed calls 0.707–0.718 s and composition
error 1.05e-15. Other workers were active; this is not an isolated speedup or
MATLAB timing comparison. Initial compilation and rounding-sensitive adaptive
lengths remain important. The LaneEmden page ran before fused nodes; four solver
controls separately qualify that integration, not a fresh full page.

Small explicit Hermite ASY remains unresolved: 28/38 source cases pass;
focused arithmetic changes did not close six remaining comparisons. Strict
bounds and failed evidence are preserved in
`hermite_asy_small_next_20261008/HANDOFF.md`. Do not substitute REC results.
Singular/delta repmat rows and general complex/array product integrals remain open.
The accepted mixed solver still has damping/stopping adapters; a separate source
algorithm package is under review. Its passing diagnostics are not yet accepted
library/page parity.

Three Astra workers cover Bessel functions, source nonlinear damping/refinement,
and VanillaOptions. Root reviews/integrates and restores omitted division tests.
The user requested maximum parallelism; the four-agent session cap is full.
CPU only; heavy examples use one serial lane. Package status is recorded in
`current_parallel_checkpoint_20261008.json` outside the repository.

Last confirmed remote main is **8a6d9c80**. Restricted networking prevents
GitHub API refresh/publication; newer commits are local. Prior successful
pushes and MATLAB captures remain valid historical evidence. Full source/test,
page/figure, performance and exact-head CI gates remain required.

## AD error functions and finite classification (2026-10-08)

Commit `d1fc3256` adds five source AD error functions and restores all15
original test clauses with fresh MATLAB seed6179 inputs.291 distinct CPU
tests passed across a preserved initial run (276 passes, one control failure)
and corrected focused run (30 passes); this is not one all-green291 run.
The correction changed a domain expectation only. Native MATLAB returned all15
clauses true. Complex/array-valued generality and speed remain unqualified.
Evidence: `ad_erf_root_acceptance_20261008.json` in the shared directory below.

Commit `eceb146d` includes breakpoint values in finite classification, matching
source semantics. All12 original finite/infinite predicates and4 controls pass;
the previous implementation failed both isolated-infinite-point predicates.
Evidence: `finite_root_acceptance_20261008.json`. These16 CPU tests were run
with the pinned environment; no fresh MATLAB run was performed for this package.

Three Astra workers remain active on repmat, small-order Hermite and the
LaneEmden solver/page, with the coordinator reviewing and publishing. This is
the session limit of four agents including the coordinator. CPU only; one heavy
example lane. CI at154923bf is still running with no failures observed; newer
commits require their own completed CI. Full parity remains incomplete.


## Fresh MATLAB logical and singular Laguerre checks (2026-10-08)

Commit `eb666fec` repairs Chebfun.any's continuous reduction: exact nonzero
coefficients and point values count, NaNs do not. Tiny functions and isolated
point values no longer disappear under an epsilon cutoff. Source singular
and unbounded all/any tests are restored, together with previously omitted
logical predicates.20 CPU tests plus2 focused final assertion checks passed.
Fresh pinned MATLAB returned all6 all-results and16 any-results true; the
Python tests now use its actual seed6178 probe vectors. Python scalar/vector
storage and exception identifiers remain documented adapters. Evidence:
`chebfun_logical_root_acceptance_20261008.json` in the shared directory below.

Commit `faee142c` permits singular alpha=-1 in Laguerre EXP as MATLAB does,
without changing formulas.54 CPU checks passed against45 fresh MATLAB
method/order outcomes. Source nonfinite outputs are retained. RH/RHW exception
and EXPW barycentric shape differences remain open, not declared parity.
Evidence: `laguerre_alpha_minus_one_root_acceptance_20261008.json`.

Fresh MATLAB is available; old startup blockers are historical. Remote CI
at1daa4990 remains in progress with no failures at the latest observation.
Exact-head green CI, remaining numerical/test contracts, page/figure audits
and performance comparisons still prevent full-goal completion.

## AD power and complete Chebfun2 evaluation tests (2026-10-08)

At `ad483fd1`, fresh static inventory has1,067 present files,25 with literal
skip/xfail markers and10 module-skipped files among1,102 source tests. This is
file coverage, not proof of all assertions or MATLAB random-stream parity.
Evidence: `matlab_test_static_inventory_ad483fd1_20261008.json` in the shared
evidence directory below.

AD power passed261 CPU checks, restoring21 source clauses, literal derivative
arithmetic, linearity/domain rules and reverse dispatch (commit8fda14fd;
`ad_power_root_acceptance_20261008.json`). Chebfun2 evaluation restores all24
source predicates, matrix2-norms, complex syntax and full1001x1001 transposed
grid;24 pass (commitad483fd1; `chebfun2_feval_root_acceptance_20261008.json`).
The old Chebfun2 path-composition skip described a different requirement from
the pinned test and was replaced by its actual source predicates.

Fresh MATLAB startup now works in the unrestricted environment; worker
Laguerre boundary capture succeeded. Current remote CI724ce10c numerical
shards are running; lint,code-quality and golden-reference gates passed.
Do not infer green CI on newer local commits. Three worker lanes continue AD
error functions, Laguerre alpha=-1 and a compressed scalar-parameter Newton
solver for LaneEmden. Its n24 diagnostic converged but public integration and
refined qualification remain open.

## Publication recovered and Fourier/EXP qualification (2026-10-08)

Normal direct push now succeeds in the unrestricted execution environment.
Main was published through `48936f2b`; CI run37827871039 was queued at last
observation, not yet green. Earlier SSH blockage below is historical.
Evidence: `/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/publication_recovered_20261008.json`.

Fourier real/imag extraction now follows source grid values, exact-zero
handling and imag length preservation. Supplied/scaled values use a dynamic
JAX leaf, and coefficient transforms invalidate it.107 CPU tests passed,
including original source assertions with matrix infinity norms, even modes,
tiny components, cache invalidation and JVP. Commit `48936f2b`; evidence
`trig_parts_root_acceptance_20261008.json` in the shared directory above.

Explicit Laguerre EXP/EXPW is ported with staged JAX source expansions.
46 frozen tests passed (40 method/grid combinations within20 cases plus26
controls). Source approximation errors, alpha=-1, fresh MATLAB and performance
qualification remain explicit limitations. Commit `d6f8f9b3`; evidence
`laguerre_exp_root_acceptance_20261008.json` in the same directory.

Prior remote CI at76c1e79e failed a sphere-vector timeout and the randnfun
mirror's missing fixture setup. The latter is under focused correction;
new-head CI must be checked. Full goal remains incomplete.

## Current qualification (2026-10-08, local ec3c4946)

Full parity with MATLAB Chebfun `7574c77680d7e82b79626300bf255498271a72df`
remains incomplete. This checkpoint supersedes older counts and disk k=7
failure reports below. CPU only. Three worker agents plus the coordinator
are active; the user requested this expansion of the older two-worker limit.

Fresh static inventory: **1,059 present, 26 containing skip/xfail markers,
17 module-skipped, zero missing after explicit consolidated mappings**, out
of 1,102 MATLAB test files. These are file classifications, not proof of all
source assertions, random streams or runtime parity. Fresh PNG dimensions:
**118 mismatches across 24 pages**, 1,304 mapped slots, one unresolved
AtmosphericTemperature mapping slot. Equal sizes do not establish visual parity.

Evidence directory:
`/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/`.
Fresh inventories: `matlab_test_static_inventory_ec3c4946_20261008.json` and
`figure_size_current_ec3c4946_20261008.json`.

Latest verified changes:

- Disk numeric BMC reconstruction now uses the actual doubled matrix rank
  capacity, an explicit correctness adaptation to the source cap. Pivots and
  tolerances are unchanged. All ten original Helmholtz predicates pass in a
  19-test CPU gate. At k=7 the actual disk L2 error is 5.81034e-12 against the
  original 2.22045e-8 bound, down from the preserved 3.75766e-8 failure.
  Commit `3a69ee84`; `disk_doubled_rank_root_acceptance_20261008.json`.
- AD elementary operations restore 129 source clause forms for 43 operations
  and repair real complex-branch evaluation and incoming-domain retention.
  371 distinct tests pass across two runs (367 original plus 39 focused,
  35 overlapping). Fixed degree7 inputs do not establish MATLAB RNG parity.
  Commit `97773b63`; `ad_elementary_root_acceptance_20261008.json`.
- Hosepipe restores four public surface/slice plots at 600x253 and records
  actual displays while preserving prose/source blocks. Lighting, pixels and
  the annulus Fourier representation (1195 coefficients versus historical
  about 101) remain open. Commit `ec3c4946`;
  `hosepipe_root_acceptance_20261008.json`.
- Lower-order Laguerre RH/RHW passed 45 tests in two bounded runs, including
  documented source nonconvergence at some n=42 parameters. The interrupted
  monolithic run remains unqualified. Commit `81045e3f`;
  `laguerre_small_rh_root_acceptance_20261008.json`.
- Earlier accepted AD inner/norm, ConformalVis and RationalHarmonic packages
  are in commits `5adbbe93`, `dd3664c3` and `6fdc4c2c` with individual root
  acceptance records in the same evidence directory.

Current parallel work: original AD arithmetic tests, explicit Laguerre EXP
compilation diagnosis, and LaneEmden source figures. Heavy examples run in
one serial lane. Broad test clauses, outputs, visual audits, known numerical
issues and matched performance measurements remain open.

Publication is unresolved: ordinary push at `81045e3f` failed with
`Bad owner or permissions on /etc/ssh/ssh_config.d/20-hostbased.conf`.
Evidence: `ordinary_push_81045e3f_20261008.json`. No exact-head green CI or
fresh MATLAB execution is claimed. Local commits and immutable evidence are
preserved; an authorized working publication environment or corrected system
SSH configuration is needed for remote publication.

## AD calculus source clauses and RECW (2026-10-08)

AD calculus154 passed:27 clause forms from all9 operations in the original
cumsumDiffSumMean test,8 new domain/vector controls and119 prior regressions.
Sum, mean and numeric-order derivative evaluation preserve Jacobians and
complex/vector values. One module skip removed. Source norms and tolerances
retained; fixed degree7 inputs do not establish MATLAB RNG parity.
Evidence: `/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/ad_calculus_root_acceptance_20261008.json`.

RECW62 passed: JAX recurrence stops after retaining the first exactzero weight,
with bitwise zero detection, staged overflow and subnormal continuation.
At1000/alpha=-0.5 two source-emulated subnormal weights precede the finalzero.
Ordinary REC/default algorithms retained. Initial59pass/2fail evidence is
preserved: the failed tests incorrectly assumed alpha0 must have subnormals;
reference-pattern checks corrected that premise without changing bounds.
Variable output length is eager; bary signs use returned length, as in RHW,
repairing the source n-size dimension defect. Both gates had stable inputs,
verified runtime origins and no survivors. No fresh MATLAB or performance claim.
Evidence: `/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/laguerre_recw_root_acceptance_20261008.json`.

## Domain remapping state preservation (2026-10-08)

12 CPU source/regression checks passed with stable source/environment inputs
and no surviving processes. new_domain now uses JAX affine mapping, retains
coefficient objects, stored point values and row/column orientation, and maps
delta locations while retaining magnitudes and derivative orders, as specified
by newDomain/changeMap. The old constructor discarded this state. Existing
remapping and CF output contracts pass. Scope is finite bounded pieces;
unbounded/other FUN representations, fresh MATLAB and full native closure
are not qualified.
Evidence: `/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/newdomain_root_v1_20261008/review.json`.

## Anchored AD cumulative integration (2026-10-08)

119 CPU checks passed (16new plus103 unchanged AD/saved-reference cases),
with stable inputs, verified runtime origins and no surviving processes.
The old pseudoinverse differentiation Jacobian mapped a constant perturbation
to x instead of x+1 on[-1,1], giving error1. The source anchored cumsum_op
now composes with the prior Jacobian, supports repeated integration and
preserves linearity and interior domain breaks. Original Taylor tolerances
are retained on fixed polynomial controls; no MATLAB RNG or fresh capture
claim. Remaining operations in the original combined AD test remain open.
Evidence: `/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/ad_cumsum_root_acceptance_20261008.json`.

## Fourier coefficient page corrections (2026-10-08)

Full CPU page run passed with stable inputs, runtime origins verified and no
surviving processes. Four610x276 figures use source coefficient coordinates,
periodic conversion and dotted jumps without artificial coefficient floors.
Eight actual output blocks replace stale text, including the duplicated and
incomplete cosine/sine display. Source prose/MATLAB blocks remain byte-identical.
Numerical parity remains open: abs(sin(x))^3 has length2513 vs historical3697,
cosine coefficient norm0 vs8.5e-17, and coefficient differences around2e-15.
Exact pixel/antialiasing parity also remains open.
Evidence: `/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/fourier_coefficients_root_acceptance_20261008.json`.

## Chebgui export cohorts and PDE parser correction (2026-10-08)

149 CPU checks passed with verified runtime origins, stable full guard inputs
and no surviving processes. All64 original demos export through BVP/IVP/EIG/PDE
adapters; fixtures match the pinned source byte-for-byte. This ports the four
previously missing export-only MATLAB test cohorts, which do not solve or run
the exported scripts.129 existing parser tests remain unchanged. PDE unary
negation now precedes infix simplification, matching the source prefix stage.
Literal demo loading and bounded numeric metadata parsing are supported; full
MATLAB eval/num2str/vectorize semantics and error-side partial file contents
remain open. No fresh MATLAB exporter output or solve/performance claim.
Evidence: `/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/chebgui_root_acceptance_20261008.json`.

## JAX PCHIP and complete source test assertions (2026-10-08)

61 CPU checks passed in40.15s pytest with stable source/environment inputs
and no surviving processes. JAX harmonic-slope Hermite interpolation replaces
SciPy in Chebfun.pchip, supports complex components and array orientation,
retains requested domain breaks and four coefficients per interval. All12
original MATLAB test assertions retain10eps norms and exact lengths/domains.
Independent SciPy comparisons, plateaus, nonuniform/extrapolated values and
existing spline regressions pass. No fresh MATLAB, full native-runtime closure
or isolated performance claim. General MATLAB input-cleaning/error behavior
remains unqualified beyond the finite data controls.
Evidence: `/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/pchip_root_v1_20261008/review.json`.

## Trigonometric CF page correction (2026-10-08)

Full CPU page run and figure audit passed with stable inputs, verified runtime
origins and no surviving processes. Source f-p/q error curve and600x269
reference layout restored. Hankel norm, CF bound and Remez error match historical
15-digit printed values. All7 output blocks come from the run; prose/MATLAB
blocks preserved. Elapsed output is a concurrent diagnostic, not a paired
performance comparison. Exact pixels and antialiasing remain unqualified.
Evidence: `/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/trigcf_root_acceptance_20261008.json`.

## Shared JAX spline and source spline tests (2026-10-08)

68 CPU checks passed:49 spline/source/derivative controls and19 unchanged
contour regressions, including cross0.01. Stable inputs, correct runtime origins,
no surviving processes. Shared JAX tridiagonal cubic interpolation replaces
SciPy in Chebfun.spline and contour fitting, retaining complex/array samples,
endpoint slopes, nonuniform knots, small-node conventions and requested domains.
All13 nonconstant MATLAB spline assertions retain their10eps norms, exact
lengths and domains. Continuous array differentiation no longer casts vectors
to scalar complex. Array-jump breakpoint infVals/pointValues and source warning
multiplicity remain unfinished; no full diff parity or performance claim.
Evidence: `/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/jax_spline_root_acceptance_20261008.json`.

## Chebyshev and phase portrait page corrections (2026-10-08)

Two serial CPU page runs completed with stable inputs, verified runtime origins,
no surviving processes and figure audits passing. Eleven figures regenerated;
nine reference-size mismatches removed. Chebyshev coefficient plots now reuse
the source helper without an artificial floor; function overlays use source
sampling. Two printed last digits changed by less than 1e-15; prose and MATLAB
blocks are unchanged. Phase portraits reuse the existing source color helper,
500-square sampling, alpha and subplot layout; source stdout is exactly empty.
Exact historical pixels, antialiasing and marker fidelity remain unqualified.
Evidence: `/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/cheb_phase_pages_root_acceptance_20261008.json`.

## Laguerre RHW source truncation (2026-10-08)

44 CPU checks passed (151.76 s pytest), stable source inputs and no surviving
owned processes. RHW now uses source capacity min(n,ceil(17*sqrt(n))), updates
initial-guess regions for that capacity, and stops before the first zero weight
following a positive one. The JAX kernel returns a live length; the eager API
slices it and normalizes only the returned rule. It does not solve the full
n-node rule before truncation. Barycentric signs use the returned node count.
Tests retain full-rule prefixes for alpha -0.5/0/0.3/0.5, exercise early stopping,
public barycentric/interval ordering, and degree10000/100000 moments/capacity,
plus all33 prior general-alpha controls. No isolated speed claim.
Evidence: `/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/laguerre_rhw_root_v2_20261008/review.json`.
Small explicit RH/RHW, RECW/EXP/EXPW, singular alpha=-1, traced-alpha dispatch,
and arbitrary-argument special-function qualification remain open.

## Spherefun constructor source dispatch (2026-10-08)

Integrated constructor changes passed 274 unique CPU checks, then 13 focused
checks against the committed compiled QR helper. All source bounds retained,
inputs stable, exact runtime origins, no surviving owned processes. Adds the
public `spherefun` factory for spherical/Cartesian callables, bounded string
expressions, numeric/coefficient inputs, fixed ranks/grids and preferences.
Restores empty-input precedence, numeric-scalar recursion and sampleTest
preference handling. Only the existing from_function/from_values methods change;
other sampling, QR, Helmholtz and contour fixes are preserved.
Evidence: `/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/sphere_constructor_root_acceptance_20261008.json`.
The frozen five vector regressions include the old Helmholtz test and do not
replace the separately verified corrected literal nine-case suite. No fresh
MATLAB or isolated performance claim; strings use a restricted arithmetic
parser rather than arbitrary MATLAB evaluation.

## Laguerre default dispatch ceiling removed (2026-10-08)

Nine CPU checks passed with stable inputs and no surviving processes. Default
n>=3000 now selects RH for every static alpha, exactly as the source dispatcher;
the artificial alpha10 ceiling and GW fallback are removed. Dispatch boundaries,
actual alpha12 moments/default-vs-explicit equality, and propagation of the
source alpha20.3 Newton failure are verified without relaxed bounds. This
supersedes the default-alpha>10 limitation recorded below. Arbitrary-order
Bessel accuracy, traced-alpha dispatch, other unported methods, and full
performance qualification remain open.
Evidence: `/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/laguerre_default_dispatch_root_20261008/review.json`.

## General-alpha Laguerre RH implementation (2026-10-08)

The final affected CPU gate passed 42 cases: general-alpha rule moments,
selected nodes/weights, independent Bessel controls, source-expression tables,
subnormal weights/barycentric/interval controls, and source Newton failure.
Earlier 46-case integration also passed, including unchanged MATLAB and
half-alpha controls. Inputs stable, exact runtime origins checked, no owned
process survivors. Evidence: `/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/laguerre_general42_v5_cpu_20261008/`.
General RH dispatch now covers the qualified default alpha range through 10;
explicit RH uses source stopping/error guards. Alpha 20.3 at n=3000 reaches
the source nine-iteration failure, also reproduced by independently interpreted
pinned expressions. Do not relax it or claim all finite alpha succeeds.
Subnormal source arithmetic stages are retained. New runtime numerical kernels
use JAX; fixed scalar Gauss128 tables replace runtime Python quadrature.
The coefficient tables are shared with the existing half-alpha driver.
Source-expression fixtures are interpreted references, not fresh MATLAB captures.
Default alpha above 10 retains the previous fallback and remains parity work;
arbitrary-order/argument special-function accuracy and isolated performance
remain unqualified. Hermite LAG documentation now reflects its implementation.

## Persistent sphere QR compilation (2026-10-08)

CPU correctness gate: 141 passed in 1193.52 s pytest / 1211.62 s supervised
wall time, no skips/failures or surviving owned processes, stable source
inputs. The only algorithm-file change is `@jax.jit` on
`real_trig_matrix_qr`; its function body is unchanged. Includes 128 existing
sphere controls, all nine literal/protocol Helmholtz checks, and four new
odd/even reconstruction/physical-measure controls with JIT on and off.
Evidence: `/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/sphere_qr_integration141_root_20261008/review.json`.
The gate binds source files and the pinned environment shell, not the full
native dependency closure. No isolated performance claim under parallel load.

## Marching Squares rendering checkpoint (2026-10-08)

The actual corrected CPU example completed and regenerated four 600x270 PNGs
with empty stdout. Source canvas, axes, tick and color audits pass. The reusable
bounded complex-curve plotting helper has six passing source controls. The
export helper adds optional DPI with its existing 100 default preserved.
Evidence: `/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/marchingsquares_v9_terminal_page_publication_review_20261008/review.json`.
Inputs stable, no owned process survivors. Import whitespace was normalized
for repository lint with exact AST equality to the tested files.
Full visual parity remains OPEN: Trott endpoint gaps are about 0.00403922,
smaller than the historical reference gaps; font/antialias differences remain.
No artificial closure or segment masking was added. Fresh prose and MATLAB
rendering qualification remain pending. These images are an improved verified
rendering checkpoint, not a claim that the page is fully matched.

## Scalar zero-curve source correction (2026-10-08)

125 CPU checks passed: 19 existing roots regressions, 79 numeric constructor
checks, and 27 spin checks, all with unchanged bounds. Restores source grid,
duplicate removal, endpoint snapping, six-step refinement, and last accepted
curve semantics. Reuses JAX paired Clenshaw for field slice evaluation; this
fixes the crossing-curve regression without changing contour connectivity.
Evidence: `/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/trott_root_acceptance_20261008.json`.
All reviewed hashes match, no surviving owned processes. Legacy skimage and
SciPy contour/spline adapters remain; full JAX-only and visual parity are open.
Corrected page rendering is a separate pending audit. No speed claim.

## Helmholtz source wrapper and literal tests (2026-10-08)

CPU gate: 9 passed (five wrapper controls and all four pinned MATLAB test
clauses), 1316.70 s pytest. Restores tangent projection, source warning bound,
rows-first Poisson dimensions, and two empty Spherefun outputs. Replaces the
old test port that omitted the third field and weakened the norm assertion.
Evidence: `/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/helmholtz_sampling_integration_root_20261008/review.json`.
Source hashes stable, no surviving owned processes. This gate binds the pinned
environment shell and source files; it is not a full native dependency audit
or an isolated performance measurement. Full parity and latest-main CI remain
open; preceding local commits have not been pushed from this restricted session.

## Sphere sampling and vscale source semantics (2026-10-08)

Spherefun sampling now aliases Fourier coefficients before evaluation, retains
source real-factor extraction and the nonconjugating CDR product, and uses the
source singleton phase and physical-grid selection. vscale samples each factor
length clamped to9..2000 and takes the source maximum magnitude. Original
missing/nonpositive dimension checks and source identifiers are preserved;
the empty three-output error remains a documented Python adapter.

91 CPU cases passed:43 source controls plus48 unchanged sampling regressions,
exact collection/JUnit/runtime origins, stable inputs and no surviving children.
The two production files match the tested snapshot; test import whitespace is
the only post-test formatting change. Qualification was concurrent correctness
work, not a performance comparison. Coefficient-only storage still reconstructs
values where MATLAB may retain stored samples, so bitwise parity is not claimed.

Evidence: sphere_sample91_terminal_review_20261008_oe7bpzzx/review.json.
Verified archive sphere_sample_vscale91_reference_20261008.zip SHA256
ad99b4bd94be15126af71fe53e5f62c48f1d91bddb66b111c0bddf0cddc1663c
requires the full-source base archive documented below. Constructor-V2 changes
are separate and require method-level integration and combined tests.

## Hermite fresh-process JIT import correction (2026-10-08)

An additional existing fresh-process regression exposed six module-level uint64
masks retaining tracers when ASY helpers are first imported inside jax.jit.
Their values and arithmetic are unchanged; ensure_compile_time_eval now
materializes the constants outside the enclosing trace. Independent AST review
confirmed the initializer wrappers are the only algorithmic-source change.

The original installed80 cases plus three existing independent ASY comparisons
and the existing fresh-subprocess import/leak regression now pass together:
84 passed in63.66s, stable source guards and no surviving children. Evidence:
hermite_outer_jit_regression_root_v3_20261008/review.json under the shared20261005
root. The preceding3pass/1fail run is preserved as v2; the initial launcher
failed collection because its supervisor working directory was omitted.
This additional gate uses the pinned environment shell; the earlier80/high-n
archive retains its full dependency closure evidence. Full parity/CI remain open.

## Hermite gradual-underflow and source normalization (2026-10-07)

Hermite asymptotic weights and barycentric factors retain representable tiny
binary64 values through the source separately rounded normalization stages.
JAX bit-level arithmetic handles gradual underflow; mapped division explicitly
broadcasts before rounding, with an explicit JVP rule. Source odd-center
behavior is preserved rather than replaced by an exact-zero assertion.

The installed-library test package passed 80 CPU cases (64.32s), including
18 unchanged MATLAB-port cases, exact rounding controls, mapped division and
public normalization. Collection, JUnit and runtime-origin checks agree; input
hashes were stable and no owned processes survived. Four source-equivalent
high-degree runs (default and ASY at n=10,000 and 100,000) passed finite/order/
symmetry and original second-moment checks, with no captured MATLAB nonzero
weights or barycentric values lost. Default and ASY outputs match at each size.

Maximum absolute differences from captured MATLAB at n=100,000 are
2.984e-13 for nodes, 8.344e-16 for weights and 9.637e-14 for barycentric values.
These are numerical comparisons, not bitwise MATLAB parity. Timings are
unpaired, and the first high-degree arm overlapped other work; no isolated
speed claim is made. Subnormal automatic differentiation remains unqualified.

Evidence in the shared 20261005 root: hermite_v9_root_resume_cpu_20261007,
hermite_high_n_integrated_v8_cpu_20261007 and
hermite_v9_combined_archive_context_20261007_2xt6bmcg. Verified archive
hermite80_highn_reference_v9_20261007.zip SHA256
e7b735b0e1cdd17cf4ce6a0750b9a537ede4369c2b5c3f00a607586d1fb5fe58
requires the full-source base archive identified below. Full library/example
parity and latest-main green CI remain open.

## Numeric constructor complex infinity follow-up (2026-10-07)

The JAX numeric constructor now preserves MATLAB magnitude infinity precedence
when one complex component is infinite and the other is NaN. The predicate also
retains magnitude overflow from finite components. Scalar rejection and inverse
CDR weight masking share this predicate; NaN without infinity stays unmasked.

Qualification: 77 constructor cases and 27 unchanged spin consumers passed on
the exact V4 numerical source. Two original direct zero-factor storage controls
were retained and separately passed against that source, giving 79 constructor
cases in the published module. Inputs were stable, runtime origin checks passed,
and supervisors reported no survivors. Test bounds and captured fixtures are
unchanged. These are separate correctness cohorts, not performance measurements.

Evidence under the shared 20261005 root: root_operator_chebfun2_constructor77_v1_cpu_20261007,
root_operator_chebfun2_spin27_v1_cpu_20261007,
constructor_restored2_root_resume_cpu_20261007, and
helmholtz_remaining_root_resume_review_20261007/terminal4.json.
The combined archive context is chebfun2_v5_combined_archive_context_20261007_2rcsdyjs.
Verified archive: chebfun2_numeric106_source_reference_v5_20261007.zip, SHA256
ab9b7cfb2ea6f5b8096e44f8ee319191a45b3236a1c8d833e4d1ee6d13b7a643.
It requires the same full-source base archive identified below.
Broader complex/extreme scalar parity, full API coverage and latest-main green
CI remain unresolved. This section supersedes the complex-infinity exclusion
in the historical baseline entry below.

## Numeric Chebfun2 construction uses JAX (2026-10-07)

The public numeric from_values path now uses source-ordered JAX ACA and factor
transforms: column-major pivot ties, selected-diagonal stopping norm, numeric
rank cap, row division before outer product, source factor lengths, source grids,
and real zero storage in the no-pivot branch. It reuses shared Chebtech2
extrapolation for missing rows. Scalar recursion preserves the source default
Chebyshev technology, including a requested trig scalar input. Default numeric
chopping is off to preserve source nonadaptive factor lengths; explicit chop=True
remains a Python adapter. Adaptive callable construction is unchanged.

CPU qualification:77 constructor checks and27 unchanged spin consumers passed
in separate fresh processes (117.50s and74.86s). These include32 actual MATLAB
public constructor captures in both JIT modes and the existing constructor
aggregate. Exact collection/JUnit/runtime checks passed, inputs stayed stable,
and no descendants survived. The preceding failed gate is preserved:63pass,
12fail, caused by empty fixture decoding and complex no-pivot zero storage;
the correction keeps original assertions and fixture bytes unchanged.

This is eager construction with JAX arithmetic, not an outer-jit API. Inherited
host evaluation and adaptive paths still contain NumPy. General complex/extreme
scalar behavior, complex Inf/NaN precedence, MATLAB RNG identity, full consumer
coverage and full source parity remain open. Captured comparisons use explicitly
additional64eps rounding diagnostics; original MATLAB bounds were not widened.

Evidence root: `/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/`.
Archive chebfun2_numeric104_source_reference_v3_20261007.zip SHA256
6d4cfa2da49a9ca83736206058d1f48efcc0299d622a88d84811f244d3dd92ed
requires corrected_runtime_union_v3_full_source_20261006.zip SHA256
1e7c77153df60503d57a940869e2922174db86f7b55ab633d16a6225be2d90a7.
The separately developed complex-infinity predicate is excluded from this
qualification and publication. Latest-main green CI is still required.


## Sphere constructor tolerance and CF padding (2026-10-07)

Spherefun callable construction now propagates the source grid-dependent
absolute tolerance into the small-pivot check and final simplification. Numeric
zero samples preserve source factor dimensions; missing pole samples raise the
source identifier. The exact-zero classifier uses JAX binary64 bits so CPU
flush-to-zero cannot mistake a nonzero subnormal for zero. General nonzero
subnormal construction parity is not established.

Qualification: 53 constructor checks, 128 combined sphere regressions and five
sphere-vector regressions passed in three separate CPU cohorts. Seven literal
constructor clauses (1, 2, 8, 9, 16, 26, 30) are published with unchanged bodies,
inputs and bounds. The full 30-clause draft remains preserved in the evidence
packet; the other 23 clauses and their missing public overloads remain open.
The publication subset is checked by AST against the qualified selected cases.

The JAX CF Clenshaw-Lord fallback now pads short coefficient vectors with
source epsilon-scaled real normal draws before c0 scaling, and handles n=0.
Padding shares randnfun's advancing default stream; explicit keys/seeds bypass
that stream. This does not reproduce MATLAB RNG bits. Forty checks passed,
including six matched-draw comparisons from three actual MATLAB public captures,
15 stream/branch controls and the preceding 19 CF/minimax checks. Public CF
reachability of insufficient padding, full ChebPade migration, large Hankel
qualification and the separate strict coefficient diagnostic remain open.

All four Python cohorts had stable inputs, exact collection/JUnit reconciliation
and no surviving processes. Their 226 passes are not a combined runtime claim.
Evidence root: `/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/`.
Archive constructor_cf226_reference_v1_20261007.zip SHA256
34f5c9796827eb866c51a7637204dc695f6cc93e5bca6679fb4d122037074afb
requires corrected_runtime_union_v3_full_source_20261006.zip SHA256
1e7c77153df60503d57a940869e2922174db86f7b55ab633d16a6225be2d90a7.
Publication copy/subset evidence: constructor_cf_publication_review_p130_20261007.
Latest-main green CI and full function/test/page/figure/speed parity remain open.


## Spin2 public outputs, preferences and requested times (2026-10-07)

The public operators.spinop2.spin2 route now returns actual Chebfun2 objects and
an explicitly bounded variables-by-times function-matrix adapter. It forwards
SpinPref2/name-value preferences, preserves persistent multistep history and
collects requested times with the source cursor rules. Source default dealiasing
is off; enabled dealiasing changes saved outputs, not next-step history.
The source rectangular-domain Fourier symbol uses the x width for both axes;
this literal behavior is retained and covered rather than silently corrected.

CPU qualification:27 passed in49.85s with stable inputs, exact collection/JUnit/
runtime origins and no survivors. Sixteen cases compare eight actual MATLAB
captures in both JIT modes; further controls cover preferences, masking and the
source symbol. Unchanged nonlinear Gray-Scott ETDRK4/ABNorsett4 reference tests,
three custom-operator tests and the surface interpolation test also pass.
MATLAB capture confirms that a missed requested time blocks subsequent collection
and public spin2 rejects three outputs while direct solvepde supports three.
Requested times are collected without interpolation or interval restarts.

New stepping and time/output-selection arithmetic is JAX. The inherited
Chebfun2.from_values numeric constructor still uses NumPy ACA/Trigtech paths;
therefore the entire public route is not yet JAX-only. Full Chebmatrix algebra,
graphics, named-call defaults, system initial-input surface, source random streams
and broader solver/performance qualification remain open. The legacy low-level
array spin2 and other dimensional solvers are unchanged.

Evidence root: `/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/`.
Archive spin2_public27_source_reference_v1_20261007.zip SHA256
1efbfa267419dc0024badcbd5672b6aea276cd59d6da15e6452b989f34449f94
includes the actual MATLAB capture and qualified Python run with their separate
guards/environments; its archival input union is not a combined-runtime claim.
It requires corrected_runtime_union_v3_full_source_20261006.zip SHA256
1e7c77153df60503d57a940869e2922174db86f7b55ab633d16a6225be2d90a7.
Latest-main green CI and full library/test/page/figure parity remain unresolved.

## Shared JAX randnfun construction (2026-10-07)

The source-shaped top-level, utils.random and chebfun1d.randfuns entry points now
share one JAX engine, including periodic/nonperiodic normalization, full-coefficient
endpoint values, common-column chopping, NaN defaults and zero-count/Inf ordering.
Explicit keys/seeds are repeatable; no-key calls advance an entropy-seeded stream.
The explicitly legacy utils.randnfun convenience defaults are retained.
NumPy random.seed no longer controls the production randnfun stream.

CPU qualification:49 passed in249.84s, stable inputs, exact collection/JUnit/runtime
origins and no survivors. This covers44 construction/API controls (both JIT modes),
the original23 predicates as one aggregate, three convenience tests and an actual
inpainting consumer. The aggregate deliberately injects its historical primitive
normal draws; its assertions are unchanged. It does not establish MATLAB RNG-stream
identity. Separate public tests exercise actual JAX key/seed/default behavior.

Historical random example pages still need stream migration and actual reruns.
Malformed-input identifiers, empty-domain metadata and MATLAB normal-transform
identity remain open; returned-object consumers retain their existing backends.
No whole-construction JIT or full random-example parity claim is made.

Evidence root: `/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/`.
Archive randnfun_public49_reference_v4_20261007.zip SHA256
9bb8ff45b583971dfd92215e8571dd182a13676334c99b9f7d98ed706b368b70
requires corrected_runtime_union_v3_full_source_20261006.zip SHA256
1e7c77153df60503d57a940869e2922174db86f7b55ab633d16a6225be2d90a7.
Fresh CI and the full library/test/page/figure/performance goal remain unresolved.

## CF public source behavior and minimax consumer (2026-10-07)

CF public source package: 18 literal/API tests and separate exact minimax clause2 consumer passed, guards stable/no survivors. All eight pinned MATLAB test_cf clauses retain original inputs/norms/bounds; public orientation, complex warning, endpoint and p/q contracts covered. New methods contain no oracle answer arrays or source eigenpair substitutions.

The additional strict16 coefficient-golden diagnostic remains15passed/1failed (exp(4,3)); its assertions, fixtures and bound are unchanged. It is not a pinned MATLAB source assertion. Same-input eigenswap isolates the coefficient discrepancy mainly to the eigenvector; high-precision diagnosis does not justify degrading the native eigenpair. Exact Fraction certificate bounds the stored native/source rational ratios uniformly by <2.496002668e-15 on[-1,1], not general exp error or floating public evaluation. Functional/public acceptance is documented separately from unresolved coefficient matching.

Large Hankel>1024 LOBPCG route is enabled by existing dispatch but remains unqualified; documentation now says so. Other explicit unsupported branches, non-Chebyshev/preferences and historical BestApprox page scope remain open. No full-CF parity or performance claim. Consumer preserves exp(sin(exp(x))),degree7,continuousL2<0.0003; it is one clause, not the full minimax aggregate.

Evidence root: `/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/`.
CF18 archive: cf_public18_literal_api_reference_v2_20261007.zip SHA256
5b0fbad8496d40907e389b37fc5f1e7a54e231afc877032d6dc0d4630a29e93f.
Consumer and publication archive: cf_consumer_publication_reference_v1_20261007.zip
SHA256 3f9a0bcc9b6c0041db8a38c0ee084ba4bcbf2ce7ae143e6f6d6c3bf288ab2505.
Both require full base corrected_runtime_union_v3_full_source_20261006.zip SHA256
1e7c77153df60503d57a940869e2922174db86f7b55ab633d16a6225be2d90a7.
Publication removes stale prototype wording in two docstrings; normalized AST
comparison confirms unchanged executable code. The amendment and explicit
unsupported-branch audit are in cf_publication_doc_amendment_v1_20261007.
Missing random-padding Cheb-Pade fallback and large-degree solver qualification
remain concrete follow-up work. Full parity and fresh CI are not established.

## Sphere numerical rank, singular functions and compilation reuse (2026-10-07)

Spherefun now exposes source spectral rank as numerical_rank(tol), retaining the
Python rank property as stored factor count. Singular-function SVD uses JAX
weighted QR and the source transform sequence; the default values-only path is
unchanged. Complex longitude outputs preserve MATLAB's convention and do not
promise arbitrary complex U*S*V' reconstruction. Persistent JAX staging of the
existing real Horner and sample FFT kernels reuses compiled functions without
changing sampling, aliasing, shapes or zero predicates.

CPU qualification:128 combined rank/SVD/arithmetic/cache tests passed; the five
unchanged sphere-vector regressions separately passed in946.29s total. Helmholtz
was810.255s on this machine. This is not a paired performance comparison or proof
of the CI900s limit on CI hardware. Both runs reconcile exact collection/JUnit,
runtime origins, stable inputs and no surviving processes. The separately
qualified literal vectorRelations replacement restores all three source clauses,
continuous norms and original bounds, removing weakened assertions.

Evidence root: `/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/`.
Archives:
- sphere_union128_reference_v3_20261007.zip SHA256
  b4e2c1c083eb2a5867c15ec015010cc635b5a910d4a4a63664f479e07efccf65
- sphere_union_vector5_reference_v1_20261007.zip SHA256
  0a514cbdaf520e5ffff0b235143b2d745232e263508ca4db14c0db4161135119
- sphere_vector_relations_literal_reference_v1_20261007.zip SHA256
  29da0cea3e72a9806abc4fdee89ad21b3cbef1879262fc16bac7a87276339570
All require corrected_runtime_union_v3_full_source_20261006.zip SHA256
1e7c77153df60503d57a940869e2922174db86f7b55ab633d16a6225be2d90a7.

Full source-clause coverage, literal Helmholtz coverage, constructor overloads,
legacy NumPy consumers, BMCsvd, other pages/figures and latest-main green CI remain
open. The earlier iszero-only vector run's bytecode guard failure remains a failed
gate; this fresh combined run passed its complete guard. No unrun Rotate changes
or constructor drafts are included here.

## Localization page and reference-size figures (2026-10-07)

The standalone Localization script now follows the recovered 2016 publisher
geometry: both figures are610x276 at5630 pixels/metre, with source line/marker
sizes, axes/ticks and unclipped boundary markers. All six printed-output blocks
match the cached chebfun.org page exactly. The two plotted pivot arrays match
MATLAB source captures bit-for-bit (field cases2/4, ranks14/17).

The final CPU page run and independent artifact audit passed:60.58s, sampled
summed RSS1,095,976KiB, stable runtime inputs and no surviving processes.
Generated Markdown is unchanged. Renderer-specific antialiasing differs from
MATLAB; scientific data, artists, labels, layout and reference sizes were checked.
The publication script changes only two checkpoint metadata keys after the run
(`source_head` to `qualification_baseline_head`); normalized AST comparison
confirms numerical and plot code is unchanged. The old home-checkout edits are
preserved; this verified script and freshly generated figures supersede its
Localization candidate in shared main.

Evidence root: `/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/`.
Full-closure archive `localization_v8_reference_v2_20261007.zip`, SHA256
`bd210fc74055297900aa8a1d25dc4975c2c482471d6624780f78329165c063d7`,
contains53,379 payloads and requires verified full base
`corrected_runtime_union_v3_full_source_20261006.zip`, SHA256
`1e7c77153df60503d57a940869e2922174db86f7b55ab633d16a6225be2d90a7`.
The earlier246-payload V1 archive is partial-scope and is not the full closure.
Other pages, literal source tests, performance and latest-main green CI remain
open; P124 sphere-vector CI had40passes and one900s Helmholtz timeout.

## Sphere rank-one multiplication regression repair (2026-10-07)

Rank-one multiplication now follows pinned spherefun/times.m: masked CDR scaling
is distributed into column/row factors, right-operand pivots and locations survive,
pole flags are ANDed, and parity switches when the left plus group is empty.
The previous helper dropped pole metadata; literal addition then projected valid
polar terms away. Exact P123 curl and arithmetic CI failures were reproduced,
along with six independent metadata failures, before the repair.

CPU qualification: **90 passed in464.08s**, no errors/skips, peak sampled summed
RSS5,569,740KiB. This includes the unchanged prior69 addition/CDR/default-SVD cases,
two exact failing CI tests, six metadata controls, twelve source CDR scaling controls,
and all three literal MATLAB multiplication clauses. Collection/JUnit/runtime
origins reconcile; source/runtime inputs stayed unchanged and no processes survived.
Full configured Ruff, F821 and repository NumPy/provenance presence policies pass.
The new scaling arithmetic is JAX; inherited Trigtech NumPy transform delegation
remains unresolved and is being replaced in a separate, unqualified draft.

Five additional P123 sphere-vector CI failures (Helmholtz, tangent/normal and three
vorticity cases) are under a separate serial gate on these exact library bytes.
Their cause and resolution are not yet established by the90-case result. Fresh
CI on this commit is required. A source coverage audit also found weakened or
missing vector assertions, including an unconditional `or True`; those gaps remain
open and passing existing tests does not establish full source parity.

Evidence root: `/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/`.
Verified archive `sphere_rank1_times_repair90_reference_v1_20261007.zip`, SHA256
`93445de8152450882afa3aedf4a46798ea048c38c6d5c2f9d9842af595bb78bd`.
Failed baseline archive `sphere_p123_product_regression8_failure_reference_v1_20261007.zip`,
SHA256 `797a2ebd1caaa196c7ae6c4f38841483bbce5b6b636a25140ccc3cb8220b4dc6`.
Both require full base `corrected_runtime_union_v3_full_source_20261006.zip`, SHA256
`1e7c77153df60503d57a940869e2922174db86f7b55ab633d16a6225be2d90a7`.
Full library/test/page/figure/performance parity and latest-main green CI remain open.

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
