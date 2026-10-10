# Parity status — 2026-10-09

## Latest CPU qualification and open failures (2026-10-09)

Local implementation head 6b5a7101 includes two newly qualified packages:

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
remain unqualified. The next candidate extracts only the unchanged recurrence
into a persistent compiled helper. Shared evidence: sphere_heat_public_source_20261009/
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

**Full parity is incomplete.** The target is MATLAB Chebfun commit
`7574c77680d7e82b79626300bf255498271a72df`, all native functions/tests, and
all 322 example pages, including computations, prose, outputs and figures.
Work is CPU-only. [HANDOFF_CODEX.md](HANDOFF_CODEX.md) takes precedence over
the July handoff. No defensible overall completion percentage is available.

## Latest verified changes

| Local commit | Change | Qualification |
| --- | --- | --- |
| `17fb15c6` | AtmosphericTemperature captured cameras and contour framing | Full source replay plus isolated corrected figure 3; all 10 images; historical graphics/scientific-reference differences remain |
| `cf3cde14` | Ballfun default source slices and coefficient emptiness | 27 distinct checks across staged runs plus a normal-pytest artifact check; native lighting remains open |
| `f5d7d450` | SolidHarmonics full source page | All 13 constructions, including degree 150; 11 public plots/55 surfaces; two 600×253 figures; lighting/layout differ |
| `510b3f8a` | WaveDecay adaptive eigensolves and continuous normalization | Full source run, two 600×480 figures and eight matching formatted eigenvalue labels; solver/rendering parity remains open |

These are scoped qualifications, not a complete passing suite or exact-head CI.

## Remaining gates

- **Tests and functions:** complete semantic coverage and full CPU suite.
  Exact-commit inventory at `ad02a54d` maps all 1,102 native test files:
  1,091 have no literal skip/xfail marker; 11 contain markers. File presence
  is not complete predicate coverage or a passing execution result.
- **Figures:** fresh audit at `510b3f8a` checks 1,305 mapped image pairs with
  no mapping holes. **71 dimension mismatches remain across 14 pages.**
  Correct dimensions do not establish matching content, colors or rendering.
- **Correctness:** trigonometric constructors and mixed-column realness,
  rational poles, Hermite edge cases, eigensolver semantics, difficult
  examples and the remaining source-linked handoff backlog.
- **Performance:** inverse-function and other slow paths need qualification;
  no full MATLAB performance-parity claim is supported.
- **Implementation:** remaining host numerical paths must become JAX-only.
- **Acceptance:** native MATLAB comparisons, publication and final green CI.
  Latest observed session failures are GitHub network/SSH access and MATLAB
  IPC startup. Last confirmed remote main was `8a6d9c80`; newer listed commits
  are local. Useful local implementation and verification continue.

## Current work

The C2 eigensolver package restores all four original basic predicates and
passes 14 controls plus four affected consumer/dispatch checks. Its inherited
host eigendecomposition and other backend/adapter semantics remain open.
Solid harmonics pass all 14 original predicates and 25 controls, with a fresh
full-page run including degree 150. New recurrence and radial contraction code
is JAX; inherited Ballfun host numerics and rendering differences remain.
The shared Trigtech constructor/composition core passes 82 distinct checks,
including nine original constructor and eleven original composition clauses.

The shared JAX Chebtech extrema policy passes all 16 original C1/C2 predicates
and 12 focused controls; empty-sentinel and nonfinite edge cases remain open.

The private JAX two-dimensional Nelder–Mead fallback passes 12 controls.
Public optimizer dispatch and native active-set parity remain unfinished.

C1/C2 fixed-zero callback/population and empty vscale pass 14 source controls;
public retained-empty-FUN behavior is part of the ongoing constructor work.

Active work: full public constructor context and callback/endpoint semantics;
JAX Ballfun norms and triple integration with native dimensions and Nyquist
handling; Gibbs2D continuous extrema and public constructor dependencies.
These scoped qualifications are not full-suite or exact-head CI evidence.
Resource-heavy examples run serially; independent bounded work uses up to
three worker slots. Work remains CPU-only.

## Evidence

Immutable run receipts, runtime hashes and reviews are under:
`/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/`

- `atmospheric_composite_root_integration_20261009.json`
- `ball_plot_composite_root_runtime_review_20261009.json`
- `ball_plot_ci_fallback_root_runtime_review_20261009.json`
- `solid_harmonics_root_integration_20261009.json`
- `wavedecay_root_integration_20261009.json`
- `matlab_test_static_inventory_1bacf226_20261009.json`
- `figure_size_current_510b3f8a_20261009.json`

[PARITY_MATRIX.md](PARITY_MATRIX.md) and the handoff retain dated historical
qualification records. The old initial-port phase labels and percentage in this
file did not establish MATLAB parity; their historical inventory remains in git.

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
