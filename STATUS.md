## Native inverse degree and cotangent branches — 2026-10-14

TreeVar acosd/asind/atand/acot now use JAX with captured branch/storage/pole
behavior and componentwise angular scaling. Baseline53/144 native cases fail;
root250 merged boundary, original unary and JIT/JVP/RHS checks now pass24.79s.
All144 compact records independently match the retained native packet. Review
caught an initial raw-atan JIT regression; the successor preserves that typed
primitive and qualifies the new value-dependent adapters as eager. Opaque
builtin internals, acotd and extreme reciprocal underflow remain unqualified.
Evidence: `docs/treevar_inverse_degree_cpu_20261014.json`.

## Public spin3 JAX integration and native time outputs — 2026-10-14

Public scalar/system stepping now uses the shared JAX exponential integrator,
with continuous checkpoint collection, native output-only cube dealiasing and
numeric Chebfun3 results. Multiple times/systems return an indexed function
matrix; return_times exposes saved times. Seven native cases agree within
2.61e-15; root independently verified all fixture fields. Root27 merged tests
pass48.85s, stable inputs/no survivors; original GL error .000114705<.01.
The q6 startup end case is captured natively and replaces an explicitly
source-inconsistent supplemental assertion; four native scheme tests unchanged.
Named-PDE/default-argument dispatch, full Chebmatrix algebra, movie output and
separate low-level solver host paths remain open.
Evidence: `docs/spin3_public_jax_cpu_20261014.json`.

## Scalar preference reset and public Chebfun3 zero semantics — 2026-10-14

Scalar numeric construction now drops call-local technology/tolerance and uses
session defaults, as confirmed by native preference/coefficient captures.
Fixed-length scalar callbacks follow the same route. Public iszero now checks
zero factors after its core/sample checks; rank retains native core-only
semantics and length uses full iszero. Three zero-factor baseline failures
are fixed. Root98 merged checks plus all4 literal original iszero predicates
pass; raw native MAT and driver are retained. Equi remains unimplemented.
Evidence: `docs/chebfun3_scalar_zero_cpu_20261014.json`.

## Native inverse reciprocal hyperbolic branches — 2026-10-14

ACOTH/ACSCH/ASECH now use JAX reciprocal and inverse-function compositions
with native real-axis/storage behavior. Baseline44/108 captured checks failed;
all108 native controls,49 original unary and6 precision/storage checks pass.
Root178 merged checks pass and all108 fixture records match the raw capture.
Native ASECH(-0)=NaN is retained explicitly. Builtin internals remain opaque;
these are qualified mathematical compositions with eager storage selection,
not blanket JIT or extreme-reciprocal qualification.
Evidence: `docs/treevar_reciprocal_hyperbolic_cpu_20261014.json`.

## Native logarithm branches and JAX magnitude accuracy — 2026-10-14

TreeVar log/log1p/log2/log10 now use JAX with native exact-axis branch and
real-storage behavior. Baseline47/144 captured boundary checks failed;
144 native,49 original unary and20 independent Decimal controls now pass.
Root228 merged checks pass; all144 fixture records independently match raw
MATLAB capture. A compensated log1p identity fixes an additional accuracy
error found by the independent controls. Base logarithms use polar magnitude;
this is a qualified mathematical implementation, not inspected MATLAB builtin
code. Storage selection remains eager; generic JIT/global Chebfun branches
remain separate. Evidence: `docs/treevar_log_family_cpu_20261014.json`.

## SH source random tensor initialization — 2026-10-14

SH now uses the shared advancing JAX normal stream for the source0.1*32³
random tensor, followed by numeric periodic Chebfun3 construction. Source
parameters, grid recovery, real representation and a short solve pass in
both private and public-import gates. MATLAB normal-stream identity and
full SH evolution remain unresolved.
Evidence: `docs/spinop3_sh_random_cpu_20261014.json`.

## Original Chebfun3 constructor syntax predicates — 2026-10-14

Added the original callback/string/domain/lower-variable-dependency cases,
rank/domain clauses8–13, and existing-object preference clause14 using the
source sin(x*y*z) object on[-2,1]^3. Nine isolated CPU cases pass with
unchanged source predicates and stable inputs. Python keyword adapters are
explicit; native split-domain positional clause1 and positional character
selectors remain unported. Evidence: `docs/chebfun3_constructor_syntax_cpu_20261014.json`.

## Native spin3 numeric output and Nyquist correction — 2026-10-14

Public spin3 now returns native numeric periodic Chebfun3 outputs through
the shared component helper. This removes the host FFT callback, whose
unsplit Nyquist mode produced spurious imaginary values for a real field
(off-grid error0.99588 in the focused control). Both baseline failures are
fixed;11 merged solver/output checks pass, including original GL convergence
0.000114705<0.01. The system container and inherited solver host paths remain
separate gaps. Evidence: `docs/spin3_native_output_cpu_20261014.json`.

## GL3 random initial field and native dealiasing default — 2026-10-14

GL3 now constructs the source0.1*randn(32,32,32) numeric trig field using
the advancing JAX normal stream. This exposed a spin3 default mismatch:
native dealiasing is off, while Python had enabled it. Corrected both public
scalar/system adapters; explicit on remains available. Same-input original
convergence error fell from0.0252037 to0.000114790 against unchanged0.01.
Nine combined constructor, routing and original convergence checks pass.
MATLAB normal-stream/full-evolution parity, SH preset and low-level separate
solver API remain outside this qualification.
Evidence: `docs/spinop3_gl_random_dealias_cpu_20261014.json`.

## Fixed-length Chebfun3 callback construction — 2026-10-14

Added n=(nx,ny,nz), sampling a callable once on the native second-kind
tensor grid before numeric construction. Original syntax clauses3–7 exact
length/domain predicates and six invalid-length controls pass; ten checks
total. Python keyword order uses the final domain, so MATLAB positional
length-before-domain sampling behavior remains an explicit parser difference.
Evidence: `docs/chebfun3_fixed_lengths_cpu_20261014.json`.

## Quasimatrix numeric replacement domains — 2026-10-14

Numeric replacement columns now use only the original domain endpoints, as
in native assignColumns; logical values are rejected by the numeric branch.
Five assignment groups pass, including piecewise row/column domain,
orientation and value controls. Published source/tests exactly match the
qualified private candidate. Singleton wrapper differences remain explicit.
Evidence: `docs/quasimatrix_numeric_endpoints_cpu_20261014.json`.

## Numeric Chebfun3 source constructor — 2026-10-14

Fixed numeric tensor construction now follows native complete ACA and block-term
compression using JAX. Trig tensors are supported and complex values retain
their imaginary parts. Eight native fixtures match ranks and stored lengths;
dense32 grid difference is1.8541e-14. All72 fixture array fields independently
match the raw MATLAB captures. Coarse-tolerance zero results retain stored
factor lengths while reporting native public rank0 and length1. Private25
and merged69 numeric/plus checks pass with stable inputs and no survivors.
Exact-zero native constructor crashes and scalar requested-tech differences
remain explicit compatibility gaps; C1 numeric fields are not yet qualified.
Evidence: `docs/chebfun3_numeric_tensor_cpu_20261014.json`.

## Genuine Quasimatrix column assignment — 2026-10-14

Added column replacement, repeated indices, deletion and zero-padded growth
with native error identifiers and row orientation. All15 original k=2
assignment clauses pass with captured native queries and unchanged bounds;
28 merged assignment/division checks pass (50.56seconds, CPU only).
Python retains its singleton Quasimatrix wrapper convention; native singleton
isQuasi metadata and general numeric shape parity remain separate gaps.
Evidence: `docs/quasimatrix_assign_columns_cpu_20261014.json`.

## Original array-valued assignment predicates and native RNG — 2026-10-14

Restored original array-valued assignColumns clauses1–15 across the mapped and
focused modules: exact native seed6178/continuation probes, source breakpoint
layouts, exact equality, native error identifiers, and literal deletion norm.
Unbounded clause13 now uses the original100x3 matrix infinity norm, with its
unchanged100*vscale*eps bound. Root14 finite/adapter checks and separate literal
14 and unbounded13 checks pass. Native1–13 passes for both loop kinds; all10
extracted fixture fields independently checked against retained raw MAT files.
The genuine Python Quasimatrix assignment loop remains separate work.
Evidence: `docs/assign_columns_native_predicates_cpu_20261014.json`.

## Complex column assignment and native quasimatrix collation — 2026-10-14

Numeric column assignment now preserves complex dtype through the JAX constant
constructor. The shared quasi2cheb helper unifies and restricts breakpoints
before concatenation, including native periodic-to-regular conversion; singleton
inputs remain unchanged. Literal array-valued assignment pass15 and complex
partial/full-row controls pass, together with division regressions. Private11
and merged23 checks pass with stable inputs/no survivors. Original bounds are
unchanged. Source/RNG fidelity gaps in the older assignment module and genuine
Quasimatrix assignment remain open, explicitly audited separately.
Evidence: `docs/assign_columns_quasi2cheb_cpu_20261014.json`.

## Native TreeVar sqrt signed-zero and storage branches — 2026-10-14

Sqrt now follows the native positive imaginary branch on the negative real
axis for either imaginary signed zero, normalizes zero, and retains native
real storage when the result has zero imaginary part. Baseline10/38 boundary
checks failed; private108 and merged211 checks pass after correction.
Root independently verified36 compact records against the retained native
capture. Original49 unary and98 first-order assertions/bounds are unchanged.
Dynamic dtype selection remains eager; log and global Chebfun branch behavior
remain separately unqualified. Evidence: `docs/treevar_sqrt_cpu_20261014.json`.

## Native TreeVar atanh branch and real-kernel accuracy — 2026-10-14

Atanh now uses native real-axis branches and handles pure-imaginary values,
signed zeros and nonfinite classifications from retained native captures.
A standard single-log1p identity also fixes a real-kernel accuracy failure;
the initial failing run and independent high-precision diagnosis are retained.
Private118 and merged173 checks pass, including unchanged original unary and
first-order bounds. Root independently verified all36 fixture records against
raw native capture. Dtype selection remains eager; reciprocal wrappers and
other branch cases remain separate. Evidence: `docs/treevar_atanh_cpu_20261014.json`.

## Direct column extraction metadata and bounds — 2026-10-14

Direct extract_columns now handles empty selectors, retains row orientation and
scalar stored point values, and rejects out-of-range indices instead of JAX's
silent clipping. Python zero-based and valid negative indexing remain supported;
noninteger/boolean selectors are rejected. Nine merged original/adapter tests
pass; final exact-equivalence indexing regression also passes separately.
No original bounds changed. Evidence: `docs/extract_columns_bounds_cpu_20261014.json`.

## Native inverse-trig branches and GL preset initialization — 2026-10-14

TreeVar asin/acos now follow native real-axis branches, including signed zero
and nonfinite cases. Private146 and merged125 checks pass; root independently
verified all72 compact fixture records against retained native capture. Original
49 unary and98 first-order assertions/bounds are unchanged. Dtype-changing
paths remain eager; degree wrappers and other inverse functions remain separate.
Evidence: `docs/treevar_asin_acos_cpu_20261014.json`.

The GL preset now constructs a real periodic randnfun2(4, domain, trig=True)
object and normalizes its infinity norm, matching native construction order.
Constructor, short solver and unchanged original convergence gates each pass
in separate bounded CPU processes. Source/test hashes match reviewed candidate;
convergence uses one shared initial object for both time steps. The test's stale
deterministic-initialization description is corrected. MATLAB normal-stream
identity and full GL evolution/raster parity remain open.
Evidence: `docs/gl_preset_random_init_cpu_20261014.json`.

## Restore native derivative probes and literal unbounded callback — 2026-10-14

Derivative tests now use captured native seed7681 probes and actual sequential
seed6178 draws for singular/unbounded clauses10–11. Removed the test's manual
zero-limit workaround: the literal native unbounded callback and derivative
pass at the original bound. Native10/11 captures pass; private focused7+2
input gates and affected literal11 successor pass. Root13 finite/adapter tests
also pass on the integrated file, with stable inputs/no survivors. No library
behavior change; no tolerance changes. Raw native capture/driver are retained.
Evidence: `docs/diff_native_queries_cpu_20261014.json`.

## TreeVar dtype-stable unary JAX kernels; unbounded division predicate — 2026-10-14

Sixteen unary runtime primitives now stay in JAX, bypassing the eager NumPy
fallback. Baseline sin JIT failed; private118 checks pass afterward, including
unchanged49 unary and98 first-order expressions. Merged81 checks pass:32 new
real/complex JIT/analytic derivative controls plus49 original unary tests.
Value-dependent real-to-complex branches remain eager; no tolerance/reference
changes. Evidence: `docs/treevar_unary16_jax_cpu_20261014.json`.

Native mrdivide1–9 prefix replay passes and supplies actual clause9 probes
after intervening constructors. The original unbounded100x3 matrix-norm
predicate passes on guarded publication source in12.75seconds/914,872KiB.
Root independently checked all8 extracted MAT/NPZ fields; raw capture and driver
are retained. Evidence: `docs/mrdivide_unbounded_source_cpu_20261014.json`.
This does not close general LS flags, all unary branches, or final CI.

## Numeric evaluation columns and empty column slices — 2026-10-14

Added zero-based numeric component selection: f(x,1) or f(x,columns=1)
selects the second component. Native pass12 uses exact equality on captured
queries. Existing positional string/side keyword evaluation remains intact;
public feval numeric side arguments still raise the original error. Static
selectors, negative indices, sequences/slices, packed query shapes, complex
values and JIT adapters are qualified. Dynamic traced selectors remain unsupported.
Empty column slices/lists now return empty Chebfuns, with source23 and Python
orientation/domain controls; integer out-of-range selection still raises.
The merged54-test gate passes with stable hashes/no survivors. Evidence:
`docs/subsref_columns_cpu_20261014.json`. Full source/API parity and CI remain open.

## Restore initial cumulative-integration source predicates — 2026-10-14

Original cumsum1–9 now use native seed7681 probes and source bounds; added
missing array-transpose, two-column and repeated-integration checks. Ten tests
pass (nine native plus retained supplemental control) on exact guarded current
publication source:21.22seconds/1,399,760KiB, stable/no survivors. This is test
coverage restoration with no cumsum library edits. Later native10–30 are outside
this gate. Evidence: `docs/cumsum_initial_predicates_cpu_20261014.json`.

## Native matrix division for Singfun and array-valued inputs — 2026-10-14

Restored native QR/adjoint branches and avoided scalar Singfun reconstruction
through mat2cell. Numeric matrix adjoints and complex Q columns are conjugated;
row/column output orientation follows the source. Root current-source gates pass
separately:8 singular/adjoint checks and2 grouped array/complex checks, stable
inputs/no survivors. Original mldivide7–9 and mrdivide1–8 are represented with
source operands and bounds. Inherited list adapters/full flag API and main CI
remain open. Evidence: `docs/matrix_division_native_cpu_20261014.json`.

## Preserve complex abs breakpoints; complete subtraction predicates — 2026-10-14

Complex abs now follows native addBreaksAtRoots, per-FUN abs, point-value abs,
and simplify for bounded scalar C1/C2 inputs. This prevents a nearby computed
root replacing an existing breakpoint and triggering global resampling.
Merged13 checks pass at unchanged bounds, with stable guards/no survivors.
CI's exact numerical abs failure was not reproduced locally; the breakpoint
fault was independently demonstrated. Fresh CI still must review this fix.
Evidence: `docs/complex_abs_breakpoints_cpu_20261014.json`.

All15 meaningful native subtraction assertions (1–14,17) now pass in4 grouped
checks on guarded publication source:41.32seconds/2,356,064KiB sampled RSS.
Native matrix/vector infinity norms, output shapes and seed6178 probes are
preserved. Original15–16 are vacuous unconditional-success error blocks, not
meaningful rejection checks. Superseded weaker matrix and overly strong row
norm diagnostics are retained. Evidence: `docs/minus_original_predicates_cpu_20261014.json`.

## TreeVar scalar leaves and RHS assembly use JAX — 2026-10-14

Preserved complex coefficients and RHS leaves, accepted JAX scalar coefficients,
and removed host state/output conversions. The private36-test gate includes all98
original first-order assertions. Merged64 checks pass in19.85seconds, sampled peak
1,090,492KiB, stable inputs/no survivors:15 focused controls plus49 independent
unary conversion predicates. Polynomial JIT/state AD and time-dependent leaves
are qualified; eager construction and unary fallback tracing remain open.
The RHS now returns a JAX column array. Evidence:
`docs/treevar_rhs_jax_cpu_20261014.json`. Full parity and main CI remain unresolved.

## TreeVar Airy uses the existing JAX kernel — 2026-10-14

Removed the TreeVar SciPy Airy runtime call in favor of the qualified JAX
airy_all kernel. Private25 checks pass, including existing JIT/complex kernel
controls. Both affected merged TreeVar checks pass at the original10eps bound:
9.88seconds, sampled peak740,680KiB, stable/no survivors.
This is one numerical dependency removal; general TreeVar tracing and remaining
host operations are still open. Evidence: `docs/treevar_airy_jax_cpu_20261014.json`.

## Division cancellation, native seeded indices and unary branches — 2026-10-14

Merged CPU gate:84 tests pass in49.86seconds wall, sampled peak
1,597,464KiB, stable guards/no survivors. Three scoped packages:

- Bounded scalar division cancels removable roots correctly and preserves numeric
  reciprocal row orientation. All28 original expressions are now represented;
  27 valid expressions pass in scoped gates. Overwritten24 also fails in pinned
  MATLAB and remains strict xfail, with a separate native merge control.
- Chebfun3 index initialization now uses JAX MT uniform/floor/index arithmetic.
  Ten native draw/state cases match;8 final index vectors match, while2 retain
  the existing clipping discrepancy. Root verified350 raw-MAT/JSON/fixture fields.
- TreeVar acsc/acscd/asec/asecd now use the correct real-input complex branch.
  All49 source conversion predicates pass with independent references and original
  bounds. Native primitive residuals are reported separately (max1.42e-14);
  one-ULP backend differences are not hidden by tolerance changes.

Evidence: `docs/rdivide_cancellation_source_cpu_20261014.json`,
`docs/chebfun3_seeded_indices_cpu_20261014.json`, and
`docs/treevar_unary_branches_cpu_20261014.json`. Raw native captures are retained
in tests/fixtures. Full API, normal RNG, plots, speed and final-main CI remain open.

## Preserve leading coefficients in first-order conversion — 2026-10-14

Fixed a source-backed numerical fault: a nonconstant quintic equal at five
fixed probes was replaced by a scalar coefficient, changing the ODE. Native
coefficient-tree evaluation now composes Chebfun leaves directly, preserving
constant-function representation too and removing host sampled reconstruction.
Both baseline controls fail before the fix;24 merged checks pass afterward,
including all98 original expressions and3 controls. Wall27.60seconds,
sampled peak1,634,096KiB, stable guards/no survivors.
Inherited RHS host evaluation and broader TreeVar/CI parity remain open.
Evidence: `docs/treevar_coefficient_source_cpu_20261014.json`.

## Complete original first-order conversion checks — 2026-10-14

Restored all98 native treeVar/toFirstOrder expressions across19 problems and
2 error cases, including omitted metadata/coefficient checks and the extra
linear-index overwrite. Original5e-14 bounds replace old relaxed bounds;
scalar-output slicing and dependent RHS references were removed.
All21 tests pass with no library edits:30.68seconds wall, sampled peak
1,637,248KiB, stable guards/no survivors. Inherited host numerical evaluation,
general TreeVar behavior and main CI remain open.
Evidence: `docs/treevar_firstorder_native_cpu_20261014.json`.

## Quasimatrix division and native error predicates — 2026-10-14

Native rdivide8–12 and both13 assignments plus14–15 now pass with original
operands and bounds. Added native Quasimatrix column dispatch, numeric-zero
representation preservation, singleton expansion and row-oriented array assembly.
Root source review corrected earlier orientation and error-order controls.
The merged22-test CPU gate passes in22.69seconds, sampled peak1,276,416KiB,
stable guards/no survivors:18 native predicates plus4 supplemental groups.
Later16–24 numerical predicates and full main CI remain under qualification.
Evidence: `docs/rdivide_quasimatrix_cpu_20261014.json`.

## Native absolute-value assertions and surface framing — 2026-10-13

## Complete original abs expressions and unbounded fixes — 2026-10-14

All44 distinct expressions in native `tests/chebfun/test_abs.m` now pass at
factory preferences, including both overwritten pass7 assignments. Merged19-test
CPU gate also covers scalar and map-ownership controls; 99.47seconds wall,
sampled peak3,941,944KiB, stable guards/no survivors.
Native random probes, reference operations, norms and bounds are unchanged.

Fixed an infinite root-dedup tolerance that erased unbounded sign-change splits,
preserved nonlinear FUN maps through simplify, and implemented native Singfun
abs on the smooth factor. Wholeline error fell from1.43e-4 to3.33e-16 against
unchanged1.416e-15 bound; halfline also passes. The blow-up TypeError is resolved.
General unbounded complex/array behavior, broader native coverage and green main
CI remain open. Evidence: `docs/chebfun_abs_complete_cpu_20261014.json`.

## Native uniform stream and captured feval inputs — 2026-10-14

Added an explicit-state JAX MT19937 uniform helper, verified exactly against
R2025b draws/state for four seeds and multiple twist cycles. Six private and
six merged checks pass, including JIT/eager, empty and column-major shapes.
Root independently verified all168 extracted fixture fields against raw MAT.
The existing49-clause Chebfun3 feval port now loads native seed42 probes;
all16 arrays/36,633 values equal its prior NumPy inputs bitwise, and all test
predicate ASTs are unchanged. Seven affected source clauses pass privately on
byte-identical production source. Normal random draws and production RNG
consumers remain unchanged; full49 execution and main CI are not claimed.
Evidence: `docs/matlab_twister_uniform_cpu_20261014.json`.

## Direct Chebtech1 preference construction — 2026-10-14

Explicit/session Chebfun3 C1 preferences now follow native C2 sampling grids,
C1 transforms, column-major ACA and nested refinement, JAX QR/DEIM/core solves,
and the source ten-attempt restart/warning behavior. The original type/origin
predicates pass in a guarded full diagnostic. Native factor lengths615/389/389
and warning are reproduced; five off-grid values differ by at most3.12e-14.
The native warned approximation itself has6.81e-6 analytic error; these results
do not certify uniform accuracy. The original bound is unchanged.

The merged35-check CPU gate passes in60.69seconds, sampled peak2,287,740KiB;
original full diagnostic took161.21seconds/4,194,492KiB. New session nonstandard
happiness rejection closes a root-review finding. Default C2/trig regression
controls pass; legacy keyword conversion remains distinct. Inherited host
sampling/RNG adapters, general complex/vectorize/all-preference parity,
classic/numeric tensor routes and full CI remain open. Evidence:
`docs/chebfun3_direct_c1_cpu_20261014.json`.

## Division source predicates and SVD CI repair — 2026-10-14

Restored native rdivide predicates1–7 with exact captured seed6178 inputs and
original bounds, plus25–26 for zero entries in numeric divisor vectors. Both
zero-vector predicates failed before the JAX guard. The unchanged original
SVD CI failures now pass: Trigtech numeric action runs eagerly to preserve its
value-dependent native zero-result storage; polynomial action stays compiled.
Merged46-check CPU gate passes (10 division,22 two-dimensional SVD,14 disk SVD),
62.21seconds wall, sampled peak2,887,864KiB, stable guards/no survivors.
Full division coverage and green main CI remain open. Evidence:
`docs/rdivide_svd_ci_cpu_20261014.json`.

Restored30 smooth Chebfun abs predicates with original reference construction,
operation order, exact native random probes and unchanged bounds. Nine merged
CPU tests pass, including the retained supplemental check. Two original periodic
predicates pass in a separate test. Earlier substituted-reference/probe tests
were rejected before publication. Remaining identity-conversion, singular and
unbounded predicates are still open. See `docs/chebfun_abs_native_cpu_20261013.json`.

Random2D now uses the native rectangular viewport, orthographic projection,
explicit x/y limits and source inset layout. The full CPU example passes in
125.16seconds and emits all seven600-by-270 figures; only panel4 changes.
Native fixed-mesh property captures support these structural changes. Automatic
camera aperture/data aspect for the random surface, RNG stream and color
interpolation remain unqualified. See `docs/random2d_framing_cpu_20261014.json`.

Full native assertion coverage, known dense constructor failure, remaining
page/figure and speed gaps, and green final-main CI remain required.

## Direct periodic Chebfun3 construction — 2026-10-13

Periodic Chebfun3 now samples native half-open grids and builds Fourier factors
directly, with source envelope happiness, refinement and simplification. Copied
explicit preferences and selected session fields control technology, epsilon,
sample testing and constructor dispatch. The merged CPU gate passes25 controls,
including native off-grid queries and independent complex/nonunit-domain values;
new controls forbid postconversion. Default C2 controls still pass.

Unsupported C1 preference construction, classic construction, numeric Fourier
tensors and custom preference routes remain explicit gaps. The old explicit
C1 keyword keeps its legacy conversion behavior. Shared ACA/QR implementation
and full native algorithm parity are not qualified by this package. Evidence:
`docs/chebfun3_direct_trig_cpu_20261013.json`.

The dense Chebfun3T composition test now executes original predicate1's callback
construction, replacing a different product-of-approximations test. The isolated
original predicate passes private and merged CPU gates with its original bound;
remaining predicates2-9 are unchanged and are not claimed freshly rerun. See
`docs/chebfun3t_compose_predicate_cpu_20261013.json`.

Full native assertions, known dense accuracy failure, pages/figures, remaining
CPU speed gaps and green final-main CI remain unresolved.

## Named left division and Random2D lighting — 2026-10-13

Added Chebfun3T `ldivide` and `mldivide` adapters with native direction,
empty-input handling and type-error identifiers. At least one operand must be
Chebfun3T; Python numeric literals adapt MATLAB double literals. Three private
and three merged CPU controls pass. See `docs/chebfun3t_leftdivide_cpu_20261013.json`.

Random2D now uses the source 200-point surface grid, Parula, and captured local
light/material properties. The full CPU script completes with all seven
600-by-270 figures; eight private and eight merged controls pass. The page
records remaining JAX-versus-MATLAB random stream, color interpolation and
surface framing differences. This is a qualified rendering improvement, not
pixel parity. See `docs/random2d_lighting_cpu_20261014.json` and
`docs/random2d_lighting_integration_cpu_20261013.json`.

Full native assertions, constructor accuracy, page/figure parity, remaining
speed gaps and green final-main CI remain unresolved.

## Chebfun3 constructor defaults and session preferences — 2026-10-13

The public factory now starts with the native 17-point grid instead of9 and
reads the session Chebfun3 epsilon when tolerance is omitted. Explicit tolerance
and eps override precedence are preserved. Baseline controls reproduce both
faults. The combined CPU gate passes12 checks, including unchanged original
constructor clauses8–11 and constructor2 clauses13–14. Global preferences are
restored after each test. Evidence: `docs/chebfun3_constructor_defaults_cpu_20261013.json`.

The ZetaZeros matched-coefficient control reproduces native continuation
coefficients to1.55e-14 and Python continuation coefficients exactly. The larger
between-run continuation difference follows the initial coefficient perturbation;
no separate composition defect was found. Updated evidence is in
`docs/zetazeros_native_comparison_cpu_20261013.json`.

Full constructor technology/preference semantics, native coverage, pages/figures,
remaining numerical issues and current-main green CI remain unresolved.

## CPU compilation reuse, coefficient plotting and native page evidence — 2026-10-13

Adaptive sample metadata now groups existing JAX operations into reusable
compilations. Twenty captured sample batches and complete forward/inverse
coefficients and point values remain byte-identical. All32 private controls and
five merged controls pass. One coordinated serial CPU pair reduces setup plus
first inverse30.34→29.00seconds and compiler events594→541. Warm medians are
0.6044/0.6073seconds; no warm-speed or general performance guarantee is claimed.
See `docs/inverse_sample_metadata_cpu_20261013.json`.

Added Chebfun3T coefficient plotting with source index order, zero floors,
log colors, and native-captured scatter properties. Camera conversion is applied
through the existing MATLAB adapter. Four private and four merged controls pass.
This qualifies data/artist contracts, not raster identity. See
`docs/chebfun3t_plotcoeffs_cpu_20261013.json`.

Fresh full-source ZetaZeros runs on current Python and pinned MATLAB both emit
the same composition warning. Seven of eight printed root rows match; the last
differs in its tenth decimal. Coefficient counts match, but critical-line
coefficient differences still need diagnosis. The322-page static audit confirms
all2408 MATLAB code blocks and1305 figure slot/dimension matches; numerical and
raster parity remain incomplete. See `docs/zetazeros_native_comparison_cpu_20261013.json`
and `docs/page_audit_cpu_20261013.json`. Full native coverage, known constructor
failure, remaining correctness/speed gaps and current-main green CI stay open.

## Dense Chebfun3T degree tangent and display — 2026-10-13

Added the missing `tand` and `display` methods. Degree reduction preserves
captured exact signed zeros, poles and unit values, finite neighboring poles,
large angles, and tiny complex components. Review corrected cancellation for
negative angles before integration. JIT and analytic derivative controls pass.
The display matches four native compact/loose, empty/nonempty transcripts exactly.

Merged CPU gate: 16 tests pass plus four exact native display comparisons;
pinned JAX/JAXlib0.11.0, x64, stable inputs and no surviving processes. Full static
checks pass. Evidence: `docs/chebfun3t_tand_display_cpu_20261013.json`.
This is scoped API progress; the existing dense constructor expected failure,
complete native coverage, page/figure parity and green main CI remain open.

## Dense Chebfun3T source representation — 2026-10-13

Chebfun3T now stores the native full coefficient tensor and uses source adaptive
construction, evaluation, integration and arithmetic. The prior callable path
incorrectly delegated to low-rank Chebfun3. Original grids and bounds are restored:
17 evaluation predicates, 9 composition predicates, all 40 battery functions,
and get/ndf/sum3 pass. Nine of ten constructor predicates pass. A merged gate
passes 16 tests with one strict expected failure; this is partial parity.

The remaining original rational-constructor predicate fails at2.81e-14 against
100eps=2.22e-14 (native8.33e-15). Identical samples isolate FFT-roundoff-sensitive
source truncation; both full tensors pass, but the JAX-selected crop fails.
The unchanged assertion executes as a strict expected failure, never counted
as a pass. No tolerance/rank/cutoff fitting is used. Python Tucker bridge values,
derivatives and factor roundtrips pass independently. See
`docs/dense_chebfun3t_cpu_20261013.json` for immutable evidence and limitations.
Full library/example/figure parity and green main CI remain unresolved.

## Scalar periodic source Newton and endpoint CI repair — 2026-10-13

Supported scalar periodic trigcolloc problems now use continuous AD and the
source Newton recurrence, correction resolution, preferences and history.
Final28controls pass; original first-/second-order and nonlinear integral
predicates pass unchanged within the documented selector qualification scope.
The merged checkout passes the original first-order and native constant-history
tests (58.48s pytest), with stable inputs/no survivors. Matched native first-order
results have43coefficients/eightupdates, sampledelta2.25e-15. Intermediate cutoff
and unsupported numeric callbacks remain different; systems/trigspec/complex
trajectories are unqualified. See `docs/periodic_scalar_newton_cpu_20261013.json`.

The CI tiny-power endpoint failure was a stale supplemental assertion. A clean
native capture gives g(1)=0 and pointValues=[1,0], even while its smooth factor
is1. The test-only correction passes; native zero-exponent Singfun versus Python
smooth Tech remains a representation distinction. See
`docs/power_endpoint_ci_cpu_20261013.json`. Full CI remains unresolved.

Restoring exact chebfun3t constructor grids/bounds in a private test package
exposes nativepass7 error1.01e-11 versus2.22e-14. Native chebfun3t is dense;
current Python delegates to low-rank chebfun3. Diagnosis/qualification are open;
no tolerance weakening or full constructor parity is claimed.

## Native multiplication errors and plain-data point glyphs — 2026-10-13

Trigtech matrix multiplication now reports the exact native error identifiers
and messages for predicates11–14. Scalar, complex-sequence, empty-operand and
result-zero contracts are covered. The19-check gate passed before the final
storage-isreal correction; its affected complex-zero/nonfinite control passed
again afterward. See `docs/trigtech_mtimes_cpu_20261013.json`.

Plain-data matlab_plot now reuses qualified small-dot geometry while retaining
source marker sizes, coordinates and color-cycle behavior. Related92checks
pass; an amended cycle test also passes independently. The full ConformalL
rerun completed cleanly in25.38s. All6inputs/14proseparagraphs preserve source
order, all5outputblocks match actual stdout, and all3figures are600×270.
Oversized dots are corrected; historical counts/location/figure differences
remain. See `docs/plain_dot_glyph_cpu_20261013.json`. Full CI remains unresolved.

## Rectangular basic least squares and ConformalL — 2026-10-13

ConformalL now uses a JAX pivoted-QR basic solution for its rectangular fit,
matching the native solver choice. Twenty independent MATLAB controls and
four scope checks pass. Exact native inputs independently produce rank73
and the same eight zero variables; no rank or tolerance was fitted.
See `docs/basic_lstsq_cpu_20261013.json`.

The full CPU page run completed cleanly in32.00s, with stable inputs and no
survivors. All five output blocks and three600×270 figures were regenerated
from actual output. Boundary error prints5.1268e-7, matching the historical
page. Pole counts remain46/67 versus fresh native46/69 and historical27/44.
Point glyph sizes, historical distance sequence and exact raster/output parity
remain unresolved. Existing NumPy example arithmetic outside the solver call
also remains. See `docs/conformal_l_page_cpu_20261013.json`. Full CI is pending.

## Nested technology preferences — 2026-10-13

Nested provider defaults now resolve through the source preference contract.
Nested writes update raw overrides while copied preferences remain independent.
Held views refresh after writes, and existing nested dictionary operations
preserve removed values. The final candidate passes 40 focused preference
checks; both integrated files are byte-identical to the guarded candidate.
See `docs/nested_techprefs_cpu_20261013.json`. Full CI remains pending.

## CPU truncation compilation reuse — 2026-10-13

standard_chop now shares compiled kernels across nearby lengths while retaining
the original active length in every source stopping rule. The disk helper uses
the same kernel. Seventy-four focused checks pass in the private candidate;
no source bounds or tolerances changed. Retained Krylov Tcall14 input and both
outputs remain byte-identical. Observed first-call time falls from 7.385s to
6.376s; warm repeat remains near 46ms. Setup compiler time also falls, but total
supervised wall time is nearly unchanged and this is not a full Krylov solve.
Four isolated warm chops cost about 0.332ms more. See
`docs/standard_chop_compilation_cpu_20261013.json` for scope and evidence.

## Surface CI tests and ConformalL diagnosis — 2026-10-13

Two f6a50aaa CI surface-color failures came from tests intercepting the old
plot_surface path. The tests now inspect the actual interpolated artist's
source scalar data, preserving both numerical assertions. Related private
checks pass26/26; the two original failures pass in the publication checkout.
See `docs/surface_color_ci_test_cpu_20261013.json`. Full CI remains unresolved.

Fresh pinned MATLAB and current Python both return46 forward and69 inverse
ConformalL poles. The cached original instead reports27/44, so historical
extra-pole/output parity remains unresolved. With exact MATLAB
boundary inputs, inverse poles agree to approximately3e-11 in absolute distance,
but forward poles differ by up to0.093. The current example's ill-conditioned
boundary fit also differs; matching counts do not establish full parity.
Native supports/weights are being captured to separate greedy-selection from
pole-extraction differences. See `docs/conformal_l_native_cpu_20261013.json`.

## Root powers and IVP page refresh — 2026-10-13

Root-split powers now map stored point values and preserve transposed orientation,
following native power.m. Both new regression cases pass in the publication
checkout; the private candidate also passed the focused native sqrt/power
controls at unchanged bounds. See `docs/sqrt_pointvalues_cpu_20261013.json`.

All three IVPCapabilities solves and four 600×270 figures completed in a guarded
CPU run (126.33s, peak summed RSS 4,122,476 KiB). The regenerated page matches
seven source input blocks and seventeen prose paragraphs and prints actual
execution output. It uses source marker size and MATLAB default insets.
Exact parity remains open: coefficient count 3881 versus 3765, timings,
automatic Y limits/ticks, typography and raster differences. The execution
snapshot is 2f47429e; later root-power and input-promotion changes were not part
of this full-page run. See `docs/ivp_capabilities_page_cpu_20261013.json`.

## Unbounded sqrt and exact query promotion — 2026-10-13

Root powers now retain unbounded FUN mappings and classify the principal
branch at a canonical interior point. All eleven native sqrt clauses passed
in the private candidate. The publication integration passed 17 checks,
including the original CI sqrt failure, both infinite-tail controls and the
new float32/complex64 JIT promotion tests. Full source bounds are unchanged.
Exact primal widening preserves tiny input values; subnormal AD tangents and
dynamic cotangents retain existing cast limitations. See
`docs/promotion_sqrt_integration_cpu_20261013.json`,
`docs/sqrt_unbounded_cpu_20261013.json` and
`docs/exact_primal_promotion_cpu_20261013.md`.

## Variable-coefficient MINRES comparison — 2026-10-13

The Krylov source example now completes in Python with flag 0 and 22 iterations,
as in MATLAB. Source L2 solution errors are 3.813e-13 and 2.907e-13 respectively.
The normalized residual curves differ by at most 6.14e-14. Python runtime is
still slow: the instrumented capture took 268.62s under its supervisor.
Its numerical checkpoints completed before a final JSON-write harness error;
the retained exit 1 is explicitly reported, not counted as a clean process gate.
Coefficient counts/RHS values and full example/figure parity remain different
or unqualified. See `docs/krylov_varcoeff_native_cpu_20261013.json`.

## IVP first-solve comparison — 2026-10-13

The first IVPCapabilities solve now returns one smooth piece in both current
Python and pinned MATLAB, superseding the historical 16-piece Python result.
MATLAB returns 3,765 coefficients (matching the cached page); Python returns
3,881. The maximum difference on 5,001 common sample points is 1.526e-9.
Both guarded CPU runs completed with stable inputs and no survivors.
Representation, complete example outputs and figures remain unqualified;
no acceptance threshold is inferred from these observations. See
`docs/ivp_capabilities_first_solve_cpu_20261013.json`.

## Logical comparison CI repair — 2026-10-13

Equality and inequality now follow their separate native source paths,
preserving unbounded FUN mappings and native singular pointValue behavior.
The private candidate passed 23 distinct focused tests. The publication
checkout passed all 12 native-ne tests, including the original CI failure,
with stable guards and no surviving processes. No tolerance was widened.
The native singular capture saved its scientific results before a shutdown
wall cap; it is not claimed as a clean native process gate. See
`docs/logical_eq_ne_cpu_20261013.json`.

## Public continuous spin2 example path — 2026-10-13

Ginzburg–Landau now calls the existing public Spinop2/spin2 API directly,
including continuous multi-time stepping and native matrix output indexing.
The example-local reconstruction/restart wrapper is removed. A guarded full
candidate run completed in 119.74s at 3,714,812 KiB sampled RSS, with stable
inputs and no survivors. Its page retains 15 input blocks, 20 prose paragraphs,
five outputs and nine figures at the reference dimensions.

Historical late-field/pixel parity remains open. Matched native Fourier
states and ETDRK4 factors demonstrate amplification of small arithmetic
perturbations; no solver tolerance or equation was altered to fit the images.
See `docs/gl_public_spin2_cpu_20261013.json` for exact run provenance,
integration scope and hashes.

## Chebfun2 composition CI repair — 2026-10-13

Array-valued outer Chebfuns now compose one scalar column at a time, as in
the MATLAB source, preserving native matrix-query output layout. Scalar and
vector composition also inherit periodicity separately on each axis.
Thirteen focused private controls passed; the merged publication checkout
passed nine composition/phase-plot integration tests in 37.26s, with stable
source guards and no surviving processes. The original CI predicate passes
with its unchanged 1000-epsilon bound. See
`docs/chebfun2_compose_array_cpu_20261013.json`.

The preceding-main unbounded logical inequality failure is addressed by the
source eq/ne repair above. Core shard3 stopped on a runner shutdown signal
(no assertion failure shown); the coverage dependency gate consequently failed.
Current-main CI, full native assertion coverage, and complete page/figure
parity remain open.

## Surface and complex phase rendering — 2026-10-13

Source-based surface rendering now interpolates vertex colors, uses the exact
Parula table, and preserves equal physical units in topdown views. Complex
Chebfun2 plots now follow the native 500-by-500 phase-portrait branch with
HSV(600). The graphics candidate passed 61 focused CPU controls, including
existing sphere/camera tests and colorbar/custom-option compatibility.

The full Ginzburg–Landau candidate completed in 102.46s with nine 600-by-270
figures and stable input guards. Its first spiral now closely matches cached
reference geometry and palette. The final phase portrait still differs in
its evolved field; matched native state/coefficient replays demonstrate roundoff amplification
but do not establish late-field parity. See
`docs/gl_evolution_roundoff_cpu_20261013.md` for evidence.
The explicit R2016a palette is a reference-compatibility inference, not proof
of the reference's generating runtime. Exact pixels and broader page figure
parity remain open. See `docs/gl_surface_phase_cpu_20261013.json`.

CI run 38421149742 on preceding main e73957d4 reported one Chebfun2 composition
failure with 454 tests passing in that shard. The array-valued composition
caller repair is qualified above; the separate unbounded logical failure
is still under repair. Full goal completion remains open.

## Fresh Krylov indefinite source comparison — 2026-10-13

The exact indefinite MINRES clause from `ode-linear/Krylov.m` now has fresh
CPU captures against MATLAB Chebfun 7574c77. Both use breakpoints [-1,0,1]
and identical piece lengths; solution samples differ by at most 1.34e-14.
The source quantity `norm(u_minres-u_colloc)` is 3.13e-15 in Python and
1.37e-14 in MATLAB. The example defines no acceptance threshold for this
quantity, so these are observations, not a newly invented pass criterion.

Both completed captures exited successfully with stable source guards and
no surviving owned processes. Resource and harness failures remain recorded;
run timings are unpaired observations. See
`docs/krylov_indefinite_native_cpu_20261013.json` for hashes and comparisons.
The complete page, preconditioned variable-coefficient clause, residual plots,
full parity, and current-main CI remain unresolved.

## Fresh public inverse CPU measurements — 2026-10-13

At 7c750029, Python and pinned MATLAB Chebfun independently constructed the
same flower arc-length function, then made six public inverse calls. Warm
medians were 0.6065s (Python) and 0.6399s (MATLAB). First calls after setup were
13.2855s and 0.7476s; setup took 18.7860s and 0.8390s. The first-call/setup
costs remain unresolved. These are descriptive paired CPU measurements,
not pristine cold starts or identical internal representations.

Both runs exited successfully with stable inputs and no surviving owned
processes. All six results repeated identically within each implementation;
129-point roundtrip errors were 1.05e-15 and 6.66e-16, below the unchanged
1e-10 diagnostic bound. Forward/inverse lengths differ: Python 2092/8878,
MATLAB 2016/8706. The initial MATLAB startup failure from a long temporary
path is preserved and excluded. See
`docs/inverse_public_cpu_benchmark_20261013.json` for timings, hashes and scope.
Full inverse/example parity and current-main CI remain open.

## Core CI partition and progress diagnostics — 2026-10-13

Core CI now uses eight shards instead of six. All 797 current core test files
are assigned exactly once; per-test timeout (600 seconds), two workers per
shard, complete manifest checks and aggregate 79% coverage remain unchanged.
Worker logs retain test start/finish node IDs in separate shard directories.
The local plugin smoke preserved the expected failing-test exit status.
See `docs/ci_core_partition_progress_20261013.json` for checks and evidence.

This follows run 38402535328 reaching its 90-minute limit in core shard 0.
Its quiet log reached 92% of items but does not identify the last test or
predict remaining runtime. Eight file-based shards do not guarantee balanced
runtimes. Full current-main CI is still pending; parity is incomplete.

## Ordered CPU evaluation and empty functions — 2026-10-13

The final JAX revision passed 83 tests together in the publication checkout.
Traced piece selection now distinguishes subnormal queries and uses native
half-open intervals. Empty Chebfuns return empty before query validation,
including fixed-length-zero representations. Bounded nonfinite evaluation
matches the native Clenshaw initialization for constants through quadratics;
NaN and positive infinity select no piece in the multi-piece source masks.
See `docs/feval_ordered_routing_cpu_20261013.json` for source evidence,
compiler review, retained failed runs and final receipts.

The JAX mask adds about 75 microseconds (49%) to the measured scalar calls;
full inverse timing remains pending. Float32/complex64 value promotion can
still flush subnormals on narrow domains; a concrete eager/JIT reproducer is
retained. This is not full evaluation or performance parity. Older CI run
38402535328 reached its 90-minute limit in core shard 0 after 23 other jobs
passed; current-main CI and full parity remain unresolved.

## Public evaluation and quasimatrix source semantics — 2026-10-13

Public `cj.feval` now supports documented evaluation syntax. Quasimatrices
handle endpoint strings and one-sided values, preserve empty query shapes,
and apply row orientation once after native component packing. Their missing
`isempty` method is implemented. All 13 numbered native `isempty` predicates
are represented, including literal horizontal concatenation and singular and
unbounded constructors. The package has 32 distinct passing CPU checks;
27 ran together in the publication checkout. See
`docs/feval_public_quasimatrix_isempty_cpu_20261013.json` for scope and receipts.

Empty scalar Chebfun evaluation and JIT subnormal/nonfinite routing are still
under separate repair. Full parity and green CI on current main remain open.

## Exact one-sided evaluation and native feval coverage — 2026-10-13

This package has 93 passing CPU checks across bounded fresh processes, covering
all 35 numbered MATLAB `test_feval` cases and 16 sign regressions. The tail uses
captured native MATLAB queries and operator values with original inequalities;
no source tolerance was relaxed. One-sided evaluation now preserves nearby and
complex queries and native endpoint ownership. Numeric `cj.sign` preserves real
subnormal signs, including fused CPU JIT; 24 additional IEEE robustness checks
passed. See `docs/feval_exact_sides_sign_cpu_20261013.json` for hashes, receipts,
native fixture provenance, adaptations and limits.

The initial large sign run hit its memory cap; all tests subsequently passed
in smaller batches. General traced subnormal piece routing remains open, as do
full repository assertion coverage, page/figure parity, speed qualification and
green CI on current main. The preceding main CI run is pending; docs passed.

## Native feval matrix and tensor shapes — 2026-10-13

All18 source cases1,14–30 now pass using captured MATLAB seedRNG(7681)
queries and original bounds. Seven former shape failures are fixed: component
blocks pack along the second axis, and transpose swaps the first two axes.
The combined CPU gate passed37 tests, including the published JIT breakpoint
fix and a new matrix point-value JIT control. Empty query shapes are preserved;
endpoint orientation and native error identifiers are implemented. See
`docs/feval_matrix_shapes_cpu_20261013.json` for receipts and compatibility.
Python scalar/rank-one conventions remain explicit adapters. Sign callback
semantics, exact one-sided selection and clauses31–35 remain open; full parity
and current-main CI are not complete.

## CPU JIT breakpoint equality — 2026-10-13

Stored point values now use exact IEEE comparisons under CPU JIT, including
subnormal queries, signed zero, complex offsets and float32 widening. The
compiler previously rewrote integer zero canonicalization into floating equality;
uint32 word comparisons pass the public reproducer and all12 focused tests.
Original eight test bodies/bounds are unchanged. Eager explicit override behavior
is unchanged. See `docs/feval_jit_breakpoint_equality_cpu_20261013.json` for
receipts and scope. General subnormal piece routing, tensor shape parity and full
main CI remain open. This is not a full feval parity qualification.

# Current CPU parity checkpoint — 2026-10-13

Main through `d8e079cb` is published; this checkpoint adds the packages below.
Full parity remains incomplete. Pushes work; qualification is CPU only.

Recent verified changes:

- `diffbarytrig` now uses JAX recurrences and source option parsing. Twelve
  native captured cases and unchanged analytic bounds are covered by 34 passing
  CPU checks; static-argument JIT is covered. The trig utilities module no longer
  imports NumPy. Exact duplicate support nodes and MATLAB interactive option
  display remain unqualified. See `docs/diffbarytrig_jax_source_cpu_20261013.json`.

- Floquet periodic panels now use the native data range and compact tick
  presentation. All 32 x/y tick sequences and labels match the fresh native
  capture; y-limit differences are at most 6.56e-15. All 16 plotted arrays
  remain byte-identical; rendering is 600x270. Glyph/raster parity remains open.
  See `docs/floquet_periodic_ticks_cpu_20261013.json`.

- `trigpoly` now returns a periodic Chebfun with source Fourier coefficients,
  phase, orientation and default-domain preferences. All six native captured
  cases are covered; coefficients match exactly and sampled differences are
  at most 2.30e-16. Twenty-six CPU checks pass. This corrects the old sampled-array
  return API; see `docs/trigpoly_native_source_cpu_20261013.json`.

- Public Bessel roots now use the existing JAX Piessens/McMahon kernel,
  removing duplicate NumPy execution. Native count validation and early return
  are preserved; 36 CPU checks pass, including all six original numerical
  predicates with unchanged bounds. Dynamic JIT parameters and complex counts
  remain unqualified. See `docs/besselroots_public_jax_cpu_20261013.json`.

- Legacy `utils.transforms.dst/idst` now use the existing source JAX provider.
  This corrects the former SciPy factor-two forward scaling, inverse scaling
  and complex truncation; adds source DST V–VIII and IDST V–VII; and preserves
  the native IDST VIII error. Final 61 focused checks pass, including all 32
  native DST slots; 48 other transform definitions are unchanged. See
  `docs/dst_wrapper_source_cpu_20261013.json` for compatibility and evidence.
- Floquet periodic-matrix figure now uses measured native subplot positions,
  source font/stroke sizes and 72dpi at 600x270 pixels. All 16 saved curve
  coordinate arrays and data limits are unchanged. Tick selection and raster
  differences remain open. See `docs/floquet_periodic_layout_cpu_20261013.json`.
- Floquet exponent and multiplier output now uses the source four-decimal
  display. Both blocks match the actual native capture exactly when replayed
  from retained computed eigenvalues; all non-print executable AST is unchanged.
  See `docs/floquet_print_format_cpu_20261013.json`. Figures remain open.
- Ten division tests now load packaged native MATLAB fixtures on CI instead
  of external scratch paths. All ten pass locally; assertions are unchanged.
  See `docs/rdivide_fixture_packaging_cpu_20261013.json`.
- CPU inverse subdivision reuses compilation across sampled grid sizes. The
  controlled stage improves 5.69 to 2.87 seconds; all 106 subdivision outputs
  and the complete inverse representation remain identical. Original accuracy
  passes, with 11 additional same-backend portability controls. This is a stage
  timing claim, not a matched whole-inverse speed claim. See
  `docs/inverse_sampled_cpu_reuse_qualification_20261011.json`.
- Polynomial Trigremez exchange now retains true jump roots, matching native
  `roots(..., "nobreaks")`. Original polynomial predicates 1–7 and three native
  root controls pass. The symmetric extremum-selection discrepancy remains
  open; retained-stage analysis traces its first observed difference to
  trigonometric-to-Chebyshev conversion. See
  `docs/trigremez_polynomial_exchange_source_cpu_20261011.json`.
- Floquet now has a complete fresh matched-input numerical capture: all 16
  periodic entries differ from native samples by at most 7.22e-15 and trajectory
  samples by 6.79e-14. These are observations, since the example defines no
  acceptance bound. The original final recorder failure and offline recovery
  are retained. Representation lengths, historical RNG and figure differences
  remain open. See `docs/floquet_matched_math_cpu_20261011.json`.
- Empty numeric/cell/existing Chebfun2 inputs now follow native vector-field
  construction: all-empty inputs yield an empty field; mixed empty/nonempty
  components raise the source domain error. All 12 native controls failed
  before the fix; the candidate passes 51 focused controls, including existing
  constructor and empty-operation tests. General parser and empty-object storage
  parity remain open. See `docs/chebfun2v_empty_inputs_cpu_20261011.json`.
- Finite scalar Chebop deflation now uses the existing source Newton solver and
  preserves the nonlocal AD derivative. The original Painleve test passes:
  solution lengths 38/75 match native, and sampled differences are below 6.3e-15.
  Seventeen AD controls, 15 small public controls and the original Bratu and
  Herceg tests pass. Two further public end-to-end controls pass. A legacy
  restart-warning expectation is replaced with the source unsuccessful-Newton
  contract; retained-root replay reproduces the original result exactly and
  verifies failure history. The full edited test was not rerun locally.
  Explicit `n` now follows scalar Newton's equation-point convention; the
  finite route uses its source tolerance instead of legacy `newton_tol`.
  Other backend/domain dispatches are documented separately; this is not
  a qualification of every solver route. See
  `docs/painleve_deflation_source_cpu_20261011.json`.
- Trigremez now accepts source name/value options and emits the native iteration
  display and plotting callbacks. Thirty-one focused controls pass, with native
  numeric, parser and graphics captures. Existing numerical-provider arithmetic
  is unchanged. A symmetric polynomial exchange-reference difference and
  automatic axes/layout parity remain open. See
  `docs/trigremez_iteration_options_source_cpu_20261011.json`.
- Floquet now emits the native matrix-logarithm warning for its actual endpoint
  matrix. Seven native controls qualify the diagnostic while preserving SciPy
  output bytes and other warnings. The page includes the captured warning;
  historical random trajectories and full figure parity remain open. See
  `docs/floquet_logm_warning_cpu_20261011.json`.

- Disk cutoff compilation is now reused across completed FFT magnitude buffers
  without changing FFT lengths, active source lengths or tolerances. A second
  controlled CPU pair measures29.19 to17.80seconds and12.1% lower sampledRSS;
  all68factors/cutoffs are preserved and15tests pass. This is stage performance,
  not a new whole-constructor timing. See `docs/disk_chop_active_cpu_20261011.json`.
  A separate scale fusion preserves all68scales/factors, passes11checks, and
  improves its controlled stage17.92 to15.15seconds with13.4%lower sampledRSS.
  Cached values and tolerance arithmetic are unchanged. See
  `docs/disk_scale_fusion_cpu_20261011.json`.
- Vector construction now handles one-level cell inputs and numeric scalar/
  matrix components through existing JAX constructors. All35focused checks pass
  (20original constructor predicates,4original vectorize clauses,11controls).
  Native numeric/cell captures agree with analytic targets within4.45e-16.
  General argument parsing, empty numeric inputs and exact diagnostic adapters
  remain open. See `docs/chebfun2v_numeric_cell_cpu_20261011.json`.
- Public plotting now reproduces native complex-coordinate warnings and creates
  separate point artists for scalar/vector coordinate pairs. Lorenz uses that
  public path and no longer suppresses all warnings. Seventeen scoped controls
  pass; full-matrix input retains its prior behavior. No Lorenz trajectory or
  pole-table parity claim. See `docs/lorenz_plot_coordinate_cpu_20261011.json`.

- Disk slice simplification fuses the existing JAX operations while retaining
  each slice's FFT length and tolerance. One controlled CPU stage pair improves
  45.78 to 29.41 seconds, with 43.6% lower sampled memory; all 68 output factors
  are bit-identical. Ten focused checks pass. The actual default constructor
  preserves all factors/samples and its known supplemental accuracy failure;
  no whole-constructor paired speed claim. See
  `docs/disk_simplify_fusion_cpu_20261011.json`.
- Public trigremez input validation now preserves native guard ordering and
  diagnostic identifiers. Sixteen native cases and six existing adapters pass.
  Display, plotting and general name/value options remain in progress. See
  `docs/trigremez_input_policy_source_cpu_20261011.json`.
- Empty Chebfun2v/Chebfun3v norms now return empty JAX arrays, matching the
  native empty result and existing Diskfunv convention. Twelve focused checks
  pass. Three supplemental norm tests no longer skip errors; all three pass.
  The refreshed inventory contains all 1,102 native filenames with no literal
  masks in mapped files; this is not assertion-completeness or runtime parity.
  Eleven false native-test path claims were corrected. See
  `docs/empty_vector_norm_inventory_cpu_20261011.json`.
- TransientGrowth now has two 600-by-270 figures using source plotting and
  measured native geometry. Full source runs agree on the printed maximum;
  the renderer preserves all 56 coefficient arrays and recorded samples.
  Font rasterization and plot-grid differences remain explicit. See
  `docs/transient_growth_renderer_cpu_20261011.json`.
- DDE is now verified at all 23 output slots (24 figures). Earlier 20/23
  checkpoints are historical; this does not establish pixel-identical plots.

- Two supplemental CI expectations are reconciled with native behavior.
  Single-component vectors are valid. The variable-coefficient cubic solve's
  analytic100eps bound also fails in MATLAB; exact projection and independent
  native LU comparisons now separate projection from conditioned solve error,
  retaining100eps for native comparisons. All14 focused checks pass; the old
  failures remain recorded in `docs/piecewise_assembly_ci_cpu_20261011.json`.
  No library behavior or original native-test bound changed.

- Rational trigremez now follows native root exchange and best-iterate stopping
  with a JAX finite-pencil solve. All 13 original predicates pass. Independent
  native comparisons give abs(x) five iterations and exp(sin(pi*x)) one, with
  no convergence warnings and sample differences1.22e-15/6.22e-15. Persistent
  tests cover the former zero-iteration bug. Public input/options gaps and
  roundoff-scale status differences remain open. See
  `docs/trigremez_rational_source_cpu_20261011.json` and the finite-pencil proof.

- EightShades now executes the actual public approximation calls, with a
  retained full CPU run (191.50 seconds, 4.65 GiB peak), all 20 sampled objects
  and actual warnings compared with native execution. Source polynomial
  exchange, periodic coefficients, real rational input and division point
  values are repaired. Four 600-by-270 figures use measured native geometry;
  renderer corrections preserve all 22 curve arrays. Near-pole and polynomial
  trig convergence gaps remain explicit in
  `docs/examples/approx/EightShades.qualification.json`. Source inventory confirms
  this page has only polynomial trigremez; its rational slot is a placeholder.
- Vector integrals delegate to the existing JAX representations; vector norms
  use the native component SVD formula. All six constructor2 clauses pass,
  as do the affected 20 constructor and 22 three-component predicates.
  Complete numeric/cell parser parity remains open. The empty-norm successor
  is qualified above. Earlier evidence:
  `docs/chebfun2v_integral_constructor2_source_cpu_20261011.json`.
- Disk Phase-II rank-one updates preserve source arithmetic order using JAX.
  Nine native plus-stage fixtures match bitwise both directly and under JIT;
  13 focused checks pass. This does not establish full constructor speed or
  correctness. Evidence: `docs/disk_phase2_rank_one_cpu_20261011.json`.

- Disk refinement now forwards the global scale to the existing JAX slice
  checkers and uses native Fourier cutoff semantics. Nine independent native
  fixture cases plus two original constructor assertions pass. The effective
  Phase-I-derived tolerance is retained; a proposed change to that tolerance
  was rejected during source review. A subsequent independent native capture also fails the unchanged supplemental
  tiltedpeg bound; JAX/native samples agree within1.14e-15. Full accuracy and
  CPU speed remain open; see docs/disk_tiltedpeg_default_cpu_20261011.json.
- The exact19-operation native empty-vector test passes, restoring omitted
  multiplication calls without changing library behavior.

- All20 native Chebfun2v constructor checks and18 compatibility/source-body
  controls pass. Existing components retain their representations; omitted
  domains are inferred, and one-component complex coordinate composition
  follows the source real/imaginary dispatch. Full numeric/cell parsing remains
  outside this package. All6 native Chebfun2 rank predicates also pass.

- Besselh now uses a JAX-only complex Hankel kernel and honors native scaling,
  output preferences and zero-error identifiers. All12 native predicates,
  9 compatibility controls and29 kernel controls pass. Integer, near-integer,
  half-integer and complex inputs have retained native oracles; arbitrary
  extreme orders/arguments remain outside the numerical qualification.
- All10 native Chebfun2v composition predicates pass, restoring 3D vector
  operands, periodic outputs and direct call dispatch. Seven additional
  compatibility/domain/complex-value controls pass. One-component constructor
  and composition source branches are now covered by the constructor package.

- Deltafun integration correctly shares residual derivative impulses between
  adjacent pieces and preserves complex integration constants. Nine focused
  checks pass, including three independent MATLAB captures and native branch
  controls. This method now uses JAX arrays for its numerical bookkeeping.

- The original PDE continuous Frobenius predicate passes at **3.52943e-12**
  against its unchanged **2.22045e-11** bound. Native defaults and C2 nodes
  are preserved. 200 actual native RHS states and 76 differentiation-matrix
  comparisons match exactly; 141 distinct focused controls are covered.
  Source arithmetic is bounded to documented CPU binary64 paths.
- Constructor warnings belong to public construction, matching native behavior.
  ConformalSquare now emits four actual warnings; its 1,222 saved arrays and
  five images are unchanged. The page retains two actual output blocks.
- Chebfun2v now covers all 22 native three-component predicates, with numeric
  addition, elementwise multiplication and matrix multiplication adapters.
- Native NE, CELL2QUASI, breakpoint values and Deltafun coefficient delegation
  gaps are repaired. Numeric vector infinity-norm adapters now use max-absolute
  semantics; qualification records explicitly correct earlier sum-based checks.
- Strict endpoint and delta-convolution CI failures were repaired at their
  original bounds. Inverse cold compilation remains a performance target.

Verification still required:

- Full native API and assertion coverage, including full vector-field
  argument parsing and wider distribution/threshold-edge behavior.
- Remaining page computations, printed output, random inputs and figure details.
  The frozen `64cf5c3b` inventory finds all **322 pages**, **2,408 MATLAB blocks**,
  original prose/math and **1,305 reference-sized assets**. These static checks
  do not establish numerical or pixel parity. Nine blank blocks remain in that
  inventory: four historical renderer warnings and five algorithm/plot warnings.
  EightShades source computation and warning behavior are under active repair.
- CPU performance, remaining disk/Hermite correctness and memory issues,
  general DDE behavior, and remaining JAX-only migration.
- Complete green CI on current main. Local focused results do not establish
  a green full CI run; no green-main claim is made here.

Evidence: dated qualification records in `docs/`, especially
`pde15s_cpu_source_arithmetic_20261011.json`,
`constructor_warning_policy_source_cpu_20261011.json`,
`conformal_square_warning_policy_cpu_20261011.json`,
`threecomponents_native_source_cpu_20261011.json`, and
`page_inventory_64cf5c3b_cpu_20261011.json`.
Live status and original process receipts remain under
`/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/`.

---

# Parity status — 2026-10-09

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
