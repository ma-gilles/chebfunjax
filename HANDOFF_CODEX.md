# chebfunjax: handoff to a Codex agent (2026-10-02)

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
