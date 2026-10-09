# Parity status — 2026-10-09

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
