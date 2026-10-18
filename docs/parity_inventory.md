# Chebfun MATLAB parity inventory

This is a deterministic, source-only tracking baseline generated from
Chebfunjax `45c04b294107ee1b4b5155650b39fa760a63af70` and native Chebfun
`7574c77680d7e82b79626300bf255498271a72df`. It does not establish parity.
Full-item qualification statuses are `unverified` in
[`parity_inventory.json`](parity_inventory.json). Port-path matches are
navigation hints only. The generator requires explicit `--python-root` and `--out` arguments. It
checks the native source pin and clean input trees, records the selected Python
commit, and refuses to overwrite an existing ledger. Generate a fresh baseline
and reconcile reviewed evidence; regeneration must never erase completed work.

Recent scoped qualifications are linked from the
[current checklist](parity_checklist.md) and recorded per source in the JSON
ledger. Full-item statuses remain unverified.

## Inventory scope

| Level | Count | Tracked state |
|---|---:|---|
| Native library `.m` files outside `tests/` | 2,333 | Source and semantic assertion coverage unverified |
| Function declaration lines detected in library sources | 3,707 | Implementation and semantic assertion coverage unverified |
| Original MATLAB test `.m` files | 1,102 | Runtime, assertions, and tolerances unverified |
| Function declaration lines detected in original tests | 1,256 | Assertion/runtime coverage unverified |
| Cached example HTML pages | 322 | Prose, computations, stdout, figure content, figure size, and performance separately unverified |

The file inventory is distinct from a public API inventory. A MATLAB class
file can define methods and local/nested functions; package, private, and
class visibility are not inferred here. **Parser limitation:** declarations
are found with a line-based regular expression, not a MATLAB parser. It can
miss multiline or unusual declarations and does not determine declaration
scope, class dispatch, visibility, or whether a function is public. The
3,707/1,256 counts are detected declaration lines, not public API counts.

MATLAB example paths and Python script candidates are separate fields.
Candidate port paths are matched by basename and may be incomplete or
ambiguous. Their presence, original test-file counts, or historical package
passes never set an inventory status. Evidence must be reviewed and attached
to the specific function/assertion/page dimension before any status changes.

## Current status and scoped evidence

Latest [source numerical integration](numerical_source_integration_cpu_20261018.json):168PASS/1strictXF and228PASS/2strictXF in two actual-parent stages (23protocol-node overlap). Finite tiny tangents, literal ACA division and source Dirac-aware extrema repaired at unchanged original bounds; all1131owners/13lanes/Hermite12 retained. Full Checkmark numeric loop completes but degree5, exact provider/knot words and standalone page/figures/stdout remain open. Full-item statuses are unchanged.

Latest [paired adaptive/EQUI qualification](chebfun2_paired_equi_parent_cpu_20261018.json):196actual parent PASS/7strictXFAIL; full original constructor2 and EQUI/mixed-technology child tests, per-axis grids/rank budgets and source sample-before-finalize staging. Seven factor/cutoff/location contracts, callback orientation and full-item statuses remain open.

Latest [minimax panel compilation qualification](minimax_panel_compilation_cpu_20261018.json):44actual parent PASS/66isolated PASS, ten original polynomial clauses unchanged. Full degree3Checkmark replay preserves517callback words and19snapshot arrays; recorded383.88→296.44s is not matched MATLAB performance. Seven-curve page and whole-item statuses remain unverified.

Latest [edge failure handling and original tests](edge_roundoff_original_tests_cpu_20261018.json):44parentPASS, all7default originalpredicates at unchanged bounds; actualdegree3 Checkmark representation recovered. Exact knot/provider words/fullpage/speed remain open. Full-item statuses remain unverified.

Latest [default numeric disk route](disk_numeric_direct_cpu_20261018.json):145parentPASS/5strictXFAIL, native rank-cap limitations retained. [Checkmark edge localization](checkmark_edge_localization_cpu_20261018.json) narrows the next diagnosis; no edge repair claimed. Full-item statuses remain unverified.

Latest [disk shared-matrix simplification](disk_matrix_simplify_cpu_20261018.json):195parentPASS/2strictXFAIL; complete PhaseTwo matrices, native global tolerance and original rank102 physical/storage checks. Full-item statuses remain unverified.

Latest [coefficient and paired fixed preferences](chebfun2_coefficient_preferences_cpu_20261018.json):141parentPASS/1strictfactorXFAIL;45native constructors and prior preference/CI controls. Fourier provider factor choices and full-item qualification remain open.

Latest [Chebfun2 preferences/disk source grids](chebfun2_disk_preferences_spacing_cpu_20261018.json):194parentPASS in two disjoint scopes; single-preference fixed/numeric/rank support and native sampling/tolerance/rank stopping. Remaining full-item statuses stay unverified. Prior constructor timeout attribution is explicitly corrected in this report.

Latest [TAN phase/disk errors/division](tan_disk_remaining_parent_cpu_20261018.json):251parentPASS/1strictXFAIL; original native assertions/bounds retained. Expected failures/full-item qualification remain open.

Latest [root filter and adaptive scalar qualification](root_disk_parent_cpu_20261018.json):223parent passes in two disjoint scopes. Inverse first-call compilation reduced; native words/speed and full-item statuses remain unverified.

Latest [four-method unary and numeric disk construction qualification](unary_disk_constructor_parent_cpu_20261018.json):231parent passes in two disjoint scoped gates; child full unary2,064PASS22XFAIL and disk376PASS3XFAIL are separate scopes. Four original unary records repaired; remaining full-item statuses stay unverified.

Latest [root mapping, SIN overflow and disk tolerance qualification](roots_sin_tolerance_parent_cpu_20261018.json): 795 scientific parent passes / 3 strict expected failures, with a separate targeted TAN preservation gate and exact CI file-partition audit. Specific native rows carry scoped evidence; whole-item statuses remain unverified.

Latest [SIN output-underflow and JAX disk PhaseOne qualification](sin_disk_jax_parent_cpu_20261018.json):387parentPASS/3strictXFAIL; child full unary and disk gates are separate scopes. Native disk constructor evidence is attached without promoting complete source/API/test statuses.

The [completed/to-complete checklist](parity_checklist.md) is the running human
work list. This inventory supplies the denominator and per-item evidence slots.
Scoped GET evidence for original predicates 31 and 35–38 is attached to the
native `@chebfun/get.m` and `tests/chebfun/test_get.m` rows. It comes from an
isolated candidate based on `43e5a5b7`, and does not qualify the complete GET
function, test file or public API.
The earlier ThreeBody observation records numerical differences and one
missing pre-run standard-library guard entry. A newer combined-source run
completed with all 1,290 actual imports preguarded; its numerical artifacts
match the earlier Python observation exactly, leaving the MATLAB discrepancy
open. See [combined review](residue_render_joint_cpu_20261017.json).
Other historical evidence still needs reconciliation into these rows; this
is an evidence ledger, not a finished parity audit.


See the [repository parity status](parity_status.md), the
[current goal checklist](/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/GOAL_CHECKLIST.md),
and [current goal status](/scratch/gpfs/CRYOEM/gilleslab/mg6942/_agent_scratch/CURRENT_GOAL_STATUS.md).
These documents record specific completed scopes and explicit open work; this
inventory does not promote those scopes into file-wide or goal-wide completion.
For example, the status links to scoped
[polynomial conversion / CI lifecycle evidence](poly_port_joint_cpu_20261017.json)
and [Spinop2 norm evidence](spinop2_gl_norm_cpu_20261017.json). Those receipts
are not universal library, test, or page evidence, so all corresponding
inventory statuses remain unverified until reviewed at the relevant
source/function/assertion level.

Per-page rows now link the official reference archive plus all 1,305 PNG
dimension and decoded-pixel observations across 322 pages. All 1,305 local PNG
dimensions match their recorded references; decoded RGBA comparisons are exact
for 0 and differ for 1,305. These are observations, not page qualification:
all prose, computation, stdout, figure content, figure size, and performance
statuses remain `unverified`. The reference binding and image measurements do
not establish computation or raster parity.

The newest ThreeBody color-cycle rerun is linked separately, with unchanged
retained numerical artifacts and new hashes for images 2 and 4. Basic PDE
consumer and scoped color-cycle evidence is also attached to native source
rows; full-item statuses remain unverified.

The evidence register in `parity_inventory.json` records artifact paths,
SHA-256 values, source commits and scope limitations, including the current
ThreeBody image observation. The original global dimension/pixel artifacts
remain linked to their historical 43e5a5b7 image binding. Cached HTML structure or
port presence does not qualify those dimensions.

## Regeneration

Using a clean Python source checkout and a new output path:

```bash
python scripts/build_parity_inventory.py \
  --python-root /path/to/chebfunjax \
  --out /shared/scratch/new_parity_baseline.json
```

The generator creates an unverified baseline; attached reviewed evidence and
manual tracking metadata must be reconciled separately into the live ledger.
It refuses an existing output path to prevent erasing that work.

The generator uses source trees only; it runs no MATLAB, Python package code,
examples, or numerical tests. Native source and cached-page SHA-256 values are
stored in the JSON. The adjacent `parity_inventory.sha256` records checksums
for the generator, JSON, and this summary.

The spin/complex-constructor package adds scoped evidence to selected native
source rows and the std3 original predicate. Seven complex scientific controls
pass, while the final runtime auditor reports one unpreguarded dynamic file;
full runtime closure and full-item parity are not claimed. See
[review](spin_complex_integration_cpu_20261017.json).

## Paired original test invocations — 2026-10-17

Five default original MATLAB test invocations pass all **141 predicates**;
the matching current-source Python selection passes **133 nodes**, with no
failures or skips. The scopes are GET (38), interp1 (35), ratinterp (23),
Chebtech compose (44), and the sole constant std3 predicate (1). Rational
clauses are grouped into 15 Python nodes. Each original-test inventory row
now has a `verified_scopes` entry bound to source hashes and
[paired evidence](original_five_predicates_cpu_20261017.json). Full-file and
API statuses remain unverified: alternate preferences, unsupported inputs,
and broader semantics are outside these invocations. This gate observes
external runtime origins without claiming every dependency was preguarded.

## NDCT and exact-seeded Chebtech evaluation — 2026-10-17

The original Chebtech feval row now records another paired scope: all 18
default predicates pass using MATLAB's actual seeded vector, with unchanged
strict bounds and Fortran-order reshapes. Native also confirms that the
fixture matches its draw bit-for-bit. Existing linspace controls remain
supplemental. [Seeded evidence](chebtech_feval_seeded_cpu_20261017.json).

The missing strict large-input NDCT dispatch is restored for both kinds.
Eight same-word native captures agree within 2.45e-15; 23 authored JAX controls
qualify the real-node polynomial differentiation convention and first-import
JIT fix. BestApprox completes in 151.93s versus about 201s in the retained local
baseline; these are descriptive shared-host observations. Its five figures
match 600×269 reference sizes and all still differ in pixels. The rational8
curve discrepancy remains 8.86e-8. Per-page observations are appended without
promoting full computation, graphics, API or matched-performance statuses.
[Integrated evidence](chebtech_ndct_cpu_20261017.json).

The packed Clenshaw and iteration-color package adds scoped source/performance evidence and a whole ComplexArcLength observation. Four dimensions match; axes/markers/raster and representation differences remain. See [review](inverse_packed_color_cpu_20261017.json). Full-item statuses are unchanged.

A [manual source review](chebgui_inventory_mapping_20261017.json) maps the four grouped Chebgui exporter cohorts missed by basename matching. All 64 demo fixtures match pinned native bytes. This supplies navigation only; it adds no runtime or complete assertion qualification.

Factory/conversion and rectangle integration now have source-bound evidence in
[the joint CPU report](quasimatrix_integral2_cpu_20261017.json). Original
cell2quasi predicate 1 and integral2 predicates 1–3 have paired native/Python
scopes attached; broad API/item statuses remain unverified.

The [paired exporter report](chebgui_paired_originals_cpu_20261017.json)
adds four native/Python smoke scopes to the original Chebgui exporter rows.
These qualify successful export, not full API or generated-text equality.

[Legendre coefficient evidence](legcoeffs_original_cpu_20261017.json) adds
all six literal Python original predicates. Native JSON records six true
results, with no retained terminal receipt; the scope explicitly distinguishes
that result-only observation from supervised native qualification.

[ND evaluation and subspace evidence](ndarray_subspace_cpu_20261017.json)
records the public adapter fix and six original subspace predicates with the
source distinct breakpoints and row case. The broader subspace API and its
existing host numerical implementation remain open.

[Sphere optimization evidence](sphere_extrema_tie_cpu_20261017.json) attaches
ten original default predicates with a byte-identical forward source bridge.
[Open numerical investigations](numerical_parity_gaps_20261017.md) separately
record unresolved transform-sensitive PDE/Hermite/rational paths without
promoting diagnostic observations into parity completion.

[Subspace and unbounded inner product](subspace_unbnd_inner_cpu_20261017.json)
attaches the six and three original predicates, respectively. Full API and
near-rank quantitative subspace parity remain open.

[First-kind Chebfun2 and string operators](c1_nodots_cpu_20261017.json)
attaches three technology predicates and seven no-dots predicates, with
component and current-source integration evidence. Full API remains open.

[Complex-fit pre-IFFT follow-up](ratinterp_preifft_cpu_20261017.json)
localizes same-input differences to transform/normalization arithmetic and
proves public native capture neutrality for this input; no parity credit.

Both complex-fit diagnostic scopes attach to native root-level `ratinterp.m`.
Earlier attachment code incorrectly looked for `@chebfun/ratinterp.m`; that
nonexistent target attached no row. The evidence register was present and is
now linked to the actual source row; full-item status remains unverified.

[Subspace empty/rank-zero evidence](subspace_empty_rank0_cpu_20261017.json)
records native-matched values/errors and preserves the unrelated moving-HEAD
guard failure. Exact tested function/source bridges carry the scoped Python
gates; broader API and near-rank quantitative fidelity remain open.

## Published roots, power, extrema and division scopes — 2026-10-17

[Scoped reconciliation](scoped_originals_inventory_reconciliation_20261017.json)
adds precise function evidence and original-test `verified_scopes` for roots
(19 predicates), power (41), minandmax (14), and rdivide (26 final slots).
Literal test navigation paths are attached without removing historical
basename hints. Every full-item and function status remains unchanged and
unverified. Source hashes and bindings refer to the published reports,
not a new execution against the latest main branch.

Power retains its 24 final reruns / 17 unchanged-node bridge distinction.
The later 18-control zero-preflight report supersedes predicate 33's extra
warning only for the supported bounded fractional-power path; standalone
norm/extrema and nonfinite complex primitives remain open. Division's
28 assignments include overwritten slots 13 and 24. The overwritten slot 24
native failure remains uncredited. Neither its strict-xfail nor the 26 true
final slots proves all 28 assigned expressions. No direct gamma-method
qualification is inferred from string gamma input.

## Padé source and original predicates — 2026-10-17

The `padeapprox.m` and original `tests/misc/test_padeapprox.m` rows now link
[published evidence](pade_jax_source_cpu_20261017.json): all three default
native predicates and the corresponding Python tests pass. The final 17-check
Python gate also covers ten existing authored tests and four new JAX/correctness
contracts. Full function, API and alternate-parameter statuses remain unverified;
this attachment preserves all prior item statuses and page records.

## Ten remaining output-slot discrepancies — 2026-10-17

[Reviewed mapping](output_slot_reconciliation_cpu_20261017.json) attaches scoped
stdout evidence to ten page rows. Four blank blocks contain native JOGL graphics
messages; the other three contain native Newton, complex-plot or unresolved
constructor warnings. Three count differences concern EightShades, ZetaZeros
and TransientGrowth warnings. No missing numerical print was found among these
specific slots. These differences remain explicit; no page or stdout status is
promoted, and no warning is synthesized from cached native output.

## Current Chebfun3T unresolved predicate — 2026-10-17

[Fresh current-source evidence](chebfun3t_current_transform_cpu_20261017.json)
records original constructor predicate 7 still failing at its unchanged
`100*eps` bound after the shared inverse-transform change. The error is
`2.864375403532904e-14`; the bound is `2.220446049250313e-14`.
The strict expected-failure marker remains. This negative result is attached
to the constructor and original-test rows; no full-item status changes.

## Lorenz warning capture and TransientGrowth constructor — 2026-10-17

[Lorenz actual execution](lorenz_warning_cpu_20261017.json) closes its previously
blank fourth output slot with the actual complex-plot warning. Explicit source
slot markers prevent punctuation differences from misgrouping it. The unchanged
numerical/plot operands run in 33.63 seconds with all 1,381 runtime files
preguarded. Four reference-size figures complete; all four reference rasters
still differ. Numerical stdout/native trajectory/pole parity remains open.

[TransientGrowth constructor diagnosis](transient_growth_constructor_cpu_20261017.json)
finds 28 happy pieces and zero warning attempts/emissions with original operands
and preferences. The cached native unresolved warning remains different; no
warning is synthesized and no full example/figure/energy parity is claimed.
These two page rows now link scoped evidence; full-item statuses are unchanged.

## Padé optional default grammar — 2026-10-17

[Fresh empty-default evidence](pade_optional_defaults_cpu_20261017.json) attaches
the six repaired empty-argument cases and final 23-check Python gate to the
Padé source/test entries. A fresh pinned MATLAB invocation confirms empty
numeric/callable defaults and nonempty zero tolerance. Full-item statuses
remain unverified; this is a source-bound grammar scope.

## ThreePlanets actual warning visibility — 2026-10-17

[Parent review](threeplanets_parent_cpu_20261017.json) links the full actual
application, unchanged four Python frames and zero solve warnings. The raw
resource receipt retains an independently diagnosed root-only exit race;
no bare uncensored claim is made. The cached warning, native trajectory and
existing SciPy highorder provider remain open. Page/figure files and full-item
statuses are unchanged.

## Legacy ChebPadé scoped JAX conversion — 2026-10-17

[Evidence](legacy_chebpade_cpu_20261017.json) attaches15authored Python checks
and7fresh pinned-native numerical controls to the ChebPadé library row.
This is not an original MATLAB test-file qualification. Exact rank-warning,
full API and performance gaps remain. All full-item statuses are unchanged.

## Public ChebPadé source padding — 2026-10-17

[Evidence](public_chebpade_padding_cpu_20261017.json) records51regression
checks and6additional public matched-native-draw controls. Exact MATLAB random
bits, full API and performance remain unqualified. All full-item statuses stay
unchanged; existing fixture tests do not constitute a fresh native run.

[Shared generalized eig integration](generalized_qz_parent_cpu_20261018.json)
adds scoped provider/C1/original-predicate evidence.60parent checks across two
explicitly bridged stages do not certify whole files or transformations. Native
FFT/HC2C4 and complex QR observations remain separately registered diagnostics.
All whole-item status fields are unchanged.
