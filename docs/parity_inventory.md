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
and [current goal status](/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/CURRENT_GOAL_STATUS.md).
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
