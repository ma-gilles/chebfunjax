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
Historical scoped evidence is attached to the interpolation, GET and rational
tests and the ThreeBody page; it does not promote full-item qualification.
The ThreeBody observation explicitly records numerical differences and one
missing pre-run standard-library guard entry. Other historical evidence still
needs reconciliation into these rows; this is a baseline, not a finished audit.


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

Page output dimensions are independent: prose, computation, stdout, figure
content, figure size, and performance each require their own evidence. Cached
HTML structure or port presence does not qualify those dimensions.

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
