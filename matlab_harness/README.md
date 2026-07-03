# MATLAB golden-reference parity harness

This directory pins chebfunjax numeric outputs against **MATLAB Chebfun**
(the ground truth) at `rtol = 1e-12`. It is the *function axis* of the
whole-codebase parity matrix (`/scratch/.../parity_matrix/master_matrix.csv`).
Golden-ref comparison catches construction bugs the unit suite never touches —
this pass alone found three (chebfun2d ACA tolerance, diskfun + spherefun
row/col-major flat-index). Any agent can add a module by mirroring the pattern
below.

## The pipeline (per module)

1. **`matlab_harness/refs/<module>_refs.m`** — a MATLAB script that builds a
   `ref` struct of named fields and saves `tests/references/<module>.mat`.
2. **`tests/references/<module>.mat`** — the committed golden values.
3. **`tests/test_<module>/test_<module>_matlab.py`** — a `pytest` that loads
   the `.mat` and asserts chebfunjax matches at `rtol 1e-12`.

`generate_refs.m` runs every `refs/*.m` in one go; you do **not** edit it.

## Rules (non-negotiable)

- **Naming:** name the ref script `<module>_refs.m`, **never** `<module>.m`, if
  `<module>` is a Chebfun class (`chebfun`, `chebfun2`, `diskfun`, `spherefun`,
  `ballfun`, …). A bare `diskfun.m` collides with Chebfun's `@diskfun` class and
  MATLAB's `run()` fails to resolve it. The saved `.mat` may keep the plain name
  (`diskfun.mat`) — only the *script* filename collides.
- **rtol policy:** pin VALUES at `rtol 1e-12` (project Gate 3). A value mismatch
  (eval / sum / norm / integral off) is a **BUG** — report it, do **not** widen
  the tolerance. Only genuinely implementation-dependent *discrete* metrics
  (e.g. numerical `rank` right at the stopping-tolerance boundary) may be
  documented per Gate 3, and even then prefer a strict `xfail` over a widened
  assertion.
- **Bug found → report, don't hide.** Send the team lead the exact repro. Mark
  just the failing case as a strict `pytest.param(..., marks=pytest.mark.xfail(
  reason="… reported"))` (or a conditional `pytest.xfail()` inside the method)
  so the suite stays green while the bug is tracked. When the library fix lands,
  remove the xfail and re-run.
- **No commits.** Leave everything in the working tree; the team lead integrates.
- **Lint-clean** the test file (`ruff check … --select E,F,W,I`). Common traps:
  `E741` (don't name a var/param `l`, `I`, or `O`) and `E501` (line length).
  Do NOT run ruff on the `.m` file (it's MATLAB, not Python).

## Coordinate & eval conventions — verify BEFORE generating

chebfunjax and MATLAB often differ in coordinate convention; build the MATLAB
ref in the mode that matches chebfunjax's **native** one, and confirm it
empirically (construct `x`/`y`/`z` and eval at known points) first.

| module    | chebfunjax construct / eval | MATLAB ref built as |
|-----------|-----------------------------|---------------------|
| chebfun2  | `chebfun2(f(x,y))`, element-wise eval | `chebfun2(@(x,y) …)` |
| diskfun   | `Diskfun.from_function(f(theta,r))`, θ∈[-π,π], r∈[0,1] | `diskfun(@(t,r) …, 'polar')`, `feval(f,t,r,'polar')` |
| spherefun | `Spherefun.from_function(f(lam,theta))`, λ∈[-π,π], θ∈[0,π] colatitude | `spherefun(@(lam,th) …)` (2-arg = spherical), `feval(f,lam,th)` |
| ballfun   | `Ballfun.from_function(f(x,y,z))` Cartesian; eval `f(r,lam,th)` **grid** (x=r sinθ cosλ, y=r sinθ sinλ, z=r cosθ) | `ballfun(@(x,y,z) …)`, `feval(f,r,lam,th,'spherical')` |

**Element-wise vs grid eval:** Chebfun2/Diskfun/Spherefun evaluate element-wise
on matched point vectors. **Ballfun `__call__` does a meshgrid eval** and
returns a 3-D tensor — for a single matched point index `[0,0,0]`.

## Bug-pattern watchlist

- **Row/col-major flat-index:** any `j = idx % rows; k = idx // rows` (or
  `unravel` with `order='F'`) applied to a **numpy** `argmax` flat index is a bug
  — numpy is row-major, so the correct conversion is `j, k = divmod(idx, ncols)`.
  This exact bug froze diskfun/spherefun GE pivots (scrambled pivots →
  zero-divisions → NaN residuals → rank stuck). Grep new construction code for it.
- **Stopping tolerance too loose:** if chebfunjax `rank` is one *below* MATLAB
  while eval/sum still ~match, the low-rank stopping tolerance is under-scaled
  (the chebfun2d ACA bug). If eval is grossly off (%-level), it's worse.

## MATLAB invocation (login node, no GPU, ~1–2 min)

```bash
/usr/licensed/matlab-R2025b/bin/matlab -batch \
  "addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); \
   run('/home/mg6942/chebfunjax/matlab_harness/refs/<module>_refs.m')"
```

Run it as a background command (it's slow); poll for `tests/references/<module>.mat`.
`chebfun_matlab_ref` is the Chebfun clone at commit `7574c77` (see `project.conf`).

## After each module

1. Run the pytest; triage every mismatch (bug report + xfail, never a widen).
2. Update the master matrix: set the covered `function` rows to `PASS`
   (`metric = matlab golden-ref rtol 1e-12`) or `BUG` with a note + repro pointer.
3. Report the module to the team lead: files, pass/xfail counts, and any bug repro.

## Template

Copy `matlab_harness/refs/chebfun2d.m` + `tests/test_chebfun2d/test_chebfun2_matlab.py`
(evaluation, integral, norm, rank on a small function battery incl. one
higher-rank case like `exp(x)` that stresses the low-rank construction).
The test uses the module-level lazy guard:

```python
_REF_PATH = Path(__file__).resolve().parents[1] / "references" / "<module>.mat"
if not _REF_PATH.exists():
    pytest.skip("<module>.mat golden ref not generated (run matlab_harness/refs/<module>_refs.m)",
                allow_module_level=True)
_REF = scipy.io.loadmat(str(_REF_PATH), squeeze_me=True)
```
