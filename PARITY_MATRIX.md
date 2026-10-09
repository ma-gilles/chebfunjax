# chebfunjax ↔ MATLAB Chebfun — Parity Status

For the current summary, see [STATUS.md](STATUS.md) and
[HANDOFF_CODEX.md](HANDOFF_CODEX.md). The dated entries below are historical
qualification records; their active assignments and environment availability
are not current status. Full parity remains incomplete.

## Historical CPU qualification (2026-10-08, local 6cda66b9)

Full parity remains incomplete. Latest static inventory revalidated all 1,102
source/port mappings: **1,075 present, 19 with skip/xfail markers, 8 module-skipped**.
File presence does not establish complete assertions or runtime parity.

| Commit | Verified change | CPU evidence |
| --- | --- | --- |
| ca70c195 | Source repmat forms, point values and orientation | 64 passing tests; native seeded source inputs |
| ea3fb9e2 | Exact singular equality, source cancellation, complex-transform conjugacy | 207 passing tests; fresh MATLAB equality/complex predicates pass |
| 73772a94 | Explicit small Hermite REC source outcomes, including source defects | 60 passing tests against captured native outcomes; bounds unchanged |
| 6cda66b9 | AD product integrals and plain cumprod | 65 passing tests plus 12 integration checks; six original clauses |

Evidence directory: `/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/`.
See `repmat_root_acceptance_20261008.json`,
`singfun_equality_root_acceptance_20261008.json`,
`hermite_small_root_acceptance_20261008.json`,
`ad_cumprod_prod_root_acceptance_20261008.json`, and
`matlab_test_static_inventory_6cda66b9_20261008.json`.

The complex cosine transform uses two real JAX IFFTs, an explicit numerical
adaptation preserving conjugacy; no performance claim. The real-valued eager
NumPy path still needs porting. Small Hermite ASY, singular/delta repmat rows,
and general complex/array product-integral behavior remain open.

Three workers continue public JAX Airy/AD functions, small Hermite ASY, and
LaneEmden complex parameter correctness. The earlier two-figure LaneEmden run
is not accepted for publication: review found imaginary equation components
were discarded. A corrected complex solve passes a manufactured case; final
adaptive/page qualification remains outstanding. Heavy examples run serially.

Last confirmed remote main is **8a6d9c80**. The execution profile subsequently
changed to restricted networking; the GitHub API now fails to connect, so CI
status cannot be refreshed and newer commits are local. Previous successful
pushes and MATLAB captures remain valid historical evidence. Full source/test,
page/figure, performance and exact-head CI gates remain required.

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

## Historical snapshot — not current completion evidence

*Formalizes task #23. Snapshot maintained by Claude Opus 4.8; see
`HANDOFF.md` for the narrative and `STATUS.md` for module history.*

The parity effort spans three axes — **functions** (numerical
correctness), **plots** (pixel-faithful renders), and **examples**
(chebfun.org reproductions). This document tallies verified coverage on
each.

## 1. Summary

| Axis | Population | Verified | Method |
|---|---:|---:|---|
| Functions / methods | ~473 public | **2,728 test fns / 3,084 collected** | unit + golden-ref |
| MATLAB golden-ref (machine precision) | — | **488 tests / 34 `*_matlab.py` files** | `.mat` fixtures vs MATLAB R2025b, rtol 1e-12–1e-13 |
| Guide figures | 323 | 323 regenerated; all 20 chapters genuine | `compare_plots` + montage |
| Example figures | 826 numbered (1,594 images total) | 826 genuinely computed; 631 (76%) pass strict 0.06 gate | `compare_plots` badness ≤ 0.06 |
| Example categories | 21 | 21 complete | per-block regeneration |
| `cheb.gallery` / `gallerytrig` | 27 / 11 | 27 / 11 (full MATLAB set) | per-entry vs MATLAB |

Both suites green: `test-fast` **2,584 passed / 4 skipped**;
MATLAB-marked **488, all pass** (14 skipped where a ref is absent, 1
documented xfail).

## 2. Functions — golden-ref cross-validation coverage

Machine-precision MATLAB cross-validation now spans the whole
implemented library:

| Layer | Classes with a `*_matlab.py` golden-ref |
|---|---|
| Tech | chebtech1, chebtech2, trigtech |
| Fun | bndfun, singfun, deltafun, unbndfun |
| Chebfun 1D | core (×2 batches), extras, linalg/QR/SVD |
| Operators | chebop (operators, nonlinear ×2, extras, Mathieu, periodic) |
| 2D / 3D | chebfun2 (+extras), chebfun2v, chebfun3 (+extras), chebfun3v |
| Sphere | spherefun (+ calculus + poisson) |
| Disk | diskfun (+ calculus + poisson) |
| Ball | ballfun (+ calculus + poisson), ballfunv (+ div/curl/helmholtz) |
| Utils | quadrature, transforms, interpolation, diffmat, polynomials, aaa, minimax |
| Misc | spin (ETDRK4), autodiff, discretization |

**Not yet golden-ref cross-validated at the file level:** chebmatrix,
linop internal blocks, chebop2 (2D PDE class), chebgui. These are
covered (if at all) by independent Python tests, not exact MATLAB
comparison.

## 3. Plots — chebfun.org parity

- **Guide (ATAP-style docs):** 323/323 figures regenerated; **all 20
  chapters at genuine parity** — ch.17 now uses the library spherefun
  calculus (sphharm / laplacian / poisson), not scipy stand-ins.
- **Examples:** all 21 categories, 826 numbered figures genuinely
  computed. 631 (76%) pass the strict badness ≤ 0.06 gate; the other
  195 are content-verified but fall in four documented exception
  classes: 3D-renderer aspect (mae=0, hist≥0.99), seeded-random
  instance, data-dependent (unbundled climate data), and version-drift
  (page revision no longer published).

## 4. Examples — chebfun.org example pages

All 21 categories complete (roots, complex, geom, temp, approx, approx3,
linalg, integro, quad, cheb, calc, applics, fourier, opt, fun,
ode-random, stats, ode-eig, ode-linear, ode-nonlin, sphere). Every
per-block code snippet executes; ~60 broken stubs were replaced with
runnable code during the campaign.

## 4bis. MATLAB unit-test replication (tests/test_matlab_port)

Every non-chebgui MATLAB test file (1,095 across 43 modules) now has a
port counterpart under ``tests/test_matlab_port/<module>/`` -- either a
genuine port of its assertions at MATLAB tolerances, or a skip/xfail
stub naming the precise missing feature.  Authored by Opus-4.8-driven
subagents (wave 1, salvaged + verified) and Claude Fable 5 (everything
else).  The port drive found and fixed 8+ real library bugs and
documented ~10 more as evidence-carrying xfails (see MATLAB_PORT_LEDGER
in the audit scratch dir and the git log).

## 5. Known gaps

The entire library backlog (#8–#25) is closed:

- **#9 done** — `Chebfun.diff` attaches Dirac deltas at jumps (static
  `deltas` field; `sum` includes them).
- **#13 done** — `cheb.gallery` covers all **27/27** MATLAB entries
  (gamma via Singfun pole-pieces, daubechies via the cascade algorithm,
  vandermonde/vandercheb quasimatrices, blasius via a chebop initial
  guess).
- **#24 done** — chebop periodic BCs (Fourier collocation) and IVP
  time-marching routing.

Remaining (not library gaps, lower priority): chebmatrix / linop /
chebop2 / chebgui do not yet have file-level MATLAB golden-ref ports
(they have independent Python tests). Some sphere example pages use
`scipy.special.sph_harm_y` as a numerical reference / input generator
(analogous to test code using numpy), not as a library stand-in.
