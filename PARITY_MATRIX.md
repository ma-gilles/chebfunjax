# chebfunjax ↔ MATLAB Chebfun — Parity Status

## Current qualification (2026-10-08)

Full parity with Chebfun `7574c77680d7e82b79626300bf255498271a72df`
remains incomplete. `HANDOFF_CODEX.md` takes precedence over the historical
snapshot below and the older `HANDOFF.md`.

At local commit `a817dfed`, the static
MATLAB test-file inventory is:

| Classification | Files |
|---|---:|
| Present without literal skip/xfail markers | 1,051 |
| Contains skip/skipif/xfail markers | 27 |
| Module-skipped | 24 |
| Missing after explicit consolidated test mappings | 0 |
| Total pinned MATLAB test files | 1,102 |

These counts describe files and literal markers, not verified assertions or
numerical parity. All 24 module skips are in `adchebfun`. Four formerly missing `chebgui`
export cohorts now map explicitly to `test_toFile_exporters_matlab.py` and
passed with all64 original demos in the149-case CPU qualification. This does
not establish general GUI or AD parity; both remain in the full scope.
Existing aggregate tests also need clause and tolerance audits: recent fixes
restored an omitted Helmholtz field and the original norm bound.

Immutable inventory with source and port hashes:
`/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/matlab_test_static_inventory_a817dfed_20261008.json`.
The refreshed PNG audit at a817dfed reports129 image-size mismatches across
27 of322 pages,1304 mapped slots and one unresolved AtmosphericTemperature
slot. Mapping is inherited; matching dimensions alone does not qualify figures.
Evidence: `/scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/figure_size_current_a817dfed_20261008.json`. Local
numerical packages have individual evidence in `HANDOFF_CODEX.md`; current
local commits are not published, and green CI on their exact head is unproved.

Latest bounded packages at this checkpoint:

- AD calculus and independent seeds restore two original test modules. The
  seed package passes179 CPU checks, including all five original seed clauses
  at1e-15 and the unchanged154-case calculus cohort.
- Disk Helmholtz coefficient input now performs the source Fourier/Chebyshev
  resizing.66 alias controls and the original coefficient-input assertion
  pass. The separate k=7 case still fails its original2.22e-8 bound with
  error3.76e-8; its diagnostic is preserved.
- Rouche regenerates all seven600x253 figures, including the previously
  omitted seventh script output. Four equal-axis helper controls pass;
  source stdout is empty and Markdown is unchanged. Pixel/font parity is
  not established by these checks.

Acceptance records in the shared evidence directory are
`ad_seed_root_acceptance_20261008.json`,
`disk_alias_root_acceptance_20261008.json`, and
`rouche_root_acceptance_20261008.json`.

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
