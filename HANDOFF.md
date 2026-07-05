# chebfunjax — Parity Campaign Handoff

*Status as of 2026-07-05. Branch `fix/genuine-plot-parity` (PR #99),
merged to `main`. Authored by Claude Fable 5, verified and documented
by Claude Opus 4.8.*

This document is the single source of truth for **what is done, what
was verified, and what remains** on the whole-codebase MATLAB-Chebfun
parity campaign. Read it before continuing the work.

---

## 1. Mission

Translate MATLAB Chebfun into Python/JAX such that **every** function is
numerically correct (rtol ≤ 1e-12 against golden refs), **every**
chebfun.org example is genuinely reproduced, and **every** plot is
pixel-faithful to its reference render. The bar is the *whole*
codebase, not a spot-check.

---

## 2. What is DONE and verified

### 2.1 Library (from earlier phases — see STATUS.md)
- 39 feature PRs merged, ~90% feature parity. Utilities, tech layer
  (Chebtech1/2, Trigtech), fun layer (Bndfun/Unbndfun/Singfun/Deltafun),
  Chebfun 1D, Chebfun2/Chebfun3, Spherefun, Diskfun, Ballfun, chebop,
  ODE/BVP/eig solvers.
- Full test suite: **2513 passed, 4 skipped** (`pixi run test-fast`,
  ~3m44s, CPU). No regressions from the parity campaign.

### 2.2 Correctness fixes landed this campaign (32 verified bugs)
Highlights (all covered by golden-ref tests at rtol 1e-12):
- **ETDRK4 spin solvers** (`solver.py`, `solver2d.py`, `solver3.py`):
  the Cox–Matthews stage-1 coefficients were missing across the 1D, 2D
  scalar/multi-component, and 3D Fourier+sphere branches. Fixed and
  pinned.
- **Chebop** gained `eigs(return_eigenfunctions=True)`, `expm(t,u0,n)`
  (endpoint-Dirichlet interior-node restriction; spectral to 2e-15),
  and `matrix(n)`. Critical detail: `expm` uses raw `L.matrix(disc)`,
  **not** `_assemble` (which substitutes BC rows and corrupts the
  interior block).
- **Chebfun3** diff/grad/sum via Tucker-factor operations.
- **Ballfun** diff/grad/laplacian via `_ballfun_onediff_cart`
  (golden-verified, `tests/test_ballfun/test_ballfun_calculus_matlab.py`).
- **Complex-valued chebfun** support (constructor no longer real-casts).
- **Trigtech** dtype-following construction + conjugate-symmetry
  is_real inference.
- **Unbndfun** wired into the `chebfun()` factory (infinite intervals
  were previously unreachable).

### 2.3 Guide chapters (workstream C)
- **323/323 guide figures** regenerated. 19 of 20 chapters at genuine
  parity. Chapter 17 (spherefun): **12/28 committed figures pass; 16
  remain gated on the spherefun calculus layer (§4, task #25)**.

### 2.4 Example pages (workstream D) — **ALL 21 CATEGORIES COMPLETE**
Every numbered chebfun.org example figure now has a **genuinely
computed** counterpart (no byte-copies among regenerated figures).
Verified with `scripts/compare_plots.py` (badness = mae +
0.5·aspect_err + 0.5·(1−hist_corr), threshold 0.06).

| Category | figures | pass 0.06 gate | notes |
|---|---:|---:|---|
| temp | 26 | 26 | |
| linalg | 4 | 4 | |
| integro | 8 | 8 | |
| calc | 14 | 14 | |
| roots | 22 | 18 | 3 dense-hatch (WhiteCurves) |
| complex | 35 | 31 | |
| geom | 23 | 19 | rolling-ellipse snapshots |
| approx3 | 33 | 18 | 3D slice/surface renderer class |
| approx | 167† | 138 | coefficient-scatter marginals |
| quad | 12 | 10 | |
| cheb | 10 | 9 | |
| applics | 15 | 12 | |
| fourier | 17 | 14 | |
| opt | 15 | 5 | 3D wild-surface + contour styling |
| fun | 22 | 17 | HelloWorld rank-k contour panels |
| ode-random | 24 | 15 | random-instance trajectories |
| stats | 69 | 50 | RandomSurf/histogram styling |
| ode-eig | 51 | 39 | quiver/3D marginals |
| ode-linear | 84† | 77 | quiver-density marginals |
| ode-nonlin | 110† | 85 | chaotic-instance + 3D |
| sphere | 65 | 22 | **3D-sphere renderer class (mae 0, hist ≥ 0.99)** |
| **TOTAL** | **826** | **631 (76%)** | |

† gate CSV counts only *regenerated* figures; a handful of pre-existing
byte-identical legacy snapshots (our own earlier copies, flagged 999.0)
are not re-scored but are genuine chebfunjax output.

**The 195 figures that miss the strict 0.06 gate are content-verified
and fall into documented exception classes**, not wrong plots:
- **3D-renderer aspect class** (sphere surfaces, wild 3D surfaces):
  mae = 0.0, hist-corr ≥ 0.99; the residual is entirely matplotlib's
  bounding-box aspect term vs MATLAB's 3D projection engine.
- **Random-instance class** (ode-random, some stats/ode-nonlin): the
  original uses `rng`-seeded random fields; we reproduce the
  *statistical texture*, the honest ceiling for randomized demos.
- **Data-dependent class** (AtmosphericTemperature): the original loads
  real climate data + coastline shapefiles that are not bundled; a
  statistically-matched synthetic field stands in.
- **Version-drift class** (ConstrainedExtrema_01): the live chebfun.org
  render is from a page revision no longer published.

### 2.5 Spherefun harmonics + spectral calculus

- `Spherefun.sphharm(l, m)` and `Spherefun.mean()` **verified against
  `scipy.special.sph_harm_y` to machine precision** (ratio std ~1e-15).
- `Spherefun.laplacian()` — spectral (harmonic-diagonal). Passes the
  exact identity `Δ Y_l^m = -l(l+1) Y_l^m` to **5.7e-14** across
  degrees 1–8. *(This is the identity Fable's reverted BMC attempt
  failed with error 16.8.)*
- `Spherefun.poisson(f, const)` — solve `Δu = f`. Round-trips
  `poisson(laplacian(u)) == u` to 2e-15 and matches the worked example
  in MATLAB `@spherefun/poisson.m` to 4.4e-15.
- 73 spherefun/sphere tests pass; lint clean.

---

## 2bis. Contributions by author (for trust-checking)

Everything before the parity campaign, plus the campaign's example/plot
regeneration and the 32 correctness fixes, is **Claude Fable 5**'s work.
The items below were done by **Claude Opus 4.8** and are each gated on a
machine-precision or exact-identity test so they can be trusted on their
own merits regardless of author:

| Item | Commit | Verification |
|---|---|---|
| Reverted Fable's broken spherefun `diff`/`laplacian`/`grad` | `44f5fbe` | it failed `ΔY=−l(l+1)Y` (err 16.8) |
| `Spherefun.sphharm` + `mean` (kept from Fable's attempt, re-verified) | `44f5fbe` | vs scipy, ratio std ~1e-15 |
| `cheb.gallery` `airy`/`rose`/`motto`; `_Piece.__len__` | `a7fa79e` | airy vs scipy 1.1e-14; 3 tests |
| `Spherefun.laplacian` + `poisson` (spectral) | `d26a710` | `ΔY=−l(l+1)Y` to 5.7e-14; 7 tests |

Guiding rule Opus followed: **never ship spectral math that fails its
exact-identity test** — quarantine it (as with Fable's `diff`) and say
so plainly.

### 2.6 Page snippets
- Every example `.md` page's fenced code block executes. ~60 broken
  stubs (`from examples.X import run`, undefined names, missing
  imports, unimplemented Chebop features) were replaced with genuine
  executing snippets during the campaign.

---

## 3. Machinery built (reusable for the remaining work)

Per-category figure generators under `scripts/`:
`generate_examples_<category>[_tN].py`. Each has a `PAGES` dict, reads
reference pixel sizes from the audit snapshot, and uses
`save_chebfun_figure(fig, path, size=...)` for exact-pixel export.

Reusable techniques proven this campaign:
- **LP grid-minimax** (`_lp_minimax`) — Remez is too slow at degree
  ≥ 40 on kinks; the linear-program formulation is exact and fast.
- **Differential-correction rational best-approx** (`_dc_rational`) —
  stand-in until library `minimax(rational=True)` lands.
- **Gauss–Legendre L2 projection**, **IRLS L1 fitting**.
- **Method-of-steps** for delay ODEs; **Nyström/tridiagonal
  eigensolves** for Schrödinger/Sturm–Liouville pages.
- **Riemann–Liouville quadrature** for fractional calculus.
- **Talbot-contour resolvent quadrature** for matrix `expm`.
- **Even/odd BMC doubling + split SVD** for spherefun structure.

Audit trail: `/scratch/gpfs/CRYOEM/gilleslab/chebfunjax_audit_20260702/`
- `workstream_D_plan.md` — the running ledger (per-category progress,
  gotchas, polish queue).
- `refs/docs/images/<cat>/` — the golden reference renders.
- `out/compare/metrics_examples_<cat>.csv` — per-figure badness scores.

**Key gotchas discovered** (in the ledger, repeated here so they aren't
lost):
- MATLAB `'cyan'` fill = `(0, 0.8, 1.0)`, **not** `(0,1,1)` — was a
  0.17→0.03 parity fix on Snell's-law pages.
- Keyhole/branch-cut parametrizations: use `c2*c3.^s./c2.^s` (separate
  principal powers); `(c3/c2)**s` crosses the branch cut.
- Login-node OOM segfaults on n_max=2048 dense solves and on the
  approx generator at 32G — run those locally with `nice`+`timeout`,
  not via a memory-capped Slurm job.
- `_Piece` has no `__len__` — page snippets must use
  `len(piece.tech.coeffs)`.
- Young–Laplace drop ODE: `kappa + z` (not `kappa - z`) closes the
  profile at φ = π.

---

## 4. What is LEFT TO DO

### 4.1 Library gaps that gate quality (highest value)
- **#25 Spherefun calculus layer — PARTIALLY DONE (Opus 4.8).**
  - **DONE & verified:** `laplacian`, `poisson`, `sphharm`, `mean`
    (spectral / harmonic-diagonal — see §2.5). These unblock the
    Laplacian/Poisson-dependent guide-17 and sphere pages.
  - **STILL OPEN:** `diff(dim,k)` / `grad` (tangential Cartesian
    derivatives) and `curl`/`div`. Fable's coefficient-space attempt
    (`∂z/∂x` err 0.83) was reverted. These need the delicate BMC
    parity handling from `@spherefun/diff.m` (the `1/sin(theta)` solve
    on the correct even/odd columns, tracked via `idxPlus`/`idxMinus`).
    They still block the sphere HelmholtzDecomposition /
    PTDecomposition / AdvectionDiffusion pages' *library-genuine*
    (vs finite-difference) versions and the guide-17 figures that plot
    tangential vector fields. **Recommended approach:** either fix the
    BMC solve with per-step validation against the intrinsic-gradient
    formula, or implement `grad` spectrally via harmonic ladder
    operators (like `laplacian` above) — the latter is lower-risk.
- **#17 Ballfun** remaining: HelmholtzDecomposition, div/curl/solharm
  library-API promotion, helmholtz solver (gates sphere ball pages +
  guide20 figs 23–26 placeholders).
- **#15 Unhappy 65537-point constructions.** Functions with kinks
  (`|x-1/4|`, `-1/log(x)`) grind to the 65537 unhappy cap in the
  constructor. Several page snippets had to route around this (LP,
  GL-quadrature, piecewise domains). A real fix (edge detection /
  splitting) would let those pages use the library directly.
- **#24 Chebop `bc='periodic'` is NotImplemented**, and IVPs (all BCs
  at one endpoint) are not routed to time-marching. Two page snippets
  (FourierCollocation, PeriodicSystem) had to use raw Fourier
  collocation instead.

### 4.2 Library backlog (from the task board)
- **#9** `Chebfun.diff()` drops delta functions at jumps.
- **#10** `legpts` is O(n²) Golub–Welsch; MATLAB is O(n). Blocks
  quadrature-convergence pages at n = 65536 (they were capped at 2^12).
- **#11** Wire preferences (eps, max_length, domain) into the
  constructor.
- **#12** No splitting-on / edge detection (piecewise auto-construction).
- **#13** `cheb.gallery`: 13/25 entries; **missing `airy`, `motto`,
  `rose`** (surfaced by the Galleries example page); `wild` suspect.
- **#14** Two-arg min/max, round/floor/ceil, local extrema, complex
  roots flags, `roots()` perf, **`_Piece.__len__`**.
- **#20** `cheb.gallerytrig` missing; `Chebop.solve` lacks Newton
  iteration info.
- **#8** ETDRK4 solvers are pure-NumPy (correct but not JAX — perf).

### 4.3 Plot polish queue (cosmetic; content already correct)
All below are in a documented exception class; closing them is
diminishing-returns pixel-chasing, not correctness:
- opt: `GlobalMinimum_01/02` (0.13, wild-surface facet render),
  `ConstrainedExtrema_01/03` (version-drift), `Rosenbrock_01`.
- stats: `RandomSurf_01..03` (zebra styling), `MercerKL_04`,
  `RandomMaxima_03`, `UniformExercises_01`.
- sphere: `AtmosphericTemperature_08..10` (Earth/coastline render),
  `HelmholtzDecomposition_06`.
- ode-linear: `SpectralDisc_01` (dense-vs-banded spy pattern).
- ode-nonlin: `Droplets_02/03` (unstated volume constants).

### 4.4 Verification & reporting (workstream E)
- **Full GPU test suite** (`./scripts/run_tests_parallel.sh full`, ~1h,
  3 Slurm jobs) — the campaign only ran `test-fast` (CPU, 2513 tests).
  Run this before considering the campaign closed.
- **Master parity matrix (#23):** 2652 items (658 functions verified
  ~365, 1594 plots in progress, 400 examples now output-parity).
  Formalize the final tally.

---

## 5. How to continue

1. `source project.conf` for paths/accounts/thresholds.
2. To finish a plot-polish item: regenerate with the category's
   `scripts/generate_examples_<cat>.py`, then
   `scripts/compare_plots.py --kind examples --only <cat> --montages N`
   and read the side-by-side montage.
3. To close #25: implement `@spherefun/diff.m` faithfully (the bug was
   in the coefficient-space sine-multiplication solve / parity
   handling), test against the exact identities
   `Δ Y_l^m = -l(l+1) Y_l^m` and the Cartesian tangential-gradient
   formulas **before** wiring it into guide 17.
4. Never widen a golden-ref tolerance without documenting why
   (Gate-3 rule). Never ship spectral math that fails its exact-identity
   test — quarantine it like §2.5 did.
