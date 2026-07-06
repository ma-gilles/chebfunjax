# chebfunjax ↔ MATLAB Chebfun — Master Parity Matrix

*Formalizes task #23. Snapshot maintained by Claude Opus 4.8; see
`HANDOFF.md` for the narrative and `STATUS.md` for module history.*

The parity effort spans three axes — **functions** (numerical
correctness), **plots** (pixel-faithful renders), and **examples**
(chebfun.org reproductions). This document tallies verified coverage on
each.

## 1. Summary

| Axis | Population | Verified | Method |
|---|---:|---:|---|
| Functions / methods | ~473 public | ~2,717 tests pass (3,073 collected) | unit + golden-ref |
| MATLAB golden-ref (machine precision) | — | **488 tests / 34 `*_matlab.py` files** | `.mat` fixtures vs MATLAB R2025b, rtol 1e-12–1e-13 |
| Guide figures | 323 | 323 regenerated | `compare_plots` + montage |
| Example figures | 826 numbered (1,594 images total) | 826 genuinely computed; 631 (76%) pass strict 0.06 gate | `compare_plots` badness ≤ 0.06 |
| Example categories | 21 | 21 complete | per-block regeneration |

Both suites green: `test-fast` **2,569 passed / 4 skipped**;
MATLAB-marked **488 collected, all pass** (14 skipped where a ref is
absent, 1 documented xfail).

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

- **Guide (ATAP-style docs):** 323/323 figures regenerated; 19/20
  chapters at genuine parity, ch.17 now unblocked by the spherefun
  calculus layer.
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

## 5. Known gaps (see HANDOFF.md §4)

- **#9** deltas through `Chebfun.diff` (needs a `deltas` field on the
  core eqx.Module).
- **#13** remaining `cheb.gallery` entries: gamma (blowup),
  vandermonde/vandercheb (quasimatrix), daubechies (wavelets), blasius
  (BVP initial guess). 22/27 done.
- **#24 (part)** IVP → time-marching routing (optimization; collocation
  already solves IVPs).
- chebmatrix / linop / chebop2 / chebgui golden-ref ports.
