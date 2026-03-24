# chebfunjax — Phase 2 Plan

## Goal
Fill the remaining gaps to reach ~90% feature parity with MATLAB Chebfun.
Skip: GUI, MATLAB-specific interop, plotting (use matplotlib directly).

## Priority 1: ODE/PDE Solvers (the biggest gap)

| Unit | What | MATLAB source | Est. LOC |
|------|------|--------------|----------|
| V01 | `spinop2` — 2D PDE time-stepping | @spinop2/ (14 methods) | ~500 |
| V02 | `spinop3` + `spinopsphere` — 3D + sphere PDE | @spinop3/, @spinopsphere/ | ~600 |
| V03 | `imex` — IMEX time-stepping scheme | @imex/ (4 methods) | ~200 |
| V04 | ODE integrators: `ode45`, `ode113` on Chebfun | @chebfun/ode45.m, ode113.m | ~400 |
| V05 | `bvp4c`, `bvp5c` wrappers | @chebfun/bvp4c.m, bvp5c.m | ~300 |

## Priority 2: Operator AD (Fréchet derivatives)

| Unit | What | MATLAB source | Est. LOC |
|------|------|--------------|----------|
| V06 | `adchebfun` — AD for operators | @adchebfun/ (7 methods) | ~400 |
| V07 | `treeVar` — automatic linearization | @treeVar/ (12 methods) | ~500 |

These enable Newton iteration on nonlinear operators to work automatically
(currently Chebop uses finite-difference Jacobians as a fallback).

## Priority 3: Missing Chebfun Methods (~100 methods)

| Unit | What | Examples | Est. LOC |
|------|------|---------|----------|
| V08 | Array-valued / quasimatrix ops | horzcat, vertcat, cat, colon, subsref | ~400 |
| V09 | Interp/fitting | polyfit, interp1, spline, pchip | ~400 |
| V10 | Convolution + misc | conv, circconv, fliplr, flipud | ~200 |
| V11 | Special functions (rare) | besselj, bessely, airy, ellipj, erf, erfc | ~300 |
| V12 | Logical/type ops | isnan, isinf, isreal, logical, any, all | ~200 |
| V13 | `pde15s` — PDE solver via method of lines | @chebfun/pde15s.m (~500 lines) | ~500 |

## Priority 4: Missing Root Utilities

| Unit | What | MATLAB source | Est. LOC |
|------|------|--------------|----------|
| V14 | `sing` — singularity detection | sing.m (320 lines) | ~300 |
| V15 | `pswf` — prolate spheroidal wave functions | pswf.m, pswfpts.m | ~300 |
| V16 | `lebesgue` — Lebesgue constant/function | lebesgue.m (215 lines) | ~200 |
| V17 | `conformal` — conformal mapping | conformal.m (293 lines) | ~300 |
| V18 | Remaining quadrature: `gauss`, custom weights | ~5 minor .m files | ~200 |

## Priority 5: Infrastructure Gaps

| Unit | What | Est. LOC |
|------|------|----------|
| V19 | Plotting: `Chebfun.plot()`, `Chebfun2.plot()` via matplotlib | ~300 |
| V20 | `+cheb` gallery functions: gallery, gallery2, gallery3 | ~300 |
| V21 | Spin preferences: spinpref, spinpref2, spinpref3 | ~100 |
| V22 | Full benchmark suite with MATLAB comparison | ~400 |
| V23 | Comprehensive MATLAB golden ref coverage (run chebtest) | ~500 |

## Execution Plan

**Wave 1** (5 agents parallel): V01-V05 (ODE/PDE) — the biggest user-facing gap
**Wave 2** (2 agents parallel): V06-V07 (operator AD) — enables proper Newton
**Wave 3** (5 agents parallel): V08-V12 (missing Chebfun methods)
**Wave 4** (5 agents parallel): V13-V18 (PDE + utilities)
**Wave 5** (3 agents parallel): V19-V23 (polish)

## Priority 6: Full Test Coverage

| Unit | What | Est. LOC |
|------|------|----------|
| V24 | Backfill unit tests for uncovered branches (target 95%+ coverage) | ~1,000 |
| V25 | Translate MATLAB's 1,102 test files → Python equivalents for every module | ~3,000 |
| V26 | MATLAB golden ref generation for ALL translated functions | ~500 |
| V27 | Cross-platform CI tests (CPU + GPU parity verification) | ~300 |
| V28 | Property-based tests: mathematical invariants across all modules | ~500 |

Key invariants to test:
- `diff(cumsum(f)) == f` for every function class
- `sum(f*g) == inner(f,g)`
- `roots(f - f(x0))` contains `x0`
- `cheb2leg(leg2cheb(c)) == c` round-trips
- `curl(grad(f)) == 0`, `div(curl(F)) == 0` for vector fields
- CPU vs GPU agreement at `rtol=1e-12` for all JIT-safe operations

**Estimated total: ~12,000 LOC new code + tests, ~25 agents, ~4-5 hours wall time**

After this, the library will be at ~90% feature parity with full test coverage —
the remaining 10% is MATLAB-specific functionality that has no Python equivalent.
