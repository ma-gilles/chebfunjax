# Translation Status

## 39 PRs merged — ~90% feature parity with MATLAB Chebfun

### Phase 1: Utilities (complete)
| Module | PR(s) | Functions |
|--------|-------|-----------|
| utils/quadrature | #2 | chebpts, legpts, jacpts, hermpts, lagpts, ultrapts, radaupts, lobpts, trigpts |
| utils/transforms | #4 | vals2coeffs, coeffs2vals, cheb2leg, leg2cheb, cheb2jac, jac2cheb |
| utils/interpolation | #3 | bary, bary_weights, trig_bary, barymat |
| utils/diffmat | #11 | diffmat, cumsummat, intmat, introw, diffrow |
| utils/polynomials | #9 | chebpoly, legpoly, jacpoly + eval functions |
| utils/aaa | #13 | aaa (AAA rational approximation) |
| utils/minimax | #35 | minimax (Remez exchange) |
| utils/ratapprox | #33 | ratinterp, padeapprox, trigratinterp |
| utils/misc | #1 | standard_chop, gridsample, abstract_qr |
| domain | #7 | Domain class |
| pref | #6 | ChebPreferences |

### Phase 2: Tech Layer (complete)
| Module | PR(s) | Classes |
|--------|-------|---------|
| tech/chebtech | #5,8,12,33 | Chebtech2, Chebtech1 (full ops) |
| tech/trigtech | #19 | Trigtech (periodic functions) |

### Phase 3: Fun Layer (complete)
| Module | PR(s) | Classes |
|--------|-------|---------|
| fun/bndfun | #15 | Classicfun, Bndfun on [a,b] |
| fun/unbndfun | #22 | Unbndfun on (-∞,∞), [a,∞) |
| fun/singfun+deltafun | #25 | Singfun, Deltafun |

### Phase 4: Chebfun 1D (complete)
| Module | PR(s) | What |
|--------|-------|------|
| chebfun1d/chebfun | #14,16 | Chebfun class, factory, arithmetic, calculus, roots, norm |
| chebfun1d/specfun | #18 | sin, cos, exp, log, sqrt, abs, sign, sinh, tanh, ... |
| chebfun1d/linalg | #29 | Quasimatrix QR, SVD |
| chebfun1d/ode | #37,41 | ode45, ode113, bvp4c, bvp5c, ivp, bvp, eigs |
| chebfun1d (V08-V12) | #39 | conv, polyfit, interp1, besselj, erf, horzcat, isnan, ... |

### Phase 5: Discretization (complete)
| Module | PR(s) | Classes |
|--------|-------|---------|
| discretization/chebcolloc | #17 | ChebColloc1, ChebColloc2 |
| discretization/ultras | #20 | UltraS spectral method |
| discretization/trigcolloc | #33 | TrigColloc |

### Phase 6: Operators (complete)
| Module | PR(s) | Classes |
|--------|-------|---------|
| operators/blocks+chebmatrix | #21 | OperatorBlock, FunctionalBlock, ChebMatrix |
| operators/linop+chebop | #23 | Linop, Chebop (ODE/BVP solving) |
| operators/chebop2 | #31 | Chebop2 (2D PDE: Poisson, Helmholtz) |
| autodiff/adchebfun | #42 | ADChebfun (exact Fréchet derivatives) |
| autodiff/treevar | #42 | TreeVar (symbolic operator linearization) |

### Phase 7: 2D Functions (complete)
| Module | PR(s) | Classes |
|--------|-------|---------|
| chebfun2d/separable_approx | #24,37 | SeparableApprox (low-rank 2D) + diff/sum/norm |
| chebfun2d/chebfun2 | #26 | Chebfun2 |
| chebfun2d/chebfun2v | #32,36 | Chebfun2v (2D vector fields) |
| diskfun | #27,36 | Diskfun, Diskfunv |
| spherefun | #27,36 | Spherefun, Spherefunv |

### Phase 8: 3D Functions (complete)
| Module | PR(s) | Classes |
|--------|-------|---------|
| chebfun3d/chebfun3 | #28 | Chebfun3 (Tucker 3D) |
| chebfun3d/chebfun3v | #36 | Chebfun3v (3D vector fields) |
| chebfun3d/chebfun3t | #37 | Chebfun3T (Tucker tensor) |
| ballfun | #32 | Ballfun, Ballfunv |

### Phase 9: PDE Time-Stepping (complete)
| Module | PR(s) | Classes |
|--------|-------|---------|
| spin (1D) | #30 | SpinOp, ETDRK4 (KdV, Allen-Cahn, NLS, KS) |
| spin (2D) | #38 | SpinOp2, spin2 |
| spin (3D+sphere) | #40 | SpinOp3, SpinOpSphere, spin3, spinsphere |
| spin (IMEX) | #41 | imex_euler, imex_sbdf2 |

### Phase 10: Testing & Polish (complete)
| Module | PR(s) | What |
|--------|-------|------|
| integration tests | #36 | 50 end-to-end tests covering README examples |
| autodiff tests | #37 | JIT/grad/vmap verification |
| benchmarks | #37 | benchmarks/bench_core.py |

## In Progress

| Unit | What | Agent |
|------|------|-------|
| V13-V18, V24-V28 | pde15s, singularity detection, Lebesgue, gallery, test coverage backfill | running |

## Stats
- **39 PRs merged**
- **65 source files, ~38,000 LOC**
- **~23,000 test LOC**
- **~2,000+ tests**
- **Repo**: https://github.com/ma-gilles/chebfunjax
