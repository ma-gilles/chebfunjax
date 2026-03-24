# Translation Status

## Completed (32 units)

| Unit | Module | PR | Key functions/classes |
|------|--------|-----|---------------------|
| U10 | utils/quadrature | #2 | chebpts, legpts, jacpts, hermpts, lagpts, ultrapts, radaupts, lobpts, trigpts |
| U11 | utils/transforms | #4 | vals2coeffs, coeffs2vals, cheb2leg, leg2cheb, cheb2jac, jac2cheb |
| U12 | utils/interpolation | #3 | bary, bary_weights, trig_bary, barymat, cheb_bary_weights |
| U13 | utils/diffmat | #11 | diffmat, cumsummat, intmat, introw, diffrow |
| U14 | utils/polynomials | #9 | chebpoly, legpoly, jacpoly, ultrapoly, chebeval, legeval, jaceval, hermeval, lageval |
| U15a | utils/aaa | #13 | aaa (AAA rational approximation) |
| U15b | utils/minimax | #35 | Remez exchange algorithm |
| U15c | utils/ratapprox | #33 | ratinterp, padeapprox, trigratinterp |
| U16 | utils/misc | #1 | standard_chop, gridsample, abstract_qr |
| U17 | domain | #7 | Domain class |
| U18 | pref | #6 | ChebPreferences |
| U20a | tech/chebtech core | #5 | Chebtech2: adaptive construction, Clenshaw eval, prolong, simplify |
| U20b | tech/chebtech construct | #8 | compose, restrict, happiness_check |
| U20c | tech/chebtech ops | #12 | arithmetic, diff, cumsum, sum, inner, norm, roots |
| U20d | tech/chebtech misc | #33 | Chebtech1 variant |
| U21a | tech/trigtech | #19 | Trigtech: periodic functions, spectral diff/integration |
| U30 | fun/classicfun + bndfun | #15 | Classicfun, Bndfun on [a,b] |
| U31 | fun/unbndfun | #22 | Unbndfun on (-inf,inf), [a,inf), (-inf,b] |
| U32+U33 | fun/singfun + deltafun | #25 | Singfun, Deltafun |
| U40 | chebfun1d/chebfun core | #14 | Chebfun class, chebfun() factory, evaluation, repr |
| U41+U42 | chebfun1d/ops | #16 | arithmetic, diff, cumsum, sum, roots, norm, min, max |
| U44 | chebfun1d/specfun | #18 | sin, cos, exp, log, sqrt, abs, sign, etc. |
| U45 | chebfun1d/linalg | #29 | qr, svd on quasimatrices |
| U46 | chebfun1d/ode | — | bvp(), ivp(), eigs() wrappers around Chebop |
| U50 | discretization/chebcolloc | #17 | ChebColloc1, ChebColloc2 |
| U51 | discretization/ultras | #20 | UltraS spectral method |
| U52 | discretization/trig | #33 | TrigColloc discretization |
| U60+U61 | operators/blocks + chebmatrix | #21 | OperatorBlock, FunctionalBlock, ChebMatrix |
| U62+U63 | operators/linop + chebop | #23 | Linop, Chebop (ODE/BVP solving) |
| U64 | operators/chebop2 | #31 | Chebop2: 2D PDE solver |
| U70a | chebfun2d/separable_approx | #24 | SeparableApprox (low-rank 2D) |
| U70b-c | chebfun2d/separable ops | — | diff, sum, sum2, norm on SeparableApprox |
| U71a | chebfun2d/chebfun2 | #26 | Chebfun2, chebfun2() factory |
| U71b | chebfun2d/chebfun2v | #32 | Chebfun2v (2D vector fields) |
| U72a+U73a | diskfun + spherefun | #27 | Diskfun, Spherefun |
| U80a | chebfun3d/chebfun3 | #28 | Chebfun3 (Tucker 3D) |
| U80c | chebfun3d/chebfun3t | — | Chebfun3T: Tucker tensor wrapper |
| U81a | ballfun/ballfun | #32 | Ballfun |
| U90a-c | spin | #30 | Spin: ETDRK4 time-stepping |
| U100 | autodiff | — | JAX AD tests (jit/grad/vmap through all eval paths) |
| U102 | benchmarks | — | benchmarks/bench_core.py |

## TODO (~5 units remaining)

### Medium Priority (extensions)
| Unit | Module | Description | Est. LOC |
|------|--------|------------|----------|
| U21b | tech/trigtech ops | Remaining Trigtech methods | ~500 |
| U72b | diskfun/diskfunv | Disk vector fields | ~400 |
| U73b | spherefun/spherefunv | Sphere vector fields | ~400 |
| U80b | chebfun3d/chebfun3v | 3D vector fields | ~400 |
| U81b | ballfun/ballfunv | Ball vector fields | ~300 |
| U101 | integration tests | End-to-end workflows | ~300 |

## Stats
- **Source**: 47 files, ~25,000 LOC
- **Tests**: 30 files, ~16,000 LOC, ~1,500+ tests
- **Repo**: https://github.com/ma-gilles/chebfunjax
