# Translation Status

## Completed (32 units merged)

| Unit | Module | PR | Key functions/classes |
|------|--------|-----|---------------------|
| U10 | utils/quadrature | #2 | chebpts, legpts, jacpts, hermpts, lagpts, ultrapts, radaupts, lobpts, trigpts |
| U11 | utils/transforms | #4 | vals2coeffs, coeffs2vals, cheb2leg, leg2cheb, cheb2jac, jac2cheb |
| U12 | utils/interpolation | #3 | bary, bary_weights, trig_bary, barymat, cheb_bary_weights |
| U13 | utils/diffmat | #11 | diffmat, cumsummat, intmat, introw, diffrow |
| U14 | utils/polynomials | #9 | chebpoly, legpoly, jacpoly, ultrapoly + eval functions |
| U15a | utils/aaa | #13 | aaa (AAA rational approximation) |
| U15b | utils/minimax | #35 | minimax (Remez exchange algorithm) |
| U15c | utils/ratapprox | #33 | ratinterp, padeapprox, trigratinterp |
| U16 | utils/misc | #1 | standard_chop, gridsample, abstract_qr |
| U17 | domain | #7 | Domain class |
| U18 | pref | #6 | ChebPreferences |
| U20a-d | tech/chebtech | #5,8,12,33 | Chebtech2, Chebtech1, full ops |
| U21a | tech/trigtech | #19 | Trigtech: periodic functions |
| U30 | fun/classicfun + bndfun | #15 | Classicfun, Bndfun on [a,b] |
| U31 | fun/unbndfun | #22 | Unbndfun on (-inf,inf) |
| U32+U33 | fun/singfun + deltafun | #25 | Singfun, Deltafun |
| U40 | chebfun1d/chebfun core | #14 | Chebfun class, chebfun() factory |
| U41+U42 | chebfun1d/ops | #16 | arithmetic, diff, cumsum, sum, roots, norm |
| U44 | chebfun1d/specfun | #18 | sin, cos, exp, log, sqrt, abs, etc. |
| U45 | chebfun1d/linalg | #29 | Quasimatrix QR, SVD |
| U50 | discretization/chebcolloc | #17 | ChebColloc1, ChebColloc2 |
| U51 | discretization/ultras | #20 | UltraS spectral method |
| U52 | discretization/trigcolloc | #33 | TrigColloc |
| U60+U61 | operators/blocks + chebmatrix | #21 | OperatorBlock, FunctionalBlock, ChebMatrix |
| U62+U63 | operators/linop + chebop | #23 | Linop, Chebop (ODE/BVP) |
| U64 | operators/chebop2 | #31 | 2D PDE solver |
| U70a | chebfun2d/separable_approx | #24 | SeparableApprox |
| U71a | chebfun2d/chebfun2 | #26 | Chebfun2 |
| U72a+U73a | diskfun + spherefun | #27 | Diskfun, Spherefun |
| U80a | chebfun3d/chebfun3 | #28 | Chebfun3 (Tucker 3D) |
| U90a-c | spin PDE | #30 | SpinOp, ETDRK4 (KdV, Allen-Cahn, NLS, KS) |

## Pending

| Unit | Module | PR | Status |
|------|--------|-----|--------|
| U81a+U71b | ballfun + chebfun2v | #32 | CI running |

## TODO (~12 units)

| Unit | Module | Priority | Description |
|------|--------|----------|------------|
| U21b | tech/trigtech ops | medium | Remaining Trigtech methods |
| U46 | chebfun1d/ode | medium | ODE integrator wrappers |
| U72b | diskfun/diskfunv | medium | Disk vector fields |
| U73b | spherefun/spherefunv | medium | Sphere vector fields |
| U81b | ballfun/ballfunv | medium | Ball vector fields |
| U43 | chebfun1d/rootfinding | low | Edge cases |
| U70b-c | chebfun2d/separable ops | low | Remaining 2D methods |
| U80b | chebfun3d/chebfun3v | low | 3D vector fields |
| U80c | chebfun3d/chebfun3t | low | Tucker tensor class |
| U100 | autodiff | low | AD for chebfun operations |
| U101 | integration tests | low | End-to-end workflows |
| U102 | benchmarks | low | CPU/GPU performance suite |

## Stats
- **Source**: ~25,000+ LOC
- **Tests**: ~1,800+
- **Repo**: https://github.com/ma-gilles/chebfunjax
