# Translation Status

| Unit | Module | Status | PR | Notes |
|------|--------|--------|-----|-------|
| U00 | repo skeleton | done | — | pyproject, pixi, CI, CLAUDE.md |
| U01 | matlab harness | done | — | generate_refs.m per-module pattern |
| U02 | test infra | done | — | conftest.py, generic matlab_ref fixture |
| U03 | utils/quadrature (initial) | done | — | chebpts, chebweights |
| U10 | utils/quadrature (full) | done | #2 | legpts, jacpts, hermpts, lagpts, ultrapts, radaupts, lobpts, trigpts |
| U11 | utils/transforms | done | #4 | vals2coeffs, coeffs2vals, cheb2leg, leg2cheb, cheb2jac, jac2cheb +7 |
| U12 | utils/interpolation | done | #3 | bary, bary_weights, trig_bary, trig_bary_weights, barymat, cheb_bary_weights |
| U13 | utils/diffmat | done | #11 | diffmat, cumsummat, intmat, introw, diffrow |
| U14 | utils/polynomials | done | #9 | chebpoly, legpoly, jacpoly, ultrapoly, chebeval, legeval, jaceval, ultraeval, hermeval, lageval |
| U15a | utils/aaa | done | #13 | aaa (AAA rational approximation) |
| U15b | utils/minimax | todo | | minimax (Remez exchange) |
| U15c | utils/ratapprox | todo | | ratinterp, padeapprox, trigratinterp |
| U16 | utils/misc | done | #1 | standard_chop, gridsample, abstract_qr |
| U17 | domain | done | #7 | Domain class with mapping, containment, union, restrict |
| U18 | pref | done | #6 | ChebPreferences with context manager |
| U20a | tech/chebtech core | done | #5 | Chebtech2: adaptive construction, Clenshaw eval, prolong, simplify |
| U20b | tech/chebtech construct | done | #8 | compose, restrict, happiness_check |
| U20c | tech/chebtech ops | done | #12 | arithmetic, diff, cumsum, sum, inner, norm, roots |
| U20d | tech/chebtech misc | todo | | Chebtech1 variant, remaining methods |
| U21a | tech/trigtech core | todo | | Trigtech class, trig coefficients, evaluation |
| U21b | tech/trigtech ops | todo | | Trigtech arithmetic, calculus, roots |
| U30 | fun/classicfun + bndfun | done | #15 | Classicfun abstract + Bndfun on [a,b] wrapping Chebtech2 |
| U31 | fun/unbndfun | todo | | Unbounded domains via mapping |
| U32 | fun/singfun | todo | | Algebraic/log singularities |
| U33 | fun/deltafun | todo | | Dirac delta support |
| U40 | chebfun1d/chebfun core | done | #14 | Chebfun class, chebfun() factory, evaluation, repr |
| U41+U42 | chebfun1d/ops | in_progress | | arithmetic, calculus, roots, norm |
| U43 | chebfun1d/rootfinding | todo | | (may be covered by U41+U42) |
| U44 | chebfun1d/specfun | todo | | sin, cos, exp, log, abs via compose |
| U45 | chebfun1d/linalg | todo | | qr, svd, eig on quasimatrices |
| U46 | chebfun1d/ode | todo | | ode45, bvp4c, pde15s |
| U50 | discretization/chebcolloc | todo | | Chebyshev collocation |
| U51 | discretization/ultras | todo | | Ultraspherical spectral |
| U52 | discretization/trig | todo | | Trig collocation + spectral |
| U60 | operators/chebmatrix | todo | | Block matrix of chebfuns |
| U61 | operators/blocks | todo | | linBlock, operatorBlock, functionalBlock |
| U62 | operators/linop | todo | | Linear operators + BCs |
| U63 | operators/chebop | todo | | Nonlinear operators, Newton iteration |
| U64 | operators/chebop2 | todo | | 2D operators |
| U70a-c | chebfun2d/separable_approx | todo | | Low-rank 2D base |
| U71a-b | chebfun2d/chebfun2 | todo | | 2D rectangles + vector fields |
| U72a-b | diskfun | todo | | 2D disk + vector fields |
| U73a-b | spherefun | todo | | 2D sphere + vector fields |
| U80a-c | chebfun3d | todo | | 3D cuboids + vector + Tucker |
| U81a-b | ballfun | todo | | 3D ball + vector fields |
| U90a-c | spin | todo | | PDE time-stepping |
| U100 | autodiff | todo | | AD for chebfuns |
| U101 | integration tests | todo | | End-to-end |
| U102 | benchmarks | todo | | CPU/GPU suite |
