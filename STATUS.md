# Translation Status

| Unit | Module | Status | PR | Agent | Notes |
|------|--------|--------|-----|-------|-------|
| U00 | repo skeleton | done | — | initial | pyproject, pixi, CLAUDE.md |
| U01 | matlab harness | done | — | initial | generate_refs.m, run_chebfun_tests.m |
| U02 | test infra | done | — | initial | conftest.py |
| U03 | utils/quadrature (partial) | done | — | initial | chebpts, chebweights — 13 tests pass |
| U10 | utils/quadrature (full) | todo | | | legpts, jacpts, hermpts, lagpts, ultrapts, etc. |
| U11 | utils/transforms | todo | | | cheb2leg, leg2cheb, jac2cheb, etc. |
| U12 | utils/interpolation | todo | | | bary, trig_bary, bary_weights, barymat |
| U13 | utils/diffmat | todo | | | diffmat, intmat, cumsummat |
| U14 | utils/polynomials | todo | | | chebpoly, legpoly, jacpoly, etc. |
| U15 | utils/approximation | todo | | | aaa, minimax, ratinterp, padeapprox |
| U16 | utils/misc | todo | | | standard_chop, gridsample, abstract_qr |
| U17 | domain | todo | | | Domain class |
| U18 | pref | todo | | | Preferences singleton |
| U20 | tech/chebtech | todo | | | Chebtech2, Chebtech1 — adaptive construction |
| U21 | tech/trigtech | todo | | | Trigtech — periodic functions |
| U30 | fun/classicfun + bndfun | todo | | | Maps [a,b] to [-1,1] |
| U31 | fun/unbndfun | todo | | | Unbounded domains |
| U32 | fun/singfun | todo | | | Algebraic/log singularities |
| U33 | fun/deltafun | todo | | | Dirac delta support |
| U40 | chebfun1d/chebfun (core) | todo | | | Construction, evaluation |
| U41 | chebfun1d/arithmetic | todo | | | +, -, *, /, **, compose |
| U42 | chebfun1d/calculus | todo | | | diff, sum, cumsum, norm |
| U43 | chebfun1d/rootfinding | todo | | | roots, max, min |
| U44 | chebfun1d/specfun | todo | | | sin, cos, exp, log, abs, ... |
| U45 | chebfun1d/linalg | todo | | | qr, svd, eig |
| U46 | chebfun1d/ode | todo | | | ode45, bvp4c, pde15s |
| U50 | discretization/chebcolloc | todo | | | Chebyshev collocation |
| U51 | discretization/ultras | todo | | | Ultraspherical spectral |
| U52 | discretization/trig | todo | | | Trig collocation + spectral |
| U60 | operators/chebmatrix | todo | | | Block matrix of chebfuns |
| U61 | operators/blocks | todo | | | linBlock, operatorBlock, functionalBlock |
| U62 | operators/linop | todo | | | Linear operators + BCs |
| U63 | operators/chebop | todo | | | Nonlinear operators |
| U64 | operators/chebop2 | todo | | | 2D operators |
| U70 | chebfun2d/separable_approx | todo | | | Low-rank 2D base |
| U71 | chebfun2d/chebfun2 | todo | | | 2D rectangles |
| U72 | diskfun | todo | | | 2D disk |
| U73 | spherefun | todo | | | 2D sphere |
| U80 | chebfun3d | todo | | | 3D cuboids |
| U81 | ballfun | todo | | | 3D ball |
| U90 | spin | todo | | | PDE time-stepping |
| U100 | autodiff | todo | | | AD for chebfuns |
| U101 | integration tests | todo | | | End-to-end |
| U102 | benchmarks | todo | | | CPU/GPU suite |
