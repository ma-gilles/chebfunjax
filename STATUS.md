# Translation Status

## Completed (38+ units merged across 36 PRs)

| Phase | Units | PRs | What's included |
|-------|-------|-----|----------------|
| 1. Utilities | U10-U18, U15a-c | #1-4,6-7,9,11,13,33,35 | quadrature, transforms, interpolation, diffmat, polynomials, aaa, minimax, ratapprox, misc, domain, pref |
| 2. Tech | U20a-d, U21a | #5,8,12,19,33 | Chebtech2, Chebtech1, Trigtech |
| 3. Fun | U30-U33 | #15,22,25 | Bndfun, Unbndfun, Singfun, Deltafun |
| 4. Chebfun 1D | U40-U45 | #14,16,18,29 | Chebfun core, ops, specfun, QR/SVD |
| 5. Discretization | U50-U52 | #17,20,33 | ChebColloc, UltraS, TrigColloc |
| 6. Operators | U60-U64 | #21,23,31 | blocks, ChebMatrix, Linop, Chebop, Chebop2 |
| 7. 2D | U70a, U71a-b, U72a-b, U73a-b | #24,26,27,36 | SeparableApprox, Chebfun2, Chebfun2v, Diskfun, Diskfunv, Spherefun, Spherefunv |
| 8. 3D | U80a-b, U81a-b | #28,32,36 | Chebfun3, Chebfun3v, Ballfun, Ballfunv |
| 9. PDE | U90a-c | #30 | SpinOp, ETDRK4 (KdV, Allen-Cahn, NLS, KS) |
| 10. Tests | U101 | #36 | 50 integration tests covering all README examples |

## TODO (~5 remaining)

| Unit | Module | Priority | Description |
|------|--------|----------|------------|
| U46 | chebfun1d/ode | medium | ODE integrator wrappers (ode45, bvp4c) |
| U80c | chebfun3d/chebfun3t | low | Tucker tensor class |
| U70b-c | chebfun2d/separable ops | low | Remaining SeparableApprox methods |
| U100 | autodiff | low | AD for chebfun operations |
| U102 | benchmarks | low | CPU/GPU performance suite |

## Stats
- **36 PRs merged**
- **Source**: ~27,000+ LOC
- **Tests**: ~1,800+
- **Repo**: https://github.com/ma-gilles/chebfunjax
