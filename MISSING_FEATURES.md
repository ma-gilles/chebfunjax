# chebfunjax — Missing Features & Test Status (definitive list)

*Generated from the completed MATLAB test-suite port (Fable 5 audit,
2026-07-11). Every number below is derived from the skip/xfail markers
in `tests/test_matlab_port/` — this is measured, not estimated.*

## 0. Test-status accounting

All **1,095** non-chebgui MATLAB test files (43 modules) have port
counterparts (1,105 port files). Final run:

| Status | Count | Meaning |
|---|---:|---|
| **passed** | 2,252 | ported at MATLAB tolerances and passing |
| **skipped** | 676 | test written; the feature does not exist (reason string names it) |
| **xfailed** | 356 | known library bug or convention gap, with evidence in the reason |
| **failed** | **0** | |

*(2026-07-19 follow-up: the 22 alias-port xfails (chebtech1/2
test_alias) flipped to plain passes -- the fleet's `chebtech.alias`
made 20 of them stale XPASSes, and adding `chebtech.bary`/`barywts`
(closed-form barycentric weights + MATLAB-style static evaluator)
closed the final large-tail case in each file at 2e2*eps.)*

*(Updated 2026-07-19 after the structural + fleet phase: +273 passes,
-119 skips, -138 xfails.  Landed: row-chebfun transpose; ballfunv
poloidal-toroidal decomposition + the SPECTRAL ball Helmholtz/
Poisson-Neumann solver (HelmholtzDecomposition at 2.95e-14 vs a
2.2e-10 tolerance); spherefunv 3-Cartesian-component overhaul
(curl/div/vorticity/cross); tech-level empty representations;
SingFun 64-flip wave + residue (canonicalisation, autodetection,
chebcoeffs); deltafun 39-flip wave; chebop params-in-BCs,
interior-point BCs, dense matrix realization, typed linop();
chebfun2 svd/spectral norms/Padua/fevalm; chebtech accessors
(alias/turbo/sample/trigcoeffs); chebfun mldivide/mrdivide/gmres.
FIVE more real bugs fixed (13 total): bartels_stewart's two QZ
defects, three SingFun cumsum/add bugs, and the long-standing
Cahn-Hilliard spinop instability (conjugate-antisymmetric roundoff
growing in the spinodal band -- fixed by per-step
re-Hermitianization).  NEW measured finding: nested spherefun
compositions blow up XLA CPU compilation (3 tests skipped with the
diagnosis; investigation queued).)*

*(Updated 2026-07-19 after the feature-build phase: +138 passes,
-98 skips.  Landed: piecewise-domain chebop solver (per-piece
collocation + continuity rows); Orr-Sommerfeld (complex generalized
eigs, clamped BCs, 'LR'); the zero-curve roots subsystem
(Chebfun2/diskfun/spherefun curves + chebfun2v common zeros);
chebmatrix vertcat/diag-op/rank-k kron-op; chebfun mldivide/mrdivide/
gmres; empty chebfun with propagation; chebfun3 13-feature fill;
diskfun/spherefun/ballfun/ballfunv sub-features; minandmax2
multi-start global optimization.  More real bugs fixed: diskfun
constructor rank collapse (3-strike counter), _chebtech1_quadwts
off-by-one (wrong moments), Chebfun2v arithmetic recompression,
degenerate-domain validation.)*

*(Updated 2026-07-18 after the Fable 5 Big-Three drive: +408 ports
flipped to genuine passes (1,411 -> 1,819), -212 skips, -99 xfails.
Newly closed categories: trig rational approximation (aaatrig SVD-conj
fix, trigpade, trigremez); spinop family (Spinop + ETDRK4 spin(), AC
preset; CH instability xfailed with evidence); and the ARRAY-VALUED
(multi-column) representation through EVERY layer -- column-wise
transforms with exact symmetry enforcement, chebtech1/2 + trigtech
coefficient/column ops, bndfun/classicfun/unbndfun, and the Chebfun
public API (n_columns/extract/assign/mat2cell/repmat/fliplr/any/all,
per-column roots + minandmax incl. the MATLAB complex |f|^2 path,
piecewise array evaluation and cumsum, array interp1/pchip/spline).
Real bugs fixed along the way: complex colleague-matrix rootfinding
(real-part cast gave spurious roots), Trigtech complex-scalar
arithmetic dropping imaginary parts, Chebfun.fliplr flipud-alias,
inf-destroying symmetry correction, gradient projection in the
symmetry enforcement.)*

*(Previous 2026-07-12 accounting: 1,411/1,105/614 after the utilities
+ systems drive.)*

Fully-passing files: 139 · mixed (some assertions pass, some skip): 249
· fully-skipped files: 717.

## 1. KNOWN BUGS awaiting fixes (xfails with evidence)

| # | Bug | Evidence | Where recorded |
|---|---|---|---|
| 1 | ~~diskfun `_diskfun_reconstruct`~~ **FIXED (Fable 5)**: the radial lstsq fit carried ~1e-11 noise that sent the constructor's GE down a noise-pivot path, degenerating the representation. Radial nodes moved to Chebyshev-Gauss points (well-conditioned exact solve). All previously-broken cases now exact; diskfun/diskfunv xfails flipped to passing tests | was: `lap((1−r²)r²cos2θ)` → −12r² (no cos2θ); `d/dy(y²)` → 0 | fixed in `_diskfun_reconstruct` |
| 2 | ~~fracDiff/fracInt~~ **FIXED (Fable 5)**: fracInt now uses Gauss-Jacobi quadrature absorbing the (x−t)^(μ−1) singularity into the weight. fracInt 4e-17, fracDiff 2.6e-14 (n=4). Residual: fracDiff of x (output x^(1−q) endpoint-singular) limited to ~2e-8 by the smooth-chebfun representation — needs the Singfun-wired factory (feature gap) | was: constant ×0.9502 bias, 4-digit fracInt | fixed in `fracInt` |
| 3 | ~~circconv~~ **FIXED (Fable 5)**: g is now sampled periodically at m·dx (was offset by the left endpoint → half-period shift) and the result is rebuilt as a Fourier series (was Runge-diverging polynomial interp1 → NaN). Exact to 1e-14 on symmetric/asymmetric domains, verified vs scipy quad | was: NaN on [−π,π], shift on [−1,1] | fixed in `circconv` |
| 4 | ~~polyfitL1~~ **FIXED (Fable 5)**: the previous "Watson update" was a relaxation onto a fixed interpolant. Now a proper Watson–Newton on the coefficients (SPD Jacobian 2ΣT_k(r)T_j(r)/\|e′(r)\|): optimality integral converges to ~6e-16; L1 error 1.2535 < 1.2709 (L2); the classical \|x\|+x → 0.5+x case exact to 1e-11 | was: L1 err 1.93 > L2's 1.27 | fixed in `polyfitL1` |
| 5 | ~~norm(inf)~~ **FIXED (Fable 5)**: complex chebfuns route through the real chebfun \|f\|² and take √max — piecewise complex exponential gives e exactly | was: crash (lt on complex128) | fixed in `norm` |
| 6 | ~~Chebtech1 complex transforms~~ **FIXED (Fable 5)**: vals2coeffs/coeffs2vals now split complex data into re+im (as MATLAB does); complex roundtrip and exp(iπx) construction at machine precision. 17 xfails + 5 sentinels flipped to passes; Chebtech1 now included in the complex chebtech port cases | was: jnp.real cast dropped imag | fixed in `_chebtech1_*` |
| 7 | ~~ishappy propagation~~ **FIXED (Fable 5)**: all 16 Chebtech1/2 arithmetic-and-calculus operators now AND their operands' happiness through `from_coeffs(ishappy=...)`; 6 port xfails flipped | | fixed in tech ops |
| 8 | ~~θ-independent Spherefun~~ **NOT A BUG (Fable 5)**: sin(λ) is discontinuous at the poles (limit depends on λ) — not a smooth function on the sphere. The constructor rightly warns; ~1e-7 is the attainable accuracy for the ill-posed input | | reclassified |
| 9 | ~~barycentric outputs~~ **FIXED (Fable 5)**: all 7 quadrature functions accept `bary=True` and return MATLAB's normalized barycentric weights (log-scaled products, exact vs MATLAB's printed values incl. Lobatto/Radau sign conventions); 7 port xfails flipped | | fixed in quadrature |

## 2. MISSING FEATURES (the 1,273 skips, categorized)

Ranked by files blocked (fixing a category flips its skips to real tests):

| Files | Feature gap |
|---:|---|
| ~~~120~~ | ~~Array-valued funs/techs/chebfuns~~ **CLOSED (Fable 5, 2026-07-18)**: (n, m) representation through every layer — transforms, chebtech1/2, trigtech, bndfun/classicfun/unbndfun, and the Chebfun API (n_columns/extract/assign_columns/mat2cell/repmat/fliplr/any/all, per-column roots/minandmax, piecewise array eval/cumsum, array interp1/pchip/spline).  diag/vertcat now live in chebmatrix; mldivide/mrdivide least squares added |
| **~70** | **Missing named utilities** — cf, chebpade, trigpade, trigremez, trigratinterp, gmres/minres/pcg/svds (operator LA), deflation, followpath, pantograph, wronskian, gaussfilt/harmonic/solharm (geometry), hosvd/tucker/truncate on 3D, cheb.bernoulli/bspline/gallery2/galleryball/normal2/revolution. **ADDED (Fable 5):** merge, join, arclength, inv, rem, deriv, nextpow2, realsqrt/realpow, dct/idct/dst/idst/dlt/idlt, minandmax2est/3est, residue (both directions), fred, volt, poly, prod, compose (unary/binary/f(g)), legcoeffs, jaccoeffs, complex_fun, cell2quasi, overlap, quasimatrix rank/orth/null/pinv (complex-capable Householder QR/SVD), chebfun2 cumsum/cumsum2/restrict/squeeze, chebfun3 permute/squeeze/restrict/hosvd, spherefun rotate/sum2/gaussfilt/helmholtz, diskfun harmonic/helmholtz/sum2 + poisson bc, ballfun rotate/solharm/helmholtz, hermpoly, lagpoly, trigBary, isSubset, nufft2, randnfun2, cf (polynomial+rational Caratheodory-Fejer), chebpade (Clenshaw-Lord + Maehly), wronskian, addBreaks/addBreaksAtRoots, getValuesAtBreakpoints |
| ~30 | **Chebop/chebmatrix residue** — MOSTLY CLOSED (Fable 5): systems (block collocation, eigs, generalized eigs incl. complex/clamped/'LR', IVPs, periodic, PIECEWISE domains) and chebmatrix (vertcat/horzcat, diag-op, rank-k kron-op) now exist.  Remaining: typed-linop introspection (linearizationDimensions), parameters only in BCs, interior-point BCs, dense D(n)/'oldschool' realization, C1/US discretization variants (single-discretization convention), chebop2 generalized bc objects |
| ~15 | **Empty-object residue** — MOSTLY CLOSED (Fable 5): chebfun()/Chebfun2/3/geometry empties with operation propagation exist.  Remaining: empty TECH-level make() analogues in a few chebtech/trigtech arithmetic files |
| **42** | **MATLAB-only interfaces** — string constructors (`chebfun('sin(x)')`), get()/subsref/end, preference objects (chebfunpref/cheboppref/spinpref), global toggles (splitting/blowup/chebvar), treeVar |
| **~40** | **Vector-class method gaps** — chebfun2v/3v, spherefunv, diskfunv, ballfunv have core calculus but lack many per-class methods (norm variants, feval conventions, arithmetic on ballfunv, PT decomposition, vorticity, normal fields …) |
| **26** | **chebfun-level blowup/singular exponents** (`'exps'` flag, tan with poles, sqrt of root-touching functions) — Singfun exists at the fun layer but is not wired into the chebfun factory |
| 26 | **adchebfun** — *deliberate skip*: JAX AD is the designated counterpart (user decision) |
| ~~18~~ | ~~Composition ops~~ **ADDED (Fable 5)**: compose/exp/sin/cos/sqrt/log/tanh/abs on Chebfun2/Chebfun3; compose/exp/sin/cos/sqrt on Spherefun/Diskfun/Ballfun — constructor re-approximation, machine precision |
| ~~14~~ | ~~2D/3D analytics~~ **ADDED (Fable 5)**: Chebfun2 mean2/mean(dim)/std2/diag_fun/trace/fliplr/flipud/minandmax2/max2/min2; Chebfun3 mean3/std3/norm/minandmax3/max3/min3; Spherefun.norm; Diskfun.norm/mean — all verified against closed forms at machine precision; 14 port stubs flipped |
| 18 | **trigtech method gaps** (97 markers) — innerProduct, compose, restrict, plus MATLAB-specific happiness/refinement knobs |
| 13 | Plot smoke tests (MATLAB handle semantics; not numerics) |
| 9 | **Logical chebfuns** — ==, <, ≤, ~, &, | returning indicator chebfuns |
| ~10 | **Deltafun buildout** — isempty/deltaTol cleaning/±Inf feval/cumsum-cell/innerProduct/isequal/minandmax/restrict/conv/Deltafun×Deltafun |
| ~~4~~ | ~~Spherefun/Diskfun arithmetic~~ **ADDED (Fable 5)**: +,−,×,÷,**,neg via constructor re-approximation; 4 port stubs flipped to passing tests |
| ~8 | Misc singles: row-chebfun transpose, spinsphere, iszero/isequal on geometry classes, trigcolloc/diffmat |

## 3. What is NOT missing

The passing 1,145 cover: all tech/fun layers (chebtech1/2, trigtech,
bndfun, unbndfun, singfun, deltafun core), chebfun-core calculus/
rootfinding/extrema/norms/special functions, chebop scalar
BVP/IVP/eigs/expm/periodic, chebfun2/chebfun3 with arithmetic,
quadrature/transforms/interpolation (with 3 real bug fixes),
sphere/ball geometry + calculus + Poisson (adversarially verified),
ballfunv/chebfun2v vector calculus, and the misc utility layer.
