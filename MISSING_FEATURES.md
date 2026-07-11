# chebfunjax — Missing Features & Test Status (definitive list)

*Generated from the completed MATLAB test-suite port (Fable 5 audit,
2026-07-11). Every number below is derived from the skip/xfail markers
in `tests/test_matlab_port/` — this is measured, not estimated.*

## 0. Test-status accounting

All **1,095** non-chebgui MATLAB test files (43 modules) have port
counterparts (1,105 port files). Final run:

| Status | Count | Meaning |
|---|---:|---|
| **passed** | 1,145 | ported at MATLAB tolerances and passing |
| **skipped** | 1,273 | test written; the feature does not exist (reason string names it) |
| **xfailed** | 688 | known library bug or convention gap, with evidence in the reason |
| **failed** | **0** | |

Fully-passing files: 139 · mixed (some assertions pass, some skip): 249
· fully-skipped files: 717.

## 1. KNOWN BUGS awaiting fixes (xfails with evidence)

| # | Bug | Evidence | Where recorded |
|---|---|---|---|
| 1 | ~~diskfun `_diskfun_reconstruct`~~ **FIXED (Fable 5)**: the radial lstsq fit carried ~1e-11 noise that sent the constructor's GE down a noise-pivot path, degenerating the representation. Radial nodes moved to Chebyshev-Gauss points (well-conditioned exact solve). All previously-broken cases now exact; diskfun/diskfunv xfails flipped to passing tests | was: `lap((1−r²)r²cos2θ)` → −12r² (no cos2θ); `d/dy(y²)` → 0 | fixed in `_diskfun_reconstruct` |
| 2 | **fracDiff** off by a constant factor (~0.9502 at q=√2/2); **fracInt** only ~4-digit accurate | vs Γ(n+1)/Γ(n+1∓q)·x^(n∓q); MATLAB passes at 100·eps | `chebfun/test_fracCalc` |
| 3 | **circconv** NaN on non-unit trig domains; on [−1,1] output shifted by half the period (~0.6% amplitude error too) | cos(10x)⊛cos(10x) | `chebfun/test_circconv` |
| 4 | **polyfitL1** not L1-optimal | its deg-5 fit of exp(x)sin(10x) has L1 error 1.93 > 1.27 (plain L2 projection); optimality integral ∫T_k·sign(f−p) ≠ 0 | `chebfun/test_polyfitL1` |
| 5 | **norm(inf)** crashes on complex chebfuns (`lt` on complex128) | needs a |f| path via minandmax | `chebfun/test_norm` |
| 6 | **Chebtech1 vals2coeffs/coeffs2vals drop imaginary parts** (jnp.real cast) | 17 xfail markers | `chebtech1/` ports |
| 7 | Arithmetic does not propagate `ishappy=False` | 6 markers | tech ports |
| 8 | θ-independent Spherefun constructions (e.g. sin λ) only ~8e-9 accurate, warn "column slices not resolved" | | `spherefunv/test_constructor` |
| 9 | legpts/jacpts/lobpts/radaupts/hermpts/lagpts/ultrapts return no barycentric-weight third output | minor API gap | misc ports |

## 2. MISSING FEATURES (the 1,273 skips, categorized)

Ranked by files blocked (fixing a category flips its skips to real tests):

| Files | Feature gap |
|---:|---|
| **~120** | **Array-valued (multi-column) funs/techs/chebfuns** (209 markers). The single biggest architectural gap. Blocks: per-column ops, qr/svd of array-valued objects, cov/var, horzcat/vertcat/diag/repmat/mat2cell/extractColumns/assignColumns, cummax/cummin, mean of columns, kron, mldivide/mrdivide least squares |
| **116** | **Missing named utilities** — cf, chebpade, trigpade, trigremez, trigratinterp, merge, join, arclength, inv (compositional), residue, dct/dst/dlt/idlt, fred/volt (integral operators), gmres/minres/pcg/svds/null/orth/pinv (operator LA), deflation, followpath, pantograph, wronskian, rotate/gaussfilt/harmonic/solharm (geometry), hosvd/tucker/permute/squeeze/restrict/truncate on 3D, minandmax2est/3est, cheb.bernoulli/bspline/gallery2/galleryball/normal2/revolution, randnfun2, nufft2, trigBary, hermpoly, lagpoly, isSubset |
| **84** | **Systems of ODEs / chebmatrix / block operators** — chebop is scalar-only; no chebmatrix class; linop block algebra internal; chebop2 accepts only scalar lbc/rbc/ubc/dbc |
| **46** | **Empty-object representations** (`chebfun()`, empty techs/funs/2D/3D objects) |
| **42** | **MATLAB-only interfaces** — string constructors (`chebfun('sin(x)')`), get()/subsref/end, preference objects (chebfunpref/cheboppref/spinpref), global toggles (splitting/blowup/chebvar), treeVar |
| **~40** | **Vector-class method gaps** — chebfun2v/3v, spherefunv, diskfunv, ballfunv have core calculus but lack many per-class methods (norm variants, feval conventions, arithmetic on ballfunv, PT decomposition, vorticity, normal fields …) |
| **26** | **chebfun-level blowup/singular exponents** (`'exps'` flag, tan with poles, sqrt of root-touching functions) — Singfun exists at the fun layer but is not wired into the chebfun factory |
| 26 | **adchebfun** — *deliberate skip*: JAX AD is the designated counterpart (user decision) |
| 18 | **Composition ops on 2D/3D/geometry classes** — cos(f), exp(f), sqrt(f)… of chebfun2/chebfun3/spherefun/diskfun/ballfun (1-D Chebfun has them) |
| 18 | **trigtech method gaps** (97 markers) — innerProduct, compose, restrict, plus MATLAB-specific happiness/refinement knobs |
| 13 | Plot smoke tests (MATLAB handle semantics; not numerics) |
| 9 | **Logical chebfuns** — ==, <, ≤, ~, &, | returning indicator chebfuns |
| ~10 | **Deltafun buildout** — isempty/deltaTol cleaning/±Inf feval/cumsum-cell/innerProduct/isequal/minandmax/restrict/conv/Deltafun×Deltafun |
| 4 | **Spherefun/Diskfun arithmetic** (+,−,×,scalar) — Chebfun2/3/Ballfun have it (added this audit); Spherefun/Diskfun don't |
| ~8 | Misc singles: row-chebfun transpose, spinsphere, iszero/isequal on geometry classes, trigcolloc/diffmat |

## 3. What is NOT missing

The passing 1,145 cover: all tech/fun layers (chebtech1/2, trigtech,
bndfun, unbndfun, singfun, deltafun core), chebfun-core calculus/
rootfinding/extrema/norms/special functions, chebop scalar
BVP/IVP/eigs/expm/periodic, chebfun2/chebfun3 with arithmetic,
quadrature/transforms/interpolation (with 3 real bug fixes),
sphere/ball geometry + calculus + Poisson (adversarially verified),
ballfunv/chebfun2v vector calculus, and the misc utility layer.
