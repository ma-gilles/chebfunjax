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
