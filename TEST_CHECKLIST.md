# chebfunjax — Full MATLAB Test Checklist

One line per MATLAB Chebfun test file (commit 7574c77). `[x]` ported & enabled (runs green in the suite), `[~]` ported with some cases skipped, `[ ]` skipped or missing (reason shown). GUI-only dirs (chebgui) and adchebfun are excluded by project policy.


**Totals: 1102 MATLAB tests — 905 ported+enabled, 4 partial, 186 skipped, 7 missing (82% enabled).**


## adchebfun  (0/26 enabled)

- [ ] `test_airy` — chebfunjax uses JAX automatic differentiation instead of the adchebfun operator-overloadin
- [ ] `test_bessel` — chebfunjax uses JAX automatic differentiation instead of the adchebfun operator-overloadin
- [ ] `test_cumprodProd` — chebfunjax uses JAX automatic differentiation instead of the adchebfun operator-overloadin
- [ ] `test_cumsumDiffSumMean` — chebfunjax uses JAX automatic differentiation instead of the adchebfun operator-overloadin
- [ ] `test_deflationFun` — chebfunjax uses JAX automatic differentiation instead of the adchebfun operator-overloadin
- [ ] `test_ellipj` — chebfunjax uses JAX automatic differentiation instead of the adchebfun operator-overloadin
- [ ] `test_erf` — chebfunjax uses JAX automatic differentiation instead of the adchebfun operator-overloadin
- [ ] `test_expLog` — chebfunjax uses JAX automatic differentiation instead of the adchebfun operator-overloadin
- [ ] `test_fevalJump` — chebfunjax uses JAX automatic differentiation instead of the adchebfun operator-overloadin
- [ ] `test_fred` — chebfunjax uses JAX automatic differentiation instead of the adchebfun operator-overloadin
- [ ] `test_innerProduct` — chebfunjax uses JAX automatic differentiation instead of the adchebfun operator-overloadin
- [ ] `test_linearityDetection` — chebfunjax uses JAX automatic differentiation instead of the adchebfun operator-overloadin
- [ ] `test_lintest_rdivide` — chebfunjax uses JAX automatic differentiation instead of the adchebfun operator-overloadin
- [ ] `test_lintest_times` — chebfunjax uses JAX automatic differentiation instead of the adchebfun operator-overloadin
- [ ] `test_norm` — chebfunjax uses JAX automatic differentiation instead of the adchebfun operator-overloadin
- [ ] `test_plusMinus` — chebfunjax uses JAX automatic differentiation instead of the adchebfun operator-overloadin
- [ ] `test_pow2Sqrt` — chebfunjax uses JAX automatic differentiation instead of the adchebfun operator-overloadin
- [ ] `test_power` — chebfunjax uses JAX automatic differentiation instead of the adchebfun operator-overloadin
- [ ] `test_rdivide` — chebfunjax uses JAX automatic differentiation instead of the adchebfun operator-overloadin
- [ ] `test_seed` — chebfunjax uses JAX automatic differentiation instead of the adchebfun operator-overloadin
- [ ] `test_times` — chebfunjax uses JAX automatic differentiation instead of the adchebfun operator-overloadin
- [ ] `test_trig1` — chebfunjax uses JAX automatic differentiation instead of the adchebfun operator-overloadin
- [ ] `test_trig2` — chebfunjax uses JAX automatic differentiation instead of the adchebfun operator-overloadin
- [ ] `test_trig3` — chebfunjax uses JAX automatic differentiation instead of the adchebfun operator-overloadin
- [ ] `test_trig4` — chebfunjax uses JAX automatic differentiation instead of the adchebfun operator-overloadin
- [ ] `test_volt` — chebfunjax uses JAX automatic differentiation instead of the adchebfun operator-overloadin

## ballfun  (49/49 enabled)

- [x] `test_abs`
- [x] `test_ballfun`
- [x] `test_coeffs2vals`
- [x] `test_coeffs3`
- [x] `test_conj`
- [x] `test_constructor`
- [x] `test_cos`
- [x] `test_cosh`
- [x] `test_diff`
- [x] `test_diskfun`
- [x] `test_exp`
- [x] `test_feval`
- [x] `test_gradient`
- [x] `test_helmholtz`
- [x] `test_imag`
- [x] `test_isempty`
- [x] `test_isequal`
- [x] `test_iszero`
- [x] `test_laplacian`
- [x] `test_log`
- [x] `test_mean`
- [x] `test_mean2`
- [x] `test_mean3`
- [x] `test_minus`
- [x] `test_mrdivide`
- [x] `test_mtimes`
- [x] `test_norm`
- [x] `test_plus`
- [x] `test_poisson`
- [x] `test_power`
- [x] `test_real`
- [x] `test_rotate`
- [x] `test_sample`
- [x] `test_sin`
- [x] `test_sinh`
- [x] `test_size`
- [x] `test_solharm`
- [x] `test_spherefun`
- [x] `test_sqrt`
- [x] `test_sum`
- [x] `test_sum2`
- [x] `test_sum3`
- [x] `test_tan`
- [x] `test_tanh`
- [x] `test_times`
- [x] `test_uminus`
- [x] `test_uplus`
- [x] `test_vals2coeffs`
- [x] `test_vscale`

## ballfunv  (25/25 enabled)

- [x] `test_HelmholtzDecomposition`
- [x] `test_PTdecomposition`
- [x] `test_conj`
- [x] `test_constructor`
- [x] `test_cross`
- [x] `test_curl`
- [x] `test_divergence`
- [x] `test_dot`
- [x] `test_feval`
- [x] `test_imag`
- [x] `test_isempty`
- [x] `test_isequal`
- [x] `test_iszero`
- [x] `test_laplacian`
- [x] `test_minus`
- [x] `test_mrdivide`
- [x] `test_mtimes`
- [x] `test_norm`
- [x] `test_plus`
- [x] `test_power`
- [x] `test_real`
- [x] `test_size`
- [x] `test_times`
- [x] `test_uminus`
- [x] `test_uplus`

## bndfun  (14/14 enabled)

- [x] `test_changeMap`
- [x] `test_compose`
- [x] `test_constructor`
- [x] `test_createMap`
- [x] `test_cumsum`
- [x] `test_diff`
- [x] `test_feval`
- [x] `test_innerProduct`
- [x] `test_mldivide`
- [x] `test_mrdivide`
- [x] `test_poly`
- [x] `test_qr`
- [x] `test_restrict`
- [x] `test_sum`

## cheb  (3/8 enabled)

- [ ] `test_bernoulli` — chebfunjax has no cheb.bernoulli
- [ ] `test_bspline` — chebfunjax has no cheb.bspline
- [x] `test_gallery`
- [x] `test_gallery2`
- [ ] `test_galleryball` — chebfunjax has no cheb.galleryball
- [x] `test_gallerytrig`
- [ ] `test_normal2` — chebfunjax has no cheb.normal2
- [ ] `test_revolution` — chebfunjax has no cheb.revolution

## chebfun  (138/166 enabled)

- [x] `test_aaa`
- [x] `test_aaatrig`
- [x] `test_abs`
- [x] `test_addBreaks`
- [x] `test_addBreaksAtRoots`
- [x] `test_airy`
- [x] `test_all`
- [x] `test_and`
- [x] `test_any`
- [x] `test_arclength`
- [x] `test_assignColumns`
- [x] `test_atan2`
- [x] `test_besselh`
- [x] `test_besselj`
- [x] `test_besselyk`
- [ ] `test_bvp4c` — chebfunjax has no bvp4c wrapper (chebop.solve covers BVPs)
- [ ] `test_bvp5c` — chebfunjax has no bvp5c wrapper
- [x] `test_cell2quasi`
- [x] `test_cf`
- [ ] `test_changeTech` — chebfunjax has no changeTech
- [x] `test_chebcoeffs`
- [ ] `test_chebfun_lu` — chebfunjax has no chebfun LU factorization
- [x] `test_chebpade`
- [x] `test_chebpoly`
- [x] `test_circconv`
- [x] `test_comet`
- [x] `test_comet3`
- [x] `test_complex`
- [x] `test_compose_binary`
- [x] `test_compose_chebfuns`
- [x] `test_compose_unary`
- [ ] `test_constructor_basic` — string ctor syntaxes ('x', 'sin(x)') do not exist
- [ ] `test_constructor_basic_periodic` — string + 'periodic' ctor syntaxes do not exist
- [x] `test_constructor_equi`
- [ ] `test_constructor_inputs` — numeric-matrix/string ctor inputs do not exist
- [ ] `test_constructor_inputs_periodic` — periodic ctor input variants do not exist
- [x] `test_constructor_singfun`
- [ ] `test_constructor_splitting` — covered by tests/test_chebfun1d SplittingOn tests (chebfunjax splitting=True)
- [x] `test_constructor_turbo`
- [x] `test_constructor_unbndfun`
- [x] `test_conv`
- [x] `test_cov`
- [x] `test_cummax`
- [x] `test_cummin`
- [x] `test_cumsum`
- [x] `test_dct`
- [ ] `test_defineInterval` — MATLAB subsasgn interval redefinition has no counterpart
- [ ] `test_definePoint` — MATLAB subsasgn point assignment has no counterpart
- [ ] `test_deltaOps` — delta-function chebfun ops (dirac arithmetic at the chebfun level) limited to diff/sum; de
- [x] `test_deriv`
- [x] `test_diag`
- [x] `test_diff`
- [x] `test_dlt`
- [ ] `test_doubleLength` — chebfunjax has no doubleLength
- [x] `test_dst`
- [x] `test_ellipj`
- [x] `test_ellipke`
- [ ] `test_end` — MATLAB end-indexing has no counterpart
- [x] `test_eq`
- [x] `test_erfX`
- [x] `test_exp`
- [x] `test_extractColumns`
- [x] `test_feval`
- [ ] `test_find` — chebfunjax has no find (logical chebfun indexing)
- [x] `test_fix`
- [x] `test_fliplr`
- [x] `test_flipud`
- [x] `test_floor`
- [x] `test_fracCalc`
- [x] `test_fred`
- [x] `test_get`
- [x] `test_getValuesAtBreakpoints`
- [x] `test_gmres`
- [x] `test_horzcat`
- [x] `test_hypot`
- [x] `test_idlt`
- [x] `test_imag`
- [x] `test_innerProduct`
- [x] `test_interp1`
- [x] `test_inv`
- [x] `test_isempty`
- [x] `test_isequal`
- [x] `test_isfinite`
- [x] `test_isinf`
- [x] `test_isnan`
- [x] `test_iszero`
- [ ] `test_ivp` — MATLAB ode113/15s/45 chebfun wrappers; chebop IVP routing tested in operators ports
- [x] `test_jaccoeffs`
- [x] `test_join`
- [x] `test_kron`
- [x] `test_kronOp`
- [x] `test_le`
- [x] `test_legcoeffs`
- [x] `test_log`
- [x] `test_logical`
- [x] `test_lt`
- [x] `test_mat2cell`
- [x] `test_max`
- [x] `test_mean`
- [x] `test_merge`
- [x] `test_min`
- [x] `test_minandmax`
- [x] `test_minus`
- [x] `test_mldivide`
- [x] `test_mrdivide`
- [x] `test_mtimes`
- [x] `test_ne`
- [x] `test_nextpow2`
- [ ] `test_nodots` — MATLAB dot-syntax parser test; not applicable
- [x] `test_norm`
- [x] `test_not`
- [x] `test_null`
- [x] `test_or`
- [x] `test_orth`
- [x] `test_overlap`
- [x] `test_pchip`
- [x] `test_permute`
- [x] `test_pinv`
- [x] `test_plot`
- [x] `test_plot_xylim`
- [x] `test_plotcoeffs`
- [x] `test_plus`
- [ ] `test_points` — chebfunjax has no points() accessor
- [x] `test_polyfit`
- [x] `test_polyfitL1`
- [ ] `test_polyval` — MATLAB polyval-style coefficient evaluation not applicable
- [x] `test_power`
- [x] `test_prod`
- [x] `test_qr`
- [ ] `test_range` — chebfunjax has no range
- [x] `test_rdivide`
- [x] `test_real`
- [x] `test_realpow`
- [x] `test_realsqrt`
- [ ] `test_removeDeltas` — chebfunjax has no removeDeltas (deltas field is static metadata)
- [x] `test_repmat`
- [x] `test_residue`
- [x] `test_restrict`
- [x] `test_roots`
- [x] `test_round`
- [x] `test_sign`
- [x] `test_simplify`
- [x] `test_spline`
- [ ] `test_splitting_abs` — covered by SplittingOn unit tests + abs port
- [x] `test_sqrt`
- [x] `test_subspace`
- [x] `test_subsref`
- [x] `test_sum`
- [x] `test_svd`
- [x] `test_tan`
- [x] `test_times`
- [ ] `test_trig` — MATLAB 'trig' flag broad test; trig construction covered by trigtech ports + factory tests
- [x] `test_trigcasting`
- [x] `test_trigcoeffs`
- [x] `test_trigpade`
- [ ] `test_trigratinterp` — chebfunjax has no trigratinterp
- [x] `test_trigremez`
- [ ] `test_truncate` — chebfunjax has no truncate
- [ ] `test_tweakDomain` — chebfunjax has no tweakDomain
- [ ] `test_ultracoeffs` — chebfunjax Chebfun has no ultracoeffs (ultra2ultra tested in misc)
- [x] `test_unwrap`
- [x] `test_var`
- [ ] `test_vectorCheck` — MATLAB 'vectorize' flag does not exist
- [x] `test_vertcat`
- [x] `test_volt`
- [x] `test_waterfall`

## chebfun2  (61/75 enabled)

- [ ] `test_CLA` — Chebfun2 has no CDR/CLA decomposition accessors
- [x] `test_abs`
- [x] `test_battery`
- [x] `test_biharm`
- [x] `test_cdr`
- [x] `test_chebcoeffs2`
- [x] `test_chebpolyval2`
- [x] `test_chebpts2`
- [ ] `test_chol` — Chebfun2 has no chol() factorization
- [x] `test_coefficients`
- [x] `test_complex`
- [x] `test_composition_operators`
- [x] `test_conj`
- [ ] `test_constructor` — most ctor syntaxes tested (strings, coefficient matrices, values arrays, 'coeffs'/'trig' f
- [ ] `test_constructor2` — adaptive-grid ctor internals (minSamples/maxLength prefs) are not exposed
- [x] `test_contour`
- [ ] `test_contour3` — Chebfun2 has no contour3
- [x] `test_ctorsyntax`
- [x] `test_cumsum`
- [x] `test_diag`
- [x] `test_diff`
- [x] `test_divide`
- [x] `test_eig`
- [x] `test_emptyObjects`
- [ ] `test_end` — MATLAB end-indexing has no Python counterpart on Chebfun2
- [ ] `test_equiOption` — Chebfun2 constructor has no 'equi' (equispaced) option
- [x] `test_feval`
- [x] `test_fevalm`
- [x] `test_gradys_function1`
- [x] `test_gradys_function2`
- [x] `test_guide`
- [x] `test_imag`
- [x] `test_integral`
- [x] `test_integral2`
- [ ] `test_integralEqns` — Fredholm integral-equation solves need fred/volt on Chebfun2 (absent)
- [x] `test_interpaccuracy`
- [x] `test_isPeriodicTech`
- [x] `test_isequal`
- [x] `test_lu`
- [x] `test_max`
- [x] `test_mean`
- [x] `test_min`
- [x] `test_minandmax2est`
- [x] `test_minus`
- [ ] `test_mixed_tech` — Chebfun2 has no trig/periodic tech option to mix
- [x] `test_norm`
- [x] `test_ode45`
- [x] `test_optimization`
- [x] `test_padua`
- [x] `test_plotting`
- [x] `test_plus`
- [ ] `test_poisson` — chebfunjax has no chebfun2.poisson fast solver (chebop2 covers Poisson separately)
- [ ] `test_poldec` — Chebfun2 has no polar decomposition (poldec)
- [x] `test_qr`
- [x] `test_rank`
- [x] `test_repeatedArithmetic`
- [x] `test_restriction`
- [x] `test_roots`
- [x] `test_roots_syntax`
- [x] `test_scl`
- [x] `test_squeeze`
- [x] `test_std`
- [ ] `test_subsref` — MATLAB subsref indexing semantics have no Python counterpart
- [x] `test_sum`
- [x] `test_sumdisk`
- [x] `test_surf`
- [x] `test_techs`
- [x] `test_times`
- [x] `test_transpose`
- [ ] `test_trig` — Chebfun2 has no trig option
- [x] `test_uminus`
- [x] `test_uplus`
- [ ] `test_vectoriseFlag` — Chebfun2 constructor has no 'vectorize' flag
- [x] `test_vertcat`
- [x] `test_zerofunction`

## chebfun2v  (39/40 enabled)

- [x] `test_arithmetic`
- [x] `test_coeffs_vals`
- [x] `test_compose`
- [x] `test_conj`
- [x] `test_constructor`
- [x] `test_constructor2`
- [x] `test_cross`
- [x] `test_curl`
- [x] `test_divergence`
- [x] `test_divgrad`
- [x] `test_dot`
- [x] `test_empty`
- [x] `test_imag`
- [x] `test_integral`
- [x] `test_isPeriodicTech`
- [x] `test_isreal`
- [x] `test_jacobian`
- [x] `test_laplacian`
- [x] `test_minandmax2est`
- [x] `test_plotting`
- [x] `test_real`
- [x] `test_roots01`
- [x] `test_roots02`
- [x] `test_roots03`
- [x] `test_roots04`
- [x] `test_roots05`
- [x] `test_roots06`
- [x] `test_roots07`
- [x] `test_roots08`
- [x] `test_roots09`
- [ ] `test_roots10` — Degenerate common-zero set (a whole line x=1 of solutions); the marching-squares + Newton
- [x] `test_roots_slow`
- [x] `test_roots_syntax`
- [x] `test_size`
- [x] `test_subsref`
- [x] `test_syntax`
- [x] `test_threecomponents`
- [x] `test_times`
- [x] `test_twocomponents`
- [x] `test_vertcat`

## chebfun3  (70/82 enabled)

- [x] `test_abs`
- [x] `test_battery`
- [x] `test_biharm`
- [ ] `test_chebcoeffs3` — Chebfun3 has no chebcoeffs3 accessor
- [ ] `test_chebfun3f` — chebfunjax has no chebfun3f (alternative constructor) variant
- [ ] `test_chebpolyval3` — Chebfun3 has no chebpolyval3 accessor
- [x] `test_chebpts3`
- [ ] `test_coefficients` — Chebfun3 has no coefficient accessors
- [x] `test_complex`
- [x] `test_compose`
- [x] `test_conj`
- [ ] `test_constructor` — most ctor syntaxes (strings, arrays, flags) do not exist on Chebfun3.from_function
- [ ] `test_constructor2` — adaptive-grid ctor internals not exposed
- [x] `test_construnctorsyntax`
- [x] `test_cumsum`
- [x] `test_cumsum3`
- [x] `test_diff`
- [x] `test_diffx`
- [x] `test_diffy`
- [x] `test_diffz`
- [x] `test_divide`
- [ ] `test_domainChck` — domain-check helper semantics are MATLAB-internal
- [x] `test_domainvolume`
- [x] `test_emptyObjects`
- [x] `test_equiFlag`
- [x] `test_feval`
- [x] `test_fevalt`
- [x] `test_fold_unfold`
- [x] `test_get`
- [x] `test_gradient`
- [~] `test_guide` — some cases skipped
- [x] `test_hosvd`
- [x] `test_imag`
- [x] `test_integral`
- [x] `test_integral2`
- [x] `test_integral3`
- [ ] `test_isPeriodicTech` — Chebfun3 has no trig tech option
- [x] `test_isequal`
- [x] `test_isreal`
- [x] `test_iszero`
- [x] `test_laplacian`
- [x] `test_max`
- [x] `test_max2`
- [x] `test_max3`
- [x] `test_mean`
- [x] `test_mean2`
- [x] `test_mean3`
- [x] `test_min`
- [x] `test_min2`
- [x] `test_min3`
- [ ] `test_minandmax3est` — Chebfun3 has no minandmax3est
- [x] `test_minus`
- [x] `test_mtimes`
- [x] `test_ndf`
- [x] `test_norm`
- [x] `test_optimization`
- [x] `test_permute`
- [x] `test_plotting`
- [x] `test_plus`
- [x] `test_rank`
- [x] `test_repeatedArithmetic`
- [x] `test_restrict`
- [x] `test_root`
- [x] `test_scl`
- [x] `test_sin`
- [x] `test_squeeze`
- [x] `test_std`
- [x] `test_std2`
- [x] `test_std3`
- [x] `test_subsref`
- [x] `test_sum`
- [x] `test_sum2`
- [x] `test_sum3`
- [ ] `test_techs` — Chebfun3 has no alternative-tech options
- [x] `test_times`
- [ ] `test_trigs` — Chebfun3 has no trig option
- [x] `test_tucker`
- [x] `test_uminus`
- [x] `test_uplus`
- [x] `test_vectoriseFlag`
- [x] `test_vertcat`
- [ ] `test_zerofunction` — chebfunjax cannot represent the zero-rank Chebfun3 the file tests

## chebfun3t  (6/7 enabled)

- [x] `test_battery`
- [x] `test_compose`
- [x] `test_constructor`
- [x] `test_feval`
- [x] `test_get`
- [ ] `test_ndf` — ndf pins the full-tensor degrees of freedom: MATLAB chebfun3t.ndf == prod(size(f.coeffs)),
- [x] `test_sum3`

## chebfun3v  (29/29 enabled)

- [x] `test_arithmetic`
- [x] `test_compose`
- [x] `test_conj`
- [x] `test_constructor`
- [x] `test_constructor2`
- [x] `test_cross`
- [x] `test_curl`
- [x] `test_divergence`
- [x] `test_divgrad`
- [x] `test_dot`
- [x] `test_empty`
- [x] `test_imag`
- [x] `test_integral`
- [x] `test_integral2`
- [x] `test_isPeriodicTech`
- [x] `test_isreal`
- [x] `test_jacobian`
- [x] `test_laplacian`
- [x] `test_minandmax3est`
- [x] `test_quiver3`
- [x] `test_real`
- [x] `test_root`
- [x] `test_size`
- [x] `test_subsref`
- [x] `test_syntax`
- [x] `test_threecomponents`
- [x] `test_times`
- [x] `test_twocomponents`
- [x] `test_vertcat`

## chebgui  (0/7 enabled)

- [ ] `test_multipleOutputs` — no port file
- [ ] `test_parSimp` — no port file
- [ ] `test_stringParser` — no port file
- [ ] `test_toFileBVP` — no port file
- [ ] `test_toFileEIG` — no port file
- [ ] `test_toFileIVP` — no port file
- [ ] `test_toFilePDE` — no port file

## chebmatrix  (15/15 enabled)

- [x] `test_cellfun`
- [x] `test_changeTech`
- [x] `test_constructor`
- [x] `test_deal`
- [x] `test_flip`
- [x] `test_isNotMultOrDiff`
- [x] `test_length`
- [x] `test_matrixOutput`
- [x] `test_norm`
- [x] `test_plot`
- [x] `test_plotcoeffs`
- [x] `test_size`
- [x] `test_subsassgn`
- [x] `test_times`
- [x] `test_waterfall`

## chebop  (62/99 enabled)

- [x] `test_LorenzIVP`
- [ ] `test_adjoint` — chebop has no adjoint()
- [x] `test_autoVectorize`
- [ ] `test_basic_arithmetic` — chebop objects have no + / - arithmetic or direct application A(u)
- [x] `test_bc`
- [x] `test_bcVectorInput`
- [ ] `test_bcsyntax` — MATLAB bc string syntaxes ('dirichlet', 'neumann', @(x,u) ...) partially exist; string for
- [x] `test_carrier_C1`
- [ ] `test_carrier_C2` — covered assertion-for-assertion by tests/test_operators/test_chebop_nonlinear_matlab.py::t
- [x] `test_carrier_US`
- [x] `test_cellOperator`
- [x] `test_chap21`
- [x] `test_cumsum`
- [x] `test_deflate_bratu`
- [x] `test_deflate_herceg`
- [x] `test_deflate_painleve`
- [ ] `test_determineDiscretization` — single discretization; not applicable
- [ ] `test_diff` — chebop has no D*f operator application (linearize/apply not exposed)
- [ ] `test_domain` — constructor accepts domains leniently; MATLAB's error-identifier checks are MATLAB-specifi
- [x] `test_eigs_basic`
- [x] `test_eigs_drum`
- [x] `test_eigs_foxli`
- [x] `test_eigs_orrsom`
- [x] `test_eigs_periodic`
- [x] `test_eigs_piecewise`
- [x] `test_eigs_schrodinger`
- [x] `test_eigs_system`
- [x] `test_eigs_system2`
- [ ] `test_ellipjODE` — nonlinear pendulum BVP needs ellipj-based exact solution machinery and N.init tuning beyon
- [ ] `test_exactInitial` — N.init exact-solution shortcut semantics not implemented
- [x] `test_expm`
- [ ] `test_feval` — chebop N(x, u) direct evaluation not implemented
- [x] `test_feval2`
- [ ] `test_firstOrderIntegralEqn` — integral-equation operators (fred/volt) not implemented
- [ ] `test_followpath` — path following not implemented
- [ ] `test_gmres` — operator gmres not implemented
- [ ] `test_initialConditions` — MATLAB N.lbc string forms and chebmatrix ICs; scalar IC solving covered by ivp/vdpIVP port
- [ ] `test_intops` — integral operators not implemented
- [x] `test_ivp`
- [ ] `test_ivp_chebmatrix_syntax` — MATLAB chebmatrix cell-syntax variant of system IVPs; the functionality is ported in test_
- [x] `test_jump_scaled`
- [x] `test_jumps_manual`
- [ ] `test_linearInit` — linear-solve init path internal; covered by scalarODE ports
- [x] `test_linearScalarODEs`
- [x] `test_linearSystem1`
- [ ] `test_linearSystem2` — MATLAB u{1}/u{2} cell-indexing NOTATION; the same class of linear systems is ported in tes
- [x] `test_linearizationDimensions`
- [ ] `test_linearize` — linearize() not exposed publicly
- [ ] `test_linearize_init_fails` — linearize() diagnostics not exposed
- [ ] `test_manualNewton` — manual Newton stepping interface not exposed
- [x] `test_matrix`
- [ ] `test_maxnorm` — maxnorm option not implemented
- [ ] `test_minres` — operator minres not implemented
- [ ] `test_mtimes` — chebop scalar*op composition not implemented
- [ ] `test_multOutputs_simplify` — multiple-output simplify not exposed
- [ ] `test_multipleOutputs` — multiple-output solve diagnostics not exposed
- [x] `test_nonlinSys1Breaks_C1`
- [x] `test_nonlinSys1Breaks_C2`
- [x] `test_nonlinSys1Breaks_US`
- [x] `test_nonlinSys1_C1`
- [x] `test_nonlinSys1_C2`
- [x] `test_nonlinSys1_US`
- [x] `test_nonlinSys2_C1`
- [ ] `test_nonlinSys2_C2` — MATLAB chebmatrix u{1}/u{2} cell-indexing NOTATION for the same system ported in test_nonl
- [x] `test_nonlinSys2_US`
- [x] `test_nonlinSysDampingBreaks_C1`
- [x] `test_nonlinSysDampingBreaks_C2`
- [x] `test_nonlinSysDampingBreaks_US`
- [x] `test_nonlinSysDamping_C1`
- [x] `test_nonlinSysDamping_C2`
- [x] `test_nonlinSysDamping_US`
- [ ] `test_null` — operator null space not implemented
- [ ] `test_pantograph` — pantograph (delay) equations not supported
- [x] `test_paramODE`
- [x] `test_paramODE_inBCs`
- [x] `test_paramODE_linearization`
- [x] `test_paramODE_nonlin_C1`
- [x] `test_paramODE_nonlin_C2`
- [x] `test_paramODE_nonlin_US`
- [ ] `test_pcg` — operator pcg not implemented
- [x] `test_periodic`
- [ ] `test_periodic_nonlin` — nonlinear periodic solve not implemented (linear periodic is)
- [x] `test_periodic_system`
- [x] `test_promote_functional`
- [x] `test_quiver`
- [x] `test_scalarODE`
- [x] `test_scalarODE_breakpoints`
- [ ] `test_scalarODE_damping` — MATLAB inspects info.normDelta damping diagnostics; solve() does not expose Newton step in
- [x] `test_scalarODE_sign`
- [ ] `test_shortPulses` — requires breakpoint preservation in solve
- [ ] `test_stringConstructor` — string operator constructor ('0.01*diff(u,2)+...') not implemented
- [ ] `test_svds` — operator svds not implemented
- [x] `test_system3`
- [ ] `test_uminusOp` — chebop unary minus not implemented
- [ ] `test_undampedNewton` — damping-off pref not exposed
- [x] `test_vdpIVP`
- [x] `test_vectorizeOp`
- [x] `test_wronskian`
- [x] `test_zerothOrder`

## chebop2  (22/30 enabled)

- [x] `test_BartelsStewart`
- [ ] `test_adaptivity` — Requires multi-condition BC syntax rbc=@(t,u)[u;diff(u)], a 3rd-order-in-x term diffx(u,3)
- [x] `test_adtest`
- [x] `test_advectionDiffusion1`
- [x] `test_advectionDiffusion2`
- [ ] `test_backwardsWaveEquation` — Two-condition BC on one edge ubc=@(x,u)[u-...;diff(u)-...] now works via the coefficient-s
- [x] `test_basicArithmetic`
- [x] `test_battery`
- [x] `test_bc`
- [x] `test_construction`
- [x] `test_domain`
- [x] `test_eulerTricomi`
- [x] `test_generalVariableCoefficients`
- [x] `test_heatequation`
- [x] `test_helmholtz`
- [x] `test_linearKDV`
- [x] `test_linearSchrodinger`
- [x] `test_neumann`
- [x] `test_plus`
- [x] `test_rhs`
- [ ] `test_rhs2` — Relies on constant/function forcing terms embedded in the operator (laplacian(u)-1, laplac
- [x] `test_schrodinger`
- [ ] `test_separableFormat` — Needs the chebop2.separableFormat low-rank-of-PDO API returning {U,S,V} cells, plus variab
- [ ] `test_squarewaveequation` — Two-condition initial BC dbc=@(x,u)[u-...;diff(u)-...] now works via the coefficient-space
- [ ] `test_subsref` — Requires N(m,n) returning the discretization matrix and N*f / N(f) applying the PDO to a c
- [x] `test_transport`
- [x] `test_univariate`
- [ ] `test_waveequation` — Two-condition initial BC dbc=@(x,u)[u-...;diff(u)-...] now works via the coefficient-space
- [x] `test_weakcornersingularities`
- [ ] `test_withoutAD` — Injects a manual low-rank operator via N.U/N.S/N.V (bypassing AD) and uses variable coeffi

## chebpref  (2/2 enabled)

- [x] `test_chebfunpref`
- [x] `test_cheboppref`

## chebtech  (55/55 enabled)

- [x] `test_abs`
- [x] `test_alias`
- [x] `test_angles`
- [x] `test_any`
- [x] `test_assignColumns`
- [x] `test_bary`
- [x] `test_cell2mat`
- [x] `test_chebTcoeffs2chebUcoeffs`
- [x] `test_clenshaw`
- [x] `test_compose`
- [x] `test_conj`
- [x] `test_cumsum`
- [x] `test_diff`
- [x] `test_extractBoundaryRoots`
- [x] `test_feval`
- [x] `test_fliplr`
- [x] `test_flipud`
- [x] `test_happinessCheck`
- [x] `test_imag`
- [x] `test_innerProduct`
- [x] `test_isempty`
- [x] `test_isequal`
- [x] `test_isfinite`
- [x] `test_isinf`
- [x] `test_isnan`
- [x] `test_isreal`
- [x] `test_iszero`
- [x] `test_legcoeffs`
- [x] `test_length`
- [x] `test_mat2cell`
- [x] `test_max`
- [x] `test_min`
- [x] `test_minandmax`
- [x] `test_minus`
- [x] `test_mldivide`
- [x] `test_mrdivide`
- [x] `test_mtimes`
- [x] `test_plus`
- [x] `test_poly`
- [x] `test_prolong`
- [x] `test_qr`
- [x] `test_quadpts`
- [x] `test_rdivide`
- [x] `test_real`
- [x] `test_restrict`
- [x] `test_roots`
- [x] `test_sample`
- [x] `test_scaleInvariance`
- [x] `test_sign`
- [x] `test_simplify`
- [x] `test_size`
- [x] `test_sum`
- [x] `test_times`
- [x] `test_trigcoeffs`
- [x] `test_turbo`

## chebtech1  (6/6 enabled)

- [x] `test_alias`
- [x] `test_chebpts`
- [x] `test_coeffs2vals`
- [x] `test_constructor`
- [x] `test_extrapolate`
- [x] `test_vals2coeffs`

## chebtech2  (6/6 enabled)

- [x] `test_alias`
- [x] `test_chebpts`
- [x] `test_coeffs2vals`
- [x] `test_constructor`
- [x] `test_extrapolate`
- [x] `test_vals2coeffs`

## classicfun  (12/12 enabled)

- [x] `test_isempty`
- [x] `test_isequal`
- [x] `test_mat2cell`
- [x] `test_max`
- [x] `test_min`
- [x] `test_minandmax`
- [x] `test_minus`
- [x] `test_mtimes`
- [x] `test_plus`
- [x] `test_rdivide`
- [x] `test_roots`
- [x] `test_times`

## deltafun  (19/19 enabled)

- [x] `test_anyDelta`
- [x] `test_chebcoeffs`
- [x] `test_constructor`
- [x] `test_conv`
- [x] `test_cumsum`
- [x] `test_diff`
- [~] `test_feval` — some cases skipped
- [x] `test_imag`
- [x] `test_innerProduct`
- [x] `test_isempty`
- [x] `test_isequal`
- [x] `test_iszero`
- [x] `test_minandmax`
- [x] `test_plus`
- [x] `test_real`
- [x] `test_restrict`
- [x] `test_sum`
- [x] `test_times`
- [x] `test_zeroDeltaFun`

## diskfun  (38/47 enabled)

- [ ] `test_BMCsvd` — internal BMC svd
- [x] `test_Poisson`
- [x] `test_abs`
- [ ] `test_biharm` — Diskfun has no biharm
- [x] `test_cdr`
- [x] `test_coeffs2`
- [x] `test_coeffs2diskfun`
- [ ] `test_coeffs2vals_vals2coeffs` — 2D coefficient transforms not exposed
- [x] `test_composition_operators`
- [x] `test_constructor`
- [x] `test_contour3`
- [ ] `test_curl` — scalar diskfun curl (stream-function) lives on Diskfunv; div/curl tested there
- [ ] `test_diag` — no diag
- [x] `test_diff`
- [x] `test_emptyObjects`
- [x] `test_feval`
- [x] `test_fevalm`
- [x] `test_flipshiftrotate`
- [x] `test_get`
- [ ] `test_grad` — Diskfun has no grad (diffx/diffy tested in diff port)
- [x] `test_harmonic`
- [x] `test_helmholtz`
- [ ] `test_inherited` — inherited separableApprox methods not implemented
- [x] `test_integral`
- [x] `test_integral2`
- [x] `test_isempty`
- [x] `test_iszero`
- [x] `test_laplacian`
- [x] `test_mean`
- [ ] `test_median` — no median
- [x] `test_minandmax2est`
- [x] `test_norm`
- [x] `test_optimization`
- [x] `test_partitionCombine`
- [x] `test_plotting`
- [x] `test_plus`
- [x] `test_power`
- [ ] `test_projection` — BMC projection internal
- [x] `test_rank`
- [x] `test_roots`
- [x] `test_sample`
- [x] `test_subsref`
- [x] `test_sum`
- [x] `test_sum2`
- [x] `test_svd`
- [x] `test_times`
- [x] `test_vertcat`

## diskfunv  (18/24 enabled)

- [x] `test_arithmetic`
- [ ] `test_coeffs_vals` — diskfunv: 'coeffs_vals' targets a missing feature (MATLAB accessor/op not implemented in c
- [ ] `test_compose` — diskfunv: 'compose' targets a missing feature (MATLAB accessor/op not implemented in chebf
- [x] `test_conj_imag_real`
- [x] `test_constructor`
- [x] `test_cross`
- [x] `test_curl`
- [ ] `test_diff` — diskfunv: 'diff' targets a missing feature (MATLAB accessor/op not implemented in chebfunj
- [x] `test_div`
- [x] `test_dot`
- [x] `test_empty`
- [ ] `test_feval` — diskfunv: 'feval' targets a missing feature (MATLAB accessor/op not implemented in chebfun
- [ ] `test_get` — diskfunv: 'get' targets a missing feature (MATLAB accessor/op not implemented in chebfunja
- [x] `test_jacobian`
- [x] `test_minandmax2est`
- [x] `test_plotting`
- [x] `test_size`
- [ ] `test_subsref` — diskfunv: 'subsref' targets a missing feature (MATLAB accessor/op not implemented in chebf
- [x] `test_syntax`
- [x] `test_times_divide`
- [x] `test_transpose`
- [x] `test_vectorRelations`
- [x] `test_vertcat`
- [x] `test_vscale`

## domain  (1/3 enabled)

- [x] `test_merge`
- [ ] `test_poly` — MATLAB domain-class poly accessor has no counterpart
- [ ] `test_polyfit` — MATLAB domain-class polyfit; chebfun-level polyfit ported in chebfun

## fun  (1/1 enabled)

- [x] `test_detectEdge`

## functionalBlock  (0/1 enabled)

- [ ] `test_isNotMultOrDiff` — chebfunjax functional blocks are internal; covered via chebop BC handling tests

## linop  (27/27 enabled)

- [x] `test_chebmatrix`
- [x] `test_coeffs`
- [x] `test_concatenation`
- [x] `test_discretization`
- [x] `test_eigs`
- [x] `test_eigsGeneralized`
- [x] `test_eigsGeneralizedSys`
- [x] `test_eigsPiecewise`
- [~] `test_eigsRayleigh` — some cases skipped
- [x] `test_expm`
- [~] `test_feval_lr` — some cases skipped
- [x] `test_fitBCs`
- [x] `test_functionForm`
- [x] `test_functionals`
- [x] `test_integralops`
- [x] `test_iszero`
- [x] `test_linearsystems`
- [x] `test_linjump`
- [x] `test_linop`
- [x] `test_linopAdjoint`
- [x] `test_mult_op`
- [x] `test_oldschool`
- [x] `test_operarith`
- [x] `test_periodicbvp`
- [x] `test_svds`
- [x] `test_systemapply`
- [x] `test_times`

## misc  (33/52 enabled)

- [x] `test_bary`
- [x] `test_besselroots`
- [ ] `test_blowup` — MATLAB global blowup() toggle has no counterpart; blowup is a constructor concern (Singfun
- [x] `test_cheb2jac`
- [x] `test_cheb2leg`
- [ ] `test_chebpoly` — MATLAB test builds chebfun quasimatrices; chebfunjax chebpoly returns coefficient arrays
- [ ] `test_chebpolyval` — chebfunjax has no chebpolyval (quasimatrix of Chebyshev polys as chebfuns); coefficient tr
- [ ] `test_chebpolyvalm` — chebfunjax has no chebpolyvalm (matrix polynomial evaluation)
- [ ] `test_chebvar` — MATLAB 'chebvar x' workspace magic has no Python counterpart; chebfun(lambda x: x) covers
- [x] `test_coeffs2vals`
- [ ] `test_conformal` — MATLAB test checks chebfun-valued conformal maps; chebfunjax conformal returns discrete bo
- [ ] `test_conformal2` — MATLAB test checks chebfun-valued rectangle maps; chebfunjax conformal2 covered by tests/t
- [x] `test_cumsummat`
- [x] `test_diffmat`
- [x] `test_fov`
- [ ] `test_gpr` — MATLAB test checks chebfun-valued GPR outputs; chebfunjax gpr returns dict of arrays, cove
- [x] `test_hermpoly`
- [x] `test_hermpts`
- [x] `test_inufft`
- [x] `test_isSubset`
- [x] `test_jac2cheb`
- [x] `test_jac2jac`
- [x] `test_jacpoly`
- [x] `test_jacpts`
- [x] `test_lagpoly`
- [x] `test_lagpts`
- [x] `test_lebesgue`
- [x] `test_leg2cheb`
- [ ] `test_legpoly` — MATLAB test builds a 201-column quasimatrix at degrees 900-1100; chebfunjax legpoly return
- [x] `test_legpts`
- [x] `test_lobpts`
- [ ] `test_minimax` — MATLAB test operates on chebfun inputs incl. cf() comparison; chebfunjax minimax(callable)
- [x] `test_nufft`
- [x] `test_nufft2`
- [x] `test_padeapprox`
- [ ] `test_pde15s` — MATLAB test uses chebfun/chebmatrix PDE syntax; chebfunjax pde15s covered by tests/test_co
- [ ] `test_pswf` — MATLAB test checks chebfun-valued PSWFs incl. WolframAlpha point values; chebfunjax pswf r
- [ ] `test_pswfpts` — chebfunjax pswfpts exists; MATLAB test needs pswf chebfun machinery (NOT YET PORTED assert
- [ ] `test_quantumstates` — MATLAB test checks chebfun eigenstates; chebfunjax quantumstates covered by unit tests (NO
- [x] `test_radaupts`
- [ ] `test_randnfun` — MATLAB test checks chebfun-valued randnfun statistics; chebfunjax randnfun covered by test
- [x] `test_randnfun2`
- [ ] `test_randnfundisk` — MATLAB test checks diskfun-valued output; chebfunjax returns grid samples (NOT YET PORTED
- [ ] `test_randnfunsphere` — MATLAB test checks spherefun-valued output; chebfunjax returns grid samples (NOT YET PORTE
- [x] `test_ratinterp`
- [x] `test_scribble`
- [ ] `test_smoothie` — MATLAB test checks chebfun-valued smoothie; chebfunjax returns grid samples (NOT YET PORTE
- [ ] `test_splitting` — MATLAB global splitting() toggle has no counterpart; splitting=True kwarg is tested in tes
- [x] `test_trigBary`
- [x] `test_ultra2ultra`
- [x] `test_ultrapoly`
- [x] `test_ultrapts`

## operatorBlock  (0/1 enabled)

- [ ] `test_isNotMultOrDiff` — chebfunjax operator blocks are internal (chebfunjax.operators.blocks); the public chebop s

## singfun  (24/24 enabled)

- [x] `test_chebcoeffs`
- [x] `test_compose`
- [x] `test_conj`
- [x] `test_cumsum`
- [x] `test_diff`
- [x] `test_feval`
- [x] `test_flipud`
- [x] `test_imag`
- [x] `test_innerProduct`
- [x] `test_isempty`
- [x] `test_isequal`
- [x] `test_isfinite`
- [x] `test_isnan`
- [x] `test_make`
- [x] `test_minandmax`
- [x] `test_plus`
- [x] `test_rdivide`
- [x] `test_real`
- [x] `test_restrict`
- [x] `test_roots`
- [x] `test_singfun_constructor`
- [x] `test_sum`
- [x] `test_times`
- [x] `test_zeroSingFun`

## spherefun  (33/41 enabled)

- [ ] `test_BMCsvd` — BMC-structured svd internal; no public accessor
- [x] `test_HelmholtzSolver`
- [x] `test_Poisson`
- [x] `test_abs`
- [x] `test_biharm`
- [ ] `test_cdr` — Spherefun has no cdr accessor
- [ ] `test_coeffs2` — Spherefun has no coeffs2 accessor
- [ ] `test_coeffs2vals_vals2coeffs` — Spherefun 2D coefficient transforms not exposed
- [x] `test_composition_operators`
- [x] `test_constructor`
- [x] `test_contour3`
- [ ] `test_curl` — scalar Spherefun has no curl (vorticity of a scalar stream fn); Spherefunv tested separate
- [x] `test_diff`
- [x] `test_emptyObjects`
- [x] `test_feval`
- [x] `test_fevalm`
- [x] `test_gaussfilt`
- [x] `test_get`
- [x] `test_grad`
- [ ] `test_inherited` — inherited separableApprox methods (flipud/trace/...) not implemented on Spherefun
- [x] `test_isempty`
- [x] `test_iszero`
- [x] `test_laplacian`
- [x] `test_minandmax2est`
- [x] `test_norm`
- [x] `test_optimization`
- [x] `test_partitionCombine`
- [x] `test_plotting`
- [x] `test_plus`
- [x] `test_power`
- [ ] `test_projection` — Spherefun BMC projection internal
- [x] `test_rank`
- [x] `test_roots`
- [x] `test_rotate`
- [x] `test_sample`
- [x] `test_sphharm`
- [x] `test_subsref`
- [x] `test_sum2`
- [ ] `test_svd` — Spherefun has no svd
- [x] `test_times`
- [x] `test_vertcat`

## spherefunv  (23/23 enabled)

- [x] `test_arithmetic`
- [x] `test_coeffs_vals`
- [x] `test_compose`
- [x] `test_conj_imag_real`
- [x] `test_constructor`
- [x] `test_cross`
- [x] `test_curl`
- [x] `test_div`
- [x] `test_dot`
- [x] `test_empty`
- [x] `test_feval`
- [x] `test_helmholtzdecomp`
- [x] `test_minandmax2est`
- [x] `test_plotting`
- [x] `test_size`
- [x] `test_subsref`
- [x] `test_syntax`
- [x] `test_tangentnormal`
- [x] `test_times_divide`
- [x] `test_transpose`
- [x] `test_vectorRelations`
- [x] `test_vertcat`
- [x] `test_vort`

## spinop  (1/2 enabled)

- [x] `test_spin`
- [ ] `test_spinop` — SpinOp preset/timestep plumbing; ETDRK4 numerics are golden-ref tested in tests/test_spin/

## spinop2  (2/2 enabled)

- [x] `test_spin2`
- [x] `test_spinop2`

## spinop3  (2/2 enabled)

- [x] `test_spin3`
- [x] `test_spinop3`

## spinopsphere  (2/2 enabled)

- [x] `test_spinopsphere`
- [x] `test_spinsphere`

## spinpref  (0/1 enabled)

- [ ] `test_spinpref` — chebfunjax spin has no preference object (kwargs instead)

## spinpref2  (0/1 enabled)

- [ ] `test_spinpref2` — chebfunjax spin2 has no preference object

## spinpref3  (0/1 enabled)

- [ ] `test_spinpref3` — chebfunjax spin3 has no preference object

## spinprefsphere  (0/1 enabled)

- [ ] `test_spinprefsphere` — chebfunjax spinsphere (operators.spinopsphere) takes kwargs and has no SpinPrefSphere pref

## spinscheme  (0/2 enabled)

- [ ] `test_phiFun` — chebfunjax has no public expinteg.phiFun; the exponential phi-function weights are compute
- [ ] `test_startMultistep` — test compares ETDRK4 vs the multistep PECEC736 (1D/2D) and LIRK4 vs IMEXBDF4 (sphere) via

## treeVar  (7/7 enabled)

- [x] `test_bivariate`
- [x] `test_diffArguments`
- [x] `test_plotTree`
- [x] `test_printTree`
- [x] `test_sortConditions`
- [x] `test_toFirstOrder`
- [x] `test_univariate`

## trigspec  (0/1 enabled)

- [ ] `test_multmat` — chebfunjax has no trigspec discretization class; periodic solves use Fourier collocation t

## trigtech  (53/53 enabled)

- [x] `test_abs`
- [x] `test_alias`
- [x] `test_any`
- [x] `test_assignColumns`
- [x] `test_cell2mat`
- [x] `test_circconv`
- [x] `test_coeffs2vals`
- [x] `test_compose`
- [x] `test_conj`
- [x] `test_constructor`
- [x] `test_cumsum`
- [x] `test_diff`
- [x] `test_diffmat`
- [x] `test_feval`
- [x] `test_fliplr`
- [x] `test_flipud`
- [x] `test_happinessCheck`
- [x] `test_imag`
- [x] `test_innerProduct`
- [x] `test_isempty`
- [x] `test_isequal`
- [x] `test_isfinite`
- [x] `test_isinf`
- [x] `test_isnan`
- [x] `test_isreal`
- [x] `test_iszero`
- [x] `test_length`
- [x] `test_mat2cell`
- [x] `test_max`
- [x] `test_min`
- [x] `test_minandmax`
- [x] `test_minus`
- [x] `test_mldivide`
- [x] `test_mrdivide`
- [x] `test_mtimes`
- [x] `test_plus`
- [x] `test_poly`
- [x] `test_prolong`
- [x] `test_qr`
- [x] `test_quadpts`
- [x] `test_rdivide`
- [x] `test_real`
- [x] `test_restrict`
- [x] `test_roots`
- [x] `test_sample`
- [x] `test_scaleInvariance`
- [x] `test_sign`
- [x] `test_simplify`
- [x] `test_size`
- [x] `test_sum`
- [x] `test_times`
- [x] `test_trigcoeffs`
- [x] `test_vals2coeffs`

## unbndfun  (11/11 enabled)

- [x] `test_changeMap`
- [x] `test_compose`
- [x] `test_constructor`
- [x] `test_createMap`
- [x] `test_cumsum`
- [x] `test_diff`
- [x] `test_innerProduct`
- [x] `test_mldivide`
- [x] `test_mrdivide`
- [x] `test_restrict`
- [x] `test_sum`


# Examples (chebfun.org reproductions)

All 21 categories ported; every numbered figure regenerated (826 figures; 631 = 76% pass the strict 0.06 visual-difference gate vs chebfun.org renders — see PARITY_MATRIX.md for the figure-level audit).


## examples/applics (9 scripts)

- [x] `__init__.py`
- [x] `blackscholes2d_replica.py`
- [x] `bode2tf_replica.py`
- [x] `europeancall_replica.py`
- [x] `europeanoptions_replica.py`
- [x] `gompertz_replica.py`
- [x] `greeks_replica.py`
- [x] `step2tf_replica.py`
- [x] `vanillaoptions_replica.py`

## examples/approx (103 scripts)

- [x] `AAAApprox.py`
- [x] `AAASpline.py`
- [x] `AAAZeros.py`
- [x] `AbsoluteValue.py`
- [x] `AbsoluteValueScaled.py`
- [x] `AliasingCoefficients.py`
- [x] `AliasingCoefficientsLeg.py`
- [x] `BSplineConv.py`
- [x] `BernsteinPolys.py`
- [x] `BestApprox.py`
- [x] `BestL1.py`
- [x] `BestL2Approximation.py`
- [x] `CF30.py`
- [x] `ChebfunFFT.py`
- [x] `Checkmark.py`
- [x] `CommunicationSystem.py`
- [x] `DivergentSeries.py`
- [x] `EdgeDetection.py`
- [x] `EightShades.py`
- [x] `Entire.py`
- [x] `EntireBound.py`
- [x] `EquispacedData.py`
- [x] `FermiDirac.py`
- [x] `FiltersCF.py`
- [x] `Galleries.py`
- [x] `GammaFun.py`
- [x] `GreedyInterp.py`
- [x] `Halphen.py`
- [x] `HermiteBasis.py`
- [x] `Inpainting1D.py`
- [x] `InteractiveInterp.py`
- [x] `LebesgueConst.py`
- [x] `Local.py`
- [x] `MinimaxSqrt.py`
- [x] `NearestOrthFun.py`
- [x] `Noisy.py`
- [x] `NoisyNonsmooth.py`
- [x] `OddEven.py`
- [x] `OrthPolys.py`
- [x] `OrthPolysLanczos.py`
- [x] `OscError.py`
- [x] `Prolate.py`
- [x] `PthComposite.py`
- [x] `Pushnitski.py`
- [x] `RationalAbsx.py`
- [x] `RationalInterp.py`
- [x] `Rationalxn.py`
- [x] `ResolutionWiggly.py`
- [x] `RestrictedDenominatorApproximations.py`
- [x] `ScalingAndSquaring.py`
- [x] `SmoothCompact.py`
- [x] `Splines.py`
- [x] `WeierstrassFunction.py`
- [x] `WigglyApprox.py`
- [x] `__init__.py`
- [x] `aaa_approx_replica.py`
- [x] `aaa_spline_replica.py`
- [x] `absolute_value_replica.py`
- [x] `absolute_value_scaled_replica.py`
- [x] `aliasing_coefficients_leg_replica.py`
- [x] `aliasing_coefficients_replica.py`
- [x] `bernstein_polys_replica.py`
- [x] `best_approx_replica.py`
- [x] `best_l1_replica.py`
- [x] `best_l2_approximation_replica.py`
- [x] `bspline_conv_replica.py`
- [x] `cf30_replica.py`
- [x] `checkmark_replica.py`
- [x] `divergent_series_replica.py`
- [x] `edge_detection_replica.py`
- [x] `eight_shades_replica.py`
- [x] `entire_bound_replica.py`
- [x] `entire_replica.py`
- [x] `equispaced_data_replica.py`
- [x] `fermi_dirac_replica.py`
- [x] `filters_cf_replica.py`
- [x] `galleries_replica.py`
- [x] `gamma_fun_replica.py`
- [x] `greedy_interp_replica.py`
- [x] `halphen_replica.py`
- [x] `inpainting1d_replica.py`
- [x] `lebesgue_const_replica.py`
- [x] `local_replica.py`
- [x] `minimax_sqrt_replica.py`
- [x] `nearest_orth_fun_replica.py`
- [x] `noisy_nonsmooth_replica.py`
- [x] `noisy_replica.py`
- [x] `odd_even_replica.py`
- [x] `orth_polys_replica.py`
- [x] `osc_error_replica.py`
- [x] `polyfitL1.py`
- [x] `prolate_replica.py`
- [x] `pth_composite_replica.py`
- [x] `pushnitski_replica.py`
- [x] `rational_absx_replica.py`
- [x] `rational_interp_replica.py`
- [x] `rationalxn_replica.py`
- [x] `resolution_wiggly_replica.py`
- [x] `scaling_and_squaring_replica.py`
- [x] `smooth_compact_replica.py`
- [x] `splines_replica.py`
- [x] `weierstrass_function_replica.py`
- [x] `wiggly_approx_replica.py`

## examples/approx2 (19 scripts)

- [x] `__init__.py`
- [x] `alignment_replica.py`
- [x] `belyaev_replica.py`
- [x] `bumpfunction_replica.py`
- [x] `continuous_skeletonization_study.py`
- [x] `gibbs2d_replica.py`
- [x] `hosepipe_replica.py`
- [x] `localization_replica.py`
- [x] `maxtrace_replica.py`
- [x] `nearestpsdkernel_replica.py`
- [x] `other2ddomains_replica.py`
- [x] `paduapoints_replica.py`
- [x] `pegs_replica.py`
- [x] `polyfitL1.py`
- [x] `prettyfunctions_replica.py`
- [x] `random2d_replica.py`
- [x] `randomponds_replica.py`
- [x] `tucker_replica.py`
- [x] `zebra_replica.py`

## examples/approx2_new (5 scripts)

- [x] `__init__.py`
- [x] `gibbs_2d.py`
- [x] `low_rank_alignment.py`
- [x] `pretty_functions_2d.py`
- [x] `random_functions_2d.py`

## examples/approx3 (12 scripts)

- [x] `ChangeVar3D.py`
- [x] `Chebfun3Speedup.py`
- [x] `Complexity.py`
- [x] `FindingRankOne.py`
- [x] `FluxIntegral3D.py`
- [x] `GaussGreenStokes.py`
- [x] `Hello3.py`
- [x] `LineIntegral3D.py`
- [x] `SurfaceIntegral3D.py`
- [x] `Tolerance.py`
- [x] `Wagon.py`
- [x] `__init__.py`

## examples/calc (7 scripts)

- [x] `__init__.py`
- [x] `delta_derivs_replica.py`
- [x] `for_the_birds_replica.py`
- [x] `integrals_replica.py`
- [x] `mean_value_theorem_replica.py`
- [x] `snells_law_replica.py`
- [x] `surface_revolution_replica.py`

## examples/cheb (12 scripts)

- [x] `__init__.py`
- [x] `cheb_explain_replica.py`
- [x] `cheb_polys_higham_replica.py`
- [x] `chebyshev_coefficients.py`
- [x] `chebyshev_coeffs_replica.py`
- [x] `convergence_replica.py`
- [x] `doublelength_flag_replica.py`
- [x] `exact_cheb_coeffs_replica.py`
- [x] `fast_cheb_leg_transform_replica.py`
- [x] `fast_dlt_replica.py`
- [x] `fast_transforms.py`
- [x] `turbo_replica.py`

## examples/complex (21 scripts)

- [x] `__init__.py`
- [x] `analytic_continuation_replica.py`
- [x] `arguments_replica.py`
- [x] `closed_contours_replica.py`
- [x] `complex_arc_length_replica.py`
- [x] `complex_minimax_replica.py`
- [x] `conformal_l_replica.py`
- [x] `conformal_mapping2_replica.py`
- [x] `conformal_mapping_replica.py`
- [x] `conformal_square_replica.py`
- [x] `conformal_vis_replica.py`
- [x] `hyperfuns_replica.py`
- [x] `keyhole_ablowitz_fokas_replica.py`
- [x] `keyhole_contour_replica.py`
- [x] `phase_portraits_replica.py`
- [x] `phaseplot_command_replica.py`
- [x] `portraits_with_poles_replica.py`
- [x] `rational_harmonic_replica.py`
- [x] `rouche_theorem_replica.py`
- [x] `singularities_replica.py`
- [x] `zeta_zeros_replica.py`

## examples/disk (4 scripts)

- [x] `__init__.py`
- [x] `disk_functions.py`
- [x] `eigenfunctions_replica.py`
- [x] `heat_eqn_replica.py`

## examples/fourier (6 scripts)

- [x] `__init__.py`
- [x] `best_trig_approx_replica.py`
- [x] `fejer_jackson_replica.py`
- [x] `fourier_based_chebfuns.py`
- [x] `fourier_coefficients_replica.py`
- [x] `trig_cf_replica.py`

## examples/fourier_new (2 scripts)

- [x] `__init__.py`
- [x] `fourier_coefficients.py`

## examples/fun (9 scripts)

- [x] `__init__.py`
- [x] `birthday_odds_replica.py`
- [x] `birthday_replica.py`
- [x] `encryption_replica.py`
- [x] `fun_examples.py`
- [x] `hello_world_replica.py`
- [x] `valentines_day2_replica.py`
- [x] `valentines_day_replica.py`
- [x] `writing_3d_replica.py`

## examples/geom (15 scripts)

- [x] `__init__.py`
- [x] `area_replica.py`
- [x] `constant_width_replica.py`
- [x] `curves_and_lengths.py`
- [x] `curves_replica.py`
- [x] `ellipse_replica.py`
- [x] `ellipses_replica.py`
- [x] `lissajous_replica.py`
- [x] `parametric_surfaces.py`
- [x] `parametric_surfaces_replica.py`
- [x] `procrustes_replica.py`
- [x] `rose_curves_replica.py`
- [x] `rounding_corners_replica.py`
- [x] `two_circles_replica.py`
- [x] `volume_of_heart_replica.py`

## examples/integro (5 scripts)

- [x] `__init__.py`
- [x] `fox_li_replica.py`
- [x] `frac_calc2_replica.py`
- [x] `frac_calc_replica.py`
- [x] `wiki_integro_diff_replica.py`

## examples/linalg (19 scripts)

- [x] `__init__.py`
- [x] `analytic_svd_replica.py`
- [x] `cond_nos_replica.py`
- [x] `cond_vandermonde_replica.py`
- [x] `constrained_least_squares_replica.py`
- [x] `crossings_analyticity_replica.py`
- [x] `crouzeix_replica.py`
- [x] `eig_landscapes_replica.py`
- [x] `eigs_via_det_replica.py`
- [x] `field_of_values_replica.py`
- [x] `level_repulsion_replica.py`
- [x] `mercury_earth_conjunctions_replica.py`
- [x] `nonnormal_quiz_replica.py`
- [x] `nonsmooth_fov_replica.py`
- [x] `quasi_qr_replica.py`
- [x] `resolvent_norm_replica.py`
- [x] `sor_replica.py`
- [x] `transient_growth_replica.py`
- [x] `vandermonde_arnoldi_replica.py`

## examples/linalg_new (3 scripts)

- [x] `__init__.py`
- [x] `condition_numbers.py`
- [x] `eigenvalue_problems.py`

## examples/ode-eig (18 scripts)

- [x] `__init__.py`
- [x] `_rayleighquotient_data.py`
- [x] `continuouswilkinson_replica.py`
- [x] `contourprojeig_replica.py`
- [x] `doublewell_replica.py`
- [x] `drum_replica.py`
- [x] `eigenstates_replica.py`
- [x] `fouriereigs_replica.py`
- [x] `landscape_replica.py`
- [x] `levelrepulsionode_replica.py`
- [x] `nullspace_replica.py`
- [x] `opticalresponse_replica.py`
- [x] `orrsommerfeld_replica.py`
- [x] `randfuneig_replica.py`
- [x] `rayleighquotient_replica.py`
- [x] `solarqda_replica.py`
- [x] `thermoelasticrod_replica.py`
- [x] `wavedecay_replica.py`

## examples/ode-linear (26 scripts)

- [x] `__init__.py`
- [x] `adjoints_replica.py`
- [x] `adv_diff_jump_replica.py`
- [x] `boundary_layer_replica.py`
- [x] `breakpoints_replica.py`
- [x] `contour_expm_replica.py`
- [x] `dawson_integral_replica.py`
- [x] `dynamical_systems_replica.py`
- [x] `floquet_replica.py`
- [x] `fourier_collocation_replica.py`
- [x] `frozen_coeffs_replica.py`
- [x] `jump_green_replica.py`
- [x] `krylov_replica.py`
- [x] `lee_greengard_replica.py`
- [x] `lin_exp_ivp_replica.py`
- [x] `linear_ivp_replica.py`
- [x] `matched_asymp_replica.py`
- [x] `near_nonuniqueness_replica.py`
- [x] `nonstandard_bcs_replica.py`
- [x] `order_stars_replica.py`
- [x] `parameter_ode_replica.py`
- [x] `periodic_system_replica.py`
- [x] `regions_replica.py`
- [x] `resonant_vandal_replica.py`
- [x] `spectral_disc_replica.py`
- [x] `wiki_ode_replica.py`

## examples/ode-nonlin (27 scripts)

- [x] `__init__.py`
- [x] `allen_cahn_replica.py`
- [x] `blasius_replica.py`
- [x] `bloodhound_replica.py`
- [x] `blowup_fk_replica.py`
- [x] `bvp_system_replica.py`
- [x] `carrier_replica.py`
- [x] `chebop_quiver_replica.py`
- [x] `delay_differential_equations_replica.py`
- [x] `droplets_replica.py`
- [x] `exact_solns_replica.py`
- [x] `fourier_collocation_nonlin_replica.py`
- [x] `guckenheimer_holmes_replica.py`
- [x] `gulf_stream_replica.py`
- [x] `ivp_capabilities_replica.py`
- [x] `lane_emden_replica.py`
- [x] `logistic2_replica.py`
- [x] `logistic_replica.py`
- [x] `lorenz_attractor_replica.py`
- [x] `lyapunov_exponents_replica.py`
- [x] `modelling_diseases_replica.py`
- [x] `orbits_replica.py`
- [x] `picard_replica.py`
- [x] `square_cycle_replica.py`
- [x] `three_body_problem_replica.py`
- [x] `three_planets_replica.py`
- [x] `two_electrons_replica.py`

## examples/ode-random (11 scripts)

- [x] `__init__.py`
- [x] `consensus_replica.py`
- [x] `gbm_replica.py`
- [x] `levelhopping_replica.py`
- [x] `phaselocking_replica.py`
- [x] `pitchfork_replica.py`
- [x] `random2sde_replica.py`
- [x] `randomonasphere_replica.py`
- [x] `randomswitching_replica.py`
- [x] `tunnelling_replica.py`
- [x] `whitenoiseparadox_replica.py`

## examples/opt (12 scripts)

- [x] `__init__.py`
- [x] `catenary_replica.py`
- [x] `constrained_extrema_replica.py`
- [x] `constrained_optimization_replica.py`
- [x] `dixon_szego_replica.py`
- [x] `extreme_extrema_replica.py`
- [x] `global_minimum_replica.py`
- [x] `mercury_earth_replica.py`
- [x] `needle_replica.py`
- [x] `optim_int_replica.py`
- [x] `rosenbrock2_replica.py`
- [x] `rosenbrock_replica.py`

## examples/opt_new (3 scripts)

- [x] `__init__.py`
- [x] `optimization_1d.py`
- [x] `optimization_2d.py`

## examples/pde (11 scripts)

- [x] `__init__.py`
- [x] `bsexponential_replica.py`
- [x] `erosion_replica.py`
- [x] `fourierexpm_replica.py`
- [x] `ginzburglandau_replica.py`
- [x] `grayscott_replica.py`
- [x] `kdv_replica.py`
- [x] `kswave_replica.py`
- [x] `kuramoto_replica.py`
- [x] `swifthohenberg_replica.py`
- [x] `trapezoideigs_replica.py`

## examples/pde_new (4 scripts)

- [x] `__init__.py`
- [x] `allen_cahn.py`
- [x] `heat_equation.py`
- [x] `kdv_equation.py`

## examples/quad (9 scripts)

- [x] `__init__.py`
- [x] `gauss_clen_curt_replica.py`
- [x] `hermite_quad_replica.py`
- [x] `quadrature_convergence_replica.py`
- [x] `spike_integral_replica.py`
- [x] `sumdisk_demo_replica.py`
- [x] `symbolic_numeric_replica.py`
- [x] `tjtkdisk_replica.py`
- [x] `tricky_replica.py`

## examples/roots (19 scripts)

- [x] `__init__.py`
- [x] `aaa_zeros_replica.py`
- [x] `average_degree_reduction_1d_replica.py`
- [x] `average_degree_reduction_2d_replica.py`
- [x] `bessel_roots_replica.py`
- [x] `bivariate_roots_replica.py`
- [x] `complex_roots_replica.py`
- [x] `fundamental_theorem_of_algebra_replica.py`
- [x] `marching_squares_replica.py`
- [x] `newton_raphson_replica.py`
- [x] `random_polynomials_replica.py`
- [x] `random_polys_replica.py`
- [x] `resultant_method_replica.py`
- [x] `roots_near_axis_replica.py`
- [x] `roots_speed_replica.py`
- [x] `secular_roots_replica.py`
- [x] `subramanian_replica.py`
- [x] `tiger_replica.py`
- [x] `white_curves_replica.py`

## examples/sphere (16 scripts)

- [x] `__init__.py`
- [x] `advectiondiffusion_replica.py`
- [x] `atmospherictemperature_replica.py`
- [x] `gravity_replica.py`
- [x] `helmholtzdecomposition_replica.py`
- [x] `helmholtzdecompositionball_replica.py`
- [x] `laplaceball_replica.py`
- [x] `ptdecomposition_replica.py`
- [x] `rayleighquotientexample_replica.py`
- [x] `solidharmonics_replica.py`
- [x] `sphere_operations.py`
- [x] `spherefunpartition_replica.py`
- [x] `spherefunrotate_replica.py`
- [x] `sphereheatconduction_replica.py`
- [x] `spherical_harmonics.py`
- [x] `sphericalharmonics_replica.py`

## examples/stats (17 scripts)

- [x] `__init__.py`
- [x] `bayesian_gradebook_replica.py`
- [x] `bivariate_normal_distribution_replica.py`
- [x] `central_limit_theorem_replica.py`
- [x] `expectations_replica.py`
- [x] `histogram_replica.py`
- [x] `least_squares_replica.py`
- [x] `mercer_karhunen_loeve_replica.py`
- [x] `normal_exercises_replica.py`
- [x] `probability_convolution_replica.py`
- [x] `random_maxima_replica.py`
- [x] `random_polynomials_replica.py`
- [x] `random_surf_replica.py`
- [x] `resampling_random_variables_replica.py`
- [x] `smooth_random_walk_replica.py`
- [x] `smoothies_replica.py`
- [x] `uniform_exercises_replica.py`

## examples/temp (4 scripts)

- [x] `__init__.py`
- [x] `binousshaikhbellagi_replica.py`
- [x] `compactingcolloids_replica.py`
- [x] `taylorstheorem_replica.py`

## examples/veccalc (4 scripts)

- [x] `__init__.py`
- [x] `autonomous_systems_replica.py`
- [x] `checking_vector_calculus_replica.py`
- [x] `vector_calculus.py`


# Guide (docs/guide)

All 20 chapters translated with regenerated figures (323 figures, all 20 chapters verified against chebfun.org).

- [x] `guide01.md`
- [x] `guide02.md`
- [x] `guide03.md`
- [x] `guide04.md`
- [x] `guide05.md`
- [x] `guide06.md`
- [x] `guide07.md`
- [x] `guide08.md`
- [x] `guide09.md`
- [x] `guide10.md`
- [x] `guide11.md`
- [x] `guide12.md`
- [x] `guide13.md`
- [x] `guide14.md`
- [x] `guide15.md`
- [x] `guide16.md`
- [x] `guide17.md`
- [x] `guide18.md`
- [x] `guide19.md`
- [x] `guide20.md`
