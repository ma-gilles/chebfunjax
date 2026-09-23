# chebfunjax — Full MATLAB Test Checklist

One line per MATLAB Chebfun test file (commit 7574c77). `[x]` ported & enabled (runs green in the suite), `[~]` ported with some cases skipped, `[ ]` skipped or missing (reason shown). GUI-only dirs (chebgui) and adchebfun are excluded by project policy.


**Totals: 1102 MATLAB tests — 1069 ported+enabled, 0 partial, 26 skipped, 7 missing (97% enabled).**


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

## cheb  (8/8 enabled)

- [x] `test_bernoulli`
- [x] `test_bspline`
- [x] `test_gallery`
- [x] `test_gallery2`
- [x] `test_galleryball`
- [x] `test_gallerytrig`
- [x] `test_normal2`
- [x] `test_revolution`

## chebfun  (166/166 enabled)

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
- [x] `test_bvp4c`
- [x] `test_bvp5c`
- [x] `test_cell2quasi`
- [x] `test_cf`
- [x] `test_changeTech`
- [x] `test_chebcoeffs`
- [x] `test_chebfun_lu`
- [x] `test_chebpade`
- [x] `test_chebpoly`
- [x] `test_circconv`
- [x] `test_comet`
- [x] `test_comet3`
- [x] `test_complex`
- [x] `test_compose_binary`
- [x] `test_compose_chebfuns`
- [x] `test_compose_unary`
- [x] `test_constructor_basic`
- [x] `test_constructor_basic_periodic`
- [x] `test_constructor_equi`
- [x] `test_constructor_inputs`
- [x] `test_constructor_inputs_periodic`
- [x] `test_constructor_singfun`
- [x] `test_constructor_splitting`
- [x] `test_constructor_turbo`
- [x] `test_constructor_unbndfun`
- [x] `test_conv`
- [x] `test_cov`
- [x] `test_cummax`
- [x] `test_cummin`
- [x] `test_cumsum`
- [x] `test_dct`
- [x] `test_defineInterval`
- [x] `test_definePoint`
- [x] `test_deltaOps`
- [x] `test_deriv`
- [x] `test_diag`
- [x] `test_diff`
- [x] `test_dlt`
- [x] `test_doubleLength`
- [x] `test_dst`
- [x] `test_ellipj`
- [x] `test_ellipke`
- [x] `test_end`
- [x] `test_eq`
- [x] `test_erfX`
- [x] `test_exp`
- [x] `test_extractColumns`
- [x] `test_feval`
- [x] `test_find`
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
- [x] `test_ivp`
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
- [x] `test_nodots`
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
- [x] `test_points`
- [x] `test_polyfit`
- [x] `test_polyfitL1`
- [x] `test_polyval`
- [x] `test_power`
- [x] `test_prod`
- [x] `test_qr`
- [x] `test_range`
- [x] `test_rdivide`
- [x] `test_real`
- [x] `test_realpow`
- [x] `test_realsqrt`
- [x] `test_removeDeltas`
- [x] `test_repmat`
- [x] `test_residue`
- [x] `test_restrict`
- [x] `test_roots`
- [x] `test_round`
- [x] `test_sign`
- [x] `test_simplify`
- [x] `test_spline`
- [x] `test_splitting_abs`
- [x] `test_sqrt`
- [x] `test_subspace`
- [x] `test_subsref`
- [x] `test_sum`
- [x] `test_svd`
- [x] `test_tan`
- [x] `test_times`
- [x] `test_trig`
- [x] `test_trigcasting`
- [x] `test_trigcoeffs`
- [x] `test_trigpade`
- [x] `test_trigratinterp`
- [x] `test_trigremez`
- [x] `test_truncate`
- [x] `test_tweakDomain`
- [x] `test_ultracoeffs`
- [x] `test_unwrap`
- [x] `test_var`
- [x] `test_vectorCheck`
- [x] `test_vertcat`
- [x] `test_volt`
- [x] `test_waterfall`

## chebfun2  (75/75 enabled)

- [x] `test_CLA`
- [x] `test_abs`
- [x] `test_battery`
- [x] `test_biharm`
- [x] `test_cdr`
- [x] `test_chebcoeffs2`
- [x] `test_chebpolyval2`
- [x] `test_chebpts2`
- [x] `test_chol`
- [x] `test_coefficients`
- [x] `test_complex`
- [x] `test_composition_operators`
- [x] `test_conj`
- [x] `test_constructor`
- [x] `test_constructor2`
- [x] `test_contour`
- [x] `test_contour3`
- [x] `test_ctorsyntax`
- [x] `test_cumsum`
- [x] `test_diag`
- [x] `test_diff`
- [x] `test_divide`
- [x] `test_eig`
- [x] `test_emptyObjects`
- [x] `test_end`
- [x] `test_equiOption`
- [x] `test_feval`
- [x] `test_fevalm`
- [x] `test_gradys_function1`
- [x] `test_gradys_function2`
- [x] `test_guide`
- [x] `test_imag`
- [x] `test_integral`
- [x] `test_integral2`
- [x] `test_integralEqns`
- [x] `test_interpaccuracy`
- [x] `test_isPeriodicTech`
- [x] `test_isequal`
- [x] `test_lu`
- [x] `test_max`
- [x] `test_mean`
- [x] `test_min`
- [x] `test_minandmax2est`
- [x] `test_minus`
- [x] `test_mixed_tech`
- [x] `test_norm`
- [x] `test_ode45`
- [x] `test_optimization`
- [x] `test_padua`
- [x] `test_plotting`
- [x] `test_plus`
- [x] `test_poisson`
- [x] `test_poldec`
- [x] `test_qr`
- [x] `test_rank`
- [x] `test_repeatedArithmetic`
- [x] `test_restriction`
- [x] `test_roots`
- [x] `test_roots_syntax`
- [x] `test_scl`
- [x] `test_squeeze`
- [x] `test_std`
- [x] `test_subsref`
- [x] `test_sum`
- [x] `test_sumdisk`
- [x] `test_surf`
- [x] `test_techs`
- [x] `test_times`
- [x] `test_transpose`
- [x] `test_trig`
- [x] `test_uminus`
- [x] `test_uplus`
- [x] `test_vectoriseFlag`
- [x] `test_vertcat`
- [x] `test_zerofunction`

## chebfun2v  (40/40 enabled)

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
- [x] `test_roots10`
- [x] `test_roots_slow`
- [x] `test_roots_syntax`
- [x] `test_size`
- [x] `test_subsref`
- [x] `test_syntax`
- [x] `test_threecomponents`
- [x] `test_times`
- [x] `test_twocomponents`
- [x] `test_vertcat`

## chebfun3  (82/82 enabled)

- [x] `test_abs`
- [x] `test_battery`
- [x] `test_biharm`
- [x] `test_chebcoeffs3`
- [x] `test_chebfun3f`
- [x] `test_chebpolyval3`
- [x] `test_chebpts3`
- [x] `test_coefficients`
- [x] `test_complex`
- [x] `test_compose`
- [x] `test_conj`
- [x] `test_constructor`
- [x] `test_constructor2`
- [x] `test_construnctorsyntax`
- [x] `test_cumsum`
- [x] `test_cumsum3`
- [x] `test_diff`
- [x] `test_diffx`
- [x] `test_diffy`
- [x] `test_diffz`
- [x] `test_divide`
- [x] `test_domainChck`
- [x] `test_domainvolume`
- [x] `test_emptyObjects`
- [x] `test_equiFlag`
- [x] `test_feval`
- [x] `test_fevalt`
- [x] `test_fold_unfold`
- [x] `test_get`
- [x] `test_gradient`
- [x] `test_guide`
- [x] `test_hosvd`
- [x] `test_imag`
- [x] `test_integral`
- [x] `test_integral2`
- [x] `test_integral3`
- [x] `test_isPeriodicTech`
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
- [x] `test_minandmax3est`
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
- [x] `test_techs`
- [x] `test_times`
- [x] `test_trigs`
- [x] `test_tucker`
- [x] `test_uminus`
- [x] `test_uplus`
- [x] `test_vectoriseFlag`
- [x] `test_vertcat`
- [x] `test_zerofunction`

## chebfun3t  (7/7 enabled)

- [x] `test_battery`
- [x] `test_compose`
- [x] `test_constructor`
- [x] `test_feval`
- [x] `test_get`
- [x] `test_ndf`
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

## chebop  (99/99 enabled)

- [x] `test_LorenzIVP`
- [x] `test_adjoint`
- [x] `test_autoVectorize`
- [x] `test_basic_arithmetic`
- [x] `test_bc`
- [x] `test_bcVectorInput`
- [x] `test_bcsyntax`
- [x] `test_carrier_C1`
- [x] `test_carrier_C2`
- [x] `test_carrier_US`
- [x] `test_cellOperator`
- [x] `test_chap21`
- [x] `test_cumsum`
- [x] `test_deflate_bratu`
- [x] `test_deflate_herceg`
- [x] `test_deflate_painleve`
- [x] `test_determineDiscretization`
- [x] `test_diff`
- [x] `test_domain`
- [x] `test_eigs_basic`
- [x] `test_eigs_drum`
- [x] `test_eigs_foxli`
- [x] `test_eigs_orrsom`
- [x] `test_eigs_periodic`
- [x] `test_eigs_piecewise`
- [x] `test_eigs_schrodinger`
- [x] `test_eigs_system`
- [x] `test_eigs_system2`
- [x] `test_ellipjODE`
- [x] `test_exactInitial`
- [x] `test_expm`
- [x] `test_feval`
- [x] `test_feval2`
- [x] `test_firstOrderIntegralEqn`
- [x] `test_followpath`
- [x] `test_gmres`
- [x] `test_initialConditions`
- [x] `test_intops`
- [x] `test_ivp`
- [x] `test_ivp_chebmatrix_syntax`
- [x] `test_jump_scaled`
- [x] `test_jumps_manual`
- [x] `test_linearInit`
- [x] `test_linearScalarODEs`
- [x] `test_linearSystem1`
- [x] `test_linearSystem2`
- [x] `test_linearizationDimensions`
- [x] `test_linearize`
- [x] `test_linearize_init_fails`
- [x] `test_manualNewton`
- [x] `test_matrix`
- [x] `test_maxnorm`
- [x] `test_minres`
- [x] `test_mtimes`
- [x] `test_multOutputs_simplify`
- [x] `test_multipleOutputs`
- [x] `test_nonlinSys1Breaks_C1`
- [x] `test_nonlinSys1Breaks_C2`
- [x] `test_nonlinSys1Breaks_US`
- [x] `test_nonlinSys1_C1`
- [x] `test_nonlinSys1_C2`
- [x] `test_nonlinSys1_US`
- [x] `test_nonlinSys2_C1`
- [x] `test_nonlinSys2_C2`
- [x] `test_nonlinSys2_US`
- [x] `test_nonlinSysDampingBreaks_C1`
- [x] `test_nonlinSysDampingBreaks_C2`
- [x] `test_nonlinSysDampingBreaks_US`
- [x] `test_nonlinSysDamping_C1`
- [x] `test_nonlinSysDamping_C2`
- [x] `test_nonlinSysDamping_US`
- [x] `test_null`
- [x] `test_pantograph`
- [x] `test_paramODE`
- [x] `test_paramODE_inBCs`
- [x] `test_paramODE_linearization`
- [x] `test_paramODE_nonlin_C1`
- [x] `test_paramODE_nonlin_C2`
- [x] `test_paramODE_nonlin_US`
- [x] `test_pcg`
- [x] `test_periodic`
- [x] `test_periodic_nonlin`
- [x] `test_periodic_system`
- [x] `test_promote_functional`
- [x] `test_quiver`
- [x] `test_scalarODE`
- [x] `test_scalarODE_breakpoints`
- [x] `test_scalarODE_damping`
- [x] `test_scalarODE_sign`
- [x] `test_shortPulses`
- [x] `test_stringConstructor`
- [x] `test_svds`
- [x] `test_system3`
- [x] `test_uminusOp`
- [x] `test_undampedNewton`
- [x] `test_vdpIVP`
- [x] `test_vectorizeOp`
- [x] `test_wronskian`
- [x] `test_zerothOrder`

## chebop2  (30/30 enabled)

- [x] `test_BartelsStewart`
- [x] `test_adaptivity`
- [x] `test_adtest`
- [x] `test_advectionDiffusion1`
- [x] `test_advectionDiffusion2`
- [x] `test_backwardsWaveEquation`
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
- [x] `test_rhs2`
- [x] `test_schrodinger`
- [x] `test_separableFormat`
- [x] `test_squarewaveequation`
- [x] `test_subsref`
- [x] `test_transport`
- [x] `test_univariate`
- [x] `test_waveequation`
- [x] `test_weakcornersingularities`
- [x] `test_withoutAD`

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
- [x] `test_feval`
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

## diskfun  (47/47 enabled)

- [x] `test_BMCsvd`
- [x] `test_Poisson`
- [x] `test_abs`
- [x] `test_biharm`
- [x] `test_cdr`
- [x] `test_coeffs2`
- [x] `test_coeffs2diskfun`
- [x] `test_coeffs2vals_vals2coeffs`
- [x] `test_composition_operators`
- [x] `test_constructor`
- [x] `test_contour3`
- [x] `test_curl`
- [x] `test_diag`
- [x] `test_diff`
- [x] `test_emptyObjects`
- [x] `test_feval`
- [x] `test_fevalm`
- [x] `test_flipshiftrotate`
- [x] `test_get`
- [x] `test_grad`
- [x] `test_harmonic`
- [x] `test_helmholtz`
- [x] `test_inherited`
- [x] `test_integral`
- [x] `test_integral2`
- [x] `test_isempty`
- [x] `test_iszero`
- [x] `test_laplacian`
- [x] `test_mean`
- [x] `test_median`
- [x] `test_minandmax2est`
- [x] `test_norm`
- [x] `test_optimization`
- [x] `test_partitionCombine`
- [x] `test_plotting`
- [x] `test_plus`
- [x] `test_power`
- [x] `test_projection`
- [x] `test_rank`
- [x] `test_roots`
- [x] `test_sample`
- [x] `test_subsref`
- [x] `test_sum`
- [x] `test_sum2`
- [x] `test_svd`
- [x] `test_times`
- [x] `test_vertcat`

## diskfunv  (24/24 enabled)

- [x] `test_arithmetic`
- [x] `test_coeffs_vals`
- [x] `test_compose`
- [x] `test_conj_imag_real`
- [x] `test_constructor`
- [x] `test_cross`
- [x] `test_curl`
- [x] `test_diff`
- [x] `test_div`
- [x] `test_dot`
- [x] `test_empty`
- [x] `test_feval`
- [x] `test_get`
- [x] `test_jacobian`
- [x] `test_minandmax2est`
- [x] `test_plotting`
- [x] `test_size`
- [x] `test_subsref`
- [x] `test_syntax`
- [x] `test_times_divide`
- [x] `test_transpose`
- [x] `test_vectorRelations`
- [x] `test_vertcat`
- [x] `test_vscale`

## domain  (3/3 enabled)

- [x] `test_merge`
- [x] `test_poly`
- [x] `test_polyfit`

## fun  (1/1 enabled)

- [x] `test_detectEdge`

## functionalBlock  (1/1 enabled)

- [x] `test_isNotMultOrDiff`

## linop  (27/27 enabled)

- [x] `test_chebmatrix`
- [x] `test_coeffs`
- [x] `test_concatenation`
- [x] `test_discretization`
- [x] `test_eigs`
- [x] `test_eigsGeneralized`
- [x] `test_eigsGeneralizedSys`
- [x] `test_eigsPiecewise`
- [x] `test_eigsRayleigh`
- [x] `test_expm`
- [x] `test_feval_lr`
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

## misc  (52/52 enabled)

- [x] `test_bary`
- [x] `test_besselroots`
- [x] `test_blowup`
- [x] `test_cheb2jac`
- [x] `test_cheb2leg`
- [x] `test_chebpoly`
- [x] `test_chebpolyval`
- [x] `test_chebpolyvalm`
- [x] `test_chebvar`
- [x] `test_coeffs2vals`
- [x] `test_conformal`
- [x] `test_conformal2`
- [x] `test_cumsummat`
- [x] `test_diffmat`
- [x] `test_fov`
- [x] `test_gpr`
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
- [x] `test_legpoly`
- [x] `test_legpts`
- [x] `test_lobpts`
- [x] `test_minimax`
- [x] `test_nufft`
- [x] `test_nufft2`
- [x] `test_padeapprox`
- [x] `test_pde15s`
- [x] `test_pswf`
- [x] `test_pswfpts`
- [x] `test_quantumstates`
- [x] `test_radaupts`
- [x] `test_randnfun`
- [x] `test_randnfun2`
- [x] `test_randnfundisk`
- [x] `test_randnfunsphere`
- [x] `test_ratinterp`
- [x] `test_scribble`
- [x] `test_smoothie`
- [x] `test_splitting`
- [x] `test_trigBary`
- [x] `test_ultra2ultra`
- [x] `test_ultrapoly`
- [x] `test_ultrapts`

## operatorBlock  (1/1 enabled)

- [x] `test_isNotMultOrDiff`

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

## spherefun  (41/41 enabled)

- [x] `test_BMCsvd`
- [x] `test_HelmholtzSolver`
- [x] `test_Poisson`
- [x] `test_abs`
- [x] `test_biharm`
- [x] `test_cdr`
- [x] `test_coeffs2`
- [x] `test_coeffs2vals_vals2coeffs`
- [x] `test_composition_operators`
- [x] `test_constructor`
- [x] `test_contour3`
- [x] `test_curl`
- [x] `test_diff`
- [x] `test_emptyObjects`
- [x] `test_feval`
- [x] `test_fevalm`
- [x] `test_gaussfilt`
- [x] `test_get`
- [x] `test_grad`
- [x] `test_inherited`
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
- [x] `test_projection`
- [x] `test_rank`
- [x] `test_roots`
- [x] `test_rotate`
- [x] `test_sample`
- [x] `test_sphharm`
- [x] `test_subsref`
- [x] `test_sum2`
- [x] `test_svd`
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

## spinop  (2/2 enabled)

- [x] `test_spin`
- [x] `test_spinop`

## spinop2  (2/2 enabled)

- [x] `test_spin2`
- [x] `test_spinop2`

## spinop3  (2/2 enabled)

- [x] `test_spin3`
- [x] `test_spinop3`

## spinopsphere  (2/2 enabled)

- [x] `test_spinopsphere`
- [x] `test_spinsphere`

## spinpref  (1/1 enabled)

- [x] `test_spinpref`

## spinpref2  (1/1 enabled)

- [x] `test_spinpref2`

## spinpref3  (1/1 enabled)

- [x] `test_spinpref3`

## spinprefsphere  (1/1 enabled)

- [x] `test_spinprefsphere`

## spinscheme  (2/2 enabled)

- [x] `test_phiFun`
- [x] `test_startMultistep`

## treeVar  (7/7 enabled)

- [x] `test_bivariate`
- [x] `test_diffArguments`
- [x] `test_plotTree`
- [x] `test_printTree`
- [x] `test_sortConditions`
- [x] `test_toFirstOrder`
- [x] `test_univariate`

## trigspec  (1/1 enabled)

- [x] `test_multmat`

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

All 21 categories ported; every numbered figure regenerated (see PARITY_MATRIX.md for the figure-level audit).


## examples/applics (9 scripts)

- [x] `__init__.py`
- [x] `blackscholes2d.py`
- [x] `bode2tf.py`
- [x] `europeancall.py`
- [x] `europeanoptions.py`
- [x] `gompertz.py`
- [x] `greeks.py`
- [x] `step2tf.py`
- [x] `vanillaoptions.py`

## examples/approx (55 scripts)

- [x] `ChebfunFFT.py`
- [x] `CommunicationSystem.py`
- [x] `HermiteBasis.py`
- [x] `InteractiveInterp.py`
- [x] `OrthPolysLanczos.py`
- [x] `RestrictedDenominatorApproximations.py`
- [x] `__init__.py`
- [x] `aaa_approx.py`
- [x] `aaa_spline.py`
- [x] `absolute_value.py`
- [x] `absolute_value_scaled.py`
- [x] `aliasing_coefficients.py`
- [x] `aliasing_coefficients_leg.py`
- [x] `bernstein_polys.py`
- [x] `best_approx.py`
- [x] `best_l1.py`
- [x] `best_l2_approximation.py`
- [x] `bspline_conv.py`
- [x] `cf30.py`
- [x] `checkmark.py`
- [x] `divergent_series.py`
- [x] `edge_detection.py`
- [x] `eight_shades.py`
- [x] `entire.py`
- [x] `entire_bound.py`
- [x] `equispaced_data.py`
- [x] `fermi_dirac.py`
- [x] `filters_cf.py`
- [x] `galleries.py`
- [x] `gamma_fun.py`
- [x] `greedy_interp.py`
- [x] `halphen.py`
- [x] `inpainting1d.py`
- [x] `lebesgue_const.py`
- [x] `local.py`
- [x] `minimax_sqrt.py`
- [x] `nearest_orth_fun.py`
- [x] `noisy.py`
- [x] `noisy_nonsmooth.py`
- [x] `odd_even.py`
- [x] `orth_polys.py`
- [x] `osc_error.py`
- [x] `polyfitL1.py`
- [x] `prolate.py`
- [x] `pth_composite.py`
- [x] `pushnitski.py`
- [x] `rational_absx.py`
- [x] `rational_interp.py`
- [x] `rationalxn.py`
- [x] `resolution_wiggly.py`
- [x] `scaling_and_squaring.py`
- [x] `smooth_compact.py`
- [x] `splines.py`
- [x] `weierstrass_function.py`
- [x] `wiggly_approx.py`

## examples/approx2 (19 scripts)

- [x] `__init__.py`
- [x] `alignment.py`
- [x] `belyaev.py`
- [x] `bumpfunction.py`
- [x] `continuous_skeletonization_study.py`
- [x] `gibbs2d.py`
- [x] `hosepipe.py`
- [x] `localization.py`
- [x] `maxtrace.py`
- [x] `nearestpsdkernel.py`
- [x] `other2ddomains.py`
- [x] `paduapoints.py`
- [x] `pegs.py`
- [x] `polyfitL1.py`
- [x] `prettyfunctions.py`
- [x] `random2d.py`
- [x] `randomponds.py`
- [x] `tucker.py`
- [x] `zebra.py`

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
- [x] `delta_derivs.py`
- [x] `for_the_birds.py`
- [x] `integrals.py`
- [x] `mean_value_theorem.py`
- [x] `snells_law.py`
- [x] `surface_revolution.py`

## examples/cheb (12 scripts)

- [x] `__init__.py`
- [x] `cheb_explain.py`
- [x] `cheb_polys_higham.py`
- [x] `chebyshev_coefficients.py`
- [x] `chebyshev_coeffs.py`
- [x] `convergence.py`
- [x] `doublelength_flag.py`
- [x] `exact_cheb_coeffs.py`
- [x] `fast_cheb_leg_transform.py`
- [x] `fast_dlt.py`
- [x] `fast_transforms.py`
- [x] `turbo.py`

## examples/complex (21 scripts)

- [x] `__init__.py`
- [x] `analytic_continuation.py`
- [x] `arguments.py`
- [x] `closed_contours.py`
- [x] `complex_arc_length.py`
- [x] `complex_minimax.py`
- [x] `conformal_l.py`
- [x] `conformal_mapping.py`
- [x] `conformal_mapping2.py`
- [x] `conformal_square.py`
- [x] `conformal_vis.py`
- [x] `hyperfuns.py`
- [x] `keyhole_ablowitz_fokas.py`
- [x] `keyhole_contour.py`
- [x] `phase_portraits.py`
- [x] `phaseplot_command.py`
- [x] `portraits_with_poles.py`
- [x] `rational_harmonic.py`
- [x] `rouche_theorem.py`
- [x] `singularities.py`
- [x] `zeta_zeros.py`

## examples/disk (4 scripts)

- [x] `__init__.py`
- [x] `disk_functions.py`
- [x] `eigenfunctions.py`
- [x] `heat_eqn.py`

## examples/fourier (6 scripts)

- [x] `__init__.py`
- [x] `best_trig_approx.py`
- [x] `fejer_jackson.py`
- [x] `fourier_based_chebfuns.py`
- [x] `fourier_coefficients.py`
- [x] `trig_cf.py`

## examples/fourier_new (2 scripts)

- [x] `__init__.py`
- [x] `fourier_coefficients.py`

## examples/fun (9 scripts)

- [x] `__init__.py`
- [x] `birthday.py`
- [x] `birthday_odds.py`
- [x] `encryption.py`
- [x] `fun_examples.py`
- [x] `hello_world.py`
- [x] `valentines_day.py`
- [x] `valentines_day2.py`
- [x] `writing_3d.py`

## examples/geom (14 scripts)

- [x] `__init__.py`
- [x] `area.py`
- [x] `constant_width.py`
- [x] `curves.py`
- [x] `curves_and_lengths.py`
- [x] `ellipse.py`
- [x] `ellipses.py`
- [x] `lissajous.py`
- [x] `parametric_surfaces.py`
- [x] `procrustes.py`
- [x] `rose_curves.py`
- [x] `rounding_corners.py`
- [x] `two_circles.py`
- [x] `volume_of_heart.py`

## examples/integro (5 scripts)

- [x] `__init__.py`
- [x] `fox_li.py`
- [x] `frac_calc.py`
- [x] `frac_calc2.py`
- [x] `wiki_integro_diff.py`

## examples/linalg (19 scripts)

- [x] `__init__.py`
- [x] `analytic_svd.py`
- [x] `cond_nos.py`
- [x] `cond_vandermonde.py`
- [x] `constrained_least_squares.py`
- [x] `crossings_analyticity.py`
- [x] `crouzeix.py`
- [x] `eig_landscapes.py`
- [x] `eigs_via_det.py`
- [x] `field_of_values.py`
- [x] `level_repulsion.py`
- [x] `mercury_earth_conjunctions.py`
- [x] `nonnormal_quiz.py`
- [x] `nonsmooth_fov.py`
- [x] `quasi_qr.py`
- [x] `resolvent_norm.py`
- [x] `sor.py`
- [x] `transient_growth.py`
- [x] `vandermonde_arnoldi.py`

## examples/linalg_new (3 scripts)

- [x] `__init__.py`
- [x] `condition_numbers.py`
- [x] `eigenvalue_problems.py`

## examples/ode-eig (18 scripts)

- [x] `__init__.py`
- [x] `_rayleighquotient_data.py`
- [x] `continuouswilkinson.py`
- [x] `contourprojeig.py`
- [x] `doublewell.py`
- [x] `drum.py`
- [x] `eigenstates.py`
- [x] `fouriereigs.py`
- [x] `landscape.py`
- [x] `levelrepulsionode.py`
- [x] `nullspace.py`
- [x] `opticalresponse.py`
- [x] `orrsommerfeld.py`
- [x] `randfuneig.py`
- [x] `rayleighquotient.py`
- [x] `solarqda.py`
- [x] `thermoelasticrod.py`
- [x] `wavedecay.py`

## examples/ode-linear (26 scripts)

- [x] `__init__.py`
- [x] `adjoints.py`
- [x] `adv_diff_jump.py`
- [x] `boundary_layer.py`
- [x] `breakpoints.py`
- [x] `contour_expm.py`
- [x] `dawson_integral.py`
- [x] `dynamical_systems.py`
- [x] `floquet.py`
- [x] `fourier_collocation.py`
- [x] `frozen_coeffs.py`
- [x] `jump_green.py`
- [x] `krylov.py`
- [x] `lee_greengard.py`
- [x] `lin_exp_ivp.py`
- [x] `linear_ivp.py`
- [x] `matched_asymp.py`
- [x] `near_nonuniqueness.py`
- [x] `nonstandard_bcs.py`
- [x] `order_stars.py`
- [x] `parameter_ode.py`
- [x] `periodic_system.py`
- [x] `regions.py`
- [x] `resonant_vandal.py`
- [x] `spectral_disc.py`
- [x] `wiki_ode.py`

## examples/ode-nonlin (27 scripts)

- [x] `__init__.py`
- [x] `allen_cahn.py`
- [x] `blasius.py`
- [x] `bloodhound.py`
- [x] `blowup_fk.py`
- [x] `bvp_system.py`
- [x] `carrier.py`
- [x] `chebop_quiver.py`
- [x] `delay_differential_equations.py`
- [x] `droplets.py`
- [x] `exact_solns.py`
- [x] `fourier_collocation_nonlin.py`
- [x] `guckenheimer_holmes.py`
- [x] `gulf_stream.py`
- [x] `ivp_capabilities.py`
- [x] `lane_emden.py`
- [x] `logistic.py`
- [x] `logistic2.py`
- [x] `lorenz_attractor.py`
- [x] `lyapunov_exponents.py`
- [x] `modelling_diseases.py`
- [x] `orbits.py`
- [x] `picard.py`
- [x] `square_cycle.py`
- [x] `three_body_problem.py`
- [x] `three_planets.py`
- [x] `two_electrons.py`

## examples/ode-random (11 scripts)

- [x] `__init__.py`
- [x] `consensus.py`
- [x] `gbm.py`
- [x] `levelhopping.py`
- [x] `phaselocking.py`
- [x] `pitchfork.py`
- [x] `random2sde.py`
- [x] `randomonasphere.py`
- [x] `randomswitching.py`
- [x] `tunnelling.py`
- [x] `whitenoiseparadox.py`

## examples/opt (12 scripts)

- [x] `__init__.py`
- [x] `catenary.py`
- [x] `constrained_extrema.py`
- [x] `constrained_optimization.py`
- [x] `dixon_szego.py`
- [x] `extreme_extrema.py`
- [x] `global_minimum.py`
- [x] `mercury_earth.py`
- [x] `needle.py`
- [x] `optim_int.py`
- [x] `rosenbrock.py`
- [x] `rosenbrock2.py`

## examples/opt_new (3 scripts)

- [x] `__init__.py`
- [x] `optimization_1d.py`
- [x] `optimization_2d.py`

## examples/pde (11 scripts)

- [x] `__init__.py`
- [x] `bsexponential.py`
- [x] `erosion.py`
- [x] `fourierexpm.py`
- [x] `ginzburglandau.py`
- [x] `grayscott.py`
- [x] `kdv.py`
- [x] `kswave.py`
- [x] `kuramoto.py`
- [x] `swifthohenberg.py`
- [x] `trapezoideigs.py`

## examples/pde_new (4 scripts)

- [x] `__init__.py`
- [x] `allen_cahn.py`
- [x] `heat_equation.py`
- [x] `kdv_equation.py`

## examples/quad (9 scripts)

- [x] `__init__.py`
- [x] `gauss_clen_curt.py`
- [x] `hermite_quad.py`
- [x] `quadrature_convergence.py`
- [x] `spike_integral.py`
- [x] `sumdisk_demo.py`
- [x] `symbolic_numeric.py`
- [x] `tjtkdisk.py`
- [x] `tricky.py`

## examples/roots (19 scripts)

- [x] `__init__.py`
- [x] `aaa_zeros.py`
- [x] `average_degree_reduction_1d.py`
- [x] `average_degree_reduction_2d.py`
- [x] `bessel_roots.py`
- [x] `bivariate_roots.py`
- [x] `complex_roots.py`
- [x] `fundamental_theorem_of_algebra.py`
- [x] `marching_squares.py`
- [x] `newton_raphson.py`
- [x] `random_polynomials.py`
- [x] `random_polys.py`
- [x] `resultant_method.py`
- [x] `roots_near_axis.py`
- [x] `roots_speed.py`
- [x] `secular_roots.py`
- [x] `subramanian.py`
- [x] `tiger.py`
- [x] `white_curves.py`

## examples/sphere (16 scripts)

- [x] `__init__.py`
- [x] `advectiondiffusion.py`
- [x] `atmospherictemperature.py`
- [x] `gravity.py`
- [x] `helmholtzdecomposition.py`
- [x] `helmholtzdecompositionball.py`
- [x] `laplaceball.py`
- [x] `ptdecomposition.py`
- [x] `rayleighquotientexample.py`
- [x] `solidharmonics.py`
- [x] `sphere_operations.py`
- [x] `spherefunpartition.py`
- [x] `spherefunrotate.py`
- [x] `sphereheatconduction.py`
- [x] `spherical_harmonics.py`
- [x] `sphericalharmonics.py`

## examples/stats (17 scripts)

- [x] `__init__.py`
- [x] `bayesian_gradebook.py`
- [x] `bivariate_normal_distribution.py`
- [x] `central_limit_theorem.py`
- [x] `expectations.py`
- [x] `histogram.py`
- [x] `least_squares.py`
- [x] `mercer_karhunen_loeve.py`
- [x] `normal_exercises.py`
- [x] `probability_convolution.py`
- [x] `random_maxima.py`
- [x] `random_polynomials.py`
- [x] `random_surf.py`
- [x] `resampling_random_variables.py`
- [x] `smooth_random_walk.py`
- [x] `smoothies.py`
- [x] `uniform_exercises.py`

## examples/temp (4 scripts)

- [x] `__init__.py`
- [x] `binousshaikhbellagi.py`
- [x] `compactingcolloids.py`
- [x] `taylorstheorem.py`

## examples/veccalc (4 scripts)

- [x] `__init__.py`
- [x] `autonomous_systems.py`
- [x] `checking_vector_calculus.py`
- [x] `vector_calculus.py`


# Guide (docs/guide)

All 20 chapters translated with regenerated figures.

- [x] `guide01.md` (15 figures)
- [x] `guide02.md` (14 figures)
- [x] `guide03.md` (18 figures)
- [x] `guide04.md` (22 figures)
- [x] `guide05.md` (21 figures)
- [x] `guide06.md` (12 figures)
- [x] `guide07.md` (19 figures)
- [x] `guide08.md` (12 figures)
- [x] `guide09.md` (16 figures)
- [x] `guide10.md` (11 figures)
- [x] `guide11.md` (12 figures)
- [x] `guide12.md` (10 figures)
- [x] `guide13.md` (7 figures)
- [x] `guide14.md` (7 figures)
- [x] `guide15.md` (10 figures)
- [x] `guide16.md` (28 figures)
- [x] `guide17.md` (28 figures)
- [x] `guide18.md` (16 figures)
- [x] `guide19.md` (18 figures)
- [x] `guide20.md` (26 figures)
