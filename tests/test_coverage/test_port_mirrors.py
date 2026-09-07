"""Core-suite mirrors of MATLAB-port tests (Fable 5).

The coverage job runs only the core suites (tests/ minus
tests/test_matlab_port); the MATLAB-port tests that exercise the
newer modules (spherefun/diskfun BMC arithmetic, expinteg schemes,
linop adjoint/svds, followpath, randfuns, cheb gallery, spin
preferences, ...) are re-run here so that code counts towards the
coverage gate.  Each port module is one parametrised case; module-level
skip marks are honoured, parametrised port tests are skipped (they need
pytest's own collection), and pytest.skip/xfail raised inside count as
skips.  The heaviest port modules (>100 s each: carrier_C2, legpoly,
shortPulses, diskfun helmholtz, chebfun3 guide/constructor, chebfun2
poisson) are left to the port-tree shards.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

MODULES = [
    'tests.test_matlab_port.chebfun2.test_battery_matlab',
    'tests.test_matlab_port.chebfun2.test_chol_matlab',
    'tests.test_matlab_port.chebfun2.test_CLA_matlab',
    'tests.test_matlab_port.chebfun2.test_constructor2_matlab',
    'tests.test_matlab_port.chebfun2.test_constructor_matlab',
    'tests.test_matlab_port.chebfun2.test_contour3_matlab',
    'tests.test_matlab_port.chebfun2.test_end_matlab',
    'tests.test_matlab_port.chebfun2.test_equiOption_matlab',
    'tests.test_matlab_port.chebfun2.test_integralEqns_matlab',
    'tests.test_matlab_port.chebfun2.test_mixed_tech_matlab',
    'tests.test_matlab_port.chebfun2.test_poldec_matlab',
    'tests.test_matlab_port.chebfun2.test_subsref_matlab',
    'tests.test_matlab_port.chebfun2.test_trig_matlab',
    'tests.test_matlab_port.chebfun2.test_vectoriseFlag_matlab',
    'tests.test_matlab_port.chebfun2v.test_roots10_matlab',
    'tests.test_matlab_port.chebfun2v.test_roots_slow_matlab',
    'tests.test_matlab_port.chebfun3._battery',
    'tests.test_matlab_port.chebfun3.test_battery_matlab',
    'tests.test_matlab_port.chebfun3.test_chebcoeffs3_matlab',
    'tests.test_matlab_port.chebfun3.test_chebfun3f_matlab',
    'tests.test_matlab_port.chebfun3.test_chebpolyval3_matlab',
    'tests.test_matlab_port.chebfun3.test_coefficients_matlab',
    'tests.test_matlab_port.chebfun3.test_constructor2_matlab',
    'tests.test_matlab_port.chebfun3.test_domainChck_matlab',
    'tests.test_matlab_port.chebfun3.test_isPeriodicTech_matlab',
    'tests.test_matlab_port.chebfun3.test_minandmax3est_matlab',
    'tests.test_matlab_port.chebfun3.test_techs_matlab',
    'tests.test_matlab_port.chebfun3.test_trigs_matlab',
    'tests.test_matlab_port.chebfun3.test_zerofunction_matlab',
    'tests.test_matlab_port.chebfun3t.test_ndf_matlab',
    'tests.test_matlab_port.chebfun.test_bvp4c_matlab',
    'tests.test_matlab_port.chebfun.test_bvp5c_matlab',
    'tests.test_matlab_port.chebfun.test_changeTech_matlab',
    'tests.test_matlab_port.chebfun.test_chebfun_lu_matlab',
    'tests.test_matlab_port.chebfun.test_constructor_basic_matlab',
    'tests.test_matlab_port.chebfun.test_constructor_basic_periodic_matlab',
    'tests.test_matlab_port.chebfun.test_constructor_inputs_matlab',
    'tests.test_matlab_port.chebfun.test_constructor_inputs_periodic_matlab',
    'tests.test_matlab_port.chebfun.test_constructor_splitting_matlab',
    'tests.test_matlab_port.chebfun.test_defineInterval_matlab',
    'tests.test_matlab_port.chebfun.test_definePoint_matlab',
    'tests.test_matlab_port.chebfun.test_deltaOps_matlab',
    'tests.test_matlab_port.chebfun.test_doubleLength_matlab',
    'tests.test_matlab_port.chebfun.test_end_matlab',
    'tests.test_matlab_port.chebfun.test_find_matlab',
    'tests.test_matlab_port.chebfun.test_ivp_matlab',
    'tests.test_matlab_port.chebfun.test_nodots_matlab',
    'tests.test_matlab_port.chebfun.test_points_matlab',
    'tests.test_matlab_port.chebfun.test_polyval_matlab',
    'tests.test_matlab_port.chebfun.test_range_matlab',
    'tests.test_matlab_port.chebfun.test_removeDeltas_matlab',
    'tests.test_matlab_port.chebfun.test_splitting_abs_matlab',
    'tests.test_matlab_port.chebfun.test_trig_matlab',
    'tests.test_matlab_port.chebfun.test_trigratinterp_matlab',
    'tests.test_matlab_port.chebfun.test_truncate_matlab',
    'tests.test_matlab_port.chebfun.test_tweakDomain_matlab',
    'tests.test_matlab_port.chebfun.test_ultracoeffs_matlab',
    'tests.test_matlab_port.chebfun.test_vectorCheck_matlab',
    'tests.test_matlab_port.chebop2.test_adaptivity_matlab',
    'tests.test_matlab_port.chebop2.test_backwardsWaveEquation_matlab',
    'tests.test_matlab_port.chebop2.test_rhs2_matlab',
    'tests.test_matlab_port.chebop2.test_separableFormat_matlab',
    'tests.test_matlab_port.chebop2.test_squarewaveequation_matlab',
    'tests.test_matlab_port.chebop2.test_subsref_matlab',
    'tests.test_matlab_port.chebop2.test_waveequation_matlab',
    'tests.test_matlab_port.chebop2.test_withoutAD_matlab',
    'tests.test_matlab_port.chebop.test_ellipjODE_matlab',
    'tests.test_matlab_port.chebop.test_followpath_matlab',
    'tests.test_matlab_port.chebop.test_ivp_chebmatrix_syntax_matlab',
    'tests.test_matlab_port.chebop.test_linearize_init_fails_matlab',
    'tests.test_matlab_port.chebop.test_linearize_matlab',
    'tests.test_matlab_port.chebop.test_linearSystem2_matlab',
    'tests.test_matlab_port.chebop.test_nonlinSys2_C2_matlab',
    'tests.test_matlab_port.chebop.test_pantograph_matlab',
    'tests.test_matlab_port.cheb.test_bernoulli_matlab',
    'tests.test_matlab_port.cheb.test_bspline_matlab',
    'tests.test_matlab_port.cheb.test_galleryball_matlab',
    'tests.test_matlab_port.cheb.test_normal2_matlab',
    'tests.test_matlab_port.cheb.test_revolution_matlab',
    'tests.test_matlab_port.deltafun.test_feval_matlab',
    'tests.test_matlab_port.diskfun._cart',
    'tests.test_matlab_port.diskfun.test_biharm_matlab',
    'tests.test_matlab_port.diskfun.test_BMCsvd_matlab',
    'tests.test_matlab_port.diskfun.test_coeffs2vals_vals2coeffs_matlab',
    'tests.test_matlab_port.diskfun.test_curl_matlab',
    'tests.test_matlab_port.diskfun.test_diag_matlab',
    'tests.test_matlab_port.diskfun.test_grad_matlab',
    'tests.test_matlab_port.diskfun.test_inherited_matlab',
    'tests.test_matlab_port.diskfun.test_median_matlab',
    'tests.test_matlab_port.diskfun.test_projection_matlab',
    'tests.test_matlab_port.diskfunv.test_coeffs_vals_matlab',
    'tests.test_matlab_port.diskfunv.test_compose_matlab',
    'tests.test_matlab_port.diskfunv.test_diff_matlab',
    'tests.test_matlab_port.diskfunv.test_dot_matlab',
    'tests.test_matlab_port.diskfunv.test_feval_matlab',
    'tests.test_matlab_port.diskfunv.test_get_matlab',
    'tests.test_matlab_port.diskfunv.test_subsref_matlab',
    'tests.test_matlab_port.domain.test_polyfit_matlab',
    'tests.test_matlab_port.domain.test_poly_matlab',
    'tests.test_matlab_port.functionalBlock.test_isNotMultOrDiff_matlab',
    'tests.test_matlab_port.linop.test_eigsRayleigh_matlab',
    'tests.test_matlab_port.linop.test_feval_lr_matlab',
    'tests.test_matlab_port.linop.test_linopAdjoint_matlab',
    'tests.test_matlab_port.linop.test_svds_matlab',
    'tests.test_matlab_port.misc.test_blowup_matlab',
    'tests.test_matlab_port.misc.test_chebpoly_matlab',
    'tests.test_matlab_port.misc.test_chebpolyval_matlab',
    'tests.test_matlab_port.misc.test_chebpolyvalm_matlab',
    'tests.test_matlab_port.misc.test_chebvar_matlab',
    'tests.test_matlab_port.misc.test_conformal2_matlab',
    'tests.test_matlab_port.misc.test_conformal_matlab',
    'tests.test_matlab_port.misc.test_gpr_matlab',
    'tests.test_matlab_port.misc.test_minimax_matlab',
    'tests.test_matlab_port.misc.test_pde15s_matlab',
    'tests.test_matlab_port.misc.test_pswf_matlab',
    'tests.test_matlab_port.misc.test_pswfpts_matlab',
    'tests.test_matlab_port.misc.test_quantumstates_matlab',
    'tests.test_matlab_port.misc.test_randnfundisk_matlab',
    'tests.test_matlab_port.misc.test_randnfun_matlab',
    'tests.test_matlab_port.misc.test_randnfunsphere_matlab',
    'tests.test_matlab_port.misc.test_smoothie_matlab',
    'tests.test_matlab_port.misc.test_splitting_matlab',
    'tests.test_matlab_port.operatorBlock.test_isNotMultOrDiff_matlab',
    'tests.test_matlab_port.spherefun._cart',
    'tests.test_matlab_port.spherefun.test_BMCsvd_matlab',
    'tests.test_matlab_port.spherefun.test_cdr_matlab',
    'tests.test_matlab_port.spherefun.test_coeffs2_matlab',
    'tests.test_matlab_port.spherefun.test_coeffs2vals_vals2coeffs_matlab',
    'tests.test_matlab_port.spherefun.test_curl_matlab',
    'tests.test_matlab_port.spherefun.test_inherited_matlab',
    'tests.test_matlab_port.spherefun.test_norm_matlab',
    'tests.test_matlab_port.spherefun.test_projection_matlab',
    'tests.test_matlab_port.spherefun.test_svd_matlab',
    'tests.test_matlab_port.spinop.test_spinop_matlab',
    'tests.test_matlab_port.spinpref2.test_spinpref2_matlab',
    'tests.test_matlab_port.spinpref3.test_spinpref3_matlab',
    'tests.test_matlab_port.spinprefsphere.test_spinprefsphere_matlab',
    'tests.test_matlab_port.spinpref.test_spinpref_matlab',
    'tests.test_matlab_port.spinscheme.test_phiFun_matlab',
    'tests.test_matlab_port.spinscheme.test_startMultistep_matlab',
    'tests.test_matlab_port.trigspec.test_multmat_matlab',
]


def _module_skip(mod) -> str | None:
    marks = getattr(mod, "pytestmark", None)
    if marks is None:
        return None
    if not isinstance(marks, (list, tuple)):
        marks = [marks]
    for m in marks:
        if getattr(m, "name", "") == "skip":
            return str(m.kwargs.get("reason", "module skip"))
    return None


def _plain(fn) -> bool:
    params = [p for p in inspect.signature(fn).parameters.values()
              if p.name != "self"]
    return not params and not hasattr(fn, "pytestmark")


@pytest.mark.parametrize("modname", MODULES)
@pytest.mark.timeout(900)
def test_port_mirror(modname):
    mod = importlib.import_module(modname)
    reason = _module_skip(mod)
    if reason:
        pytest.skip(reason)
    ran = 0
    skipped = 0
    for name, obj in list(vars(mod).items()):
        if inspect.isclass(obj) and name.startswith("Test"):
            inst = obj()
            for mname, meth in inspect.getmembers(obj, inspect.isfunction):
                if not mname.startswith("test") or not _plain(meth):
                    continue
                try:
                    getattr(inst, mname)()
                    ran += 1
                except (pytest.skip.Exception, pytest.xfail.Exception):
                    skipped += 1
        elif inspect.isfunction(obj) and name.startswith("test") and _plain(obj):
            try:
                obj()
                ran += 1
            except (pytest.skip.Exception, pytest.xfail.Exception):
                skipped += 1
    if ran == 0:
        pytest.skip(f"no plain test callables in {modname} ({skipped} skipped)")
