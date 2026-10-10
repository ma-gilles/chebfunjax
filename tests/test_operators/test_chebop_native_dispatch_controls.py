"""Real native scalar dispatch, option forwarding, and stale-marker controls.

Provenance
----------
MATLAB source : @chebop/solveivp.m, @cheboppref/cheboppref.m,
    @chebfun/constructODEsol.m
Chebfun commit: 7574c77
Analytic oracle: u'=u, u(0)=1 or u(1)=e. Diagnostic accuracy bound2e-8
matches existing scalar-IVP fit controls; it does not replace source bounds.
"""
import importlib
import math

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.integrate

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.operators.chebop import Chebop


@pytest.mark.parametrize('reverse, explicit', [(False, False), (True, True)])
def test_public_chebop_uses_actual_native_without_fallback(monkeypatch, reverse, explicit):
    module = importlib.import_module('chebfunjax.chebfun1d.chebfun')
    original = module.ode113
    calls = []

    def observe(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail('native scalar solve attempted a SciPy/collocation fallback')

    monkeypatch.setattr(module, 'ode113', observe)
    monkeypatch.setattr(scipy.integrate, 'solve_ivp', forbidden)
    for name in ('_solve_linear', '_solve_nonlinear', '_solve_ivp_system_highorder',
                 '_solve_piecewise'):
        monkeypatch.setattr(Chebop, name, forbidden)
    problem = Chebop(lambda u: u.diff() - u, (0.0, 1.0))
    forcing = chebfun(lambda x: jnp.zeros_like(x), domain=(0.0, 0.375, 1.0))
    if reverse:
        problem.rbc = math.e
    else:
        problem.lbc = 1.0
    # Explicit native selection must override an existing explicit adapter.
    if explicit:
        problem.ivp_method = 'LSODA'
    options = {'ivp_solver': 'chebfun.ode113'} if explicit else {}
    solution = problem.solve(forcing, **options)
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert kwargs['backend'] == 'native'
    expected_span = (1.0, 0.375, 0.0) if reverse else (0.0, 0.375, 1.0)
    assert tuple(args[1]) == expected_span
    assert args[3]['RelTol'] == 100 * np.finfo(float).eps
    assert args[3]['AbsTol'] == 1e5 * np.finfo(float).eps
    assert args[3]['restartSolver'] is True
    assert problem._ivp_backend_used == 'native_ode113'
    x = jnp.asarray([0.0, 0.125, 0.5, 0.875, 1.0])
    np.testing.assert_allclose(solution(x), jnp.exp(x), rtol=0, atol=2e-8)
    assert 0.375 in tuple(solution.domain.breakpoints)


def test_reused_operator_resets_native_marker_before_extraction_error():
    problem = Chebop(lambda u: u.diff() - u, (0.0, 2.0), 1.0, None)
    problem.ivp_method = 'ode113'
    problem.maxnorm = 2.0
    solution = problem.solve(0.0)
    # Native maxnorm now terminates this solve at log(2), with NaN padding.
    np.testing.assert_allclose(solution(0.5), math.exp(0.5), rtol=0, atol=2e-8)
    assert np.isnan(float(solution(1.0)))
    assert problem._ivp_backend_used == 'native_ode113'
    problem.maxnorm = None
    with pytest.raises(TypeError, match='forcing must be a real scalar'):
        problem.solve_ivp(jnp.asarray([1.0, 2.0]))
    assert problem._ivp_backend_used is None
