"""Independent forcing/fit checks; source restart-off assertion stays separate.

Provenance
----------
MATLAB source : @chebfun/constructODEsol.m, tests/chebop/test_shortPulses.m
Chebfun commit: 7574c77
The 2e-8 absolute accuracy target is retained from the existing scalar-IVP
fit controls, with rtol0; original source restart-off remains strict1e-10.
"""
import importlib

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebpref import ChebopPref
from chebfunjax.operators.chebop import Chebop


def test_restart_on_analytic_pulse_accuracy():
    t = chebfun(lambda x: x, domain=(0.0, 2.0))
    forcing = 20 * (t > 1) * (t < 1.05)
    operator = Chebop(lambda u: u.diff() + u, (0.0, 2.0), 1.0, None)
    operator.ivp_restart_solver = True
    u = operator.solve(forcing)
    x = jnp.asarray([0.25, 0.75, 1.01, 1.025, 1.049, 1.1, 1.5, 2.0])
    expected = (jnp.exp(-x) + 20 * (
        jnp.where(x > 1, -jnp.expm1(-(x - 1)), 0)
        - jnp.where(x > 1.05, -jnp.expm1(-(x - 1.05)), 0)))
    np.testing.assert_allclose(u(x), expected, atol=2e-8, rtol=0)
    assert tuple(u.domain.breakpoints) == (0.0, 1.0, 1.05, 2.0)


def test_restart_off_retains_supplied_fit_breaks():
    # A smooth analytic problem with an explicit domain break isolates fit
    # partition policy from the source/SciPy narrow-pulse stepping difference.
    operator = Chebop(lambda u: u.diff() + u, (0.0, 0.75, 2.0), 1.0, None)
    operator.ivp_restart_solver = False
    u = operator.solve_ivp(0.0)
    assert 0.75 in tuple(u.domain.breakpoints)
    x = jnp.asarray([0.125, 0.5, 1.25, 1.75])
    np.testing.assert_allclose(u(x), jnp.exp(-x), atol=2e-8, rtol=0)


def test_explicit_lsoda_adapter_restart_agreement():
    # Preserve the former Python-specific predicate as an explicit adapter
    # check, not as MATLAB pass3. Its original 1e-4 bound is unchanged.
    t = chebfun(lambda x: x, domain=(0.0, 2.0))
    forcing = 20 * (t > 1) * (t < 1.05)
    operator = Chebop(lambda u: u.diff() + u, (0.0, 2.0), 1.0, None)
    operator.ivp_method = 'LSODA'
    operator.ivp_restart_solver = True
    restarted = operator.solve(forcing)
    operator.ivp_restart_solver = False
    uninterrupted = operator.solve(forcing)
    assert float((restarted - uninterrupted).norm(2)) < 1e-4


def test_explicit_native_terminal_event_does_not_fall_back(monkeypatch):
    # solveivp.m269-280: abs(y)-maxnorm, terminal1, either direction.
    # constructODEsol.m40-52: stop at xe and join NaNs to the original end.
    provider = importlib.import_module('chebfunjax.utils.native_ode113')
    original = provider.native_ode113
    calls = []

    def observed(fun, span, initial, options=None, **kwargs):
        result = original(fun, span, initial, options, **kwargs)
        calls.append((span, initial, dict(options or {}), result))
        return result

    def forbidden_fallback(*args, **kwargs):
        raise AssertionError('explicit native event must not use SciPy fallback')

    monkeypatch.setattr(provider, 'native_ode113', observed)
    monkeypatch.setattr('scipy.integrate.solve_ivp', forbidden_fallback)
    operator = Chebop(lambda u: u.diff() - u, (0.0, 2.0), 1.0, None)
    operator.ivp_method = 'chebfun.ode113'
    operator.maxnorm = 2.0
    result = operator.solve(0.0)
    assert operator._ivp_backend_used == 'native_ode113'
    assert len(calls) == 1
    span, initial, options, native = calls[0]
    np.testing.assert_array_equal(span, [0., 2.])
    np.testing.assert_array_equal(initial, [1.])
    pref = ChebopPref()
    assert options['RelTol'] == pref.ivpRelTol
    assert options['AbsTol'] == pref.ivpAbsTol
    value, terminal, direction = options['Events'](0., jnp.asarray([-2.]))
    np.testing.assert_array_equal(value, [0.])
    np.testing.assert_array_equal(terminal, [1.])
    assert float(direction) == 0.
    np.testing.assert_array_equal(native['ie'], [1])
    np.testing.assert_allclose(native['xe'], [jnp.log(2.)], atol=2e-8, rtol=0)
    np.testing.assert_allclose(native['ye'], [[2.]], atol=2e-8, rtol=0)
    cutoff = float(native['xe'][0])
    np.testing.assert_array_equal(result.domain.breakpoints, [0., cutoff, 2.])
    points = jnp.asarray([0., .25, .5])
    np.testing.assert_allclose(result(points), jnp.exp(points), atol=2e-8, rtol=0)
    assert bool(jnp.all(jnp.isnan(result(jnp.asarray([1., 2.])))))
