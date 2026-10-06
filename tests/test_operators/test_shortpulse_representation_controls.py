"""Independent forcing/fit checks; source restart-off assertion stays separate.

Provenance
----------
MATLAB source : @chebfun/constructODEsol.m, tests/chebop/test_shortPulses.m
Chebfun commit: 7574c77
The 2e-8 absolute accuracy target is retained from the existing scalar-IVP
fit controls, with rtol0; original source restart-off remains strict1e-10.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
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


def test_explicit_native_unsupported_event_does_not_fall_back():
    operator = Chebop(lambda u: u.diff() - u, (0.0, 2.0), 1.0, None)
    operator.ivp_method = 'chebfun.ode113'
    operator.maxnorm = 2.0
    with pytest.raises(NotImplementedError, match='native scalar ode113 events'):
        operator.solve(0.0)
