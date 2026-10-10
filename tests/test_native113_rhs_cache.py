"""RHS compilation reuse with source restart semantics and closure recapture.

Provenance: @chebfun/constructODEsol.m, MATLAB Chebfun 7574c77.
Supplemental performance/analytic controls, not captured MATLAB outputs.
"""
import jax.numpy as jnp
import numpy as np

import chebfunjax as cj
from chebfunjax.utils.native_ode113 import _advance_step, native_ode113


def test_restart_reuses_kernel_but_public_call_recaptures_closure():
    state = {'rate': 1.}

    def rhs(t, y):
        return -state['rate'] * y

    options = {'RelTol': 1e-8, 'AbsTol': 1e-10}
    before = _advance_step._cache_size()
    f = cj.ode113(rhs, [0., .2, .4, .6], [1.], options)
    assert _advance_step._cache_size() - before == 1
    x = jnp.linspace(0, .6, 61)
    np.testing.assert_allclose(np.asarray(f(x)).reshape(-1),
                               np.exp(-np.asarray(x)), rtol=0, atol=2e-8)
    state['rate'] = 2.
    g = cj.ode113(rhs, [0., .2], [1.], options)
    x = jnp.linspace(0, .2, 21)
    np.testing.assert_allclose(np.asarray(g(x)).reshape(-1),
                               np.exp(-2*np.asarray(x)), rtol=0, atol=2e-8)


def test_direct_calls_recapture_same_callable_changed_closure():
    state = {'rate': 3.}

    def rhs(t, y):
        return -state['rate'] * y

    for rate in (3., 4.):
        state['rate'] = rate
        sol = native_ode113(rhs, [0., .2], [1.],
                           {'RelTol': 1e-8, 'AbsTol': 1e-10})
        assert abs(float(sol['y'][0, -1]) - np.exp(-rate*.2)) < 2e-8
