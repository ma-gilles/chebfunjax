"""Literal native tests/chebop/test_LorenzIVP.m endpoint predicate.

Chebfun source pin7574c77680d7e82b79626300bf255498271a72df.
The independent operator/direct routes share the accepted R2025b Adams
provider; this is not a fresh MATLAB R2017 runtime comparison.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.operators.chebop import Chebop
from chebfunjax.utils.native_ode113 import native_ode113

EPS = 2.220446049250313e-16


@pytest.fixture(autouse=True)
def factory_preferences(monkeypatch):
    from chebfunjax.chebpref import ChebfunPref, ChebopPref
    # monkeypatch restores each exact prior object even after test failure.
    monkeypatch.setattr(ChebfunPref, '_defaults', None)
    monkeypatch.setattr(ChebopPref, '_defaults', None)
    pref = ChebopPref()
    assert pref.ivpSolver == 'ode113'
    assert pref.ivpAbsTol == 1e5*EPS and pref.ivpRelTol == 100*EPS
    yield


def test_native_lorenz_original_predicate():
    n = Chebop(lambda t, u, v, w: [u.diff()-10*(v-u),
                                  v.diff()-u*(28-w)+v,
                                  w.diff()-u*v+(8/3)*w], domain=(0, 5))
    n.lbc = lambda u, v, w: [w-20, v+15, u+14]
    solved = n.solve([0, 0, 0])
    def rhs(t, y):
        return jnp.asarray([10*(y[1]-y[0]), y[0]*(28-y[2])-y[1],
                            y[0]*y[1]-(8/3)*y[2]])
    reference = native_ode113(rhs, [0, 5], [-14, -15, 20],
                             {'AbsTol': 1e5*EPS, 'RelTol': 100*EPS})
    error = jnp.stack([u(5.) for u in solved])-reference['sol'](jnp.asarray([5.]))[:, 0]
    assert float(jnp.linalg.norm(error)) < 1e-14
