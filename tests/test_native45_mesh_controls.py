"""Bounded actual native45 numerical controls, not historical MATLAB capture."""
# uses-numpy: independent assertions only.
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
from chebfunjax.utils import native_ode45 as solver_module


def constant(t, y):
    return jnp.array([1., -2.], dtype=jnp.float64)


@pytest.mark.parametrize('span', [(0., 1.), (1., 0.)])
def test_constant_mesh_and_fsal(span):
    sol = solver_module.native_ode45(constant, span, [1., 2.])
    expected = jnp.array([1., 2.])[:, None]+jnp.array([1., -2.])[:, None]*(sol['x']-span[0])
    np.testing.assert_allclose(sol['y'], expected, rtol=0, atol=64*np.finfo(float).eps)
    assert float(sol['x'][0]) == span[0] and float(sol['x'][-1]) == span[-1]
    assert sol['stats']['nfevals'] == 1+6*(sol['stats']['nsteps']+sol['stats']['nfailed'])
    assert sol['stats']['nfailed'] == 0


def diagonal(t, y):
    return jnp.array([-y[0], 2*y[1]], dtype=jnp.float64)


def test_actual_tight_linear_flow():
    sol = solver_module.native_ode45(diagonal, (0., 2.), [.5, .125],
                                     {'RelTol': 1e-10, 'AbsTol': 1e-12, 'InitialStep': 1.})
    expected = jnp.array([.5, .125])[:, None]*jnp.exp(jnp.array([-1., 2.])[:, None]*sol['x'])
    # Existing explicit tight-options analytic phase-plane bound, unchanged.
    np.testing.assert_allclose(sol['y'], expected, rtol=0, atol=1e-8)
    assert sol['stats']['nfailed'] > 0


def test_public_boundary_mesh_construction(monkeypatch):
    actual = solver_module.native_ode45
    captured = []
    def capture(*args, **kwargs):
        sol = actual(*args, **kwargs)
        captured.append(sol)
        return sol
    monkeypatch.setattr(solver_module, 'native_ode45', capture)
    field = Chebfun2v.from_functions(lambda x, y: 1.+0*x+0*y, lambda x, y: 0*x+0*y)
    T, Y = field.ode45((0., 2.), [0., 0.])
    sol = captured[0]
    assert sol['stats']['nfailed'] > 0
    assert sol['x'][-1] == 2.
    assert jnp.iscomplexobj(Y(jnp.asarray(0.)))
    # Raw-mesh constructor must interpolate these values on Chebyshev nodes;
    # the pinned wrapper does not fit a dense-time solution here.
    n = sol['y'].shape[1]
    nodes = 1-jnp.cos(jnp.linspace(0., jnp.pi, n))
    expected = sol['y'][0]+1j*sol['y'][1]+jnp.finfo(jnp.float64).eps*1j
    np.testing.assert_allclose(Y(nodes), expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(T(jnp.array([0., .5, 2.])), [0., .5, 2.], rtol=0, atol=1e-14)
