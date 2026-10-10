"""Independent source boundaries; no MATLAB executable capture claim."""
# uses-numpy: independent host test oracle and assertions only.
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun2d.chebfun2v import Chebfun2v, _field_ode_rhs
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.utils import native_ode45 as solver_module


def nonlinear(t, y):
    return jnp.array([t+y[1], y[0]*y[1]-t], dtype=jnp.float64)


def test_literal_stage_products():
    # Rows of the published DP tableau, evaluated as a separate eager oracle.
    A = np.array([1/5, 3/10, 4/5, 8/9, 1, 1], dtype=float)
    B = np.array([[1/5, 3/40, 44/45, 19372/6561, 9017/3168, 35/384],
                  [0, 9/40, -56/15, -25360/2187, -355/33, 0],
                  [0, 0, 32/9, 64448/6561, 46732/5247, 500/1113],
                  [0, 0, 0, -212/729, 49/176, 125/192],
                  [0, 0, 0, 0, -5103/18656, -2187/6784],
                  [0, 0, 0, 0, 0, 11/84], [0, 0, 0, 0, 0, 0]], dtype=float)
    t, h = .125, .0625
    y = np.array([.75, -.25])
    f = np.zeros((2, 7))
    def rhs(t, y):
        return np.array([t+y[1], y[0]*y[1]-t])
    f[:, 0] = rhs(t, y)
    before = f.copy()
    for k in range(5):
        f[:, k+1] = rhs(t+h*A[k], y+f@(h*B[:, k]))
    expected = y+f@(h*B[:, 5])
    f[:, 6] = rhs(t+h, expected)
    tn, yn, hn, actual = solver_module._stages(nonlinear, jnp.asarray(t), jnp.asarray(y),
                                                jnp.asarray(h), jnp.asarray(before),
                                                jnp.asarray(t+h), jnp.asarray(True), ())
    assert float(tn) == t+h and float(hn) == h
    np.testing.assert_allclose(yn, expected, rtol=0, atol=32*np.finfo(float).eps)
    np.testing.assert_allclose(actual, f, rtol=0, atol=32*np.finfo(float).eps)


@pytest.mark.parametrize('point', [(0., 0.), (-1., 1.), (1., -1.),
                                  (-1.0001, 0.), (0., 1.0001), (1.0001, -1.0001)])
def test_inclusive_rectangle(point):
    field = Chebfun2v.from_functions(lambda x, y: x+2*y, lambda x, y: y)
    x, y = point
    indicator = float(x >= -1)*float(x <= 1)*float(y >= -1)*float(y <= 1)
    got = _field_ode_rhs(jnp.asarray(0.), jnp.asarray(point), field)
    np.testing.assert_allclose(got, np.array([x+2*y, y])*indicator,
                               rtol=0, atol=32*np.finfo(float).eps)


@pytest.mark.parametrize('components,columns', [(2, 2), (2, 4), (3, 2), (3, 4)])
def test_literal_raw_mesh_constructor(monkeypatch, components, columns):
    field = Chebfun2v.from_functions(*([lambda x, y: x+0*y]*components))
    module = importlib.import_module('chebfunjax.chebfun1d.chebfun')
    captured = []
    marker = object()
    def constructor(values, **kwargs):
        captured.append((jnp.asarray(values), kwargs))
        return marker
    monkeypatch.setattr(module, 'chebfun', constructor)
    values = jnp.arange(components*columns, dtype=jnp.float64).reshape(components, columns)
    monkeypatch.setattr(solver_module, 'native_ode45', lambda *a, **k:
                        {'x': jnp.array([0., 2.]), 'y': values})
    T, Y = field.ode45((0., 2.), [0.]*components)
    assert T is marker and Y is marker
    np.testing.assert_array_equal(captured[0][0], [0., 2.])
    expected = (values[:, 0]+1j*values[:, 1] if columns == 2
                else values[0, :]+1j*values[1, :])+np.finfo(float).eps*1j
    np.testing.assert_array_equal(captured[1][0], expected)
    if components == 3:
        np.testing.assert_array_equal(captured[2][0], values if columns == 2 else values.T)
    assert all(item[1] == {'domain': (0., 2.)} for item in captured)


def test_source_defaults_and_explicit_overrides(monkeypatch):
    captured = []
    def solver(fun, span, initial, options, **kwargs):
        captured.append((fun, options, kwargs))
        return {'x': jnp.array([0., 1.]), 'y': jnp.zeros((2, 4), dtype=jnp.float64)}
    monkeypatch.setattr(solver_module, 'native_ode45', solver)
    field = Chebfun2v.from_functions(lambda x, y: x, lambda x, y: y)
    field.ode45((0., 1.), [0., 0.])
    field.ode45((0., 1.), [0., 0.], {'RelTol': 1e-7}, atol=1e-13)
    field.ode45((0., 1.), [0., 0.], {'RelTol': None, 'AbsTol': []})
    eps = ChebfunPref().techPrefs.chebfuneps
    assert captured[0][1] == {'RelTol': 1e8*(100*eps), 'AbsTol': 100*eps}
    assert captured[1][1] == {'RelTol': 1e-7, 'AbsTol': 1e-13}
    assert captured[2][1] == captured[0][1]
    assert all(item[0] is _field_ode_rhs and item[2]['args'][0] is field for item in captured)
    assert jax.config.x64_enabled
