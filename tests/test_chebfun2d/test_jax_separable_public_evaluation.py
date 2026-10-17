"""Public CDR evaluation without host numerical dispatch.

Provenance
----------
MATLAB source: @separableApprox/{feval,cdr}.m, Chebfun7574c77.
Low-degree monomial and trigonometric functions give independent value and
coordinate-derivative oracles. Host dispatch guards are JAX adapter controls.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun2d import separable_approx as module
from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.separable_approx import SeparableApprox
from chebfunjax.tech import trigtech
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

DOMAIN = (.1, 2.7, -4.3, .2)


def _field(kind, complex_values=False):
    first = (1 + .5j) if complex_values else 1.
    weight = (1 - .25j) if complex_values else 1.
    cols = [kind.from_coeffs(first*jnp.array([1., .5, .125])),
            kind.from_coeffs(jnp.array([2., -.25]))]
    rows = [kind.from_coeffs(jnp.array([.75, 2., -.25])),
            kind.from_coeffs(jnp.array([-.5, .125, .5]))]
    return SeparableApprox(cols, rows, jnp.array([.75, -.25*weight]), DOMAIN)


def _exact(x, y, complex_values=False):
    tx = (x-DOMAIN[0])/(DOMAIN[1]-DOMAIN[0])-(DOMAIN[1]-x)/(DOMAIN[1]-DOMAIN[0])
    ty = (y-DOMAIN[2])/(DOMAIN[3]-DOMAIN[2])-(DOMAIN[3]-y)/(DOMAIN[3]-DOMAIN[2])
    a = (1+.5j) if complex_values else 1.
    b = (1-.25j) if complex_values else 1.
    return (.75*a*(.875+.5*ty+.25*ty**2)*(1.+2*tx-.5*tx**2)
            -.25*b*(2.-.25*ty)*(-1.+.125*tx+tx**2))


class _NoHostNumerics:
    def __getattr__(self, name):
        raise AssertionError(f"SeparableApprox host NumPy dispatch: {name}")


@pytest.mark.parametrize('kind', [Chebtech1, Chebtech2])
@pytest.mark.parametrize('complex_values', [False, True])
@pytest.mark.parametrize('layout', ['points', 'xy', 'ij', 'broadcast', 'complex_coordinates'])
def test_public_values_use_jax(monkeypatch, kind, complex_values, layout):
    field = _field(kind, complex_values)
    x, y = jnp.array([.1, .35, 1.9, 2.7]), jnp.array([-4.3, -2.7, -.1, .2])
    if layout == 'xy':
        x, y = jnp.meshgrid(x, y)
    elif layout == 'ij':
        x, y = jnp.meshgrid(x, y, indexing='ij')
    elif layout == 'broadcast':
        x, y = x[None, :], y[:, None]
    elif layout == 'complex_coordinates':
        x, y = x+.125j, y-.25j
    monkeypatch.setattr(module, 'np', _NoHostNumerics())
    expected = _exact(x, y, complex_values)
    for function in (field, Chebfun2(approx=field)):
        np.testing.assert_allclose(function(x, y), expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize('kind', [Chebtech1, Chebtech2])
def test_public_jit_vmap_and_coordinate_derivatives(kind):
    field = _field(kind)
    x, y = jnp.array([.35, 1.9]), jnp.array([-2.7, -.1])
    compiled = jax.jit(lambda a, b: field(a, b))
    for function in (field, compiled, jax.vmap(field)):
        np.testing.assert_allclose(function(x, y), _exact(x, y), rtol=1e-13, atol=1e-13)
    tangent = (jnp.array([.2, -.3]), jnp.array([-.1, .4]))
    for function in (field, compiled):
        actual = jax.jvp(function, (x, y), tangent)
        expected = jax.jvp(_exact, (x, y), tangent)
        for a, e in zip(actual, expected):
            np.testing.assert_allclose(a, e, rtol=1e-13, atol=1e-13)


def test_native_near_mesh_recognition_precedes_mapping():
    field = _field(Chebtech2)
    # Native compares physical coordinates to 10eps, not mapped coordinates.
    x, y = jnp.meshgrid(jnp.array([.35, 1.9]), jnp.array([-2.7, -.1]))
    x = x.at[1, 0].add(4*jnp.finfo(float).eps)
    expected = _exact(jnp.broadcast_to(x[:1, :], x.shape), y)
    np.testing.assert_allclose(field(x, y), expected, rtol=0, atol=5e-15)


def test_trig_slices_avoid_eager_numpy(monkeypatch):
    cols = [trigtech.Trigtech.from_function(lambda t: jnp.cos(jnp.pi*t)),
            trigtech.Trigtech.from_function(lambda t: jnp.sin(2*jnp.pi*t))]
    rows = [trigtech.Trigtech.from_function(lambda t: jnp.sin(jnp.pi*t)),
            trigtech.Trigtech.from_function(lambda t: jnp.ones_like(t))]
    field = SeparableApprox(cols, rows, jnp.array([.75, .5]), (-1., 1., -1., 1.))
    monkeypatch.setattr(module, 'np', _NoHostNumerics())
    def forbidden(*args, **kwargs):
        raise AssertionError('Trigtech eager NumPy evaluator called')
    monkeypatch.setattr(trigtech, '_trig_eval_np', forbidden)
    x, y = jnp.meshgrid(jnp.array([-.8, -.1, .4]), jnp.array([-.6, .2]))
    expected = .75*jnp.cos(jnp.pi*y)*jnp.sin(jnp.pi*x)+.5*jnp.sin(2*jnp.pi*y)
    np.testing.assert_allclose(field(x, y), expected, rtol=1e-13, atol=1e-13)
