"""Exact tensor-grid recognition, complex values and generic fallback."""
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d._tensor_feval import source_tensor_grid
from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.tech.chebtech import Chebtech2


def field(complex_values=False):
    ranks = (2, 3, 4)
    factors = [[Chebtech2.from_coeffs(jnp.eye(n)[k]) for k in range(n)] for n in ranks]
    core = jnp.arange(1., 25.).reshape(ranks)/100
    if complex_values:
        core = core+1j*core[::-1, ::-1, ::-1]
        factors = [[Chebtech2.from_coeffs(t.coeffs*(1+.1j*(k+1)))
                    for k, t in enumerate(group)] for group in factors]
    return Chebfun3(*factors, core, (-2., 3., .5, 2., -4., -1.))


def points(axes=(0, 1, 2), singleton=False):
    x = jnp.asarray([-.8]) if singleton else jnp.asarray([-1.7, -.3, 2.4])
    y = jnp.asarray([.51, .8, 1.1, 1.8])
    z = jnp.asarray([-3.8, -3.3, -2.7, -2.2, -1.1])
    permutation = tuple(axes.index(k) for k in range(3))
    return tuple(jnp.transpose(v, permutation) for v in jnp.meshgrid(x, y, z, indexing='ij'))


@pytest.mark.parametrize('axes', [(0, 1, 2), (2, 0, 1), (1, 2, 0), (1, 0, 2)])
@pytest.mark.parametrize('complex_values', [False, True])
def test_native_grid_patterns_against_generic(monkeypatch, axes, complex_values):
    f = field(complex_values)
    xyz = points(axes)
    expected = f._evaluate_numeric(*xyz)
    monkeypatch.setattr(Chebfun3, '_evaluate_numeric', lambda *a: pytest.fail('generic grid allocation'))
    actual = f(*xyz)
    assert actual.shape == xyz[0].shape
    assert jnp.max(jnp.abs(actual-expected)) < 500*jnp.finfo(jnp.float64).eps*jnp.maximum(1., jnp.max(jnp.abs(expected)))


def test_singleton_first_axis(monkeypatch):
    f = field(True)
    xyz = points(singleton=True)
    expected = f._evaluate_numeric(*xyz)
    monkeypatch.setattr(Chebfun3, '_evaluate_numeric', lambda *a: pytest.fail('generic grid allocation'))
    actual = f(*xyz)
    assert actual.shape == (1, 4, 5)
    assert jnp.max(jnp.abs(actual-expected)) < 500*jnp.finfo(jnp.float64).eps


def test_near_grid_requires_generic():
    f = field(True)
    x, y, z = points()
    x = x.at[1, 1, 1].add(1e-10)
    assert source_tensor_grid(f, x, y, z) is NotImplemented
    assert jnp.array_equal(f(x, y, z), f._evaluate_numeric(x, y, z))


@pytest.mark.parametrize('kind', ['vector', 'matrix'])
def test_other_dimensions_retain_generic(kind):
    f = field(True)
    x = jnp.asarray([-.5, .1, .4])
    if kind == 'matrix':
        x = jnp.stack((x, x+.1))
    y, z = 1.+0*x, -2.+0*x
    assert source_tensor_grid(f, x, y, z) is NotImplemented
    assert jnp.array_equal(f(x, y, z), f._evaluate_numeric(x, y, z))


def test_traced_grid_and_core_gradients_retain_generic():
    f = field()
    x, y, z = points()
    assert jnp.array_equal(jax.jit(lambda t: f(t, y, z))(x),
                           jax.jit(lambda t: f._evaluate_numeric(t, y, z))(x))
    got = jax.grad(lambda shift: jnp.sum(f(x+shift, y, z)))(0.)
    expected = jax.grad(lambda shift: jnp.sum(f._evaluate_numeric(x+shift, y, z)))(0.)
    assert jnp.array_equal(got, expected)
    public = jax.grad(lambda core: jnp.sum(eqx.tree_at(lambda g: g.core, f, core)(x, y, z)))(f.core)
    reference = jax.grad(lambda core: jnp.sum(eqx.tree_at(lambda g: g.core, f, core)._evaluate_numeric(x, y, z)))(f.core)
    assert jnp.array_equal(public, reference)
