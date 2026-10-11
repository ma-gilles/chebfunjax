"""Coefficient metadata and resize controls.

Provenance
----------
MATLAB source: @ballfun/constructor.m, @ballfun/coeffs3.m.
Chebfun commit: 7574c77
"""
import jax
import jax.numpy as jnp
import pytest

from chebfunjax.ballfun.ballfun import Ballfun
from chebfunjax.tech.chebtech import Chebtech2
from chebfunjax.tech.trigtech import Trigtech


@pytest.mark.parametrize("value, expected", [(1., True), (1.j, False)])
def test_inferred_constant(value, expected):
    f = Ballfun.from_coeffs(jnp.full((1, 1, 1), value))
    assert f.is_real is expected
    assert jnp.array_equal(f.coeffs, jnp.full((1, 1, 1), value))


def test_explicit_realness():
    f = Ballfun.from_coeffs(jnp.ones((1, 1, 1)), is_real=False)
    assert f.is_real is False


def test_traced_default_preserves_complex():
    def construct(c):
        f = Ballfun.from_coeffs(c, fixed_size=(3, 4, 5))
        assert f.is_real is False
        return f.coeffs
    c = jnp.full((1, 1, 1), 1+2j)
    out = jax.jit(construct)(c)
    assert out.shape == (3, 4, 5)
    assert out[0, 2, 2] == 1+2j
    assert jnp.count_nonzero(out) == 1


@pytest.mark.parametrize("shape", [(2, 3, 4), (7, 8, 9)])
def test_coeffs3_source_axis_order(shape):
    c = jnp.arange(5*6*7).reshape(5, 6, 7) * (1+2j)
    f = Ballfun(coeffs=c, is_real=False, domain=(0., 1., -jnp.pi, jnp.pi, 0., jnp.pi))
    m, n, p = shape
    g = jnp.stack([Chebtech2.alias(Trigtech.alias(c[:, :, k].T, n).T, m)
                   for k in range(c.shape[2])], axis=2)
    expected = jnp.stack([Trigtech.alias(g[i].T, p).T for i in range(m)])
    assert jnp.array_equal(f.coeffs3(*shape), expected)


def test_scalar_callback():
    def scalar(x,y,z):
        assert x.ndim == y.ndim == z.ndim == 0
        return x*y
    f=Ballfun.from_function(scalar,vectorize=True)
    exact=Ballfun.from_function(lambda x,y,z:x*y)
    assert (f-exact).norm() < jnp.finfo(jnp.float64).eps
