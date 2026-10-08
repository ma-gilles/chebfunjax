"""Independent analytic orientation, endpoint, and empty mtimes controls."""

import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.linalg import Quasimatrix


def test_complex_bilinear_and_sesquilinear_products():
    f = cj.chebfun(lambda x: 1+1j*x)
    assert jnp.abs((f.T @ f)[0, 0]-4/3) < 2e-15
    assert jnp.abs((f.H @ f)[0, 0]-8/3) < 2e-15
    x, y = jnp.array([-.7, .2]), jnp.array([.1, .6])
    assert jnp.max(jnp.abs((f @ f.T)(x, y)-(1+1j*y)*(1+1j*x))) < 2e-15
    assert jnp.max(jnp.abs((f @ f.H)(x, y)-(1+1j*y)*(1-1j*x))) < 2e-15


def test_numeric_matrix_preserves_isolated_point_values():
    f = cj.chebfun(lambda x: jnp.stack([x, x*x], axis=-1))
    object.__setattr__(f, '_point_values', jnp.array([[3., 4.], [5., 6.]]))
    a = jnp.array([[2., 0.], [0., 3.]])
    h = f @ a
    assert jnp.array_equal(h(jnp.array([-1., 1.])), jnp.array([[6., 12.], [10., 18.]]))


def test_zero_and_empty_products():
    f = cj.chebfun(lambda x: 1+1j*x)
    for h in (f @ 0., 0. @ f, f.T @ 0., 0. @ f.T):
        assert jnp.all(h(jnp.array([-.3, .8])) == 0)
    for h in (f @ [], [] @ f, cj.chebfun() @ f, f @ cj.chebfun()):
        assert h.isempty()


def test_quasimatrix_row_evaluation_and_complex_numeric_left():
    f = cj.chebfun(lambda x: 1+1j*x)
    q = Quasimatrix([f, f*2], f.domain)
    a = jnp.array([[1j, 2.], [-1., .5j]])
    x = jnp.array([-.5, .3, .9])
    assert jnp.max(jnp.abs((a @ q.T)(x)-a @ q(x).T)) < 3e-15


def test_inner_product_requires_matching_domains():
    f = cj.chebfun(lambda x: x, domain=(-1, 1))
    g = cj.chebfun(lambda x: x, domain=(-2, 1))
    with pytest.raises(ValueError):
        f.T @ g
