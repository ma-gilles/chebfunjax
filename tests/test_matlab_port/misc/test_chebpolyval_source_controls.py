"""Source shape/complex controls for chebpolyval.

Provenance
----------
MATLAB source: chebpolyval.m, @chebtech/clenshaw.m, @chebtech/feval.m.
Chebfun commit: 7574c77.
Manufactured low-degree controls use explicit 100eps regression bounds.
"""

import jax
import jax.numpy as jnp

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.utils.polynomials import chebpolyval


def test_complex_numeric():
    p = jnp.array([1 + 2j, 3 + 0j])
    x = jnp.array([0.25 + 0.5j])
    y = chebpolyval(p, x)
    assert jnp.iscomplexobj(y) and jnp.array_equal(y, p[0] * x + p[1])

def test_matrix_numeric():
    p = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    x = jnp.array([[0.25, 0.5], [0.75, 1.0]])
    y = chebpolyval(p, x)
    expected = jnp.concatenate([x + 3, 2 * x + 4], axis=1)
    assert y.shape == (2, 4), y.shape
    assert jnp.array_equal(y, expected)

def test_chebfun_matrix():
    x = cj.chebfun('x')
    y = chebpolyval(jnp.array([[1.0, 2.0], [3.0, 4.0]]), x)
    assert isinstance(y, Chebfun) and y.n_columns == 2, type(y)
    assert (y - cj.Chebfun.horzcat([x + 3, 2 * x + 4])).norm(jnp.inf) < 100 * jnp.finfo(jnp.float64).eps

def test_scalar_guard():
    x = cj.chebfun(lambda t: jnp.stack((t, t * t), axis=-1))
    try:
        chebpolyval(jnp.array([1.0, 2.0]), x)
    except ValueError as e:
        assert 'CHEBFUN:chebpolyval:columnInput' in str(e)
        return
    raise AssertionError('missing scalar-X guard')

def test_transpose():
    x = cj.chebfun('x').transpose()
    y = chebpolyval(jnp.array([[1.0, 2.0], [3.0, 4.0]]), x)
    assert isinstance(y, Chebfun) and y.is_transposed and (y.n_columns == 2)

def test_numeric_shapes():
    p = jnp.array([[2.0, 3.0]])
    assert jnp.array_equal(chebpolyval(p, jnp.array([0.25, 0.5])), jnp.array([[2.0, 3.0], [2.0, 3.0]]))
    assert chebpolyval(p, 0.25).shape == (1, 2)
    assert chebpolyval(jnp.array([1.0, 2.0]), jnp.ones((2, 3, 4))).shape == (2, 3, 4)
    try:
        chebpolyval(jnp.array([[1.0, 2.0], [3.0, 4.0]]), jnp.ones((2, 3, 4)))
    except ValueError as e:
        assert 'evalArrayAtNDArray' in str(e)
    else:
        raise AssertionError('missing ND array guard')

def test_numeric_jit():
    fun = jax.jit(chebpolyval)
    p = jnp.array([[1 + 2j, 2 - 1j], [3 + 0j, 4 + 1j]])
    x = jnp.array([[0.25 + 0.5j, 0.5 - 0.25j]])
    assert jnp.array_equal(fun(p, x), jnp.concatenate([p[0, 0] * x + p[1, 0], p[0, 1] * x + p[1, 1]], axis=1))

def test_nonlinear_chebfun():
    t = cj.chebfun('x')
    x = t * t + 0.25 * t
    p = jnp.array([1 + 1j, 2 - 1j, 3 + 0j])
    y = chebpolyval(p, x)
    expected = p[0] * (2 * x * x - 1) + p[1] * x + p[2]
    assert (y - expected).norm(jnp.inf) < 100 * jnp.finfo(jnp.float64).eps

def test_complex_chebfun():
    t = cj.chebfun('x')
    x = t + 1j * 0.25
    y = chebpolyval(jnp.array([1 + 2j, 3 + 0j]), x)
    assert (y - ((1 + 2j) * x + 3)).norm(jnp.inf) < 100 * jnp.finfo(jnp.float64).eps

