# uses-numpy: analytic host reference assertions; numerical bounds unchanged.
"""Literal sphere sample grid and alias controls, Chebfun7574c77.

Provenance: @spherefun/sample.m, @trigtech/alias.m. Expected values are
analytic trigonometric values on the literal source grids; no candidate oracle.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.tech.trigtech import Trigtech


def field(nc=9, nr=10, complex_factors=False):
    # C(theta)=2+cos(3theta); R(lambda)=1+sin(2lambda)/4.
    c = jnp.zeros(nc, dtype=jnp.complex128).at[nc//2].set(2.)
    c = c.at[nc//2-3].set(.5).at[nc//2+3].set(.5)
    r = jnp.zeros(nr, dtype=jnp.complex128).at[nr//2].set(1.)
    r = r.at[nr//2-2].set(.125j).at[nr//2+2].set(-.125j)
    if complex_factors:
        c = c.at[nc//2].add(3.j)
        r = r.at[nr//2].add(-2.j)
    return Spherefun(cols=[Trigtech.from_coeffs(c, is_real=not complex_factors)],
                     rows=[Trigtech.from_coeffs(r, is_real=not complex_factors)],
                     pivots=jnp.asarray([2.]), idx_plus=(0,), idx_minus=())


@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("m,n", [(1, 1), (3, 2), (4, 3), (5, 4), (12, 8)])
@pytest.mark.parametrize("complex_factors", [False, True])
def test_literal_grid_alias_and_real_factors(disabled, m, n, complex_factors):
    with jax.disable_jit(disabled):
        f = field(complex_factors=complex_factors)
        values = f.sample(m, n)
        c, diagonal, r = f.sample_cdr(m, n)
    lam = -np.pi + 2*np.pi*np.arange(m)/m
    theta = np.asarray([-np.pi]) if n == 1 else np.linspace(0., np.pi, n)
    cv = 2+np.cos(3*theta)
    rv = 1+np.sin(2*lam)/4
    expected = np.outer(cv/2, rv)
    np.testing.assert_allclose(values, expected, rtol=0., atol=100*np.finfo(float).eps)
    np.testing.assert_allclose(c[:,0], cv, rtol=0., atol=50*np.finfo(float).eps)
    np.testing.assert_allclose(r[:,0], rv, rtol=0., atol=50*np.finfo(float).eps)
    np.testing.assert_array_equal(diagonal, [[.5]])
    np.testing.assert_array_equal(values, (c@diagonal)@r.T)


def test_default_dimensions_use_rows_then_columns():
    f = field()
    assert f.sample().shape == (9, 10)
    c, diagonal, r = f.sample_cdr()
    assert c.shape == (9, 1) and r.shape == (10, 1)
    np.testing.assert_array_equal(f.sample(), (c@diagonal)@r.T)


@pytest.mark.parametrize("dimensions", [(3,), (0, 3), (3, 0), (-2, 4), (4, -2)])
def test_source_dimension_errors(dimensions):
    with pytest.raises(ValueError, match="sample:inputs"):
        field().sample(*dimensions)


def test_empty_precedes_dimension_validation():
    assert Spherefun.empty().sample(-1, 0).shape == (0, 0)
    with pytest.raises(ValueError, match="sample:outputs"):
        Spherefun.empty().sample_cdr()
