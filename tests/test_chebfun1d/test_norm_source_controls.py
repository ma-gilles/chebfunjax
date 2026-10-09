"""Analytic controls for source norm branches omitted from the original test.

Provenance
----------
MATLAB source : @chebfun/norm.m
Chebfun commit: 7574c77
"""
import jax
import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.linalg import Quasimatrix

EPS = jnp.finfo(jnp.float64).eps


def test_string_extrema_and_interior_zero():
    f = cj.chebfun(lambda x: x-.2)
    value, location = f.norm('-inf', return_location=True)
    assert abs(value) < 32*EPS and abs(location-.2) < 32*EPS
    value, location = f.norm('inf', return_location=True)
    assert abs(value-1.2) < 32*EPS and location == -1.


def test_array_generic_power_is_maximum_row_sum():
    f = cj.chebfun(lambda x: jnp.stack((1+x, 2+0*x), axis=-1), domain=[0., 1.])
    value, location = f.norm(3, return_location=True)
    assert abs(value-16**(1/3)) < 32*EPS and location == 1.
    assert abs(f.T.norm(3)-value) < 32*EPS


def test_quasimatrix_one_norm_and_tied_column_index():
    a = cj.chebfun(lambda x: 1+x, domain=[0., 1.])
    b = cj.chebfun(lambda x: 2+0*x, domain=[0., 1.])
    q = Quasimatrix([a, b, b], a.domain)
    value, column = q.norm(1, return_location=True)
    assert abs(value-2.) < 32*EPS and column == 2
    value, location = q.norm(-jnp.inf, return_location=True)
    assert abs(value-5.) < 32*EPS and location == 0.


def test_unknown_selector_precedes_second_output_validation():
    f = cj.chebfun(lambda x: 1+x)
    with pytest.raises(ValueError, match='CHEBFUN:CHEBFUN:norm:unknownNorm'):
        f.norm('bad', return_location=True)
    with pytest.raises(ValueError, match='CHEBFUN:CHEBFUN:norm:unknownNorm'):
        f.norm(1+2j)


def test_transposed_complex_scalar_norms():
    f = cj.chebfun(lambda x: 1+1j*x)
    assert abs(f.T.norm()-jnp.sqrt(8/3)) < 32*EPS
    assert abs(f.T.norm(4)-(2+4/3+2/5)**.25) < 32*EPS


def test_two_norm_jit_derivative():
    f = cj.chebfun(lambda x: x)
    evaluate = jax.jit(lambda a: (a*f).norm(2))
    expected = jnp.sqrt(2/3)
    assert abs(evaluate(2.)-2*expected) < 32*EPS
    assert abs(jax.grad(evaluate)(2.)-expected) < 32*EPS


def test_transposed_quasimatrix_columns_use_source_column_orientation():
    a = cj.chebfun(lambda x: 1+x, domain=[0., 1.])
    b = cj.chebfun(lambda x: 2+0*x, domain=[0., 1.])
    q = Quasimatrix([a.T, b.T], a.domain)
    assert abs(q.norm()-jnp.sqrt(19/3)) < 32*EPS
    value, column = q.norm(1, return_location=True)
    assert abs(value-2.) < 32*EPS and column == 2
    value, location = q.norm(jnp.inf, return_location=True)
    assert abs(value-4.) < 32*EPS and location == 1.
    assert abs(q.norm(3)-16**(1/3)) < 32*EPS


def test_unbounded_scalar_restriction_keeps_bounded_piece_protocol():
    f = cj.chebfun(lambda x: 1/x, domain=[1., jnp.inf])
    g = f.restrict(2., 4.).simplify()
    x = jnp.array([2., 2.5, 3., 4.])
    assert g.n_columns == 1
    assert jnp.max(jnp.abs(g(x)-1/x)) < 64*EPS
    assert abs(g.norm(1)-jnp.log(2.)) < 128*EPS


def test_unbounded_array_restriction_preserves_columns_and_integrals():
    f = cj.chebfun(lambda x: jnp.stack((jnp.exp(x), 1/x), axis=-1),
                  domain=[-jnp.inf, -1.])
    g = f.restrict(-4., -1.).simplify()
    x = jnp.array([-4., -2.5, -1.])
    assert g.n_columns == 2
    assert jnp.max(jnp.abs(g(x)-jnp.stack((jnp.exp(x), 1/x), axis=-1))) < 64*EPS
    expected = jnp.array([jnp.exp(-1.)-jnp.exp(-4.), -jnp.log(4.)])
    assert jnp.max(jnp.abs(g.sum()-expected)) < 128*EPS
    value, column = g.norm(1, return_location=True)
    assert abs(value-jnp.log(4.)) < 128*EPS and column == 2
