"""Native CDR column evaluation preserves each genuine polynomial.

Provenance
----------
MATLAB source: @separableApprox/feval.m, @chebtech/clenshaw.m
Chebfun commit: 7574c77
These are authored operation-preservation controls, not original predicates.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun2d.separable_approx import SeparableApprox
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


def _bits_equal(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


def _coefficients(n, rank):
    k = jnp.arange(n)[:, None]
    r = jnp.arange(rank)[None, :]
    return ((3*k+5*r) % 17-8) * jnp.exp2(-k.astype(jnp.float64)-4)


@pytest.mark.parametrize('n,rank', [(1, 3), (2, 3), (17, 33), (202, 33)])
@pytest.mark.parametrize('coordinate_kind', ['real', 'float32', 'complex'])
def test_batched_columns_preserve_individual_values(n, rank, coordinate_kind):
    coeffs = _coefficients(n, rank)
    funs = [Chebtech2.from_coeffs(coeffs[:, j]) for j in range(rank)]
    t = jnp.array([-1., -.75, -.125, 0., .5, 1.])
    if coordinate_kind == 'float32':
        t = t.astype(jnp.float32)
    elif coordinate_kind == 'complex':
        t = t + .125j
    expected = jnp.stack([fn(t) for fn in funs], axis=-1)
    _bits_equal(SeparableApprox._slice_vals_jax(funs, t), expected)
    _bits_equal(jax.jit(lambda t: SeparableApprox._slice_vals_jax(funs, t))(t), expected)


def test_column_batch_retains_coefficient_and_coordinate_derivatives():
    coeffs = _coefficients(7, 3)
    t = jnp.array([-.75, -.125, .5])
    def actual(c, t):
        funs = [Chebtech2.from_coeffs(c[:, j]) for j in range(c.shape[1])]
        return SeparableApprox._slice_vals_jax(funs, t)
    def expected(c, t):
        return jnp.stack([Chebtech2.from_coeffs(c[:, j])(t)
                          for j in range(c.shape[1])], axis=-1)
    tangents = (jnp.full_like(coeffs, .125), jnp.full_like(t, -.25))
    av, ad = jax.jvp(actual, (coeffs, t), tangents)
    ev, ed = jax.jvp(expected, (coeffs, t), tangents)
    _bits_equal(av, ev)
    np.testing.assert_allclose(ad, ed, rtol=4e-15, atol=4e-15)


@pytest.mark.parametrize('kind', ['unequal', 'complex', 'c1'])
def test_other_representations_keep_individual_evaluation(kind):
    coeffs = _coefficients(7, 3)
    funs = [Chebtech2.from_coeffs(coeffs[:, j]) for j in range(3)]
    if kind == 'unequal':
        funs[1] = Chebtech2.from_coeffs(coeffs[:3, 1])
    elif kind == 'complex':
        funs[1] = Chebtech2.from_coeffs(coeffs[:, 1] * (1+.25j))
    else:
        funs[1] = Chebtech1.from_coeffs(coeffs[:, 1])
    t = jnp.array([-1., -.5, 0., .75, 1.])
    _bits_equal(SeparableApprox._slice_vals_jax(funs, t),
                jnp.stack([fn(t) for fn in funs], axis=-1))
