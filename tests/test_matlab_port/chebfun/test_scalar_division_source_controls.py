"""Scalar division metadata and literal source rdivide assertions3/4.

Provenance
----------
MATLAB source: @chebfun/rdivide.m, @chebfun/mrdivide.m,
    @chebfun/times.m, @deltafun/rdivide.m, tests/chebfun/test_rdivide.m
Chebfun commit:7574c77
Metadata controls use exact representable values; diagnostic JIT/AD bounds
are explicit. The two exp assertions retain source10*vscale*eps bounds.
NumPy seeded probes are a Python reproducibility adapter, not MATLAB RNG parity.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain
from chebfunjax.tech.chebtech import Chebtech2

EPS = np.finfo(float).eps


def _constant_with_point_values(row=False, deltas=()):
    tech = Chebtech2.from_coeffs(jnp.array([2.0]))
    f = Chebfun(funs=[_Piece(tech=tech, interval=(-1.0, 1.0))],
                domain=Domain((-1.0, 1.0)), deltas=deltas)
    f = f.set_point_values(jnp.array([4.0, 8.0]))
    return f.T if row else f


@pytest.mark.parametrize('row', [False, True])
@pytest.mark.parametrize('divisor', [2.0, 1.0 + 2.0j])
def test_scalar_division_preserves_orientation_and_explicit_point_values(row, divisor):
    f = _constant_with_point_values(row)
    g = f / divisor
    assert g.is_transposed == row
    assert getattr(g, '_point_values', None) is not None
    np.testing.assert_array_equal(g.point_values, np.array([4.0, 8.0]) / divisor)
    np.testing.assert_allclose(np.asarray(g(jnp.array([0.0]))).ravel(),
                               [2.0 / divisor], rtol=0, atol=4 * EPS)
    np.testing.assert_array_equal(f.point_values, [4.0, 8.0])


@pytest.mark.parametrize('row', [False, True])
def test_reciprocal_preserves_denominator_orientation_and_explicit_point_values(row):
    f = _constant_with_point_values(row)
    g = 2.0 / f
    assert g.is_transposed == row
    assert getattr(g, '_point_values', None) is not None
    np.testing.assert_array_equal(g.point_values, [0.5, 0.25])
    np.testing.assert_allclose(np.asarray(g(jnp.array([0.0]))).ravel(),
                               [1.0], rtol=0, atol=4 * EPS)


@pytest.mark.parametrize('order', [0, 2])
@pytest.mark.parametrize('divisor', [2.0, 1.0 + 2.0j])
def test_scalar_division_scales_delta_magnitude_and_preserves_order(order, divisor):
    delta = (0.25, 2.0) if order == 0 else (0.25, 2.0, order)
    f = _constant_with_point_values(True, (delta,))
    g = f / divisor
    expected = (0.25, 2.0 / divisor) if order == 0 else (0.25, 2.0 / divisor, order)
    assert g.deltas == (expected,)
    assert g.is_transposed
    np.testing.assert_array_equal(f.point_values, [4.0, 8.0])
    assert f.deltas == (delta,)


@pytest.mark.parametrize('zero', [0.0, 0j, jnp.array(0.0)])
def test_scalar_zero_divisor_has_source_error_identifier(zero):
    f = _constant_with_point_values()
    with pytest.raises(ValueError, match='CHEBFUN:CHEBFUN:rdivide:columnRdivide:divisionByZero: Division by zero'):
        f / zero


@pytest.mark.parametrize('row', [False, True])
def test_zero_numerator_uses_source_zero_times_denominator_before_delta_rejection(row):
    f = _constant_with_point_values(row, ((0.25, 2.0),))
    g = 0.0 / f
    assert g.is_transposed == row
    assert g.deltas == ()
    np.testing.assert_array_equal(g.point_values, [0.0, 0.0])
    assert float(jnp.max(jnp.abs(g(jnp.array([0.0]))))) == 0.0


@pytest.mark.parametrize('numerator', [1.0, 1.0 + 2.0j])
def test_nonzero_scalar_divided_by_delta_denominator_has_source_rejection(numerator):
    f = _constant_with_point_values(False, ((0.25, 2.0),))
    with pytest.raises(ValueError, match='CHEBFUN:DELTAFUN:rdivide:rdivide: Division by delta functions is not defined'):
        numerator / f


def test_chebfun_scalar_division_uses_literal_reciprocal_multiplication():
    coefficients = jnp.array([0.1, 0.2, 0.3])
    f = Chebfun(funs=[_Piece(tech=Chebtech2.from_coeffs(coefficients),
                            interval=(-1.0, 1.0))], domain=Domain((-1.0, 1.0)))
    expected = np.asarray(coefficients) * (1.0 / 10.0)
    # The binary64 values distinguish the source multiplication from c/10.
    assert expected[0] != float(coefficients[0]) / 10.0
    np.testing.assert_array_equal((f / 10.0).funs[0].tech.coeffs, expected)


def test_scalar_division_remains_jittable_and_differentiable():
    f = cj.chebfun(lambda x: 2.0 + x)
    x = jnp.array([-0.75, 0.0, 0.75])
    actual = jax.jit(lambda scale: (f / scale)(x))(jnp.array(2.0))
    np.testing.assert_allclose(actual, np.asarray(2.0 + x) / 2,
                               rtol=0, atol=8 * EPS)
    derivative = jax.grad(lambda scale: (f / scale)(jnp.array(0.0)))(jnp.array(2.0))
    assert abs(float(derivative) + 0.5) < 8 * EPS


@pytest.mark.parametrize('source_assertion', [3, 4])
def test_literal_source_scalar_exp_assertions_original_bounds(source_assertion):
    x = jnp.asarray(2 * np.random.default_rng(6178).uniform(size=100) - 1)
    f = cj.chebfun(jnp.exp, domain=[-1.0, -0.5, 0.0, 0.5, 1.0])
    if source_assertion == 3:
        h = 1.0 / f
        exact = jnp.exp(-x)
    else:
        g = cj.chebfun(lambda t: jnp.exp(-t), domain=[-1.0, 1.0])
        h = f / g
        exact = jnp.exp(2 * x)
    assert float(jnp.max(jnp.abs(h(x) - exact))) < 10 * float(h.vscale) * EPS
