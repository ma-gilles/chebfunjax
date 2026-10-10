"""Pinned @chebfun/minus.m plus/uminus and explicit pointValues contract."""

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax import Chebfun
from chebfunjax.chebfun1d.chebfun import _Piece
from chebfunjax.domain import Domain


def fixture(*, complex_values=False, row=False, cut=1.0):
    values = jnp.array([2.0, 7.0, -3.0])
    coeffs = [jnp.array([1.0, 0.25]), jnp.array([4.0, -0.5])]
    if complex_values:
        values = values + 1j * jnp.array([3.0, -2.0, 1.0])
        coeffs = [a * (1 + 2j) for a in coeffs]
    domain = Domain((0.0, cut, 2.0))
    f = Chebfun(
        funs=[
            _Piece.from_coeffs(c, a, b)
            for c, a, b in zip(coeffs, domain.breakpoints[:-1], domain.breakpoints[1:])
        ],
        domain=domain,
    )
    object.__setattr__(f, "_point_values", values)
    return f.T if row else f


@pytest.mark.parametrize("row", [False, True])
@pytest.mark.parametrize("complex_values", [False, True])
@pytest.mark.parametrize("operation", ["negative", "minus_scalar", "reflected"])
def test_source_stored_values_and_interior(row, complex_values, operation):
    f = fixture(row=row, complex_values=complex_values)
    points = jnp.array([0.0, 0.25, 1.0, 1.5, 2.0])
    before = np.asarray(f(points))
    stored = np.asarray(f._point_values).copy()
    scalar = 3.0 - 2j if complex_values else 3.0
    if operation == "negative":
        g = -f
        expected = -before
    elif operation == "minus_scalar":
        g = f - scalar
        expected = before + (-scalar)
    else:
        g = scalar - f
        expected = scalar + (-before)
    np.testing.assert_array_equal(np.asarray(g(points)), expected)
    assert g.is_transposed == row
    np.testing.assert_array_equal(np.asarray(f._point_values), stored)
    assert g._point_values is not None


@pytest.mark.parametrize("row", [False, True])
def test_source_two_functions_union_and_points(row):
    f = fixture(row=row)
    g = fixture(complex_values=True, row=row, cut=0.75)
    points = jnp.array([0.0, 0.25, 0.75, 1.0, 1.5, 2.0])
    expected = np.asarray(f(points)) + (-np.asarray(g(points)))
    actual = f - g
    np.testing.assert_allclose(np.asarray(actual(points)), expected, rtol=0, atol=4e-15)
    assert actual.is_transposed == row
    assert 0.75 in actual.domain.breakpoints and 1.0 in actual.domain.breakpoints


def test_native_orientation_error_and_empty():
    f = fixture()
    with pytest.raises(ValueError, match="matdim"):
        _ = f - f.T
    assert (f - Chebfun.empty()).isempty()
    assert (Chebfun.empty() - f).isempty()


def test_array_columns_and_delta_negation():
    f = fixture()
    object.__setattr__(f, "deltas", ((0.5, 2.0, 0),))
    g = f - jnp.array([1.0, 2.0])
    expected = np.asarray(f(jnp.array([0.0, 1.0, 2.0])))[:, None] - np.array([1.0, 2.0])
    np.testing.assert_array_equal(np.asarray(g(jnp.array([0.0, 1.0, 2.0]))), expected)
    assert tuple((*d, 0) if len(d) == 2 else d for d in (-f).deltas) == ((0.5, -2.0, 0),)
    assert tuple((*d, 0) if len(d) == 2 else d for d in (5 - f).deltas) == ((0.5, -2.0, 0),)
