"""Focused controls for MATLAB's complex-power branch-cut handling.

These checks exercise the source rule that noninteger powers of a complex
Chebfun split at roots of its imaginary part. At a negative-real-axis cut,
MATLAB maps the stored breakpoint value with complex principal power even
though the two one-sided limits differ.

Provenance
----------
MATLAB sources : @chebfun/power.m (columnPower), @chebfun/getRootsForBreaks.m,
                 @chebfun/addBreaks.m, @chebfun/feval.m
Chebfun commit: 7574c77
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

EPS = float(np.finfo(np.float64).eps)


def _has_break(fun, point):
    return any(abs(float(t) - point) <= 20 * EPS for t in fun.domain.breakpoints)


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
def test_complex_power_splits_negative_real_branch_cut_and_maps_pointvalue(Tech):
    f = Chebfun(funs=[_Piece(tech=Tech.from_coeffs(jnp.asarray([-1.0, 1j])),
                            interval=(-1.0, 1.0))],
                domain=Domain((-1.0, 1.0)))
    result = f**0.5
    assert _has_break(result, 0.0)
    points = np.asarray([-1e-7, 1e-7])
    np.testing.assert_allclose(result(points), np.sqrt(-1.0 + 1j * points),
                               rtol=0.0, atol=100 * EPS)
    np.testing.assert_allclose(result(0.0, side="left"), -1j,
                               rtol=0.0, atol=100 * EPS)
    np.testing.assert_allclose(result(0.0, side="right"), 1j,
                               rtol=0.0, atol=100 * EPS)
    # Source @chebfun/power.m maps g.pointValues = f.pointValues.^b, so the
    # exact cut value is the principal (+pi angle) value, not the side mean.
    np.testing.assert_allclose(result(0.0), 1j, rtol=0.0, atol=100 * EPS)


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
def test_complex_power_also_splits_positive_real_imaginary_roots(Tech):
    f = Chebfun(funs=[_Piece(tech=Tech.from_coeffs(jnp.asarray([1.0, 1j])),
                            interval=(-1.0, 1.0))],
                domain=Domain((-1.0, 1.0)))
    result = f**0.5
    assert _has_break(result, 0.0)
    np.testing.assert_allclose(result(0.0), 1.0, rtol=0.0, atol=100 * EPS)
    points = np.asarray([-1e-7, 1e-7])
    np.testing.assert_allclose(result(points), np.sqrt(1.0 + 1j * points),
                               rtol=0.0, atol=100 * EPS)


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
@pytest.mark.parametrize("transposed", [False, True])
def test_complex_power_preserves_old_stored_point_values_when_rebreaking(Tech, transposed):
    f = Chebfun(funs=[_Piece(tech=Tech.from_coeffs(jnp.asarray([-1.0, 1j])),
                            interval=(-1.0, 1.0))],
                domain=Domain((-1.0, 1.0)))
    f = f._with_breakpoints((-1.0, 0.5, 1.0))
    f = f.set_point_values(jnp.asarray([4.0, 9.0, 16.0], dtype=jnp.complex128))
    f = Chebfun._as_transposed(f, transposed)
    result = f**0.5
    assert result.is_transposed == transposed
    assert _has_break(result, 0.0)
    np.testing.assert_allclose(result(jnp.asarray([-1.0, 0.5, 1.0])),
                               [2.0, 3.0, 4.0], rtol=0.0, atol=100 * EPS)
    expected_limit = np.sqrt(-1.0 + 0.5j)
    np.testing.assert_allclose(result(0.5, side="left"), expected_limit,
                               rtol=0.0, atol=100 * EPS)
    np.testing.assert_allclose(result(0.5, side="right"), expected_limit,
                               rtol=0.0, atol=100 * EPS)
