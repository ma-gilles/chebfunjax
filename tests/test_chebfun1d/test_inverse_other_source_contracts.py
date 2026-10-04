"""Focused controls for literal MATLAB bisection and Newton inverses.

Provenance
----------
MATLAB source : @chebfun/inv.m, fInverseBisection/fInverseNewton
Chebfun commit: 7574c77

These are additional algorithm controls; the unchanged original source test
battery remains in tests/test_matlab_port/chebfun/test_inv_matlab.py.
"""

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.inverse import _bisection, _newton

EPS = float(jnp.finfo(jnp.float64).eps)


@pytest.mark.parametrize("slope, offset", [(1.0, 0.0), (-1.0, 0.0)])
def test_bisection_collapses_at_source_residual_dead_band(slope, offset):
    """The source freezes an iterate when signed residual magnitude is < EPS."""
    target = slope * (0.5 + EPS / 2.0) + offset
    result = _bisection(lambda x: slope * x + offset,
                        jnp.asarray([target]), 0.0, 1.0)
    assert float(result[0]) == 0.5


def test_bisection_explicit_safety_cap_raises():
    with pytest.raises(RuntimeError, match="bisection still open"):
        result = _bisection(lambda x: x, jnp.asarray([0.3]), 0.0, 1.0,
                            max_iterations=1)
        result.block_until_ready()


def test_newton_returns_source_raw_iterate_outside_bracket():
    """MATLAB applies the Newton step without an interval guard or rescue."""
    result = _newton(lambda x: x, lambda x: jnp.ones_like(x),
                     jnp.asarray([2.0]), 0.0, 1.0, EPS)
    assert float(result[0]) == 2.0


def test_newton_returns_the_eleventh_source_iterate_without_brent_rescue():
    """A double root gives linear Newton convergence and exposes the 11-step cap."""
    root = 0.9
    result = _newton(lambda x: (x - root) ** 2,
                     lambda x: 2.0 * (x - root),
                     jnp.asarray([0.0]), 0.0, 1.0, EPS)
    expected = root * (1.0 - 2.0**-11)
    assert abs(float(result[0]) - expected) <= 20.0 * EPS
    assert abs(float(result[0]) - root) > 1.0e-4
