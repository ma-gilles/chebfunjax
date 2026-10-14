"""Focused controls for native @spherefunv/power.m dispatch branches."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.spherefun.spherefunv import Spherefunv

jax.config.update("jax_enable_x64", True)
TOL = 1e3 * ChebfunPref().cheb2Prefs.chebfun2eps


def _assert_vector_close(actual, expected):
    errors = [(left - right).norm()
              for left, right in zip(actual.components, expected.components)]
    assert jnp.sqrt(sum(error ** 2 for error in errors)) < TOL


def test_native_scalar_power_branches_and_orientation():
    f = Spherefun.from_function(lambda lam, th: jnp.cos(th))
    F = Spherefunv(f, f, f)

    # Native original pass(9) is unchanged in test_times_divide_matlab.py;
    # these controls exercise scalar-left/right and scalar-shaped arrays.
    right = F.power(2)
    square = Spherefun.from_function(lambda lam, th: jnp.cos(th) ** 2)
    reference_right = Spherefunv(square, square, square)
    _assert_vector_close(right, reference_right)
    _assert_vector_close(F ** jnp.asarray([[2.0]]), right)
    _assert_vector_close(F.power([2.0]), right)

    left = Spherefunv.power(2.0, F)
    scalar_left_expected = Spherefun.from_function(
        lambda lam, th: jnp.exp(jnp.log(2.0) * jnp.cos(th)))
    reference_left = Spherefunv(scalar_left_expected,
                                scalar_left_expected,
                                scalar_left_expected)
    _assert_vector_close(left, reference_left)
    _assert_vector_close(2.0 ** F, left)
    _assert_vector_close(Spherefunv.power(jnp.asarray([[2.0]]), F), left)

    row = F.transpose()
    row_right = row.power(2)
    row_left = 2.0 ** row
    assert row_right.size == row_left.size == (jnp.inf, jnp.inf, 3)
    _assert_vector_close(row_right, reference_right.transpose())
    _assert_vector_close(row_left, reference_left.transpose())

    assert Spherefunv.empty().power(2).isempty()
    with pytest.raises(ValueError, match="SPHEREFUN:SPHEREFUNV:power:double"):
        _ = F.power(jnp.asarray([2.0, 3.0]))
    with pytest.raises(ValueError, match="SPHEREFUN:SPHEREFUNV:power:double"):
        _ = Spherefunv.power([2.0, 3.0], F)
    with pytest.raises(ValueError, match="SPHEREFUN:SPHEREFUNV:power:size"):
        _ = F.power(F)
    with pytest.raises(ValueError, match="SPHEREFUN:SPHEREFUNV:power:inputs"):
        _ = F.power(f)
