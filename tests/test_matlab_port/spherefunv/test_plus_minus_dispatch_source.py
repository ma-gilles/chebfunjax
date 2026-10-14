"""Focused controls for native @spherefunv/plus.m and minus.m branches."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.spherefun.spherefunv import Spherefunv

jax.config.update("jax_enable_x64", True)
TOL = 1e3 * ChebfunPref().cheb2Prefs.chebfun2eps


def _vnorm(field):
    return field.norm()


def _field():
    a = Spherefun.from_function(
        lambda lam, th: jnp.sin(th) * jnp.cos(lam))
    b = Spherefun.from_function(
        lambda lam, th: jnp.sin(th) * jnp.sin(lam))
    c = Spherefun.from_function(lambda lam, th: jnp.cos(th))
    return Spherefunv(a, b, c), (a, b, c)


def _assert_components_close(actual, expected):
    component_errors = [(left - right).norm()
                        for left, right in zip(actual.components,
                                               expected.components)]
    assert jnp.sqrt(sum(error ** 2 for error in component_errors)) < TOL


def test_native_plus_minus_operand_branches():
    F, (a, b, c) = _field()
    G = F

    # Numeric scalar, including reflected scalar addition.
    scalar = F + 0.375
    _assert_components_close(scalar, Spherefunv(a + 0.375, b + 0.375, c + 0.375))
    _assert_components_close(0.375 + F, scalar)

    # MATLAB numel-three arrays are consumed by linear indexing.
    numeric = jnp.asarray([[1.0], [2.0], [3.0]])
    array_sum = F.plus(numeric)
    _assert_components_close(array_sum, Spherefunv(a + 1.0, b + 2.0, c + 3.0))
    array_difference = numeric - F
    _assert_components_close(array_difference, Spherefunv(1.0 - a, 2.0 - b, 3.0 - c))
    list_sum = F.plus([1.0, 2.0, 3.0])
    _assert_components_close(list_sum, Spherefunv(a + 1.0, b + 2.0, c + 3.0))
    list_difference = F - [1.0, 2.0, 3.0]
    _assert_components_close(list_difference,
                             Spherefunv(a - 1.0, b - 2.0, c - 3.0))
    reflected_list_difference = [1.0, 2.0, 3.0] - F
    _assert_components_close(reflected_list_difference,
                             Spherefunv(1.0 - a, 2.0 - b, 3.0 - c))

    # A scalar Spherefun is broadcast to each component.
    offset = Spherefun.from_function(lambda lam, th: 0.25 + 0.0 * th)
    _assert_components_close(F + offset,
                             Spherefunv(a + offset, b + offset, c + offset))

    # Vector addition/subtraction preserve row orientation; mixed orientation
    # follows the native plus transpose mismatch branch.
    row_f, row_g = F.transpose(), G.transpose()
    row_sum = row_f + row_g
    row_difference = row_f - row_g
    assert row_sum.size == (jnp.inf, jnp.inf, 3)
    assert row_difference.size == (jnp.inf, jnp.inf, 3)
    _assert_components_close(row_sum, Spherefunv(a + a, b + b, c + c).transpose())
    _assert_components_close(row_difference,
                             Spherefunv(a - a, b - b, c - c).transpose())
    with pytest.raises(ValueError, match="SPHEREFUN:SPHEREFUNV:plus:transposed"):
        _ = F + row_g
    with pytest.raises(ValueError, match="SPHEREFUN:SPHEREFUNV:plus:doubleSize"):
        _ = F + jnp.asarray([1.0, 2.0])
