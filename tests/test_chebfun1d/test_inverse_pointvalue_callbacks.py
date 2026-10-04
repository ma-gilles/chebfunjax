"""Breakpoint values through eager, JIT and PyTree evaluation.

Provenance
----------
MATLAB source : @chebfun/feval.m, @chebfun/getValuesAtBreakpoints.m,
    tests/chebfun/test_feval.m, tests/chebfun/test_definePoint.m
Chebfun commit: 7574c77
"""

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece, chebfun
from chebfunjax.domain import Domain


def manual_jump():
    return Chebfun(funs=[
        _Piece.from_coeffs(jnp.array([1.0]), -1.0, 0.0),
        _Piece.from_coeffs(jnp.array([3.0]), 0.0, 1.0),
    ], domain=Domain((-1.0, 0.0, 1.0)))

def test_manual_jump_uses_source_mean_with_and_without_jit():
    f = manual_jump()
    points = jnp.array([-2.0, -1.0, -0.4, 0.0, 0.3, 1.0, 2.0])
    expected = [1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0]
    npt.assert_array_equal(f.point_values, [1.0, 2.0, 3.0])
    npt.assert_array_equal(f(points), expected)
    npt.assert_array_equal(jax.jit(lambda g, x: g(x))(f, points), expected)

@pytest.mark.parametrize("single_piece", [True, False])
@pytest.mark.parametrize("complex_values", [True, False])
def test_stored_values_survive_as_pytree_argument(single_piece, complex_values):
    breaks = (-1.0, 1.0) if single_piece else (-1.0, 0.0, 1.0)
    f = chebfun(lambda x: 2+x, domain=breaks)
    values = jnp.arange(len(breaks), dtype=jnp.float64)+10
    if complex_values:
        values = values+1j*(values+2)
    f = f.set_point_values(values)
    points = jnp.asarray(breaks)
    npt.assert_array_equal(jax.jit(lambda g, x: g(x))(f, points), values)
    npt.assert_array_equal(jax.jit(lambda x: f(x))(points), values)
    npt.assert_array_equal(jax.vmap(lambda x: f(x))(points), values)
    assert float(jax.jit(lambda g: g(0.25).real)(f)) == 2.25

def test_callable_constructor_retains_its_value_at_jump_and_side_limits():
    x = chebfun(lambda x: x)
    f = x.sign()
    points = jnp.array([-0.4, 0.0, 0.4])
    npt.assert_array_equal(jax.jit(lambda g, x: g(x))(f, points), [-1.0, 0.0, 1.0])
    npt.assert_array_equal(f(points, side="left"), [-1.0, -1.0, 1.0])
    npt.assert_array_equal(f(points, side="right"), [-1.0, 1.0, 1.0])

def test_point_value_leaf_has_its_own_derivative():
    f = manual_jump()
    # Point assignments are independent of the represented neighboring FUNs.
    derivative = jax.grad(lambda v: f.set_point_values(v)(0.0))(
        jnp.array([1.0, 7.0, 3.0]))
    npt.assert_array_equal(derivative, [0.0, 1.0, 0.0])
    assert float(jax.grad(lambda x: f(x))(0.25)) == 0.0
