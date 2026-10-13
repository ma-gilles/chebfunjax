"""Exact one-sided point replacement from @chebfun/feval.m, pin7574c77."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain


def _linear_jump():
    # x on [-1,0], x+1 on [0,1], represented without adaptive construction.
    return Chebfun(funs=[
        _Piece.from_coeffs(jnp.asarray([-.5, .5]), -1., 0.),
        _Piece.from_coeffs(jnp.asarray([1.5, .5]), 0., 1.),
    ], domain=Domain((-1., 0., 1.)))


@pytest.mark.parametrize("side,at_zero", [("left", 0.), ("right", 1.)])
@pytest.mark.parametrize("compiled", [False, True])
def test_sides_replace_only_exact_breakpoints(side, at_zero, compiled):
    f = _linear_jump()
    delta = 2.**-48
    x = jnp.asarray([[-delta, 0., delta]])
    expected = np.asarray([[-delta, at_zero, 1.+delta]])
    def evaluate(q):
        return f(q, side)
    if compiled:
        evaluate = jax.jit(evaluate)
    got = np.asarray(evaluate(x))
    assert got.shape == expected.shape
    assert np.max(np.abs(got-expected)) < 8*np.finfo(float).eps


@pytest.mark.parametrize("side", ["left", "right"])
def test_complex_offsets_are_not_breakpoint_members(side):
    f = _linear_jump()
    x = np.asarray([-1e-8+1e-8j, 1e-8j, 1e-8+1e-8j])
    expected = x + np.asarray([0., 1., 1.])
    for evaluate in (lambda q: f(q, side), jax.jit(lambda q: f(q, side))):
        assert np.max(np.abs(np.asarray(evaluate(jnp.asarray(x)))-expected)) < 8*np.finfo(float).eps


@pytest.mark.parametrize("side,expected", [
    ("left", [-9., 0., 2.]), ("right", [-1., 1., 9.]),
])
def test_source_endpoint_point_value_ownership(side, expected):
    # Source columnFeval uses [pointValues(1); rvals] for left limits,
    # and [lvals; pointValues(end)] for right limits.
    f = _linear_jump().set_point_values(jnp.asarray([-9., 7., 9.]))
    x = jnp.asarray([-1., 0., 1.])
    for evaluate in (lambda q: f(q, side), jax.jit(lambda q: f(q, side))):
        np.testing.assert_array_equal(np.asarray(evaluate(x)), expected)
