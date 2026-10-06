"""Literal join bounds, orientation, mapping and interval accumulation.

The probe grids are deterministic Python adapters, not MATLAB's RNG stream.
Original MATLAB assertion bounds are retained for covered source cases.
Quasimatrix container equivalence is not established by these array controls.

Provenance
----------
MATLAB source : @chebfun/join.m, tests/chebfun/test_join.m
Chebfun commit: 7574c77
"""
import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.fun.unbndfun import Unbndfun

EPS = float(jnp.finfo(jnp.float64).eps)


@pytest.mark.parametrize("array", [False, True])
@pytest.mark.parametrize("row", [False, True])
def test_source_scalar_array_and_row_bounds(array, row):
    # Source pass(1), pass(2), pass(4), all at 10*vscale*eps.
    op = (lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1)) if array else jnp.sin
    f = cj.chebfun(op, domain=(-1., -.5, 0.))
    g = cj.chebfun(op, domain=(0., .5, 1.))
    if row:
        f, g = f.T, g.T
    h = f.join(g)
    x = jnp.linspace(-.999, .999, 1000)
    exact = op(x)
    if array and row:
        exact = exact.T
    assert h.is_transposed == row
    assert h.domain.breakpoints == (-1., -.5, 0., .5, 1.)
    assert float(jnp.max(jnp.abs(h(x)-exact))) < 10*h.vscale*EPS


def test_source_translated_join_bound():
    # Source pass(6), including its exact domain assertion.
    f = cj.chebfun(jnp.sin, domain=(-1., -.5, 0.))
    g = cj.chebfun(jnp.cos, domain=(1., 1.5, 2.))
    h = f.join(g)
    x = jnp.linspace(-.999, .999, 1000)
    exact = jnp.where(x <= 0, jnp.sin(x), jnp.cos(x+1))
    assert h.domain.breakpoints == (-1., -.5, 0., .5, 1.)
    assert float(jnp.max(jnp.abs(h(x)-exact))) < 10*h.vscale*EPS


def test_interval_lengths_accumulate_in_literal_source_order():
    f = cj.chebfun(lambda x: x, domain=(0., .2, 1.))
    g = cj.chebfun(lambda x: x, domain=(.2, .4))
    h = f.join(g)
    assert h.domain.breakpoints == (0., .2, 1., 1.+(.4-.2))
    assert h.domain.b == 1.2
    # changeMap retains onefun coefficients rather than resampling.
    assert h.funs[-1].tech is g.funs[0].tech
    x = jnp.array([1.05, 1.15])
    assert float(jnp.max(jnp.abs(h(x)-(x-1.+.2)))) < 10*h.vscale*EPS


def test_source_orientation_error():
    # Source pass(7).
    f = cj.chebfun(jnp.sin)
    with pytest.raises(ValueError, match="transposition state"):
        f.join(f.T)


def test_source_dimension_error():
    f = cj.chebfun(jnp.sin)
    g = cj.chebfun(lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1))
    with pytest.raises(ValueError, match="dimensions must agree"):
        f.join(g)


def test_source_unbounded_join_keeps_nonlinear_mapping():
    # Source pass(9): deterministic finite grid, unchanged 10*vscale*eps.
    f = cj.chebfun(lambda x: x*jnp.exp(x), domain=(-jnp.inf, -3.))
    g = cj.chebfun(lambda x: (1-jnp.exp(-x))/x, domain=(1., jnp.inf))
    h = f.join(g)
    x = jnp.linspace(-99.7, 99.7, 100)
    exact = jnp.where(x < -3., x*jnp.exp(x), (1-jnp.exp(-(x+4)))/(x+4))
    assert h.domain.breakpoints == (-float("inf"), -3., float("inf"))
    assert isinstance(h.funs[-1], Unbndfun)
    assert h.funs[-1].onefun is g.funs[0].onefun
    assert float(jnp.max(jnp.abs(h(x)-exact))) < 10*h.vscale*EPS


def test_source_single_operand_returns_operand():
    f = cj.chebfun(jnp.sin)
    assert f.join() is f


def test_source_singular_join_keeps_endpoint_exponents():
    # Source pass(8), unchanged 100000*vscale*eps.
    def op(x):
        return jnp.sin(x)/(x+1)
    f = cj.chebfun(op, domain=(-1., -.5, 0.), exps=(-1., 0., 0.))
    g = cj.chebfun(jnp.sin, domain=(1., 2.))
    h = f.join(g)
    x = jnp.linspace(-.999, .999, 1000)
    exact = jnp.where(x < 0, op(x), jnp.sin(x+1))
    assert h.domain.breakpoints == (-1., -.5, 0., 1.)
    assert h.funs[0] is f.funs[0]
    assert float(jnp.max(jnp.abs(h(x)-exact))) < 100000*h.vscale*EPS
