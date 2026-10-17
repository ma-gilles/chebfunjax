"""Complex extrema preserve native stored values and breakpoint side repair.

Provenance
----------
MATLAB source : @chebfun/minandmax.m
Chebfun commit: 7574c77
"""

import math

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun


def _jump(left, right, breakpoint=1.):
    return chebfun([lambda x: left + 1j*(x-breakpoint),
                    lambda x: right + 1j*(x-breakpoint)],
                   domain=[-1., breakpoint, breakpoint+1])


@pytest.mark.parametrize('left,right,sign', [(1, 2, -1), (2, 1, 1)])
def test_one_sided_minimum_repairs_location(left, right, sign):
    f = _jump(left, right)
    (xmin, vmin), _ = f.minandmax()
    assert float(xmin) == 1. + sign*math.ulp(1.)
    assert abs(complex(vmin) - 1) < 2e-14
    assert abs(complex(f(xmin)) - complex(vmin)) < 2e-14


def test_zero_breakpoint_repairs_side():
    f = _jump(1, 2, 0.)
    (xmin, vmin), _ = f.minandmax()
    # Native rootfinding puts this at -1.1102230246251565e-16.
    assert -2e-14 < float(xmin) < 0.
    assert abs(complex(vmin) - 1) < 2e-14


@pytest.mark.parametrize('location,value,which', [(-1., 100j, 1), (1., 0j, 0)])
def test_complex_isolated_point_values(location, value, which):
    f = chebfun(lambda x: 1 + 1j*x)
    index = 0 if location == -1 else -1
    f = f.set_point_values(f.point_values.at[index].set(value))
    pos, val = f.minandmax()[which]
    assert float(pos) == location
    assert abs(complex(val) - value) < 2e-14


def test_array_columns_use_scalar_complex_policy():
    f = chebfun([lambda x: jnp.stack([1+1j*(x-1), 2+1j*(x-1)], axis=-1),
                 lambda x: jnp.stack([2+1j*(x-1), 1+1j*(x-1)], axis=-1)],
                domain=[-1., 1., 2.])
    (xmin, vmin), _ = f.minandmax()
    expected = jnp.asarray([1-math.ulp(1.), 1+math.ulp(1.)])
    assert bool(jnp.array_equal(xmin, expected))
    assert float(jnp.max(jnp.abs(vmin - 1))) < 2e-14
