"""Native dimension preference forwarding through public nonlinear solves.

Provenance
----------
MATLAB source: @linop/linsolve.m; @valsDiscretization/valsDiscretization.m;
    @coeffsDiscretization/coeffsDiscretization.m dimensionValues.
Chebfun commit: 7574c77
"""
import warnings

import pytest

import chebfunjax as cj
from chebfunjax.chebpref import ChebopPref
from chebfunjax.operators.chebop import Chebop


@pytest.fixture(autouse=True)
def restore_session():
    previous = ChebopPref._defaults
    ChebopPref.setDefaults('factory')
    try:
        yield
    finally:
        ChebopPref._defaults = previous


@pytest.mark.parametrize('route', ['scalar', 'parameter', 'coupled'])
def test_session_dimensions_resolved_newton_grid(route):
    session, kwargs, expected = 32, {}, 32
    ChebopPref.setDefaults('minDimension', session, 'maxDimension', session)
    x = cj.chebfun(lambda t: t)
    if route == 'scalar':
        op = Chebop(lambda x, u: u**2-1)
        op.init = cj.chebfun(1.001)
    elif route == 'parameter':
        op = Chebop(lambda x, u, a: u.diff(2)+a**2-1)
        op.lbc = lambda u, a: [u-1, u.diff()-1]
        op.rbc = lambda u, a: u-3
        op.init = [x+2, 1.001]
    else:
        op = Chebop(lambda x, u, v: [u**2-1, v-1])
        op.bc = lambda u, v: []
        op.init = [cj.chebfun(1.001), cj.chebfun(1)]
    op.damping = True
    with warnings.catch_warnings(record=True):
        result, info = op.solvebvp(0, max_iter=5, **kwargs)
    assert info is op._last_info
    assert info['linearDimensions']
    grids = info['linearDimensions']
    if route == 'coupled':
        grids = [size for row in grids for size in row]
    assert set(grids) == {expected}
    assert info['converged']
    value = 1.0
    if route == 'parameter':
        u, a = result
        assert abs(a-value) < 1e-10
        assert float((u-x-2).norm()) < 1e-10
    elif route == 'coupled':
        u, v = result
        assert float((u-value).norm()) < 1e-10
        assert float((v-1).norm()) < 1e-10
    else:
        assert float((result-value).norm()) < 1e-10
