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
@pytest.mark.parametrize('session,kwargs,expected', [
    (8, {}, 8), (16, {}, 16),
    (16, {'n_min': 8, 'n_max': 8}, 8), (16, {'n': 8}, 8)])
def test_session_dimensions_actual_newton_grid(route, session, kwargs, expected):
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
        result, info = op.solvebvp(0, max_iter=0, **kwargs)
    assert info is op._last_info
    assert info['linearDimensions']
    grids = info['linearDimensions']
    if route == 'coupled':
        grids = [size for row in grids for size in row]
    assert set(grids) == {expected}
    # Native standardChop returns unresolved for length <17. A capped
    # adaptive scalar/coupled correction therefore leaves the guess intact;
    # explicit n bypasses adaptivity. The parameter adapter trims first.
    exhausted = route != 'parameter' and kwargs.get('n') is None
    value = 1.001 if exhausted else 1.001-(1.001**2-1)/(2*1.001)
    if exhausted:
        assert len(info['normDelta']) == 0
        assert len(info['dampingHistory']) == 0
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


@pytest.mark.parametrize('kwargs,expected', [
    ({}, (8, 64)), ({'n_min': 16}, (16, 64)),
    ({'n_max': 32}, (8, 32)), ({'n': 7}, (8, 64))])
def test_independent_public_dimension_precedence(monkeypatch, kwargs, expected):
    ChebopPref.setDefaults('minDimension', 8, 'maxDimension', 64)
    received = {}
    def capture(self, f, **kw):
        received.update(kw)
        return None
    monkeypatch.setattr(Chebop, '_solve_impl', capture)
    op = Chebop(lambda x, u: u)
    op.solve(**kwargs)
    assert (received['n_min'], received['n_max']) == expected
    assert received['n'] == kwargs.get('n')


def test_factory_dimensions_unchanged(monkeypatch):
    received = {}
    def capture(self, f, **kw):
        received.update(kw)
        return None
    monkeypatch.setattr(Chebop, '_solve_impl', capture)
    Chebop(lambda x, u: u).solve()
    assert (received['n_min'], received['n_max']) == (32, 4096)
