"""Source session bvpTol and explicit solve argument precedence.

Provenance
----------
MATLAB source: @chebop/solvebvpNonlinear.m28-31,110-116.
Chebfun commit: 7574c77
Manufactured quadratic has correction/scale between factory and .01 tolerances.
Source initial-guess acceptance is tested; comparison bound1e-10 is inherited.
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
@pytest.mark.parametrize('session,override', [(.01, None), (5e-13, None),
                                             (.01, 5e-13), (5e-13, .01)])
def test_source_bvptol_preference_and_explicit_override(route, session, override):
    ChebopPref.setDefaults('bvpTol', session)
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
    kwargs = {} if override is None else {'tol': override}
    with warnings.catch_warnings(record=True):
        if route == 'scalar':
            result, returned_info = op.solvebvp(0, n=8, max_iter=0, **kwargs)
            assert returned_info is op._last_info
        else:
            result = op.solve(0, n=8, max_iter=0, **kwargs)
    tolerance = session if override is None else override
    initial_accepted = tolerance == .01
    expected = 1.001 if initial_accepted else 1.001-(1.001**2-1)/(2*1.001)
    if route == 'parameter':
        u, a = result
        assert abs(a-expected) < 1e-10
        assert float((u-x-2).norm()) < 1e-10
    elif route == 'coupled':
        u, v = result
        assert float((u-expected).norm()) < 1e-10
        assert float((v-1).norm()) < 1e-10
    else:
        assert float((result-expected).norm()) < 1e-10
    info = op._last_info
    assert info['bvpTol'] == tolerance
    assert info['newtonTolerance'] == 200*tolerance
    assert len(info['normDelta']) == 1
    assert len(info['dampingHistory']) == (0 if initial_accepted else 1)
