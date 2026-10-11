"""Source damping preference, with existing explicit operator override.

Provenance
----------
MATLAB source: @chebop/solvebvpNonlinear.m33-36,151-182.
Chebfun commit: 7574c77
Manufactured cubic from the existing lambdaMin control. Native first undamped
iteration computes its correction without updating u. Bound1e-10 is inherited.
"""
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
@pytest.mark.parametrize('session,override', [(False, None), (True, None),
                                             (False, True), (True, False)])
def test_source_damping_preference_and_explicit_override(route, session, override):
    ChebopPref.setDefaults('damping', session)
    x = cj.chebfun(lambda t: t)
    if route == 'scalar':
        op = Chebop(lambda x, u: u**3-2*u+2)
        op.init = cj.chebfun(1)
    elif route == 'parameter':
        op = Chebop(lambda x, u, a: u.diff(2)+a**3-2*a+2)
        op.lbc = lambda u, a: [u-1, u.diff()-1]
        op.rbc = lambda u, a: u-3
        op.init = [x+2, 1.0]
    else:
        op = Chebop(lambda x, u, v: [u**3-2*u+2, v-1])
        op.bc = lambda u, v: []
        op.init = [cj.chebfun(1), cj.chebfun(1)]
    if override is not None:
        op.damping = override
    with pytest.warns(RuntimeWarning, match='maximum number of iterations'):
        result = op.solve(0, n=8, max_iter=0)
    damping = session if override is None else override
    expected = .75 if damping else 1.
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
    assert len(info['normDelta']) == len(info['dampingHistory']) == 1
    assert info['dampingHistory'][0]['damping'] is damping
