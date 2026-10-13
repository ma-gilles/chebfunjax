"""Independent native spin3 checkpoints, Chebfun 7574c77.

The 1e-11 relative comparison is supplemental, consistent with the existing
native alternate-scheme fixture controls; it is not an original MATLAB test.
Time, shape and checkpoint selection are compared exactly.
"""
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.operators.spinop3 import Spinop3, spin3
from chebfunjax.utils.quadrature import trigpts

DATA = Path(__file__).with_name('data')
RECORDS = json.loads((DATA/'public_time_native.json').read_text())['cases']


@pytest.mark.parametrize('record', RECORDS, ids=lambda r: r['name'])
def test_native_checkpoint(record):
    r = record
    with np.load(DATA/'public_time_native.npz') as fixture:
        values = {key: fixture[r['name']+'__'+key] for key in ('input','initial','grid','query')}
    domain = tuple(r['domain'])
    s = Spinop3(domain, r['tspan'])
    init = [chebfun3(jnp.asarray(v),domain,trig=True) for v in values['input']]
    if r['model'] == 'system':
        s.lin = '@(u,v) [.1*lap(u); .2*lap(v)]'
        s.nonlin = [lambda u,v: .3*v, lambda u,v: -.2*u]
        s.init = init
    else:
        s.lin = '@(u) (.1+.05i)*lap(u)' if r['model']=='complex' else '@(u) .1*lap(u)'
        s.nonlin = (lambda u: .2*u) if r['model']=='complex' else (lambda u: u-u**3)
        s.init = init[0]
    axes = [trigpts(8,domain[k:k+2])[0] for k in (0,2,4)]
    grid = jnp.meshgrid(*axes,indexing='xy')
    for k, f in enumerate(init):
        np.testing.assert_allclose(f(*grid),values['initial'][...,k],rtol=0,atol=1e-11)
    result,times = spin3(s,8,.01,'plot','off','dealias',r['dealias'],return_times=True)
    np.testing.assert_array_equal(times,r['times'])
    scalar = len(r['tspan'])==2 and r['nvars']==1
    if not scalar:
        assert result.shape == (r['nvars'],r['nout'])
    errors = []
    for k in range(r['nvars']):
        for col in range(r['nout']):
            f = result if scalar else result[k,col]
            actual = np.asarray(f(*grid))
            expected=values['grid'][...,k,col]
            error = np.max(np.abs(actual-expected))/np.max(np.abs(expected))
            assert error < 1e-11
            q = r['query']
            actual = np.asarray(f(jnp.asarray(q['x']),jnp.asarray(q['y']),jnp.asarray(q['z'])))
            expected=values['query'][k,col]
            qerror = np.max(np.abs(actual-expected))/np.max(np.abs(expected))
            assert qerror < 1e-11
            errors.append((float(error),float(qerror)))
    print('NATIVE_RESULT='+json.dumps({'case':r['name'],'errors':errors}))
