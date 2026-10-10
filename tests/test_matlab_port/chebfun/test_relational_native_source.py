"""Native test_lt/test_le.m clauses 1--10, Chebfun 7574c77.

First probes use the retained MATLAB seed6178 capture. Unbounded probes use
NumPy MT19937 validated against both captured 100-word uniform blocks;
this is an explicit stream adapter, not a fresh MATLAB execution.
"""
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj

FIX = json.loads((Path(__file__).parent/'fixtures/logical_source_matlab.json').read_text())
X = jnp.asarray(FIX['x'])
D = [-1., -.5, 0., .5, 1.]

def gop(x):
    return (jnp.exp(.5)-jnp.exp(-.5))*(x+.5)+jnp.exp(-.5)

def crossing(h, root, value):
    b = np.asarray(h.domain.breakpoints)
    ind = np.flatnonzero(np.abs(b-root) < 10*float(h.vscale)*np.finfo(float).eps)
    assert ind.size
    assert np.all(np.asarray(h.point_values)[ind] == value)

@pytest.mark.parametrize('op', ['lt', 'le'])
@pytest.mark.parametrize('clause', range(1, 11))
def test_native_relational(op, clause):
    compare = lambda f, g: getattr(f, op)(g)
    equal = int(op == 'le')
    if clause == 1:
        f, g = cj.chebfun(jnp.sin, domain=D), cj.chebfun()
        assert compare(f, g).isempty()
        assert compare(g, f).isempty()
    elif clause == 2:
        f = cj.chebfun(jnp.sin, domain=D)
        g = cj.chebfun(lambda x: 0*x+jnp.sqrt(2)/2)
        h = compare(f, g)
        crossing(h, np.pi/4, equal)
        np.testing.assert_array_equal(h(X), .5*(jnp.sign(jnp.sqrt(2)/2-jnp.sin(X))+1))
    elif clause in (3, 6):
        f = cj.chebfun(jnp.exp, domain=D if clause == 6 else [-1., 1.])
        h = compare(f, cj.chebfun(gop))
        crossing(h, .5, equal)
        crossing(h, -.5, equal)
        np.testing.assert_array_equal(h(X), .5*(jnp.sign(gop(X)-jnp.exp(X))+1))
    elif clause in (4, 5):
        f = cj.chebfun(jnp.exp)
        h = compare(f, f) if clause == 4 else (compare(-f, f) if op == 'lt' else compare(f, -f))
        assert len(h.funs) == 1
        expected = equal if clause == 4 else 1-equal
        np.testing.assert_array_equal(h(X), jnp.full_like(X, expected))
    elif clause in (7, 8):
        f = cj.chebfun(lambda x: jnp.stack((jnp.sin(x), jnp.cos(x), jnp.exp(x)), axis=-1), domain=D)
        g = cj.chebfun(lambda x: jnp.exp(2*jnp.pi*1j*x))
        with pytest.raises(Exception, match=f'CHEBFUN:CHEBFUN:{op}:array'):
            compare(f, g) if clause == 7 else compare(g, f)
    elif clause == 9:
        f = cj.chebfun(lambda x: -jnp.sin(x)/(x+1), exps=(-1, 0))
        h = compare(f, cj.chebfun(lambda x: 0*x))
        assert np.asarray(h.point_values)[0] == 0
        np.testing.assert_array_equal(h(X), .5*(jnp.sign(X)+1))
    else:
        rng = np.random.RandomState(6178)
        np.testing.assert_array_equal(2*rng.rand(100)-1, FIX['x'])
        second = rng.rand(100)
        np.testing.assert_array_equal(9*second-2, FIX['y'])
        x = jnp.asarray(101*second-1)
        fop = lambda x: jnp.exp(-x)-1
        gg = lambda x: x*jnp.exp(-x)
        h = compare(cj.chebfun(fop, domain=(-1., jnp.inf)), cj.chebfun(gg, domain=(-1., jnp.inf)))
        np.testing.assert_array_equal(h(x), fop(x) <= gg(x))
        np.testing.assert_array_equal(h.point_values, [0, equal, 1])
