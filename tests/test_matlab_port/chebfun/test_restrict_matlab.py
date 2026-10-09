"""Literal tests/chebfun/test_restrict.m at MATLAB7574c77.

The first1000 seedRNG(7681) transformed uniforms are reused from the native
repmat fixture: both source files use exactly xr=2*rand(1000,1)-1. Later
100+100 draws for slot24 are pending; no substitute generator is used.
"""
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.tech.trigtech import Trigtech

EPS = np.finfo(float).eps
XR = jnp.asarray(json.loads(Path(__file__).with_name('repmat_matlab_inputs.json').read_text())['xr'])
DOMAINS = ([-1, .5], [-.2, 1], [-.2, .5], [-.35, .1, .2, .5])


def check_function(f, exact, dom):
    fr = f.restrict(dom)
    x = ((dom[-1]-dom[0])/2)*(XR+1)+dom[0]
    error = np.linalg.norm(np.asarray(fr(x)-exact(x)), np.inf)
    assert all(x in fr.domain.breakpoints for x in dom)
    assert error < 1e5*fr.vscale*EPS


def test_source_01():
    assert cj.chebfun().restrict([-.5, .5]).isempty()


def test_source_02():
    f = cj.chebfun(lambda x: x)
    f.restrict([-.5, .5])
    assert f.domain.breakpoints == (-1., 1.)


def test_source_03():
    f = cj.chebfun(jnp.sin)
    fr = f.restrict([-1, 1])
    assert fr.domain.breakpoints == (-1., 1.)
    assert np.linalg.norm(np.asarray(fr(XR)-jnp.sin(XR)), np.inf) < 10*fr.vscale*EPS


@pytest.mark.parametrize('slot,dom', [(4, [-2, .5]), (5, [-.5, 2]), (6, [-1, -.25, .3, .1, 1])])
def test_source_bad_domain(slot, dom):
    f = cj.chebfun(jnp.sin)
    with pytest.raises(ValueError, match='CHEBFUN:CHEBFUN:restrict:subdom' if slot != 6 else None):
        f.restrict(dom)


@pytest.mark.parametrize('slot', range(7, 23))
def test_source_smooth(slot):
    group, k = divmod(slot-7, 4)
    exact = (lambda x: 1/(1+25*(x-.1)**2)) if group < 2 else (
        lambda x: jnp.stack([jnp.sin(x-.1), jnp.cos(x+.2), jnp.exp(x)], axis=-1))
    # MATLAB -1:.1:1 constructs a symmetric colon grid, not accumulated adds.
    domain = np.linspace(-1, 1, 21) if group % 2 else [-1, 1]
    f = cj.chebfun(exact, domain=domain)
    check_function(f, exact, DOMAINS[k])


def test_source_23_singular():
    dom = [-2, 7]
    def op(x):
        return (x-dom[0])**-.5*jnp.sin(100*x)*(x-dom[1]+0j)**-.5
    f = cj.chebfun(op, domain=dom, exps=[-.5, -.5], splitting=True)
    check_function(f, op, [-2, 1, 3.5, 6.5, 7])


def test_source_24_native_pending():
    fixture = Path(__file__).with_name('restrict_matlab_inputs.json')
    if not fixture.exists():
        pytest.skip('Native source-order seed7681 subsequent100+100 uniforms not captured')
    native = json.loads(fixture.read_text())
    # x1/x2 must be captured after the source1000-point prefix, already
    # mapped by MATLAB to [-100,1] and [1,2*pi], respectively.
    x1, x2 = jnp.asarray(native['x1']), jnp.asarray(native['x2'])
    f = cj.chebfun([jnp.exp, lambda x: jnp.sin(3*x)], domain=[-np.inf, 1, 3*np.pi])
    g = f.restrict([-np.inf, -1, np.pi, 2*np.pi])
    err = jnp.concatenate([g(x1)-jnp.exp(x1), g(x2)-jnp.sin(3*x2)])
    assert np.linalg.norm(np.asarray(err), np.inf) < 1e2*EPS*g.vscale


def test_source_25():
    f = cj.chebfun(lambda x: jnp.abs(x+.04), domain=[-1, .04, 1], splitting=True)
    f = f.restrict([-.04, .04])
    g = cj.chebfun(lambda x: jnp.abs(x+.04), domain=[-.04, .04], splitting=True)
    assert (f-g).norm(np.inf) < 10*EPS


def test_source_26_unbounded():
    f = cj.chebfun(lambda x: 4*x**2-2, domain=[-np.inf, np.inf])
    g = f.restrict([-1, 1])
    assert abs(float(f(1)-g(1))) < EPS*f.vscale


def test_source_27_unbounded():
    f = cj.chebfun(lambda t: t**.5/jnp.exp(t), domain=[0, np.inf], exps=[.5, 0])
    g = f
    f = f.define_point(1, f(1))
    assert (f-g).norm(np.inf) < 1e2*EPS


def test_source_28():
    f = cj.chebfun(lambda x: jnp.sin(2*jnp.pi*x), domain=[1, 2], trig=True)
    g = f.restrict(f.domain)
    assert not isinstance(g.funs[0].tech, Trigtech)


def test_source_29():
    f = cj.chebfun(lambda x: jnp.sin(2*jnp.pi*x), domain=[1, 2], trig=True)
    g = f.restrict([1.5, 1.6])
    h = cj.chebfun(lambda x: jnp.sin(2*jnp.pi*x), domain=[1.5, 1.6])
    assert (g-h).norm(np.inf) < 1e-12
