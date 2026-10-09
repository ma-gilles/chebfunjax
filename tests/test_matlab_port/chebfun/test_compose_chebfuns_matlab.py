"""Literal source Chebfun-object composition predicates.

Provenance
----------
MATLAB source : tests/chebfun/test_compose_chebfuns.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
Seed7681 first100 xr words are the native repmat fixture prefix. Source11 recovers primitive uniforms from the next100 captured xr words;
its final100*rand mapping is derived, not freshly captured in MATLAB.
"""
import json
from functools import lru_cache
from pathlib import Path

import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.linalg import Quasimatrix
from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.chebfun3d.chebfun3v import Chebfun3v

EPS = jnp.finfo(jnp.float64).eps
XR = jnp.asarray(json.loads(Path(__file__).with_name('repmat_matlab_inputs.json').read_text())['xr'][:100])


def pair(a, b):
    return jnp.stack((a, b), axis=-1)


def quasi(f):
    return Quasimatrix(f.mat2cell(), f.domain)


@pytest.mark.parametrize('slot', range(1, 11))
def test_original_small(slot):
    domain = (-2., 7.) if slot == 2 else (-1., 1.)
    splitting = slot in (3, 4, 5)
    if slot in (1, 2):
        opf = lambda x: jnp.cos(2*(x+.2))
        opg = lambda x: jnp.sin(x-.1)
    elif slot == 3:
        opf = lambda x: jnp.abs(jnp.cos(2*(x+.2)))
        opg = lambda x: jnp.sin(x-.1)
    elif slot == 4:
        opf = lambda x: jnp.sin(x-.1)
        opg = lambda x: jnp.abs(jnp.cos(2*(x+.2)))
    elif slot == 5:
        opf = lambda x: jnp.abs(jnp.sin(2*(x-.1)))
        opg = lambda x: jnp.abs(jnp.cos(2*(x+.2)))
    elif slot in (6, 8):
        opf = lambda x: pair(jnp.sin(x-.1), jnp.cos(x-.2))
        opg = jnp.exp
    elif slot in (7, 9):
        opf = lambda x: jnp.exp(x)/jnp.exp(1.)
        opg = lambda x: pair(jnp.sin(x-.1), jnp.cos(x-.2))
    else:
        opf = lambda x: pair(jnp.sin(x), jnp.cos(x))
        opg = lambda x: pair(jnp.exp(x), -jnp.sin(x))
    f = cj.chebfun(opf, domain=domain, splitting=splitting)
    g = cj.chebfun(opg, domain=domain, splitting=splitting)
    if slot == 8:
        f = quasi(f)
    if slot == 9:
        g = quasi(g)
    if slot == 10:
        # Source catches any exception; independent controls check the ID.
        with pytest.raises(Exception):
            f.compose(g, pref={'splitting': splitting})
        return
    h = f.compose(g, pref={'splitting': splitting})
    x = ((domain[-1]-domain[0])/2)*XR+domain[0]+(domain[-1]-domain[0])/2
    error = jnp.linalg.norm(h(x)-opg(opf(x)), ord=jnp.inf)
    scale = h.vscale() if isinstance(h, Quasimatrix) else h.vscale
    assert error < 20*jnp.max(scale)*EPS
    assert jnp.array_equal(g(f(jnp.asarray(domain))), h(jnp.asarray(domain)))


def test_original_11_unbounded():
    # Source helper10 resets7681 and draws100 uniforms before the expected
    # dimension error; source11 then draws the next100. Recover those native
    # primitive uniforms losslessly from the captured first1000 2*rand-1 words.
    # The final 100*rand mapping is derived, not a fresh MATLAB capture.
    words = json.loads(Path(__file__).with_name('repmat_matlab_inputs.json').read_text())['xr'][100:200]
    primitive = [(value+1.)/2. for value in words]
    assert all(2.*u-1. == value for u, value in zip(primitive, words))
    assert all((u*2**53).is_integer() for u in primitive)
    x = jnp.asarray([100.*u for u in primitive])
    f = cj.chebfun(lambda t: jnp.exp(-t), domain=[0., jnp.inf])
    g = cj.chebfun(jnp.cos, domain=[-1., 1.])
    h = f.compose(g)
    assert jnp.linalg.norm(h(x)-jnp.cos(jnp.exp(-x)), ord=jnp.inf) < 10*EPS*jnp.max(h.vscale)


def test_original_12_discontinuous_breakpoint():
    x = cj.chebfun('x')
    f = (abs(x) < .25)/(.5)
    assert all(p.tech.ishappy for p in f(x).funs)


@lru_cache(None)
def periodic_inner():
    return cj.chebfun(lambda x: jnp.sin(jnp.pi*x), trig=True)


def test_original_13_periodic():
    x = cj.chebfun('x')
    assert (x**2)(periodic_inner()).isPeriodicTech()


@lru_cache(None)
def mixed_outer_result():
    x = cj.chebfun('x')
    return Quasimatrix([x.exp(), abs(x)], x.domain)(periodic_inner())


@pytest.mark.parametrize('slot', [14, 15])
def test_original_mixed_technology(slot):
    result = mixed_outer_result()
    assert result[slot-14].isPeriodicTech() == (slot == 14)


@pytest.mark.parametrize('slot', range(16, 31))
def test_original_multidimensional(slot):
    domain = [-jnp.pi, jnp.pi]
    periodic = slot in (18, 19, 23, 24, 28, 29, 30)
    if slot <= 24:
        if slot in (17, 22):
            f = cj.chebfun(lambda t: jnp.exp(1j*t), domain=domain)
        elif slot == 20:
            c = cj.chebfun(jnp.cos, domain=domain, trig=True)
            f = Quasimatrix([c, cj.chebfun(jnp.sin, domain=domain)], c.domain)
        else:
            f = cj.chebfun(lambda t: pair(jnp.cos(t), jnp.sin(t)), domain=domain, trig=periodic)
        if slot <= 17:
            outer = Chebfun2.from_function(lambda x, y: x**2+y**2)
            exact = lambda t: 1+0*t
        elif slot <= 20:
            outer = Chebfun2.from_function(lambda x, y: x+y)
            exact = lambda t: jnp.cos(t)+jnp.sin(t)
        elif slot <= 22:
            outer = Chebfun2v.from_functions(lambda x, y: x**2+y**2, lambda x, y: x)
            exact = lambda t: pair(1+0*t, jnp.cos(t))
        else:
            outer = Chebfun2v.from_functions(lambda x, y: x, lambda x, y: y)
            exact = lambda t: pair(jnp.cos(t), jnp.sin(t))
    else:
        if periodic:
            third = lambda t: jnp.cos(2*t)
        elif slot == 26:
            third = lambda t: 1j*t
        else:
            third = lambda t: t
        f = cj.chebfun(lambda t: jnp.stack((jnp.cos(t), jnp.sin(t), third(t)), axis=-1), domain=domain, trig=periodic)
        if slot in (27, 30):
            outer = Chebfun3v.from_functions(lambda x, y, z: x**2+y**2, lambda x, y, z: z)
            exact = lambda t: pair(1+0*t, third(t))
        else:
            outer = chebfun3(lambda x, y, z: x**2+y**2+z**2)
            exact = lambda t: 1+third(t)**2
    if slot == 26:
        with pytest.raises(ValueError, match='CHEBFUN:CHEBFUN:compose:complex3'):
            f.compose(outer)
        return
    result = outer(f) if slot == 30 else f.compose(outer)
    if slot in (19, 20, 24, 29, 30):
        assert result.isPeriodicTech() == (slot != 20)
    else:
        truth = cj.chebfun(exact, domain=domain, trig=periodic)
        assert (result-truth).norm() < (10 if slot == 28 else 100)*EPS
