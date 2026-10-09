"""Literal MATLAB binary composition predicates, including matrix norms.

Provenance
----------
MATLAB source: tests/chebfun/test_compose_binary.m; commit7574c77.
Source1–9 use native seed7681 xr fixture. Source10 recovers the primitive
uniforms from native seed6178 xr words and applies the source affine map;
that mapping is derived, not a fresh native capture. Source11 awaits the
subsequent seed6178 words, with its literal body retained below.
"""
import json
import struct
from pathlib import Path

import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.linalg import Quasimatrix

EPS = jnp.finfo(jnp.float64).eps
XR = jnp.asarray(json.loads(Path(__file__).with_name('repmat_matlab_inputs.json').read_text())['xr'][:100])


def scale(f):
    return jnp.max(f.vscale() if isinstance(f, Quasimatrix) else f.vscale)


def pair(a, b):
    return jnp.stack((a, b), axis=-1)


@pytest.mark.parametrize('slot', range(1, 10))
def test_original_binary(slot):
    fd = [-2., 7.] if slot == 2 else [-1., 1.]
    gd = [-1., .1, 1.] if slot in (3, 5) else fd
    splitting = slot in (4, 5)
    opf = lambda x: jnp.cos(2*(x+.2))
    opg = lambda x: jnp.abs(x-.1) if slot in (3, 5) else jnp.sin(x-.1)
    if slot == 1:
        op = lambda a, b: a*b
    elif slot == 2:
        op = lambda a, b: a*a-b*b
    elif slot == 3:
        op = lambda a, b: jnp.sin(a*b)
    elif slot == 4:
        op = lambda a, b: jnp.abs(a*b)
    elif slot == 5:
        op = lambda a, b: jnp.cos(jnp.abs(a*b)-.1)
    else:
        opf = lambda x: pair(jnp.cos(2*(x+.2)), jnp.sin(2*(x-.1)))
        opg = lambda x: pair(jnp.exp(x), 1/(1+25*(x-.1)**2))
        op = lambda a, b: a+2*b
    f, g = cj.chebfun(opf, domain=fd, splitting=splitting), cj.chebfun(opg, domain=gd, splitting=splitting)
    if slot in (8, 9):
        f = Quasimatrix([c.set_point_values(f.point_values[:, k]) for k, c in enumerate(f.mat2cell())], f.domain)
    if slot in (7, 9):
        g = Quasimatrix([c.set_point_values(g.point_values[:, k]) for k, c in enumerate(g.mat2cell())], g.domain)
    h = f.compose(op, g, pref={'splitting': splitting})
    dom = jnp.asarray(sorted(set(fd+gd)))
    x = ((dom[-1]-dom[0])/2)*XR+dom[0]+(dom[-1]-dom[0])/2
    assert jnp.linalg.norm(h(x)-op(opf(x), opg(x)), ord=jnp.inf) < 20*scale(h)*EPS
    assert jnp.array_equal(op(opf(dom), opg(dom)), h(dom))


def test_original_binary_10_singular():
    path = Path(__file__).resolve().parents[2]/'fixtures/trig_times_rng6178_2025b.json'
    data = json.loads(path.read_text())
    assert data['source_expression'] == 'seedRNG(6178); x = 2*rand(100,1)-1'
    xr = [struct.unpack('>d', bytes.fromhex(w))[0] for w in data['query_words']]
    u = [(v+1)/2 for v in xr]
    assert all(2*v-1 == x and (v*2**53).is_integer() for v, x in zip(u, xr))
    x = jnp.asarray([9*v-2 for v in u])
    f = cj.chebfun(lambda x: jnp.sin(20*x)/(x+2), domain=[-2, 7], exps=(-1, 0))
    g = cj.chebfun(lambda x: x*x/(x+2), domain=[-2, 7], exps=(-1, 0))
    h = f.compose(lambda a, b: a+b, g)
    exact = lambda x: (jnp.sin(20*x)+x*x)/(x+2)
    assert jnp.linalg.norm(h(x)-exact(x), ord=jnp.inf) < 1e6*scale(h)*EPS
    assert h(7.) == exact(jnp.asarray(7.))


def test_original_binary_11_unbounded():
    path = Path(__file__).with_name('compose_binary_native_subsequent6178.json')
    if not path.exists():
        pytest.skip('Native seed6178 subsequent100 uniforms for source11 not captured')
    x = jnp.asarray(json.loads(path.read_text())['x'])
    f = cj.chebfun(lambda x: jnp.exp(-x), domain=[0, jnp.inf])
    g = cj.chebfun(lambda x: x*jnp.exp(-x), domain=[0, jnp.inf])
    h = f.compose(lambda a, b: a+b, g)
    assert jnp.linalg.norm(h(x)-(x+1)*jnp.exp(-x), ord=jnp.inf) < 10*EPS*scale(h)
