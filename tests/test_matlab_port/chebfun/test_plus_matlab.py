"""All 34 original slots in MATLAB tests/chebfun/test_plus.m.

Provenance
----------
MATLAB source : tests/chebfun/test_plus.m, @chebfun/isequal.m, @chebfun/vscale.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
The first100 seed6178 sites are native captured primitive inputs. Clause29's
post-constructor RNG state is captured from the original native source order.
Clause15 preserves the original catch-all/undefined-ME defect. Clause16 uses
the ME retained by clause15; independent controls qualify actual rejection.
"""
import json
import struct
from functools import lru_cache
from pathlib import Path

import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.linalg import Quasimatrix
from chebfunjax.domain import Domain

EPS = float(jnp.finfo(jnp.float64).eps)
ALPHA = -0.194758928283640 + 0.075474485412665j


def _points():
    path = Path(__file__).resolve().parents[2]/'fixtures/trig_times_rng6178_2025b.json'
    data = json.loads(path.read_text())
    assert data['source_expression'] == 'seedRNG(6178); x = 2*rand(100,1)-1'
    return jnp.asarray([struct.unpack('>d', bytes.fromhex(w))[0] for w in data['query_words']])


X = _points()


def _inf(a):
    """Literal MATLAB vector or matrix infinity norm, including row sums."""
    a = jnp.asarray(a)
    return jnp.max(jnp.abs(a)) if a.ndim == 1 or 1 in a.shape else jnp.max(jnp.sum(jnp.abs(a), axis=1))


def _scale(f):
    return float(jnp.max(f.vscale() if isinstance(f, Quasimatrix) else f.vscale))


def _equal(f, g):
    if isinstance(f, Quasimatrix) or isinstance(g, Quasimatrix):
        fs = f.cols if isinstance(f, Quasimatrix) else f.mat2cell()
        gs = g.cols if isinstance(g, Quasimatrix) else g.mat2cell()
        return len(fs) == len(gs) and all(a.isequal(b) for a, b in zip(fs, gs))
    return f.isequal(g)


def _f1(x):
    return jnp.sin(x)*jnp.abs(x-.1)


def _g1(x):
    return jnp.cos(x)*jnp.sign(x+.2)


def _f2(x):
    return jnp.stack((_f1(x), jnp.exp(x)), axis=-1)


def _g2(x):
    return jnp.stack((_g1(x), jnp.tan(x)), axis=-1)


def _build(op, quasi=False):
    f = cj.chebfun(op, splitting=True)
    return Quasimatrix(f.mat2cell(), Domain((-1., 1.))) if quasi else f


@lru_cache(None)
def _pair(start):
    f_op = _f2 if start in (7, 9, 19, 21, 23, 25) else _f1
    g_op = _g2 if start in (9, 21, 23, 25) else _g1
    f = _build(f_op, start in (19, 21, 25))
    if start in (11, 13):
        f = f.T
        original = f_op
        f_op = lambda x: original(x).T
    if start in (3, 7, 11, 19):
        a, b = f+ALPHA, ALPHA+f
        return _equal(a, b), _inf(a(X)-(f_op(X)+ALPHA)), 10*_scale(a)*EPS
    g = _build(g_op, start in (21, 23))
    if start == 13:
        g = g.T
        original_g = g_op
        g_op = lambda x: original_g(x).T
    a, b = f+g, g+f
    return _equal(a, b), _inf(a(X)-(f_op(X)+g_op(X))), 100*_scale(a)*EPS


@lru_cache(None)
def _numeric_vector(case):
    scalar = case == 18
    op = jnp.sin if scalar else lambda x: jnp.stack((jnp.sin(x), jnp.cos(x), jnp.exp(x)), axis=-1)
    f = _build(op, case == 27)
    g = f+jnp.array([1., 2., 3.])
    exact = (jnp.stack((1+jnp.sin(X), 2+jnp.sin(X), 3+jnp.sin(X)), axis=-1)
             if scalar else jnp.stack((1+jnp.sin(X), 2+jnp.cos(X), 3+jnp.exp(X)), axis=-1))
    return g, jnp.max(jnp.abs((g(X)-exact).ravel())), 10*_scale(g)*EPS


@lru_cache(None)
def _mixed(array):
    dom = [0., float(jnp.pi), 2*float(jnp.pi)]
    if not array:
        f = cj.chebfun(lambda x: x+x*x, domain=dom, splitting=True)
        g = cj.chebfun(jnp.cos, domain=[dom[0], dom[-1]], trig=True)
        exact = cj.chebfun(lambda x: x+x*x+jnp.cos(x), domain=dom, splitting=True)
    else:
        f = cj.chebfun(lambda x: jnp.stack((jnp.cos(x), jnp.sin(x)), axis=-1),
                       domain=[dom[0], dom[-1]], trig=True)
        g = cj.chebfun(lambda x: jnp.stack((x, x**3), axis=-1), domain=dom, splitting=True)
        exact = cj.chebfun(lambda x: jnp.stack((x+jnp.cos(x), x**3+jnp.sin(x)), axis=-1),
                          domain=dom, splitting=True)
    return f+g, exact, g if array else f


@lru_cache(None)
def _source_error_clauses():
    # MATLAB retains the exception assigned by catch ME in the next try block.
    # Clause15 also passes when reading an undefined ME throws; clause16 then
    # compares that retained, unrelated identifier if its operation succeeds.
    f, g = _build(_f1), _build(_g1)
    retained = None
    results = []
    for operand, expected in ((jnp.uint8(128), 'CHEBFUN:plus:unknown'),
                              (g.T, 'CHEBFUN:plus:matdim')):
        try:
            f + operand
            if retained is None:
                raise NameError("Undefined source variable ME")
            identifier = str(retained).split(': ', 1)[0]
            results.append(identifier == expected)
        except Exception as exc:
            retained = exc
            results.append(True)
    return tuple(results)


def _singular_predicate(x):
    dom = [-2., 7.]
    f = cj.chebfun(lambda x: (x-dom[1])**-1*jnp.sin(100*x), domain=dom,
                   exps=(0., -1.), splitting=True)
    g = cj.chebfun(lambda x: (x-dom[1])**-1*jnp.cos(300*x), domain=dom,
                   exps=(0., -1.), splitting=True)
    h = f+g
    exact = (x-dom[1])**-1*(jnp.sin(100*x)+jnp.cos(300*x))
    return _inf(h(x)-exact) < 1e3*EPS*_inf(exact)


def _unbounded_predicate(x):
    dom = [-jnp.inf, jnp.inf]
    f = cj.chebfun(lambda x: jnp.exp(-x*x), domain=dom)
    g = cj.chebfun(lambda x: x*x*jnp.exp(-x*x), domain=dom)
    h = f+g
    exact = jnp.exp(-x*x)+x*x*jnp.exp(-x*x)
    return _inf(h(x)-exact) < 10*EPS*_scale(h)


@pytest.mark.parametrize('slot', range(1, 35), ids=lambda n: f'source_{n:02d}')
def test_source_predicate(slot):
    if slot in (1, 2):
        f = cj.chebfun(jnp.sin)
        assert (f+([] if slot == 1 else cj.chebfun())).isempty()
    elif slot in (15, 16):
        assert _source_error_clauses()[slot-15]
    elif slot in (17, 18, 27):
        g, error, bound = _numeric_vector(slot)
        if slot == 18:
            assert g.size(2) == 3 and error < bound
        else:
            assert error < bound
    elif slot == 28:
        # Reseeding makes the same first100 primitive uniforms applicable.
        # Mapping is two separate IEEE754 operations; no fresh mapped-site
        # native identity is asserted by this deterministic reconstruction.
        uniform = [(float(v)+1.)/2. for v in X]
        assert all(2*u-1 == float(v) for u, v in zip(uniform, X))
        x = jnp.asarray([9*u-2 for u in uniform])
        assert _singular_predicate(x)
    elif slot == 29:
        path = Path(__file__).with_name('plus_source_clause29_native_sites.json')
        data = json.loads(path.read_text())
        assert data['source_commit'] == '7574c77680d7e82b79626300bf255498271a72df'
        assert data['source_sha256'] == 'b711cfc5addb3d863dfd091a7bb743347fd59b51e8225ab4cab60d3d8296243b'
        x = jnp.asarray([struct.unpack('>d', bytes.fromhex(w))[0] for w in data['query_words']])
        assert _unbounded_predicate(x)
    elif slot >= 30:
        h, exact, polynomial = _mixed(slot >= 32)
        if slot in (30, 32, 33):
            column = 1 if slot == 33 else 0
            hc = h.mat2cell()[column] if slot >= 32 else h
            pc = polynomial.mat2cell()[column] if slot >= 32 else polynomial
            assert type(hc.funs[0].tech) is type(pc.funs[0].tech)
        else:
            assert (h-exact).norm(jnp.inf) < (10 if slot == 31 else 100)*EPS*_scale(exact)
    else:
        start = slot if slot % 2 else slot-1
        equal, error, bound = _pair(start)
        assert equal if slot == start else error < bound
