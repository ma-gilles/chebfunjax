"""All 31 original predicates of MATLAB tests/chebfun/test_times.m.

MATLAB ``.*`` uses the explicit Python ``.times`` adapter; scalar-left
multiplication uses the existing scalar operator. Native seed6178 draws
are captured in original order. Constructors, default continuous norms,
structural equality, error identifiers, and bounds retain source semantics.

Provenance
----------
MATLAB source : tests/chebfun/test_times.m
Chebfun commit: 7574c77
"""

import json
from pathlib import Path

import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.linalg import Quasimatrix

EPS = float(jnp.finfo(jnp.float64).eps)
ALPHA = -0.194758928283640 + 0.075474485412665j
DRAWS = json.loads((Path(__file__).parent / "fixtures" / "times_source_matlab.json").read_text())
X = jnp.asarray(DRAWS["x1"])


def scale(f):
    return float(jnp.max(f.vscale() if isinstance(f, Quasimatrix) else f.vscale))


def equal(a, b):
    return a.isequal(b)


def f1_op(x):
    return jnp.sin(x) * jnp.abs(x - 0.1)


def g1_op(x):
    return jnp.cos(x) * jnp.sign(x + 0.2)


def f2_op(x):
    return jnp.stack([f1_op(x), jnp.exp(x)], axis=-1)


def g2_op(x):
    return jnp.stack([g1_op(x), jnp.tan(x)], axis=-1)


def quasi(op):
    f = cj.chebfun(op, splitting=True)
    return Quasimatrix(f.mat2cell(), f.domain)


@pytest.fixture(scope="module")
def basic():
    f1 = cj.chebfun(f1_op, splitting=True)
    g1 = cj.chebfun(g1_op, splitting=True)
    f2 = cj.chebfun(f2_op, splitting=True)
    g2 = cj.chebfun(g2_op, splitting=True)
    return f1, g1, f2, g2, quasi(f2_op), quasi(g2_op)


@pytest.mark.parametrize("clause", range(1, 24))
def test_source_01_23(clause, basic):
    f1, g1, f2, g2, f2q, g2q = basic
    if clause in (1, 2):
        f = cj.chebfun(jnp.sin, domain=(-1, 1))
        assert f.times([] if clause == 1 else cj.chebfun()).isempty()
    elif clause in (21, 22, 23):
        arg = "X" if clause == 21 else f1.T if clause == 22 else g2q.T
        error = "unknown" if clause == 21 else "matdim"
        with pytest.raises((TypeError, ValueError), match="CHEBFUN:CHEBFUN:times:" + error + ":"):
            f1.times(arg)
    elif clause in (3, 4, 7, 8, 11, 12, 15, 16):
        f, op = (
            (f1, f1_op)
            if clause in (3, 4)
            else (f2, f2_op)
            if clause in (7, 8)
            else (f1.T, lambda x: f1_op(x).T)
            if clause in (11, 12)
            else (f2q, f2_op)
        )
        a, b = f.times(ALPHA), ALPHA * f
        if clause % 2:
            assert equal(a, b)
        else:
            assert jnp.max(jnp.abs((a(X) - op(X) * ALPHA).reshape(-1))) < 1e2 * scale(a) * EPS
    else:
        f, g, fo, go = (
            (f1, g1, f1_op, g1_op)
            if clause in (5, 6)
            else (f2, g2, f2_op, g2_op)
            if clause in (9, 10)
            else (f1.T, g1.T, lambda x: f1_op(x).T, lambda x: g1_op(x).T)
            if clause in (13, 14)
            else (f2q, g2, f2_op, g2_op)
            if clause in (17, 18)
            else (f2q, g2q, f2_op, g2_op)
        )
        a, b = f.times(g), g.times(f)
        if clause % 2:
            assert (a - b).norm() < 10 * scale(a) * EPS
        else:
            assert jnp.max(jnp.abs((a(X) - fo(X) * go(X)).reshape(-1))) < 1e2 * scale(a) * EPS


@pytest.mark.parametrize("clause", (24, 25))
def test_source_singular(clause):
    dom = (-2, 7)
    x = jnp.asarray(DRAWS["x2"])
    # MATLAB negative real bases to fractional powers use principal complex powers.
    power = lambda x, p: (x - dom[1] + 0j) ** p
    if clause == 24:
        op = lambda x: power(x, -0.5) * jnp.sin(x)
        f = cj.chebfun(op, domain=dom, exps=(0, -0.5), splitting=True)
        g = f.times(3)
        exact = cj.chebfun(lambda x: 3 * op(x), domain=dom, exps=(0, -0.5), splitting=True)
        bound = 50
    else:
        f = cj.chebfun(
            lambda x: power(x, -0.3) * jnp.sin(100 * x), domain=dom, exps=(0, -0.3), splitting=True
        )
        g = cj.chebfun(
            lambda x: power(x, -0.5) * jnp.cos(300 * x), domain=dom, exps=(0, -0.5), splitting=True
        )
        g = f.times(g)
        exact = cj.chebfun(
            lambda x: power(x, -0.3 - 0.5) * jnp.sin(100 * x) * jnp.cos(300 * x),
            domain=dom,
            exps=(0, -0.3 - 0.5),
            splitting=True,
        )
        bound = 1e4
    assert jnp.max(jnp.abs(g(x) - exact(x))) < bound * EPS * jnp.max(jnp.abs(exact(x)))


def test_source_26():
    x = jnp.asarray(DRAWS["x3"])
    f = cj.chebfun(lambda x: x**2 * jnp.exp(-(x**2)), domain=(-jnp.inf, jnp.inf))
    g = cj.chebfun(lambda x: (1 - jnp.exp(-(x**2))) / x, domain=(-jnp.inf, jnp.inf))
    h = f.times(g)
    assert jnp.max(
        jnp.abs(h(x) - x * jnp.exp(-(x**2)) * (1 - jnp.exp(-(x**2))))
    ) < 10 * EPS * scale(f)


@pytest.mark.parametrize("clause", range(27, 32))
def test_source_27_31(clause):
    dom = (0, jnp.pi, 2 * jnp.pi)
    if clause <= 28:
        f = cj.chebfun(lambda x: x**2, domain=dom, splitting=True)
        g = cj.chebfun(jnp.cos, domain=(dom[0], dom[-1]), periodic=True)
        h = f.times(g)
        if clause == 27:
            assert type(h.funs[0].tech) is type(f.funs[0].tech)
        else:
            exact = cj.chebfun(lambda x: x**2 * jnp.cos(x), domain=dom, splitting=True)
            assert (h - exact).norm(jnp.inf) < 1e1 * EPS * scale(exact)
    else:
        f = cj.chebfun(
            lambda x: jnp.stack([jnp.cos(x), jnp.sin(x)], axis=-1),
            domain=(dom[0], dom[-1]),
            periodic=True,
        )
        g = cj.chebfun(lambda x: jnp.stack([x, x**3], axis=-1), domain=dom, splitting=True)
        h = f.times(g)
        if clause in (29, 30):
            k = clause - 29
            assert type(h.mat2cell()[k].funs[0].tech) is type(g.mat2cell()[k].funs[0].tech)
        else:
            exact = cj.chebfun(
                lambda x: jnp.stack([x * jnp.cos(x), x**3 * jnp.sin(x)], axis=-1),
                domain=dom,
                splitting=True,
            )
            assert (h - exact).norm(jnp.inf) < 1e2 * EPS * scale(exact)
