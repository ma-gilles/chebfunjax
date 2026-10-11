"""Literal native tests/chebtech/test_times.m, Chebfun7574c77:25x2.
Seed6178 first100 mapped draws verified against trreal1 native capture.
Defaults adapt techPref(eps,17,65537); scalar Nx1 samples represented1D.
"""

import os
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

EPS = np.finfo(float).eps
X = jnp.asarray(2 * np.random.RandomState(6178).rand(100) - 1)
ALPHA = -0.194758928283640 + 0.075474485412665j
BETA = -0.526634844879922 - 0.685484380523668j
TECHS = (
    [{"1": Chebtech1, "2": Chebtech2}[os.environ["TIMES_TECH"]]]
    if "TIMES_TECH" in os.environ
    else [Chebtech1, Chebtech2]
)


@pytest.fixture(params=TECHS, ids=lambda cls: cls.__name__)
def Tech(request):
    return request.param


def ni(a):
    a = np.asarray(a)
    return float(np.max(np.sum(abs(a), axis=1))) if a.ndim == 2 else float(np.max(abs(a)))


def cols(*fs):
    return lambda x: jnp.stack([f(x) for f in fs], axis=1)


def scalar(f, op):
    a = f * ALPHA
    b = ALPHA * f
    assert a.isequal(b)
    assert ni(a(X) - op(X) * ALPHA) < 10 * float(jnp.max(a.vscale_columns * EPS))


def function(f, op, g, gop, pos=False):
    h = f * g
    tol = 1e4 * float(jnp.max(h.vscale_columns * EPS))
    assert ni(h(X) - op(X) * gop(X)) < tol
    if pos:
        values = h.coeffs2vals(h.coeffs)
        # Native positive-product branch produces real values; no imaginary discard.
        assert bool(jnp.all(values >= -tol))
    return h


def test_scalar_1_3_6_8_11(Tech):
    f = Tech.empty()
    g = Tech.from_function(lambda x: x)
    assert (f * f).isempty() and (f * g).isempty() and (g * f).isempty()
    f = Tech.from_function(jnp.sin)
    scalar(f, jnp.sin)
    gop = lambda x: ALPHA * jnp.ones_like(x)
    function(f, jnp.sin, Tech.from_function(gop), gop)
    op = jnp.ones_like
    f = Tech.from_function(op)
    function(f, op, f, op)
    op = lambda x: jnp.exp(x) - 1
    f = Tech.from_function(op)
    for gop in [
        lambda x: 1 / (1 + x * x),
        lambda x: jnp.cos(1e4 * x),
        lambda x: jnp.sinh(x * jnp.exp(2j * jnp.pi / 6)),
    ]:
        function(f, op, Tech.from_function(gop), gop)


def test_array_4_5_7_12_15(Tech):
    op = cols(jnp.sin, jnp.cos)
    f = Tech.from_function(op)
    scalar(f, op)
    gop = lambda x: jnp.tile(jnp.array([ALPHA, BETA]), (x.shape[0], 1))
    function(f, op, Tech.from_function(gop), gop)
    op = cols(jnp.sin, jnp.cos, jnp.exp)
    f = Tech.from_function(op)
    g = Tech.from_function(jnp.tanh)
    h1 = f * g
    h2 = g * f
    assert h1.coeffs.shape == h2.coeffs.shape
    assert float(jnp.linalg.norm(h1.coeffs - h2.coeffs, ord=2)) < 10 * EPS
    exact = cols(
        lambda x: jnp.tanh(x) * jnp.sin(x),
        lambda x: jnp.tanh(x) * jnp.cos(x),
        lambda x: jnp.tanh(x) * jnp.exp(x),
    )
    assert float(jnp.max(abs(h1(X) - exact(X)))) < 10 * EPS
    gop = cols(jnp.sinh, jnp.cosh, jnp.tanh)
    g = Tech.from_function(gop)
    h = f * g
    exact = cols(
        lambda x: jnp.sinh(x) * jnp.sin(x),
        lambda x: jnp.cosh(x) * jnp.cos(x),
        lambda x: jnp.tanh(x) * jnp.exp(x),
    )
    assert float(jnp.max(abs(h(X) - exact(X)))) < 10 * EPS
    g = Tech.from_function(cols(jnp.sinh, jnp.cosh))
    with pytest.raises(ValueError) as exc:
        f * g
    assert exc.value.identifier == "CHEBFUN:CHEBTECH:times:dim2"


def test_special_16_22_25(Tech):
    op = lambda x: jnp.sinh(x * jnp.exp(2j * jnp.pi / 6))
    f = Tech.from_function(op)
    function(f, op, f, op)
    function(f, op, f.conj(), lambda x: jnp.conj(op(x)), True)
    op = lambda x: jnp.exp(x) - 1
    f = Tech.from_function(op)
    function(f, op, f, op, True)
    gop = lambda x: 1 / (1 + x * x)
    g = Tech.from_function(gop)
    h1 = f * g
    h2 = Tech.from_function(lambda x: op(x) * gop(x)).prolong(len(h1.coeffs))
    assert h1.coeffs.shape == h2.coeffs.shape
    assert ni(h1.coeffs - h2.coeffs) < 50 * EPS
    f = Tech.from_function(lambda x: x)
    f8 = Tech.cell2mat([f] * 8)
    f20 = Tech.cell2mat([f8, f8, f, f, f, f])
    a = f8 * f8
    b = f20 * f20
    assert ni(a.coeffs[:, 0] - b.coeffs[:, 0]) < 50 * EPS
    xi = np.linspace(-1, 1, 4)
    y = Tech.from_function(lambda x: x)
    p1 = ((y - xi[1]) * (y - xi[2]) * (y - xi[3])) ** 2
    p = (y - xi[0]) * p1
    assert len(p.coeffs) == 8 and len(p1.coeffs) == 7
    print("DIRECT_AND_MODERATE_SHAPES", h1.coeffs.shape, a.coeffs.shape, b.coeffs.shape)


def test_unhappy_23_24(Tech):
    # Native warning off/on: suppress only these expected unhappy constructions.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = Tech.from_function(lambda x: jnp.cos(x + 1))
        g = Tech.from_function(lambda x: jnp.sqrt(x + 1))
        h = f * g
        assert not g.ishappy and not h.ishappy
        h = g * f
        assert not g.ishappy and not h.ishappy
    print("UNHAPPY_LENGTH", g.coeffs.shape)
