"""Literal tests/chebtech/test_plus.m, Chebfun7574c77:23 per technology.
First100 seed6178 MATLAB draws verified against native trreal1 capture. Scalar callback samples adapt Nx1 to(n,).
Default from_function options adapt native default techPref(eps,17,65537).
SciPy Airy Ai is a host input callback adapter. Existing port remains unchanged.
"""

import os

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

EPS = np.finfo(float).eps
X = jnp.asarray(2 * np.random.RandomState(6178).rand(100) - 1)
ALPHA = -0.194758928283640 + 0.075474485412665j
TECHS = (
    [{"1": Chebtech1, "2": Chebtech2}[os.environ["PLUS_TECH"]]]
    if "PLUS_TECH" in os.environ
    else [Chebtech1, Chebtech2]
)


@pytest.fixture(params=TECHS, ids=lambda cls: cls.__name__)
def Tech(request):
    return request.param


def ni(a):
    a = np.asarray(a)
    return float(np.max(np.sum(abs(a), axis=1))) if a.ndim == 2 else float(np.max(abs(a)))


def scalar(f, op):
    a = f + ALPHA
    b = ALPHA + f
    assert a.isequal(b)
    assert ni(a(X) - (op(X) + ALPHA)) <= 10 * float(jnp.max(a.vscale_columns * EPS))


def function(f, op, g, gop):
    a = f + g
    b = g + f
    assert a.isequal(b)
    assert ni(a(X) - (op(X) + gop(X))) <= 1e4 * float(jnp.max(a.vscale_columns * EPS))


def airy(x):
    return jnp.asarray(scipy.special.airy(1j * np.asarray(x))[0])


def cols(*fs):
    return lambda x: jnp.stack([f(x) for f in fs], axis=1)


def test_scalar_1_11(Tech):
    f = Tech.empty()
    g = Tech.from_function(lambda x: x)
    assert (f + f).isempty() and (f + g).isempty() and (g + f).isempty()
    scalar(Tech.from_function(jnp.sin), jnp.sin)
    z = jnp.zeros_like
    f = Tech.from_function(z)
    function(f, z, f, z)
    op = lambda x: jnp.exp(x) - 1
    f = Tech.from_function(op)
    for gop in [
        lambda x: 1 / (1 + x * x),
        lambda x: jnp.cos(1e4 * x),
        lambda x: jnp.sinh(x * jnp.exp(2j * jnp.pi / 6)),
    ]:
        function(f, op, Tech.from_function(gop), gop)


def test_array_12_19(Tech):
    z = cols(jnp.zeros_like, jnp.zeros_like, jnp.zeros_like)
    f = Tech.from_function(z)
    function(f, z, f, z)
    op = cols(jnp.sin, jnp.cos, jnp.exp)
    f = Tech.from_function(op)
    scalar(f, op)
    gop = cols(jnp.cosh, airy, jnp.sinh)
    function(f, op, Tech.from_function(gop), gop)
    g = Tech.from_function(jnp.sin)
    h = f + g  # noqa: F841 — Native18 tests successful evaluation.
    f = Tech.from_function(lambda x: x)
    g = Tech.from_function(lambda x: jnp.cos(x) - 1)
    h1 = f + g
    h2 = Tech.from_function(lambda x: x + jnp.cos(x) - 1)
    assert h1.coeffs.shape == h2.coeffs.shape
    assert ni(h1.coeffs - h2.coeffs) < 10 * EPS
    print("DIRECT_COEFF_SHAPE", h1.coeffs.shape)


def test_unhappy_20_21(Tech):
    f = Tech.from_function(lambda x: jnp.cos(x + 1))
    g = Tech.from_function(lambda x: jnp.sqrt(x + 1))
    h = f + g
    assert not g.ishappy and not h.ishappy
    h = g + f
    assert not g.ishappy and not h.ishappy
    print("UNHAPPY_LENGTH", g.coeffs.shape)


def test_expansion_22_23(Tech):
    op = cols(jnp.sin, jnp.cos, jnp.exp)
    f = Tech.from_function(op)
    g = f + jnp.array([1.0, 2.0, 3.0])
    err = g(X) - (op(X) + jnp.array([1.0, 2.0, 3.0]))
    assert float(jnp.max(abs(err))) < 10 * float(jnp.max(g.vscale_columns * EPS))
    f = Tech.from_function(jnp.sin)
    g = f + jnp.array([1.0, 2.0, 3.0])
    err = g(X) - (jnp.sin(X)[:, None] + jnp.array([1.0, 2.0, 3.0]))
    assert g.coeffs.shape[1] == 3
    assert float(jnp.max(abs(err))) < 10 * float(jnp.max(g.vscale_columns * EPS))
