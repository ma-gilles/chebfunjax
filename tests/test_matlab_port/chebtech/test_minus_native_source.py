"""Native MINUS21 per technology; Chebfun7574c77.
Uniform6178 first100 verified; scalar alpha bound to native uniform100 then normal2 continuation.
SciPy host Airy Ai callback; Nx1 scalar samples represented1D.
Default constructor options adapt native default techPref(eps,17,65537).
"""

import os

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

EPS = np.finfo(float).eps
X = jnp.asarray(2 * np.random.RandomState(6178).rand(100) - 1)
TECHS = (
    [{"1": Chebtech1, "2": Chebtech2}[os.environ["MINUS_TECH"]]]
    if "MINUS_TECH" in os.environ
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


def airy(x):
    return jnp.asarray(scipy.special.airy(1j * np.asarray(x))[0])


def scalar(f, op, alpha):
    a = f - alpha
    b = alpha - f
    assert a.isequal(-b)
    assert ni(a(X) - (op(X) - alpha)) <= 10 * float(jnp.max(a.vscale_columns * EPS))


def function(f, op, g, gop):
    a = f - g
    b = g - f
    assert a.isequal(-b)
    assert ni(a(X) - (op(X) - gop(X))) <= 1e4 * float(jnp.max(a.vscale_columns * EPS))


def test_scalar_nonalpha_1_4_11(Tech):
    f = Tech.empty()
    g = Tech.from_function(lambda x: x)
    assert (f - f).isempty() and (f - g).isempty() and (g - f).isempty()
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


def test_array_nonalpha_12_13_16_19(Tech):
    z = cols(jnp.zeros_like, jnp.zeros_like, jnp.zeros_like)
    f = Tech.from_function(z)
    function(f, z, f, z)
    op = cols(jnp.sin, jnp.cos, jnp.exp)
    f = Tech.from_function(op)
    gop = cols(jnp.cosh, airy, jnp.sinh)
    function(f, op, Tech.from_function(gop), gop)
    g = Tech.from_function(jnp.sin)
    h = f - g  # noqa: F841 — Native18 checks successful evaluation.
    f = Tech.from_function(lambda x: x)
    g = Tech.from_function(lambda x: jnp.cos(x) - 1)
    h1 = f - g
    h2 = Tech.from_function(lambda x: x - (jnp.cos(x) - 1))
    h3 = h1 - h2
    assert ni(h3.coeffs) < 10 * EPS
    print("DIRECT_RESIDUAL_SHAPE", h3.coeffs.shape)


def test_unhappy_20_21(Tech):
    f = Tech.from_function(lambda x: jnp.cos(x + 1))
    g = Tech.from_function(lambda x: jnp.sqrt(x + 1))
    h = f - g
    assert not g.ishappy and not h.ishappy
    h = g - f
    assert not g.ishappy and not h.ishappy
    print("UNHAPPY_LENGTH", g.coeffs.shape)


# Literal native scalar after100uniform draws; see ALPHA_CAPTURE.json.
ALPHA = -1.1710612811833176 - 0.11663804393033587j


def test_alpha_2_3_14_15(Tech):
    scalar(Tech.from_function(jnp.sin), jnp.sin, ALPHA)
    op = cols(jnp.sin, jnp.cos, jnp.exp)
    scalar(Tech.from_function(op), op, ALPHA)
