"""Literal native RDIVIDE15x2, Chebfun7574c77.
Source6178 uniformfirst100 verified in trreal1; Nx1scalar samples represented1D.
Default constructors adapt native default techPref(eps,17,65537).
Python size errors expose the full native identifier in ValueError text.
"""

import os

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

EPS = np.finfo(float).eps
X = jnp.asarray(2 * np.random.RandomState(6178).rand(100) - 1)
ALPHA = -0.194758928283640 + 0.075474485412665j
BETA = -0.526634844879922 - 0.685484380523668j
TECHS = (
    [{"1": Chebtech1, "2": Chebtech2}[os.environ["RDIVIDE_TECH"]]]
    if "RDIVIDE_TECH" in os.environ
    else [Chebtech1, Chebtech2]
)


@pytest.fixture(params=TECHS, ids=lambda cls: cls.__name__)
def Tech(request):
    return request.param


def ni(a):
    a = np.asarray(a)
    return float(np.max(np.sum(abs(a), axis=1))) if a.ndim == 2 else float(np.max(abs(a)))


def scale(g):
    return float(jnp.max(g.vscale_columns * EPS))


def arrayop(x):
    return jnp.stack([jnp.sin(x), jnp.cos(x)], axis=1)


def test_small_1_9_11(Tech):
    f = Tech.from_function(jnp.sin)
    g = f / ALPHA
    assert ni(g(X) - jnp.sin(X) / ALPHA) < 10 * scale(g)
    g = f / 0
    assert g.isnan()
    f = Tech.from_function(arrayop)
    g = f / ALPHA
    assert ni(g(X) - arrayop(X) / ALPHA) < 10 * scale(g)
    g = f / 0
    assert g.isnan()
    g = f / jnp.array([ALPHA, BETA])
    exact = jnp.stack([jnp.sin(X) / ALPHA, jnp.cos(X) / BETA], axis=1)
    assert ni(g(X) - exact) < 10 * EPS
    g = f / jnp.array([ALPHA, 0])
    assert (
        g.isnan()
        and not bool(jnp.any(jnp.isnan(g.coeffs[:, 0])))
        and bool(jnp.all(jnp.isnan(g.coeffs[:, 1])))
    )
    f = Tech.from_function(jnp.exp)
    g = ALPHA / f
    assert ni(g(X) - ALPHA / jnp.exp(X)) < 10 * scale(g)
    g = Tech.from_function(jnp.exp)
    for op in [
        lambda x: jnp.exp(x) - 1,
        lambda x: 1 / (1 + x * x),
        lambda x: jnp.sinh(x * jnp.exp(2j * jnp.pi / 6)),
    ]:
        f = Tech.from_function(op)
        h = f / g
        assert ni(h(X) - op(X) / jnp.exp(X)) < 1e4 * scale(h)


def test_highfreq_10(Tech):
    g = Tech.from_function(jnp.exp)
    f = Tech.from_function(lambda x: jnp.cos(1e4 * x))
    h = f / g
    err = ni(h(X) - jnp.cos(1e4 * X) / jnp.exp(X))
    bound = 1e4 * scale(h)
    print("SOURCE10_ERROR_BOUND_LENGTHS", err, bound, f.coeffs.shape, h.coeffs.shape)
    assert err < bound


def test_errors_direct_12_15(Tech):
    f = Tech.from_function(jnp.sin)
    with pytest.raises(ValueError) as exc:
        f / jnp.array([[1.0], [2.0]])
    assert str(exc.value).split(": matrix")[0] == "CHEBFUN:CHEBTECH:rdivide:size"
    f = Tech.from_function(arrayop)
    with pytest.raises(ValueError) as exc:
        f / jnp.array([1.0, 2.0, 3.0])
    assert str(exc.value).split(": matrix")[0] == "CHEBFUN:CHEBTECH:rdivide:size"
    f = Tech.from_function(jnp.sin)
    h1 = f / ALPHA
    h2 = Tech.from_function(lambda x: jnp.sin(x) / ALPHA)
    assert h1.coeffs.shape == h2.coeffs.shape
    assert ni(h1.coeffs - h2.coeffs) < 100 * EPS
    g = Tech.from_function(jnp.exp)
    h1 = f / g
    h2 = Tech.from_function(lambda x: jnp.sin(x) / jnp.exp(x))
    n = max(len(h1.coeffs), len(h2.coeffs))
    h1 = h1.prolong(n)
    h2 = h2.prolong(n)
    assert ni(h1.coeffs - h2.coeffs) < 100 * EPS
    print("DIRECT_COEFF_SHAPES", h1.coeffs.shape, h2.coeffs.shape)
