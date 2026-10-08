"""All 15 original classicfun division clauses at their source bounds.

Provenance
----------
MATLAB source : tests/classicfun/test_rdivide.m, tests/seedRNG.m
Chebfun commit: 7574c77
Uniform sites use MT19937 seed 6178. The first 100 draws are independently
checked against captured MATLAB logical-test inputs; the remaining draws are
an input-generation adapter, not newly captured native output. MATLAB column
vectors map to 1-D arrays; array-valued infinity norms retain max-row-sum rules.
Negative real fractional powers explicitly use MATLAB's complex branch.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.fun.unbndfun import Unbndfun

EPS = np.finfo(np.float64).eps
DOM = Domain((-2.0, 7.0))
ALPHA = -0.194758928283640 + 0.075474485412665j
BETA = -0.526634844879922 - 0.685484380523668j


@pytest.fixture(scope="module")
def sites():
    rng = np.random.RandomState(6178)
    u = rng.rand(1100)
    fixture = Path(__file__).parents[1] / "chebfun/fixtures/logical_source_matlab.json"
    native = json.loads(fixture.read_text())
    np.testing.assert_array_equal(2 * u[:100] - 1, np.asarray(native["x"]).ravel())
    return jnp.asarray(9 * u[:1000] - 2), jnp.asarray(98 * u[1000:] + 2)


def _ninf(values):
    a = jnp.abs(jnp.asarray(values))
    return float(jnp.max(jnp.sum(a, axis=1) if a.ndim == 2 else a))


def _array_op(x):
    return jnp.stack((jnp.sin(x), jnp.cos(x)), axis=-1)


@pytest.mark.parametrize("array,clause", [(False, 1), (True, 3)])
def test_function_by_scalar(sites, array, clause):
    x, _ = sites
    op = _array_op if array else jnp.sin
    g = Bndfun.from_function(op, DOM) / ALPHA
    assert _ninf(g(x) - op(x) / ALPHA) < 10 * g.vscale * EPS, clause


@pytest.mark.parametrize("array,clause", [(False, 2), (True, 4)])
def test_zero_divisor(array, clause):
    g = Bndfun.from_function(_array_op if array else jnp.sin, DOM) / 0
    assert g.isnan(), clause


def test_row_divisor_clause5(sites):
    x, _ = sites
    g = Bndfun.from_function(_array_op, DOM) / jnp.asarray([ALPHA, BETA])
    exact = jnp.stack((jnp.sin(x) / ALPHA, jnp.cos(x) / BETA), axis=-1)
    assert _ninf(g(x) - exact) < 10 * g.vscale * EPS


def test_row_zero_clause6(sites):
    x, _ = sites
    g = Bndfun.from_function(_array_op, DOM) / jnp.asarray([ALPHA, 0])
    isn = jnp.isnan(g(x))
    assert g.isnan() and not bool(jnp.any(isn[:, 0])) and bool(jnp.all(isn[:, 1]))


def test_scalar_by_function_clause7(sites):
    x, _ = sites
    g = ALPHA / Bndfun.from_function(jnp.exp, DOM)
    assert _ninf(g(x) - ALPHA / jnp.exp(x)) < 1e4 * g.vscale * EPS


@pytest.mark.parametrize("clause", [8, 9, 10, 11])
def test_function_by_function(sites, clause):
    x, _ = sites
    ops = {
        8: lambda t: jnp.exp(t) - 1,
        9: lambda t: 1 / (1 + t ** 2),
        10: lambda t: jnp.cos(1e4 * t),
        11: lambda t: jnp.sinh(t * jnp.exp(2 * jnp.pi * 1j / 6)),
    }
    op = ops[clause]
    h = Bndfun.from_function(op, DOM) / Bndfun.from_function(jnp.exp, DOM)
    assert _ninf(h(x) - op(x) / jnp.exp(x)) < 1e5 * h.vscale * EPS


@pytest.mark.parametrize("clause", [12, 13])
def test_direct_construction(sites, clause):
    x, _ = sites
    f = Bndfun.from_function(jnp.sin, DOM)
    if clause == 12:
        h1 = f / ALPHA
        h2 = Bndfun.from_function(lambda t: jnp.sin(t) / ALPHA, DOM)
        factor = 10
    else:
        h1 = f / Bndfun.from_function(jnp.exp, DOM)
        h2 = Bndfun.from_function(lambda t: jnp.sin(t) / jnp.exp(t), DOM)
        factor = 5e3
    assert _ninf(h1(x) - h2(x)) < factor * h2.vscale * EPS


def test_singular_clause14(sites):
    x, _ = sites
    pow1, pow2 = -0.5, -0.3
    power = lambda t, p: jnp.asarray(t - 7, dtype=jnp.complex128) ** p
    f = Bndfun.from_function(lambda t: power(t, pow1) * jnp.sin(t), DOM,
                            exponents=(0, pow1))
    g = Bndfun.from_function(lambda t: power(t, pow2) * (jnp.cos(t) ** 2 + 1), DOM,
                            exponents=(0, pow2))
    exact = power(x, pow1 - pow2) * jnp.sin(x) / (jnp.cos(x) ** 2 + 1)
    assert _ninf((f / g)(x) - exact) < 100 * EPS * _ninf(exact)


def test_unbounded_clause15(sites):
    _, x = sites
    domain = Domain((2.0, float("inf")))
    f = Unbndfun.from_function(lambda t: jnp.exp(-t ** 2), domain)
    g = Unbndfun.from_function(lambda t: t ** 2, domain)
    exact = jnp.exp(-x ** 2) / x ** 2
    assert _ninf((f / g)(x) - exact) < 10 * EPS * f.vscale
