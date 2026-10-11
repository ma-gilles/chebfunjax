"""All6 literal tests/chebfun/test_atan2.m predicates, Chebfun7574c77.

Actual public x→arithmetic inputs, original100-site grids, both argument
orders, storedendpoint values/defaultL2 norm and strictnativevscale bounds.
Nativechecks do not suppress warnings; the old relaxed probe remains
separately supplemental below.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

A = -2.25 * np.pi
B = 2.25 * np.pi
XX = jnp.asarray(np.linspace(0.99 * A, 0.99 * B, 100))


class TestChebfunAtan2:
    def test_scalar_valued_both_orders(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = cj.chebfun(lambda x: 0.5 + jnp.sin(x) * jnp.exp(-0.1 * x**2), domain=(A, B))
            g = cj.chebfun(lambda x: jnp.cos(x) * (1 + x**2), domain=(A, B))
            h = cj.atan2(f, g)
        ff = np.asarray(f(XX))
        gg = np.asarray(g(XX))
        # pass(1)
        assert (
            np.max(np.abs(np.asarray(h(XX)) - np.arctan2(ff, gg))) < 1e3 * 10 * np.finfo(float).eps
        )

        # pass(3): reversed arguments
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h2 = cj.atan2(g, f)
        vs = max(np.max(np.abs(np.asarray(h2(XX)))), 1.0)
        assert (
            np.max(np.abs(np.asarray(h2(XX)) - np.arctan2(gg, ff))) < 1e4 * np.finfo(float).eps * vs
        )


import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.chebpref import ChebfunPref

EPS = float(np.finfo(np.float64).eps)


@pytest.fixture(scope="module")
def native_atan2_inputs():
    pref = ChebfunPref()
    x = cj.chebfun(lambda x: x, domain=(A, B), pref=pref)
    # Literal source public arithmetic, not direct callback replacements.
    f = 0.5 + x.sin() * (-0.1 * (x**2)).exp()
    g = x.cos() * (1 + x**2)
    xx = jnp.asarray(np.linspace(0.99 * A, 0.99 * B, 100))
    return f, g, xx


@pytest.mark.parametrize("reversed_order", [False, True])
def test_native_atan2_orders_and_endpoints(reversed_order, native_atan2_inputs):
    f, g, xx = native_atan2_inputs
    y, x = (g, f) if reversed_order else (f, g)
    h = cj.atan2(y, x)
    exact = jnp.arctan2(y(xx), x(xx))
    actual = h(xx)
    assert actual.shape == exact.shape == (100,)
    tol = 1000 * EPS * float(h.vscale) if reversed_order else 10 * EPS
    bound = tol if reversed_order else 100 * tol
    error = float(jnp.linalg.norm(actual - exact, ord=jnp.inf))
    ends = jnp.asarray((h.domain.breakpoints[0], h.domain.breakpoints[-1]))
    expected_ends = jnp.arctan2(y(ends), x(ends))
    stored_ends = h.point_values[jnp.asarray([0, -1])]
    assert expected_ends.shape == stored_ends.shape == (2,)
    endpoint_error = float(jnp.linalg.norm(expected_ends - stored_ends))
    endpoint_bound = tol if reversed_order else 2 * tol
    print(
        {
            "clauses": [3, 4] if reversed_order else [1, 2],
            "error": error,
            "bound": bound,
            "endpoint_error_L2": endpoint_error,
            "endpoint_bound": endpoint_bound,
            "vscale": float(h.vscale),
        },
        flush=True,
    )
    assert error < bound
    assert endpoint_error < endpoint_bound


@pytest.fixture(scope="module")
def native_atan2_zero_inputs():
    x = cj.chebfun("x")
    return 0 * x, (4 * x).sin()


@pytest.mark.parametrize("reversed_order", [False, True])
def test_native_atan2_zero_fun(reversed_order, native_atan2_zero_inputs):
    f, g = native_atan2_zero_inputs
    y, x = (g, f) if reversed_order else (f, g)
    h = cj.atan2(y, x)
    xx = jnp.asarray(np.linspace(-0.99, 0.99, 100))
    actual = h(xx)
    exact = jnp.arctan2(y(xx), x(xx))
    assert actual.shape == exact.shape == (100,)
    error = float(jnp.linalg.norm(actual - exact, ord=jnp.inf))
    bound = 10 * float(h.vscale) * EPS
    print(
        {
            "clause": 6 if reversed_order else 5,
            "error": error,
            "bound": bound,
            "vscale": float(h.vscale),
        },
        flush=True,
    )
    assert error <= bound


def test_atan2_source_pref_and_error_guards():
    """Additional source API checks, separate from native six predicates."""
    x = cj.chebfun("x")
    one = 0 * x + 1
    pref = ChebfunPref()
    h = cj.atan2(x, one, pref)
    sites = jnp.asarray([-0.5, 0, 0.5])
    assert float(jnp.max(jnp.abs(h(sites) - jnp.arctan2(sites, 1)))) < 100 * EPS
    with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN:atan2:real"):
        cj.atan2(1j * x, one, pref)
    with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN:atan2:dim"):
        cj.atan2(Chebfun.horzcat(x, x), one, pref)
