"""Port of MATLAB Chebfun tests/singfun/test_flipud.m (Opus 4.8).

chebfunjax Singfun implements no ``flipud`` method (reflection x -> -x with
swapped exponents), so every assertion is xfailed (the call raises
``AttributeError``).  Analytic exacts from the MATLAB test are preserved.

Provenance
----------
MATLAB source : tests/singfun/test_flipud.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.fun.singfun import Singfun

EPS = float(np.finfo(np.float64).eps)

A = 0.64
B = -0.64
C = 1.28
D = -1.28

X = jnp.asarray(np.linspace(-0.99, 0.99, 100))
_REASON = "chebfunjax Singfun has no flipud() method"


def _sf(f, exps):
    return Singfun.from_function(f, exps)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestSingfunFlipud:
    @pytest.mark.xfail(reason=_REASON, strict=True)
    def test_frac_root_left(self):
        f = _sf(lambda x: (1 + x) ** A * jnp.exp(x), (A, 0.0))
        g = f.flipud()
        exact = (1 - X) ** A * jnp.exp(-X)
        assert _ninf(g(X) - exact) < 1e1 * EPS * _ninf(exact)

    @pytest.mark.xfail(reason=_REASON, strict=True)
    def test_frac_pole_left(self):
        f = _sf(lambda x: (1 + x) ** D * jnp.sin(x), (D, 0.0))
        g = f.flipud()
        exact = -(1 - X) ** D * jnp.sin(X)
        assert _ninf(g(X) - exact) < 10 * EPS * _ninf(exact)

    @pytest.mark.xfail(reason=_REASON, strict=True)
    def test_frac_root_right(self):
        f = _sf(lambda x: (1 - x) ** C * jnp.cos(x), (0.0, C))
        g = f.flipud()
        exact = (1 + X) ** C * jnp.cos(X)
        assert _ninf(g(X) - exact) < 1e1 * EPS * _ninf(exact)

    @pytest.mark.xfail(reason=_REASON, strict=True)
    def test_frac_pole_right(self):
        f = _sf(lambda x: (1 - x) ** B * (x ** 5), (0.0, B))
        g = f.flipud()
        exact = -(1 + X) ** B * (X ** 5)
        assert _ninf(g(X) - exact) < 1e1 * EPS * _ninf(exact)

    @pytest.mark.xfail(reason=_REASON, strict=True)
    def test_pole_and_root(self):
        f = _sf(lambda x: (1 + x) ** B * jnp.sin(x) * (1 - x) ** C, (B, C))
        g = f.flipud()
        exact = -(1 - X) ** B * jnp.sin(X) * (1 + X) ** C
        assert _ninf(g(X) - exact) < 1e1 * EPS * _ninf(exact)

    @pytest.mark.xfail(reason=_REASON, strict=True)
    def test_two_poles(self):
        f = _sf(lambda x: (1 + x) ** B * jnp.sin(2 * x) * (1 - x) ** B, (B, B))
        g = f.flipud()
        exact = -(1 - X) ** B * jnp.sin(2 * X) * (1 + X) ** B
        assert _ninf(g(X) - exact) < 1e1 * EPS * _ninf(exact)

    @pytest.mark.xfail(reason=_REASON, strict=True)
    def test_root_and_pole(self):
        f = _sf(lambda x: (1 + x) ** A * jnp.sin(x) * (1 - x) ** B, (A, B))
        g = f.flipud()
        exact = -(1 - X) ** A * jnp.sin(X) * (1 + X) ** B
        assert _ninf(g(X) - exact) < 5 * EPS * _ninf(exact)
