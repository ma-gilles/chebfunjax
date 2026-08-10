"""Port of MATLAB Chebfun tests/singfun/test_diff.m (Opus 4.8).

Self-validating: each derivative is checked against its analytic exact at
the SAME tolerance MATLAB uses (multiples of ``eps*norm(vals_exact, inf)``).
Test points are an interior grid ``[-0.99, 0.99]`` matching MATLAB's own
sampling window (``d = 2`` -> points in ``[-0.99, 0.99]``); the assertion
``err < tol`` holds at any interior point.

Provenance
----------
MATLAB source : tests/singfun/test_diff.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.fun.singfun import Singfun

EPS = float(np.finfo(np.float64).eps)

# The order of the exponents (as in the MATLAB test):
A = 0.56
B = -0.56
C = 1.28
D = -1.28

X = jnp.asarray(np.linspace(-0.99, 0.99, 100))


def _sf(f, exps):
    return Singfun.from_function(f, exps)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestSingfunDiff:
    def test_empty(self):
        # MATLAB: diff(singfun()) is empty.
        assert Singfun.empty().diff().isempty()

    def test_smooth_returns_nonsingfun(self):
        f = _sf(lambda x: jnp.sin(x), (0.0, 0.0))
        g = f.diff()
        assert not isinstance(g, Singfun)

    def test_frac_root_left(self):
        # fractional root at the left endpoint
        f = _sf(lambda x: (1 + x) ** A * jnp.exp(x), (A, 0.0))
        df = f.diff()
        exact = (1 + X) ** (A - 1) * (A + 1 + X) * jnp.exp(X)
        assert _ninf(df(X) - exact) < 1e2 * EPS * _ninf(exact)

    def test_frac_pole_left(self):
        # fractional pole at the left endpoint
        f = _sf(lambda x: (1 + x) ** D * jnp.sin(x), (D, 0.0))
        df = f.diff()
        exact = (1 + X) ** (D - 1) * (D * jnp.sin(X) + (1 + X) * jnp.cos(X))
        assert _ninf(df(X) - exact) < 1e2 * EPS * _ninf(exact)

    def test_frac_root_right(self):
        # fractional root at the right endpoint
        f = _sf(lambda x: (1 - x) ** C * jnp.cos(x), (0.0, C))
        df = f.diff()
        exact = -(1 - X) ** (C - 1) * (C * jnp.cos(X) + (1 - X) * jnp.sin(X))
        assert _ninf(df(X) - exact) < 1e2 * EPS * _ninf(exact)

    def test_frac_pole_right(self):
        # fractional pole at the right endpoint
        f = _sf(lambda x: (1 - x) ** B * (x ** 5), (0.0, B))
        df = f.diff()
        exact = (1 - X) ** (B - 1) * (5 - 5 * X - B * X) * (X ** 4)
        assert _ninf(df(X) - exact) < 1e2 * EPS * _ninf(exact)

    def test_pole_and_root(self):
        # combination of fractional pole (left) and fractional root (right)
        f = _sf(lambda x: (1 + x) ** B * jnp.sin(x) * (1 - x) ** C, (B, C))
        df = f.diff()
        exact = (
            jnp.cos(X) * (1 - X) ** C * (X + 1) ** B
            + B * jnp.sin(X) * (1 - X) ** C * (X + 1) ** (B - 1)
            - C * jnp.sin(X) * (1 - X) ** (C - 1) * (X + 1) ** B
        )
        assert _ninf(df(X) - exact) < 1e2 * EPS * _ninf(exact)

    def test_diff_vs_direct_construction(self):
        f = _sf(lambda x: (1 + x) ** B * jnp.sin(2 * x) * (1 - x) ** B, (B, B))
        df = f.diff()
        df_exact = _sf(
            lambda x: -2
            * (1 - x) ** (B - 1)
            * (x + 1) ** (B - 1)
            * (x ** 2 * jnp.cos(2 * x) - jnp.cos(2 * x) + B * x * jnp.sin(2 * x)),
            (B - 1, B - 1),
        )
        assert _ninf(df(X) - df_exact(X)) < 20 * EPS * _ninf(df_exact(X))

    def test_second_derivative(self):
        f = _sf(lambda x: (1 + x) ** A * jnp.sin(x) * (1 - x) ** B, (A, B))
        df2 = f.diff(2)
        exact = (
            2 * A * jnp.cos(X) * (1 - X) ** B * (X + 1) ** (A - 1)
            - jnp.sin(X) * (1 - X) ** B * (X + 1) ** A
            - 2 * B * jnp.cos(X) * (1 - X) ** (B - 1) * (X + 1) ** A
            + A * jnp.sin(X) * (A - 1) * (1 - X) ** B * (X + 1) ** (A - 2)
            - 2 * A * B * jnp.sin(X) * (1 - X) ** (B - 1) * (X + 1) ** (A - 1)
            + B * jnp.sin(X) * (B - 1) * (1 - X) ** (B - 2) * (X + 1) ** A
        )
        assert _ninf(df2(X) - exact) < 1e2 * EPS * _ninf(exact)
