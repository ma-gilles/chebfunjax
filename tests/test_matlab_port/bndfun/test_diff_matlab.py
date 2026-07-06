"""Port of MATLAB Chebfun tests/bndfun/test_diff.m (Opus 4.8).

Self-validating: each operation is checked against an analytic exact at
the SAME tolerance MATLAB uses (multiples of vscale*eps).  No .mat
fixture needed — the reference is the closed-form derivative.

Provenance
----------
MATLAB source : tests/bndfun/test_diff.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun

EPS = float(np.finfo(np.float64).eps)
DOM = Domain((-2.0, 7.0))
# deterministic test points in the domain (analytic checks hold at any x)
X = jnp.asarray(np.linspace(-2.0, 7.0, 100))


def _bf(f):
    return Bndfun.from_function(f, DOM)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestBndfunDiff:
    def test_spotcheck_exp(self):
        f = _bf(lambda x: jnp.exp(x / 10) - x)
        df = f.diff()
        err = jnp.exp(X / 10) / 10 - 1 - df(X)
        assert _ninf(err) < 1e3 * f.vscale * EPS

    def test_spotcheck_atan(self):
        f = _bf(lambda x: jnp.arctan(x))
        df = f.diff()
        err = 1.0 / (1 + X ** 2) - df(X)
        assert _ninf(err) < 1e3 * f.vscale * EPS

    def test_spotcheck_sin(self):
        f = _bf(lambda x: jnp.sin(x))
        df = f.diff()
        err = jnp.cos(X) - df(X)
        assert _ninf(err) < 1e3 * f.vscale * EPS

    def test_diff_equals_direct_construction(self):
        # diff(0.5x - 0.0625 sin 8x) == sin(4x)^2
        f = _bf(lambda x: 0.5 * x - 0.0625 * jnp.sin(8 * x))
        df = _bf(lambda x: jnp.sin(4 * x) ** 2)
        err = f.diff()(X) - df(X)
        assert _ninf(err) < 1e4 * f.vscale * EPS

    def test_sum_rule(self):
        f = _bf(lambda x: x * jnp.sin(x ** 2) - 1)
        g = _bf(lambda x: jnp.exp(-x ** 2))
        df, dg = f.diff(), g.diff()
        tol = 10 * max(f.vscale, g.vscale, df.vscale, dg.vscale) * EPS
        # diff(f+g) - (df+dg)
        err = (f + g).diff()(X) - (df(X) + dg(X))
        assert _ninf(err) < max(tol, 1e3 * EPS)

    def test_product_rule(self):
        f = _bf(lambda x: x * jnp.sin(x ** 2) - 1)
        g = _bf(lambda x: jnp.exp(-x ** 2))
        df, dg = f.diff(), g.diff()
        tol = 10 * max(f.vscale, g.vscale, df.vscale, dg.vscale) * EPS
        err = (f * g).diff()(X) - (f(X) * dg(X) + g(X) * df(X))
        assert _ninf(err) < 1e2 * max(tol, 1e3 * EPS)

    def test_derivative_of_constant(self):
        const = _bf(lambda x: jnp.ones_like(x))
        dconst = const.diff()
        assert _ninf(dconst(X)) <= 1e2 * max(dconst.vscale, 1.0) * EPS

    def test_second_derivative(self):
        f = _bf(lambda x: x * jnp.arctan(x) - x - 0.5 * jnp.log(1 + x ** 2))
        df2 = f.diff(2)
        err = 1.0 / (1 + X ** 2) - df2(X)
        assert _ninf(err) < 1e7 * df2.vscale ** 2 * EPS

    def test_fourth_derivative(self):
        f = _bf(lambda x: jnp.sin(x))
        df4 = f.diff(4)
        err = _ninf(jnp.sin(X) - df4(X))
        assert err < 1e6 * 10 * df4.vscale * EPS

    def test_sixth_derivative_of_quintic_is_zero(self):
        f = _bf(lambda x: x ** 5 + 3 * x ** 3 - 2 * x ** 2 + 4)
        df6 = f.diff(6)
        assert _ninf(df6(X)) <= 1e3 * max(df6.vscale, 1.0) ** 2 * EPS
