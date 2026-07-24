"""Port of MATLAB Chebfun tests/bndfun/test_sum.m (Opus 4.8).

Self-validating: every definite integral is compared to a closed-form exact
at the SAME tolerance MATLAB uses.

Provenance
----------
MATLAB source : tests/bndfun/test_sum.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun

EPS = float(np.finfo(np.float64).eps)
DOM = Domain((-2.0, 7.0))
A, B = -2.0, 7.0
XR = np.linspace(-2.0, 7.0, 1000)
X = jnp.asarray(XR)


def _bf(f, n=None):
    # xfail cases pass a small fixed n so a non-converging build stays fast.
    return Bndfun.from_function(f, DOM, n=n)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestBndfunSum:
    def test_spotcheck_exp(self):
        f = _bf(lambda x: jnp.exp(x) - 1)
        assert abs(float(f.sum()) - 1.087497823145222e3) < 10 * f.vscale * EPS

    def test_spotcheck_atan_deriv(self):
        f = _bf(lambda x: 1.0 / (1 + x ** 2))
        exact = np.arctan(-A) + np.arctan(B)
        assert abs(float(f.sum()) - exact) < 10 * f.vscale * EPS

    @pytest.mark.xfail(
        reason="Numerical: definite integral of cos(1e4*x) on [-2,7] has "
        "error ~3.3e-14, exceeding MATLAB's 100*vscale*eps ~= 2.2e-14 bound. "
        "chebfunjax needs ~45322 Chebyshev points here; the Clenshaw-Curtis "
        "quadrature accumulates ~150*eps rather than <100*eps.",
        strict=True,
    )
    def test_spotcheck_high_freq_cos(self):
        f = _bf(lambda x: jnp.cos(1e4 * x))
        exact = (np.sin(1e4 * B) - np.sin(1e4 * A)) / 1e4
        assert abs(float(f.sum()) - exact) < 100 * f.vscale * EPS

    def test_spotcheck_complex_sinh(self):
        z = np.exp(2 * np.pi * 1j / 6)
        f = _bf(lambda t: jnp.sinh(t * z))
        exact = (
            (np.cos(np.sqrt(3) - 1j) - np.cos((7 * np.sqrt(3)) / 2 - 7j / 2))
            * (np.sqrt(3) + 1j)
            * 1j
        ) / 2
        assert abs(complex(f.sum()) - exact) < 10 * f.vscale * EPS

    def test_linearity(self):
        a = 2.0
        b = -1j
        f = _bf(lambda x: x * jnp.sin(x ** 2) - 1)
        g = _bf(lambda x: jnp.exp(-((x / 10) ** 2)))
        tol_f = 10 * f.vscale * EPS
        tol_g = 10 * g.vscale * EPS
        lhs = complex((a * f + b * g).sum())
        rhs = a * complex(f.sum()) + b * complex(g.sum())
        assert abs(lhs - rhs) < max(tol_f, tol_g)

    def test_integration_by_parts(self):
        f = _bf(lambda x: x * jnp.sin(x ** 2) - 1)
        g = _bf(lambda x: jnp.exp(-((x / 10) ** 2)))
        df, dg = f.diff(), g.diff()
        fg = f * g
        gdf = g * df
        fdg = f * dg
        tol = max(10 * fg.vscale * EPS, 10 * gdf.vscale * EPS, 10 * fdg.vscale * EPS)
        lhs = complex(fdg.sum())
        rhs = (
            complex(fg(jnp.float64(B)))
            - complex(fg(jnp.float64(A)))
            - complex(gdf.sum())
        )
        assert abs(lhs - rhs) < tol

    def test_ftc_df(self):
        f = _bf(lambda x: x * jnp.sin(x ** 2) - 1)
        df = f.diff()
        tol = max(10 * df.vscale * EPS, 10 * f.vscale * EPS)
        err = complex(df.sum()) - (
            complex(f(jnp.float64(B))) - complex(f(jnp.float64(A)))
        )
        assert abs(err) < tol

    def test_ftc_dg(self):
        g = _bf(lambda x: jnp.exp(-((x / 10) ** 2)))
        dg = g.diff()
        tol = max(10 * dg.vscale * EPS, 10 * g.vscale * EPS)
        err = complex(dg.sum()) - (
            complex(g(jnp.float64(B))) - complex(g(jnp.float64(A)))
        )
        assert abs(err) < tol

    def test_array_valued(self):
        # pass(9): sum([sin x^2 exp(1i x)]) == per-column definite integrals.
        # FIXED (Fable 5, Big-Three array-valued epic): (n, m) Bndfun; sum
        # returns one integral per column.
        f = _bf(lambda x: jnp.stack([jnp.sin(x), x ** 2, jnp.exp(1j * x)], axis=-1))
        got = np.asarray(f.sum())
        exact = np.array(
            [
                np.cos(A) - np.cos(B),
                (B ** 3 - A ** 3) / 3,
                1j * (np.exp(1j * A) - np.exp(1j * B)),
            ]
        )
        assert _ninf(got - exact) < 10 * f.vscale * EPS

    # FIXED (Fable 5, Big-Three array-valued epic): Bndfun.sum(dim=2)
    # collapses the columns pointwise (MATLAB sum(f, 2)).
    def test_dim_option_array_valued(self):
        f = _bf(lambda x: jnp.stack([jnp.sin(x), x ** 2, jnp.exp(1j * x)], axis=-1))
        g = f.sum(dim=2)

        def h(x):
            return np.sin(x) + x ** 2 + np.exp(1j * x)

        assert _ninf(np.asarray(g(X)) - h(XR)) < 10 * g.vscale * EPS

    def test_dim_option_scalar(self):
        # pass(11): sum(h, 2) on a scalar fun is a no-op.
        h = _bf(jnp.cos)
        sumh2 = h.sum(dim=2)
        assert _ninf(h(X) - sumh2(X)) == 0.0

    def test_singular_function(self):
        pow_ = -0.5

        def op(x):
            return (x - A) ** pow_ * jnp.sin(x)

        # exponents=[pow 0]: algebraic blowup at the left endpoint.
        f = Bndfun.from_function(op, DOM, exponents=(pow_, 0.0))
        I_exact = -1.92205524578386613
        assert abs(complex(f.sum()) - I_exact) < 200 * EPS * abs(I_exact)
