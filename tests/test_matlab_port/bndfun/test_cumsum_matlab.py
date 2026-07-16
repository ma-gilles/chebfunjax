"""Port of MATLAB Chebfun tests/bndfun/test_cumsum.m (Opus 4.8).

Self-validating: each antiderivative is compared to a closed-form exact.
Because ``cumsum`` fixes F(a)=0 while the analytic antiderivative may not,
MATLAB compares ``norm(diff(err), inf)`` (which cancels the constant offset)
and separately checks ``|F(a)|``.  We reproduce both, at the SAME tolerances.

Provenance
----------
MATLAB source : tests/bndfun/test_cumsum.m
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
A = -2.0
XR = np.linspace(-2.0, 7.0, 100)
X = jnp.asarray(XR)


def _bf(f, n=None):
    # xfail cases pass a small fixed n so a non-converging build stays fast.
    return Bndfun.from_function(f, DOM, n=n)


def _diff_ninf(err):
    return float(np.max(np.abs(np.diff(np.asarray(err)))))


class TestBndfunCumsum:
    def test_spotcheck_exp(self):
        f = _bf(lambda x: jnp.exp(x / 10) - 1)
        F = f.cumsum()
        err = np.asarray(F(X)) - (10 * np.exp(XR / 10) - XR)
        assert _diff_ninf(err) < 100 * f.vscale * EPS
        assert abs(float(F(jnp.float64(A)))) <= f.vscale * EPS

    def test_spotcheck_atan(self):
        f = _bf(lambda x: 1.0 / (1 + x ** 2))
        F = f.cumsum()
        err = np.asarray(F(X)) - np.arctan(XR)
        assert _diff_ninf(err) < 1e3 * f.vscale * EPS
        assert abs(float(F(jnp.float64(A)))) <= 1e1 * f.vscale * EPS

    def test_spotcheck_high_freq_cos(self):
        f = _bf(lambda x: jnp.cos(1e4 * x))
        F = f.cumsum()
        err = np.asarray(F(X)) - np.sin(1e4 * XR) / 1e4
        assert _diff_ninf(err) < 1e3 * f.vscale * EPS
        assert abs(float(F(jnp.float64(A)))) <= 1e1 * f.vscale * EPS

    def test_spotcheck_complex_sinh(self):
        z = np.exp(2 * np.pi * 1j / 6)
        f = _bf(lambda t: jnp.sinh(t * z))
        F = f.cumsum()
        err = np.asarray(F(X)) - np.cosh(XR * z) / z
        assert _diff_ninf(err) < 100 * f.vscale * EPS
        assert abs(complex(F(jnp.float64(A)))) <= f.vscale * EPS

    def test_cumsum_matches_direct_construction(self):
        f = _bf(lambda x: jnp.sin(4 * x) ** 2)
        F = _bf(lambda x: 0.5 * x - 0.0625 * jnp.sin(8 * x))
        G = f.cumsum()
        err = np.asarray((G - F)(X))
        assert _diff_ninf(err) < 1e2 * f.vscale * EPS
        assert abs(float(G(jnp.float64(A)))) < 1e2 * f.vscale * EPS

    def test_diff_of_cumsum_is_identity(self):
        f = _bf(lambda x: x * (x - 1) * jnp.sin(x))
        g = f.cumsum().diff()
        tol_f = 10 * f.vscale * EPS
        tol_g = 10 * g.vscale * EPS
        err = np.asarray(f(X)) - np.asarray(g(X))
        assert _diff_ninf(err) < 1e2 * max(tol_f, tol_g)

    def test_cumsum_of_diff_is_identity(self):
        f = _bf(lambda x: x * (x - 1) * jnp.sin(x))
        tol_f = 10 * f.vscale * EPS
        h = f.diff().cumsum()
        tol_h = 10 * h.vscale * EPS
        err = np.asarray(f(X)) - np.asarray(h(X))
        assert _diff_ninf(err) < 10 * max(tol_f, tol_h)
        assert abs(float(h(jnp.float64(A)))) < 10 * max(tol_f, tol_h)

    def test_array_valued(self):
        # pass(8): cumsum([sin x^2 exp(1i x)]) matches the analytic
        # antiderivative up to a per-column constant, with F(a) == 0.
        # FIXED (Fable 5, Big-Three array-valued epic): column-wise cumsum.
        # MATLAB's norm(diff(err), inf) differences down the SAMPLE axis
        # (axis=0), cancelling each column's constant of integration.
        f = _bf(lambda x: jnp.stack([jnp.sin(x), x ** 2, jnp.exp(1j * x)], axis=-1))
        F = f.cumsum()
        F_exact = _bf(
            lambda x: jnp.stack(
                [-jnp.cos(x), x ** 3 / 3, jnp.exp(1j * x) / 1j], axis=-1
            )
        )
        err = np.asarray(F(X)) - np.asarray(F_exact(X))
        assert float(np.max(np.abs(np.diff(err, axis=0)))) < 10 * f.vscale * EPS
        assert bool(np.all(np.abs(np.asarray(F(jnp.float64(A)))) < f.vscale * EPS))

    @pytest.mark.xfail(
        reason="chebfunjax lacks singular (blowup) Bndfun: (x-a)^-0.64 "
        "cannot be constructed via Bndfun.from_function."
    )
    def test_singular_function(self):
        pow_ = -0.64
        f = _bf(lambda x: (x - A) ** pow_, n=17)
        g = f.cumsum()
        exact = (XR - A) ** (pow_ + 1) / (pow_ + 1)
        err = np.asarray(g(X)) - exact
        assert float(np.max(np.abs(err))) < 1e2 * EPS * float(np.max(np.abs(exact)))
