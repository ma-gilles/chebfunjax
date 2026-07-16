"""Port of MATLAB Chebfun tests/bndfun/test_feval.m (Opus 4.8).

Self-validating: every evaluation is checked against the analytic exact at
the SAME tolerance MATLAB uses.  Test points are our own deterministic grid
over the domain (the assertion ``error < tol`` holds at any point, so
MATLAB's RNG stream is not needed).

Provenance
----------
MATLAB source : tests/bndfun/test_feval.m
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
Z = np.exp(2 * np.pi * 1j / 6)
# 1000 deterministic points in the domain (MATLAB uses 1000 random ones).
XR = np.linspace(-2.0, 7.0, 1000)
X = jnp.asarray(XR)


def _bf(f, n=None):
    # xfail cases pass a small fixed n so a non-converging (array-valued /
    # singular) build stays fast; the assertion still fails as it should.
    return Bndfun.from_function(f, DOM, n=n)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestBndfunFeval:
    def test_spotcheck_exp(self):
        f = _bf(lambda x: jnp.exp(x) - 1)
        assert _ninf(f(X) - (np.exp(XR) - 1)) < 1e2 * f.vscale * EPS

    def test_spotcheck_runge(self):
        f = _bf(lambda x: 1.0 / (1 + x ** 2))
        assert _ninf(f(X) - 1.0 / (1 + XR ** 2)) < 1e2 * f.vscale * EPS

    def test_spotcheck_high_freq_cos(self):
        f = _bf(lambda x: jnp.cos(1e4 * x))
        assert _ninf(f(X) - np.cos(1e4 * XR)) < 1e5 * EPS

    def test_spotcheck_complex_sinh(self):
        f = _bf(lambda t: jnp.sinh(t * Z))
        assert _ninf(f(X) - np.sinh(XR * Z)) < 10 * f.vscale * EPS

    def test_row_vector_input(self):
        f = _bf(lambda t: jnp.sinh(t * Z))
        xrow = jnp.asarray(XR.reshape(1, 1000))
        err = f(xrow) - np.sinh(XR.reshape(1, 1000) * Z)
        assert err.shape == (1, 1000)
        assert _ninf(err) < 10 * f.vscale * EPS

    def test_matrix_input(self):
        f = _bf(lambda t: jnp.sinh(t * Z))
        xm = jnp.asarray(XR.reshape(100, 10))
        err = f(xm) - np.sinh(XR.reshape(100, 10) * Z)
        assert err.shape == (100, 10)
        assert _ninf(err) < 10 * f.vscale * EPS

    def test_3d_tensor_input(self):
        f = _bf(lambda t: jnp.sinh(t * Z))
        x3 = jnp.asarray(XR.reshape(10, 10, 10))
        err = f(x3) - np.sinh(XR.reshape(10, 10, 10) * Z)
        assert err.shape == (10, 10, 10)
        assert _ninf(err) < 10 * f.vscale * EPS

    def test_array_valued(self):
        # pass(8): array-valued [sin(x) x^2 exp(1i x)] on [-2, 7].
        # FIXED (Fable 5, Big-Three array-valued epic): (n, m) Bndfun with
        # adaptive construction; __call__ returns one column per component.
        f = _bf(lambda x: jnp.stack([jnp.sin(x), x ** 2, jnp.exp(1j * x)], axis=-1))
        exact = np.stack(
            [np.sin(XR), XR ** 2, np.exp(1j * XR)], axis=-1
        )
        assert _ninf(f(X) - exact) < 10 * f.vscale * EPS

    def test_array_valued_matrix_args(self):
        # pass(9): array-valued fun evaluated at a matrix argument.
        # FIXED (Fable 5, Big-Three array-valued epic): chebfunjax returns the
        # natural (rows, cols, m) layout; MATLAB flattens the m columns to
        # (rows, cols*m), but the values are identical, so we compare in native
        # layout against the stacked exact.
        f = _bf(lambda x: jnp.stack([jnp.sin(np.pi * x), jnp.cos(np.pi * x)], axis=-1))
        x2 = jnp.asarray(np.array([[-1.0, 0.0, 5.0], [-1.75, 0.5, 4.75]]))
        fx = np.asarray(f(x2))
        f_exact = np.stack(
            [np.sin(np.pi * np.asarray(x2)), np.cos(np.pi * np.asarray(x2))], axis=-1
        )
        assert fx.shape == (2, 3, 2)
        assert _ninf(fx - f_exact) < 1e2 * f.vscale * EPS

    @pytest.mark.xfail(
        reason="chebfunjax lacks singular (blowup) Bndfun: Bndfun.from_function "
        "has no `exponents`/blowup support for (x-a)^p factors."
    )
    def test_singular_function(self):
        pow_ = -0.5

        def op(x):
            return (x - DOM.a) ** pow_ * jnp.sin(x)

        f = _bf(op, n=17)
        exact = (XR - DOM.a) ** pow_ * np.sin(XR)
        assert _ninf(f(X) - exact) < 1e2 * EPS * float(np.max(np.abs(exact)))
