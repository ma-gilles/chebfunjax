"""Port of MATLAB Chebfun tests/trigtech/test_feval.m (Opus 4.8).

Spot-checks trigtech evaluation against analytic values; accuracy is only
expected on the order of the truncation level, so the criterion is a
multiple of vscale*eps.  Also checks shape preservation for row-vector,
matrix, and 3-D evaluation arguments.

Provenance
----------
MATLAB source : tests/trigtech/test_feval.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.trigtech import Trigtech

EPS = float(np.finfo(np.float64).eps)
X = jnp.asarray(np.linspace(-1.0, 1.0, 1000, endpoint=False))


def _tt(f):
    return Trigtech.from_function(f)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestTrigtechFeval:
    def test_ones(self):
        f = _tt(lambda x: jnp.ones_like(x))
        assert _ninf(f(X) - jnp.ones_like(X)) < 10 * f.vscale * EPS

    def test_sin(self):
        f = _tt(lambda x: jnp.sin(jnp.pi * x))
        assert _ninf(f(X) - jnp.sin(jnp.pi * X)) < 10 * f.vscale * EPS

    def test_exp_cos(self):
        f = _tt(lambda x: jnp.exp(jnp.cos(jnp.pi * x)) - 1)
        assert _ninf(f(X) - (jnp.exp(jnp.cos(jnp.pi * X)) - 1)) < 10 * f.vscale * EPS

    def test_cos_of_sin(self):
        f = _tt(lambda x: jnp.cos(100 * jnp.sin(jnp.pi * x)))
        assert _ninf(f(X) - jnp.cos(100 * jnp.sin(jnp.pi * X))) < 1e3 * f.vscale * EPS

    def test_complex_exponential(self):
        f = _tt(lambda x: jnp.exp(1j * jnp.pi * x))
        assert _ninf(f(X) - jnp.exp(1j * jnp.pi * X)) < 10 * f.vscale * EPS

    def test_even_expansion_real(self):
        # coeffs [2 0.25i 5 -0.25i] -> 2 cos(2 pi x) + 0.5 sin(pi x) + 5
        coeffs = jnp.array([2, 0.25j, 5, -0.25j], dtype=jnp.complex128)
        f = Trigtech.from_coeffs(coeffs)
        f_exact = 2 * jnp.cos(2 * jnp.pi * X) + 0.5 * jnp.sin(jnp.pi * X) + 5
        assert _ninf(f(X) - f_exact) < 10 * f.vscale * EPS

    def test_even_expansion_complex(self):
        coeffs = (
            jnp.array([2, 0.25j, 5, -0.25j]) + jnp.array([2j, 0.25, 5j, 0.25])
        ).astype(jnp.complex128)
        f = Trigtech.from_coeffs(coeffs)
        f_exact = (
            2 * (1 + 1j) * jnp.cos(2 * jnp.pi * X)
            + 0.5 * (jnp.cos(jnp.pi * X) + jnp.sin(jnp.pi * X))
            + 5 * (1 + 1j)
        )
        assert _ninf(f(X) - f_exact) < 10 * f.vscale * EPS

    def test_row_vector_input(self):
        coeffs = (
            jnp.array([2, 0.25j, 5, -0.25j]) + jnp.array([2j, 0.25, 5j, 0.25])
        ).astype(jnp.complex128)
        f = Trigtech.from_coeffs(coeffs)
        xrow = X.reshape(1, 1000)
        f_exact = (
            2 * (1 + 1j) * jnp.cos(2 * jnp.pi * xrow)
            + 0.5 * (jnp.cos(jnp.pi * xrow) + jnp.sin(jnp.pi * xrow))
            + 5 * (1 + 1j)
        )
        out = f(xrow)
        assert out.shape == (1, 1000)
        assert _ninf(out - f_exact) < 10 * f.vscale * EPS

    def test_matrix_input(self):
        coeffs = (
            jnp.array([2, 0.25j, 5, -0.25j]) + jnp.array([2j, 0.25, 5j, 0.25])
        ).astype(jnp.complex128)
        f = Trigtech.from_coeffs(coeffs)
        xm = X.reshape(100, 10)
        f_exact = (
            2 * (1 + 1j) * jnp.cos(2 * jnp.pi * xm)
            + 0.5 * (jnp.cos(jnp.pi * xm) + jnp.sin(jnp.pi * xm))
            + 5 * (1 + 1j)
        )
        out = f(xm)
        assert out.shape == (100, 10)
        assert _ninf(out - f_exact) < 10 * f.vscale * EPS

    def test_3d_input(self):
        coeffs = (
            jnp.array([2, 0.25j, 5, -0.25j]) + jnp.array([2j, 0.25, 5j, 0.25])
        ).astype(jnp.complex128)
        f = Trigtech.from_coeffs(coeffs)
        xm = X.reshape(10, 10, 10)
        f_exact = (
            2 * (1 + 1j) * jnp.cos(2 * jnp.pi * xm)
            + 0.5 * (jnp.cos(jnp.pi * xm) + jnp.sin(jnp.pi * xm))
            + 5 * (1 + 1j)
        )
        out = f(xm)
        assert out.shape == (10, 10, 10)
        assert _ninf(out - f_exact) < 10 * f.vscale * EPS

    def test_array_valued_feval(self):
        # pass(11): array-valued feval.
        # FIXED (Fable 5, Big-Three array-valued epic): trigtechs carry (n, m) coeffs.
        fun = lambda x: jnp.stack(
            [
                (2 + jnp.sin(jnp.pi * x)) * jnp.exp(1j * jnp.pi * x),
                -(2 + jnp.sin(jnp.pi * x)) * jnp.exp(1j * jnp.pi * x),
                2 + jnp.sin(jnp.pi * x),
            ],
            axis=-1,
        )
        f = Trigtech.from_function(fun)
        assert _ninf(f(X) - fun(X)) < 10 * f.vscale * EPS

    def test_array_valued_matrix_args(self):
        # pass(12): array-valued trigtech evaluated at a matrix argument.
        # FIXED (Fable 5, Big-Three array-valued epic).  chebfunjax returns
        # shape (x_rows, x_cols, m); MATLAB horzcat's the m columns into
        # (x_rows, x_cols*m) but the values are identical.
        fun = lambda x: jnp.stack(
            [jnp.sin(jnp.pi * x), jnp.cos(jnp.pi * x), jnp.exp(1j * jnp.pi * x)],
            axis=-1,
        )
        f = Trigtech.from_function(fun)
        x2 = jnp.array([[-1.0, 0.0, 1.0], [0.25, 0.5, 0.75]], dtype=jnp.float64)
        fx = f(x2)
        assert fx.shape == (2, 3, 3)
        assert _ninf(fx - fun(x2)) < 10 * f.vscale * EPS

    @pytest.mark.xfail(reason="chebfunjax lacks the chebfun 'trig'/'trunc' construction path")
    def test_chebfun_trig_trunc(self):
        raise AssertionError("chebfun trig/trunc not exercised here")

    @pytest.mark.xfail(reason="chebfunjax lacks the chebfun 'trig' constant construction path")
    def test_chebfun_trig_constant(self):
        raise AssertionError("chebfun trig constant not exercised here")
