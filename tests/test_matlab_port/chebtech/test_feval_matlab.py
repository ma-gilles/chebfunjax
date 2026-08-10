"""Port of MATLAB Chebfun tests/chebtech/test_feval.m (Fable 5).

Self-validating: interpolant evaluations are compared against the analytic
function at the SAME tolerances MATLAB uses.  The MATLAB test loops
``for n = 1:2`` over ``{chebtech1(), chebtech2()}``; we parametrize over
``[Chebtech1, Chebtech2]`` for every assertion, including the complex
``sinh(t*z)`` sub-tests (pass 4-7) -- Chebtech1 now carries complex data
through ``vals2coeffs``/``coeffs2vals`` rather than discarding the
imaginary part.

chebfunjax ``f(x)`` returns an array with the SAME shape as ``x`` (Clenshaw
broadcasts), so the row-vector / matrix / 3-D reshape checks (pass 5-7) port
directly, and the array-valued assertions (pass 8-9) port on genuine
``(n, m)`` coefficient matrices.  All nine MATLAB assertions are covered.

Deviation: MATLAB draws its 1000 evaluation points from ``seedRNG(7681)``;
here they are 1000 deterministic points spanning [-1, 1], which the
reshape checks (pass 5-7) reuse.

Provenance
----------
MATLAB source : tests/chebtech/test_feval.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

EPS = float(np.finfo(np.float64).eps)
# 1000 deterministic points in [-1, 1] (reshaped for the shape checks).
X = jnp.asarray(np.linspace(-1.0, 1.0, 1000))

BOTH = [Chebtech1, Chebtech2]

_Z6 = np.exp(2 * np.pi * 1j / 6)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


@pytest.mark.parametrize("Tech", BOTH)
class TestChebtechFeval:
    def test_spotcheck_exp(self, Tech):
        # pass(n, 1)
        f = Tech.from_function(lambda x: jnp.exp(x) - 1)
        assert _ninf(f(X) - (jnp.exp(X) - 1)) < 10 * f.vscale * EPS

    def test_spotcheck_lorentzian(self, Tech):
        # pass(n, 2)
        f = Tech.from_function(lambda x: 1.0 / (1 + x ** 2))
        assert _ninf(f(X) - 1.0 / (1 + X ** 2)) < 10 * f.vscale * EPS

    def test_spotcheck_high_frequency(self, Tech):
        # pass(n, 3)
        f = Tech.from_function(lambda x: jnp.cos(1e4 * x))
        assert _ninf(f(X) - jnp.cos(1e4 * X)) < 2e4 * f.vscale * EPS

    def test_spotcheck_sinh_complex(self, Tech):
        # pass(n, 4): complex-valued sinh(t*z), z = exp(2*pi*1i/6).
        f = Tech.from_function(lambda t: jnp.sinh(t * _Z6))
        assert f.coeffs.dtype == jnp.complex128
        assert _ninf(f(X) - jnp.sinh(X * _Z6)) < 10 * f.vscale * EPS

    def test_row_vector_shape(self, Tech):
        # pass(n, 5): row-vector input keeps its shape.
        f = Tech.from_function(lambda t: jnp.sinh(t * _Z6))
        xrow = X.reshape(1, 1000)
        err = f(xrow) - jnp.sinh(xrow * _Z6)
        assert err.shape == (1, 1000)
        assert _ninf(err) < 10 * f.vscale * EPS

    def test_matrix_shape(self, Tech):
        # pass(n, 6): matrix input keeps its shape.
        f = Tech.from_function(lambda t: jnp.sinh(t * _Z6))
        xm = X.reshape(100, 10)
        err = f(xm) - jnp.sinh(xm * _Z6)
        assert err.shape == (100, 10)
        assert _ninf(err) < 10 * f.vscale * EPS

    def test_3d_array_shape(self, Tech):
        # pass(n, 7): 3-D array input keeps its shape.
        f = Tech.from_function(lambda t: jnp.sinh(t * _Z6))
        x3 = X.reshape(10, 10, 10)
        err = f(x3) - jnp.sinh(x3 * _Z6)
        assert err.shape == (10, 10, 10)
        assert _ninf(err) < 10 * f.vscale * EPS

    def test_array_valued(self, Tech):
        # pass(n, 8): array-valued spot check [sin(x), x.^2, exp(1i*x)]
        f = Tech.from_function(
            lambda x: jnp.stack(
                [jnp.sin(x), x ** 2, jnp.exp(1j * x)], axis=-1))
        exact = jnp.stack(
            [jnp.sin(X), X ** 2, jnp.exp(1j * X)], axis=-1)
        assert _ninf(f(X) - exact) < 10 * f.vscale * EPS

    def test_array_valued_matrix_argument(self, Tech):
        # pass(n, 9): array-valued tech evaluated at a matrix argument.
        # MATLAB returns a (p, m*q) block layout [f1(x) f2(x) f3(x)];
        # chebfunjax returns (p, q, m) -- transpose to compare.
        f = Tech.from_function(
            lambda x: jnp.stack(
                [jnp.sin(jnp.pi * x), jnp.cos(jnp.pi * x),
                 jnp.exp(jnp.pi * x)], axis=-1))
        x2 = jnp.asarray([[-1.0, 0.0, 1.0], [0.25, 0.5, 0.75]])
        fx = np.asarray(f(x2))            # (2, 3, 3)
        assert fx.shape == (2, 3, 3)
        blocked = np.transpose(fx, (0, 2, 1)).reshape(2, 9)
        s2 = np.sqrt(2.0)
        f_exact = np.array([
            [0, 0, 0, -1, 1, -1, np.exp(-np.pi), 1, np.exp(np.pi)],
            [1 / s2, s2 / s2, 1 / s2, 1 / s2, 0, -1 / s2,
             np.exp(np.pi * 0.25), np.exp(np.pi * 0.5),
             np.exp(np.pi * 0.75)],
        ])
        assert np.max(np.abs(blocked - f_exact)) < 10 * f.vscale * EPS


def test_chebtech1_supports_complex():
    # Sentinel for the pass(n, 4)-(7) extension to Chebtech1: complex data
    # survives vals2coeffs/coeffs2vals instead of being silently realified.
    f = Chebtech1.from_function(lambda t: jnp.sinh(t * _Z6))
    assert f.ishappy
    assert f.coeffs.dtype == jnp.complex128
    assert _ninf(f(X) - jnp.sinh(X * _Z6)) < 10 * f.vscale * EPS
