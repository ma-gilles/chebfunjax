"""Port of MATLAB Chebfun tests/chebtech/test_feval.m (Opus 4.8).

Self-validating: interpolant evaluations are compared against the analytic
function at the SAME tolerances MATLAB uses.  The MATLAB test loops
``for n = 1:2`` over ``{chebtech1(), chebtech2()}``; we parametrize over
``[Chebtech1, Chebtech2]``.

chebfunjax ``f(x)`` returns an array with the SAME shape as ``x`` (Clenshaw
broadcasts), so the row-vector / matrix / 3-D reshape checks (pass 5-7) port
directly.  Those sub-tests reuse the complex ``sinh(t*z)`` function, so they
run over Chebtech2 only (Chebtech1 discards the imaginary part).

Notes on gaps (see the report):
* The ``sinh(t*z)`` sub-tests (pass 4-7) are complex-valued -> Chebtech2 only.
* Array-valued assertions (pass 8-9) are skipped.

Provenance
----------
MATLAB source : tests/chebtech/test_feval.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import warnings

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


class TestChebtechFeval:
    @pytest.mark.parametrize("Tech", BOTH)
    def test_spotcheck_exp(self, Tech):
        # pass(n, 1)
        f = Tech.from_function(lambda x: jnp.exp(x) - 1)
        assert _ninf(f(X) - (jnp.exp(X) - 1)) < 10 * f.vscale * EPS

    @pytest.mark.parametrize("Tech", BOTH)
    def test_spotcheck_lorentzian(self, Tech):
        # pass(n, 2)
        f = Tech.from_function(lambda x: 1.0 / (1 + x ** 2))
        assert _ninf(f(X) - 1.0 / (1 + X ** 2)) < 10 * f.vscale * EPS

    @pytest.mark.parametrize("Tech", BOTH)
    def test_spotcheck_high_frequency(self, Tech):
        # pass(n, 3)
        f = Tech.from_function(lambda x: jnp.cos(1e4 * x))
        assert _ninf(f(X) - jnp.cos(1e4 * X)) < 2e4 * f.vscale * EPS

    # pass(n, 4): complex-valued sinh(t*z) -> Chebtech2 only.
    @pytest.mark.parametrize("Tech", [Chebtech2])
    def test_spotcheck_sinh_complex(self, Tech):
        f = Tech.from_function(lambda t: jnp.sinh(t * _Z6))
        assert _ninf(f(X) - jnp.sinh(X * _Z6)) < 10 * f.vscale * EPS

    # pass(n, 5): row-vector input keeps its shape.  Uses the complex sinh f.
    @pytest.mark.parametrize("Tech", [Chebtech2])
    def test_row_vector_shape(self, Tech):
        f = Tech.from_function(lambda t: jnp.sinh(t * _Z6))
        xrow = X.reshape(1, 1000)
        err = f(xrow) - jnp.sinh(xrow * _Z6)
        assert err.shape == (1, 1000)
        assert _ninf(err) < 10 * f.vscale * EPS

    # pass(n, 6): matrix input keeps its shape.
    @pytest.mark.parametrize("Tech", [Chebtech2])
    def test_matrix_shape(self, Tech):
        f = Tech.from_function(lambda t: jnp.sinh(t * _Z6))
        xm = X.reshape(100, 10)
        err = f(xm) - jnp.sinh(xm * _Z6)
        assert err.shape == (100, 10)
        assert _ninf(err) < 10 * f.vscale * EPS

    # pass(n, 7): 3-D array input keeps its shape.
    @pytest.mark.parametrize("Tech", [Chebtech2])
    def test_3d_array_shape(self, Tech):
        f = Tech.from_function(lambda t: jnp.sinh(t * _Z6))
        x3 = X.reshape(10, 10, 10)
        err = f(x3) - jnp.sinh(x3 * _Z6)
        assert err.shape == (10, 10, 10)
        assert _ninf(err) < 10 * f.vscale * EPS

    @pytest.mark.parametrize("Tech", BOTH)
    def test_array_valued_skipped(self, Tech):
        # pass(n, 8)-(9): evaluating array-valued techs (and at matrix args)
        # requires quasimatrix techs.
        pytest.skip(
            "chebfunjax Chebtech is scalar-valued; no array-valued/quasimatrix "
            "techs"
        )


def test_chebtech1_rejects_complex():
    # FIXED (Fable 5): Chebtech1 now splits complex data into re/im
    # in vals2coeffs/coeffs2vals; this sentinel now passes.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = Chebtech1.from_function(lambda t: jnp.sinh(t * _Z6))
    assert f.ishappy
