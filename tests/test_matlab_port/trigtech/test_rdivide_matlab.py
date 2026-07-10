"""Port of MATLAB Chebfun tests/trigtech/test_rdivide.m (Opus 4.8).

Pointwise division (./).  Division of two resolved real functions and
scalar/function division with a real result match direct evaluation.
Gaps: complex-scalar division (is_real not cleared / raises), division by
zero (needs isnan), and all array-valued / error-condition cases.

Provenance
----------
MATLAB source : tests/trigtech/test_rdivide.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.trigtech import Trigtech

EPS = float(np.finfo(np.float64).eps)
X = jnp.asarray(np.linspace(-1.0, 1.0, 10, endpoint=False))
XX = jnp.asarray(np.linspace(-1.0, 1.0, 10))


def _tt(f):
    return Trigtech.from_function(f)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestTrigtechRdivide:
    def test_scalar_over_function(self):
        # 2 ./ exp(cos(pi x)) : real scalar over real function.
        f = _tt(lambda x: jnp.exp(jnp.cos(jnp.pi * x)))
        g = 2.0 / f
        exact = 2.0 / jnp.exp(jnp.cos(jnp.pi * X))
        assert _ninf(g(X) - exact) < 100 * g.vscale * EPS

    def test_function_over_function_expcos(self):
        g = _tt(lambda x: jnp.exp(jnp.cos(20 * jnp.pi * x)))
        f = _tt(lambda x: jnp.exp(jnp.cos(20 * jnp.pi * x)) - 1)
        h = f / g
        exact = (jnp.exp(jnp.cos(20 * jnp.pi * X)) - 1) / jnp.exp(jnp.cos(20 * jnp.pi * X))
        assert _ninf(h(X) - exact) < 1e3 * h.vscale * EPS

    def test_function_over_function_cos(self):
        g = _tt(lambda x: jnp.exp(jnp.cos(20 * jnp.pi * x)))
        f = _tt(lambda x: jnp.cos(1e3 * jnp.pi * x))
        h = f / g
        exact = jnp.cos(1e3 * jnp.pi * X) / jnp.exp(jnp.cos(20 * jnp.pi * X))
        assert _ninf(h(X) - exact) < 1e3 * h.vscale * EPS

    def test_direct_construction_matches(self):
        # sin(10 pi x) ./ exp(cos(pi x)) : real.
        f = _tt(lambda x: jnp.sin(10 * jnp.pi * x))
        g = _tt(lambda x: jnp.exp(jnp.cos(jnp.pi * x)))
        h1 = f / g
        h2 = _tt(lambda x: jnp.sin(10 * jnp.pi * x) / jnp.exp(jnp.cos(jnp.pi * x)))
        assert _ninf(h1(XX) - h2(XX)) < 100 * EPS

    @pytest.mark.xfail(
        reason="chebfunjax __truediv__ by a complex scalar does not clear is_real, so "
        "the imaginary part of the quotient is dropped"
    )
    def test_scalar_division_complex(self):
        raise AssertionError("complex-scalar division drops imaginary part")

    @pytest.mark.xfail(
        reason="chebfunjax __rtruediv__ raises for a complex scalar over a real function "
        "(casts to float64) and does not clear is_real"
    )
    def test_complex_scalar_over_function(self):
        raise AssertionError("complex scalar / real function not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no isnan(); f/0 yields inf/nan coeffs silently")
    def test_scalar_division_by_zero_is_nan(self):
        raise AssertionError("isnan() not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued (multi-column) trigtech")
    def test_array_scalar_division(self):
        raise AssertionError("array-valued trigtech not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued (multi-column) trigtech")
    def test_array_division_by_zero(self):
        raise AssertionError("array-valued trigtech not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued (multi-column) trigtech")
    def test_array_division_by_row(self):
        raise AssertionError("array-valued trigtech not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued (multi-column) trigtech")
    def test_array_division_by_row_with_zero(self):
        raise AssertionError("array-valued trigtech not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued dimension-mismatch detection")
    def test_size_error_column_vector(self):
        raise AssertionError("array-valued trigtech not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued dimension-mismatch detection")
    def test_size_error_row_mismatch(self):
        raise AssertionError("array-valued trigtech not implemented")
