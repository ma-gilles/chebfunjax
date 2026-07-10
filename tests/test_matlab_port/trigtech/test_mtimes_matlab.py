"""Port of MATLAB Chebfun tests/trigtech/test_mtimes.m (Opus 4.8).

mtimes (*) with a scalar is elementwise scaling (commutes); mtimes with a
matrix, and the associated error paths, require array-valued trigtechs
which chebfunjax lacks.

Provenance
----------
MATLAB source : tests/trigtech/test_mtimes.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.trigtech import Trigtech

EPS = float(np.finfo(np.float64).eps)
X = jnp.asarray(np.linspace(-1.0, 1.0, 100, endpoint=False))
rng = np.random.default_rng(6178)
ALPHA = complex(rng.standard_normal(), rng.standard_normal())


def _tt(f):
    return Trigtech.from_function(f)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestTrigtechMtimes:
    def test_scalar_mult_commutes(self):
        f = _tt(lambda x: jnp.sin(10 * jnp.pi * x))
        g1, g2 = ALPHA * f, f * ALPHA
        assert bool(jnp.all(g1.coeffs == g2.coeffs))

    def test_scalar_mult_value(self):
        f = _tt(lambda x: jnp.sin(10 * jnp.pi * x))
        g1 = ALPHA * f
        exact = ALPHA * jnp.sin(10 * jnp.pi * X)
        assert _ninf(g1(X) - exact) < 100 * g1.vscale * EPS

    def test_mult_by_zero(self):
        f = _tt(lambda x: jnp.sin(10 * jnp.pi * x))
        g = 0.0 * f
        assert bool(jnp.all(g.coeffs == 0))

    @pytest.mark.xfail(reason="chebfunjax lacks empty-argument arithmetic")
    def test_empty_arguments(self):
        raise AssertionError("empty trigtech arithmetic not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued (multi-column) trigtech")
    def test_array_scalar_mult_commutes(self):
        raise AssertionError("array-valued trigtech not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued (multi-column) trigtech")
    def test_array_scalar_mult_value(self):
        raise AssertionError("array-valued trigtech not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued (multi-column) trigtech")
    def test_array_mult_by_zero(self):
        raise AssertionError("array-valued trigtech not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued trigtech / matrix mtimes")
    def test_matrix_mult_real(self):
        raise AssertionError("array-valued trigtech not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued trigtech / matrix mtimes")
    def test_matrix_mult_complex_trigtech(self):
        raise AssertionError("array-valued trigtech not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued trigtech / matrix mtimes")
    def test_matrix_mult_complex_matrix(self):
        raise AssertionError("array-valued trigtech not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks the mtimes size-error paths (array-valued)")
    def test_error_nonscalar_double_times_trigtech(self):
        raise AssertionError("array-valued trigtech not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks the mtimes size-error paths (array-valued)")
    def test_error_trigtech_times_mismatched_double(self):
        raise AssertionError("array-valued trigtech not implemented")

    @pytest.mark.xfail(reason="chebfunjax does not forbid trigtech*trigtech via *")
    def test_error_trigtech_times_trigtech(self):
        raise AssertionError("mtimes trigtech*trigtech guard not implemented")

    @pytest.mark.xfail(reason="chebfunjax does not raise a typed error for unknown mtimes operands")
    def test_error_unknown_type(self):
        raise AssertionError("mtimes type guard not implemented")
