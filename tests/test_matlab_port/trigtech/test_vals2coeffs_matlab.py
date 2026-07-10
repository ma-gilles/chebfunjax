"""Port of MATLAB Chebfun tests/trigtech/test_vals2coeffs.m (Opus 4.8).

Static transform from equispaced values to Fourier coefficients.  The
reference coefficient vectors are the exact analytic Fourier coefficients
in descending-wavenumber order (c_{-M}, ..., c_0, ..., c_M), which is the
same ordering chebfunjax uses.

Provenance
----------
MATLAB source : tests/trigtech/test_vals2coeffs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.trigtech import Trigtech, trigpts

EPS = float(np.finfo(np.float64).eps)
TOL = 100 * EPS


def _v2c(v):
    return Trigtech.vals2coeffs(v)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestTrigtechVals2Coeffs:
    def test_single_value(self):
        vals = jnp.array([np.sqrt(2.0)])
        c = _v2c(vals)
        assert _ninf(c - np.sqrt(2.0)) == 0.0

    # Odd case: f = 1 + cos(2 pi x), n = 5, cTrue = [.5 0 1 0 .5]
    def test_odd_real(self):
        x = trigpts(5)
        vals = 1 + jnp.cos(2 * jnp.pi * x)
        cTrue = jnp.array([0.5, 0, 1, 0, 0.5], dtype=jnp.complex128)
        assert _ninf(_v2c(vals) - cTrue) < TOL

    def test_odd_imag(self):
        x = trigpts(5)
        vals = 1 + jnp.cos(2 * jnp.pi * x)
        cTrue = jnp.array([0.5, 0, 1, 0, 0.5], dtype=jnp.complex128)
        assert _ninf(_v2c(1j * vals) - 1j * cTrue) < TOL

    def test_odd_general(self):
        x = trigpts(5)
        vals = 1 + jnp.cos(2 * jnp.pi * x)
        cTrue = jnp.array([0.5, 0, 1, 0, 0.5], dtype=jnp.complex128)
        assert _ninf(_v2c((1 + 1j) * vals) - (1 + 1j) * cTrue) < TOL

    # Even case: f = 1 + sin(2 pi x) + cos(3 pi x), n = 6, cTrue = [1 .5i 0 1 0 -.5i]
    def test_even_real(self):
        x = trigpts(6)
        vals = 1 + jnp.sin(2 * jnp.pi * x) + jnp.cos(3 * jnp.pi * x)
        cTrue = jnp.array([1, 0.5j, 0, 1, 0, -0.5j], dtype=jnp.complex128)
        assert _ninf(_v2c(vals) - cTrue) < TOL

    def test_even_imag(self):
        x = trigpts(6)
        vals = 1 + jnp.sin(2 * jnp.pi * x) + jnp.cos(3 * jnp.pi * x)
        cTrue = jnp.array([1, 0.5j, 0, 1, 0, -0.5j], dtype=jnp.complex128)
        assert _ninf(_v2c(1j * vals) - 1j * cTrue) < TOL

    def test_even_general(self):
        x = trigpts(6)
        vals = 1 + jnp.sin(2 * jnp.pi * x) + jnp.cos(3 * jnp.pi * x)
        cTrue = jnp.array([1, 0.5j, 0, 1, 0, -0.5j], dtype=jnp.complex128)
        assert _ninf(_v2c((1 + 1j) * vals) - (1 + 1j) * cTrue) < TOL

    @pytest.mark.xfail(reason="chebfunjax vals2coeffs is 1-D only; no multi-column array input")
    def test_odd_array_input(self):
        raise AssertionError("array (multi-column) vals2coeffs not implemented")

    @pytest.mark.xfail(reason="chebfunjax vals2coeffs is 1-D only; no multi-column array input")
    def test_even_array_input(self):
        raise AssertionError("array (multi-column) vals2coeffs not implemented")

    @pytest.mark.xfail(reason="chebfunjax vals2coeffs is 1-D only; no multi-column array input")
    def test_symmetry_array(self):
        raise AssertionError("array (multi-column) vals2coeffs not implemented")
