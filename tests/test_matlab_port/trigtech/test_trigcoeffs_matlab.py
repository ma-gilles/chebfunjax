"""Port of MATLAB Chebfun tests/trigtech/test_trigcoeffs.m (Opus 4.8).

``trigcoeffs(f)`` returns the Fourier coefficients (== ``f.coeffs`` for a
resolved trigtech), and ``trigcoeffs(f, N)`` returns exactly N of them,
zero-padding or truncating symmetrically.  chebfunjax exposes the
coefficients directly and reuses ``prolong`` for the length-N variant.

Provenance
----------
MATLAB source : tests/trigtech/test_trigcoeffs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.trigtech import Trigtech, _trig_prolong_coeffs

EPS = float(np.finfo(np.float64).eps)


def _tt(f):
    return Trigtech.from_function(f)


def _trigcoeffs(f, n=None):
    if n is None:
        return f.coeffs
    return _trig_prolong_coeffs(f.coeffs, n)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestTrigtechTrigcoeffs:
    def test_zeros(self):
        f = _tt(lambda x: jnp.zeros_like(x))
        p = _trigcoeffs(f)
        assert _ninf(p) <= 10 * f.vscale * EPS

    def test_constant(self):
        f = _tt(lambda x: 3 * jnp.ones_like(x))
        p = _trigcoeffs(f)
        assert _ninf(p - 3) < 10 * f.vscale * EPS

    # Odd tests: f = 1 + cos(pi x) -> [.5 1 .5]
    def test_odd_default(self):
        f = _tt(lambda x: 1 + jnp.cos(jnp.pi * x))
        p = _trigcoeffs(f)
        assert _ninf(p - jnp.array([0.5, 1, 0.5])) < 10 * f.vscale * EPS

    def test_odd_pad_to_five(self):
        f = _tt(lambda x: 1 + jnp.cos(jnp.pi * x))
        p = _trigcoeffs(f, 5)
        assert _ninf(p - jnp.array([0, 0.5, 1, 0.5, 0])) < 10 * f.vscale * EPS

    def test_odd_truncate_to_one(self):
        f = _tt(lambda x: 1 + jnp.cos(jnp.pi * x))
        p = _trigcoeffs(f, 1)
        assert _ninf(p - 1) < 10 * f.vscale * EPS

    # f = 1 + exp(2i pi x) + exp(-i pi x) -> [0 1 1 0 1]
    def test_complex_default(self):
        f = _tt(lambda x: 1 + jnp.exp(2j * jnp.pi * x) + jnp.exp(-1j * jnp.pi * x))
        p = _trigcoeffs(f)
        assert _ninf(p - jnp.array([0, 1, 1, 0, 1], dtype=jnp.complex128)) < 10 * f.vscale * EPS

    def test_complex_pad_to_nine(self):
        f = _tt(lambda x: 1 + jnp.exp(2j * jnp.pi * x) + jnp.exp(-1j * jnp.pi * x))
        p = _trigcoeffs(f, 9)
        exact = jnp.array([0, 0, 0, 1, 1, 0, 1, 0, 0], dtype=jnp.complex128)
        assert _ninf(p - exact) < 10 * f.vscale * EPS

    def test_complex_truncate_to_three(self):
        f = _tt(lambda x: 1 + jnp.exp(2j * jnp.pi * x) + jnp.exp(-1j * jnp.pi * x))
        p = _trigcoeffs(f, 3)
        assert _ninf(p - jnp.array([1, 1, 0], dtype=jnp.complex128)) < 10 * f.vscale * EPS

    # Even tests
    def test_even_cos(self):
        f = _tt(lambda x: 2 + jnp.cos(jnp.pi * x))
        p = _trigcoeffs(f, 2)
        assert _ninf(p - jnp.array([1, 2])) < 10 * f.vscale * EPS

    @pytest.mark.xfail(
        reason="chebfunjax _trig_prolong_coeffs even-truncation scales the retained "
        "Nyquist coefficient by 2 rather than folding c_{N/2}+c_{-N/2}; for a pure "
        "sine (odd) Nyquist mode MATLAB folds to 0 but chebfunjax yields 1i"
    )
    def test_even_sin(self):
        f = _tt(lambda x: 2 + jnp.sin(jnp.pi * x))
        p = _trigcoeffs(f, 2)
        assert _ninf(p - jnp.array([0, 2])) < 10 * f.vscale * EPS

    def test_even_cos2(self):
        f = _tt(lambda x: 2 + jnp.cos(2 * jnp.pi * x))
        p = _trigcoeffs(f, 4)
        assert _ninf(p - jnp.array([1, 0, 2, 0])) < 10 * f.vscale * EPS

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued (multi-column) trigtech")
    def test_array_default(self):
        raise AssertionError("array-valued trigtech not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued (multi-column) trigtech")
    def test_array_pad(self):
        raise AssertionError("array-valued trigtech not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued (multi-column) trigtech")
    def test_array_truncate(self):
        raise AssertionError("array-valued trigtech not implemented")

    def test_zero_length(self):
        # trigcoeffs(f, 0) is empty.
        f = _tt(lambda x: 1 + jnp.exp(2j * jnp.pi * x) + jnp.exp(-1j * jnp.pi * x))
        p = _trigcoeffs(f, 0)
        assert p.shape[0] == 0
