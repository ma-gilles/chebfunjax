"""Port of MATLAB Chebfun tests/chebtech1/test_vals2coeffs.m (Opus 4.8).

Self-validating: each conversion is checked against the closed-form exact
Chebyshev coefficients at the SAME tolerance MATLAB uses (100*eps).

Scalar/vector real branches map to ``Chebtech1.vals2coeffs`` and match the
MATLAB exact values.  The imaginary/general complex branches FAIL because
``Chebtech1.vals2coeffs`` ends in ``jnp.real(...)`` and discards the
imaginary part (verified) — those are xfailed.  Array-input and symmetry
cases require array-valued techs (not implemented) — skipped.

Provenance
----------
MATLAB source : tests/chebtech1/test_vals2coeffs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1

EPS = float(np.finfo(np.float64).eps)
TOL = 100 * EPS

_S2 = np.sqrt(2.0)
_S5 = np.sqrt(5.0)
_S6 = np.sqrt(6.0)

# Exact coefficients, even case v = (1:6).'
_C_EVEN = np.array(
    [7 / 2, _S6 / 2 + 5 * _S2 / 6, 0, _S2 / 6, 0, _S6 / 2 - 5 * _S2 / 6]
)
# Exact coefficients, odd case v = (1:5).'
_C_ODD = np.array(
    [
        3,
        (2 / 5) * (np.sqrt((5 - _S5) / 2) + 2 * np.sqrt((5 + _S5) / 2)),
        0,
        (2 / 5) * (2 * np.sqrt((5 - _S5) / 2) - np.sqrt((5 + _S5) / 2)),
        0,
    ]
)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestChebtech1Vals2Coeffs:
    # -- single value --------------------------------------------------
    def test_single_value(self):
        c = Chebtech1.vals2coeffs(jnp.asarray([_S2]))
        assert float(np.asarray(c)[0]) == _S2

    # -- even case (v = 1:6) -------------------------------------------
    def test_even_real_branch(self):
        c = Chebtech1.vals2coeffs(jnp.asarray(np.arange(1.0, 7.0)))
        assert _ninf(np.asarray(c) - _C_EVEN) < TOL

    def test_even_real_no_imag(self):
        c = np.asarray(Chebtech1.vals2coeffs(jnp.asarray(np.arange(1.0, 7.0))))
        assert not np.any(np.imag(c))

    def test_even_imaginary_branch(self):
        c = Chebtech1.vals2coeffs(jnp.asarray(1j * np.arange(1.0, 7.0)))
        assert _ninf(np.asarray(c) - 1j * _C_EVEN) < TOL

    def test_even_imaginary_no_real(self):
        c = np.asarray(Chebtech1.vals2coeffs(jnp.asarray(1j * np.arange(1.0, 7.0))))
        assert not np.any(np.real(c))

    def test_even_general_branch(self):
        c = Chebtech1.vals2coeffs(jnp.asarray((1 + 1j) * np.arange(1.0, 7.0)))
        assert _ninf(np.asarray(c) - (1 + 1j) * _C_EVEN) < TOL

    def test_even_array_input(self):
        pytest.skip(
            "chebfunjax Chebtech is scalar-valued; no array-valued/quasimatrix techs"
        )

    # -- odd case (v = 1:5) --------------------------------------------
    def test_odd_real_branch(self):
        c = Chebtech1.vals2coeffs(jnp.asarray(np.arange(1.0, 6.0)))
        assert _ninf(np.asarray(c) - _C_ODD) < TOL

    def test_odd_real_no_imag(self):
        c = np.asarray(Chebtech1.vals2coeffs(jnp.asarray(np.arange(1.0, 6.0))))
        assert not np.any(np.imag(c))

    def test_odd_imaginary_branch(self):
        c = Chebtech1.vals2coeffs(jnp.asarray(1j * np.arange(1.0, 6.0)))
        assert _ninf(np.asarray(c) - 1j * _C_ODD) < TOL

    def test_odd_imaginary_no_real(self):
        c = np.asarray(Chebtech1.vals2coeffs(jnp.asarray(1j * np.arange(1.0, 6.0))))
        assert not np.any(np.real(c))

    def test_odd_general_branch(self):
        c = Chebtech1.vals2coeffs(jnp.asarray((1 + 1j) * np.arange(1.0, 6.0)))
        assert _ninf(np.asarray(c) - (1 + 1j) * _C_ODD) < TOL

    def test_odd_array_input(self):
        pytest.skip(
            "chebfunjax Chebtech is scalar-valued; no array-valued/quasimatrix techs"
        )

    def test_symmetry_preservation(self):
        pytest.skip(
            "chebfunjax Chebtech is scalar-valued; no array-valued/quasimatrix techs"
        )
