"""Port of MATLAB Chebfun tests/chebtech1/test_coeffs2vals.m (Opus 4.8).

Self-validating: each conversion is checked against the closed-form exact
values at 1st-kind Chebyshev points at the SAME tolerance MATLAB uses
(100*eps).

The real scalar/vector branches map to ``Chebtech1.coeffs2vals`` and match
the MATLAB exact values.  The imaginary/general complex branches FAIL
because ``Chebtech1.coeffs2vals`` ends in ``jnp.real(...)`` and discards
the imaginary part (verified) — those are xfailed.  Array-input and
symmetry cases require array-valued techs (not implemented) — skipped.

Provenance
----------
MATLAB source : tests/chebtech1/test_coeffs2vals.m
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
_S3 = np.sqrt(3.0)
_S5 = np.sqrt(5.0)
_S6 = np.sqrt(6.0)

# Exact values, even case c = (6:-1:1).'
_V_EVEN = np.array(
    [
        -3 * _S6 / 2 - 5 / _S2 + 2 * _S3 + 7,
        4 - _S2 / 2,
        -3 * _S6 / 2 + 5 / _S2 - 2 * _S3 + 7,
        3 * _S6 / 2 - 5 / _S2 - 2 * _S3 + 7,
        4 + _S2 / 2,
        3 * _S6 / 2 + 5 / _S2 + 2 * _S3 + 7,
    ]
)
# Exact values, odd case c = (5:-1:1).'
_V_ODD = np.array(
    [
        11 / 2 + _S5 - 2 * np.sqrt((5 + _S5) / 2) - np.sqrt((5 - _S5) / 2),
        11 / 2 - _S5 - 2 * np.sqrt((5 - _S5) / 2) + np.sqrt((5 + _S5) / 2),
        3,
        11 / 2 - _S5 + 2 * np.sqrt((5 - _S5) / 2) - np.sqrt((5 + _S5) / 2),
        11 / 2 + _S5 + 2 * np.sqrt((5 + _S5) / 2) + np.sqrt((5 - _S5) / 2),
    ]
)

_C_EVEN = np.arange(6.0, 0.0, -1.0)
_C_ODD = np.arange(5.0, 0.0, -1.0)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestChebtech1Coeffs2Vals:
    # -- single coefficient --------------------------------------------
    def test_single_coefficient(self):
        v = Chebtech1.coeffs2vals(jnp.asarray([_S2]))
        assert float(np.asarray(v)[0]) == _S2

    # -- even case (c = 6:-1:1) ----------------------------------------
    def test_even_real_branch(self):
        v = Chebtech1.coeffs2vals(jnp.asarray(_C_EVEN))
        assert _ninf(np.asarray(v) - _V_EVEN) < TOL

    def test_even_real_no_imag(self):
        v = np.asarray(Chebtech1.coeffs2vals(jnp.asarray(_C_EVEN)))
        assert not np.any(np.imag(v))

    def test_even_imaginary_branch(self):
        v = Chebtech1.coeffs2vals(jnp.asarray(1j * _C_EVEN))
        assert _ninf(np.asarray(v) - 1j * _V_EVEN) < TOL

    def test_even_imaginary_no_real(self):
        v = np.asarray(Chebtech1.coeffs2vals(jnp.asarray(1j * _C_EVEN)))
        assert not np.any(np.real(v))

    def test_even_general_branch(self):
        v = Chebtech1.coeffs2vals(jnp.asarray((1 + 1j) * _C_EVEN))
        assert _ninf(np.asarray(v) - (1 + 1j) * _V_EVEN) < TOL

    def test_even_array_input(self):
        pytest.skip(
            "chebfunjax Chebtech is scalar-valued; no array-valued/quasimatrix techs"
        )

    # -- odd case (c = 5:-1:1) -----------------------------------------
    def test_odd_real_branch(self):
        v = Chebtech1.coeffs2vals(jnp.asarray(_C_ODD))
        assert _ninf(np.asarray(v) - _V_ODD) < TOL

    def test_odd_real_no_imag(self):
        v = np.asarray(Chebtech1.coeffs2vals(jnp.asarray(_C_ODD)))
        assert not np.any(np.imag(v))

    def test_odd_imaginary_branch(self):
        v = Chebtech1.coeffs2vals(jnp.asarray(1j * _C_ODD))
        assert _ninf(np.asarray(v) - 1j * _V_ODD) < TOL

    def test_odd_imaginary_no_real(self):
        v = np.asarray(Chebtech1.coeffs2vals(jnp.asarray(1j * _C_ODD)))
        assert not np.any(np.real(v))

    def test_odd_general_branch(self):
        v = Chebtech1.coeffs2vals(jnp.asarray((1 + 1j) * _C_ODD))
        assert _ninf(np.asarray(v) - (1 + 1j) * _V_ODD) < TOL

    def test_odd_array_input(self):
        pytest.skip(
            "chebfunjax Chebtech is scalar-valued; no array-valued/quasimatrix techs"
        )

    def test_symmetry_preservation(self):
        pytest.skip(
            "chebfunjax Chebtech is scalar-valued; no array-valued/quasimatrix techs"
        )
