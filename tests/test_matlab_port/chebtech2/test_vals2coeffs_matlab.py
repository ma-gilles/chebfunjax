"""Port of MATLAB Chebfun tests/chebtech2/test_vals2coeffs.m (Opus 4.8).

Self-validating: each conversion is checked against the closed-form exact
Chebyshev coefficients at the SAME tolerance MATLAB uses (100*eps).

The scalar/vector real, imaginary and general complex branches all map to
``Chebtech2.vals2coeffs`` and match the MATLAB exact values (verified;
Chebtech2 keeps complex128).  Array-input and symmetry cases require
array-valued techs (not implemented) — skipped.

Provenance
----------
MATLAB source : tests/chebtech2/test_vals2coeffs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.tech.chebtech import Chebtech2

EPS = float(np.finfo(np.float64).eps)
TOL = 100 * EPS

_S2 = np.sqrt(2.0)
# Exact coefficients for v = (1:5).'
_C_TRUE = np.array([3, 1 + 1 / _S2, 0, 1 - 1 / _S2, 0])
_V = np.arange(1.0, 6.0)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestChebtech2Vals2Coeffs:
    def test_single_value(self):
        c = Chebtech2.vals2coeffs(jnp.asarray([_S2]))
        assert float(np.asarray(c)[0]) == _S2

    def test_real_branch(self):
        c = Chebtech2.vals2coeffs(jnp.asarray(_V))
        assert _ninf(np.asarray(c) - _C_TRUE) < TOL

    def test_real_no_imag(self):
        c = np.asarray(Chebtech2.vals2coeffs(jnp.asarray(_V)))
        assert not np.any(np.imag(c))

    def test_imaginary_branch(self):
        c = Chebtech2.vals2coeffs(jnp.asarray(1j * _V))
        assert _ninf(np.asarray(c) - 1j * _C_TRUE) < TOL

    def test_imaginary_no_real(self):
        c = np.asarray(Chebtech2.vals2coeffs(jnp.asarray(1j * _V)))
        assert not np.any(np.real(c))

    def test_general_branch(self):
        c = Chebtech2.vals2coeffs(jnp.asarray((1 + 1j) * _V))
        assert _ninf(np.asarray(c) - (1 + 1j) * _C_TRUE) < TOL

    # FIXED (Fable 5, Big-Three array-valued epic): pass 7-8 port now
    # that the transforms are column-wise with exact symmetry
    # enforcement (MATLAB @chebtech2/vals2coeffs.m lines 41-43).
    def test_array_input(self):
        # pass(7): [v, flipud(v)] -> [cTrue, (-1)^k .* cTrue]
        c = np.asarray(Chebtech2.vals2coeffs(
            jnp.asarray(np.column_stack([_V, _V[::-1]]))))
        tmp = np.ones_like(_C_TRUE)
        tmp[1::2] = -1.0
        assert _ninf(c[:, 0] - _C_TRUE) < TOL
        assert _ninf(c[:, 1] - tmp * _C_TRUE) < TOL

    def test_symmetry_preservation(self):
        # pass(8): exactly-even column -> odd coeffs EXACTLY zero;
        # exactly-odd column -> even coeffs EXACTLY zero.
        v = np.kron(np.array([[1.0, -1.0], [1.0, 1.0]]),
                    np.ones((10, 1)))
        c = np.asarray(Chebtech2.vals2coeffs(jnp.asarray(v)))
        assert float(np.max(np.abs(c[1::2, 0]))) == 0.0
        assert float(np.max(np.abs(c[0::2, 1]))) == 0.0
