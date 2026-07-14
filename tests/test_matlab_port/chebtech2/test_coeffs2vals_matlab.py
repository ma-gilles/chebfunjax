"""Port of MATLAB Chebfun tests/chebtech2/test_coeffs2vals.m (Opus 4.8).

Self-validating: each conversion is checked against the closed-form exact
values at 2nd-kind Chebyshev points at the SAME tolerance MATLAB uses
(100*eps).

The scalar/vector real, imaginary and general complex branches all map to
``Chebtech2.coeffs2vals`` and match the MATLAB exact values (verified;
Chebtech2 keeps complex128).  Array-input and symmetry cases require
array-valued techs (not implemented) — skipped.

Provenance
----------
MATLAB source : tests/chebtech2/test_coeffs2vals.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.tech.chebtech import Chebtech2

EPS = float(np.finfo(np.float64).eps)
TOL = 100 * EPS

_S2 = np.sqrt(2.0)
# Exact values for c = (1:5).'
_V_TRUE = np.array([3, -4 + _S2, 3, -4 - _S2, 15])
_C = np.arange(1.0, 6.0)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestChebtech2Coeffs2Vals:
    def test_single_coefficient(self):
        v = Chebtech2.coeffs2vals(jnp.asarray([_S2]))
        assert float(np.asarray(v)[0]) == _S2

    def test_real_branch(self):
        v = Chebtech2.coeffs2vals(jnp.asarray(_C))
        assert _ninf(np.asarray(v) - _V_TRUE) < TOL

    def test_real_no_imag(self):
        v = np.asarray(Chebtech2.coeffs2vals(jnp.asarray(_C)))
        assert not np.any(np.imag(v))

    def test_imaginary_branch(self):
        v = Chebtech2.coeffs2vals(jnp.asarray(1j * _C))
        assert _ninf(np.asarray(v) - 1j * _V_TRUE) < TOL

    def test_imaginary_no_real(self):
        v = np.asarray(Chebtech2.coeffs2vals(jnp.asarray(1j * _C)))
        assert not np.any(np.real(v))

    def test_general_branch(self):
        v = Chebtech2.coeffs2vals(jnp.asarray((1 + 1j) * _C))
        assert _ninf(np.asarray(v) - (1 + 1j) * _V_TRUE) < TOL

    # FIXED (Fable 5, Big-Three array-valued epic): pass 7-8 port now
    # that the transforms are column-wise with exact symmetry
    # enforcement (MATLAB @chebtech2/coeffs2vals.m lines 41-42, 64-66).
    def test_array_input(self):
        # pass(7): [c, flipud(c)] -> [vTrue, tmp .* vTrue]
        v = np.asarray(Chebtech2.coeffs2vals(
            jnp.asarray(np.column_stack([_C, _C[::-1]]))))
        tmp = np.ones_like(_V_TRUE)
        tmp[-2::-2] = -1.0
        assert _ninf(v[:, 0] - _V_TRUE) < TOL
        assert _ninf(v[:, 1] - tmp * _V_TRUE) < TOL

    def test_symmetry_preservation(self):
        # pass(8): odd coeffs exactly zero -> values EXACTLY even;
        # even coeffs exactly zero -> values EXACTLY odd.  (MATLAB's
        # file calls chebtech1.coeffs2vals here -- a copy-paste slip;
        # the property must hold for chebtech2 as well.)
        c = np.kron(np.ones((10, 1)), np.eye(2))
        v = np.asarray(Chebtech2.coeffs2vals(jnp.asarray(c)))
        assert float(np.max(np.abs(v[:, 0] - v[::-1, 0]))) == 0.0
        assert float(np.max(np.abs(v[:, 1] + v[::-1, 1]))) == 0.0
