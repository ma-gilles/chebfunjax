"""Literal MATLAB padeapprox predicates with Python vector shape guards.

Provenance: Chebfun7574c77 tests/misc/test_padeapprox.m.
Source vectors are represented by1D arrays. Shape guards prevent first-entry
checks/broadcasting from hiding additional coefficients,poles or residues.
"""

import numpy as np

from chebfunjax.utils.ratapprox import padeapprox

TOL = 1e-10


def vector(value, length):
    value = np.asarray(value)
    assert value.shape == (length,), (value.shape, length)
    return value


class TestPadeapprox:
    def test_rational_with_known_poles(self):
        r, a, b, mu, nu, poles, residues = padeapprox(
            lambda x: (x**4 - 3) / ((x + 3.2) * (x - 2.2)), 10, 10
        )
        assert mu == 4 and nu == 2
        vector(a, 5)
        vector(b, 3)
        vector(residues, 2)
        p = np.sort_complex(vector(poles, 2))
        assert np.max(np.abs(p - np.array([-3.2, 2.2]))) < TOL

    def test_coefficient_input(self):
        r, a, b, mu, nu, poles, residues = padeapprox(np.array([1.0, 1.0j]), 0, 1)
        a = vector(a, 1)
        b = vector(b, 2)
        vector(poles, 1)
        vector(residues, 1)
        assert abs(a[0] - 1) < 1e-10
        assert abs(b[1] - (-1j)) < TOL

    def test_degenerate_with_tolerance(self):
        r, a, b, mu, nu, poles, residues = padeapprox(lambda x: x / (1 - x), 5, 6, r=0.5)
        assert mu == 1 and nu == 1
        a = vector(a, 2)
        b = vector(b, 2)
        poles = vector(poles, 1)
        residues = vector(residues, 1)
        assert np.linalg.norm(a - np.array([0.0, 1.0]), ord=2) < TOL
        assert np.linalg.norm(b - np.array([1.0, -1.0]), ord=2) < TOL
        assert bool(np.all(np.abs(poles - 1) < TOL))
        assert bool(np.all(np.abs(residues + 1) < TOL))
