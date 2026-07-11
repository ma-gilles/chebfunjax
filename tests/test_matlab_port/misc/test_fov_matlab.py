"""Port of MATLAB Chebfun tests/misc/test_fov.m (Fable 5).

MATLAB fov returns a chebfun tracing the boundary; chebfunjax returns
(theta, boundary_points).  Same extremal assertions.

Provenance
----------
MATLAB source : tests/misc/test_fov.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.utils.fov import fov

EPS = float(np.finfo(np.float64).eps)


class TestFov:
    def test_two_by_two_reference(self):
        _, F = fov(np.array([[1, 2], [3, 2j]]))
        err = abs(float(np.max(np.asarray(F).real)) - 3.049509756796393)
        assert err < 1e2 * 7 * EPS

    def test_scalar_matrix(self):
        _, F = fov(np.array([[7.0]]))
        assert float(np.max(np.abs(np.asarray(F) - 7))) < 70 * EPS

    def test_hermitian_is_real_segment(self):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        _, F = fov(A)
        lam = np.linalg.eigvalsh(A)
        assert float(np.max(np.abs(np.asarray(F).imag))) < 1e-12
        assert abs(float(np.max(np.asarray(F).real)) - lam[-1]) < 1e-10
        assert abs(float(np.min(np.asarray(F).real)) - lam[0]) < 1e-10
