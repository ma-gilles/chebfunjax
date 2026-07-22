"""Port of MATLAB Chebfun tests/chebfun3v/test_twocomponents.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3v/test_twocomponents.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np
import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.chebfun3d.chebfun3v import Chebfun3v

EPS = 2.220446049250313e-16
TOL = 1e5 * EPS

P = (np.pi / 6, 1.0, 1.0)


def _val(F, pt):
    return np.asarray(F(*pt))


class TestChebfun3vTwocomponents:
    def test_constructor_equivalence(self):
        f = Chebfun3.from_function(lambda x, y, z: x)
        F1 = Chebfun3v([f, f])
        F2 = Chebfun3v.from_functions(lambda x, y, z: x, lambda x, y, z: x)
        assert float((F1 - F2).norm()) < TOL

    def test_constructor_equivalence_domain(self):
        dom = (-1, 1, -1, 1, -1, 1)
        f = Chebfun3.from_function(lambda x, y, z: x, domain=dom)
        F1 = Chebfun3v([f, f])
        F2 = Chebfun3v.from_functions(lambda x, y, z: x, lambda x, y, z: x,
                                      domain=dom)
        assert float((F1 - F2).norm()) < TOL

    def test_plus(self):
        f = Chebfun3.from_function(lambda x, y, z: x)
        F1 = Chebfun3v([f, f])
        G = Chebfun3v([f, 2 * f])

        H = F1 + G
        assert np.linalg.norm(_val(H, (1, 1, 1)) - np.array([2, 3])) < TOL

        H = G + 1
        assert np.linalg.norm(_val(H, P) - np.pi / 6 * np.array([1, 2]) - 1) < TOL

        H = G + [1, 2]
        assert np.linalg.norm(
            _val(H, P) - np.pi / 6 * np.array([1, 2]) - np.array([1, 2])) < TOL

        H = 1 + G
        assert np.linalg.norm(_val(H, P) - np.pi / 6 * np.array([1, 2]) - 1) < TOL

        H = [1, 2] + G
        assert np.linalg.norm(
            _val(H, P) - np.pi / 6 * np.array([1, 2]) - np.array([1, 2])) < TOL

    def test_times(self):
        f = Chebfun3.from_function(lambda x, y, z: x)
        G = Chebfun3v([f, 2 * f])

        H = G * 1
        assert np.linalg.norm(_val(H, P) - _val(G, P)) < TOL

        H = G * [1, 2]
        assert np.linalg.norm(_val(H, P) - np.pi / 6 * np.array([1, 4])) < TOL

        H = 1 * G
        assert np.linalg.norm(_val(H, P) - _val(G, P)) < TOL

        H = [1, 2] * G
        assert np.linalg.norm(_val(H, P) - np.pi / 6 * np.array([1, 4])) < TOL

    def test_mtimes_inner_product(self):
        f = Chebfun3.from_function(lambda x, y, z: x)
        G = Chebfun3v([f, 2 * f])

        H = G * 1
        assert np.linalg.norm(_val(H, P) - _val(G, P)) < TOL

        # G' * [1 2]'  (inner product) == f + 4f
        Hin = G.ctranspose() @ [1, 2]
        assert float((Hin - (f + 4 * f)).norm()) < TOL

        H = 1 * G
        assert np.linalg.norm(_val(H, P) - _val(G, P)) < TOL

        # [1 2]' * G is a dimension mismatch in MATLAB.
        with pytest.raises(ValueError):
            [1, 2] @ G
