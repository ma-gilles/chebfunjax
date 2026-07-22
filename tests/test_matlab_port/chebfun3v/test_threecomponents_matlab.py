"""Port of MATLAB Chebfun tests/chebfun3v/test_threecomponents.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3v/test_threecomponents.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.chebfun3d.chebfun3v import Chebfun3v

EPS = 2.220446049250313e-16
TOL = 1e5 * EPS

P = (np.pi / 6, 0.5, np.pi / 2)


def _val(F, pt):
    return np.asarray(F(*pt))


class TestChebfun3vThreecomponents:
    def test_constructor_equivalence(self):
        f = Chebfun3.from_function(lambda x, y, z: x)
        F1 = Chebfun3v([f, f, f])
        F2 = Chebfun3v.from_functions(lambda x, y, z: x, lambda x, y, z: x,
                                      lambda x, y, z: x)
        assert float((F1 - F2).norm()) < TOL

    def test_plus(self):
        f = Chebfun3.from_function(lambda x, y, z: x)
        F1 = Chebfun3v([f, f, f])
        G = Chebfun3v([f, 2 * f, 3 * f])

        H = F1 + G
        assert np.linalg.norm(_val(H, (1, 1, 1)) - np.array([2, 3, 4])) < TOL

        H = G + 1
        assert np.linalg.norm(
            _val(H, P) - np.pi / 6 * np.array([1, 2, 3]) - 1) < TOL

        H = G + [1, 2, 3]
        assert np.linalg.norm(
            _val(H, P) - np.pi / 6 * np.array([1, 2, 3])
            - np.array([1, 2, 3])) < TOL

        H = 1 + G
        assert np.linalg.norm(
            _val(H, P) - np.pi / 6 * np.array([1, 2, 3]) - 1) < TOL

        H = [1, 2, 3] + G
        assert np.linalg.norm(
            _val(H, P) - np.pi / 6 * np.array([1, 2, 3])
            - np.array([1, 2, 3])) < TOL

    def test_times(self):
        f = Chebfun3.from_function(lambda x, y, z: x)
        G = Chebfun3v([f, 2 * f, 3 * f])

        H = G * 1
        assert np.linalg.norm(_val(H, P) - _val(G, P)) < TOL

        H = G * [1, 2, 3]
        assert np.linalg.norm(_val(H, P) - np.pi / 6 * np.array([1, 4, 9])) < TOL

        H = 1 * G
        assert np.linalg.norm(_val(H, P) - _val(G, P)) < TOL

        H = [1, 2, 3] * G
        assert np.linalg.norm(_val(H, P) - np.pi / 6 * np.array([1, 4, 9])) < TOL

    def test_mtimes_inner_product(self):
        f = Chebfun3.from_function(lambda x, y, z: x)
        G = Chebfun3v([f, 2 * f, 3 * f])

        H = G * 1
        assert np.linalg.norm(_val(H, P) - _val(G, P)) < TOL

        Hin = G.ctranspose() @ [1, 2, 3]
        assert float((Hin - (f + 4 * f + 9 * f)).norm()) < TOL

        H = 1 * G
        assert np.linalg.norm(_val(H, P) - _val(G, P)) < TOL

        with pytest.raises(ValueError):
            [1, 2, 3] @ G

    def test_vector_calculus_identities(self):
        f0 = Chebfun3.from_function(lambda x, y, z: x)
        F1 = Chebfun3v([f0, f0, f0])
        G = Chebfun3v([f0, 2 * f0, 3 * f0])
        f = Chebfun3.from_function(lambda x, y, z: jnp.sin(x * y * z))

        # curl(F1 + G) == curl(F1) + curl(G)
        assert float((G.curl() + F1.curl()
                      - (F1 + G).curl()).norm()) < 100 * TOL

        # divergence(f*G) == dot(G, grad(f)) + f .* divergence(G)
        lhs = (f * G).divergence()
        rhs = G.dot(Chebfun3v.gradient(f)) + f * G.divergence()
        assert float((lhs - rhs).norm()) < 100 * TOL

        # divergence(curl(G)) == 0
        assert float(G.curl().divergence().norm()) < 10 * TOL

        # divergence(grad(f)) == laplacian(f)
        assert float((Chebfun3v.gradient(f).divergence()
                      - f.laplacian()).norm()) < 10 * TOL

        # curl(curl(G)) == grad(divergence(G)) - laplacian(G)
        lhs = G.curl().curl()
        rhs = Chebfun3v.gradient(G.divergence()) - G.laplacian()
        assert float((lhs - rhs).norm()) < 10 * TOL
