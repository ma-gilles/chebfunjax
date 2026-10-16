"""Port of MATLAB Chebfun tests/chebfun2v/test_roots06.m (Fable 5).

Original unconditional predicates and a separately labeled conditional extension.

Provenance
----------
MATLAB source : tests/chebfun2v/test_roots06.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d import chebfun2

from ._helpers import TOL, match_points


class TestChebfun2vRoots06:
    def test_original_resultant_residuals(self):
        # Residual check.
        f = chebfun2(lambda x, y: (y - 2 * x) * (y + 0.5 * x))
        g = chebfun2(lambda x, y: x * (x ** 2 + y ** 2 - 1))
        r = jnp.asarray(f.roots(g, method="resultant"))
        assert float(jnp.linalg.norm(f(r[:, 0], r[:, 1]))) < TOL
        assert float(jnp.linalg.norm(g(r[:, 0], r[:, 1]))) < TOL

    def test_conditional_branch_extension(self):
        # Native default pref.tech is a function handle: predicates3--6 are
        # assigned true. Exercise the existing non-default branch as an extension.
        # Six known common zeros.
        s = np.sqrt(5.0)
        f = chebfun2(lambda x, y: (y - 2 * x) * (y + .5 * x))
        g = chebfun2(lambda x, y: (x - .0001) * (x ** 2 + y ** 2 - 1))
        exact = np.array([
            [1 / 10000, -1 / 20000],
            [1 / 10000, 1 / 5000],
            [-2 / s, 1 / s],
            [-1 / s, -2 / s],
            [1 / s, 2 / s],
            [2 / s, -1 / s],
        ])
        assert match_points(f.roots(g, method="ms"), exact, 10 * TOL)
        assert match_points(f.roots(g, method="resultant"), exact, 100 * TOL)

    def test_original_circle_hyperbola_comparison(self):
        f = chebfun2(lambda x, y: 25*x*y - 12)
        g = chebfun2(lambda x, y: x**2 + y**2 - 1)
        r1 = np.asarray(f.roots(g, method="ms"))
        r2 = np.asarray(f.roots(g, method="resultant"))
        assert r1.shape == r2.shape
        assert np.linalg.norm(np.sort(r1[:, 0]) - np.sort(r2[:, 0])) < TOL
        assert np.linalg.norm(np.sort(r1[:, 1]) - np.sort(r2[:, 1])) < TOL
