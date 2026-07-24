"""Port of MATLAB Chebfun tests/chebfun3t/test_get.m (Fable 5).

chebfunjax's :class:`Chebfun3T` is a Tucker-backed wrapper (cols/rows/
tubes factors + core), not MATLAB's full-coefficient-tensor variant, but
the get-accessor semantics tested here (domain, positive vertical scale,
nonzero coefficients) are the same math and are exercised against it.

The ``domain`` and coefficient (``core``) assertions are exact.  MATLAB's
``f.vscale`` is a stored field; :class:`Chebfun3T` stores no vscale, so
the vertical scale is recomputed as max|f| over a sample grid (the same
quantity MATLAB's vscale estimates).

Provenance
----------
MATLAB source : tests/chebfun3t/test_get.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun3d import chebfun3t

EPS = float(np.finfo(np.float64).eps)
TOL = 1e2 * EPS


@pytest.mark.slow
class TestChebfun3tGet:
    def test_all_matlab_assertions(self):
        # f = chebfun3t(@(x,y,z) cos(x.*y.*z));
        f = chebfun3t(lambda x, y, z: jnp.cos(x * y * z))

        # pass(1) = norm([-1 1 -1 1 -1 1] - f.domain) < tol;
        dom = np.asarray(f.domain, dtype=float)
        assert np.linalg.norm(dom - np.array([-1, 1, -1, 1, -1, 1])) < TOL

        # pass(2) = f.vscale > 0;
        xs = np.linspace(-1, 1, 9)
        xx, yy, zz = np.meshgrid(xs, xs, xs)
        vscale = float(np.max(np.abs(np.asarray(f(xx, yy, zz)))))
        assert vscale > 0

        # pass(3) = norm(f.coeffs(:)) > 0;
        # (Chebfun3T stores the Tucker core in place of a full coeff
        # tensor; a nonzero core is the format-independent statement.)
        assert float(np.linalg.norm(f.core_numpy().ravel())) > 0
