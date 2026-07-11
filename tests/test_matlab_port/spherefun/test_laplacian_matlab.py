"""Port of MATLAB Chebfun tests/spherefun/test_laplacian.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefun/test_laplacian.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.spherefun.spherefun import Spherefun


class TestSpherefunLaplacian:
    @pytest.mark.parametrize("l,m", [(1, 0), (2, 1), (3, 3), (5, -2)])
    def test_eigenvalue_identity(self, l, m):
        Y = Spherefun.sphharm(l, m)
        lam, th = jnp.asarray(0.5), jnp.asarray(1.2)
        got = float(Y.laplacian()(lam, th))
        want = -l * (l + 1) * float(Y(lam, th))
        assert abs(got - want) < 1e-8 * max(abs(want), 1.0)
