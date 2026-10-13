"""Literal port of MATLAB tests/chebfun3t/test_get.m.

The native class exposes dense ``coeffs``, stored ``vscale``, and ``domain``;
this test checks those public fields directly.

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
        f = chebfun3t(lambda x, y, z: jnp.cos(x * y * z))
        assert np.linalg.norm(
            np.array([-1, 1, -1, 1, -1, 1]) - np.asarray(f.domain)
        ) < TOL
        assert float(f.vscale) > 0
        assert float(np.linalg.norm(np.asarray(f.coeffs).ravel())) > 0
