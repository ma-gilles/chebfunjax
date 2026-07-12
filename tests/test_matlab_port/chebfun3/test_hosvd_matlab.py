"""Port of MATLAB Chebfun tests/chebfun3/test_hosvd.m (Fable 5).

FIXED: Chebfun3.hosvd added in the Fable 5 audit (L2 Gram-Cholesky
orthonormalization of the Tucker factors + core HOSVD).

Provenance
----------
MATLAB source : tests/chebfun3/test_hosvd.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import Chebfun3

TOL = 1e-12


class TestChebfun3Hosvd:
    def test_hackbusch_exact_singular_values(self):
        # pass(1)-(3): Hackbusch, Tensor Spaces (2012), p. 233
        f = Chebfun3.from_function(
            lambda x, y, z: x * z + x ** 2 * y,
            domain=(0, 1, 0, 1, 0, 1))
        sv, _ = f.hosvd()
        ex1 = np.array([np.sqrt(109 / 720 + np.sqrt(46) / 45),
                        np.sqrt(109 / 720 - np.sqrt(46) / 45)])
        ex2 = np.array([np.sqrt(109 / 720 + np.sqrt(2899) / 360),
                        np.sqrt(109 / 720 - np.sqrt(2899) / 360)])
        assert np.max(np.abs(np.asarray(sv[0])[:2] - ex1)) < TOL
        assert np.max(np.abs(np.asarray(sv[1])[:2] - ex2)) < TOL
        assert np.max(np.abs(np.asarray(sv[2])[:2] - ex2)) < TOL

    def test_all_orthogonal_core_and_reconstruction(self):
        # pass(4)-(9)
        f = Chebfun3.from_function(
            lambda x, y, z: x * z + x ** 2 * y)
        sv, g = f.hosvd()
        core = np.asarray(g.core)
        if core.shape[0] > 1:
            assert abs(np.sum(core[0] * core[1])) < TOL
        if core.shape[1] > 1:
            assert abs(np.sum(core[:, 0] * core[:, 1])) < TOL
        if core.shape[2] > 1:
            assert abs(np.sum(core[:, :, 0] * core[:, :, 1])) < TOL
        xs = jnp.asarray(np.linspace(-1, 1, 6))
        XX, YY, ZZ = jnp.meshgrid(xs, xs, xs, indexing="ij")
        assert float(jnp.max(jnp.abs(
            g(XX, YY, ZZ) - f(XX, YY, ZZ)))) < 1e-11
