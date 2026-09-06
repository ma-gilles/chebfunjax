"""Port of MATLAB Chebfun tests/spinscheme/test_phiFun.m (Fable 5).

Provenance
----------
MATLAB source : tests/spinscheme/test_phiFun.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.operators.expinteg import phi_fun

jax.config.update("jax_enable_x64", True)


class TestSpinschemePhiFun:
    def test_all_matlab_assertions(self):
        tol = 1e-12
        ex = [lambda z: np.exp(z),
              lambda z: (np.exp(z) - 1) / z,
              lambda z: (np.exp(z) - z - 1) / z ** 2,
              lambda z: (np.exp(z) - z ** 2 / 2 - z - 1) / z ** 3]
        g = np.concatenate([np.arange(-10, 0), np.arange(1, 11)])
        xx, yy = np.meshgrid(g, g)
        zz = xx + 1j * yy
        for k in range(4):
            got = np.asarray(phi_fun(k)(jnp.asarray(zz)))
            assert np.max(np.abs(got - ex[k](zz))) < tol                     # pass(k+1)
