"""Port of MATLAB Chebfun tests/chebfun3/test_std.m (Fable 5).

FIXED (Fable 5): Chebfun3.std added in the audit.

Provenance
----------
MATLAB source : tests/chebfun3/test_std.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import Chebfun3

from ._helpers import EPS

TOL = 1e3 * EPS
EXACT = np.sqrt(4.0 / 45.0)


def _const2_err(g2):
    # g2 is a Chebfun2; std here is the constant sqrt(4/45).
    s = np.linspace(-0.9, 0.9, 5)
    ss, tt = np.meshgrid(s, s, indexing="ij")
    vals = np.asarray(g2(jnp.asarray(ss.ravel()), jnp.asarray(tt.ravel())))
    return float(np.max(np.abs(vals - EXACT)))


class TestChebfun3Std:
    def test_all_matlab_assertions(self):
        # x-std of x^2 + y*z equals sqrt(4/45) (the y*z term has zero x-std).
        f1 = Chebfun3.from_function(lambda x, y, z: x ** 2 + y * z)
        f2 = Chebfun3.from_function(lambda x, y, z: y ** 2 + x * z)
        f3 = Chebfun3.from_function(lambda x, y, z: z ** 2 + x * y)

        assert _const2_err(f1.std()) < TOL              # default dim=1
        assert _const2_err(f1.std(None, 1)) < TOL       # dim=1 (x)
        assert _const2_err(f2.std(None, 2)) < TOL       # dim=2 (y)
        assert _const2_err(f3.std(None, 3)) < TOL       # dim=3 (z)
