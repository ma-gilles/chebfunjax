"""Port of MATLAB Chebfun tests/chebfun3/test_std2.m (Fable 5).

FIXED (Fable 5): Chebfun3.std2 added in the audit.

Provenance
----------
MATLAB source : tests/chebfun3/test_std2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import Chebfun3

from ._helpers import EPS

TOL = 1e3 * EPS
EXACT = np.sqrt(4.0 / 45.0)


def _const1_err(g1):
    # g1 is a Chebfun; std2 here is the constant sqrt(4/45).
    s = jnp.asarray(np.linspace(-0.9, 0.9, 9))
    vals = np.asarray(g1(s))
    return float(np.max(np.abs(vals - EXACT)))


class TestChebfun3Std2:
    def test_all_matlab_assertions(self):
        # (x,y)-std of x^2 + z equals sqrt(4/45) (z-term has zero (x,y)-std).
        f1 = Chebfun3.from_function(lambda x, y, z: x ** 2 + z)
        f2 = Chebfun3.from_function(lambda x, y, z: z ** 2 + y)
        f3 = Chebfun3.from_function(lambda x, y, z: y ** 2 + x)

        assert _const1_err(f1.std2()) < TOL                 # default (1,2)
        assert _const1_err(f1.std2(None, (1, 2))) < TOL
        assert _const1_err(f2.std2(None, (1, 3))) < TOL
        assert _const1_err(f3.std2(None, (3, 2))) < TOL
