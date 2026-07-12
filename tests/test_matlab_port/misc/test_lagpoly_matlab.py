"""Port of MATLAB Chebfun tests/misc/test_lagpoly.m (Fable 5).

FIXED: lagpoly added in the Fable 5 audit.  Finite-domain polynomial
representation (see hermpoly port for the [0, inf] caveat).

Provenance
----------
MATLAB source : tests/misc/test_lagpoly.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

TOL = 1e5 * np.finfo(float).eps
XS = np.linspace(0, 1, 100)


class TestLagpoly:
    def test_laguerre_closed_forms(self):
        exact = {
            0: lambda x: 1 + 0 * x,
            1: lambda x: -x + 1,
            2: lambda x: (x ** 2 - 4 * x + 2) / 2,
            3: lambda x: (-x ** 3 + 9 * x ** 2 - 18 * x + 6) / 6,
            4: lambda x: (x ** 4 - 16 * x ** 3 + 72 * x ** 2
                          - 96 * x + 24) / 24,
            5: lambda x: (-x ** 5 + 25 * x ** 4 - 200 * x ** 3
                          + 600 * x ** 2 - 600 * x + 120) / 120,
        }
        for n, ex in exact.items():
            h = cj.lagpoly(n)
            err = np.linalg.norm(
                np.asarray(h(jnp.asarray(XS))) - ex(XS))
            assert err < (10 * TOL if n >= 5 else TOL), n
