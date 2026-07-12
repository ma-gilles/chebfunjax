"""Port of MATLAB Chebfun tests/misc/test_hermpoly.m (Fable 5).

FIXED: hermpoly added in the Fable 5 audit.  MATLAB represents H_n on
[-inf, inf] via blowup technology; chebfunjax represents the (exact)
polynomial on a finite domain, so the [-inf, inf] norm assertions are
carried as pointwise-equality assertions (equivalent for polynomials).

Provenance
----------
MATLAB source : tests/misc/test_hermpoly.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

TOL = 1e6 * np.finfo(float).eps
XS = np.linspace(-1, 1, 100)


class TestHermpoly:
    def test_physicists(self):
        exact = {
            0: lambda x: 1 + 0 * x,
            1: lambda x: 2 * x,
            2: lambda x: 4 * x ** 2 - 2,
            3: lambda x: 8 * x ** 3 - 12 * x,
            4: lambda x: 16 * x ** 4 - 48 * x ** 2 + 12,
        }
        for n, ex in exact.items():
            h = cj.hermpoly(n)
            err = np.linalg.norm(
                np.asarray(h(jnp.asarray(XS))) - ex(XS))
            assert err < TOL, n

    def test_probabilists(self):
        exact = {
            0: lambda x: 1 + 0 * x,
            1: lambda x: x,
            2: lambda x: x ** 2 - 1,
            3: lambda x: x ** 3 - 3 * x,
        }
        for n, ex in exact.items():
            h = cj.hermpoly(n, "prob")
            err = np.linalg.norm(
                np.asarray(h(jnp.asarray(XS))) - ex(XS))
            assert err < TOL, n
