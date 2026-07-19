"""Port of MATLAB Chebfun tests/spherefun/test_abs.m (Fable 5).

FIXED (Fable 5): Spherefun.abs added in the audit.

Provenance
----------
MATLAB source : tests/spherefun/test_abs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.spherefun.spherefun import Spherefun

TOL = 1000 * float(np.finfo(np.float64).eps)


def _xyz(lam, th):
    return (jnp.cos(lam) * jnp.sin(th), jnp.sin(lam) * jnp.sin(th),
            jnp.cos(th))


class TestSpherefunAbs:
    def test_all_matlab_assertions(self):
        # f = -(x^2 + y^2 + z^2) = -1 on the sphere; abs(f) + f == 0.
        def fn(lam, th):
            x, y, z = _xyz(lam, th)
            return -(x ** 2 + y ** 2 + z ** 2)

        f = Spherefun.from_function(fn)
        s = abs(f) + f
        lam = jnp.asarray(np.linspace(-np.pi, np.pi, 25))
        th = jnp.asarray(np.linspace(0.0, np.pi, 25))
        LL, TT = jnp.meshgrid(lam, th)
        assert float(jnp.max(jnp.abs(s(LL, TT)))) < TOL
