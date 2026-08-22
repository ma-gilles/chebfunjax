"""Port of MATLAB Chebfun tests/spherefun/test_rank.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefun/test_rank.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from chebfunjax.spherefun.spherefun import Spherefun

jax.config.update("jax_enable_x64", True)

TOL2 = 2.220446049250313e-16


def _sph(fc):
    def f(lam, th):
        x = jnp.cos(lam) * jnp.sin(th)
        y = jnp.sin(lam) * jnp.sin(th)
        z = jnp.cos(th)
        return fc(x, y, z)
    return Spherefun.from_function(f)


class TestSpherefunRank:
    def test_all_matlab_assertions(self):
        for fc, in [
            (lambda x, y, z: 1.0 / (1 + 10 * (z ** 2 + (x - 1) ** 2
                                              - y ** 2) ** 2),),
            (lambda x, y, z: 1.0 / (1 + 100 * ((z - 1) ** 2 + x ** 2
                                               - y ** 2) ** 2),),
            (lambda x, y, z: jnp.cos(10 * (x ** 2 + z))
             * jnp.sin(10 * (x + y ** 2)),),
            (lambda x, y, z: jnp.tanh(20 * x) * jnp.tanh(10 * y)
             * jnp.cos(50 * z * x * y + 1),),
        ]:
            f = _sph(fc)
            k = f.rank
            m, n = f.length()
            assert k <= min(m, n)
