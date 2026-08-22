"""Port of MATLAB Chebfun tests/spherefun/test_vertcat.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefun/test_vertcat.m
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


class TestSpherefunVertcat:
    def test_all_matlab_assertions(self):
        f = _sph(lambda x, y, z: jnp.cos(x))
        F = Spherefun.vertcat(f, f, f)
        assert (F.components[0] - f).iszero()   # pass(1)
        try:
            Spherefun.vertcat(f, f)
            assert False                        # pass(2)
        except ValueError:
            pass
