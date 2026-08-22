"""Port of MATLAB Chebfun tests/spherefun/test_partitionCombine.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefun/test_partitionCombine.m
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


class TestSpherefunPartitionCombine:
    def test_all_matlab_assertions(self):
        tol = 1e3 * 100 * TOL2
        feven, fodd = Spherefun.empty().partition()
        assert feven.isempty() and fodd.isempty()               # pass(1)
        f = Spherefun.combine(Spherefun.empty(), Spherefun.empty())
        assert f.isempty()                                      # pass(2)
        f = _sph(lambda x, y, z: jnp.sin(jnp.pi * x * y))
        feven, fodd = f.partition()
        assert float((feven - f).norm()) < tol                  # pass(3)
        assert fodd.isempty() or fodd.rank == 0               # pass(4)
        f = _sph(lambda x, y, z: jnp.sin(jnp.pi * x * z))
        feven, fodd = f.partition()
        assert float((fodd - f).norm()) < tol                   # pass(5)
        assert feven.isempty() or feven.rank == 0             # pass(6)
        fe = lambda x, y, z: jnp.sin(jnp.pi * x * y)
        fo = lambda x, y, z: jnp.sin(jnp.pi * x * z)
        f = _sph(lambda x, y, z: fe(x, y, z) + fo(x, y, z))
        feven, fodd = f.partition()
        assert float((_sph(fe) - feven).norm()) < tol           # pass(7)
        assert float((_sph(fo) - fodd).norm()) < tol            # pass(8)
        g = Spherefun.combine(feven, fodd)
        assert float((g - f).norm()) < tol                      # combine
