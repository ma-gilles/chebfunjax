"""Port of MATLAB Chebfun tests/spherefun/test_optimization.m (Fable 5).

FIXED (Fable 5): Spherefun.minandmax2 / max2 / min2 added in the audit
(grid seed + multi-start L-BFGS-B polish with exact JAX gradients).

Provenance
----------
MATLAB source : tests/spherefun/test_optimization.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.spherefun.spherefun import Spherefun

TOL = 5e4 * float(np.finfo(np.float64).eps)


def _xyz(lam, th):
    return (jnp.cos(lam) * jnp.sin(th), jnp.sin(lam) * jnp.sin(th),
            jnp.cos(th))


# Battery given in Cartesian (x, y, z); expressed here in (lam, theta).
def _b1(lam, th):
    return jnp.cos(np.pi * _xyz(lam, th)[2])


def _b2(lam, th):
    return jnp.sin(2 * np.pi * _xyz(lam, th)[2])


def _b3(lam, th):
    return jnp.cos(np.pi * _xyz(lam, th)[0])


def _b4(lam, th):
    return jnp.cos(np.pi * _xyz(lam, th)[1])


def _b5(lam, th):
    x, y, _z = _xyz(lam, th)
    return jnp.cos(np.pi * x) * jnp.cos(np.pi * y)


def _b6(lam, th):
    x, y, _z = _xyz(lam, th)
    return jnp.cos(2 * np.pi * x) * jnp.cos(2 * np.pi * y)


def _b7(lam, th):
    x, y, z = _xyz(lam, th)
    return jnp.exp(-10 * ((x - 1 / np.sqrt(2)) ** 2
                          + (z - 1 / np.sqrt(2)) ** 2 + y ** 2))


BATTERY = [_b1, _b2, _b3, _b4, _b5, _b6, _b7]
MINI = [-1, -1, -1, -1, -1, -1, 0]
MAXI = [1, 1, 1, 1, 1, 1, 1]


class TestSpherefunOptimization:
    def test_all_matlab_assertions(self):
        for jj, fn in enumerate(BATTERY):
            g = Spherefun.from_function(fn)
            Y, _X = g.minandmax2()
            err = (abs(float(Y[0]) - MINI[jj])
                   + abs(float(Y[1]) - MAXI[jj]))
            assert err < TOL, f"battery[{jj + 1}] err={err:.3e}"
