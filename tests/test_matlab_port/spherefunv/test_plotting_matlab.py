"""Port of MATLAB Chebfun tests/spherefunv/test_plotting.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefunv/test_plotting.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.spherefun.spherefun import Spherefun

jax.config.update("jax_enable_x64", True)

TOL = 1e3 * 2.220446049250313e-16


def _sph(fc):
    def f(lam, th):
        x = jnp.cos(lam) * jnp.sin(th)
        y = jnp.sin(lam) * jnp.sin(th)
        z = jnp.cos(th)
        return fc(x, y, z)
    return Spherefun.from_function(f)


def _norm_inf(g, n=25):
    lam = jnp.linspace(-np.pi + 1e-6, np.pi - 1e-6, n)
    th = jnp.linspace(1e-3, np.pi - 1e-3, n)
    L, T = jnp.meshgrid(lam, th)
    return float(jnp.max(jnp.abs(jnp.asarray(g(L, T)))))


def _vnorm(F, n=25):
    return max(_norm_inf(c, n) for c in F.components)

import matplotlib  # noqa: E402

matplotlib.use("Agg")


class TestSpherefunvPlotting:
    def test_all_matlab_assertions(self):
        import matplotlib.pyplot as plt
        F = _sph(lambda x, y, z: x * jnp.cos(y * z)).gradient()
        assert F.quiver() is not None
        assert F.quiver3() is not None
        plt.close("all")
