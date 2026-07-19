"""Port of MATLAB Chebfun tests/ballfunv/test_HelmholtzDecomposition.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfunv/test_HelmholtzDecomposition.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.ballfun.ballfun import Ballfun
from chebfunjax.ballfun.ballfunv import Ballfunv

# The identity v == grad(f) + PT2ballfunv(Ppsi, Tpsi) requires v - grad(f)
# to be divergence-free to ~1e-12, i.e. an accurate ball Poisson-Neumann
# solve.  This is provided by the coeff-space spectral Ballfun.helmholtz
# (Fourier x Fourier x ultraspherical with per-mode QZ Sylvester solves and
# the Legendre DC-mode Neumann branch), a faithful port of
# @ballfun/helmholtz.m; the decomposition now closes to ~1e-13.

# tol = 1e6 * pref.techPrefs.chebfuneps  (chebfuneps = machine eps)
TOL = 1e6 * float(np.finfo(np.float64).eps)


def _grad(f: Ballfun) -> Ballfunv:
    fx, fy, fz = f.grad()
    return Ballfunv(fx, fy, fz)


class TestBallfunvHelmholtzdecomposition:
    def test_all_matlab_assertions(self):
        pass_ = {}

        # ---- Two-component form ----
        # Example 1
        vx = Ballfun.from_function(lambda x, y, z: jnp.cos(x * y))
        vy = Ballfun.from_function(lambda x, y, z: jnp.sin(x * z))
        vz = Ballfun.from_function(lambda x, y, z: jnp.cos(y * z))
        v = Ballfunv(vx, vy, vz)
        f, Ppsi, Tpsi = v.HelmholtzDecomposition(nargout=3)
        w = _grad(f) + Ballfunv.PT2ballfunv(Ppsi, Tpsi)
        pass_[1] = (v - w).norm() < TOL

        # Example 2
        f0 = Ballfun.from_function(lambda x, y, z: jnp.cos(y * z))
        P = Ballfun.from_function(lambda x, y, z: jnp.cos(x * y))
        T = Ballfun.from_function(lambda x, y, z: jnp.sin(x * z))
        v = _grad(f0) + Ballfunv.PT2ballfunv(P, T)
        f, Ppsi, Tpsi = v.HelmholtzDecomposition(nargout=3)
        w = _grad(f) + Ballfunv.PT2ballfunv(Ppsi, Tpsi)
        pass_[2] = (v - w).norm() < TOL

        # ---- Three-component form ----
        # Example 3
        vx = Ballfun.from_function(lambda x, y, z: jnp.cos(x * y))
        vy = Ballfun.from_function(lambda x, y, z: jnp.sin(x * z))
        vz = Ballfun.from_function(lambda x, y, z: jnp.cos(y * z))
        v = Ballfunv(vx, vy, vz)
        f, Ppsi, Tpsi, phi = v.HelmholtzDecomposition(nargout=4)
        w = (_grad(f)
             + Ballfunv.PT2ballfunv(Ppsi, Tpsi).curl()
             + _grad(phi))
        pass_[3] = (v - w).norm() < TOL

        # Example 4
        f0 = Ballfun.from_function(lambda x, y, z: jnp.cos(y * z))
        P = Ballfun.from_function(lambda x, y, z: jnp.cos(x * y))
        T = Ballfun.from_function(lambda x, y, z: jnp.sin(x * z))
        v = _grad(f0) + Ballfunv.PT2ballfunv(P, T)
        f, Ppsi, Tpsi, phi = v.HelmholtzDecomposition(nargout=4)
        w = (_grad(f)
             + Ballfunv.PT2ballfunv(Ppsi, Tpsi).curl()
             + _grad(phi))
        pass_[4] = (v - w).norm() < TOL

        for i in range(1, 5):
            assert pass_[i], f"HelmholtzDecomposition assertion {i} failed"
