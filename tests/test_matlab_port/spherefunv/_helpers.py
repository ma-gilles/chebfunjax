"""Shared helpers for the spherefunv MATLAB-port tests (Fable 5).

The MATLAB spherefunv tests build components from Cartesian expressions,
e.g. ``spherefun(@(x,y,z) x)``, and compare vector/scalar fields with
``norm(a-b) < tol``.  Here Cartesian components are built via the intrinsic
substitution ``x = cos(lam) sin(th)``, ``y = sin(lam) sin(th)``,
``z = cos(th)``, and MATLAB ``norm`` comparisons are realised as the maximum
absolute deviation over a tensor grid strictly interior to the sphere
(avoiding the poles where ``1/sin(theta)`` factors are singular).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.spherefun.spherefun import Spherefun

EPS = float(np.finfo(np.float64).eps)

LAMS = jnp.asarray(np.linspace(-3.0, 3.0, 7))
THS = jnp.asarray(np.linspace(0.2, 2.9, 7))
LL, TT = jnp.meshgrid(LAMS, THS, indexing="ij")

# Cartesian coordinates of the grid points (numpy, for exact comparisons).
X = np.cos(np.array(LL)) * np.sin(np.array(TT))
Y = np.sin(np.array(LL)) * np.sin(np.array(TT))
Z = np.cos(np.array(TT))


def cart(fn):
    """Build a Spherefun from a Cartesian expression ``fn(x, y, z)``."""
    return Spherefun.from_function(
        lambda lam, th: fn(jnp.cos(lam) * jnp.sin(th),
                           jnp.sin(lam) * jnp.sin(th),
                           jnp.cos(th)))


def vnorm(sfv) -> float:
    """Max abs over all components of a Spherefunv on the grid (a proxy for
    MATLAB ``norm(spherefunv)``)."""
    return max(float(jnp.max(jnp.abs(np.asarray(v)))) for v in sfv(LL, TT))


def vdiff(a, b) -> float:
    """Max componentwise abs difference of two Spherefunv on the grid."""
    av, bv = a(LL, TT), b(LL, TT)
    return max(float(jnp.max(jnp.abs(np.asarray(u) - np.asarray(v))))
               for u, v in zip(av, bv))


def snorm(sf) -> float:
    """Max abs of a scalar Spherefun on the grid (MATLAB ``norm(f, inf)``)."""
    return float(jnp.max(jnp.abs(np.asarray(sf(LL, TT)))))


def sdiff(sf, exact_arr) -> float:
    """Max abs difference between a Spherefun and a numpy exact array."""
    return float(jnp.max(jnp.abs(np.asarray(sf(LL, TT)) - exact_arr)))
