"""Shared helpers for the chebfun3 MATLAB-suite ports (Fable 5)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

EPS = float(np.finfo(np.float64).eps)


def grid(dom=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0), n=8):
    """Deterministic interior evaluation lattice for a 6-tuple domain."""
    xa, xb, ya, yb, za, zb = dom
    gx = np.linspace(xa + 0.03 * (xb - xa), xb - 0.03 * (xb - xa), n)
    gy = np.linspace(ya + 0.04 * (yb - ya), yb - 0.04 * (yb - ya), n)
    gz = np.linspace(za + 0.05 * (zb - za), zb - 0.05 * (zb - za), n)
    xx, yy, zz = np.meshgrid(gx, gy, gz, indexing="ij")
    return (jnp.asarray(xx.ravel()), jnp.asarray(yy.ravel()),
            jnp.asarray(zz.ravel()))


def ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


def maxdiff(f, ref_fn, dom=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0), n=8):
    """max |f(x,y,z) - ref(x,y,z)| over the lattice."""
    X, Y, Z = grid(dom, n)
    return ninf(f(X, Y, Z) - ref_fn(X, Y, Z))
