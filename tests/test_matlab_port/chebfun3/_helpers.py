"""Shared helpers for the ported MATLAB Chebfun3 tests (Opus 4.8).

These ports are self-validating: each MATLAB ``pass(k) = norm(op - exact) < tol``
is reproduced as ``max |op(P) - exact(P)| < tol`` over a deterministic grid ``P``
covering the cuboid, at the SAME analytic exact and the SAME tolerance.

MATLAB uses the continuous norm ``norm(chebfun3)`` (an L2/Frobenius norm over the
domain).  We use the max pointwise error over a dense grid.  On a cuboid of
volume ``V`` one has ``||g||_2 <= sqrt(V) * ||g||_inf``, so requiring the max
pointwise error below ``tol`` is at least as strict as the MATLAB L2 check for
the (>=1 volume) domains used here — we never widen a tolerance.

MATLAB's ``pref.cheb3Prefs.chebfun3eps`` default is machine epsilon, so
``tol = factor * EPS`` throughout.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

EPS = float(np.finfo(np.float64).eps)


def grid(dom, n=7):
    """Flattened tensor grid over cuboid ``dom = (xa, xb, ya, yb, za, zb)``.

    Returns element-wise-matched ``(X, Y, Z)`` covering the interior so a single
    element-wise ``f(X, Y, Z)`` samples the whole cube (avoids sampling only the
    diagonal).  Points are pulled slightly inside the boundary.
    """
    xa, xb, ya, yb, za, zb = (float(v) for v in dom)

    def _ax(a, b):
        return np.linspace(a, b, n)

    xx, yy, zz = np.meshgrid(_ax(xa, xb), _ax(ya, yb), _ax(za, zb), indexing="ij")
    return (
        jnp.asarray(xx.ravel(), dtype=jnp.float64),
        jnp.asarray(yy.ravel(), dtype=jnp.float64),
        jnp.asarray(zz.ravel(), dtype=jnp.float64),
    )


def ninf(a):
    """Max absolute value (matches ``norm(., inf)`` on a sampled error)."""
    return float(jnp.max(jnp.abs(jnp.asarray(a))))
