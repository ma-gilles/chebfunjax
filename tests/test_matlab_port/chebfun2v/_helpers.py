"""Shared helpers for the chebfun2v roots MATLAB-suite ports (Fable 5).

chebfunjax provides the marching-squares common-zero finder only (the
Bezout-resultant path is not implemented), so the MATLAB assertions that
merely cross-check the 'ms' and 'resultant' methods are ported as direct
residual / count / known-root checks on the single method -- a faithful
and in fact stricter correctness test.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

EPS = float(np.finfo(np.float64).eps)
TOL = 1e3 * EPS


def residuals(f, g, r):
    """(max|f(r)|, max|g(r)|) over the returned root points."""
    if len(r) == 0:
        return 0.0, 0.0
    x = jnp.asarray(r[:, 0])
    y = jnp.asarray(r[:, 1])
    return (float(np.max(np.abs(np.asarray(f(x, y))))),
            float(np.max(np.abs(np.asarray(g(x, y))))))


def match_points(r, exact, tol):
    """True if the returned points equal ``exact`` (an (m,2) array) as sets,
    compared columnwise after sorting each column (MATLAB
    ``norm(sort(r(:,k)) - sort(exact(:,k)))``)."""
    r = np.asarray(r)
    exact = np.asarray(exact)
    if r.shape[0] != exact.shape[0]:
        return False
    ex = np.linalg.norm(np.sort(r[:, 0]) - np.sort(exact[:, 0]))
    ey = np.linalg.norm(np.sort(r[:, 1]) - np.sort(exact[:, 1]))
    return ex < tol and ey < tol
