"""Shared helpers for the chebfun2v roots MATLAB-suite ports (Fable 5).

chebfunjax provides both common-zero finders: the marching-squares path
(``method='ms'``) and the hidden-variable Bezout resultant path
(``method='resultant'``).  The MATLAB assertions that cross-check the two
methods are ported directly as ``match_points(r_ms, r_resultant, TOL)``.
Tests that check only one method use residual / count / known-root checks.
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
