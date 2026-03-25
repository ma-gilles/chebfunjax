# uses-numpy: least-squares iteration and AAA are iterative/not JIT-safe
"""Conformal mapping of a doubly-connected (annular) region to a circular annulus.

Translated from MATLAB Chebfun (commit 7574c77): conformal2.m.
Original: Copyright 2019 by L. N. Trefethen and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.

References
----------
A. Gopal and L. N. Trefethen, "Representation of conformal maps by rational
functions", Numer. Math. 142 (2019), 359–382.

L. N. Trefethen, "Numerical conformal mapping with rational functions",
Comp. Meth. Funct. Thy. 20 (2020), 369–387.
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np

from chebfunjax.utils.aaa import aaa

# ===========================================================================
# Public API
# ===========================================================================


def conformal2(
    Z1: np.ndarray,
    Z2: np.ndarray,
    *,
    tol: float = 1e-6,
) -> tuple[Callable, Callable, float, np.ndarray, np.ndarray]:
    """Conformal map from a doubly-connected (annular) region to a circular annulus.

    Given two smooth closed curves C1 (outer boundary) and C2 (inner boundary,
    enclosing the origin), compute a conformal map F of the annular region
    bounded by C1 and C2 to the circular annulus {rho < |w| < 1}, together
    with its inverse.

    The conformal modulus rho < 1 is determined as part of the calculation.

    Algorithm
    ---------
    (1) Solve a least-squares Dirichlet problem to find a harmonic function u
        such that u = -log|z| on C1 and u = -log|z| + log(rho) on C2.
    (2) Set F(z) = z * exp(u(z) + i*v(z)), where v is the harmonic conjugate.
    (3) Approximate F and its inverse by rational functions via AAA.

    Parameters
    ----------
    Z1 : array_like, complex, shape (M1,)
        Sample points on the *outer* boundary curve, ordered counter-clockwise.
    Z2 : array_like, complex, shape (M2,)
        Sample points on the *inner* boundary curve, ordered counter-clockwise,
        enclosing the origin.
    tol : float, optional
        Convergence tolerance for the harmonic least-squares step (default 1e-6).

    Returns
    -------
    f : callable
        Forward conformal map.  ``f(z)`` maps the annular region to the
        circular annulus ``{rho < |w| < 1}``.
    finv : callable
        Inverse conformal map.  ``finv(w)`` maps the annulus back to the region.
    rho : float
        Conformal modulus (inner radius of the image annulus), 0 < rho < 1.
    pol : np.ndarray, complex
        Poles of the forward map f.
    polinv : np.ndarray, complex
        Poles of the inverse map finv.

    Notes
    -----
    This is an experimental implementation suitable for smooth, simple regions.
    Regions with corners or near-degeneracies may not converge.  The polynomial
    degree is doubled every iteration until the boundary error drops below tol.

    Provenance
    ----------
    MATLAB source : conformal2.m
    Chebfun commit: 7574c77
    Original authors: L. N. Trefethen (October 2019) and The Chebfun Developers.
        Copyright 2019 by The University of Oxford and The Chebfun Developers.

    Examples
    --------
    Ellipse inside a larger ellipse:

    >>> import numpy as np
    >>> theta = np.linspace(0, 2*np.pi, 200, endpoint=False)
    >>> Z1 = 3*np.cos(theta) + 1j*np.sin(theta)   # outer ellipse
    >>> Z2 = 1.5*np.cos(theta) + 0.5j*np.sin(theta)  # inner ellipse
    >>> f, finv, rho, pol, polinv = conformal2(Z1, Z2)
    >>> 0.0 < rho < 1.0
    True

    See Also
    --------
    conformal
    """
    Z1 = np.asarray(Z1, dtype=complex).ravel()
    Z2 = np.asarray(Z2, dtype=complex).ravel()

    err = np.inf
    logn = 4.0
    W_best = None
    rho = None

    while err > tol:
        n = round(2 ** logn)
        M = 8 * n
        # Re-sample boundaries at M evenly-spaced parameter values
        idx1 = np.round(np.linspace(0, len(Z1) - 1, M)).astype(int) % len(Z1)
        idx2 = np.round(np.linspace(0, len(Z2) - 1, M)).astype(int) % len(Z2)
        Z1s = Z1[idx1]
        Z2s = Z2[idx2]
        Z = np.concatenate([Z1s, Z2s])

        # RHS for Dirichlet problem: u = -log|z|
        H = -np.log(np.abs(Z))
        # On inner boundary, add log(rho) placeholder (rho enters as extra unknown)
        H[M:] += 1.0   # will be rescaled by c[-1] = 1 - log(rho)
        rvec = np.concatenate([np.zeros(M), np.ones(M)])

        # Build Vandermonde-Arnoldi orthogonal basis for positive and negative powers
        P = _va_orthog(Z, n)
        P2 = _va_orthog(Z ** (-1), n)

        # Least-squares: real and imaginary parts
        A = np.column_stack([np.real(P), np.real(P2), -np.imag(P), -np.imag(P2), rvec])
        c_ls, _, _, _ = np.linalg.lstsq(A, H, rcond=None)

        log_rho = 1.0 - c_ls[-1]
        rho = float(np.exp(log_rho))
        err = float(np.linalg.norm(A @ c_ls - H, np.inf))

        c_ls = c_ls[:-1]
        cc = c_ls.reshape(-1, 2) @ np.array([1.0, 1j])
        F = np.concatenate([P, P2], axis=1) @ cc
        W = Z * np.exp(F)

        W_best = W
        logn += 0.5
        if err <= tol:
            break
        if logn > 8.0:
            warnings.warn("conformal2: did not converge to requested tolerance.", stacklevel=2)
            break

    # Forward map: Z -> W (annulus)
    f, pol, _, _, _, _, _ = aaa(W_best, Z, tol=tol)
    # Inverse map: W -> Z
    finv, polinv, _, _, _, _, _ = aaa(Z, W_best, tol=tol)

    return f, finv, float(rho), np.asarray(pol), np.asarray(polinv)


# ===========================================================================
# Private helpers
# ===========================================================================


def _va_orthog(Z: np.ndarray, n: int) -> np.ndarray:
    """Vandermonde–Arnoldi orthogonalisation for the monomial basis.

    Computes an ON basis Q for span{1, Z, Z^2, ..., Z^n} using the
    modified Gram-Schmidt procedure applied to the columns of the
    Vandermonde matrix V[:, k] = Z^k.

    Parameters
    ----------
    Z : np.ndarray, complex, shape (M,)
        Sample points.
    n : int
        Degree (number of columns = n + 1).

    Returns
    -------
    Q : np.ndarray, complex, shape (M, n+1)
        Orthonormal columns.
    """
    M = len(Z)
    Q = np.ones((M, 1), dtype=complex)
    for k in range(n):
        v = Z * Q[:, k]
        # Modified Gram-Schmidt orthogonalisation
        for j in range(Q.shape[1]):
            v = v - (Q[:, j].conj() @ v) / M * Q[:, j]
        norm_v = np.linalg.norm(v) / np.sqrt(M)
        if norm_v < 1e-14:
            break
        Q = np.column_stack([Q, v / norm_v])
    return Q
