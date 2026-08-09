# uses-numpy: sequential Arnoldi recurrence (data-dependent, not JIT-safe)
"""Vandermonde with Arnoldi orthogonalization (VAorthog / VAeval).

Polynomial least-squares fitting on arbitrary point sets is
catastrophically ill-conditioned in the monomial basis; running the
Arnoldi process on multiplication by ``z`` builds a discretely
orthogonal polynomial basis in which the same fit is well conditioned.

Provenance
----------
MATLAB source : VAorthog.m, VAeval.m (Chebfun examples / [1])
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford
    and The Chebfun Developers.
Algorithm: P. D. Brubeck, Y. Nakatsukasa, and L. N. Trefethen,
    "Vandermonde with Arnoldi", SIAM Review 63 (2021), 405-415.
"""
from __future__ import annotations

import numpy as np

__all__ = ["va_orthog", "va_eval"]


def va_orthog(Z, n: int):
    """Arnoldi-orthogonalized Vandermonde basis of degree ``n`` on ``Z``.

    Parameters
    ----------
    Z : array_like, shape (M,)
        Sample points (real or complex).
    n : int
        Polynomial degree (the basis has ``n + 1`` columns).

    Returns
    -------
    Hes : ndarray, shape (n+1, n)
        The Hessenberg recurrence coefficients (MATLAB ``Hes``).
    Q : ndarray, shape (M, n+1)
        The orthogonalized basis matrix (MATLAB ``R``); columns are
        discretely orthonormal, ``Q.conj().T @ Q / M = I``.

    Provenance
    ----------
    MATLAB source : VAorthog.m
    Chebfun commit: 7574c77
    """
    Z = np.asarray(Z).ravel()
    M = Z.shape[0]
    dtype = np.result_type(Z.dtype, np.float64)
    H = np.zeros((n + 1, n), dtype=np.complex128)
    Q = np.ones((M, 1), dtype=dtype)
    for k in range(n):
        q = Z * Q[:, k]
        for j in range(k + 1):
            H[j, k] = np.vdot(Q[:, j], q) / M
            q = q - H[j, k] * Q[:, j]
        H[k + 1, k] = np.linalg.norm(q) / np.sqrt(M)
        Q = np.column_stack([Q, q / H[k + 1, k]])
    if np.isrealobj(Z):
        H = np.real(H)
    return H, Q


def va_eval(Z, Hes):
    """Evaluate the Arnoldi-orthogonalized basis at new points ``Z``.

    Provenance
    ----------
    MATLAB source : VAeval.m
    Chebfun commit: 7574c77
    """
    Z = np.asarray(Z).ravel()
    M = Z.shape[0]
    n = Hes.shape[1]
    dtype = np.result_type(Z.dtype, Hes.dtype, np.float64)
    W = np.ones((M, 1), dtype=dtype)
    for k in range(n):
        w = Z * W[:, k]
        for j in range(k + 1):
            w = w - Hes[j, k] * W[:, j]
        W = np.column_stack([W, w / Hes[k + 1, k]])
    return W
