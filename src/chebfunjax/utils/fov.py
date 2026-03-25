# uses-numpy: eigenvalue decomposition at each theta is not JIT-safe
"""Field of values (numerical range) of a matrix.

Translated from MATLAB Chebfun (commit 7574c77): fov.m.
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.

References
----------
[1] C. R. Johnson, "Numerical determination of the field of values of a
    general complex matrix", SIAM J. Numer. Anal. 15 (1978), 595–602.
"""

from __future__ import annotations

import numpy as np

# ===========================================================================
# Public API
# ===========================================================================


def fov(
    A: np.ndarray,
    n_theta: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Field of values (numerical range) of a matrix.

    The field of values W(A) is the set

        W(A) = { v* A v / (v* v) : v != 0 }

    which is a convex region in the complex plane (Toeplitz-Hausdorff
    theorem).  This function traces the boundary of W(A) by computing
    the extreme eigenvalue of H(theta) = (e^{i theta} A + e^{-i theta} A*) / 2
    at each angle theta in [0, 2*pi] (Johnson's algorithm [1]).

    Parameters
    ----------
    A : array_like, shape (n, n)
        Square matrix (real or complex).
    n_theta : int, optional
        Number of angles at which to sample the boundary curve (default 512).
        The curve is returned as a complex array of this length.

    Returns
    -------
    theta : np.ndarray, shape (n_theta,)
        Angles in [0, 2*pi] at which the boundary is sampled.
    boundary : np.ndarray, complex, shape (n_theta,)
        Complex numbers tracing the boundary of W(A).

    Notes
    -----
    For a *normal* matrix the boundary is the convex hull of the eigenvalues;
    the output curve will have flat segments connecting eigenvalues.

    For a generic matrix the boundary is smooth and all points are extreme.

    The numerical abscissa equals ``max(real(boundary))``.

    Provenance
    ----------
    MATLAB source : fov.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> A = rng.standard_normal((4, 4))
    >>> theta, bdy = fov(A)
    >>> bdy.shape
    (512,)

    See Also
    --------
    numpy.linalg.eig
    """
    A = np.asarray(A, dtype=complex)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2-D array.")

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    boundary = np.array([_fov_point(t, A) for t in theta])
    return theta, boundary


# ===========================================================================
# Private helpers
# ===========================================================================


def _fov_point(theta: float, A: np.ndarray) -> complex:
    """Compute one boundary point of fov(A) at angle theta."""
    r = np.exp(1j * theta)
    B = r * A
    H = 0.5 * (B + B.conj().T)
    # Hermitian part; eigenvalues are real
    eigvals, eigvecs = np.linalg.eigh(H)
    k = np.argmax(eigvals.real)
    v = eigvecs[:, k]
    return complex(v.conj() @ A @ v / (v.conj() @ v))
