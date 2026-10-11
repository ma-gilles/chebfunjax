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

import jax
import jax.numpy as jnp

# ===========================================================================
# Public API
# ===========================================================================


def fov(
    A: jax.Array,
    n_theta: int = 512,
) -> tuple[jax.Array, jax.Array]:
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
    theta : jax.Array, shape (n_theta,)
        Angles in [0, 2*pi] at which the boundary is sampled.
    boundary : jax.Array, complex, shape (n_theta,)
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
    >>> import jax.numpy as jnp
    >>> A = jnp.array([[0., 2.], [0., 0.]])
    >>> theta, bdy = fov(A)
    >>> bdy.shape
    (512,)

    See Also
    --------
    jax.numpy.linalg.eigh
    """
    A = jnp.asarray(A, dtype=complex)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2-D array.")

    theta = jnp.linspace(0.0, 2.0 * jnp.pi, n_theta, endpoint=False)
    from chebfunjax.chebfun1d.fov import _fov_curve

    boundary = _fov_curve(theta, A)
    return theta, boundary


# ===========================================================================
# Private helpers
# ===========================================================================


def _fov_point(theta: float, A: jax.Array) -> complex:
    """Compute one boundary point of fov(A) at angle theta."""
    from chebfunjax.chebfun1d.fov import _fov_curve

    return _fov_curve(jnp.asarray(theta), jnp.asarray(A))[0]
