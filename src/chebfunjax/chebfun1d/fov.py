"""Field of values (numerical range) of a matrix as a complex Chebfun.

``fov(A)`` returns the boundary curve of the field of values of ``A`` as a
complex-valued Chebfun of the Johnson angle ``theta`` on ``[0, 2*pi]``,
constructed with splitting on (the curve of a non-normal matrix has
corners and, when the numerical range has flat sides, jumps) and merged;
``fov(A, line_segments=True)`` also returns the straight line segments
that close the jumps and the angles at which they occur.

Provenance
----------
MATLAB source : fov.m
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford and The
    Chebfun Developers.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np  # uses-numpy: dense eigensolves of the Hermitian parts

from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun


def _fov_curve(theta, A: np.ndarray) -> np.ndarray:
    """MATLAB ``fovCurve``: for each angle the point ``v'Av / v'v`` where
    ``v`` is the eigenvector of the largest eigenvalue of the Hermitian
    part of ``exp(i theta) A``."""
    th = np.atleast_1d(np.asarray(theta, dtype=np.float64))
    z = np.empty(th.shape, dtype=np.complex128)
    for j, t in enumerate(th.ravel()):
        B = np.exp(1j * t) * A
        H = (B + B.conj().T) / 2
        w, X = np.linalg.eigh(H)
        v = X[:, int(np.argmax(w))]
        z.flat[j] = (v.conj() @ A @ v) / (v.conj() @ v)
    return z


def fov(A, line_segments: bool = False):
    """Field of values boundary of the matrix ``A`` (MATLAB ``fov``).

    Parameters
    ----------
    A : array_like, shape (n, n)
    line_segments : bool
        If True return ``(f, lineSegs, theta)`` like MATLAB's three-output
        form: ``lineSegs`` is an array-valued Chebfun on [-1, 1] whose
        columns are the line segments joining the left and right limits
        of ``f`` at its jumps, and ``theta`` the angles of those jumps.

    Provenance
    ----------
    MATLAB source : fov.m
    Chebfun commit: 7574c77
    """
    A = np.asarray(A)
    f = chebfun(lambda t: jnp.asarray(_fov_curve(np.asarray(t), A)),
                domain=(0.0, 2.0 * np.pi), splitting=True)
    f = f.merge()
    if not line_segments:
        return f
    ends = np.asarray(list(f.domain.breakpoints), dtype=np.float64)
    delta = 1e-14
    if np.any(np.diff(ends) < delta):
        delta = float(np.min(np.diff(ends))) / 3
    left = np.asarray(f(jnp.asarray(ends[:-1]), "left")).ravel().astype(np.complex128)
    left[0] = complex(np.asarray(f(jnp.asarray(ends[-1]), "left")).ravel()[0])
    right = np.asarray(f(jnp.asarray(ends[:-1]), "right")).ravel().astype(np.complex128)
    tol = 10 * delta * np.maximum(np.abs(left), np.abs(right))
    discont = np.abs(left - right) > tol
    theta = ends[:-1][discont]
    cols = [Chebfun.from_values(jnp.asarray([lv, rv]), domain=(-1.0, 1.0))
            for lv, rv in zip(left[discont], right[discont])]
    return f, cols, theta
