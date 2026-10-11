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

import jax
import jax.numpy as jnp

from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun


@jax.jit
def _fov_curve(theta, A):
    """Literal Johnson Rayleigh quotient, mapped serially over angles."""
    th = jnp.atleast_1d(theta)

    def point(t):
        B = jnp.exp(1j * t) * A
        H = (B + B.conj().T) / 2
        w, X = jnp.linalg.eigh(H, symmetrize_input=False)
        v = X[:, jnp.argmax(w)]
        return ((v.conj() @ A) @ v) / (v.conj() @ v)

    # Native source processes one angle at a time. A mapped loop also avoids
    # materializing a large stack of eigensystems during adaptive refinement.
    return jax.lax.map(point, th.reshape(-1)).reshape(th.shape)


def fov(A, line_segments: bool = False, *, pref=None):
    """Field of values boundary of the matrix ``A`` (MATLAB ``fov``).

    Parameters
    ----------
    A : array_like, shape (n, n)
    pref : optional
        Constructor preferences; domain and splitting follow native FOV overrides.
    line_segments : bool
        If True return ``(f, lineSegs, theta)`` like MATLAB's three-output
        form. The Python adapter returns ``lineSegs`` as a list of scalar
        Chebfuns on [-1, 1], each joining the left and right limits of
        ``f`` at a jump; ``theta`` contains the angles of those jumps.

    Provenance
    ----------
    MATLAB source : fov.m
    Chebfun commit: 7574c77
    """
    A = jnp.atleast_2d(jnp.asarray(A))
    f = chebfun(lambda t: _fov_curve(t, A), pref=pref,
                domain=(0.0, 2.0 * jnp.pi), splitting=True)
    f = f.merge()
    if not line_segments:
        return f
    ends = jnp.asarray(list(f.domain.breakpoints), dtype=jnp.float64)
    delta = 1e-14
    if jnp.any(jnp.diff(ends) < delta):
        delta = float(jnp.min(jnp.diff(ends))) / 3
    left = jnp.asarray(f(jnp.asarray(ends[:-1]), "left")).ravel().astype(jnp.complex128)
    left = left.at[0].set(jnp.asarray(f(ends[-1], "left")).reshape(()))
    right = jnp.asarray(f(jnp.asarray(ends[:-1]), "right")).ravel().astype(jnp.complex128)
    tol = 10 * delta * jnp.maximum(jnp.abs(left), jnp.abs(right))
    discont = jnp.abs(left - right) > tol
    theta = ends[:-1][discont]
    cols = [Chebfun.from_values(jnp.asarray([lv, rv]), domain=(-1.0, 1.0))
            for lv, rv in zip(left[discont], right[discont])]
    return f, cols, theta
