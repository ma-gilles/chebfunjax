"""Dense QR with source-order real binary64 arithmetic.

Provenance: MATLAB dense QR called by @chebfun3/chebfun3f.m in Chebfun
commit 7574c77. Real kernels follow the captured CPU LAPACK/BLAS operation
order. Other dtypes retain JAX QR. The full-rank thin differential follows
A = Q R and Q.T Q = I, as in JAX's thin QR differential.
"""

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

from ._qr_api import qr_real


@jax.custom_jvp
def _real_qr(matrix):
    return qr_real(matrix)


@_real_qr.defjvp
def _real_qr_jvp(primals, tangents):
    (matrix,), (tangent,) = primals, tangents
    if matrix.shape[0] < matrix.shape[1]:
        raise NotImplementedError("Source QR differential requires rows >= columns")
    q, r = _real_qr(matrix)
    z = solve_triangular(r.T, tangent.T, lower=True).T
    projected = q.T @ z
    lower = jnp.tril(projected, -1)
    rotation = lower - lower.T
    return (q, r), (z + q @ (rotation - projected), (projected - rotation) @ r)


def qr(matrix):
    """Return reduced Q/R, preserving source order for real binary64 inputs.

    Integer, complex and other floating dtypes use JAX's existing dense QR.
    The real differential is defined for full-rank thin matrices; singular
    factors have the same nonunique differential limitation as dense QR.
    """
    matrix = jnp.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError("Dense source QR requires a matrix")
    if matrix.dtype == jnp.float64:
        return _real_qr(matrix)
    return jnp.linalg.qr(matrix, mode="reduced")


__all__ = ["qr"]
