"""Continuous small-core eigendecomposition from @chebfun2/eig.m.

Chebfun commit: 7574c77
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
Python full outputs use Quasimatrix/diagonal array; provider eigenvalue order
and eigenfunction phases are retained. Source SVD complex conventions remain.
"""
import jax.numpy as jnp

from chebfunjax.chebfun1d.linalg import Quasimatrix
from chebfunjax.chebfun2d._svd import source_svd


def source_eig(approx, *, normalize=False):
    """Literal V.H*U*S, then U*S*small eigenvectors; continuous norms."""
    left, singular, right = source_svd(approx, full=True, as_array=True)
    if left.isempty():
        return jnp.empty((0,), dtype=jnp.complex128), jnp.empty((0, 0))
    u, v = left, right
    diagonal = jnp.diag(singular)
    core = (v.H @ u) @ diagonal
    values, vectors = jnp.linalg.eig(core)
    lifted = (u @ diagonal) @ vectors
    columns = list(lifted.cols) if isinstance(lifted, Quasimatrix) else lifted.mat2cell()
    if normalize:
        columns = [column / column.norm() for column in columns]
    return values, Quasimatrix(columns, left.domain)
