"""Continuous Tucker HOSVD using the native QR and mode-SVD sequence.

Provenance
----------
MATLAB source : @chebfun3/{hosvd,discreteHOSVD,unfold,txm}.m,
    @chebfun/{qr,horzcat,mtimes}.m
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford and
    The Chebfun Developers.
"""

import jax.numpy as jnp

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.chebfun1d.mtimes import _columns
from chebfunjax.domain import Domain


def _mode_product(tensor, matrix, axis):
    """Native tensor-times-matrix with the contracted mode restored."""
    return jnp.moveaxis(jnp.tensordot(matrix, tensor, axes=(1, axis)), 0, axis)


def source_hosvd(f, *, return_factors=False):
    """Apply three public continuous QRs followed by discrete HOSVD."""
    from chebfunjax.chebfun3d.chebfun3 import Chebfun3

    orthogonal, triangular = [], []
    for axis, factors in enumerate((f.cols, f.rows, f.tubes)):
        interval = tuple(f.domain[2*axis:2*axis+2])
        domain = Domain(interval)
        columns = [Chebfun([_Piece(tech, interval)], domain) for tech in factors]
        panel = Chebfun.horzcat(*columns)
        q, r = panel.qr()
        orthogonal.append(q)
        triangular.append(r)
    core = f.core
    for axis, r in enumerate(triangular):
        core = _mode_product(core, r, axis)
    modes = []
    for axis in range(3):
        order = (axis,)+tuple(i for i in range(3) if i != axis)
        unfolded = jnp.transpose(core, order).reshape((core.shape[axis], -1), order='F')
        u, _, _ = jnp.linalg.svd(unfolded, full_matrices=False)
        modes.append(u)
    for axis, u in enumerate(modes):
        core = _mode_product(core, jnp.conj(u).T, axis)
    singular_values = [
        jnp.linalg.norm(jnp.moveaxis(core, axis, 0).reshape((core.shape[axis], -1)), axis=1)
        for axis in range(3)
    ]
    panels = [q @ u for q, u in zip(orthogonal, modes)]
    if return_factors:
        return singular_values, core, *panels
    factors = [[column.funs[0].tech for column in _columns(panel)] for panel in panels]
    g = Chebfun3(cols=factors[0], rows=factors[1], tubes=factors[2], core=core, domain=f.domain)
    return singular_values, g
