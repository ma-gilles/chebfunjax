"""Literal original separable CDR action with physical source mapping.

Provenance: @separableApprox/{fevalm,cdr}.m, @mapping/mapping.m;
Chebfun7574c77680d7e82b79626300bf255498271a72df.
Copyright University of Oxford and the Chebfun Developers.
"""

import jax.numpy as jnp


def _slice_values(slices, points, a, b):
    """Evaluate original scalar technologies at physical vector points."""
    x = jnp.asarray(points)
    # Native bndfun.feval -> mapping.linear.Inv, preserving its literal order.
    t = (x - a) / (b - a) - (b - x) / (b - a)
    return jnp.stack([piece(t) for piece in slices], axis=-1)


def _mesh_values(approx, x, y):
    """Native fevalm: (C(y) * diag(d)) * R(x).', without conjugation."""
    xa, xb, ya, yb = approx.domain
    cols = _slice_values(approx.cols, y, ya, yb)
    rows = _slice_values(approx.rows, x, xa, xb)
    # Native cdr replaces infinite reciprocal pivots by zero.
    d = jnp.asarray(approx.pivots)
    d = jnp.where(jnp.isinf(jnp.abs(d)), 0, d)
    return (cols @ jnp.diag(d)) @ rows.T

