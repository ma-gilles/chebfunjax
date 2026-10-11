"""Source one-dimensional integration of separable approximations.

Provenance
----------
MATLAB source : @separableApprox/sum.m
Chebfun commit: 7574c77
"""
import jax.numpy as jnp

from ._svd import _as_fun, _axis_panel


def source_sum(approx, dim=None):
    """Integrate native CDR factors; preserve row/column orientation."""
    if not approx.cols:
        return jnp.asarray([])
    if dim is None:
        dim = 1
    if dim not in (1, 2):
        raise ValueError("SEPARABLEAPPROX:sum:unknown: Undefined dimension")
    xa, xb, ya, yb = approx.domain
    columns = _as_fun(_axis_panel(tuple(approx.cols)), (ya, yb))
    rows = _as_fun(_axis_panel(tuple(approx.rows)), (xa, xb))
    weights = jnp.diag(jnp.asarray(approx.pivots))
    if dim == 1:
        coefficients = (jnp.asarray(columns.sum()).reshape((1, -1)) @ weights).T
        result = (rows @ coefficients).mat2cell()[0].T
    else:
        coefficients = weights @ jnp.asarray(rows.sum()).reshape((-1, 1))
        result = (columns @ coefficients).mat2cell()[0]
    return result.simplify()
