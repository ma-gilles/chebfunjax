"""Source sphere singular-value path; singular functions are outside this draft.

Provenance
----------
MATLAB source : @spherefun/svd.m, @chebfun/qr.m, @trigtech/qr.m,
               @trigtech/innerProduct.m, @bndfun/qr.m, @separableApprox/cdr.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""

import jax.numpy as jnp

from chebfunjax.tech.trigtech import (
    _trig_coeffs2vals_impl,
    _trig_eval,
    _trig_prolong_coeffs,
)
from chebfunjax.utils.quadrature import legpts


def _positive_diagonal_r(values):
    """Literal source QR sign convention; return only the triangular factor."""
    _q, r = jnp.linalg.qr(values, mode="reduced")
    diagonal = jnp.diag(r)
    signs = jnp.where(diagonal == 0, jnp.ones_like(diagonal), jnp.sign(diagonal))
    return signs[:, None] * r


def _values_at_length(tech, n):
    values = _trig_coeffs2vals_impl(_trig_prolong_coeffs(tech.coeffs, n))
    return jnp.real(values) if tech.is_real else values


def _row_r(rows):
    """Source physical [-pi,pi] QR, including the one-column inner product."""
    nf = max(row.coeffs.shape[0] for row in rows)
    if len(rows) == 1:
        # @chebfun/qr short-circuits before FUN QR for one column.
        # @trigtech/innerProduct prolongs to sum of the two lengths.
        n = 2 * nf
        values = _values_at_length(rows[0], n)
        inner = jnp.vdot((2.0 / n) * values, values)
        # Source equal-operand innerProduct forces a nonnegative diagonal;
        # the BNDFUN inner product then rescales by half the domain length.
        return jnp.sqrt(jnp.abs(inner) * jnp.pi).reshape((1, 1))
    n = max(nf, len(rows))
    values = jnp.column_stack([_values_at_length(row, n) for row in rows])
    if all(row.is_real for row in rows):
        values = jnp.real(values)
    r = _positive_diagonal_r(values)
    # Keep the two source scale operations distinct.
    r = jnp.sqrt(jnp.asarray(2.0 / n)) * r
    return r * jnp.sqrt(jnp.pi)


def singular_values(f):
    """Source weighted QR/CDR singular values, with unconditional JAX math.

    Provenance
    ----------
    MATLAB source : @spherefun/svd.m (sphereQR and reduced SVD)
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df

    Shapes and is_real flags are static metadata. JIT-disabled execution
    still calls the JAX transform body, never its NumPy dispatch wrapper.
    Complex SVD/QR differentiation and degenerate singular values have
    the underlying JAX restrictions; no general AD guarantee is added.
    """
    if f.isempty() or not f.cols:
        return jnp.empty((0,), dtype=jnp.float64)
    n = max(col.coeffs.shape[0] for col in f.cols) + 9
    x, w = legpts(n, (0.0, jnp.pi))
    c = jnp.column_stack([
        _trig_eval(col.coeffs, x / jnp.pi, is_real=col.is_real)
        for col in f.cols
    ])
    rc = _positive_diagonal_r(jnp.sqrt(w * jnp.sin(x))[:, None] * c)
    rr = _row_r(f.rows)
    reciprocal = 1.0 / jnp.asarray(f.pivots)
    reciprocal = jnp.where(jnp.isinf(jnp.abs(reciprocal)), 0, reciprocal)
    # Source uses nonconjugating transpose (.'), not Hermitian transpose.
    core = rc @ jnp.diag(reciprocal) @ rr.T
    return jnp.linalg.svd(core, compute_uv=False)
