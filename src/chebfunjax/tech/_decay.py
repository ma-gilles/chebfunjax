"""Native Chebtech endpoint-decay predicate.

Provenance: @chebtech/isdecay.m, Chebfun commit 7574c77.
"""
import jax
import jax.numpy as jnp

_EPS = float(jnp.finfo(jnp.float64).eps)


def chebtech_isdecay(onefun) -> jax.Array:
    """Return two endpoint flags per column using native root extraction."""
    if getattr(onefun, "_is_empty_object", False):
        return jnp.zeros((2, 0), dtype=jnp.bool_)
    coeffs = jnp.asarray(onefun.coeffs)
    if coeffs.size == 0:
        ncols = coeffs.shape[1] if coeffs.ndim == 2 else 0
        return jnp.zeros((2, ncols), dtype=jnp.bool_)
    if coeffs.ndim == 1:
        coeffs = coeffs[:, None]
    n, ncols = coeffs.shape
    scales = jnp.asarray(onefun.vscale_columns, dtype=jnp.float64).reshape((ncols,))
    tol = 1e2 * _EPS * scales

    if n == 1:
        # MATLAB ordering compares real parts, including complex constants.
        mask = (jnp.real(coeffs[0]) < tol) | (coeffs[0] == 0)
        return jnp.broadcast_to(mask[None, :], (2, ncols))

    end_values = jnp.asarray(onefun(jnp.asarray([-1.0, 1.0], dtype=jnp.float64)))
    if end_values.ndim == 1:
        end_values = end_values[:, None]
    endpoint_roots = jnp.abs(end_values) < tol[None, :]
    flags = jnp.zeros((2, ncols), dtype=jnp.bool_)

    for side in (0, 1):
        roots = jnp.zeros((2, ncols), dtype=jnp.int32)
        roots = roots.at[side].set(endpoint_roots[side].astype(jnp.int32))
        if bool(jnp.any(roots[side] > 0)):
            peeled, _, _ = onefun.extractBoundaryRoots(roots)
            residual = jnp.abs(peeled(jnp.asarray(-1.0 if side == 0 else 1.0)))
            residual = jnp.asarray(residual).reshape((ncols,))
            flags = flags.at[side].set(
                residual < 1e4 * tol
            )
    return flags

