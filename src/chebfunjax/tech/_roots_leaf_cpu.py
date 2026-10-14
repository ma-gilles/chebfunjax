"""Reuse CPU leaf setup while retaining exact-sized source eigensystems.

Provenance
----------
MATLAB source : @chebtech/roots.m, default colleague construction
Chebfun commit: 7574c77
"""

import jax
import jax.numpy as jnp

# uses-numpy: transfer/storage buffers and inherited exception class only.
import numpy as np


@jax.jit
def _matrix(c, n):
    """Build the literal source matrix in fixed-capacity transfer storage."""
    nn = n - 1
    product = jax.lax.optimization_barrier(-0.5 * c[:-1])
    divisor = jax.lax.optimization_barrier(jnp.broadcast_to(c[n - 1], product.shape))
    adjusted = jax.lax.optimization_barrier(product / divisor)
    amended = jax.lax.optimization_barrier(adjusted[n - 3] + 0.5)
    adjusted = adjusted.at[n - 3].set(amended)
    row = jnp.arange(49)
    matrix = jnp.where(jnp.abs(row[:, None] - row[None, :]) == 1, 0.5, 0.)
    matrix = matrix.at[nn - 2, nn - 1].set(1.)
    matrix = matrix.at[:, 0].set(adjusted[jnp.maximum(nn - 1 - row, 0)])
    active = (row[:, None] < nn) & (row[None, :] < nn)
    matrix = jnp.where(active, matrix, 0.)
    return matrix, jnp.all(jnp.isfinite(matrix))


@jax.jit
def _metadata(roots):
    """Preserve finite-output and real-spectrum adapter classification."""
    return jnp.all(jnp.isfinite(roots)), jnp.any(jnp.imag(roots) != 0)


def _eigenvalues(c, device, eigvals):
    """Eager CPU float64 adapter for coefficient counts 3 through 50.

    Only storage is padded: the eigensolver sees the original (n-1)-square
    matrix. Its dimension, entries, provider order and root filters are unchanged.
    """
    n = len(c)
    padded = np.zeros(50, dtype=np.float64)
    padded[:n] = c
    matrix, finite = _matrix(jax.device_put(padded, device),
                             jax.device_put(np.asarray(n, dtype=np.int64), device))
    if not bool(finite):
        raise np.linalg.LinAlgError("Array must not contain infs or NaNs")
    prefix = jax.device_get(matrix)[:n - 1, :n - 1].copy()
    roots = jax.device_get(eigvals(jax.device_put(prefix, device)))
    storage = np.zeros(49, dtype=np.complex128)
    storage[:len(roots)] = roots
    finite, imaginary = _metadata(jax.device_put(storage, device))
    if not bool(finite):
        raise np.linalg.LinAlgError("Eigenvalues did not converge (nonfinite JAX eig output)")
    return roots.copy() if bool(imaginary) else roots.real.copy()
