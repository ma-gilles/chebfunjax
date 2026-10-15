"""Real binary64 factorization and reduced Q generation.

Uses the unblocked, nested-panel and Q-generation source paths.
Dtype dispatch and differentiation are defined in the package entry point.
"""

import jax.numpy as jnp

from ._jax_qr_blocked import _apply_block, _orgqr, source_crossover, source_factor_block
from ._jax_qr_padded import factor_general_padded
from ._nested_panel import nested_panel


def factor_real(matrix):
    a = jnp.asarray(matrix)
    if a.ndim != 2 or a.dtype != jnp.float64:
        raise TypeError("Candidate requires a real binary64 matrix")
    m, n = a.shape
    rank = min(m, n)
    tau = jnp.zeros(rank, jnp.float64)
    if rank == 0:
        return a, tau
    start = 0
    block = source_factor_block(m, rank)
    while rank - start > max(block, source_crossover(m - start, rank - start)):
        end = start + block
        v, t, T = nested_panel(a[start:, start:end])
        a = a.at[start:, start:end].set(v)
        tau = tau.at[start:end].set(t)
        a = a.at[start:, end:].set(_apply_block(v, T, a[start:, end:], transpose=True))
        start = end
        block = min(block, source_factor_block(m - start, rank - start))
    tail, t = factor_general_padded(a[start:, start:])
    return a.at[start:, start:].set(tail), tau.at[start:].set(t)


def qr_real(matrix):
    a, tau = factor_real(matrix)
    rank = min(a.shape)
    if rank == 0:
        return jnp.zeros((a.shape[0], 0), a.dtype), a[:0, :]
    return _orgqr(a[:, :rank], tau), jnp.triu(a[:rank, :])
