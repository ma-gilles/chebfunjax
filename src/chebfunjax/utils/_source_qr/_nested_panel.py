"""Source DLAQRF panel composition using JAX primitive operations."""

import jax.numpy as jnp

from ._jax_qr import _neg
from ._jax_qr_blocked import _apply_block, _gemm_source, _left_upper, _trmm_source
from ._panel_dynamic import panel_factor


def nested_panel(a):
    a = jnp.asarray(a, jnp.float64)
    m, n = a.shape
    taus = jnp.zeros(n, jnp.float64)
    compact = jnp.zeros((n, n), jnp.float64)
    for start in range(0, n, 16):
        end = min(start + 16, n)
        v, t, T = panel_factor(a[start:, start:end])
        a = a.at[start:, start:end].set(v)
        taus = taus.at[start:end].set(t)
        compact = compact.at[start:end, start:end].set(T)
        if start:
            cross = a[start:end, :start].T
            cross = _trmm_source(a[start:end, start:end], cross, uplo="L", trans="N", diag="U")
            cross = _gemm_source(a[end:, :start], a[end:, start:end], cross, ta="C")
            cross = _left_upper(compact[:start, :start], cross)
            cross = _neg(_trmm_source(T, cross, uplo="U", trans="N", diag="N"))
            compact = compact.at[:start, start:end].set(cross)
        if end < n:
            a = a.at[start:, end:].set(_apply_block(v, T, a[start:, end:], transpose=True))
    return a, taus, compact
