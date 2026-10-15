"""Source panel factorization with scaled-reflector fallback."""

import jax
import jax.numpy as jnp

from ._jax_qr import _add, _div, _mul, _neg, _scaled_hypot
from ._jax_qr_padded import _gemv_padded, _ordinary_dynamic, _shifted_column, _write_shifted_column


@jax.jit
def panel_factor(a):
    a = jnp.asarray(a, jnp.float64)
    m, n = a.shape
    rows = jnp.arange(m)
    cols = jnp.arange(n)
    taus = jnp.zeros(n, jnp.float64)
    compact = jnp.zeros((n, n), jnp.float64)
    lower = jnp.finfo(jnp.float64).tiny / (jnp.finfo(jnp.float64).eps / 2)

    def reflector(j, state):
        mat, tau, t = state
        length = m - j - 1
        tail = _shifted_column(mat, j, j + 1, length)
        gram = _gemv_padded(mat, tail, j + 1, length, 0, n)
        bad = jnp.any((cols >= j) & ((~jnp.isfinite(gram)) | (jnp.abs(gram) < lower)))

        def normal(st):
            mat, tau, t = st
            signed = jnp.copysign(_scaled_hypot(jnp.sqrt(gram[j]), mat[j, j]), mat[j, j])
            denominator = _add(mat[j, j], signed)
            tau_j = _div(denominator, signed)
            inverse = _div(1.0, denominator)
            values = _mul(_neg(_add(_mul(gram, inverse), mat[j, :])), tau_j)
            normalized = _mul(tail, inverse)
            mat = _write_shifted_column(mat, j, j + 1, length, normalized).at[j, j].set(1.0)
            update = _add(mat, _mul(mat[:, j, None], values[None, :]))
            mat = (
                jnp.where((rows >= j)[:, None] & (cols > j)[None, :], update, mat)
                .at[j, j]
                .set(_neg(signed))
            )

            def column(c, v):
                value = v[c]
                update = _add(v, _mul(value, t[:, c]))
                out = jnp.where(cols < c, update, v).at[c].set(_mul(value, t[c, c]))
                return jnp.where(c < j, out, v)

            v = jax.lax.fori_loop(0, n, column, values)
            t = t.at[:, j].set(jnp.where(cols < j, v, jnp.where(cols == j, tau_j, t[:, j])))
            return mat, tau.at[j].set(tau_j), t

        def exceptional(st):
            mat, tau, t = st
            beta, tau_j, normalized = _ordinary_dynamic(mat[j, j], tail, length, False)
            mat = _write_shifted_column(mat, j, j + 1, length, normalized).at[j, j].set(1.0)
            v = _shifted_column(mat, j, j, m - j)
            dots = _gemv_padded(mat, v, j, m - j, 0, n)
            values = _mul(_neg(tau_j), dots)
            update = _add(mat, _mul(mat[:, j, None], values[None, :]))
            mat = (
                jnp.where((rows >= j)[:, None] & (cols > j)[None, :], update, mat)
                .at[j, j]
                .set(beta)
            )

            def column(c, v):
                value = v[c]
                update = _add(v, _mul(value, t[:, c]))
                out = jnp.where(cols < c, update, v).at[c].set(_mul(value, t[c, c]))
                return jnp.where(c < j, out, v)

            values = jax.lax.fori_loop(0, n, column, values)
            t = t.at[:, j].set(jnp.where(cols < j, values, jnp.where(cols == j, tau_j, t[:, j])))
            return mat, tau.at[j].set(tau_j), t

        return jax.lax.cond(bad, exceptional, normal, state)

    return jax.lax.fori_loop(0, n, reflector, (a, taus, compact))
