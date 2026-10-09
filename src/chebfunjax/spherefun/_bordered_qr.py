"""Compact QR solve of source sphere band equations with one dense constraint.

Provenance: Chebfun7574c77680d7e82b79626300bf255498271a72df,
@spherefun/poisson.m134–136 and @spherefun/helmholtz.m131–133.
Givens QR is an equation-equivalent JAX algorithm, not native sparse LU.
No dense Q, dense matrix factorization, pivot flooring or coefficient threshold.
"""
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp


class BorderStatus(NamedTuple):
    rank_deficient: jax.Array
    constraint_singular: jax.Array
    nonfinite: jax.Array


def complex_givens(a, b):
    """Return real c, s, r for [[c,s],[-conj(s),c]] @ [a,b]=[r,0]."""
    aa, bb = jnp.abs(a), jnp.abs(b)
    norm = jnp.hypot(aa, bb)
    safe_norm = jnp.where(norm == 0, 1, norm)
    phase = jnp.where(aa == 0, jnp.ones_like(a), a/jnp.where(aa == 0, 1, aa))
    c = jnp.where(norm == 0, 1, aa/safe_norm)
    s = phase*jnp.conj(b)/safe_norm
    return c, s, phase*norm


@partial(jax.jit, static_argnames=('lower', 'upper', 'removed_row'))
def solve_bordered(ab, rhs, border, value, *, lower, upper, removed_row):
    """Solve [border; A[I,:]] x = [value; rhs[I]] in compact storage.

    I excludes removed_row. ab is UNFACTORED row storage of shape
    (m,2*lower+upper+1), matching the root pivoted band helper. Reserved LU
    fill is ignored. rhs may have multiple columns; value broadcasts over
    those columns. Separate failure flags must be checked by the caller.
    """
    ab, rhs, border, value = map(jnp.asarray, (ab, rhs, border, value))
    m = ab.shape[0]
    if (m < 2 or lower < 0 or upper < 0 or not 0 <= removed_row < m
            or ab.shape != (m, 2*lower+upper+1)):
        raise ValueError('invalid bordered band dimensions')
    if rhs.ndim not in (1, 2) or rhs.shape[0] != m or border.shape != (m,):
        raise ValueError('invalid border or RHS shape')
    vector = rhs.ndim == 1
    dtype = jnp.result_type(ab, rhs, border, value, jnp.float64)
    ab, rhs, border, value = (x.astype(dtype) for x in (ab, rhs, border, value))
    b = rhs[:, None] if vector else rhs
    count = b.shape[1]
    d = jnp.broadcast_to(value, (count,))
    finite_input = (jnp.all(jnp.isfinite(ab)) & jnp.all(jnp.isfinite(b))
                    & jnp.all(jnp.isfinite(border)) & jnp.all(jnp.isfinite(d)))
    indices = jnp.arange(m-1)
    original_rows = indices+(indices >= removed_row)
    b = b[original_rows]
    # C=A[I,:] has lower l/upper u+1; C* has lower p=u+1/upper q=l.
    p, q = upper+1, lower
    reach = p+q
    width = p+reach+1
    rows = jnp.arange(m)[:, None]
    columns = rows+jnp.arange(width)[None, :]-p
    original = columns+(columns >= removed_row)
    slots = rows-original+lower
    valid = ((columns >= 0) & (columns < m-1)
             & (slots >= 0) & (slots <= lower+upper))
    t = jnp.where(valid, jnp.conj(ab[jnp.clip(original, 0, m-1),
                                   jnp.clip(slots, 0, ab.shape[1]-1)]), 0)
    cosines = jnp.ones((m-1, p), dtype=jnp.abs(t).dtype)
    sines = jnp.zeros((m-1, p), dtype=dtype)
    offsets = jnp.arange(reach+1)

    def column(k, state):
        def eliminate(step, state):
            t, cs, ss = state
            i = k+p-step

            def rotate(state):
                t, cs, ss = state
                j = i-1
                c, s, r = complex_givens(t[j, k-j+p], t[i, k-i+p])
                cols = k+offsets
                top_slots, bottom_slots = cols-j+p, cols-i+p
                top, bottom = t[j, top_slots], t[i, bottom_slots]
                new_top = (c*top+s*bottom).at[0].set(r)
                new_bottom = (-jnp.conj(s)*top+c*bottom).at[0].set(0)
                # Drop only columns outside the rectangular C* matrix.
                t = t.at[jnp.where(cols < m-1, j, m), top_slots].set(
                    new_top, mode='drop')
                t = t.at[jnp.where(cols < m-1, i, m), bottom_slots].set(
                    new_bottom, mode='drop')
                return t, cs.at[k, step].set(c), ss.at[k, step].set(s)

            return jax.lax.cond(i < m, rotate, lambda state: state, (t, cs, ss))
        return jax.lax.fori_loop(0, p, eliminate, state)

    t, cosines, sines = jax.lax.fori_loop(0, m-1, column, (t, cosines, sines))
    diagonal = t[:m-1, p]
    rank_bad = jnp.any(diagonal == 0)
    distances = jnp.arange(1, reach+1)

    def triangular(k, y):
        previous = k-distances
        terms = jnp.conj(t[jnp.maximum(previous, 0), p+distances, None])
        terms = terms*y[jnp.maximum(previous, 0)]
        total = jnp.sum(jnp.where((previous >= 0)[:, None], terms, 0), axis=0)
        return y.at[k].set((b[k]-total)/jnp.conj(diagonal[k]))

    y = jax.lax.fori_loop(0, m-1, triangular, jnp.zeros((m, count), dtype=dtype))
    null = jnp.zeros((m, 1), dtype=dtype).at[-1, 0].set(1)
    transformed = jnp.concatenate((y, null), axis=1)

    def undo(index, values):
        flat = (m-1)*p-1-index
        k, step = flat//p, flat % p
        i = k+p-step

        def rotate(values):
            j = i-1
            c, s = cosines[k, step], sines[k, step]
            top, bottom = values[j], values[i]
            return values.at[j].set(c*top-s*bottom).at[i].set(
                jnp.conj(s)*top+c*bottom)

        return jax.lax.cond(i < m, rotate, lambda values: values, values)

    transformed = jax.lax.fori_loop(0, (m-1)*p, undo, transformed)
    particular, null = transformed[:, :count], transformed[:, -1]
    denominator = border @ null  # Source constraint is nonconjugate.
    alpha = (d-border @ particular)/denominator
    x = particular+null[:, None]*alpha[None, :]
    nonfinite = (~finite_input | ~jnp.all(jnp.isfinite(t))
                 | ~jnp.all(jnp.isfinite(cosines)) | ~jnp.all(jnp.isfinite(sines))
                 | ~jnp.all(jnp.isfinite(transformed))
                 | ~jnp.isfinite(denominator) | ~jnp.all(jnp.isfinite(x)))
    status = BorderStatus(rank_bad, denominator == 0, nonfinite)
    return (x[:, 0] if vector else x), status
