"""Source-order binary64 arithmetic and vector reductions for real QR."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._add import add_binary64
from ._divide import divide_binary64
from ._multiply import multiply_binary64


def _add(a, b):
    return add_binary64(a, b)[0]


def _mul(a, b):
    return multiply_binary64(a, b)[0]


def _div(a, b):
    return divide_binary64(a, b)[0]


def _neg(a):
    bits = jax.lax.bitcast_convert_type(jnp.asarray(a, jnp.float64), jnp.uint64)
    return jax.lax.bitcast_convert_type(bits ^ jnp.uint64(1 << 63), jnp.float64)


def _dot(a, x, kind):
    """Reproduce the captured two/four-lane native GEMV reduction order."""
    m = a.shape[0]
    seed = _mul(a[0], x[0])
    if m < 9:

        def add_one(i, total):
            return _add(total, _mul(a[i], x[i]))

        return jax.lax.fori_loop(1, m, add_one, seed)

    stop = 1 + ((m - 1) // 8) * 8
    lanes = jnp.zeros((kind, 2), dtype=jnp.float64).at[0, 0].set(seed)

    def block(block_index, lane_values):
        start = 1 + block_index * 8

        def pair(pair_index, acc):
            lane = pair_index % kind

            def word(word_index, inner):
                i = start + pair_index * 2 + word_index
                p = _mul(a[i], x[i])
                return inner.at[lane, word_index].set(_add(inner[lane, word_index], p))

            return jax.lax.fori_loop(0, 2, word, acc)

        return jax.lax.fori_loop(0, 4, pair, lane_values)

    lanes = jax.lax.fori_loop(0, (stop - 1) // 8, block, lanes)
    if kind == 4:
        first = jnp.stack((_add(lanes[0, 0], lanes[1, 0]), _add(lanes[0, 1], lanes[1, 1])))
        second = jnp.stack((_add(lanes[2, 0], lanes[3, 0]), _add(lanes[2, 1], lanes[3, 1])))
        combined = jnp.stack((_add(first[0], second[0]), _add(first[1], second[1])))
    elif kind == 2:
        combined = jnp.stack((_add(lanes[0, 0], lanes[1, 0]), _add(lanes[0, 1], lanes[1, 1])))
    else:
        combined = lanes[0]
    total = _add(combined[0], combined[1])

    def tail(i, acc):
        return _add(acc, _mul(a[i], x[i]))

    return jax.lax.fori_loop(stop, m, tail, total)


def _abs_bits(x):
    """Magnitude bits preserve comparisons for finite subnormal operands."""
    return jax.lax.bitcast_convert_type(jnp.asarray(x, jnp.float64), jnp.uint64) & jnp.uint64(
        0x7FFFFFFFFFFFFFFF
    )


def _scaled_hypot(a, b):
    aa = _abs_bits(a)
    bb = _abs_bits(b)
    hi = jax.lax.bitcast_convert_type(jnp.maximum(aa, bb), jnp.float64)
    lo = jax.lax.bitcast_convert_type(jnp.minimum(aa, bb), jnp.float64)
    ratio = _div(lo, hi)
    square = _mul(ratio, ratio)
    inside = _add(jnp.asarray(1.0, jnp.float64), square)
    return _mul(hi, jnp.sqrt(inside))


def _pair_dot(a, x, length=None):
    """Source two-lane GEMV reduction used by groups of four columns."""
    if a.shape[0] == 0:
        return jnp.asarray(0.0, jnp.float64)
    if length is None:
        length = jnp.asarray(a.shape[0], jnp.int64)
    lanes = jnp.stack((_mul(a[0], x[0]), jnp.asarray(0.0, jnp.float64)))
    stop = 1 + ((length - 1) // 2) * 2

    def block(start_index, pair):
        start = 1 + start_index * 2
        first = _add(pair[0], _mul(a[start], x[start]))
        second = _add(pair[1], _mul(a[start + 1], x[start + 1]))
        return jnp.stack((first, second))

    lanes = jax.lax.fori_loop(0, (stop - 1) // 2, block, lanes)
    total = _add(lanes[0], lanes[1])

    def tail(i, acc):
        return _add(acc, _mul(a[i], x[i]))

    return jax.lax.fori_loop(stop, length, tail, total)


def _gemv_columns(a, x):
    """Source column-group dispatch for the captured native GEMV kernel."""
    m, n = a.shape
    if m == 0 or n == 0:
        return jnp.zeros((n,), dtype=jnp.float64)
    full = (n // 4) * 4
    output = [_pair_dot(a[:, j], x) for j in range(full)]
    remaining = n - full
    for j in range(remaining):
        kind = (2 if j == 0 else 1) if remaining >= 2 and j < 2 else 4
        output.append(_dot(a[:, full + j], x, kind))
    return jnp.stack(output)
