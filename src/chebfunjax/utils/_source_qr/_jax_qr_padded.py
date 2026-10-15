"""Fixed-shape dynamic loops for source-order real QR."""

import jax
import jax.numpy as jnp

from ._jax_qr import _abs_bits, _add, _div, _mul, _neg, _scaled_hypot


def _shifted_column(a, column, row_start, length):
    m = a.shape[0]
    offsets = jnp.arange(m, dtype=jnp.int32)
    rows = (offsets + row_start) % m
    values = a[rows, column]
    return jnp.where(offsets < length, values, jnp.asarray(0.0, jnp.float64))


def _write_shifted_column(a, column, row_start, length, values):
    m = a.shape[0]
    rows = jnp.arange(m, dtype=jnp.int32)
    offsets = (rows - row_start) % m
    valid = (rows >= row_start) & (rows < row_start + length)
    col = jnp.where(valid, values[offsets], a[:, column])
    return a.at[:, column].set(col)


def _dot_dynamic(a, x, length, kind):
    m = a.shape[0]
    z = jnp.asarray(0.0, jnp.float64)

    def empty(_):
        return z

    def nonempty(_):
        seed = _mul(a[0], x[0])

        def short(_):
            def add_one(i, total):
                return jax.lax.cond(
                    i < length,
                    lambda v: _add(v, _mul(a[i], x[i])),
                    lambda v: v,
                    total,
                )

            return jax.lax.fori_loop(1, m, add_one, seed)

        def long(_):
            stop = 1 + ((length - 1) // 8) * 8
            lanes = jnp.zeros((4, 2), jnp.float64).at[0, 0].set(seed)

            def block(bi, lane_values):
                base = 1 + bi * 8

                def pair(pi, acc):
                    lane = pi % kind

                    def word(wi, inner):
                        i = base + 2 * pi + wi
                        product = _mul(a[i], x[i])
                        old = inner[lane, wi]
                        return inner.at[lane, wi].set(_add(old, product))

                    return jax.lax.fori_loop(0, 2, word, acc)

                return jax.lax.fori_loop(0, 4, pair, lane_values)

            lanes = jax.lax.fori_loop(0, (stop - 1) // 8, block, lanes)
            pair0 = jnp.stack((_add(lanes[0, 0], lanes[1, 0]), _add(lanes[0, 1], lanes[1, 1])))
            pair1 = jnp.stack((_add(lanes[2, 0], lanes[3, 0]), _add(lanes[2, 1], lanes[3, 1])))
            four = jnp.stack((_add(pair0[0], pair1[0]), _add(pair0[1], pair1[1])))
            two = jnp.stack((_add(lanes[0, 0], lanes[1, 0]), _add(lanes[0, 1], lanes[1, 1])))
            one = lanes[0]
            combined = jax.lax.cond(
                kind == 4,
                lambda _: four,
                lambda _: jax.lax.cond(kind == 2, lambda __: two, lambda __: one, None),
                None,
            )
            total = _add(combined[0], combined[1])

            def add_tail(i, acc):
                return jax.lax.cond(
                    i < length,
                    lambda v: _add(v, _mul(a[i], x[i])),
                    lambda v: v,
                    acc,
                )

            return jax.lax.fori_loop(stop, m, add_tail, total)

        return jax.lax.cond(length < 9, short, long, None)

    return jax.lax.cond(length <= 0, empty, nonempty, None)


def _pair_dot_dynamic(a, x, length):
    m = a.shape[0]
    z = jnp.asarray(0.0, jnp.float64)

    def empty(_):
        return z

    def nonempty(_):
        initial = jnp.stack((_mul(a[0], x[0]), z))
        stop = 1 + ((length - 1) // 2) * 2

        def pair(start_index, lanes):
            i = 1 + 2 * start_index
            first = _add(lanes[0], _mul(a[i], x[i]))
            second = _add(lanes[1], _mul(a[i + 1], x[i + 1]))
            return jnp.stack((first, second))

        lanes = jax.lax.fori_loop(0, (stop - 1) // 2, pair, initial)
        total = _add(lanes[0], lanes[1])

        def tail(i, acc):
            return jax.lax.cond(i < length, lambda v: _add(v, _mul(a[i], x[i])), lambda v: v, acc)

        return jax.lax.fori_loop(stop, m, tail, total)

    return jax.lax.cond(length <= 0, empty, nonempty, None)


def _gemv_padded(a, x, row_start, length, col_start, col_count):
    m, n = a.shape
    full = (col_count // 4) * 4
    remaining = col_count - full
    result = jnp.zeros((n,), jnp.float64)

    def column(c, output):
        relative = c - col_start
        active = (relative >= 0) & (relative < col_count)
        offset = jnp.arange(m, dtype=jnp.int32)
        rows = (offset + row_start) % m
        vec = jnp.where(offset < length, a[rows, c], 0.0)
        if_x = x

        def pair(_):
            return _pair_dot_dynamic(vec, if_x, length)

        def rem(_):
            rem_index = relative - full
            kind = jnp.where(
                (remaining >= 2) & (rem_index == 0),
                2,
                jnp.where((remaining >= 2) & (rem_index == 1), 1, 4),
            )
            return _dot_dynamic(vec, if_x, length, kind)

        value = jax.lax.cond(relative < full, pair, rem, None)
        return output.at[c].set(jnp.where(active, value, output[c]))

    return jax.lax.fori_loop(0, n, column, result)


def _norm_dynamic(x, length):
    """Keep the scale in integer representation across the reduction loop."""
    m = x.shape[0]

    def step(i, state):
        scale_bits, ssq = state
        value_bits = _abs_bits(x[i])
        scale = jax.lax.bitcast_convert_type(scale_bits, jnp.float64)
        value = jax.lax.bitcast_convert_type(value_bits, jnp.float64)

        def nonzero(_):
            def increase(_):
                ratio = _div(scale, value)
                return value_bits, _add(_mul(_mul(ssq, ratio), ratio), 1.0)

            def accumulate(_):
                ratio = _div(value, scale)
                return scale_bits, _add(ssq, _mul(ratio, ratio))

            return jax.lax.cond(scale_bits < value_bits, increase, accumulate, None)

        return jax.lax.cond((i < length) & (value_bits != 0), nonzero, lambda _: state, None)

    bits, ssq = jax.lax.fori_loop(0, m, step, (jnp.uint64(0), jnp.float64(1.0)))
    return _mul(jax.lax.bitcast_convert_type(bits, jnp.float64), jnp.sqrt(ssq))


def _small_dot_dynamic(a, b, length):
    """Source zero-seeded four-pair small-body reduction with active length."""
    m = a.shape[0]
    lanes = jnp.zeros((4, 2), jnp.float64)
    stop = (length // 8) * 8

    def block(bi, lane_values):
        base = bi * 8

        def pair(pi, acc):
            def word(wi, inner):
                index = base + 2 * pi + wi
                product = _mul(a[index], b[index])
                return inner.at[pi, wi].set(_add(inner[pi, wi], product))

            return jax.lax.fori_loop(0, 2, word, acc)

        return jax.lax.fori_loop(0, 4, pair, lane_values)

    lanes = jax.lax.fori_loop(0, stop // 8, block, lanes)
    pair0lo = _add(lanes[0, 0], lanes[1, 0])
    pair1lo = _add(lanes[2, 0], lanes[3, 0])
    pair0hi = _add(lanes[0, 1], lanes[1, 1])
    pair1hi = _add(lanes[2, 1], lanes[3, 1])
    low = _add(pair0lo, pair1lo)
    high = _add(pair0hi, pair1hi)
    total = _add(low, high)

    def tail(i, acc):
        return jax.lax.cond(
            i < length, lambda value: _add(value, _mul(a[i], b[i])), lambda value: value, acc
        )

    return jax.lax.fori_loop(stop, m, tail, total)


def _ordinary_dynamic(alpha, tail, length, direct):
    z = jnp.asarray(0.0, jnp.float64)
    safe = jnp.finfo(jnp.float64).tiny / (jnp.finfo(jnp.float64).eps / 2.0)
    reciprocal = _div(1.0, safe)
    xnorm = _norm_dynamic(tail, length)

    def zero(_):
        return alpha, z, tail

    def nonzero(_):
        beta0 = _neg(jnp.copysign(_scaled_hypot(alpha, xnorm), alpha))
        state0 = (alpha, beta0, tail, jnp.asarray(0, jnp.int32))

        def condition(state):
            _, beta, _, count = state
            return (jnp.abs(beta) < safe) & (count < 20)

        def scale(state):
            aa, bb, vv, count = state
            return _mul(aa, reciprocal), _mul(bb, reciprocal), _mul(vv, reciprocal), count + 1

        aa, _, vv, count = jax.lax.while_loop(condition, scale, state0)
        bb = _neg(jnp.copysign(_scaled_hypot(aa, _norm_dynamic(vv, length)), aa))
        tau = _div(_add(bb, _neg(aa)), bb)
        denom = _add(aa, _neg(bb))
        normalized = _div(vv, denom) if direct else _mul(vv, _div(1.0, denom))
        beta = jax.lax.fori_loop(0, count, lambda _, b: _mul(b, safe), bb)
        return beta, tau, normalized

    return jax.lax.cond(_abs_bits(xnorm) == 0, zero, nonzero, None)


@jax.jit
def factor_general_padded(matrix):
    """One JAX-loop body with padded rows/columns and dynamic active bounds."""
    a = jnp.asarray(matrix, jnp.float64)
    m, n = a.shape
    rank = min(m, n)
    small = max(m, n) <= 20
    lower = jnp.finfo(jnp.float64).tiny / (jnp.finfo(jnp.float64).eps / 2.0)
    tau = jnp.zeros((rank,), jnp.float64)
    rows = jnp.arange(m, dtype=jnp.int32)
    cols = jnp.arange(n, dtype=jnp.int32)

    def factor(j, state):
        mat, taus = state
        alpha = mat[j, j]
        tail_length = m - j - 1
        tail = _shifted_column(mat, j, j + 1, tail_length)
        if small:
            beta, tau_j, normalized = _ordinary_dynamic(alpha, tail, tail_length, True)

            def skip(st):
                matrix, tau_values = st
                return matrix, tau_values.at[j].set(tau_j)

            def reflect(st):
                matrix, tau_values = st
                matrix = _write_shifted_column(matrix, j, j + 1, tail_length, normalized)
                matrix = matrix.at[j, j].set(1.0)
                row_length = m - j
                x = _shifted_column(matrix, j, j, row_length)
                updates = jnp.zeros((n,), jnp.float64)

                def update_column(c, values):
                    dot = _small_dot_dynamic(
                        x,
                        _shifted_column(matrix, c, j, row_length),
                        row_length,
                    )
                    factor = _mul(_neg(tau_j), dot)
                    return values.at[c].set(jnp.where(c > j, factor, 0.0))

                updates = jax.lax.fori_loop(0, n, update_column, updates)
                delta = _mul(matrix[:, j, None], updates[None, :])
                new_matrix = _add(matrix, delta)
                matrix = jnp.where(
                    (rows >= j)[:, None] & (cols > j)[None, :],
                    new_matrix,
                    matrix,
                )
                matrix = matrix.at[j, j].set(beta)
                return matrix, tau_values.at[j].set(tau_j)

            return jax.lax.cond(tau_j == 0.0, skip, reflect, (mat, taus))
        if m <= n:
            beta, tau_j, normalized = _ordinary_dynamic(alpha, tail, tail_length, False)

            def apply_ordinary(st):
                matrix, tau_values = st
                matrix = _write_shifted_column(matrix, j, j + 1, tail_length, normalized)
                matrix = matrix.at[j, j].set(1.0)
                v = _shifted_column(matrix, j, j, m - j)
                offsets = jnp.arange(m, dtype=jnp.int32)
                last = jnp.max(jnp.where((offsets < m - j) & (_abs_bits(v) != 0), offsets + 1, 0))
                dot = _gemv_padded(matrix, v, j, last, j + 1, n - j - 1)
                weights = _mul(_neg(tau_j), dot)
                updated = _add(matrix, _mul(matrix[:, j, None], weights[None, :]))
                matrix = jnp.where(
                    ((rows >= j) & (rows < j + last))[:, None] & (cols > j)[None, :],
                    updated,
                    matrix,
                )
                return matrix.at[j, j].set(beta), tau_values.at[j].set(tau_j)

            return jax.lax.cond(
                tau_j == 0.0,
                lambda st: (st[0].at[j, j].set(beta), st[1].at[j].set(tau_j)),
                apply_ordinary,
                (mat, taus),
            )
        gram = _gemv_padded(mat, tail, j + 1, tail_length, j, n - j)
        active_gram = cols >= j
        fallback = jnp.any(active_gram & ((~jnp.isfinite(gram)) | (jnp.abs(gram) < lower)))

        def fallback_body(st):
            mat, taus = st
            beta, tau_j, normalized = _ordinary_dynamic(mat[j, j], tail, tail_length, False)
            mat = _write_shifted_column(mat, j, j + 1, tail_length, normalized)
            mat = mat.at[j, j].set(1.0)
            length = m - j
            x = _shifted_column(mat, j, j, length)
            dot = _gemv_padded(mat, x, j, length, j + 1, n - j - 1)
            updates = _mul(_neg(tau_j), dot)
            active_rows = rows >= j
            active_cols = cols > j
            delta = _mul(mat[:, j, None], updates[None, :])
            new_mat = _add(mat, delta)
            mat = jnp.where(active_rows[:, None] & active_cols[None, :], new_mat, mat)
            mat = mat.at[j, j].set(beta)
            return mat, taus.at[j].set(tau_j)

        def normal_body(st):
            mat, taus = st
            signed = jnp.copysign(_scaled_hypot(jnp.sqrt(gram[j]), alpha), alpha)
            denom = _add(alpha, signed)
            tau_j = _div(denom, signed)
            inv = _div(1.0, denom)
            col = _mul(mat[:, j], inv)
            mat = mat.at[:, j].set(jnp.where(rows > j, col, mat[:, j]))
            mat = mat.at[j, j].set(1.0)
            updates = _mul(_neg(_add(_mul(gram, inv), mat[j, :])), tau_j)
            delta = _mul(mat[:, j, None], updates[None, :])
            new_mat = _add(mat, delta)
            mat = jnp.where((rows >= j)[:, None] & (cols > j)[None, :], new_mat, mat)
            mat = mat.at[j, j].set(_neg(signed))
            return mat, taus.at[j].set(tau_j)

        def fallback_branch(st):
            return fallback_body(st)

        def normal_branch(st):
            return normal_body(st)

        new_state = jax.lax.cond(fallback, fallback_branch, normal_branch, (mat, taus))
        return new_state

    a, tau = jax.lax.fori_loop(0, rank, factor, (a, tau))
    return a, tau
