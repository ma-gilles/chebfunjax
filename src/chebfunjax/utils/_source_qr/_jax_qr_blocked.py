"""Source-order real blocked QR and triangular matrix operations.

Block schedules and reduction orders reproduce the captured CPU reference.
The numerical operations use JAX binary64 arithmetic kernels.
"""

from functools import partial

import jax
import jax.numpy as jnp

from ._fma import fma_binary64
from ._jax_qr import _abs_bits, _add, _gemv_columns, _mul, _neg
from ._jax_qr_padded import _gemv_padded, _shifted_column


def _fma(a, b, c):
    return fma_binary64(a, b, c)[0]


def source_crossover(m, n):
    """DGEQRF source crossover for the archived CPU configuration."""
    if n <= 8:
        return 128
    return 8 if m > 4000 or m * n > 12500 else 128


def source_factor_block(m, n):
    """DGEQRF ILAENV block-size table derived from native queries."""
    n = min(m, n)
    if max(m, n) <= 10:
        return 1
    if n <= 300:
        return 16
    if n <= 350:
        return 24
    if n <= 800:
        return 48
    if n <= 1150:
        return 64
    if n <= 3400:
        return 96
    return 128


def source_q_block(m, n):
    """DORGQR source block size; n is the economy-Q column count."""
    if max(m, n) <= 10:
        return 1
    if m >= 10000:
        for limit, block in (
            (30, 16),
            (40, 32),
            (50, 16),
            (60, 24),
            (80, 32),
            (90, 40),
            (100, 32),
            (110, 24),
            (120, 56),
            (140, 32),
            (150, 48),
            (160, 24),
            (170, 32),
            (190, 24),
            (200, 32),
            (500, 48),
        ):
            if n <= limit:
                return block
        return 128
    if min(m, n) <= 64:
        return 16
    area = m * n
    if area < 256000:
        return 32
    if area < 1500000:
        return 64
    return 128


@partial(jax.jit, static_argnames=("ta", "tb", "alpha", "beta"))
def _gemm_source(a, b, c, *, ta="N", tb="N", alpha=1.0, beta=1.0):
    """Source-order real GEMM body used by the captured blocked QR paths."""
    aa = a.T if ta != "N" else a
    bb = b.T if tb != "N" else b
    m, k = aa.shape
    n = bb.shape[1]
    if m == 0 or n == 0:
        return c
    if k == 0:
        return _mul(c, beta) if beta != 0.0 else jnp.zeros_like(c)
    if m == 1 and ta in ("T", "C") and tb == "N":
        dots = _gemv_columns(bb, aa[0])
        initial = _mul(c[0], beta) if beta != 0.0 else jnp.zeros_like(c[0])
        return _add(initial, _mul(alpha, dots))[None, :]
    result = c
    special_tn = ta in ("T", "C") and tb == "N"
    fused_main = m >= 12 and n >= 4 and k >= 2

    def one_cell(index, out):
        i = index // n
        j = index % n
        init = _mul(out[i, j], beta) if beta != 0.0 else jnp.asarray(0.0, jnp.float64)

        def ordinary(_):
            def q_body(q, value):
                av, bv = aa[i, q], bb[q, j]
                if fused_main:
                    return jax.lax.cond(
                        (i < (m // 12) * 12) & (j < (n // 4) * 4),
                        lambda v: _fma(_mul(alpha, av), bv, v),
                        lambda v: _add(v, _mul(av, _mul(alpha, bv))),
                        value,
                    )
                return _add(value, _mul(av, _mul(alpha, bv)))

            value = jax.lax.fori_loop(0, k, q_body, init)
            return out.at[i, j].set(value)

        def tail_tn(_):
            rem_rows = m % 12
            local = i - (m // 12) * 12
            groups = jnp.where(local < (rem_rows // 4) * 4, 1, 4)
            width = 2 * groups
            chunk = min(256, (k + 1) // 2) if m >= 12 and k > 256 else k

            def chunk_body(ci, value):
                begin = ci * chunk
                end = jnp.minimum(begin + chunk, k)
                stop = ((end - begin) // width) * width
                lanes = jnp.zeros((4, 2), jnp.float64)

                def block(bi, sums):
                    def pair(p, pair_sums):
                        def lane(l, inner):
                            q = begin + bi * width + p * 2 + l
                            return inner.at[p, l].set(_add(inner[p, l], _mul(aa[i, q], bb[q, j])))

                        return jax.lax.fori_loop(0, 2, lane, pair_sums)

                    return jax.lax.fori_loop(0, groups, pair, sums)

                lanes = jax.lax.fori_loop(0, stop // width, block, lanes)

                def reduce_one(_):
                    return _add(lanes[0, 0], lanes[0, 1])

                def reduce_four(_):
                    lo = _add(_add(lanes[0, 0], lanes[1, 0]), _add(lanes[2, 0], lanes[3, 0]))
                    hi = _add(_add(lanes[0, 1], lanes[1, 1]), _add(lanes[2, 1], lanes[3, 1]))
                    return _add(lo, hi)

                value0 = jax.lax.cond(groups == 1, reduce_one, reduce_four, None)
                value0 = jax.lax.fori_loop(
                    begin + stop,
                    end,
                    lambda q, value: _add(value, _mul(aa[i, q], bb[q, j])),
                    value0,
                )
                return _add(value, _mul(alpha, value0))

            value = jax.lax.fori_loop(0, (k + chunk - 1) // chunk, chunk_body, init)
            return out.at[i, j].set(value)

        if special_tn:
            return jax.lax.cond(i >= (m // 12) * 12, tail_tn, ordinary, None)
        return ordinary(None)

    return jax.lax.fori_loop(0, m * n, one_cell, result)


@partial(jax.jit, static_argnames=("uplo", "trans", "diag"))
def _inner_source(a, b, *, uplo, trans, diag):
    """Right triangular diagonal block multiplication in native loop order."""
    aa = a.T if trans != "N" else a
    m, n = b.shape
    lower = (uplo == "L") == (trans == "N")
    result = b

    def cell(index, out):
        i, j = index // n, index % n
        initial = b[i, j] if diag == "U" else _mul(b[i, j], aa[j, j])
        if lower:
            start, step, count = j + 1, 1, n - j - 1
        elif uplo == "L" and trans == "T":
            start, step, count = j - 1, -1, j
        else:
            start, step, count = 0, 1, j

        def accumulate(index_q, value):
            q = start + index_q * step
            return _add(value, _mul(b[i, q], aa[q, j]))

        value = jax.lax.fori_loop(0, count, accumulate, initial)
        return out.at[i, j].set(value)

    return jax.lax.fori_loop(0, m * n, cell, result)


@partial(jax.jit, static_argnames=("uplo", "trans", "diag"))
def _trmm_source(a, b, *, uplo, trans, diag):
    """Source-order right TRMM, including its row and triangular-column tiles."""
    m, n = b.shape
    if m == 1 and n <= 64 and trans == "N":

        def column(j, out):
            seed = b[0, j] if diag == "U" else _mul(b[0, j], a[j, j])
            begin = j + 1 if uplo == "L" else 0
            length = n - j - 1 if uplo == "L" else j
            stop = length // 8 * 8

            def grouped(_):
                lanes = jnp.zeros((8,), jnp.float64).at[0].set(seed)

                def block(bi, sums):
                    def lane(l, sums):
                        q = begin + 8 * bi + l
                        return sums.at[l].set(_add(sums[l], _mul(b[0, q], a[q, j])))

                    return jax.lax.fori_loop(0, 8, lane, sums)

                lanes = jax.lax.fori_loop(0, stop // 8, block, lanes)
                lo = _add(_add(lanes[0], lanes[2]), _add(lanes[4], lanes[6]))
                hi = _add(_add(lanes[1], lanes[3]), _add(lanes[5], lanes[7]))
                return _add(lo, hi)

            value = jax.lax.cond(stop > 0, grouped, lambda _: seed, None)
            value = jax.lax.fori_loop(
                begin + stop, begin + length, lambda q, v: _add(v, _mul(b[0, q], a[q, j])), value
            )
            return out.at[0, j].set(value)

        return jax.lax.fori_loop(0, n, column, b)
    # Single-row right TRMM dispatches to opposite-transpose TRMV.
    # Native xdtrmv uses 64-column inner blocks, not TRMM's 16-column
    # blocks. Lower/N TRMV visits contributing columns in reverse order.
    if m == 1 and n <= 64 and uplo == "L" and trans == "T":
        return _inner_source(a, b, uplo=uplo, trans=trans, diag=diag)
    step = 8 if trans == "N" else 16
    lower = (uplo == "L") == (trans == "N")
    result = b
    for row in range(0, m, 32):
        row_end = min(row + 32, m)
        blocks = (
            [(col, min(col + step, n)) for col in range(0, n, step)]
            if lower
            else [(max(0, end - step), end) for end in range(n, 0, -step)]
        )
        for col, col_end in blocks:
            local_a = a[col:col_end, col:col_end]
            local_b = b[row:row_end, col:col_end]
            block = _inner_source(local_a, local_b, uplo=uplo, trans=trans, diag=diag)
            cross = slice(col_end, n) if lower else slice(0, col)
            cross_b = b[row:row_end, cross]
            cross_a = a[cross, col:col_end] if trans == "N" else a[col:col_end, cross]
            block = _gemm_source(
                cross_b,
                cross_a,
                block,
                tb=trans,
                alpha=1.0,
                beta=1.0,
            )
            result = result.at[row:row_end, col:col_end].set(block)
    return result


@partial(jax.jit, static_argnames=("transpose",))
def _apply_block(v, t, c, *, transpose=False):
    k = t.shape[0]
    result = c
    w = result[:k, :].T
    w = _trmm_source(v[:k, :], w, uplo="L", trans="N", diag="U")
    w = _gemm_source(result[k:, :], v[k:, :], w, ta="T")
    w = _trmm_source(t, w, uplo="U", trans="N" if transpose else "T", diag="N")
    lower = _gemm_source(v[k:, :], w, result[k:, :], tb="T", alpha=-1.0)
    result = result.at[k:, :].set(lower)
    w = _trmm_source(v[:k, :], w, uplo="L", trans="T", diag="U")
    top = result[:k, :]
    top = jax.lax.fori_loop(
        0,
        k * result.shape[1],
        lambda index, out: out.at[index // result.shape[1], index % result.shape[1]].set(
            _add(
                out[index // result.shape[1], index % result.shape[1]],
                _neg(w[index % result.shape[1], index // result.shape[1]]),
            )
        ),
        top,
    )
    return result.at[:k, :].set(top)


def _left_upper(t, b):
    m, n = b.shape
    if m % 2 or n % 2:
        raise ValueError("source left-upper kernel requires even dimensions")
    if m == 0 or n == 0:
        return b

    def pair_step(pair_index, result):
        pair = 2 * pair_index
        seeds = jnp.stack(
            (
                _add(_mul(t[pair, pair], b[pair, :]), _mul(t[pair, pair + 1], b[pair + 1, :])),
                _mul(t[pair + 1, pair + 1], b[pair + 1, :]),
            )
        )
        lanes = jnp.zeros((2, 2, n), jnp.float64).at[:, 0, :].set(seeds)

        def accumulate(q_index, current):
            q = 2 * q_index
            lo = jnp.stack((t[pair, q], t[pair + 1, q]))[:, None]
            hi = jnp.stack((t[pair, q + 1], t[pair + 1, q + 1]))[:, None]
            current = current.at[:, 0, :].set(_add(current[:, 0, :], _mul(lo, b[q, :])))
            return current.at[:, 1, :].set(_add(current[:, 1, :], _mul(hi, b[q + 1, :])))

        lanes = jax.lax.fori_loop(pair_index + 1, m // 2, accumulate, lanes)
        values = _add(lanes[:, 0, :], lanes[:, 1, :])
        result = result.at[pair, :].set(values[0])
        return result.at[pair + 1, :].set(values[1])

    return jax.lax.fori_loop(0, m // 2, pair_step, b)


@jax.jit
def _triangular_t(a, tau):
    """Dynamic inner loops retain host DLARFT operation order."""
    m, k = a.shape
    result = jnp.zeros((k, k), jnp.float64)
    indices = jnp.arange(k)
    for start in range(0, k, 16):
        end = min(start + 16, k)
        if start:
            cross = _gemm_source(
                a[end:, :start], a[end:, start:end], result[:start, start:end], ta="T", beta=0.0
            )
            result = result.at[:start, start:end].set(cross)

        def raw_column(col, current):
            alpha = _neg(tau[col])
            values = current[:, col]
            if start:
                length = end - col - 1
                tail = _shifted_column(a, col, col + 1, length)
                dots = _gemv_padded(a, tail, col + 1, length, 0, start)
                changed = _add(_mul(alpha, values), _mul(alpha, dots))
                changed = _add(changed, _mul(alpha, a[col, :]))
                values = jnp.where(indices < start, changed, values)
            length = m - col - 1
            tail = _shifted_column(a, col, col + 1, length)
            dots = _gemv_padded(a, tail, col + 1, length, start, col - start)
            changed = _add(_mul(alpha, dots), _mul(alpha, a[col, :]))
            values = jnp.where((indices >= start) & (indices < col), changed, values)
            values = values.at[col].set(tau[col])
            return current.at[:, col].set(values)

        result = jax.lax.fori_loop(start, end, raw_column, result)
        if start:
            result = result.at[:start, start:end].set(
                _left_upper(result[:start, :start], result[:start, start:end])
            )

        def finish_column(col, current):
            raw = current[:, col]

            def cross_update(q, values):
                changed = _add(values, _mul(current[:, q], raw[q]))
                return jnp.where(indices < start, changed, values)

            values = jax.lax.fori_loop(start, col, cross_update, raw)

            def upper_update(q, values):
                value = values[q]
                changed = _add(values, _mul(value, current[:, q]))
                values = jnp.where((indices >= start) & (indices < q), changed, values)
                return values.at[q].set(_mul(value, current[q, q]))

            values = jax.lax.fori_loop(start, col, upper_update, values)
            return current.at[:, col].set(values)

        result = jax.lax.fori_loop(start, end, finish_column, result)
    return result


@jax.jit
def _org2r(a, tau):
    """Unblocked source reflector-to-Q path with exact zero-tail trimming."""
    q = a
    m, rank = q.shape

    def reverse_step(step, matrix):
        j = rank - 1 - step
        matrix = matrix.at[j, j].set(1.0)
        v = _shifted_column(matrix, j, j, m - j)
        last = jnp.max(
            jnp.where(
                _abs_bits(v) != 0,
                jnp.arange(m, dtype=jnp.int32) + 1,
                jnp.asarray(0, jnp.int32),
            )
        )
        weights = _gemv_padded(matrix, v, j, last, j + 1, rank - j - 1)
        factor = _mul(_neg(tau[j]), weights)
        for col in range(rank):

            def update_column(mat):
                update = _mul(mat[:, j], factor[col])
                changed = _add(mat[:, col], update)
                row_active = jnp.arange(m, dtype=jnp.int32) < (j + last)
                row_active = row_active & (jnp.arange(m, dtype=jnp.int32) >= j)
                return mat.at[:, col].set(jnp.where(row_active, changed, mat[:, col]))

            matrix = jax.lax.cond(col > j, update_column, lambda mat: mat, matrix)
        column = matrix[:, j]
        scaled = _mul(column, _neg(tau[j]))
        matrix = matrix.at[:, j].set(jnp.where(jnp.arange(m, dtype=jnp.int32) > j, scaled, column))
        matrix = matrix.at[j, j].set(_add(1.0, _neg(tau[j])))
        column = matrix[:, j]
        column = jnp.where(jnp.arange(m, dtype=jnp.int32) < j, 0.0, column)
        return matrix.at[:, j].set(column)

    return jax.lax.fori_loop(0, rank, reverse_step, q)


def _orgqr(a, tau):
    """DORGQR unblocked/blocked paths driven by source dimension dispatch."""
    q = jnp.asarray(a)
    tau = jnp.asarray(tau)
    m, rank = q.shape
    block = source_q_block(m, rank)
    if rank <= 32 or block <= 1 or block >= rank:
        return _org2r(q, tau)
    kk = min(rank, ((rank - 32 - 1) // block) * block + block)
    if kk < rank:
        q = q.at[:kk, kk:].set(0.0)
        q = q.at[kk:, kk:].set(_org2r(q[kk:, kk:], tau[kk:]))
    for start in range(kk - block, -1, -block):
        end = start + block
        v = q[start:, start:end]
        t = _triangular_t(v, tau[start:end])
        q = q.at[start:, end:].set(_apply_block(v, t, q[start:, end:], transpose=False))
        q = q.at[start:, start:end].set(_org2r(v, tau[start:end]))
        q = q.at[:start, start:end].set(0.0)
    return q
