"""JAX generalized complex Schur reduction used by the source Sylvester solver.

Provenance
----------
MATLAB source: @chebop2/bartelsStewart.m (QZ consumer)
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
Algorithm source: LAPACK ZGGHRD and ZHGEQZ, version 3.2, Netlib complex16.
https://www.netlib.org/lapack/complex16/zgghrd.f
https://www.netlib.org/lapack/complex16/zhgeqz.f
Shift calculation: LAPACK 3.12.1 ZHGEQZ lines 745-776.
https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/SRC/zhgeqz.f
See the accompanying LAPACK_LICENSE.txt for the upstream license.
This translation uses JAX arrays and rotations; it does not call host SciPy.
"""

import jax
import jax.numpy as jnp


def rotation(f, g):
    af, ag = jnp.abs(f), jnp.abs(g)
    scale = jnp.maximum(af, ag)
    safe = jnp.where(scale == 0, 1.0, scale)
    norm = safe * jnp.sqrt((af / safe) ** 2 + (ag / safe) ** 2)
    denominator = jnp.where(norm == 0, 1.0, norm)
    phase = jnp.where(af == 0, 1.0 + 0j, f / jnp.where(af == 0, 1.0, af))
    c = jnp.where(scale == 0, 1.0, af / denominator)
    s = phase * jnp.conj(g) / denominator
    return (jnp.where(g == 0, 1.0, c), jnp.where(g == 0, 0j, s), jnp.where(g == 0, f, phase * norm))


def rows(a, i, j, c, s, mask):
    if a.shape[0] < 2:
        return a
    lo = jnp.minimum(i, j)
    block = jax.lax.dynamic_slice(a, (lo, 0), (2, a.shape[1]))
    forward = i < j
    x = jnp.where(forward, block[0], block[1])
    y = jnp.where(forward, block[1], block[0])
    xx = jnp.where(mask, c * x + s * y, x)
    yy = jnp.where(mask, -jnp.conj(s) * x + c * y, y)
    out = jnp.stack((jnp.where(forward, xx, yy), jnp.where(forward, yy, xx)))
    return jax.lax.dynamic_update_slice(a, out, (lo, 0))


def cols(a, i, j, c, s, mask):
    if a.shape[1] < 2:
        return a
    lo = jnp.minimum(i, j)
    block = jax.lax.dynamic_slice(a, (0, lo), (a.shape[0], 2))
    forward = i < j
    x = jnp.where(forward, block[:, 0], block[:, 1])
    y = jnp.where(forward, block[:, 1], block[:, 0])
    xx = jnp.where(mask, c * x + s * y, x)
    yy = jnp.where(mask, -jnp.conj(s) * x + c * y, y)
    out = jnp.stack((jnp.where(forward, xx, yy), jnp.where(forward, yy, xx)), axis=1)
    return jax.lax.dynamic_update_slice(a, out, (0, lo))


@jax.jit
def reduce(a, b):
    a, b = jnp.asarray(a, dtype=jnp.complex128), jnp.asarray(b, dtype=jnp.complex128)
    n = a.shape[0]
    if n == 0:
        identity = jnp.eye(0, dtype=a.dtype)
        return a, b, identity, identity
    idx = jnp.arange(n)
    q, t = jnp.linalg.qr(b)
    h = q.conj().T @ a
    z = jnp.eye(n, dtype=a.dtype)

    def column(j, state):
        def row(k, state):
            h, t, q, z = state
            i = n - 1 - k
            c, s, r = rotation(h[i - 1, j], h[i, j])
            h = rows(h, i - 1, i, c, s, idx > j).at[i - 1, j].set(r).at[i, j].set(0)
            t = rows(t, i - 1, i, c, s, idx >= i - 1)
            q = cols(q, i - 1, i, c, jnp.conj(s), idx >= 0)
            c, s, r = rotation(t[i, i], t[i, i - 1])
            h = cols(h, i, i - 1, c, s, idx < n)
            t = cols(t, i, i - 1, c, s, idx < i).at[i, i].set(r).at[i, i - 1].set(0)
            z = cols(z, i, i - 1, c, s, idx >= 0)
            return h, t, q, z

        return jax.lax.fori_loop(0, n - j - 2, row, state)

    return jax.lax.fori_loop(0, max(0, n - 2), column, (h, t, q, z))


def abs1(x):
    return jnp.abs(x.real) + jnp.abs(x.imag)


@jax.jit
def _qz_iteration(a, b):
    h, t, q, z = reduce(a, b)
    n = a.shape[0]
    idx = jnp.arange(n)
    if n == 0:
        return h, t, q, z, jnp.int64(0), jnp.int64(0)
    tiny = jnp.finfo(jnp.float64).tiny
    eps = jnp.finfo(jnp.float64).eps
    anorm = jnp.linalg.norm(h)
    bnorm = jnp.linalg.norm(t)
    atol = jnp.maximum(tiny, eps * anorm)
    btol = jnp.maximum(tiny, eps * bnorm)
    asc = 1 / jnp.maximum(tiny, anorm)
    bsc = 1 / jnp.maximum(tiny, bnorm)

    # state: matrices, last active index, iterations since deflation,
    # exceptional shift, total sweeps, status.
    def body(state):
        h, t, q, z, last, it, eshift, total, status = state
        split = (last == 0) | (abs1(h[last, last - 1]) <= atol)

        def deflate(state):
            h, t, q, z, last, it, eshift, total, status = state
            h = jax.lax.cond(last > 0, lambda h: h.at[last, last - 1].set(0), lambda h: h, h)
            mag = jnp.abs(t[last, last])
            phase = jnp.conj(t[last, last] / jnp.where(mag > tiny, mag, 1.0))
            phase = jnp.where(mag > tiny, phase, 1.0 + 0j)
            t = (
                t.at[:, last]
                .set(t[:, last] * phase)
                .at[last, last]
                .set(jnp.where(mag > tiny, mag, 0.0))
            )
            h = h.at[:, last].set(h[:, last] * phase)
            z = z.at[:, last].set(z[:, last] * phase)
            return h, t, q, z, last - 1, jnp.int64(0), jnp.complex128(0), total, status

        def sweep(state):
            h, t, q, z, last, it, eshift, total, status = state
            sub = jnp.concatenate((jnp.array([0.0]), abs1(jnp.diag(h, -1))))
            first = jnp.max(jnp.where((idx <= last) & (sub <= atol), idx, 0))
            h = jax.lax.cond(first > 0, lambda h: h.at[first, first - 1].set(0), lambda h: h, h)
            it = it + 1
            u12 = t[last - 1, last] / t[last, last]
            ad11 = asc * h[last - 1, last - 1] / (bsc * t[last - 1, last - 1])
            ad21 = asc * h[last, last - 1] / (bsc * t[last - 1, last - 1])
            ad12 = asc * h[last - 1, last] / (bsc * t[last, last])
            ad22 = asc * h[last, last] / (bsc * t[last, last])
            abi22 = ad22 - u12 * ad21
            # LAPACK 3.12.1 ZHGEQZ, lines 745-776: scaled Wilkinson
            # shift avoids cancellation for nearly repeated eigenvalues.
            abi12 = ad12 - u12 * ad11
            ct = jnp.sqrt(abi12) * jnp.sqrt(ad21)
            x = 0.5 * (ad11 - abi22)
            xnorm = abs1(x)
            scale = jnp.maximum(abs1(ct), xnorm)
            safe_scale = jnp.where(scale != 0, scale, 1.0)
            y = scale * jnp.sqrt((x / safe_scale) ** 2 + (ct / safe_scale) ** 2)
            safe_xnorm = jnp.where(xnorm != 0, xnorm, 1.0)
            direction = (x.real / safe_xnorm) * y.real + (x.imag / safe_xnorm) * y.imag
            y = jnp.where(direction < 0, -y, y)
            denominator = jnp.where(ct != 0, x + y, 1.0 + 0j)
            shift = abi22 - ct * (ct / denominator)
            exceptional = jnp.where(
                (it % 20 == 0) & (bsc * abs1(t[last, last]) > tiny),
                ad22,
                ad21,
            )
            eshift = jnp.where(it % 10 == 0, eshift + exceptional, eshift)
            shift = jnp.where(it % 10 == 0, eshift, shift)

            def choose(k, start):
                j = last - 1 - k
                ct = asc * h[j, j] - shift * bsc * t[j, j]
                v = abs1(ct)
                w = asc * abs1(h[j + 1, j])
                scale = jnp.maximum(v, w)
                safe = jnp.where((scale < 1) & (scale != 0), scale, 1.0)
                match = abs1(h[j, j - 1]) * (w / safe) <= (v / safe) * atol
                return jnp.where((start == first) & match, j, start)

            start = jax.lax.fori_loop(0, jnp.maximum(0, last - first - 1), choose, first)
            c, s, _ = rotation(
                asc * h[start, start] - shift * bsc * t[start, start], asc * h[start + 1, start]
            )

            def chase(j, state):
                h, t, q, z, c, s = state

                def next_rotation(args):
                    h, c, s = args
                    c, s, r = rotation(h[j, j - 1], h[j + 1, j - 1])
                    h = h.at[j, j - 1].set(r).at[j + 1, j - 1].set(0)
                    return h, c, s

                h, c, s = jax.lax.cond(j > start, next_rotation, lambda args: args, (h, c, s))
                h = rows(h, j, j + 1, c, s, idx >= j)
                t = rows(t, j, j + 1, c, s, idx >= j)
                q = cols(q, j, j + 1, c, jnp.conj(s), idx >= 0)
                c, s, r = rotation(t[j + 1, j + 1], t[j + 1, j])
                t = t.at[j + 1, j + 1].set(r).at[j + 1, j].set(0)
                h = cols(h, j + 1, j, c, s, idx <= jnp.minimum(j + 2, last))
                t = cols(t, j + 1, j, c, s, idx <= j)
                z = cols(z, j + 1, j, c, s, idx >= 0)
                return h, t, q, z, c, s

            h, t, q, z, _, _ = jax.lax.fori_loop(start, last, chase, (h, t, q, z, c, s))
            return h, t, q, z, last, it, eshift, total + 1, status

        def bottom_zero(state):
            h, t, q, z, last, it, eshift, total, status = state
            t = t.at[last, last].set(0)
            c, s, r = rotation(h[last, last], h[last, last - 1])
            h = (
                cols(h, last, last - 1, c, s, idx < last)
                .at[last, last]
                .set(r)
                .at[last, last - 1]
                .set(0)
            )
            t = cols(t, last, last - 1, c, s, idx < last)
            z = cols(z, last, last - 1, c, s, idx >= 0)
            return h, t, q, z, last, it, eshift, total + 1, status

        def inspect_internal(state):
            h, t, q, z, last, it, eshift, total, status = state
            sub = jnp.concatenate((jnp.array([0.0]), abs1(jnp.diag(h, -1))))
            small_h = sub <= atol
            small_t = jnp.abs(jnp.diag(t)) < btol
            j = jnp.max(jnp.where((idx < last) & (small_h | small_t), idx, 0))

            def chase_zero(state):
                h, t, q, z, last, it, eshift, total, status = state
                t = t.at[j, j].set(0)
                small_pair = abs1(h[j, j - 1]) * (asc * abs1(h[j + 1, j])) <= abs1(h[j, j]) * (
                    asc * atol
                )
                top = small_h[j] | small_pair

                def split_top(state):
                    h, t, q, z = state
                    h = jax.lax.cond(
                        (j > 0) & small_h[j], lambda h: h.at[j, j - 1].set(0), lambda h: h, h
                    )
                    c, s, r = rotation(h[j, j], h[j + 1, j])
                    h = rows(h, j, j + 1, c, s, idx > j).at[j, j].set(r).at[j + 1, j].set(0)
                    t = rows(t, j, j + 1, c, s, idx > j)
                    q = cols(q, j, j + 1, c, jnp.conj(s), idx >= 0)
                    h = jax.lax.cond(
                        (j > 0) & (~small_h[j]) & small_pair,
                        lambda h: h.at[j, j - 1].set(h[j, j - 1] * c),
                        lambda h: h,
                        h,
                    )
                    t = t.at[j + 1, j + 1].set(
                        jnp.where(abs1(t[j + 1, j + 1]) >= btol, t[j + 1, j + 1], 0)
                    )
                    return h, t, q, z

                def chase_bottom(state):
                    def step(k, state):
                        h, t, q, z = state
                        c, s, r = rotation(t[k, k + 1], t[k + 1, k + 1])
                        t = (
                            rows(t, k, k + 1, c, s, idx > k + 1)
                            .at[k, k + 1]
                            .set(r)
                            .at[k + 1, k + 1]
                            .set(0)
                        )
                        h = rows(h, k, k + 1, c, s, idx >= k - 1)
                        q = cols(q, k, k + 1, c, jnp.conj(s), idx >= 0)
                        c, s, r = rotation(h[k + 1, k], h[k + 1, k - 1])
                        h = (
                            cols(h, k, k - 1, c, s, idx <= k)
                            .at[k + 1, k]
                            .set(r)
                            .at[k + 1, k - 1]
                            .set(0)
                        )
                        t = cols(t, k, k - 1, c, s, idx < k)
                        z = cols(z, k, k - 1, c, s, idx >= 0)
                        return h, t, q, z

                    return jax.lax.fori_loop(j, last, step, state)

                h, t, q, z = jax.lax.cond(top, split_top, chase_bottom, (h, t, q, z))
                return h, t, q, z, last, it, eshift, total + 1, status

            return jax.lax.cond(small_t[j], chase_zero, sweep, state)

        return jax.lax.cond(
            split,
            deflate,
            lambda state: jax.lax.cond(
                jnp.abs(state[1][state[4], state[4]]) <= btol, bottom_zero, inspect_internal, state
            ),
            state,
        )

    state = (
        h,
        t,
        q,
        z,
        jnp.int64(n - 1),
        jnp.int64(0),
        jnp.complex128(0),
        jnp.int64(0),
        jnp.int64(0),
    )
    h, t, q, z, last, it, eshift, total, status = jax.lax.while_loop(
        lambda x: (x[4] >= 0) & (x[7] < 30 * n) & (x[8] == 0), body, state
    )
    status = jnp.where((status == 0) & (last >= 0), last + 1, status)
    return h, t, q, z, status, total


@jax.jit
def qz(a, b):
    """Complex generalized Schur factors, convergence status and sweep count.

    Input scaling follows LAPACK ZGGES: scale each pencil matrix separately
    into [sqrt(safmin)/eps, eps/sqrt(safmin)], then undo scaling on its factor.
    Provenance: @chebop2/bartelsStewart.m, Chebfun commit: 7574c77;
    Netlib ZGGES lines298–332 and443–450.
    """
    a, b = jnp.asarray(a, dtype=jnp.complex128), jnp.asarray(b, dtype=jnp.complex128)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or b.shape != a.shape:
        raise ValueError("generalized Schur requires matching square matrices")
    if a.shape[0] == 0:
        identity = jnp.eye(0, dtype=a.dtype)
        return a, b, identity, identity, jnp.int64(0), jnp.int64(0)
    small = jnp.sqrt(jnp.finfo(jnp.float64).tiny) / jnp.finfo(jnp.float64).eps
    large = 1.0 / small

    def scale(matrix):
        norm = jnp.max(jnp.abs(matrix))
        denominator = jnp.where(norm == 0, 1.0, norm)
        factor = jnp.where(
            (norm > 0) & (norm < small),
            small / denominator,
            jnp.where(norm > large, large / denominator, 1.0),
        )
        scaled = jax.lax.cond(
            factor == 1.0,
            lambda x: x,
            lambda x: jax.lax.complex(x.real * factor, x.imag * factor),
            matrix,
        )
        return scaled, factor

    a, sa = scale(a)
    b, sb = scale(b)
    h, t, q, z, status, steps = _qz_iteration(a, b)

    def undo(matrix, factor):
        # Real scalar division avoids an intermediate squared complex divisor.
        return jax.lax.cond(
            factor == 1.0,
            lambda x: x,
            lambda x: jax.lax.complex(x.real / factor, x.imag / factor),
            matrix,
        )

    return undo(h, sa), undo(t, sb), q, z, status, steps
