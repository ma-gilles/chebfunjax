"""JAX real binary64 LU with the observed source factorization schedules.

Provenance
----------
MATLAB source: @valsDiscretization/mldivide.m and native dense LU consumer
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
Independent integer arithmetic preserves binary64 operations in the observed
MATLAB small helper and larger recursive factorization. Reference controls
through151 cover real square matrices. Larger blocking, complex factors and
arbitrary NaN payload propagation remain separate qualification obligations.
Returned certificate describes NaN payload qualification, not an input gate.
"""

import jax
import jax.numpy as jnp

from chebfunjax.utils._source_qr._add import add_binary64 as add_lane
from chebfunjax.utils._source_qr._divide import divide_binary64
from chebfunjax.utils._source_qr._fma import fma_binary64 as fma_normal
from chebfunjax.utils._source_qr._multiply import multiply_binary64 as multiply_normal


def used(valid, mask):
    return jnp.all(jnp.where(mask, valid, True))


def magnitude(a):
    return jax.lax.bitcast_convert_type(a, jnp.uint64) & jnp.uint64((1 << 63) - 1)


def not_nan(a):
    return magnitude(a) <= jnp.uint64(0x7FF0000000000000)


def sub(a, b, c):
    p, pv = multiply_normal(a, b)
    z, zv = add_lane(c, -p)
    return z, pv & zv


def block_size(w):
    if w <= 10:
        return 1  # recursive square suffix: ILAENV both-dimensions<=10
    if w <= 32:
        return 8
    if w <= 128:
        return 32
    return min(32768, 1 << ((w - 1).bit_length() - 1))


@jax.jit
def _triangular_panel_update(L, X, valid):
    """Apply the source triangular schedule with reusable shape specializations.

    L/X shapes select specialization. Pivoting, slicing and trailing-region
    updates remain in general_factor. No arithmetic or certificate change.
    """
    width = L.shape[0]
    cols = X.shape[1]
    paired_cols = cols - cols % 2

    def pair(t, state):
        X, valid = state
        i = 2 * t
        if paired_cols:
            initial = jax.lax.dynamic_slice(X, (i, 0), (2, paired_cols))
            lo = initial
            hi = jnp.zeros_like(initial)

            def prior(q, state):
                lo, hi, valid = state
                a = jax.lax.dynamic_slice(L, (i, q), (2, 1))
                p, pv = multiply_normal(a, X[q, None, :paired_cols])
                lane = jnp.where(q % 2 == 0, lo, hi)
                out, ov = add_lane(lane, -p)
                return (
                    jnp.where(q % 2 == 0, out, lo),
                    jnp.where(q % 2 == 1, out, hi),
                    valid & jnp.all(pv & ov),
                )

            lo, hi, valid = jax.lax.fori_loop(0, i, prior, (lo, hi, valid))

            def combine(st):
                lo, hi, valid = st
                out, v = add_lane(lo, hi)
                return out, valid & jnp.all(v)

            out, valid = jax.lax.cond(
                i > 0, combine, lambda st: (st[0], st[2]), (lo, hi, valid)
            )
            second, v = sub(L[i + 1, i], out[0], out[1])
            valid = valid & jnp.all(v)
            X = X.at[i, :paired_cols].set(out[0]).at[i + 1, :paired_cols].set(second)
        if cols % 2:
            initial = jax.lax.dynamic_slice(X, (i, cols - 1), (2, 1))[:, 0]

            def sequential(st):
                z, valid = st

                def step(q, st):
                    z, valid = st
                    av = jax.lax.dynamic_slice(L, (i, q), (2, 1))[:, 0]
                    out, v = sub(av, X[q, -1], z)
                    return out, valid & jnp.all(v)

                return jax.lax.fori_loop(0, i, step, (z, valid))

            def grouped(st):
                z, valid = st
                four = jnp.array([z[0], 0.0, 0.0, 0.0])
                two = jnp.array([z[1], 0.0])
                boundary = (i // 8) * 8

                def prior(q, st):
                    four, two, valid = st
                    av = jax.lax.dynamic_slice(L, (i, q), (2, 1))[:, 0]
                    p, pv = multiply_normal(av, X[q, -1])
                    f, fv = add_lane(four[q % 4], -p[0])
                    tw, tv = add_lane(two[q % 2], -p[1])
                    return (
                        four.at[q % 4].set(f),
                        two.at[q % 2].set(tw),
                        valid & jnp.all(pv) & fv & tv,
                    )

                four, two, valid = jax.lax.fori_loop(0, boundary, prior, (four, two, valid))
                f0, v0 = add_lane(four[0], four[2])
                f1, v1 = add_lane(four[1], four[3])
                f, vf = add_lane(f0, f1)
                tw, vt = add_lane(two[0], two[1])
                out = jnp.array([f, tw])
                valid = valid & v0 & v1 & vf & vt

                def tail(q, st):
                    out, valid = st
                    av = jax.lax.dynamic_slice(L, (i, q), (2, 1))[:, 0]
                    z, v = sub(av, X[q, -1], out)
                    return z, valid & jnp.all(v)

                return jax.lax.fori_loop(boundary, i, tail, (out, valid))

            out, valid = jax.lax.cond(i < 8, sequential, grouped, (initial, valid))
            second, v = sub(L[i + 1, i], out[0], out[1])
            valid = valid & v
            X = X.at[i, -1].set(out[0]).at[i + 1, -1].set(second)
        return X, valid

    X, valid = jax.lax.fori_loop(0, width // 2, pair, (X, valid))
    return X, valid


def general_factor(A):
    n = A.shape[0]
    if A.ndim != 2 or A.shape[1] != n:
        raise ValueError("square matrix required")
    rows = jnp.arange(n)
    piv = jnp.zeros(n, jnp.int64)
    initial = jnp.all(not_nan(A))

    def pivot(A, j, piv, valid):
        # Integer magnitude max plus first-index tie rule is equivalent to
        # sequential strict comparisons; initialNaN keepsj, laterNaNs ignored.
        mags = magnitude(A[:, j])
        eligible = (rows >= j) & (mags <= jnp.uint64(0x7FF0000000000000))
        p = jnp.argmax(jnp.where(eligible, mags.astype(jnp.int64), -1))
        p = jnp.where(magnitude(A[j, j]) > jnp.uint64(0x7FF0000000000000), j, p)
        old = A[j]
        A = A.at[j].set(A[p]).at[p].set(old)
        piv = piv.at[j].set(p)
        return A, piv, valid

    if n == 0:
        return A, piv, initial
    if n <= 16:

        def step(j, state):
            A, piv, valid = state

            def col(q, state):
                A, valid = state
                z, v = sub(A[:, q], A[q, j], A[:, j])
                mask = (q < j) & (rows >= j)
                return A.at[:, j].set(jnp.where(mask, z, A[:, j])), valid & used(v, mask)

            A, valid = jax.lax.fori_loop(0, j, col, (A, valid))
            A, piv, valid = pivot(A, j, piv, valid)

            def upper(q, state):
                A, valid = state
                z, v = sub(A[j, q], A[q, :], A[j, :])
                mask = (q < j) & (rows > j)
                return A.at[j, :].set(jnp.where(mask, z, A[j, :])), valid & used(v, mask)

            A, valid = jax.lax.fori_loop(0, j, upper, (A, valid))

            # MATLAB helper scalar division loop. CPU XLA rewrites broadcast division
            # to reciprocal multiplication even when fast math is disabled.
            def divide_nonzero(state):
                A, valid = state

                def divide_entry(i, state):
                    A, valid = state
                    z, v = divide_binary64(A[i, j], A[j, j])
                    return A.at[i, j].set(z), valid & v

                return jax.lax.fori_loop(j + 1, n, divide_entry, (A, valid))

            A, valid = jax.lax.cond(
                magnitude(A[j, j]) != 0, divide_nonzero, lambda state: state, (A, valid)
            )
            return A, piv, valid

        return jax.lax.fori_loop(0, n, step, (A, piv, initial))

    def base(state, start, width):
        def step(t, state):
            A, piv, valid = state
            j = start + t

            def col(q, state):
                A, valid = state
                idx = start + q
                prod, pv = multiply_normal(A[:, idx], -A[idx, j])
                z, zv = add_lane(A[:, j], prod)
                mask = (q < t) & (rows >= j)
                return A.at[:, j].set(jnp.where(mask, z, A[:, j])), valid & used(pv & zv, mask)

            A, valid = jax.lax.fori_loop(0, t, col, (A, valid))
            A, piv, valid = pivot(A, j, piv, valid)
            mag = magnitude(A[j, j])
            large = (mag >= jnp.uint64(0x0010000000000000)) & (
                mag <= jnp.uint64(0x7FF0000000000000)
            )
            mask = rows > j

            def scale_nonzero(state):
                A, valid = state

                def reciprocal_scale(_):
                    reciprocal, rv = divide_binary64(1.0, A[j, j])
                    scaled, mv = multiply_normal(A[:, j], reciprocal)
                    return scaled, mv & rv

                def direct_divide(_):
                    return divide_binary64(A[:, j], A[j, j])

                scaled, v = jax.lax.cond(large, reciprocal_scale, direct_divide, operand=None)
                return A.at[:, j].set(jnp.where(mask, scaled, A[:, j])), valid & used(v, mask)

            A, valid = jax.lax.cond(mag != 0, scale_nonzero, lambda state: state, (A, valid))
            # Upper-row dot starts at first product, then sequential additions.
            initial, pv = multiply_normal(A[j, start], A[start, :])
            active = (t > 0) & (rows > j) & (rows < start + width)
            valid = valid & used(pv, active)

            def dot(q, state):
                z, valid = state
                idx = start + q
                prod, pv = multiply_normal(A[j, idx], A[idx, :])
                out, ov = add_lane(z, prod)
                mask = (q < t) & active
                return jnp.where(q < t, out, z), valid & used(pv & ov, mask)

            z, valid = jax.lax.fori_loop(1, t, dot, (initial, valid))
            out, v = add_lane(A[j, :], -z)
            valid = valid & used(v, active)
            A = A.at[j, :].set(jnp.where(active, out, A[j, :]))
            return A, piv, valid

        return jax.lax.fori_loop(0, width, step, state)

    def update(state, start, width, end):
        A, piv, valid = state
        stop = start + width
        L = A[start:stop, start:stop]
        X = A[start:stop, stop:end]
        cols = end - stop
        X, valid = _triangular_panel_update(L, X, valid)
        A = A.at[start:stop, stop:end].set(X)
        m = n - stop
        mr = (m // 12) * 12
        nc = (cols // 4) * 4
        if width < 2:
            mr = nc = 0

        def region(A, valid, r0, r1, c0, c1, fused):
            if r0 == r1 or c0 == c1:
                return A, valid
            C = A[r0:r1, c0:c1]
            left = A[r0:r1, start:stop]
            right = X[:, c0 - stop : c1 - stop]

            def step(q, st):
                C, valid = st
                if fused:
                    z, v = fma_normal(-left[:, q, None], right[q, None, :], C)
                else:
                    p, pv = multiply_normal(left[:, q, None], -right[q, None, :])
                    z, zv = add_lane(C, p)
                    v = pv & zv
                return z, valid & jnp.all(v)

            C, valid = jax.lax.fori_loop(0, width, step, (C, valid))
            return A.at[r0:r1, c0:c1].set(C), valid

        A, valid = region(A, valid, stop, stop + mr, stop, stop + nc, True)
        A, valid = region(A, valid, stop + mr, n, stop, stop + nc, False)
        A, valid = region(A, valid, stop, n, stop + nc, end, False)
        return A, piv, valid

    base_compiled = jax.jit(base, static_argnums=(2,))

    def factor(state, start, width):
        nb = block_size(width)
        if nb <= 1:
            return base_compiled(state, jnp.asarray(start, dtype=jnp.int64), width)
        # Static source panel boundaries allow exact active-array slice shapes.
        for offset in range(start, start + width, nb):
            actual = min(nb, start + width - offset)
            state = factor(state, offset, actual)
            if offset + actual < start + width:
                state = update(state, offset, actual, start + width)

        return state

    return factor((A, piv, initial), 0, n)
