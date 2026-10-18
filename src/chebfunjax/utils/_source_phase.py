"""Source scalar arithmetic used by the small Hermite ASY path.

Provenance: Chebfun hermpts.m, commit
7574c77680d7e82b79626300bf255498271a72df; AMOS Airy/Bessel algorithms
by Donald E. Amos; Netlib fdlibm e_pow.c, e_exp.c, e_log.c,
e_rem_pio2.c and k_rem_pio2.c.

Fdlibm copyright (C) 1993, 2004 Sun Microsystems, Inc. All rights reserved.
Developed at SunSoft, a Sun Microsystems, Inc. business.
Permission to use, copy, modify, and distribute this software is freely
granted, provided that this notice is preserved.

These are private ordinary-domain source kernels, not complete replacements
for public transcendental APIs. Static mathematical coefficients are kept
in the repository. No captured operands, output tables or runtime fixtures.
"""

import jax
import jax.numpy as j

from ._source_real import add, mul, sub
from ._source_scalar_constants import LARGE as _LARGE
from ._source_scalar_constants import ORDER_PHASE as _ORDER_PHASE
from ._source_scalar_constants import PHASE as _PHASE

IP = j.asarray(_LARGE["two_over_pi"], j.int64)
PI = j.asarray(_LARGE["PIo2"][:5], j.float64)
N = 20


def _word(x):
    return jax.lax.bitcast_convert_type(x, j.uint64)


def _real(x):
    return jax.lax.bitcast_convert_type(x, j.float64)


def _i(x):
    return j.trunc(x).astype(j.int64)


@jax.jit
def source_large_rem_pio2(x):
    x = j.asarray(x, j.float64)
    a = j.abs(x)
    bits = _word(a)
    e0 = (bits >> j.uint64(52) & j.uint64(2047)).astype(j.int64) - 1046
    z = _real(bits - (e0.astype(j.uint64) << j.uint64(52)))
    t0 = _i(z).astype(j.float64)
    z = mul(sub(z, t0), 16777216.0)
    t1 = _i(z).astype(j.float64)
    t2 = mul(sub(z, t1), 16777216.0)
    tx = j.stack((t0, t1, t2))
    nx = j.where(t2 != 0.0, 3, j.where(t1 != 0.0, 2, 1))
    jx = nx - 1
    jv = j.maximum((e0 - 3) // 24, 0)
    q0 = e0 - 24 * (jv + 1)
    fi = jv - jx + j.arange(N + 3, dtype=j.int64)
    f = j.where(fi < 0, 0.0, IP[j.clip(fi, 0, len(IP) - 1)].astype(j.float64))

    def convolution(i):

        def term(k, total):
            return add(total, mul(tx[k], f[jx + i - k]))

        return jax.lax.fori_loop(0, nx, term, j.asarray(0.0, j.float64))

    q = j.zeros((N,), j.float64)
    q = jax.lax.fori_loop(0, 5, lambda i, q: q.at[i].set(convolution(i)), q)
    iq = j.zeros((N,), j.int64)

    def distill(q, jz):

        def term(k, state):
            (z, iq) = state
            fw = _i(mul(5.960464477539063e-08, z)).astype(j.float64)
            return (add(q[jz - k - 1], fw), iq.at[k].set(_i(sub(z, mul(16777216.0, fw)))))

        (z, iq) = jax.lax.fori_loop(0, jz, term, (q[jz], j.zeros((N,), j.int64)))
        z = j.ldexp(z, q0.astype(j.int32))
        z = sub(z, mul(8.0, j.floor(mul(z, 0.125))))
        n = _i(z)
        z = sub(z, n.astype(j.float64))
        shift = j.clip(24 - q0, 0, 63)
        top = iq[jz - 1] >> shift
        n = n + j.where(q0 > 0, top, 0)
        iq = iq.at[jz - 1].set(j.where(q0 > 0, iq[jz - 1] - (top << shift), iq[jz - 1]))
        ih = j.where(
            q0 > 0,
            iq[jz - 1] >> j.clip(23 - q0, 0, 63),
            j.where(q0 == 0, iq[jz - 1] >> 23, j.where(z >= 0.5, 2, 0)),
        )

        def complement(state):
            (z, iq, n) = state

            def step(i, state):
                (iq, carry) = state
                v = iq[i]
                first = (carry == 0) & (v != 0)
                value = j.where(carry != 0, 16777215 - v, j.where(first, 16777216 - v, v))
                return (iq.at[i].set(value), j.where(first, 1, carry))

            (iq, carry) = jax.lax.fori_loop(0, jz, step, (iq, j.asarray(0, j.int64)))
            mask = j.where(q0 == 1, 8388607, 4194303)
            iq = iq.at[jz - 1].set(j.where(q0 > 0, iq[jz - 1] & mask, iq[jz - 1]))
            zz = sub(1.0, z)
            zz = j.where(carry != 0, sub(zz, j.ldexp(j.asarray(1.0), q0.astype(j.int32))), zz)
            return (j.where(ih == 2, zz, z), iq, n + 1)

        (z, iq, n) = jax.lax.cond(ih > 0, complement, lambda s: s, (z, iq, n))
        return (z, iq, n, ih)

    def needs_more(z, iq, jz):
        mask = (j.arange(N) >= 4) & (j.arange(N) < jz)
        accumulated = jax.lax.reduce(
            j.where(mask, iq, 0), j.asarray(0, j.int64), jax.lax.bitwise_or, (0,)
        )
        return (z == 0.0) & (accumulated == 0)

    (z, iq, n, ih) = distill(q, j.asarray(4, j.int64))
    initial = (q, j.asarray(4, j.int64), z, iq, n, ih, j.asarray(0, j.int64), j.asarray(False))

    def again(s):
        return needs_more(s[2], s[3], s[1]) & ~s[7] & (s[6] < N)

    def extend(s):
        (q, jz, z, iq, n, ih, count, exhausted) = s

        def zero_tail(k):
            return (k < 4) & (iq[4 - k] == 0)

        k = jax.lax.while_loop(zero_tail, lambda k: k + 1, j.asarray(1, j.int64))
        stop = jz + k
        exhausted = stop >= N
        q = jax.lax.fori_loop(
            jz + 1, j.minimum(stop + 1, N), lambda i, q: q.at[i].set(convolution(i)), q
        )
        jz = j.minimum(stop, N - 1)
        (z, iq, n, ih) = distill(q, jz)
        return (q, jz, z, iq, n, ih, count + 1, exhausted)

    (q, jz, z, iq, n, ih, count, exhausted) = jax.lax.while_loop(again, extend, initial)

    def zero_top(state):
        (iq, jz, q0, z) = state
        jz = jz - 1
        q0 = q0 - 24

        def drop(s):
            return (s[0] > 0) & (iq[s[0]] == 0)

        (jz, q0) = jax.lax.while_loop(drop, lambda s: (s[0] - 1, s[1] - 24), (jz, q0))
        return (iq, jz, q0)

    def nonzero_top(state):
        (iq, jz, q0, z) = state
        z = j.ldexp(z, (-q0).astype(j.int32))
        fw = _i(mul(5.960464477539063e-08, z))
        high = z >= 16777216.0
        value = j.where(high, _i(sub(z, mul(16777216.0, fw.astype(j.float64)))), _i(z))
        iq = iq.at[jz].set(value)
        iq = jax.lax.cond(
            high, lambda iq: iq.at[j.minimum(jz + 1, N - 1)].set(fw), lambda iq: iq, iq
        )
        return (iq, jz + high.astype(j.int64), q0 + 24 * high.astype(j.int64))

    (iq, jz, q0) = jax.lax.cond(z == 0.0, zero_top, nonzero_top, (iq, jz, q0, z))
    exhausted = exhausted | (jz >= N) | (count >= N)
    jz = j.minimum(jz, N - 1)

    def convert(k, state):
        (q, fw) = state
        i = jz - k
        return (q.at[i].set(mul(fw, iq[i].astype(j.float64))), mul(fw, 5.960464477539063e-08))

    (q, _) = jax.lax.fori_loop(
        0, jz + 1, convert, (j.zeros((N,), j.float64), j.ldexp(j.asarray(1.0), q0.astype(j.int32)))
    )

    def finish(k, fq):
        i = jz - k

        def term(t, total):
            return add(total, mul(PI[t], q[i + t]))

        fw = jax.lax.fori_loop(0, j.minimum(4, jz - i) + 1, term, j.asarray(0.0))
        return fq.at[k].set(fw)

    fq = jax.lax.fori_loop(0, jz + 1, finish, j.zeros((N,), j.float64))
    hi = jax.lax.fori_loop(0, jz + 1, lambda k, v: add(v, fq[jz - k]), j.asarray(0.0))
    lo = jax.lax.fori_loop(1, jz + 1, lambda i, v: add(v, fq[i]), sub(fq[0], hi))
    sign = j.where(ih == 0, 1.0, -1.0) * j.where(x < 0.0, -1.0, 1.0)
    hi = mul(hi, sign)
    lo = mul(lo, sign)
    n = (n & 7) * j.where(x < 0.0, -1, 1)
    valid = j.isfinite(x) & (a > 1.0) & ~exhausted
    return (n, j.where(valid, hi, j.nan), j.where(valid, lo, j.nan), count)


_NPIO2_HW = (
    1073291771,
    1074340347,
    1074977148,
    1075388923,
    1075800698,
    1076025724,
    1076231611,
    1076437499,
    1076643386,
    1076849274,
    1076971356,
    1077074300,
    1077177244,
    1077280187,
    1077383131,
    1077486075,
    1077589019,
    1077691962,
    1077794906,
    1077897850,
    1077968460,
    1078019932,
    1078071404,
    1078122876,
    1078174348,
    1078225820,
    1078277292,
    1078328763,
    1078380235,
    1078431707,
    1078483179,
    1078534651,
)


def high(x):
    return (jax.lax.bitcast_convert_type(j.abs(x), j.uint64) >> j.uint64(32)).astype(j.int64)


def cosine(x, y):
    z = mul(x, x)
    r = add(mul(_PHASE[53996536], z), _PHASE[53996544])
    r = sub(mul(r, z), _PHASE[53996552])
    r = add(mul(r, z), _PHASE[53996560])
    r = sub(mul(r, z), _PHASE[53996568])
    r = add(mul(r, z), _PHASE[53996576])
    r = sub(mul(mul(r, z), z), mul(x, y))
    hz = mul(0.5, z)
    bits = jax.lax.bitcast_convert_type(j.abs(x), j.uint64) & j.uint64(18446744069414584320)
    qx = mul(jax.lax.bitcast_convert_type(bits, j.float64), 0.25)
    answer = j.where(
        j.abs(x) < _PHASE[53996584],
        sub(1.0, sub(hz, r)),
        j.where(
            j.abs(x) > _PHASE[53996592],
            sub(_PHASE[53996608], sub(sub(hz, _PHASE[53996600]), r)),
            sub(sub(1.0, qx), sub(sub(hz, qx), r)),
        ),
    )
    return j.where(j.abs(x) < 2 ** (-27), j.asarray(1.0), answer)


def sine(x, y, reduced):
    z = mul(x, x)
    v = mul(x, z)
    r = sub(mul(_PHASE[53997272], z), _PHASE[53997280])
    r = add(mul(r, z), _PHASE[53997288])
    r = sub(mul(r, z), _PHASE[53997296])
    r = add(mul(r, z), _PHASE[53997304])
    unreduced = add(x, mul(v, sub(mul(r, z), _PHASE[53997312])))
    reduced_value = sub(
        x, add(sub(mul(z, sub(mul(0.5, y), mul(v, r))), y), mul(v, _PHASE[53997312]))
    )
    return j.where(j.abs(x) < 2 ** (-27), x, j.where(reduced, reduced_value, unreduced))


@jax.jit
def source_medium_phase_jit(x):
    x = j.asarray(x, j.float64)
    ax = j.abs(x)
    zero = j.asarray(0.0, j.float64)

    def small(_):
        return (sine(x, zero, False), cosine(x, zero))

    def special(_):
        sign = j.copysign(j.asarray(1.0), x)
        z = sub(x, mul(sign, _PHASE[53996768]))
        near = high(x) == 1073291771
        z = j.where(near, sub(z, mul(sign, _PHASE[53996784])), z)
        tail = mul(sign, j.where(near, _PHASE[53996792], _PHASE[53996776]))
        hi = sub(z, tail)
        lo = sub(sub(z, hi), tail)
        return (mul(sign, cosine(hi, lo)), -mul(sign, sine(hi, lo, True)))

    def medium(_):
        ix = high(x)
        n = add(mul(ax, 0.6366197723675814), 0.5).astype(j.int64)
        fn = n.astype(j.float64)
        r = sub(ax, mul(fn, 1.5707963267341256))
        w = mul(fn, 6.077100506506192e-11)
        hi = sub(r, w)

        def refine(state):
            (r, w, hi) = state
            loss = (ix >> 20) - (high(hi) >> 20 & 2047)

            def second(state):
                (temp, _, _) = state
                w = mul(fn, 6.077100506303966e-11)
                r = sub(temp, w)
                w = sub(mul(fn, 2.0222662487959506e-21), sub(sub(temp, r), w))
                hi = sub(r, w)
                loss = (ix >> 20) - (high(hi) >> 20 & 2047)

                def third(state):
                    (temp, _, _) = state
                    w = mul(fn, 2.0222662487111665e-21)
                    r = sub(temp, w)
                    w = sub(mul(fn, 8.4784276603689e-32), sub(sub(temp, r), w))
                    return (r, w, sub(r, w))

                return jax.lax.cond(loss > 49, third, lambda q: q, (r, w, hi))

            return jax.lax.cond(loss > 16, second, lambda q: q, state)

        pattern = j.asarray(_NPIO2_HW, j.int64)[j.clip(n - 1, 0, 31)]
        (r, w, hi) = jax.lax.cond((n >= 32) | (ix == pattern), refine, lambda q: q, (r, w, hi))
        lo = sub(sub(r, hi), w)
        s = sine(hi, lo, True)
        c = cosine(hi, lo)
        quad = n & 3
        ss = j.where(quad == 1, c, j.where(quad == 2, -s, j.where(quad == 3, -c, s)))
        cc = j.where(quad == 1, -s, j.where(quad == 2, -c, j.where(quad == 3, s, c)))
        return (j.where(x < 0.0, -ss, ss), cc)

    (s, c) = jax.lax.cond(
        ax <= j.pi / 4,
        small,
        lambda _: jax.lax.cond(ax <= 2 * j.pi / 3 + 1e-15, special, medium, None),
        None,
    )
    valid = j.isfinite(x) & (high(x) <= 1094263291)
    return (j.where(valid, s, j.nan), j.where(valid, c, j.nan))


@jax.jit
def source_phase_jit(x):
    x = j.asarray(x, j.float64)

    def large(_):
        (n, hi, lo, count) = source_large_rem_pio2(x)
        s = sine(hi, lo, True)
        c = cosine(hi, lo)
        quad = n & 3
        return (
            j.where(quad == 1, c, j.where(quad == 2, -s, j.where(quad == 3, -c, s))),
            j.where(quad == 1, -s, j.where(quad == 2, -c, j.where(quad == 3, s, c))),
        )

    return jax.lax.cond(
        j.isfinite(x) & (high(x) > 1094263291), large, lambda _: source_medium_phase_jit(x), None
    )


@jax.jit
def source_order_phase(order, direction):
    integer = j.trunc(order)
    x = mul(sub(order, integer), direction)
    twelfths = mul(x, 12.0)
    index = j.clip(twelfths.astype(j.int32) + 11, 0, 22)
    table_cos = j.asarray([r["cos"] for r in _ORDER_PHASE])[index]
    table_sin = j.asarray([r["sin"] for r in _ORDER_PHASE])[index]
    table_sin = j.where(x == 0.0, j.copysign(j.asarray(0.0), x), table_sin)
    axis = j.where(
        j.abs(x) < 0.25,
        0,
        j.where(j.abs(x) < 0.75, j.where(x > 0.0, 1, -1), j.where(x > 0.0, 2, -2)),
    )
    reduced = sub(x, mul(axis.astype(j.float64), 0.5))
    (sn, cs) = source_medium_phase_jit(mul(reduced, j.pi))
    n = j.mod(axis, 4)
    cos = j.where(n == 0, cs, j.where(n == 1, -sn, j.where(n == 2, -cs, sn)))
    sin = j.where(n == 0, sn, j.where(n == 1, cs, j.where(n == 2, -sn, -cs)))
    exact = (twelfths == j.trunc(twelfths)) & (twelfths >= -11) & (twelfths <= 11)
    cos = j.where(exact, table_cos, cos)
    sin = j.where(exact, table_sin, sin)
    odd = j.mod(integer.astype(j.int32), 2) == 1
    return (j.where(odd, -cos, cos), j.where(odd, -sin, sin))
