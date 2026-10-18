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

from ._source_binary import binary
from ._source_scalar_constants import EXP as _EXP
from ._source_scalar_constants import GAMMA as _GAMMA
from ._source_scalar_constants import LOG as _LOG
from ._source_scalar_constants import POW as _POW


def op(a, b, k):
    return binary(j.asarray(a, j.float64), j.asarray(b, j.float64), k)


def mul(a, b):
    return op(a, b, 0)


def add(a, b):
    return op(a, b, 1)


def sub(a, b):
    return op(a, b, 2)


def div(a, b):
    return op(a, b, 3)


def rational(t, p, q, den=1.0):
    numerator = j.asarray(0.0)
    denominator = j.asarray(den)
    for a, b in zip(_GAMMA[p], _GAMMA[q]):
        numerator = add(mul(numerator, t), a)
        denominator = add(mul(denominator, t), b)
    return div(numerator, denominator)


def source_gamma_log(x):
    x = j.asarray(x, j.float64)

    def small(_):
        return -j.log(x)

    def branch_p1(t):
        return mul(t, sub(mul(t, rational(t, "p1", "q1")), 0.5772156649015329))

    def branch_p2(t):
        return mul(t, add(mul(t, rational(t, "p2", "q2")), 0.42278433509846713))

    def central(_):
        return jax.lax.cond(
            x <= 0.5,
            lambda _: sub(branch_p1(x), j.log(x)),
            lambda _: jax.lax.cond(
                x < 0.6796875,
                lambda _: sub(branch_p2(sub(x, 1.0)), j.log(x)),
                lambda _: jax.lax.cond(
                    x <= 1.5,
                    lambda _: branch_p1(sub(x, 1.0)),
                    lambda _: branch_p2(sub(x, 2.0)),
                    None,
                ),
                None,
            ),
            None,
        )

    def upper(_):

        def middle(_):
            return add(mul(sub(x, 4.0), rational(sub(x, 4.0), "p4", "q4", -1.0)), 1.791759469228055)

        def large(_):

            def correction(_):
                ratio = div(1.0, mul(x, x))
                a = j.asarray(_GAMMA["c"][-1])
                for c in _GAMMA["c"][:-1]:
                    a = add(mul(a, ratio), c)
                return a

            a = jax.lax.cond(x <= 2.25e76, correction, lambda _: j.asarray(0.0), None)
            l = j.log(x)
            return add(mul(x, sub(l, 1.0)), sub(add(div(a, x), 0.9189385332046728), mul(0.5, l)))

        return jax.lax.cond(x <= 12.0, middle, large, None)

    def ordinary(_):
        return jax.lax.cond(
            x <= 2.220446049250313e-16,
            small,
            lambda _: jax.lax.cond(x <= 4.0, central, upper, None),
            None,
        )

    integer = j.trunc(x)
    idx = j.clip(integer.astype(j.int32) - 1, 0, 99)
    result = jax.lax.cond(
        (x >= 1.0) & (x <= 100.0) & (x == integer),
        lambda _: j.asarray(_GAMMA["integers"])[idx],
        ordinary,
        None,
    )
    return j.where(x > 0, result, j.nan)


@jax.jit
def source_exp_jit(x):
    x = j.asarray(x, j.float64)
    ax = j.abs(x)
    k = j.where(
        ax < _EXP[53997208],
        j.where(x > 0.0, 1, -1),
        j.trunc(add(mul(x, _EXP[53996856]), j.where(x > 0.0, 0.5, -0.5))).astype(j.int64),
    )
    k = j.where(ax > _EXP[53997200], k, j.asarray(0, j.int64))
    fk = k.astype(j.float64)
    hi = sub(x, mul(fk, _EXP[53996864]))
    lo = mul(fk, _EXP[53996840])
    rarg = sub(hi, lo)
    t = mul(rarg, rarg)
    r = sub(mul(_EXP[53997216], t), _EXP[53997224])
    r = add(mul(r, t), _EXP[53997232])
    r = sub(mul(r, t), _EXP[53997240])
    r = add(mul(r, t), _EXP[53997248])
    c = sub(rarg, mul(r, t))
    y0 = sub(1.0, sub(div(mul(rarg, c), sub(c, 2.0)), rarg))
    y = sub(1.0, sub(sub(lo, div(mul(rarg, c), sub(2.0, c))), hi))
    bits = jax.lax.bitcast_convert_type(y, j.uint64) + (k.astype(j.uint64) << j.uint64(52))
    scaled = jax.lax.bitcast_convert_type(bits, j.float64)
    value = j.where(k == 0, y0, j.where(k == 1, add(y, y), j.where(k == -1, mul(0.5, y), scaled)))
    value = j.where(
        ax < _EXP[53997256], add(1.0, x), j.where(x == 1.0, j.asarray(_EXP[53997184]), value)
    )
    return j.where(j.isfinite(x) & (ax < 700.0), value, j.nan)


@jax.jit
def source_log_jit(original):
    x = j.asarray(original, j.float64)
    bits = jax.lax.bitcast_convert_type(x, j.uint64)
    hi = bits >> j.uint64(32)
    tiny = hi < j.uint64(1048576)
    x = j.where(tiny, mul(x, float(2**54)), x)
    bits = jax.lax.bitcast_convert_type(x, j.uint64)
    hi = (bits >> j.uint64(32)).astype(j.int64)
    hx = hi & 1048575
    i = hx + 614244 & 1048576
    k = (hi >> 20) - 1023 + j.where(tiny, -54, 0) + (i >> 20)
    normal = (hx | i ^ 1072693248).astype(j.uint64) << j.uint64(32) | bits & j.uint64(4294967295)
    f = sub(jax.lax.bitcast_convert_type(normal, j.float64), 1.0)
    dk = k.astype(j.float64)
    lh = _LOG[53996864]
    ll = _LOG[53996840]
    r = mul(mul(f, f), sub(0.5, mul(_LOG[53997344], f)))
    small = j.where(
        f == 0.0,
        j.where(k == 0, j.asarray(0.0), add(mul(dk, lh), mul(dk, ll))),
        j.where(k == 0, sub(f, r), sub(mul(dk, lh), sub(sub(r, mul(dk, ll)), f))),
    )
    s = div(f, add(2.0, f))
    z = mul(s, s)
    w = mul(z, z)
    p = add(mul(_LOG[53997120], w), _LOG[53997136])
    p = add(mul(p, w), _LOG[53997152])
    p = add(mul(p, w), _LOG[53997168])
    p = mul(p, z)
    q = add(mul(_LOG[53997128], w), _LOG[53997144])
    q = add(mul(q, w), _LOG[53997160])
    q = mul(q, w)
    r = add(p, q)
    h = mul(mul(0.5, f), f)
    t = mul(add(r, h), s)
    wide = j.where(k == 0, sub(f, sub(h, t)), sub(mul(dk, lh), sub(sub(h, add(mul(dk, ll), t)), f)))
    t = mul(sub(f, r), s)
    close = j.where(k == 0, sub(f, t), sub(mul(dk, lh), sub(sub(t, mul(dk, ll)), f)))
    ordinary = j.where(hx - 398458 | 440401 - hx > 0, wide, close)
    value = j.where(hx + 2 & 1048575 <= 2, small, ordinary)
    return j.where((original > 0.0) & j.isfinite(original), value, j.nan)


def bits(x):
    return jax.lax.bitcast_convert_type(x, j.uint64)


def real(b):
    return jax.lax.bitcast_convert_type(b, j.float64)


def upper(x):
    return real(bits(x) & j.uint64(18446744069414584320))


def high_real(h):
    return real((h.astype(j.uint64) & j.uint64(4294967295)) << j.uint64(32))


@jax.jit
def source_pow_jit(x, y):
    x = j.asarray(x, j.float64)
    y = j.asarray(y, j.float64)
    ax = j.abs(x)
    ix = (bits(ax) >> j.uint64(32)).astype(j.int64)
    tiny = ix < 1048576
    ax = j.where(tiny, mul(ax, float(2**53)), ax)
    ix = (bits(ax) >> j.uint64(32)).astype(j.int64)
    n = (ix >> 20) - 1023 + j.where(tiny, -53, 0)
    frac = ix & 1048575
    k = j.where((frac > 235662) & (frac < 767610), 1, 0).astype(j.int64)
    renorm = frac >= 767610
    n = n + renorm.astype(j.int64)
    ix = (frac | 1072693248) - j.where(renorm, 1048576, 0)
    ax = real(ix.astype(j.uint64) << j.uint64(32) | bits(ax) & j.uint64(4294967295))
    bp = j.where(k == 0, 1.0, 1.5)
    dph = j.where(k == 0, 0.0, _POW["dph"])
    dpl = j.where(k == 0, 0.0, _POW["dpl"])
    u = sub(ax, bp)
    v = div(1.0, add(ax, bp))
    ss = mul(u, v)
    sh = upper(ss)
    th = high_real((ix >> 1 | 536870912) + 524288 + (k << 18))
    tl = sub(ax, sub(th, bp))
    sl = mul(v, sub(sub(u, mul(sh, th)), mul(sh, tl)))
    s2 = mul(ss, ss)
    r = j.asarray(_POW["L6"], j.float64)
    for key in ["L5", "L4", "L3", "L2", "L1"]:
        r = add(_POW[key], mul(s2, r))
    r = mul(mul(s2, s2), r)
    r = add(r, mul(sl, add(sh, ss)))
    s2 = mul(sh, sh)
    th = upper(add(add(3.0, s2), r))
    tl = sub(r, sub(sub(th, 3.0), s2))
    u = mul(sh, th)
    v = add(mul(sl, th), mul(tl, ss))
    ph = upper(add(u, v))
    pl = sub(v, sub(ph, u))
    zh = mul(_POW["cph"], ph)
    zl = add(add(mul(_POW["cpl"], ph), mul(pl, _POW["cp"])), dpl)
    nf = n.astype(j.float64)
    t1 = upper(add(add(add(zh, zl), dph), nf))
    t2 = sub(zl, sub(sub(sub(t1, nf), dph), zh))
    y1 = upper(y)
    pl = add(mul(sub(y, y1), t1), mul(y, t2))
    ph = mul(y1, t1)
    power = add(pl, ph)
    jh = (bits(power) >> j.uint64(32)).astype(j.int32).astype(j.int64)
    ih = jh & 2147483647
    ke = (ih >> 20) - 1023
    reduce = ih > 1071644672
    nn = jh + (j.asarray(1048576, j.int64) >> j.maximum(ke + 1, 0))
    kn = ((nn & 2147483647) >> 20) - 1023
    ks = j.clip(kn, 0, 20)
    t = high_real(nn & ~(j.asarray(1048575, j.int64) >> ks))
    scale = (nn & 1048575 | 1048576) >> 20 - ks
    scale = j.where(jh < 0, -scale, scale)
    scale = j.where(reduce, scale, 0)
    ph = j.where(reduce, sub(ph, t), ph)
    t = upper(add(pl, ph))
    u = mul(t, _POW["lg2h"])
    v = add(mul(sub(pl, sub(t, ph)), _POW["lg2"]), mul(t, _POW["lg2l"]))
    z = add(u, v)
    w = sub(v, sub(z, u))
    t = mul(z, z)
    r = j.asarray(_POW["P5"], j.float64)
    for key in ["P4", "P3", "P2", "P1"]:
        r = add(_POW[key], mul(t, r))
    t1 = sub(z, mul(t, r))
    r = sub(div(mul(z, t1), sub(t1, 2.0)), add(w, mul(z, w)))
    z = sub(1.0, sub(r, z))
    magnitude = real(bits(z) + (scale.astype(j.uint64) << j.uint64(52)))
    integral = y == j.trunc(y)
    odd = integral & (j.abs(y).astype(j.int64) & 1 == 1)
    sign = j.where(j.signbit(x) & odd, -1.0, 1.0)
    value = mul(sign, magnitude)
    valid = (
        j.isfinite(x)
        & j.isfinite(y)
        & (j.abs(y) <= 32.0)
        & (j.abs(power) < 1000.0)
        & ((x >= 0.0) | integral)
    )
    value = j.where(valid, value, j.nan)
    value = j.where(y == 0.5, j.where(x >= 0.0, j.sqrt(x), j.nan), value)
    value = j.where(y == 2.0, mul(x, x), value)
    value = j.where(y == -1.0, div(1.0, x), value)
    value = j.where(y == 1.0, x, value)
    zero = j.where(odd, j.copysign(j.asarray(0.0, j.float64), x), j.asarray(0.0, j.float64))
    value = j.where(
        x == 0.0, j.where(y > 0.0, zero, j.copysign(j.inf, j.where(odd, x, 1.0))), value
    )
    return j.where(y == 0.0, j.asarray(1.0, j.float64), value)
