"""full-exponent binary64 FMA: exact product +128-bit jammed alignment.

RNE only. NaN-input payload propagation excluded; exception flags not modeled.
Adaptation of SoftFloat3's128-bit shift/jam/cancellation/round scheme.
"""

import jax
import jax.numpy as jnp


def fma_binary64(a, b, c):
    if not jax.config.x64_enabled:
        raise ValueError("Requires x64")
    a, b, c = jnp.broadcast_arrays(*[jnp.asarray(x, jnp.float64) for x in (a, b, c)])
    u = jnp.uint64
    i = jnp.int64
    mask = u((1 << 52) - 1)
    signmask = u(1 << 63)
    aa, bb, cc = [jax.lax.bitcast_convert_type(x, u) for x in (a, b, c)]
    sp = (aa ^ bb) & signmask
    sc = cc & signmask

    def decode(x):
        e = ((x >> u(52)) & u(2047)).astype(i)
        f = x & mask
        d = jnp.clip(jax.lax.clz(f).astype(i) - 11, 0, 52)
        return (
            jnp.where(e == 0, f << d.astype(u), f | u(1 << 52)),
            jnp.where(e == 0, 1 - d, e),
            (e == 0) & (f == 0),
            (e == 2047) & (f == 0),
            (e == 2047) & (f != 0),
        )

    sa, ea, za, ia, na = decode(aa)
    sb, eb, zb, ib, nb = decode(bb)
    cs, ce, zc, ic, nc = decode(cc)

    def shl(lo, hi, d):
        s = jnp.clip(d, 0, 63).astype(u)
        t = jnp.clip(d - 64, 0, 63).astype(u)
        lh = (hi << s) | jnp.where(d == 0, u(0), lo >> (u(64) - s))
        return jnp.where(d < 64, lo << s, u(0)), jnp.where(d < 64, lh, lo << t)

    def jam(lo, hi, d):
        s = jnp.clip(d, 0, 63).astype(u)
        t = jnp.clip(d - 64, 0, 63).astype(u)
        low = (lo >> s) | jnp.where(d == 0, u(0), hi << (u(64) - s))
        low = jnp.where(d < 64, low, hi >> t)
        high = jnp.where(d < 64, hi >> s, u(0))
        lost = jnp.where(
            d < 64, (lo & ((u(1) << s) - u(1))) != 0, (lo != 0) | ((hi & ((u(1) << t) - u(1))) != 0)
        )
        lost = jnp.where(d >= 128, (lo | hi) != 0, lost)
        low = jnp.where(d >= 128, u(0), low) | lost.astype(u)
        return low, high

    limb = u((1 << 32) - 1)
    al = sa & limb
    ah = sa >> u(32)
    bl = sb & limb
    bh = sb >> u(32)
    p0 = al * bl
    p1 = al * bh
    p2 = ah * bl
    p3 = ah * bh
    middle = (p0 >> u(32)) + (p1 & limb) + (p2 & limb)
    pl = (p0 & limb) | (middle << u(32))
    ph = p3 + (p1 >> u(32)) + (p2 >> u(32)) + (middle >> u(32))
    upper = (ph >> u(41)) != 0
    pe = ea + eb - 1023 + upper.astype(i)
    pl, ph = shl(pl, ph, 22 - upper.astype(i))
    cl, ch = shl(cs, jnp.zeros_like(cs), jnp.full_like(ce, 74))
    # Zero magnitudes cannot determine the shared exponent.
    zp = za | zb
    pe = jnp.where(zp, -10000, pe)
    ce = jnp.where(zc, -10000, ce)
    swap = (pe < ce) | ((pe == ce) & ((ph < ch) | ((ph == ch) & (pl < cl))))
    e = jnp.maximum(pe, ce)
    d = jnp.abs(pe - ce)
    hl = jnp.where(swap, cl, pl)
    hh = jnp.where(swap, ch, ph)
    ll = jnp.where(swap, pl, cl)
    lh = jnp.where(swap, ph, ch)
    ll, lh = jam(ll, lh, d)
    sign = jnp.where(swap, sc, sp)
    same = sp == sc
    addlo = hl + ll
    addhi = hh + lh + (addlo < hl).astype(u)
    sublo = hl - ll
    subhi = hh - lh - (hl < ll).astype(u)
    lo = jnp.where(same, addlo, sublo)
    hi = jnp.where(same, addhi, subhi)
    carry = (hi >> u(63)) != 0
    rl, rh = jam(lo, hi, jnp.ones_like(e))
    lo = jnp.where(carry, rl, lo)
    hi = jnp.where(carry, rh, hi)
    e = e + carry.astype(i)
    zero = (lo | hi) == 0
    lead = jnp.where(hi != 0, 127 - jax.lax.clz(hi).astype(i), 63 - jax.lax.clz(lo).astype(i))
    left = jnp.where(zero, 0, 126 - lead)
    lo, hi = shl(lo, hi, left)
    e = e - left
    # Retain53 significant bits plus10 round bits, jam everything farther away.
    rounded_sig, _ = jam(lo, hi, 64 + jnp.maximum(1 - e, 0))
    q = rounded_sig >> u(10)
    r = rounded_sig & u(1023)
    q = q + ((r > u(512)) | ((r == u(512)) & ((q & u(1)) != 0))).astype(u)
    carry = q == u(1 << 53)
    q = jnp.where(carry, q >> u(1), q)
    oe = jnp.maximum(e, 1) + carry.astype(i)
    mag = jnp.where(q < u(1 << 52), q, (oe.astype(u) << u(52)) | (q & mask))
    mag = jnp.where(oe >= 2047, u(0x7FF0000000000000), mag)
    mag = jnp.where(zero, u(0), mag)
    # Tiny nonzero negative results retain-0; only exact opposite cancellation+0.
    sign = jnp.where(zero & ~same, u(0), sign)
    bits = sign | mag
    ip = ia | ib
    bits = jnp.where(ip, sp | u(0x7FF0000000000000), bits)
    bits = jnp.where(ic, cc, bits)
    invalid = (za & ib) | (zb & ia) | (ip & ic & (sp != sc))
    bits = jnp.where(na | nb | nc | invalid, u(0x7FF8000000000000), bits)
    return jax.lax.bitcast_convert_type(bits, jnp.float64), ~(na | nb | nc)
