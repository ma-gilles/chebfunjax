"""integer binary64 add/sub, round-nearest ties-even.

Shift-right-jam and10 rounding bits follow Berkeley SoftFloat3's F64 scheme.
Returns(value,payload_supported): NaN input payload propagation unsupported.
Exception flags and alternate rounding modes are not implemented.
"""

import jax
import jax.numpy as jnp


def add_binary64(a, b):
    if not jax.config.x64_enabled:
        raise ValueError("Requires x64")
    a, b = jnp.broadcast_arrays(jnp.asarray(a, jnp.float64), jnp.asarray(b, jnp.float64))
    u = jnp.uint64
    i = jnp.int64
    aa = jax.lax.bitcast_convert_type(a, u)
    bb = jax.lax.bitcast_convert_type(b, u)
    mask = u((1 << 52) - 1)
    signmask = u(1 << 63)
    magmask = signmask - u(1)

    def decode(x):
        e = ((x >> u(52)) & u(2047)).astype(i)
        f = x & mask
        shift = jnp.clip(jax.lax.clz(f).astype(i) - 11, 0, 52)
        sub = e == 0
        return (
            jnp.where(sub, f << shift.astype(u), f | u(1 << 52)),
            jnp.where(sub, 1 - shift, e),
            (e == 2047) & (f == 0),
            (e == 2047) & (f != 0),
        )

    sa, ea, ia, na = decode(aa)
    sb, eb, ib, nb = decode(bb)
    swap = (aa & magmask) < (bb & magmask)
    hi = jnp.where(swap, sb, sa) << u(10)
    lo = jnp.where(swap, sa, sb) << u(10)
    e = jnp.maximum(ea, eb)
    diff = jnp.abs(ea - eb)
    sign = jnp.where(swap, bb, aa) & signmask
    same = ((aa ^ bb) & signmask) == 0

    def jam(x, d):
        dc = jnp.clip(d, 0, 63).astype(u)
        lost = (x & ((u(1) << dc) - u(1))) != 0
        return jnp.where(d >= 64, (x != 0).astype(u), (x >> dc) | lost.astype(u))

    aligned = jam(lo, diff)
    s = jnp.where(same, hi + aligned, hi - aligned)
    carry = (s >> u(63)) != 0
    s = jnp.where(carry, jam(s, jnp.ones_like(e)), s)
    e = e + carry.astype(i)
    left = jnp.clip(jax.lax.clz(s).astype(i) - 1, 0, 63)
    s = s << left.astype(u)
    e = e - left
    # Normalization is exact; only discarded low bits in alignment/underflow jam.
    s = jam(s, jnp.maximum(1 - e, 0))
    q = s >> u(10)
    r = s & u(1023)
    q = q + ((r > u(512)) | ((r == u(512)) & ((q & u(1)) != 0))).astype(u)
    carry = q == u(1 << 53)
    q = jnp.where(carry, q >> u(1), q)
    out_e = jnp.maximum(e, 1) + carry.astype(i)
    magnitude = jnp.where(q < u(1 << 52), q, (out_e.astype(u) << u(52)) | (q & mask))
    magnitude = jnp.where(out_e >= 2047, u(0x7FF0000000000000), magnitude)
    # Exact opposite-sign cancellation is+0 under RNE; -0+-0 retains-0.
    sign = jnp.where((magnitude == 0) & ~same, u(0), sign)
    bits = sign | magnitude
    infbits = jnp.where(ia, aa, bb)
    bits = jnp.where(ia | ib, infbits, bits)
    invalid = ia & ib & ~same
    bits = jnp.where(na | nb | invalid, u(0x7FF8000000000000), bits)
    return jax.lax.bitcast_convert_type(bits, jnp.float64), ~(na | nb)
