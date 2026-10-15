"""integer binary64 division: exact restoring quotient/remainder, RNE.

NaN input payloads unsupported (canonical qNaN,false); flags not modeled.
"""

import jax
import jax.numpy as jnp


def divide_binary64(a, b):
    if not jax.config.x64_enabled:
        raise ValueError("Requires x64")
    a, b = jnp.broadcast_arrays(jnp.asarray(a, jnp.float64), jnp.asarray(b, jnp.float64))
    u = jnp.uint64
    i = jnp.int64
    aa = jax.lax.bitcast_convert_type(a, u)
    bb = jax.lax.bitcast_convert_type(b, u)
    mask = u((1 << 52) - 1)
    sign = (aa ^ bb) & u(1 << 63)

    def decode(bits):
        e = ((bits >> u(52)) & u(2047)).astype(i)
        f = bits & mask
        shift = jnp.clip(jax.lax.clz(f).astype(i) - 11, 0, 52)
        sub = e == 0
        return (
            jnp.where(sub, f << shift.astype(u), f | u(1 << 52)),
            jnp.where(sub, 1 - shift, e),
            (e == 0) & (f == 0),
            (e == 2047) & (f == 0),
            (e == 2047) & (f != 0),
        )

    sa, ea, za, ia, na = decode(aa)
    sb, eb, zb, ib, nb = decode(bb)
    less = sa < sb
    num = jnp.where(less, sa << u(1), sa)
    e = ea - eb + 1023 - less.astype(i)

    # For finite nonzero inputs:1<=num/sb<2. Invariant num*2^k=q*sb+r.
    def step(_, state):
        q, r = state
        r = r << u(1)
        bit = r >= sb
        return (q << u(1)) | bit.astype(u), jnp.where(bit, r - sb, r)

    q, r = jax.lax.fori_loop(0, 52, step, (jnp.ones_like(sa), num - sb))
    normal_inc = ((r << u(1)) > sb) | (((r << u(1)) == sb) & ((q & u(1)) != 0))
    d = jnp.maximum(1 - e, 0)
    dc = jnp.clip(d, 0, 63).astype(u)
    k = jnp.clip(d - 1, 0, 63).astype(u)
    base = jnp.where(d >= 64, u(0), q >> dc)
    guard = ((q >> k) & u(1)) != 0
    guard = guard & (d > 0) & (d <= 53)
    lowmask = (u(1) << k) - u(1)
    sticky = ((q & lowmask) != 0) | (r != 0)
    inc = guard & (sticky | ((base & u(1)) != 0))
    rounded = jnp.where(e >= 1, q + normal_inc.astype(u), base + inc.astype(u))
    carry = rounded == u(1 << 53)
    normal_q = jnp.where(carry, rounded >> u(1), rounded)
    normal_e = e + carry.astype(i)
    normal_bits = (normal_e.astype(u) << u(52)) | (normal_q & mask)
    normal_bits = jnp.where(normal_e >= 2047, u(0x7FF0000000000000), normal_bits)
    bits = sign | jnp.where(e >= 1, normal_bits, rounded)
    bits = jnp.where(za | ib, sign, bits)
    bits = jnp.where(ia | zb, sign | u(0x7FF0000000000000), bits)
    bits = jnp.where(na | nb | (za & zb) | (ia & ib), u(0x7FF8000000000000), bits)
    return jax.lax.bitcast_convert_type(bits, jnp.float64), ~(na | nb)
