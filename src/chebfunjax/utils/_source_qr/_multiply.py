"""binary64 RNE multiply; integer-only, including gradual underflow.

Returns (value, payload_supported). All finite/Inf operands are supported.
NaN operands return canonical qNaN and false: payload propagation unqualified.
0*Inf returns canonical qNaN as IEEE invalid result; flags are not exposed.
"""

import jax
import jax.numpy as jnp


def multiply_binary64(a, b):
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
        sig = jnp.where(sub, f << shift.astype(u), f | u(1 << 52))
        exponent = jnp.where(sub, 1 - shift, e)
        return sig, exponent, (e == 0) & (f == 0), (e == 2047) & (f == 0), (e == 2047) & (f != 0)

    sa, ea, za, ia, na = decode(aa)
    sb, eb, zb, ib, nb = decode(bb)
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
    lo = (p0 & limb) | (middle << u(32))
    hi = p3 + (p1 >> u(32)) + (p2 >> u(32)) + (middle >> u(32))
    upper = (hi >> u(41)) != 0
    e = ea + eb - 1023 + upper.astype(i)
    shift = 52 + upper.astype(i) + jnp.maximum(1 - e, 0)
    # Right shift a128-bit integer; explicit out-of-range handling.
    s0 = jnp.clip(shift, 0, 63).astype(u)
    s1 = jnp.clip(shift - 64, 0, 63).astype(u)
    qlow = (lo >> s0) | jnp.where(shift == 0, u(0), hi << (u(64) - s0))
    q = jnp.where(shift < 64, qlow, hi >> s1)
    q = jnp.where(shift >= 128, u(0), q)
    k = shift - 1
    kl = jnp.clip(k, 0, 63).astype(u)
    kh = jnp.clip(k - 64, 0, 63).astype(u)
    guard = jnp.where(k < 64, (lo >> kl) & u(1), (hi >> kh) & u(1))
    guard = jnp.where((k < 0) | (k >= 128), u(0), guard)
    lowmask = jnp.where(k >= 64, u(0xFFFFFFFFFFFFFFFF), (u(1) << kl) - u(1))
    highmask = jnp.where(k >= 128, u(0xFFFFFFFFFFFFFFFF), (u(1) << kh) - u(1))
    sticky = ((lo & lowmask) != 0) | ((k > 64) & ((hi & highmask) != 0))
    rounded = q + ((guard != 0) & (sticky | ((q & u(1)) != 0))).astype(u)
    carry = rounded == u(1 << 53)
    normal_q = jnp.where(carry, rounded >> u(1), rounded)
    normal_e = e + carry.astype(i)
    normal_bits = (normal_e.astype(u) << u(52)) | (normal_q & mask)
    normal_bits = jnp.where(normal_e >= 2047, u(0x7FF0000000000000), normal_bits)
    magnitude = jnp.where(e >= 1, normal_bits, rounded)
    bits = sign | magnitude
    bits = jnp.where(za | zb, sign, bits)
    bits = jnp.where(ia | ib, sign | u(0x7FF0000000000000), bits)
    invalid = (za & ib) | (zb & ia)
    bits = jnp.where(na | nb | invalid, u(0x7FF8000000000000), bits)
    return jax.lax.bitcast_convert_type(bits, jnp.float64), ~(na | nb)
