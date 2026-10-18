"""IEEE binary64 exponent transport for JAX CPU with flushed subnormal arithmetic.
Finite real operands are decoded before CPU arithmetic can flush subnormals.
Multiply/divide tiny-result paths use exact integer product/remainder rounding;
normal-result mantissas use JAX arithmetic. Nonfinite operand classes and
signed zero are selected from decoded words to avoid CPU input flushing. This is not a full IEEE library: NaN payloads, exception flags,
other rounding modes, AD and arbitrary backend/compiler behavior are unqualified.
Add retains exponent-aligned floating arithmetic; no universal exact-rounding
claim is made for it. Signed zero input sums are preserved explicitly.
"""

import jax
import jax.numpy as j
from jax import lax

from chebfunjax.utils._gradual import _rounded_product_units, _rounded_quotient_units


def magnitude(x):
    """Magnitude.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting binary64 representation arithmetic for the JAX QZ algorithm;
    exact tiny-result rounding uses integer product/remainder helpers.
    """
    bits = lax.bitcast_convert_type(x, j.uint64) & j.uint64(0x7FFFFFFFFFFFFFFF)
    return lax.bitcast_convert_type(bits, j.float64)


def signed_magnitude(value, sign):
    """Signed magnitude.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting binary64 representation arithmetic for the JAX QZ algorithm;
    exact tiny-result rounding uses integer product/remainder helpers.
    """
    return lax.bitcast_convert_type(lax.bitcast_convert_type(value, j.uint64) | sign, j.float64)


def parts(x):
    """Parts.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting binary64 representation arithmetic for the JAX QZ algorithm;
    exact tiny-result rounding uses integer product/remainder helpers.
    """
    bits = lax.bitcast_convert_type(j.asarray(x, dtype=j.float64), j.uint64)
    sign = bits & j.uint64(0x8000000000000000)
    raw = (bits >> j.uint64(52)) & j.uint64(2047)
    fraction = bits & j.uint64(0x000FFFFFFFFFFFFF)
    mantissa = fraction | j.where(raw > 0, j.uint64(1 << 52), j.uint64(0))
    shift = lax.clz(mantissa).astype(j.int32) - 11
    shift = j.where(mantissa == 0, 0, shift)
    normalized = (mantissa << shift.astype(j.uint64)).astype(j.float64) / (2.0**52)
    exponent = j.where(raw > 0, raw.astype(j.int32) - 1023, -1022) - shift
    return sign, normalized, exponent


def compose(sign, mantissa, exponent):
    # Inputs mantissa in [0,4). Normalize without touching small floats.
    """Compose.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting binary64 representation arithmetic for the JAX QZ algorithm;
    exact tiny-result rounding uses integer product/remainder helpers.
    """
    large = mantissa >= 2
    mantissa = j.where(large, mantissa * 0.5, mantissa)
    exponent = exponent + large.astype(j.int32)
    small = (mantissa < 1) & (mantissa > 0)
    mantissa = j.where(small, mantissa * 2, mantissa)
    exponent = exponent - small.astype(j.int32)
    mbits = lax.bitcast_convert_type(mantissa, j.uint64) & j.uint64(0x000FFFFFFFFFFFFF)
    normal = ((exponent + 1023).astype(j.uint64) << j.uint64(52)) | mbits
    sub = j.rint(j.ldexp(mantissa, j.clip(exponent + 1074, -53, 52))).astype(j.uint64)
    bits = j.where(exponent < -1022, sub, normal)
    bits = j.where(exponent > 1023, j.uint64(0x7FF0000000000000), bits)
    bits = j.where((exponent < -1075) | (mantissa == 0), j.uint64(0), bits)
    return lax.bitcast_convert_type(bits | sign, j.float64)


@jax.jit
def multiply(x, y):
    """Multiply.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting binary64 representation arithmetic for the JAX QZ algorithm;
    exact tiny-result rounding uses integer product/remainder helpers.
    """
    x, y = j.broadcast_arrays(j.asarray(x, j.float64), j.asarray(y, j.float64))
    sx, mx, ex = parts(x)
    sy, my, ey = parts(y)
    ordinary_finite = compose(sx ^ sy, mx * my, ex + ey)
    supported = j.isfinite(x) & j.isfinite(y)
    # For normalized significands in [1,2), ex+ey <= -1023 includes every
    # subnormal product. The integer helper rounds once in units of 2^-1074,
    # using all 106 product bits, including any sticky bits below a halfway.
    tiny = supported & (ex + ey <= -1023) & (mx != 0) & (my != 0)

    def exact(_):
        a = j.where(tiny, magnitude(x), 1.0)
        b = j.where(tiny, magnitude(y), 1.0)
        value, eligible = _rounded_product_units(a, b)
        return j.where(tiny & eligible, signed_magnitude(value, sx ^ sy), ordinary_finite)

    finite = lax.cond(j.any(tiny), exact, lambda _: ordinary_finite, operand=None)
    # Inf times a nonzero subnormal is Inf, even if hardware multiplication
    # would flush that operand to zero and incorrectly return NaN.
    invalid = j.isnan(x) | j.isnan(y) | (j.isinf(x) & (my == 0)) | (j.isinf(y) & (mx == 0))
    special = j.where(invalid, j.nan, signed_magnitude(j.full_like(finite, j.inf), sx ^ sy))
    return j.where(supported, finite, special)


@jax.jit
def divide(x, y):
    """Divide.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting binary64 representation arithmetic for the JAX QZ algorithm;
    exact tiny-result rounding uses integer product/remainder helpers.
    """
    x, y = j.broadcast_arrays(j.asarray(x, j.float64), j.asarray(y, j.float64))
    sx, mx, ex = parts(x)
    sy, my, ey = parts(y)
    # Keep direct division after broadcasting; do not replace it with a
    # separately rounded scalar reciprocal in these fixed-shape kernels.
    denominator = lax.optimization_barrier(j.where(my != 0, my, 1.0))
    mantissa = mx / denominator
    result = compose(sx ^ sy, mantissa, ex - ey)
    supported = j.isfinite(x) & j.isfinite(y)
    # Ratio of normalized significands lies in (.5,2); this exponent range
    # includes all subnormal quotients and the minimum-normal boundary.
    tiny = supported & (ex - ey <= -1022) & (mx != 0) & (my != 0)

    def exact(_):
        a = j.where(tiny, magnitude(x), 1.0)
        b = j.where(tiny, magnitude(y), 1.0)
        value, eligible = _rounded_quotient_units(a, b)
        return j.where(tiny & eligible, signed_magnitude(value, sx ^ sy), result)

    result = lax.cond(j.any(tiny), exact, lambda _: result, operand=None)
    bits = lax.bitcast_convert_type(result, j.uint64)
    special = j.where(mx == 0, j.uint64(0x7FF8000000000000), j.uint64(0x7FF0000000000000)) | (
        sx ^ sy
    )
    finite = lax.bitcast_convert_type(j.where(my == 0, special, bits), j.float64)
    invalid = j.isnan(x) | j.isnan(y) | (j.isinf(x) & j.isinf(y))
    special = signed_magnitude(j.where(j.isinf(x), j.inf, 0.0), sx ^ sy)
    return j.where(supported, finite, j.where(invalid, j.nan, special))


def nonzero(x):
    """Nonzero.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting binary64 representation arithmetic for the JAX QZ algorithm;
    exact tiny-result rounding uses integer product/remainder helpers.
    """
    return (lax.bitcast_convert_type(x, j.uint64) & j.uint64(0x7FFFFFFFFFFFFFFF)) != 0


@jax.jit
def add(x, y):
    """Add.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting binary64 representation arithmetic for the JAX QZ algorithm;
    exact tiny-result rounding uses integer product/remainder helpers.
    """
    sx, mx, ex = parts(x)
    sy, my, ey = parts(y)
    exponent = j.maximum(j.where(mx > 0, ex, -4096), j.where(my > 0, ey, -4096))
    ax = j.ldexp(mx, j.clip(ex - exponent, -1075, 0))
    ay = j.ldexp(my, j.clip(ey - exponent, -1075, 0))
    combined = j.where(sx != 0, -ax, ax) + j.where(sy != 0, -ay, ay)
    sign, mantissa, extra = parts(combined)
    finite = compose(sign, mantissa, extra + exponent)
    finite = j.where((mx == 0) & (my == 0), signed_magnitude(j.zeros_like(finite), sx & sy), finite)
    return j.where(j.isfinite(x) & j.isfinite(y), finite, x + y)
